""" ImageNet Training Script
 #train.py from vig_pytorch

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)

Modified from the training script used for the Vision GNN model found in: 
    @misc{
        https://doi.org/10.48550/arxiv.2206.00272,
        doi = {10.48550/ARXIV.2206.00272},
        url = {https://arxiv.org/abs/2206.00272},
        author = {Han, Kai and Wang, Yunhe and Guo, Jianyuan and Tang, Yehui and Wu, Enhua},
        keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
        title = {Vision GNN: An Image is Worth Graph of Nodes},
        publisher = {arXiv},
        year = {2022},
        copyright = {arXiv.org perpetual, non-exclusive license}
    }
"""
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
warnings.filterwarnings('ignore')
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import copy
import os.path as osp

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel
import timm

from timm.data import ImageDataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset #create_loader
from timm.models import create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed  # , train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes, update_data_root)

from mmdet_custom.apis.train import train_detector
import mmcv_custom.runner.epoch_based_runner
import mmcv_custom.runner.optimizer

import greedyvig_backbone


parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('config', help='train config file path')
parser.add_argument('--work-dir', help='the dir to save logs and models')
parser.add_argument('--greedyvig_model', default='greedyvig_s', type=str, metavar='MODEL',
                help='Name of model to train (default: "greedyvig_s"')
parser.add_argument(
    '--resume-from', help='the checkpoint file to resume from')
parser.add_argument(
    '--auto-resume',
    action='store_true',
    help='resume from the latest checkpoint automatically')
parser.add_argument(
    '--no-validate',
    action='store_true',
    help='whether not to evaluate the checkpoint during training')
group_gpus = parser.add_mutually_exclusive_group()
group_gpus.add_argument(
    '--gpus',
    type=int,
    help='(Deprecated, please use --gpu-id) number of gpus to use '
            '(only applicable to non-distributed training)')
group_gpus.add_argument(
    '--gpu-ids',
    type=int,
    nargs='+',
    help='(Deprecated, please use --gpu-id) ids of gpus to use '
            '(only applicable to non-distributed training)')
group_gpus.add_argument(
    '--gpu-id',
    type=int,
    default=0,
    help='id of gpu to use '
            '(only applicable to non-distributed training)')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument(
    '--diff-seed',
    action='store_true',
    help='Whether or not set different seeds for different ranks')
parser.add_argument(
    '--deterministic',
    action='store_true',
    help='whether to set deterministic options for CUDNN backend.')
parser.add_argument(
    '--options',
    nargs='+',
    action=DictAction,
    help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file (deprecate), '
            'change to --cfg-options instead.')
parser.add_argument(
    '--cfg-options',
    nargs='+',
    action=DictAction,
    help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file. If the value to '
            'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
            'Note that the quotation marks are necessary and that no white space '
            'is allowed.')
parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none',
    help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument(
    '--auto-scale-lr',
    action='store_true',
    help='enable automatically scaling LR.')

args = parser.parse_args()
print(args)
print("\n greedyvig_model = " + str(args.greedyvig_model))

def parse_args():
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update all the '
                          'configuration files to mmdet >= 2.24.1.')

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        print("NOT DISTRIBUTED LAUNCH")
    else:
        print("DIST LAUNCH")
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
        print ("GPU IDS")
        print(cfg.gpu_ids)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
        # drop_connect_rate=0.1,
        # drop_path_rate=0.1,
        # drop_block_rate=0.1)
    model.init_weights()

    print("\n\n\nCONFIG OF MODEL: \n" + str(cfg.model) + "\n\n\n")

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    print("\n\n\nTRAIN DETECTOR CONFIG OF MODEL: \n" + str(cfg) + "\n\n\n")

    print("\n\n\TRAIN DETECTOR MODEL: \n" + str(model) + "\n\n\n")

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
