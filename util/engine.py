# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
import util.utils as utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, batch_size: int, gradient_accumulation: bool, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    if gradient_accumulation == True:
        batch_idx = 0
        accum_iter = 4 # making lambda 512 (128 * 4) batch size = 2048 (128 * 16) effective batch size
        for inputs, labels in metric_logger.log_every(
                data_loader, print_freq, header, batch_idx):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if mixup_fn is not None:
                inputs, labels = mixup_fn(inputs, labels)

            with torch.cuda.amp.autocast():     # if True:
                preds = model(inputs)
                loss = criterion(inputs, preds, labels)

            loss_value = loss.item() # gradient stored, until we call optimizer.zero_grad()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
                optimizer.zero_grad()

                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(
                    optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                            parameters=model.parameters(), create_graph=is_second_order)

                torch.cuda.synchronize()
                if model_ema is not None:
                    model_ema.update(model)

                metric_logger.update(loss=loss_value)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            batch_idx += 1
    else:
        for inputs, labels in metric_logger.log_every(
                data_loader, print_freq, header):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if mixup_fn is not None:
                inputs, labels = mixup_fn(inputs, labels)

            with torch.cuda.amp.autocast():
                preds = model(inputs)
                loss = criterion(inputs, preds, labels)

            loss_value = loss.item() # gradient stored, until we call optimizer.zero_grad()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model.parameters(), create_graph=is_second_order)

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    batch_idx = 0

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(output.mean().item(), output.std().item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
