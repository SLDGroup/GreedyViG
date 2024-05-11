import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

import numpy

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'greedyvig': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}
    
class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),   
        )
        
    def forward(self, x):
        return self.stem(x)
    

class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, expansion=4):
        super().__init__()

        self.pw1 = nn.Conv2d(in_dim, in_dim * 4, 1) # kernel size = 1
        self.norm1 = nn.BatchNorm2d(in_dim * 4)
        self.act1 = nn.GELU()
        
        self.dw = nn.Conv2d(in_dim * 4, in_dim * 4, kernel_size=kernel, stride=1, padding=1, groups=in_dim * 4) # kernel size = 3
        self.norm2 = nn.BatchNorm2d(in_dim * 4)
        self.act2 = nn.GELU()
        
        self.pw2 = nn.Conv2d(in_dim * 4, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        x = self.pw2(x)
        x = self.norm3(x)
        return x

    
class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel, expansion=expansion_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.dws(x))
        else:
            x = x + self.drop_path(self.dws(x))
        return x
   


class DynamicMRConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.K = K
        self.mean = 0
        self.std = 0
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_j = x - x

        # get an estimate of the mean distance by computing the distance of points b/w quadrants. This is for efficiency to minimize computations.
        x_rolled = torch.cat([x[:, :, -H//2:, :], x[:, :, :-H//2, :]], dim=2)
        x_rolled = torch.cat([x_rolled[:, :, :, -W//2:], x_rolled[:, :, :, :-W//2]], dim=3)

        # Norm, Euclidean Distance
        norm = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)

        self.mean = torch.mean(norm, dim=[2,3], keepdim=True)
        self.std = torch.std(norm, dim=[2,3], keepdim=True)

        for i in range(0, H, self.K):
            x_rolled = torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)

            dist = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)

            # Got 83.86%
            mask = torch.where(dist < self.mean - self.std, 1, 0)

            x_rolled_and_masked = (x_rolled - x) * mask
            x_j = torch.max(x_j, x_rolled_and_masked)

        for j in range(0, W, self.K):
            x_rolled = torch.cat([x[:, :, :, -j:], x[:, :, :, :-j]], dim=3)

            dist = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)

            mask = torch.where(dist < self.mean - self.std, 1, 0)

            x_rolled_and_masked = (x_rolled - x) * mask
            x_j = torch.max(x_j, x_rolled_and_masked)
                 
        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class ConditionalPositionEncoding(nn.Module):
    """
    Implementation of conditional positional encoding. For more details refer to paper: 
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=in_channels
        )

    def forward(self, x):
        x = self.pe(x) + x
        return x


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, K):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.cpe = ConditionalPositionEncoding(in_channels, kernel_size=7)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DynamicMRConv4d(in_channels * 2, in_channels, K=self.K)  
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  # out_channels back to 1x}

       
    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)

        return x

    
class DynamicGraphConvBlock(nn.Module):
    def __init__(self, in_dim, drop_path=0., K=2, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        
        self.mixer = Grapher(in_dim, K)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(in_dim), requires_grad=True) 
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(in_dim), requires_grad=True) 
        
    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(x))
        else:
            x = x + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x))
        return x


class Downsample(nn.Module):
    """ 
    Convolution-based downsample
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class GreedyViG(torch.nn.Module):
    def __init__(self, blocks, channels, kernels, stride,
                 act_func, dropout=0., drop_path=0., emb_dims=512,
                 K=2, distillation=True, num_classes=1000,
                 pretrained=None, out_indices=None):
        super(GreedyViG, self).__init__()

        self.distillation = distillation
        self.out_indices = out_indices
        self.pretrained = pretrained
        
        self.stage_names = ['stem', 'local_1', 'local_2', 'local_3', 'global']
        
        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule 
        dpr_idx = 0

        self.stem = Stem(input_dim=3, output_dim=channels[0])
        
        self.backbone = []
        for i in range(len(blocks)):
            stage = []
            local_stages = blocks[i][0]
            global_stages = blocks[i][1]
            if i > 0:
                stage.append(Downsample(channels[i-1], channels[i]))
            for _ in range(local_stages):
                stage.append(InvertedResidual(dim=channels[i], kernel=3, expansion_ratio=4, drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            for _ in range(global_stages):
                stage.append(DynamicGraphConvBlock(channels[i], drop_path=dpr[dpr_idx], K=K[i]))
                dpr_idx += 1
            self.backbone.append(nn.Sequential(*stage))
            
        self.backbone = nn.Sequential(*self.backbone)

        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)


    def init_weights(self):
        logger = get_root_logger()
        print("Pretrained weights being loaded")
        logger.warn('Pretrained weights being loaded')
        ckpt_path = self.pretrained
        ckpt = _load_checkpoint(
            ckpt_path, logger=logger, map_location='cpu')
        print("ckpt keys: ", ckpt.keys())
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict_ema']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt

        state_dict = _state_dict
        missing_keys, unexpected_keys = \
            self.load_state_dict(state_dict, False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in self.out_indices:
                outs.append(x)
        return outs


if has_mmseg:
    @seg_BACKBONES.register_module()
    def greedyvig_s_feat(pretrained=True, **kwargs):
        model = GreedyViG(blocks=[[2,2], [2,2], [6,2], [2,2]],
                        channels=[48, 96, 192, 384],
                        kernels=3,
                        stride=1,
                        act_func='gelu',
                        dropout=0.,
                        drop_path=0.1,
                        emb_dims=768,
                        K=[8, 4, 2, 1],
                        distillation=True,
                        num_classes=1000,
                        out_indices=[0, 1, 2, 3],
                        pretrained='../Results/S_GreedyViG_81_1.pth')
        model.default_cfg = default_cfgs['greedyvig']
        return model

    @seg_BACKBONES.register_module()
    def greedyvig_b_feat(pretrained=True, **kwargs):
        model = GreedyViG(blocks=[[4,4], [4,4], [12,4], [3,3]],
                        channels=[64, 128, 256, 512],
                        kernels=3,
                        stride=1,
                        act_func='gelu',
                        dropout=0.,
                        drop_path=0.1,
                        emb_dims=768,
                        K=[8, 4, 2, 1],
                        distillation=True,
                        num_classes=1000,
                        out_indices=[0, 1, 2, 3],
                        pretrained='../Results/B_GreedyViG_83_9.pth')
        model.default_cfg = default_cfgs['greedyvig']
        return model