import warnings

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from .ops import PDConv2d, createConvFunc, DDConv2d
import torch
import cv2
from .peops import PENet
from .denet import DENet




class DCM(BaseModule):
    def __init__(self):
        super().__init__()
        self.alphas = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 1, kernel_size=7, padding=3)
        )
        # self.alphas = nn.Conv2d(3, 6, kernel_size=7, padding=1)
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(6)
        # self.alphas2 = nn.Conv2d(6, 1, kernel_size=7, padding=1)
        self.dconv = DDConv2d(3, 3, kernel_size=(3, 3), padding=[1, 1])
        # self.bn = nn.BatchNorm2d(3)
        # self.relu = nn.ReLU()
    
    def forward(self, x):
        alphas = self.alphas(x)
        # alphas = torch.nn.functional.softsign(alphas) * torch.pi/2
        alphas = torch.tanh(alphas) * torch.pi/2
        # alphas = torch.sigmoid(alphas) * torch.pi
        # alphas_r = torch.zeros_like(alphas)
        # x = self.relu(self.dconv(x, alphas))
        # x = self.bn(x)
        return self.dconv(x, alphas)
    
class PNet(BaseModule):
    def __init__(self):
        super().__init__()

        self.conv1= nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )
        self.conv3= nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )

    
    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return x
        
        
class Contour(BaseModule):
    def __init__(self):
        super().__init__()
        self.layercv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        
        self.layercd = nn.Sequential(
            PDConv2d(createConvFunc("cv"), 3, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        
        # self.dcm = DCM()
        
        self.catconv = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        
    def forward(self, x): 
        
        # exdark best
        xcd = self.layercd(x)
        # xcd = self.dcm(x)
        out = self.catconv(torch.cat([xcd, 1.2 * x], dim= 1))
        # out = self.layercv(out - xcd)
        return out + xcd, xcd



        # # exdark1
        # x4 = self.layer21(x)
        
        # out = self.catcat(torch.cat([x4, x], dim= 1))
        # x4 = self.layer12(out)
        # x4 = self.layer2(x4)
        # # x4 = torch.sum(x4, dim=1, keepdim=True)
        # out = self.catcat1(torch.cat([x4, out], dim= 1))
        # return out, x4

        # #rtts
        # x4 = self.layercv(x)
        # out = self.catconv(torch.cat([x4, x], dim= 1))
        # return out + x4.detach(), x4





# from thop import profile
# net = DENet().cuda()
# inputs = torch.randn(1, 3, 32, 32).cuda()
# flops, params = profile(net, (inputs,))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')
# exit(0)







class ResBlock(BaseModule):
    """The basic residual block used in Darknet. Each ResBlock consists of two
    ConvModules and the input is added to the final output. Each ConvModule is
    composed of Conv, BN, and LeakyReLU. In YoloV3 paper, the first convLayer
    has half of the number of the filters as much as the second convLayer. The
    first convLayer has filter size of 1x1 and the second one has the filter
    size of 3x3.

    Args:
        in_channels (int): The input channels. Must be even.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(ResBlock, self).__init__(init_cfg)
        assert in_channels % 2 == 0  # ensure the in_channels is even
        half_in_channels = in_channels // 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(in_channels, half_in_channels, 1, **cfg)
        self.conv2 = ConvModule(
            half_in_channels, in_channels, 3, padding=1, **cfg)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual

        return out


@MODELS.register_module()
class CADarknet(BaseModule):
    """Darknet backbone.

    Args:
        depth (int): Depth of Darknet. Currently only support 53.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import Darknet
        >>> import torch
        >>> self = Darknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """

    # Dict(depth: (layers, channels))
    arch_settings = {
        53: ((1, 2, 8, 8, 4), ((32, 64), (64, 128), (128, 256), (256, 512),
                               (512, 1024)))
    }

    def __init__(self,
                 depth=53,
                 out_indices=(3, 4, 5),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_eval=True,
                 pretrained=None,
                 init_cfg=None):
        super(CADarknet, self).__init__(init_cfg)
        
        self.contour = Contour()

        # self.pe = PENet()
        # self.de = DENet()
        # self.p = PNet()
            
        self.num = 0
        self.interval = 100
        
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for darknet')

        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.layers, self.channels = self.arch_settings[depth]

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(3, 32, 3, padding=1, **cfg)

        self.cr_blocks = ['conv1']
        for i, n_layers in enumerate(self.layers):
            layer_name = f'conv_res_block{i + 1}'
            in_c, out_c = self.channels[i]
            self.add_module(
                layer_name,
                self.make_conv_res_block(in_c, out_c, n_layers, **cfg))
            self.cr_blocks.append(layer_name)

        self.norm_eval = norm_eval

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):        
        self.num+=1
        if self.num == self.interval:
            # test_compare = torch.cat([x, out], dim=3)
            test_compare = x.clone()
            test_compare = test_compare.cpu().detach().numpy().transpose((0, 2, 3, 1))
            img = test_compare[0]
            cv2.imwrite('compare1' + '.png', img * 255)

        x, ooo = self.contour(x)
        # x = self.contour(x)
        # x = self.de(x)
        # x = self.p(x)
        # x = self.pe(x)
        
        # x, ooo = self.contour(x)

        # test_compare = x.clone()
        # test_compare = test_compare.cpu().detach().numpy().transpose((0, 2, 3, 1))
        # for i in range(test_compare.shape[0]):
        #     self.num+=1
        #     cv2.imwrite('showde/compare' + str(self.num) + '.png', test_compare[i] * 255)

        
        
        # if self.num == self.interval:
        #     # test_compare = torch.cat([x, out], dim=3)
        #     test_compare = ooo.clone()
        #     test_compare = test_compare.cpu().detach().numpy().transpose((0, 2, 3, 1))
        #     img = test_compare[0]
        #     cv2.imwrite('compare2' + '.png', img * 255)

        # if self.num == self.interval:
        #     # test_compare = torch.cat([x, out], dim=3)
        #     test_compare = x.clone()
        #     test_compare = test_compare.cpu().detach().numpy().transpose((0, 2, 3, 1))
        #     img = test_compare[0]
        #     # print(img.shape)
        #     cv2.imwrite('compare3' + '.png', img * 255)
        #     self.num = 0

        
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.cr_blocks[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CADarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    @staticmethod
    def make_conv_res_block(in_channels,
                            out_channels,
                            res_repeat,
                            conv_cfg=None,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            act_cfg=dict(type='LeakyReLU',
                                         negative_slope=0.1)):
        """In Darknet backbone, ConvLayer is usually followed by ResBlock. This
        function will make that. The Conv layers always have 3x3 filters with
        stride=2. The number of the filters in Conv layer is the same as the
        out channels of the ResBlock.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            res_repeat (int): The number of ResBlocks.
            conv_cfg (dict): Config dict for convolution layer. Default: None.
            norm_cfg (dict): Dictionary to construct and config norm layer.
                Default: dict(type='BN', requires_grad=True)
            act_cfg (dict): Config dict for activation layer.
                Default: dict(type='LeakyReLU', negative_slope=0.1).
        """

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        model = nn.Sequential()
        model.add_module(
            'conv',
            ConvModule(
                in_channels, out_channels, 3, stride=2, padding=1, **cfg))
        for idx in range(res_repeat):
            model.add_module('res{}'.format(idx),
                             ResBlock(out_channels, **cfg))
        return model
