import math

import cv2
import numpy as np
import torch
import torch.nn as nn
# from utils.torch_utils import initialize_weights, is_parallel, model_info
# from utils.yolo_utils import check_anchor_order


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              kernel_size // 2,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# CBL + CBL + Res_Add
class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            CBL(in_channels, in_channels // 2, kernel_size=1),
            CBL(in_channels // 2, out_channels, kernel_size=3))

    def forward(self, x):
        return x + self.block(x)  # res

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            CBL(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x):
        x = self.upsample(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3 or 5 or 7"

        self.conv = nn.Conv2d(2,
                              1,
                              kernel_size,
                              padding=kernel_size // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avgout, maxout], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention) * x


class Trans_guide(nn.Module):
    def __init__(self, ch=16):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(6, ch, 3, padding=1),
            nn.LeakyReLU(True),
            SpatialAttention(3),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.layer(x)


class Trans_low(nn.Module):
    def __init__(
        self,
        ch_blocks=64,
        ch_mask=16,
    ):
        super().__init__()

        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, ch_blocks, 3, padding=1),
                                     nn.LeakyReLU(True))

        self.mm1 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=1,
                             padding=0)
        self.mm2 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=3,
                             padding=3 // 2)
        self.mm3 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=5,
                             padding=5 // 2)
        self.mm4 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=7,
                             padding=7 // 2)

        self.decoder = nn.Sequential(nn.Conv2d(ch_blocks, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, 3, 3, padding=1))

        self.trans_guide = Trans_guide(ch_mask)

    def forward(self, x):
        x1 = self.encoder(x)
        x1_1 = self.mm1(x1)
        x1_2 = self.mm1(x1)
        x1_3 = self.mm1(x1)
        x1_4 = self.mm1(x1)
        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1 = self.decoder(x1)

        out = x + x1
        out = torch.relu(out)

        mask = self.trans_guide(torch.cat([x, out], dim=1))
        return out, mask


class SFT_layer(nn.Module):
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(inter_ch, out_ch, kernel_size, padding=kernel_size // 2))
        self.shift_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))
        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))

    def forward(self, x, guide):
        x = self.encoder(x)
        scale = self.scale_conv(guide)
        shift = self.shift_conv(guide)
        x = x * scale + shift + x
        x = self.decoder(x)
        return x


class Trans_high(nn.Module):
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        super().__init__()

        self.sft = SFT_layer(in_ch, inter_ch, out_ch, kernel_size)

    def forward(self, x, guide):
        return x + self.sft(x, guide)


class Up_guide(nn.Module):
    def __init__(self, kernel_size=1, ch=3):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch,
                      ch,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False))

    def forward(self, x):
        x = self.up(x)
        return x



class DENet(nn.Module):
    def __init__(self,
                 num_high=3,
                 ch_blocks=32,
                 up_ksize=1,
                 high_ch=32,
                 high_ksize=3,
                 ch_mask=32,
                 gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        self.trans_low = Trans_low(ch_blocks, ch_mask)

        for i in range(0, self.num_high):
            self.__setattr__('up_guide_layer_{}'.format(i),
                             Up_guide(up_ksize, ch=3))
            self.__setattr__('trans_high_layer_{}'.format(i),
                             Trans_high(3, high_ch, 3, high_ksize))

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        trans_pyrs = []
        trans_pyr, guide = self.trans_low(pyrs[-1])
        trans_pyrs.append(trans_pyr)

        commom_guide = []
        for i in range(self.num_high):
            guide = self.__getattr__('up_guide_layer_{}'.format(i))(guide)
            commom_guide.append(guide)

        for i in range(self.num_high):
            trans_pyr = self.__getattr__('trans_high_layer_{}'.format(i))(
                pyrs[-2 - i], commom_guide[i])
            trans_pyrs.append(trans_pyr)

        out = self.lap_pyramid.pyramid_recons(trans_pyrs)

        return out