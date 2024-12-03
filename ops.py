import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SOBEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.filterx = nn.Parameter(torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype= torch.float32, requires_grad=False).view(1, 1, 3, 3).repeat(1, 3, 1, 1)).cuda()
        self.filtery = nn.Parameter(torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype= torch.float32, requires_grad=False).view(1, 1, 3, 3).repeat(1, 3, 1, 1)).cuda()
        self.xbias = nn.Parameter(torch.Tensor(1))
        self.ybias = nn.Parameter(torch.Tensor(1))
        
    def forward(self, x):
        o1 = torch.nn.functional.conv2d(x, weight = self.filtery, bias=self.ybias, stride= [3, 3])
        o2 = torch.nn.functional.conv2d(x, weight = self.filterx, bias=self.xbias, stride= [3, 3])
        o = torch.sqrt(torch.abs((o1 * o1 + o2 * o2))).repeat(1,3,1,1)
        return o

class PDConv2d(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(PDConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


## cd, ad, rd convolutions
def createConvFunc(op_type):
    assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'sobel':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 7, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3*7).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 3*7)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 1, 2, 3, 4, 5, 6, 14,15,16,17,18,19,20]] = weights[:, :, [0, 1, 2, 3, 4, 5, 6, 14,15,16,17,18,19,20]]
            buffer[:, :, [7,8,9,10,11,12]] = -weights[:, :, [0, 1, 2, 3, 4, 5, 6]]-weights[:, :, [14,15,16,17,18,19,20]]
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    else:
        print('impossible to be here unless you force that')
        return None

class Dconv(nn.Module):
    def __init__(self, inc, outc, kernal_size=(3, 3), padding=(1, 1), bias=True):
        super().__init__()
        self.kernal_size = kernal_size
        self.N = 9
        self.weight = nn.Parameter(torch.Tensor(outc, inc, kernal_size[0], kernal_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(outc))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
            
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        shape = self.weight.shape
        if self.weight.is_cuda:
            buffer = torch.cuda.FloatTensor(shape[0], shape[1], self.N).fill_(0)
        else:
            buffer = torch.zeros(shape[0], shape[1], self.N)
        weights = self.weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 1, 2, 6, 7, 8]] = weights[:, :, [0, 1, 2, 6, 7, 8]]
        buffer[:, :, [3, 4, 5]] = - weights[:, :, [0, 1, 2]] - weights[:, :, [6, 7, 8]]#weights[:, :, [7,8,9,10,11,12,13]] 
        buffer = buffer.view(shape[0], shape[1], 3, 3)
        y = F.conv2d(x, buffer, self.bias, stride=self.kernal_size)
        return y


class DDConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=(3, 3), padding=(1, 1), bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.N = 9
        self.zero_padding = nn.ZeroPad2d(padding)
        # self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.dconv = Dconv(outc, inc)
        # self.sobel = SOBEL()
        
        
    
    def forward(self, x, alpha):
        
        dtype = alpha.data.type()
        # print(alpha.mean())
        if self.padding:
            x = self.zero_padding(x)
        
        p = self._get_p(alpha, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        
        q_lt = torch.cat([torch.clamp(q_lt[..., :self.N], 0, x.size(2)-1), torch.clamp(q_lt[..., self.N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :self.N], 0, x.size(2)-1), torch.clamp(q_rb[..., self.N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :self.N], q_rb[..., self.N:]], -1)
        q_rt = torch.cat([q_rb[..., :self.N], q_lt[..., self.N:]], -1)
        
        # mask = torch.cat([p[..., :self.N].lt(self.padding[0])+p[..., :self.N].gt(x.size(2)-1-self.padding[0]),
        #                   p[..., self.N:].lt(self.padding[1])+p[..., self.N:].gt(x.size(3)-1-self.padding[1])], dim=-1).type_as(p)
        # mask = mask.detach()
        # # floor_p = p - (p - torch.floor(p))
        # p = p * (1-mask) + torch.floor(p)*mask
        # p = torch.cat([torch.clamp(p[..., :self.N], 0, x.size(2)-1), torch.clamp(p[..., self.N:], 0, x.size(3)-1)], dim=-1)
        
        
        g_lt = (1 + (q_lt[..., :self.N].type_as(p) - p[..., :self.N])) * (1 + (q_lt[..., self.N:].type_as(p) - p[..., self.N:]))
        g_rb = (1 - (q_rb[..., :self.N].type_as(p) - p[..., :self.N])) * (1 - (q_rb[..., self.N:].type_as(p) - p[..., self.N:]))
        g_lb = (1 + (q_lb[..., :self.N].type_as(p) - p[..., :self.N])) * (1 - (q_lb[..., self.N:].type_as(p) - p[..., self.N:]))
        g_rt = (1 - (q_rt[..., :self.N].type_as(p) - p[..., :self.N])) * (1 + (q_rt[..., self.N:].type_as(p) - p[..., self.N:]))
        
        x_q_lt = self._get_x_q(x, q_lt)
        x_q_rb = self._get_x_q(x, q_rb)
        x_q_lb = self._get_x_q(x, q_lb)
        x_q_rt = self._get_x_q(x, q_rt)
        
        expend_map = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        
        # expend_map = x.unsqueeze(-1).repeat([1,1,1,1,21])
        expend_map = self._reshape_x_offset(expend_map, self.kernel_size)
        
        
        # y = self.dconv(expend_map)
        y = self.dconv(expend_map)
        
        # out = self.conv_kernel(expend_map)
        return y
        # return out
    
    def _reshape_x_offset(self, x_offset, ks):
        b, c, w, h, N = x_offset.size()
        
        x_offset = x_offset.permute(0,1,3,2,4)
        x_offset = torch.cat([x_offset[..., s:s+ks[0]].contiguous().view(b, c, h, w*ks[0]) for s in range(0, N, ks[0])], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks[1], w*ks[0])
        x_offset = x_offset.permute(0,1,3,2)
        return x_offset

    def _get_x_q(self, x, q):
        b, w, h, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)将图片压缩到1维，方便后面的按照index索引提取
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)这个目的就是将index索引均匀扩增到图片一样的h*w大小
        index = q[..., :self.N]*padded_w + q[..., self.N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        
        #双线性插值法就是4个点再乘以对应与 p 点的距离。获得偏置点 p 的值，这个 p 点是 9 个方向的偏置所以最后的 x_offset 是 b×c×h×w×9。
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, w, h, self.N)

        return x_offset
        
        
    def _get_p(self, alpha, dtype):
        w, h = alpha.size(2), alpha.size(3)
        # print(w,h)
        p_n = self._get_p_n(dtype)
        p_0 = self._get_p_0(w, h, dtype)
        p_n = torch.cat([p_n[:,:self.N,...] * torch.cos(alpha) + p_n[:,:self.N,...] * torch.sin(alpha), p_n[:, self.N:, ...] * torch.cos(alpha) - p_n[:, self.N:, ...] * torch.sin(alpha)], dim= 1)
        # print(p_0.size(), p_n.size())
        p = p_0 + p_n
        return p
    
    def _get_p_n(self, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-(self.kernel_size[0] // 2), (self.kernel_size[0] // 2) + 1,1), 
                      torch.arange(-(self.kernel_size[1] // 2), (self.kernel_size[1] // 2) + 1,1), indexing = 'ij')
        p_n = torch.cat([p_n_x.flatten(), p_n_y.flatten()],dim= 0)
        p_n = torch.reshape(p_n, (1, 2 * self.N, 1, 1))
        p_n = p_n.type(dtype)
        return p_n
    
    def _get_p_0(self, w, h , dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, w+1), torch.arange(1, h+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, w, h).repeat(1, self.N,1,1)
        p_0_y = p_0_y.flatten().reshape(1, 1, w, h).repeat(1, self.N,1,1)
        p_0 = torch.cat([p_0_x, p_0_y], dim=1)
        p_0 = p_0.type(dtype)

        return p_0