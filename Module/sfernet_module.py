import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

from ultralytics.nn.modules.conv import Conv
#from ultralytics.nn.modules import register_module
from .wavelet import create_1d_wavelet_filter, create_2d_wavelet_filter, wavelet_1d_transform, inverse_1d_wavelet_transform,wavelet_2d_transform, inverse_2d_wavelet_transform
from . import wavelet
from typing import Tuple
from mmcv.cnn import ConvModule
from timm.models.layers import DropPath
import numpy as np
from einops import rearrange

    

GLOBAL_WT_CACHE = {}  
  
class WSE(nn.Module):
    def __init__(self, c1, c2, kernel_size=7, k=None, stride=1, 
                 bias=True, wt_levels=1, act=True, wt_type='db1',
                 ):
        super(WSE, self).__init__()

        if k is not None:
            kernel_size = k
        

        self.in_channels = c1
        self.out_channels = c2
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1
        self.save_high_freq = False
        

        self.wt_filter, self.iwt_filter = wavelet.create_2d_wavelet_filter(wt_type, c1, c1, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(c1, c1, kernel_size, padding='same', stride=1, dilation=1, groups=c1, bias=bias)
        self.base_scale = _ScaleModule([1,c1,1,1])

        
        target_k = 7   # target_k =11  # change 3, 5 或 7 即可切换对应尺寸
        self.wavelet_convs = nn.ModuleList(
            [
                DLKC(
                    dim=c1 * 4,    
                    kernel_size=target_k  
                )
                for _ in range(self.wt_levels)
            ]
        )
        
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,c1*4,1,1], init_scale=0.01) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                #curr_x_ll = F.pad(curr_x_ll, (0, 1, 0, 1), mode='reflect')
                curr_x_ll = F.pad(curr_x_ll, curr_pads) #yuan

            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels[i]
            curr_x_h = x_h_in_levels[i]
            curr_shape = shapes_in_levels[i]

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        
        if self.save_high_freq:
            input_h, input_w = x.shape[2], x.shape[3]
            raw_highs = []
            for i in range(self.wt_levels):
                 high = x_h_in_levels[i].sum(dim=2) 
                 raw_highs.append(high)
                 
            
            
            high_freq_out = sum(raw_highs)
            return x, high_freq_out
        else:
            return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=3, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)


def create_2d_wavelet_filter(wt_type, in_channels, out_channels, dtype):
    
    filter = torch.randn(out_channels, in_channels, 2, 2, dtype=dtype)
    return filter, filter  

def wavelet_2d_transform(x, filter):
    
    return F.conv2d(x, filter, stride=2, groups=x.size(1))[:, :, None]

def inverse_2d_wavelet_transform(x, filter):
    
    return F.conv_transpose2d(x.squeeze(2), filter, stride=2, groups=x.size(1))


class SFAEM(nn.Module):
    def __init__(self, in_channels=None, width_multiple=1.0, other_param=0.5, return_high_freq=False,use_global_cache=True):
        super().__init__()

        self.in_channels = int(in_channels * width_multiple)#v11
        self.attentions = Conv(c1=self.in_channels, c2=1, k=1, act=True)
        self.values = Conv(c1=self.in_channels, c2=self.in_channels, k=1, act=True)
        self.main_conv = WSE(c1=self.in_channels, c2=self.in_channels, k=1, act=True)
        self.out_project = Conv(c1=self.in_channels, c2=self.in_channels, k=1, act=True)
        
        self.main_conv.save_high_freq = True 
        self.return_high_freq = return_high_freq
        self.use_global_cache = use_global_cache

    def forward(self, features: torch.Tensor):
        attn_logits = self.attentions(features)
        values = self.values(features)
        
        main_result = self.main_conv(features)

        if isinstance(main_result, tuple):
             x, high_freq = main_result
        else:
             x = main_result
             high_freq = None

        
        target_layer_id = 7  # 定义你想要保留的层 ID
        
        if hasattr(self, 'i') and high_freq is not None:
            if self.i == target_layer_id:
                GLOBAL_WT_CACHE[self.i] = high_freq.detach()
                
                
                if not hasattr(self, '_logged_id'):
                    print(f"📦 [SFAEM] 成功！我是 Layer {self.i}，已存入缓存。")
                    self._logged_id = True

        context_scores_W = F.softmax(attn_logits, dim=-1)
        context_vector_W = values * context_scores_W
        context_vector_W = torch.sum(context_vector_W, dim=-1, keepdim=True)
        context_scores_H = F.softmax(attn_logits, dim=-2)
        context_vector_H = values * context_scores_H
        context_vector_H = torch.sum(context_vector_H, dim=-2, keepdim=True)

        out = (x + context_vector_W.expand_as(values) + context_vector_H.expand_as(values))
        out = self.out_project(out)

        if self.return_high_freq:
            return out, high_freq
        else:
            return out

    

    
class DLKC(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        
        self.branch_local = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, 
            groups=dim, bias=False
        )

        self.branch_spatial = DRE(c1=dim, c2=dim, 
            k=3,act=False,depth_multiplier=1
        )

        self.branch_fuse = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        local = self.branch_local(x)
        
        spatial = self.branch_spatial(x)
        
        return self.branch_fuse(torch.cat([local, spatial], dim=1))
    
class DRE(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True, depth_multiplier=2):
        super().__init__()
        
        self.unshuffle = nn.PixelUnshuffle(2)
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                c1 * 4, 
                c2 * 4, 
                kernel_size=k, 
                stride=1, 
                padding=k//2, 
                groups=c1 * 4, 
                bias=False
            ),
            nn.BatchNorm2d(c2 * 4),
            nn.ReLU() if act else nn.Identity()
        )
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.unshuffle(x)  
        x=self.depthwise(x)
        x = self.shuffle(x)    
        return x
    
def autopad(k, p=None, d=1):  
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

    
class DWConv(nn.Module):
    
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, padding=autopad(k), groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1) 
        self.pw = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.pw(self.bn(self.dw(x))))
    
class SE(nn.Module):
    def __init__(self, c1, c2, r=16): 
        super().__init__()
        channel = c1 
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CA(nn.Module):
    def __init__(self, c1, c2, reduction=32):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, c1 // reduction)

        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU() 

        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class DTCA(nn.Module):
    def __init__(self, in_channels, patch=(7, 7), groups=32):
        super().__init__()
        if in_channels < groups or in_channels % groups != 0:
            groups = 1
        self.in_channels = in_channels
        self.patch = patch
        drop_prob = 0.1
        
        self.channel1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False)

        self.post_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        #self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        #tianjiagamma:1e-4 *
        #self.learnable_alpha = nn.Parameter(torch.zeros(1))
        #self.gamma = nn.Parameter(1e-4 * torch.ones((in_channels, 1, 1)), requires_grad=True)
        #self.gate = nn.Parameter(torch.ones(1))
        #self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()

    def forward(self, x):
        ph = min(self.patch[0], x.shape[2])
        pw = min(self.patch[1], x.shape[3])
        amaxp = F.adaptive_max_pool2d(x, output_size=(ph, pw))
        aavgp = F.adaptive_avg_pool2d(x, output_size=(ph, pw))

        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(x.shape[0], -1, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(x.shape[0], -1, 1, 1)

        channel_interaction = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        weight = self.sigmoid(self.channel2x1(channel_interaction))

        attended = x * weight
        
        refined = self.post_conv(attended)
         
        return x+refined
        #alpha = torch.sigmoid(self.learnable_alpha)
        #return (1 - alpha) * x + alpha * refined
        #return x + self.drop_path(self.gamma * refined) #结合
        #return x + self.gamma * refined #mmdetection有用
        #return x + self.gate * refined


    
