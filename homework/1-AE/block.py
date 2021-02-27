import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, 
                             mode='bilinear', 
                             align_corners=False, 
                             recompute_scale_factor=False)


class DenoiseLayer(nn.Module):
    def __init__(self, sigma=0.05, p=0.3):
        super().__init__()
        self.noise = Normal(loc=0.0, scale=sigma)
        self.dropout = nn.Dropout(p)
    
    def forward(self, x):
        noise_part = self.noise.sample(x.shape)
        if x.is_cuda:
            noise_part = noise_part.to('cuda')
        return self.dropout(x + noise_part)


class Block(nn.Module):
    def __init__(self, in_features, out_features, 
                 kernel_size=3, stride=2, padding=1, 
                 upsample=False, only_conv=False, denoise=False):
        super().__init__()
        self.conv = nn.Conv2d(in_features, 
                              out_features, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=False)
        
        if only_conv:
            self.before_conv = nn.Sequential()
            self.after_conv = nn.Sequential()
            return

        self.before_conv = []
        self.after_conv = []
        if upsample:
            self.before_conv.append(UpSample())   
        if denoise:
            self.before_conv.append(DenoiseLayer())
        self.after_conv.extend([nn.BatchNorm2d(out_features), 
                                nn.LeakyReLU(0.2)])
        self.before_conv = nn.Sequential(*self.before_conv)
        self.after_conv = nn.Sequential(*self.after_conv)
        

 
    def forward(self, x, l1_mode=False):
        x = self.before_conv(x)
        x_conv = self.conv(x)
        x = self.after_conv(x_conv)
        if l1_mode:
            return x, torch.abs(x_conv).sum(dim=1).mean()
        return x
