import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from block import Block


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
    
    def forward(self, x, l1_mode=False):
        l1_loss = 0.
        for module in chain(self.encoder.children(), self.decoder.children()):
            out = module(x, l1_mode)
            if l1_mode:
                x = out[0]
                l1_loss += out[1]
            else:
                x = out
        x = torch.sigmoid(x)
        return x, l1_loss
    
    def get_latent_features(self, x):
        return self.encoder(x)


class DummyAutoEncoder(AutoEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Block(1, 16, 
                                           kernel_size=3, stride=2, padding=1),
                                     Block(16, 32, 
                                           kernel_size=3, stride=2, padding=1), 
                                     Block(32, 32, 
                                           kernel_size=3, stride=2, padding=1), 
                                     Block(32, 32, 
                                           kernel_size=3, stride=2, padding=1, 
                                           only_conv=True))
        self.decoder = nn.Sequential(Block(32, 32,
                                           kernel_size=3, stride=1, padding=1, 
                                           upsample=True), 
                                     Block(32, 32,
                                           kernel_size=3, stride=1, padding=1, 
                                           upsample=True), 
                                     Block(32, 16,
                                           kernel_size=3, stride=1, padding=1, 
                                           upsample=True),
                                     Block(16, 16,
                                           kernel_size=3, stride=1, padding=1, 
                                           upsample=True), 
                                     Block(16, 16,
                                           kernel_size=3, stride=1, padding=1, 
                                           upsample=True),
                                     Block(16, 1, only_conv=True))



class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Block(1, 16, 
                                           kernel_size=3, stride=2, padding=1, 
                                           denoise=True),
                                     Block(16, 32, 
                                           kernel_size=3, stride=2, padding=1, 
                                           denoise=True), 
                                     Block(32, 32, 
                                           kernel_size=3, stride=2, padding=1, 
                                           denoise=True), 
                                     Block(32, 32, 
                                           kernel_size=3, stride=2, padding=1, 
                                           only_conv=True))
        self.decoder = nn.Sequential(Block(32, 32,
                                           kernel_size=3, stride=1, padding=1, 
                                           upsample=True, denoise=True), 
                                     Block(32, 32,
                                           kernel_size=3, stride=1, padding=1, 
                                           upsample=True, denoise=True), 
                                     Block(32, 16,
                                           kernel_size=3, stride=1, padding=1, 
                                           upsample=True, denoise=True),
                                     Block(16, 16,
                                           kernel_size=3, stride=1, padding=1, 
                                           upsample=True, denoise=True), 
                                     Block(16, 16,
                                           kernel_size=3, stride=1, padding=1, 
                                           upsample=True, denoise=True),
                                     Block(16, 1, only_conv=True))

