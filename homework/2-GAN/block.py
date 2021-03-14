import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, stride: int, padding: int,
                 in_norm: bool = False, conv_tr: bool = False,
                 add_act: bool = True, leaky: bool = False):
        super().__init__()
        self.layers = []
        if conv_tr:
            self.layers.append(nn.ConvTranspose2d(in_features, out_features,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  bias=False,
                                                  ))
        else:
            self.layers.append(nn.Conv2d(in_features, out_features,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False))
        if in_norm:
            self.layers.append(nn.InstanceNorm2d(out_features, affine=True))

        if add_act:
            if leaky:
                self.layers.append(nn.LeakyReLU(0.01))
            else:
                self.layers.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.tensor):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(Block(in_features, out_features,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          in_norm=True),
                                    Block(out_features, out_features,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          in_norm=True,
                                          add_act=False))

    def forward(self, x):
        return x + self.layers(x)