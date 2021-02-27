import torch
import torch.nn as nn
from block import Block


class Classifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()     
        self.cnns = nn.Sequential(
             # 1 x 64 x 64
             Block(1, 16, 
                   kernel_size=3, stride=2, padding=1),
             # 16 x 32 x 32
             Block(16, 16, 
                   kernel_size=3, stride=2, padding=1), 
             # 16 x 16 x 16
             Block(16, 16, 
                   kernel_size=3, stride=2, padding=1), 
             # 16 x 8 x 8
             Block(16, 32, 
                   kernel_size=3, stride=2, padding=1))
             # 32 x 4 x 4
        self.linears = nn.Sequential(nn.Linear(512, 256), 
                                     nn.LeakyReLU(0.2), 
                                     nn.Linear(256, n_classes))
        
        
    def forward(self, x):
        for layer in self.cnns:
            x = layer(x)
        return self.linears(x.view(x.shape[0], -1))
    
    def get_activations(self, x):
        x = self.cnns(x)
        x = self.linears[:-1](x.view(x.shape[0], -1))
        return x


class MLP(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features, 256), 
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(256, 256), 
                                    nn.LeakyReLU(0.2), 
                                    nn.Linear(256, n_classes))
    def forward(self, x):
        return self.layers(x.view(x.shape[0], -1))
