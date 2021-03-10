import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


class ResNetClassifier(ResNet):
    def __init__(self, n_classes=10):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=n_classes)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

    def forward(self, x):
        x = self.get_activations(x)
        x = self.fc(x)
        return x

    def get_activations(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
