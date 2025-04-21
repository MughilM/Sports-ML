# File: src/nets.py
# Author: Mughil Pari
# Creation Date: 2025-04-20
# 
# Contains all basic neural network definitions. Can range from image recognition
# models to text models, and beyond. Python encourages as many things
# to be defined in the same files as possible. These are used in conjunction
# with the Lightning Modules in the modules.py file.

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16

# A simple convolutional neural network with basic
# functionality. Good to get a basic model running...
class SimpleConvNet(nn.Module):
    def __init__(self, name: str = 'simple_conv', output_size: int = 1):
        super().__init__()

        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.model_2 = nn.Sequential(
            nn.Linear(in_features=6 * 16 * 16, out_features=1000),
            nn.Linear(1000, 500),
            nn.Linear(500, 250),
            nn.Linear(250, 120),
            nn.Linear(120, 60),
            nn.Linear(60, output_size)
        )

    def forward(self, x: torch.Tensor):
        x = self.model_1(x)
        x = x.view(-1, 6 * 16 * 16)
        return self.model_2(x)

# ResNet 50 model.
# Still something relatively basic.
class ResNet(nn.Module):
    def __init__(self, name: str = 'resnet50', output_size: int = 1, finetune: bool = False):
        super().__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # We need to freeze all the layers, so they
        # don't get trained and waste time
        if not finetune:
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False
        # Once it's frozen, we adapt the fully-connected
        # layer to the output size we need. This will
        # get trained.
        self.resnet.fc = nn.Linear(2048, output_size)

    def forward(self, x: torch.Tensor):
        return self.resnet(x)

# Vision Transformer - ViT
# A more complicated computer vision model
class ViT(nn.Module):
    def __init__(self, name: str = 'vit_b_165', output_size: int = 1, pretrained: bool = True):
        super().__init__()

        if pretrained:
            # Uses ImageNet V1's weights. Others are available...
            self.vit = vit_b_16(weights='IMAGENET1K_V1')
        else:
            self.vit = vit_b_16()
        # Change the last layer to output output_size neurons
        self.vit.heads = nn.Sequential(nn.Linear(in_features=768, out_features=output_size, bias=True))

    def forward(self, x: torch.Tensor):
        return self.vit(x)
