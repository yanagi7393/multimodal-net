import torch
from torch import nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, activation="relu"):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            getattr(F, activation),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
