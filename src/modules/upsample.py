import torch
from torch import nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode="bilinear"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
