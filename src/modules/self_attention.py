import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention1d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 8,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.g = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.h = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.attn_conv = nn.Conv2d(
            in_channels=in_channels // 2,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, ch, h, w = x.size()

        s = torch.bmm(
            self.f(x).view(-1, ch // 8, h * w // 4).permute(0, 2, 1),
            self.g(x).view(-1, ch // 8, h * w),
        )  # bmm(B X N X CH//8, B X CH//8 X N) -> B x N//4 x N
        beta = self.softmax(s)

        o = torch.bmm(
            self.h(x).view(-1, ch // 2, h * w // 4), beta
        )  # bmm(B x C//2 x N//4,  B x N//4 x N) -> B x C//2 x N
        o = self.attn_conv(o)  # B x C x N
        x = self.gamma * o + x

        return x
