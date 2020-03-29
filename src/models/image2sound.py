import torch
import torch.nn as nn
from torch.nn import functional as F
from modules.inverted_residual import InvertedRes2d
from modules.residual import FirstBlockDown2d, BlockUpsample2d
from modules.self_attention import SelfAttention2d


class Discriminator(nn.Module):
    def __init__(self, self_attention=True, sn=True):
        super().__init__()

        # DOWN:
        self.dn_block1 = FirstBlockDown2d(
            in_channels=2,
            out_channels=16,
            activation="leaky_relu",
            normalization=None,
            downscale=False,
            seblock=False,
            sn=sn,
        )

        self.dn_block2 = InvertedRes2d(
            in_channels=16,
            planes=64,
            out_channels=32,
            dropout=0,
            activation="leaky_relu",
            normalization=None,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block3 = InvertedRes2d(
            in_channels=32,
            planes=128,
            out_channels=64,
            dropout=0,
            activation="leaky_relu",
            normalization=None,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.sa_layer = None
        if self_attention is True:
            self.sa_layer = SelfAttention2d(in_channels=64, sn=sn)

        self.dn_block4 = InvertedRes2d(
            in_channels=64,
            planes=256,
            out_channels=128,
            dropout=0,
            activation="leaky_relu",
            normalization=None,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block5 = InvertedRes2d(
            in_channels=128,
            planes=512,
            out_channels=256,
            dropout=0,
            activation="leaky_relu",
            normalization=None,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block6 = InvertedRes2d(
            in_channels=256,
            planes=512,
            out_channels=256,
            dropout=0,
            activation="leaky_relu",
            normalization=None,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block7 = InvertedRes2d(
            in_channels=256,
            planes=512,
            out_channels=256,
            dropout=0,
            activation="leaky_relu",
            normalization=None,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d([1, 1])

        self.output = nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=1,
                bias=False,
                padding=0,
                stride=1,
            )
        )

    def forward(self, input):
        # DOWN:
        #   BLOCK: Inverted Residual block
        #   ACTIVATION_FUNC: LReLU
        #   NORM: SN
        # Input Dimention: [B, 2, 128, 1024]
        # Dimention -> [B, 16, 128, 1024]
        dn = self.dn_block1(input)

        # Dimention -> [B, 32, 64, 512]
        dn = self.dn_block2(dn)

        # Dimention -> [B, 64, 32, 256]
        dn = self.dn_block3(dn)

        if self.sa_layer is not None:
            dn = self.sa_layer(dn)

        # Dimention -> [B, 128, 16, 128]
        dn = self.dn_block4(dn)

        # Dimention -> [B, 256, 8, 64]
        dn = self.dn_block5(dn)

        # Dimention -> [B, 256, 4, 32]
        dn = self.dn_block6(dn)

        # Dimention -> [B, 256, 2, 16]
        dn = self.dn_block7(dn)

        # Dimention -> [B, 256, 1, 1]
        dn = self.global_avg_pool(dn)

        # Dimention -> [B, 1, 1, 1]
        dn = self.output(dn)

        return dn
