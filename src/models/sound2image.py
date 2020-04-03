import torch
import torch.nn as nn
from torch.nn import functional as F
from modules.inverted_residual import InvertedRes2d
from modules.residual import FirstBlockDown2d, BlockUpsample2d
from modules.upsample import Upsample
from modules.self_attention import SelfAttention2d
from modules.norms import NORMS


class Generator(nn.Module):
    def __init__(self, self_attention=True, sn=False, norm="BIN"):
        super().__init__()

        # DOWN:
        self.dn_block1 = FirstBlockDown2d(
            in_channels=2,
            out_channels=16,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block2 = InvertedRes2d(
            in_channels=16,
            planes=32,  # 64
            out_channels=32,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block3 = InvertedRes2d(
            in_channels=32,
            planes=64,  # 128
            out_channels=64,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block4 = InvertedRes2d(
            in_channels=64,
            planes=128,  # 256
            out_channels=128,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block5 = InvertedRes2d(
            in_channels=128,
            planes=256,  # 512
            out_channels=256,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block6 = InvertedRes2d(
            in_channels=256,
            planes=512,
            out_channels=512,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d([2, 2])

        # SKIP:
        self.skip1 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=[8, 1],
            stride=[8, 1],
            padding=[0, 0],
            groups=512,
        )
        self.skip2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=[8, 1],
            stride=[8, 1],
            padding=[0, 0],
            groups=256,
        )
        self.skip3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=[8, 1],
            stride=[8, 1],
            padding=[0, 0],
            groups=128,
        )
        self.skip4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=[8, 1],
            stride=[8, 1],
            padding=[0, 0],
            groups=64,
        )

        # UP:
        self.up_block1 = BlockUpsample2d(
            in_channels=512,
            out_channels=256,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.up_block2 = BlockUpsample2d(
            in_channels=256,
            out_channels=256,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.up_block3 = BlockUpsample2d(
            in_channels=256,
            out_channels=128,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.up_block4 = BlockUpsample2d(
            in_channels=128,
            out_channels=128,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.up_block5 = BlockUpsample2d(
            in_channels=128,
            out_channels=64,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.sa_layer = None
        if self_attention is True:
            self.sa_layer = SelfAttention2d(in_channels=64, sn=sn)

        self.up_block6 = BlockUpsample2d(
            in_channels=64,
            out_channels=32,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.up_block7 = BlockUpsample2d(
            in_channels=32,
            out_channels=16,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.last_conv = nn.Conv2d(
            in_channels=16,
            out_channels=3,
            kernel_size=1,
            bias=False,
            padding=0,
            stride=1,
        )

        self.last_tanh = nn.Tanh()

        #########################
        # RECONSTRUCTION LAYERS #
        #########################
        self.global_avg_up_pool = Upsample(scale_factor=[8, 1], mode="bilinear")

        self.rec_block1 = BlockUpsample2d(
            in_channels=512,
            out_channels=256,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.rec_block2 = BlockUpsample2d(
            in_channels=256,
            out_channels=128,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.rec_block3 = BlockUpsample2d(
            in_channels=128,
            out_channels=64,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.rec_block4 = BlockUpsample2d(
            in_channels=64,
            out_channels=32,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.rec_block5 = BlockUpsample2d(
            in_channels=32,
            out_channels=16,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
        )

        self.last_rec_up = Upsample(scale_factor=2, mode="bilinear")

        self.last_rec_conv = nn.Conv2d(
            in_channels=16,
            out_channels=2,
            kernel_size=1,
            bias=False,
            padding=0,
            stride=1,
        )

    def reconstruct(self, input):
        # DOWN:
        #   BLOCK: Inverted Residual block
        #   ACTIVATION_FUNC: LReLU
        #   NORM: IN
        # Input Dimention: [B, 2, 1024, 128]
        # Dimention -> [B, 16, 512, 64]
        dn = self.dn_block1(input)

        # Dimention -> [B, 32, 256, 32]
        dn = self.dn_block2(dn)

        # Dimention -> [B, 64, 128, 16]
        dn3 = self.dn_block3(dn)

        # Dimention -> [B, 128, 64, 8]
        dn4 = self.dn_block4(dn3)

        # Dimention -> [B, 256, 32, 4]
        dn5 = self.dn_block5(dn4)

        # Dimention -> [B, 512, 16, 2]
        dn6 = self.dn_block6(dn5)

        # Dimention -> [B, 512, 2, 2]
        latent_vec = self.global_avg_pool(dn6)

    def forward(self, input):
        # DOWN:
        #   BLOCK: Inverted Residual block
        #   ACTIVATION_FUNC: LReLU
        #   NORM: IN
        # Input Dimention: [B, 2, 1024, 128]
        # Dimention -> [B, 16, 512, 64]
        dn = self.dn_block1(input)

        # Dimention -> [B, 32, 256, 32]
        dn = self.dn_block2(dn)

        # Dimention -> [B, 64, 128, 16]
        dn3 = self.dn_block3(dn)

        # Dimention -> [B, 128, 64, 8]
        dn4 = self.dn_block4(dn3)

        # Dimention -> [B, 256, 32, 4]
        dn5 = self.dn_block5(dn4)

        # Dimention -> [B, 512, 16, 2]
        dn6 = self.dn_block6(dn5)

        # Dimention -> [B, 512, 2, 2]
        latent_vec = self.global_avg_pool(dn6)

        # UP:
        #   BLOCK: Residual block
        #   ACTIVATION_FUNC: ReLU
        #   NORM: BIN (AdaIN?)
        # Dimention -> [B, 512, 2, 2] with drop_out + Conv_spatial_wise([B, 512, 16, 2] -> [B, 512, 2, 2])
        skip1 = self.skip1(dn6)
        up = latent_vec + skip1

        # Dimention -> [B, 256, 4, 4] with drop_out + Conv_spatial_wise([B, 256, 32, 4] -> [B, 256, 4, 4])
        skip2 = self.skip2(dn5)
        up = self.up_block1(up) + skip2

        # Dimention -> [B, 256, 8, 8] with drop_out + Conv_spatial_wise([B, 128, 64, 8] -> [B, 256, 8, 8])
        skip3 = self.skip3(dn4)
        up = self.up_block2(up) + skip3

        # Dimention -> [B, 128, 16, 16] Conv_spatial_wise([B, 64, 128, 16] -> [B, 128, 16, 16])
        skip4 = self.skip4(dn3)
        up = self.up_block3(up) + skip4

        # Dimention -> [B, 128, 32, 32]
        up = self.up_block4(up)

        # Dimention -> [B, 64, 64, 64]
        up = self.up_block5(up)

        if self.sa_layer is not None:
            up = self.sa_layer(up)

        # Dimention -> [B, 32, 128, 128]
        up = self.up_block6(up)

        # Dimention -> [B, 16, 256, 256]
        up = self.up_block7(up)

        # Dimention -> [B, 3, 256, 256]
        up = self.last_tanh(self.last_conv(up))

        ########################
        # RECONSTRUCTION FLOWS #
        ########################
        # Dimention -> [B, 512, 16, 2]
        up_pooled = self.global_avg_up_pool(latent_vec)

        # Dimention -> [B, 256, 32, 4]
        rec1 = self.rec_block1(up_pooled)

        # Dimention -> [B, 128, 64, 8]
        rec2 = self.rec_block2(rec1)

        # Dimention -> [B, 64, 128, 16]
        rec3 = self.rec_block3(rec2)

        # Dimention -> [B, 32, 256, 32]
        rec4 = self.rec_block4(rec3)

        # Dimention -> [B, 16, 512, 64]
        rec5 = self.rec_block5(rec4)

        # Dimention -> [B, 2, 1024, 128]
        reconstructed = self.last_rec_conv(self.last_rec_up(rec5))

        return reconstructed, up


class Discriminator(nn.Module):
    def __init__(self, self_attention=True, sn=True, norm="IN"):
        super().__init__()

        # DOWN:
        self.dn_block1 = FirstBlockDown2d(
            in_channels=3,
            out_channels=8,
            activation="leaky_relu",
            normalization=norm,
            downscale=False,
            seblock=False,
            sn=sn,
        )

        self.dn_block2 = InvertedRes2d(
            in_channels=8,
            planes=16,  # 64
            out_channels=16,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block3 = InvertedRes2d(
            in_channels=16,
            planes=32,  # 128
            out_channels=32,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.sa_layer = None
        if self_attention is True:
            self.sa_layer = SelfAttention2d(in_channels=32, sn=sn)

        self.dn_block4 = InvertedRes2d(
            in_channels=32,
            planes=64,  # 256
            out_channels=64,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block5 = InvertedRes2d(
            in_channels=64,
            planes=64,  # 256
            out_channels=64,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block6 = InvertedRes2d(
            in_channels=64,
            planes=128,  # 512
            out_channels=128,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.dn_block7 = InvertedRes2d(
            in_channels=128,
            planes=256,  # 512
            out_channels=256,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d([1, 1])

        if sn is True:
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
        else:
            self.output = nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=1,
                bias=False,
                padding=0,
                stride=1,
            )

    def forward(self, input):
        # DOWN:
        #   BLOCK: Inverted Residual block
        #   ACTIVATION_FUNC: LReLU
        #   NORM: SN
        # Input Dimention: [B, 3, 256, 256]
        # Dimention -> [B, 8, 256, 256]
        dn = self.dn_block1(input)

        # Dimention -> [B, 16, 128, 128]
        dn = self.dn_block2(dn)

        # Get feature vector
        feature_vec = dn

        # Dimention -> [B, 32, 64, 64]
        dn = self.dn_block3(dn)

        if self.sa_layer is not None:
            dn = self.sa_layer(dn)

        # Dimention -> [B, 64, 32, 32]
        dn = self.dn_block4(dn)

        # Dimention -> [B, 64, 16, 16]
        dn = self.dn_block5(dn)

        # Dimention -> [B, 128, 8, 8]
        dn = self.dn_block6(dn)

        # Dimention -> [B, 128, 4, 4]
        dn = self.dn_block7(dn)

        # Dimention -> [B, 128, 1, 1]
        dn = self.global_avg_pool(dn)

        # Dimention -> [B, 1, 1, 1]
        dn = self.output(dn)

        return feature_vec, dn
