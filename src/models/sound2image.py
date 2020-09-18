import torch
import torch.nn as nn
from torch.nn import functional as F
from modules.inverted_residual import InvertedRes2d
from modules.residual import FirstBlockDown2d, BlockUpsample2d
from modules.upsample import Upsample
from modules.self_attention import SelfAttention2d
from modules.norms import NORMS, perform_sn


class Generator(nn.Module):
    def __init__(
        self, self_attention=True, sn=False, norm="BN", dropout=0, use_class=False
    ):
        super().__init__()
        self.use_class = use_class

        bias = False
        if norm is None:
            bias = True

        # DOWN:
        self.dn_block1 = FirstBlockDown2d(
            in_channels=2,
            out_channels=16,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
            bias=bias,
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
            bias=bias,
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
            bias=bias,
        )

        self.sa_layer1 = None
        if self_attention is True:
            self.sa_layer1 = SelfAttention2d(in_channels=64, sn=sn)

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
            bias=bias,
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
            bias=bias,
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
            bias=bias,
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
            dropout=dropout,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
            bias=bias,
        )

        self.up_block2 = BlockUpsample2d(
            in_channels=256,
            out_channels=256,
            dropout=dropout,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
            bias=bias,
        )

        self.up_block3 = BlockUpsample2d(
            in_channels=256,
            out_channels=128,
            dropout=dropout,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
            bias=bias,
        )

        self.up_block4 = BlockUpsample2d(
            in_channels=128,
            out_channels=128,
            dropout=dropout,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
            bias=bias,
        )

        self.up_block5 = BlockUpsample2d(
            in_channels=128,
            out_channels=64,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
            bias=bias,
        )

        self.sa_layer2 = None
        if self_attention is True:
            self.sa_layer2 = SelfAttention2d(in_channels=64, sn=sn)

        self.up_block6 = BlockUpsample2d(
            in_channels=64,
            out_channels=32,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
            bias=bias,
        )

        self.up_block7 = BlockUpsample2d(
            in_channels=32,
            out_channels=16,
            dropout=0,
            activation="relu",
            normalization=norm,
            seblock=False,
            sn=sn,
            bias=bias,
        )

        self.last_norm = None
        if norm is not None:
            self.last_norm = NORMS[norm](num_channels=16)

        self.last_act = getattr(F, "relu")

        self.last_conv = perform_sn(
            nn.Conv2d(
                in_channels=16,
                out_channels=3,
                kernel_size=1,
                bias=True,
                padding=0,
                stride=1,
            ),
            sn=sn,
        )

        self.last_tanh = nn.Tanh()

        ################
        # class output #
        ################

        self.class_norm = None
        self.class_act = None
        self.class_conv = None
        if use_class is True:

            if norm is not None:
                self.class_norm = NORMS[norm](num_channels=512)

            self.class_act = getattr(F, "leaky_relu")

            self.class_conv = perform_sn(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1000,
                    kernel_size=2,
                    bias=True,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
            )

    def get_latent_output(self, dn6):
        latent_vec = self.class_norm(dn6)
        latent_vec = self.class_act(latent_vec)

        latent_vec = self.global_avg_pool(latent_vec)

        class_output = self.class_conv(latent_vec)

        return latent_vec, class_output

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

        if self.sa_layer1 is not None:
            dn3 = self.sa_layer1(dn3)

        # Dimention -> [B, 128, 64, 8]
        dn4 = self.dn_block4(dn3)

        # Dimention -> [B, 256, 32, 4]
        dn5 = self.dn_block5(dn4)

        # Dimention -> [B, 512, 16, 2]
        dn6 = self.dn_block6(dn5)

        # Dimention -> [B, 512, 2, 2]
        latent_vec, class_output = self.get_latent_output(dn6)

        if self.use_class is not True:
            class_output = None

        # UP:
        #   BLOCK: Residual block
        #   ACTIVATION_FUNC: ReLU
        #   NORM: BN (AdaIN?)
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

        if self.sa_layer2 is not None:
            up = self.sa_layer2(up)

        # Dimention -> [B, 32, 128, 128]
        up = self.up_block6(up)

        # Dimention -> [B, 16, 256, 256]
        up = self.up_block7(up)

        # last norm
        if self.last_norm is not None:
            up = self.last_norm(up)

        up = self.last_act(up)

        # Dimention -> [B, 3, 256, 256]
        up = self.last_tanh(self.last_conv(up))

        return up, class_output


class Discriminator(nn.Module):
    def __init__(self, self_attention=True, sn=True, norm=None, use_class=False):
        super().__init__()
        self.use_class = use_class

        bias = False
        if norm is None:
            bias = True

        # DOWN:
        self.dn_block1 = FirstBlockDown2d(
            in_channels=3,
            out_channels=8,
            activation="leaky_relu",
            normalization=norm,
            downscale=False,
            seblock=False,
            sn=sn,
            bias=bias,
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
            bias=bias,
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
            bias=bias,
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
            bias=bias,
        )

        self.dn_block5 = InvertedRes2d(
            in_channels=64,
            planes=128,  # 256
            out_channels=128,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
            bias=bias,
        )

        self.dn_block6 = InvertedRes2d(
            in_channels=128,
            planes=256,  # 512
            out_channels=256,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
            bias=bias,
        )

        self.dn_block7 = InvertedRes2d(
            in_channels=256,
            planes=512,  # 512
            out_channels=512,
            dropout=0,
            activation="leaky_relu",
            normalization=norm,
            downscale=True,
            seblock=False,
            sn=sn,
            bias=bias,
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d([1, 1])

        self.last_norm = None
        if norm is not None:
            self.last_norm = NORMS[norm](num_channels=512)

        self.last_act = getattr(F, "leaky_relu")

        if use_class is True:
            self.output = perform_sn(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1001,
                    kernel_size=1,
                    bias=True,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
            )
        else:
            self.output = perform_sn(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1,
                    kernel_size=1,
                    bias=True,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
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
        feat_vec_1 = dn

        # Dimention -> [B, 32, 64, 64]
        dn = self.dn_block3(dn)
        feat_vec_2 = dn

        if self.sa_layer is not None:
            dn = self.sa_layer(dn)

        # Dimention -> [B, 64, 32, 32]
        dn = self.dn_block4(dn)
        feat_vec_3 = dn

        # Dimention -> [B, 128, 16, 16]
        dn = self.dn_block5(dn)
        feat_vec_4 = dn

        # Dimention -> [B, 256, 8, 8]
        dn = self.dn_block6(dn)
        feat_vec_5 = dn

        # Dimention -> [B, 512, 4, 4]
        dn = self.dn_block7(dn)

        # last norm
        if self.last_norm is not None:
            dn = self.last_norm(dn)

        # Dimention -> [B, 512, 1, 1]
        dn = self.last_act(dn)

        dn = self.global_avg_pool(dn)

        output = self.output(dn)

        if self.use_class is True:
            # Dimention -> [B, 1001, 1, 1]
            class_output, discriminated = output.split(1000, dim=1)

            return (
                [feat_vec_1, feat_vec_2, feat_vec_3, feat_vec_4, feat_vec_5],
                [discriminated],
                class_output,
            )

        else:
            return (
                [feat_vec_1, feat_vec_2, feat_vec_3, feat_vec_4, feat_vec_5],
                [output],
                None,
            )
