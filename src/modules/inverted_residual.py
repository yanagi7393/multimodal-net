import torch.nn as nn
import torch.nn.functional as F
from ml.torch.modules.norms import NORMS
from ml.torch.modules.seblock import SEBlock


class InvertedRes2d(nn.Module):
    """
    Inverted Residual block # MobileNetV2
    with pre-activation structure

    1. Use SpatialDropout
    2. Selectable activation function
    3. Selectable Normalization method
    4. Selectable Squeeze excitation block 
    """

    def __init__(
        self,
        in_channels,
        planes,
        out_channels,
        dropout=0,
        dilation=1,
        activation="relu",
        normalization="bn",
        downscale=False,
        seblock=False,
    ):
        super().__init__()
        if planes <= in_channels:
            raise ValueError(
                f"planes({planes}) should be lager than in_channels({in_channels})"
            )

        self.normalization = normalization

        stride = 1
        if downscale is True:
            stride = 2

        if normalization is not None:
            self.n1 = NORMS[normalization.upper()](num_channels=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=planes,
            kernel_size=1,
            bias=False,
            padding=0,
            stride=1,
        )

        if normalization is not None:
            self.n2 = NORMS[normalization.upper()](num_channels=planes)
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            dilation=dilation,
            bias=False,
            padding=1,
            stride=stride,
            groups=planes,
        )

        if normalization is not None:
            self.n3 = NORMS[normalization.upper()](num_channels=planes)
        self.conv3 = nn.Conv2d(
            in_channels=planes,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            padding=0,
            stride=1,
        )

        self.seblock = None
        if seblock is True:
            self.seblock = SEBlock(in_channels=out_channels, activation=activation)

        if downscale is True:
            if in_channels == out_channels:
                self.conv4 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    bias=False,
                    padding=1,
                    stride=stride,
                    groups=in_channels,
                )
            else:
                self.conv4 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    bias=False,
                    padding=1,
                    stride=stride,
                )

        elif in_channels != out_channels:
            self.conv4 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                padding=0,
                stride=1,
            )
        else:
            self.conv4 = None

        # Dropout2d can cover 1d.
        self.dropout = nn.Dropout2d(dropout)
        self.act = getattr(F, activation)

    def forward(self, x):
        h = x

        if self.normalization is not None:
            h = self.n1(h)
        h = self.conv1(self.act(h))

        if self.normalization is not None:
            h = self.n2(h)
        h = self.conv2(self.act(h))

        if self.normalization is not None:
            h = self.n3(h)
        h = self.conv3(self.dropout((self.act(h))))

        if self.conv4:
            x = self.conv4(x)

        if self.seblock is not None:
            h = self.seblock(h)

        h = h + x

        return h


class InvertedResUpsample2d(nn.Module):
    """
    Inverted Residual block # MobileNetV2
    with pre-activation structure

    1. Use SpatialDropout
    2. Selectable activation function
    3. Selectable Normalization method
    4. Selectable Squeeze excitation block 
    """

    def __init__(
        self,
        in_channels,
        planes,
        out_channels,
        dropout=0,
        dilation=1,
        activation="relu",
        normalization="bn",
        seblock=False,
    ):
        super().__init__()

        self.upsample = Upsample(scale_factor=2, mode="bilinear")

        self.inverted_residual_block = InvertedRes2d(
            in_channels=in_channels,
            planes=planes,
            out_channels=out_channels,
            dropout=dropout,
            dilation=dilation,
            activation=activation,
            normalization=normalization,
            downscale=False,
            seblock=seblock,
        )

        def forward(self, x):
            x = self.upsample(x)
            x = self.inverted_residual_block(x)

            return x
