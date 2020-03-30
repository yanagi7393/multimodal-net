import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import NORMS, perform_sn
from .seblock import SEBlock
from .upsample import Upsample


class FirstBlockDown2d(nn.Module):
    """
    This is for first block with skip connection, without pre-activation structure.
    This is more good way to prevent reducing information
    instead of using convolution on the first layer before bottlneck block.

    1. Selectable activation function
    2. Selectable Normalization method
    3. Selectable Squeeze excitation block 

    if downscale is False -> output feature size is same with input
    if downscale is True -> output feature size is output_size = math.ceil(input_size / 2)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilation=1,
        activation="relu",
        normalization="bn",
        downscale=False,
        seblock=False,
        init_channels=None,
        init_normalization=None,
        sn=False,
    ):
        super().__init__()

        self.normalization = normalization
        self.init_normalization = init_normalization

        stride = 1
        if downscale is True:
            stride = 2

        self.channel_compressor = None
        if (init_channels is not None) and (init_channels != in_channels):
            self.channel_compressor = perform_sn(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=init_channels,
                    kernel_size=1,
                    bias=False,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
            )

            if init_normalization is not None:
                self.init_n = NORMS[normalization.upper()](num_channels=init_channels)

            in_channels = init_channels

        self.conv1 = perform_sn(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=dilation,
                bias=False,
                padding=1,
                stride=stride,
            ),
            sn=sn,
        )
        if normalization is not None:
            self.n1 = NORMS[normalization.upper()](num_channels=out_channels)

        self.conv2 = perform_sn(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=dilation,
                bias=False,
                padding=1,
                stride=1,
            ),
            sn=sn,
        )

        self.seblock = None
        if seblock is True:
            self.seblock = SEBlock(
                in_channels=out_channels, activation=activation, sn=sn
            )

        if downscale is True:
            if in_channels == out_channels:
                self.conv3 = perform_sn(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        bias=False,
                        padding=1,
                        stride=stride,
                        groups=in_channels,
                    ),
                    sn=sn,
                )

            else:
                self.conv3 = perform_sn(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        bias=False,
                        padding=1,
                        stride=stride,
                    ),
                    sn=sn,
                )

        elif in_channels != out_channels:
            self.conv3 = perform_sn(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
            )

        else:
            self.conv3 = None

        self.act = getattr(F, activation)

    def forward(self, x):
        h = x

        if self.channel_compressor is not None:
            h = self.channel_compressor(h)

            if self.init_normalization is not None:
                h = self.init_n(h)

        h = self.conv1(h)
        if self.normalization is not None:
            h = self.n1(h)
        h = self.act(h)

        h = self.conv2(h)

        if self.conv3:
            x = self.conv3(x)

        if self.seblock is not None:
            h = self.seblock(h)

        h = h + x

        return h


class BlockDown2d(nn.Module):
    """
    1. Preactivation-resnet structure
    2. Use SpatialDropout
    3. Selectable activation function
    4. Selectable Normalization method
    5. Selectable Squeeze excitation block 

    if downscale is False -> output feature size is same with input
    if downscale is True -> output feature size is output_size = math.ceil(input_size / 2)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0,
        dilation=1,
        activation="relu",
        normalization="bn",
        downscale=False,
        seblock=False,
        sn=False,
    ):
        super().__init__()

        self.normalization = normalization

        stride = 1
        if downscale is True:
            stride = 2

        if normalization is not None:
            self.n1 = NORMS[normalization.upper()](num_channels=in_channels)
        self.conv1 = perform_sn(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=dilation,
                bias=False,
                padding=1,
                stride=stride,
            ),
            sn=sn,
        )

        if normalization is not None:
            self.n2 = NORMS[normalization.upper()](num_channels=out_channels)
        self.conv2 = perform_sn(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=dilation,
                bias=False,
                padding=1,
                stride=1,
            ),
            sn=sn,
        )

        self.seblock = None
        if seblock is True:
            self.seblock = SEBlock(
                in_channels=out_channels, activation=activation, sn=sn
            )

        if downscale is True:
            if in_channels == out_channels:
                self.conv3 = perform_sn(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        bias=False,
                        padding=1,
                        stride=stride,
                        groups=in_channels,
                    ),
                    sn=sn,
                )

            else:
                self.conv3 = perform_sn(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        bias=False,
                        padding=1,
                        stride=stride,
                    ),
                    sn=sn,
                )

        elif in_channels != out_channels:
            self.conv3 = perform_sn(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
            )

        else:
            self.conv3 = None

        # Dropout2d can cover 2d.
        self.dropout = nn.Dropout(dropout)
        self.act = getattr(F, activation)

    def forward(self, x):
        h = x

        if self.normalization is not None:
            h = self.n1(h)
        h = self.conv1(self.act(h))

        if self.normalization is not None:
            h = self.n2(h)
        h = self.conv2(self.dropout(self.act(h)))

        if self.conv3:
            x = self.conv3(x)

        if self.seblock is not None:
            h = self.seblock(h)

        h = h + x

        return h


class BlockUpsample2d(nn.Module):
    """
    1. Preactivation-resnet structure
    2. Use SpatialDropout
    3. Selectable activation function
    4. Selectable Normalization method
    5. Selectable Squeeze excitation block 

    if upscale is False -> output feature size is same with input
    if upscale is True -> output feature size is input_size * 2
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0,
        dilation=1,
        activation="relu",
        normalization="bn",
        seblock=False,
        sn=False,
    ):
        super().__init__()

        self.upsample = Upsample(scale_factor=2, mode="bilinear")

        self.res_block = BlockDown2d(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            dilation=dilation,
            activation=activation,
            normalization=normalization,
            downscale=False,
            seblock=seblock,
            sn=sn,
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.res_block(x)

        return x


class BlockUp2d(nn.Module):
    """
    1. Preactivation-resnet structure
    2. Use SpatialDropout
    3. Selectable activation function
    4. Selectable Normalization method
    5. Selectable Squeeze excitation block 

    if upscale is False -> output feature size is same with input
    if upscale is True -> output feature size is input_size * 2
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0,
        activation="relu",
        normalization="bn",
        upscale=False,
        seblock=False,
        sn=False,
    ):
        super().__init__()

        self.normalization = normalization

        stride = 1
        if upscale is True:
            stride = 2

        if normalization is not None:
            self.n1 = NORMS[normalization.upper()](num_channels=in_channels)
        self.deconv1 = perform_sn(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False,
                padding=1,
                stride=stride,
                output_padding=int(stride > 1),
            ),
            sn=sn,
        )

        if normalization is not None:
            self.n2 = NORMS[normalization.upper()](num_channels=out_channels)
        self.deconv2 = perform_sn(
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False,
                padding=1,
                stride=1,
            ),
            sn=sn,
        )

        self.seblock = None
        if seblock is True:
            self.seblock = SEBlock(
                in_channels=out_channels, activation=activation, sn=sn
            )

        if upscale is True:
            self.deconv3 = perform_sn(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    bias=False,
                    padding=1,
                    stride=stride,
                    output_padding=int(stride > 1),
                ),
                sn=sn,
            )

        elif in_channels != out_channels:
            self.deconv3 = perform_sn(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
            )

        else:
            self.deconv3 = None

        # Dropout2d can cover 2d.
        self.dropout = nn.Dropout(dropout)
        self.act = getattr(F, activation)

    def forward(self, x):
        h = x

        if self.normalization is not None:
            h = self.n1(h)
        h = self.deconv1(self.act(h))

        if self.normalization is not None:
            h = self.n2(h)
        h = self.deconv2(self.dropout(self.act(h)))

        if self.deconv3:
            x = self.deconv3(x)

        if self.seblock is not None:
            h = self.seblock(h)

        h = h + x

        return h


class BottleneckBlockDown2d(nn.Module):
    """
    1. Preactivation-resnet structure
    2. Use SpatialDropout
    3. Selectable activation function
    4. Selectable Normalization method
    5. Selectable Squeeze excitation block 

    if downscale is False -> output feature size is same with input
    if downscale is True -> output feature size is output_size = math.ceil(input_size / 2)
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
        sn=False,
    ):
        super().__init__()

        self.normalization = normalization

        stride = 1
        if downscale is True:
            stride = 2

        if normalization is not None:
            self.n1 = NORMS[normalization.upper()](num_channels=in_channels)
        self.conv1 = perform_sn(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=planes,
                kernel_size=1,
                bias=False,
                padding=0,
                stride=1,
            ),
            sn=sn,
        )

        if normalization is not None:
            self.n2 = NORMS[normalization.upper()](num_channels=planes)
        self.conv2 = perform_sn(
            nn.Conv2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                dilation=dilation,
                bias=False,
                padding=1,
                stride=stride,
            ),
            sn=sn,
        )

        if normalization is not None:
            self.n3 = NORMS[normalization.upper()](num_channels=planes)
        self.conv3 = perform_sn(
            nn.Conv2d(
                in_channels=planes,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                padding=0,
                stride=1,
            ),
            sn=sn,
        )

        self.seblock = None
        if seblock is True:
            self.seblock = SEBlock(
                in_channels=out_channels, activation=activation, sn=sn
            )

        if downscale is True:
            if in_channels == out_channels:
                self.conv4 = perform_sn(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        bias=False,
                        padding=1,
                        stride=stride,
                        groups=in_channels,
                    ),
                    sn=sn,
                )
            else:
                self.conv4 = perform_sn(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        bias=False,
                        padding=1,
                        stride=stride,
                    ),
                    sn=sn,
                )

        elif in_channels != out_channels:
            self.conv4 = perform_sn(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
            )
        else:
            self.conv4 = None

        # Dropout2d can cover 2d.
        self.dropout = nn.Dropout(dropout)
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


class BottleneckBlockUpsample2d(nn.Module):
    """
    1. Preactivation-resnet structure
    2. Use SpatialDropout
    3. Selectable activation function
    4. Selectable Normalization method
    5. Selectable Squeeze excitation block 

    if upscale is False -> output feature size is same with input
    if upscale is True -> output feature size is input_size * 2
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
        sn=False,
    ):
        super().__init__()

        self.upsample = Upsample(scale_factor=2, mode="bilinear")

        self.res_bottleneck_block = BottleneckBlockDown2d(
            in_channels=in_channels,
            planes=planes,
            out_channels=out_channels,
            dropout=dropout,
            dilation=dilation,
            activation=activation,
            normalization=normalization,
            downscale=False,
            seblock=seblock,
            sn=sn,
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.res_bottleneck_block(x)

        return x


class BottleneckBlockUp2d(nn.Module):
    """
    1. Preactivation-resnet structure
    2. Use SpatialDropout
    3. Selectable activation function
    4. Selectable Normalization method
    5. Selectable Squeeze excitation block 

    if upscale is False -> output feature size is same with input
    if upscale is True -> output feature size is input_size * 2
    """

    def __init__(
        self,
        in_channels,
        planes,
        out_channels,
        dropout=0,
        activation="relu",
        normalization="bn",
        upscale=False,
        seblock=False,
        sn=False,
    ):
        super().__init__()

        self.normalization = normalization
        self.upscale = upscale

        stride = 1
        if upscale is True:
            stride = 2

        if normalization is not None:
            self.n1 = NORMS[normalization.upper()](num_channels=in_channels)
        self.deconv1 = perform_sn(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=planes,
                kernel_size=1,
                bias=False,
                padding=0,
                stride=1,
            ),
            sn=sn,
        )

        if normalization is not None:
            self.n2 = NORMS[normalization.upper()](num_channels=planes)
        self.deconv2 = perform_sn(
            nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                bias=False,
                padding=1,
                stride=stride,
                output_padding=int(stride > 1),
            ),
            sn=sn,
        )

        if normalization is not None:
            self.n3 = NORMS[normalization.upper()](num_channels=planes)
        self.deconv3 = perform_sn(
            nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                padding=0,
                stride=1,
            ),
            sn=sn,
        )

        self.seblock = None
        if seblock is True:
            self.seblock = SEBlock(
                in_channels=out_channels, activation=activation, sn=sn
            )

        if upscale is True:
            self.deconv4 = perform_sn(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    bias=False,
                    padding=1,
                    stride=stride,
                    output_padding=int(stride > 1),
                ),
                sn=sn,
            )

        elif in_channels != out_channels:
            self.deconv4 = perform_sn(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
            )

        else:
            self.deconv4 = None

        # Dropout2d can cover 2d.
        self.dropout = nn.Dropout(dropout)
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
