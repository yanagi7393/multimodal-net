from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
import torch


NORMS = {
    "BN": lambda num_channels: nn.BatchNorm2d(num_features=num_channels),
    "GN": lambda num_channels: nn.GroupNorm(num_groups=3, num_channels=num_channels)
    if num_channels % 3 == 0
    else nn.GroupNorm(num_groups=2, num_channels=num_channels),
    "LN": lambda num_channels: nn.GroupNorm(num_groups=1, num_channels=num_channels),
    "IN": lambda num_channels: nn.GroupNorm(
        num_groups=num_channels, num_channels=num_channels
    ),
    "BIN": lambda num_channels: _BatchInstanceNorm2d(num_features=num_channels),
}


def perform_sn(module, sn=False):
    if sn is False:
        return module

    if sn is True:
        return nn.utils.spectral_norm(module)


class _BatchInstanceNorm2d(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, "bin_gate", True)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        out_bn = F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            bn_w,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )

        # Instance norm
        b, c = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None, True, self.momentum, self.eps
        )
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in
