from torch import nn


NORMS = {
    "BN": lambda num_channels: nn.BatchNorm2d(num_features=num_channels),
    "GN": lambda num_channels: nn.GroupNorm(num_groups=3, num_channels=num_channels),
    "LN": lambda num_channels: nn.GroupNorm(num_groups=1, num_channels=num_channels),
    "IN": lambda num_channels: nn.GroupNorm(
        num_groups=num_channels, num_channels=num_channels
    ),
}
