import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
        
class GroupNorm(nn.Module):
    def __init__(self, channels,num_groups):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Resnet block input channel = output channel
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,num_groups=16):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels,num_groups=num_groups),
            Swish(),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels,num_groups=num_groups),
            Swish(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv3d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)

class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv3d(channels, channels, 3, 2, 0)

    def forward(self, x):
        # 
        pad = (1, 1, 1, 1,1, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)