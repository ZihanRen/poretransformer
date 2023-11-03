import torch
import torch.nn as nn
import torch.nn.functional as F
        
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
    def __init__(self, in_channels, out_channels,num_groups):
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


def gradient_penalty(crit,real,fake,device="cpu"):
    BATCH_SIZE,C,H,W,L = real.shape
    epsilon = torch.rand(BATCH_SIZE, 1, 1, 1, 1).repeat(1,C,H,W,L).to(device)
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)

    # take the gradident of scores with respect to images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True
        )[0]

    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean( (gradient_norm-1)**2 )
    return penalty