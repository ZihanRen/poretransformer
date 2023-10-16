import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from lpu3dnet.modules.components import *


class Decoder(nn.Module):
    def __init__(self, image_channels=1, latent_dim=256,num_group = 16):
        super(Decoder, self).__init__()
        # 2->4->8->16->32->64
        channels = [512, 256, 128,128, 64, 16, 16]
        num_res_blocks = 3
        resolution = 2

        in_channels = channels[0]
        layers = [nn.Conv3d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels,num_groups=num_group),
                  ResidualBlock(in_channels, in_channels,num_groups=num_group)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if (i != 0) & (i != len(channels) - 1):
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels,num_groups=num_group))
        layers.append(Swish())
        layers.append(nn.Conv3d(in_channels, image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



# test
enc = Decoder()
print( 'The architecture is'+'\n{}'.format(
    summary(enc,(20,256,2,2,2)) 
    ))