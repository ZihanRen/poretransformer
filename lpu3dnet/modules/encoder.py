import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from lpu3dnet.modules.components import *
    

class Encoder(nn.Module):
    def __init__(self, image_channels=1, latent_dim=256,num_groups=16):
        super(Encoder, self).__init__()
        # compress high dimensional 3D tensor to low dimensional 3D tensor (256,2,2,2)
        # architecture 64->32->16->8->4->2. Every downsample has two residual block

        channels = [16,16,64,128,256,512]

        num_res_blocks = 2
        resolution = 256
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            
            # Downsampling for each channel
            layers.append(DownSampleBlock(channels[i+1]))
            resolution //= 2

        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels=channels[-1],num_groups=num_groups))
        layers.append(Swish())
        layers.append(nn.Conv3d(channels[-1], latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# test
enc = Encoder()
print( 'The architecture is'+'\n{}'.format(
    summary(enc,(20,1,64,64,64)) 
    ))

