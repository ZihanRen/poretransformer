import torch.nn as nn
from torchinfo import summary
from lpu3dnet.modules.components import *

class Encoder(nn.Module):
    def __init__(
            self,
            image_channels,
            latent_dim,
            num_groups,
            num_res_blocks,
            channels
            ):
        super(Encoder, self).__init__()
        # compress high dimensional 3D tensor to low dimensional 3D tensor (256,2,2,2)
        # architecture 64->32->16->8->4->2. Every downsample has two residual block
        # set up architecture in configuration yaml file

        # resolution = 256
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels,num_groups))
                in_channels = out_channels
            
            # Downsampling for each channel
            layers.append(DownSampleBlock(channels[i+1]))
            # resolution //= 2

        layers.append(ResidualBlock(channels[-1], channels[-1],num_groups))
        layers.append(ResidualBlock(channels[-1], channels[-1],num_groups))
        layers.append(GroupNorm(channels=channels[-1],num_groups=num_groups))
        layers.append(Swish())

        # experiment 1-6
        # layers.append(nn.Conv3d(channels[-1], latent_dim, 3, 1, 1))
        
        # experiment 7
        layers.append(nn.Conv3d(channels[-1], latent_dim, kernel_size=3, stride=2, dilation=1,padding=2))  # Adjust kernel size, stride, or padding as necessary
        self.model = nn.Sequential(*layers)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# test
if __name__ == "__main__":
    
    enc = Encoder(1,256,16,2,[16,32,128,256,512])
    print( 'The architecture is'+'\n{}'.format(
        summary(enc,(20,1,64,64,64)) 
        ))

