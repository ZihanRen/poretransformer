import torch.nn as nn
from torchinfo import summary
from lpu3dnet.modules.components import *    
from lpu3dnet.init_yaml import config_vqgan as config

class Encoder(nn.Module):
    def __init__(
            self,
            image_channels=config['architecture']['encoder']['img_channels'],
            latent_dim=config['architecture']['encoder']['latent_dim'],
            num_groups=config['architecture']['encoder']['num_groups'],
            num_res_blocks=config['architecture']['encoder']['num_res_blocks'],
            channels=config['architecture']['encoder']['channels']
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
        layers.append(nn.Conv3d(channels[-1], latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# test
if __name__ == "__main__":
    enc = Encoder()
    print( 'The architecture is'+'\n{}'.format(
        summary(enc,(20,1,64,64,64)) 
        ))

