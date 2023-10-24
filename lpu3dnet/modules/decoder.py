import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from lpu3dnet.modules.components import *
from lpu3dnet.init_yaml import config_vqgan as config


class Decoder(nn.Module):
    def __init__(self,
                 image_channels=config['architecture']['decoder']['img_channels'],
                 latent_dim=config['architecture']['decoder']['latent_dim'],
                 num_groups = config['architecture']['decoder']['num_groups'],
                 num_res_blocks=config['architecture']['decoder']['num_res_blocks'],
                channels=config['architecture']['decoder']['channels']
                 ):
        
        super(Decoder, self).__init__()
        # resolution = 2

        in_channels = channels[0]
        layers = [nn.Conv3d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels,num_groups=num_groups),
                  ResidualBlock(in_channels, in_channels,num_groups=num_groups)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels,num_groups))
                in_channels = out_channels
            if (i != 0) & (i != len(channels) - 1):
                layers.append(UpSampleBlock(in_channels))
                # resolution *= 2

        layers.append(GroupNorm(in_channels,num_groups=num_groups))
        layers.append(Swish())
        layers.append(nn.Conv3d(in_channels, image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



# test
if __name__ == "__main__":
    enc = Decoder()
    print( 'The architecture is'+'\n{}'.format(
        summary(enc,(20,256,2,2,2)) 
        ))