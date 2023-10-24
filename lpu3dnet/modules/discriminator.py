import torch
from torch import nn
from torchinfo import summary
from lpu3dnet.init_yaml import config_vqgan as config

class Discriminator(nn.Module):
    def __init__(
                 self, 
                 image_channels = config['architecture']['discriminator']['img_channels'],
                num_filters_last=config['architecture']['discriminator']['init_filters_num'],
                n_layers=config['architecture']['discriminator']['num_layers']
                ):
        
        super(Discriminator, self).__init__()

        layers = [nn.Conv3d(image_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv3d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm3d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv3d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



if __name__ == '__main__':
    dis = Discriminator()
    print('Summary of the discriminator is'+'\n{}'.format(summary(dis,(2,1,64,64,64))))
