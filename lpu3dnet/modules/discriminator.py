import torch
from torch import nn
from torchinfo import summary
import hydra

class Discriminator(nn.Module):
    def __init__(
                 self,
                 image_channels,
                 num_filters_last,
                 n_layers
                ):
        super(Discriminator, self).__init__()

        self.image_channels = image_channels
        self.num_filters_last = num_filters_last
        self.n_layers = n_layers
        layers = [nn.Conv3d(self.image_channels, self.num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, self.n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv3d(self.num_filters_last * num_filters_mult_last, self.num_filters_last * num_filters_mult, 4,
                          2 if i < self.n_layers else 1, 1, bias=False),
                nn.BatchNorm3d(self.num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv3d(self.num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



if __name__ == '__main__':
    dis = Discriminator(1,64,3)
    print('Summary of the discriminator is'+'\n{}'.format(summary(dis,(2,1,64,64,64))))
