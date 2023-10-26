from torch import nn
import torch
from torchinfo import summary
from lpu3dnet.init_yaml import config_vqgan as config


class Codebook(nn.Module):
    def __init__(
            self,
            num_codebook_vectors=config['architecture']['codebook']['size'],
            latent_dim=config['architecture']['codebook']['latent_dim'],
            beta_c = config['architecture']['codebook']['beta_c'],
            autoencoder=config['architecture']['codebook']['autoencoder']
            ):
        
        super(Codebook, self).__init__()
        self.num_codebook_vectors = int(num_codebook_vectors)
        self.latent_dim = latent_dim
        self.beta_c = beta_c
        self.autoencoder = autoencoder
        
        # storage vector embedding layer - this layer is learnable
        if not self.autoencoder:
            self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        if self.autoencoder:
            return z, None, None

        z = z.permute(0,2,3,4,1).contiguous()
        # print(z.shape) 
        z_flattened = z.view(-1, self.latent_dim)
        # print("Flattened z shape is {}".format(z_flattened.shape))

        # distance between z and codebook vectors
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))
        
        # arrays of indices of code vector that is minimum distance from z
        min_encoding_indices = torch.argmin(d, dim=1)

        # print(min_encoding_indices)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        # print("z_q shape is {}".format(z_q.shape))

        # make codebook close to z. Check papers for more details - embedding loss
        # commitment loss
        loss =    torch.mean((z_q.detach() - z)**2) + \
                self.beta_c * torch.mean((z_q - z.detach())**2) 

        # gradident trick since k nearest neighbor is not differentiable - consider RL??
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0,4,1,2,3)

        return z_q, min_encoding_indices, loss


if __name__ == "__main__":
    cod = Codebook()
    print( 'The architecture is'+'\n{}'.format(
        summary(cod,(20,config['architecture']['codebook']['latent_dim'],2,2,2)) 
        ))
