
from torch import nn
import torch
from torchinfo import summary

class Codebook(nn.Module):
    def __init__(
            self,
            size,
            latent_dim,
            beta_c,
            autoencoder,
            legacy
            ):
        
        super(Codebook, self).__init__()
        self.size = int(size)
        self.latent_dim = latent_dim
        self.beta_c = beta_c
        self.autoencoder = autoencoder
        self.legacy = legacy
        
        # storage vector embedding layer - this layer is learnable
        if not self.autoencoder:
            self.embedding = nn.Embedding(self.size, self.latent_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.size, 1.0 / self.size)

    def forward(self, z):
        if self.autoencoder:
            return z, None, None

        # reshape z to (#_Vectors, latent_dim) - this include all batch!
        z = z.permute(0,2,3,4,1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        # distance between z and codebook vectors
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))
        
        # arrays of indices of code vector that is minimum distance from z
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0],
            self.size).to(z)
        
        # construct one hot vector (# Vectors,num_codebook_vectors)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        # retrive z_q from matrix multiplication between one hot vector and embedding
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # make codebook close to z. Check papers for more details - embedding loss
        # commitment loss
        if self.legacy:
            loss = torch.mean((z_q.detach() - z)**2) + \
                self.beta_c * torch.mean((z_q - z.detach())**2)
        else:
            # avoid updating encoder too qucikly when beta_c is small
            # for preserving reconstruction loss
            loss =   self.beta_c * torch.mean((z_q.detach() - z)**2) + \
                    torch.mean((z_q - z.detach())**2) 

        # perplexity calculation
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # gradident trick since k nearest neighbor is not differentiable - consider RL??
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0,4,1,2,3)

        return z_q, loss, (perplexity,min_encodings,min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.size).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q




if __name__ == "__main__":
    latent_dim = 256
    cod = Codebook(3000,
                   latent_dim,
                   0.2,
                   False,
                   True)
    print( 'The architecture is'+'\n{}'.format(
        summary(cod,(20,latent_dim,2,2,2)) 
        ))
    
    # a,b,c = cod(torch.randn(20,latent_dim,2,2,2))


