from torch import nn
import torch
from torchinfo import summary
import torch.nn.functional as F

class Codebook(nn.Module):
    def __init__(
            self,
            size,
            latent_dim,
            beta_c,
            autoencoder,
            legacy,
            init_ema = None
            ):
        
        super(Codebook, self).__init__()
        self.autoencoder = autoencoder
        # initalize embedding layer based on pretrained weights or distribution
        if not self.autoencoder:
    
            if init_ema is not None:
                self.latent_dim = init_ema.shape[1]
                self.size = init_ema.shape[0]
                self.embedding = nn.Embedding(self.size, self.latent_dim)
                self.embedding.weight.data = init_ema
            else:
                self.size = int(size)
                self.latent_dim = latent_dim
                self.embedding = nn.Embedding(self.size, self.latent_dim)
                self.embedding.weight.data.uniform_(
                    -1.0 / self.size,
                    1.0 / self.size)
        
        self.beta_c = beta_c
        self.legacy = legacy
        

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




class Codebook_topk(nn.Module):
    def __init__(
            self,
            size,
            latent_dim,
            beta_c,
            legacy = False,
            init_ema = None,
            top_k = 3
            ):
        
        super(Codebook_topk, self).__init__()
        # initalize embedding layer based on pretrained weights or distribution
    
        if init_ema is not None:
            self.latent_dim = init_ema.shape[1]
            self.size = init_ema.shape[0]
            self.embedding = nn.Embedding(self.size, self.latent_dim)
            self.embedding.weight.data = init_ema
        else:
            self.size = int(size)
            self.latent_dim = latent_dim
            self.embedding = nn.Embedding(self.size, self.latent_dim)
            self.embedding.weight.data.uniform_(
                -1.0 / self.size,
                1.0 / self.size)
        
        self.beta_c = beta_c
        self.legacy = legacy
        self.top_k = top_k
        

    def forward(self, z):

        z_flattened = z.view(-1, self.latent_dim)

        # Compute distances between z and codebook vectors
        # d.shape = (160,3000) - distance information matrix
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
            
        # Find top-k nearest vectors in each row
        topk_distances, topk_indices = torch.topk(
            d,
            self.top_k,
            dim=1,
            largest=False
            )
        # Average the top-k vectors
        topk_encodings = torch.zeros(topk_indices.shape[0], self.top_k, self.size).to(z)
        topk_encodings.scatter_(2, topk_indices.unsqueeze(2), 1)
        z_q = torch.matmul(topk_encodings, self.embedding.weight)
        z_q = torch.mean(z_q, dim=1)  # Averaging across the k dimension
        z_q = z_q.view(z.shape)

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
        e_mean = torch.mean(topk_encodings, dim=0)
        e_mean = torch.mean(e_mean, dim=0)  # Average across k
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # gradident trick since k nearest neighbor is not differentiable - consider RL??
        z_q = z + (z_q - z).detach()
        # z_q = z_q.permute(0,4,1,2,3)

        return z_q, loss, (perplexity,topk_encodings,topk_indices)

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


class Codebook_EMA(nn.Module):
    # for detailed explnation, plz check
    # https://github.com/ZihanRen/what_i_learned/blob/main/gan/VQGAN.ipynb

    def __init__(self, 
                 size,
                 latent_dim,
                 beta_c,
                 decay,
                 epsilon=1e-5,
                 init_ema = None): # a super small number for divisible
        super(Codebook_EMA, self).__init__()
        # initalize embedding layer based on pretrained weights or distribution
        if init_ema is not None:
            self.latent_dim = init_ema.shape[1]
            self.size = init_ema.shape[0]
            self.embedding = nn.Embedding(self.size, self.latent_dim)
            self.embedding.weight.data = init_ema
            
        else:
            self.latent_dim = latent_dim
            self.size = size
            self.embedding = nn.Embedding(self.size, self.latent_dim)
            self.embedding.weight.data.normal_()
        
        self.beta_c = beta_c
        self.register_buffer('_ema_cluster_size', torch.zeros(size))
        self._ema_w = nn.Parameter(torch.Tensor(size, self.latent_dim))
        self._ema_w.data = self.embedding.weight.data.clone()
        
        self.decay = decay
        self.epsilon = epsilon

    def forward(self, z):
        z = z.permute(0, 2, 3, 4, 1).contiguous()        
        z_flattened = z.view(-1, self.latent_dim)        
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))
            
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            min_encoding_indices.shape[0],
            self.size).to(z)
        encodings.scatter_(1, min_encoding_indices, 1)        
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        
        # Use EMA to update the embedding vectors
        if self.training:

            
            self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.size * self.size) * n)
            
            dw = torch.matmul(encodings.t(), z_flattened)
            self._ema_w = nn.Parameter(
                self._ema_w * self.decay + (1 - self.decay) * dw
                )
            self.embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
                )
        
        # commitment loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        loss = self.beta_c * commitment_loss
        
        # Straight Through Estimator
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0,4,1,2,3)
        
        # perplexity calculation
        e_mean = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return z_q,loss,(perplexity,encodings,min_encoding_indices)


if __name__ == "__main__":
    from lpu3dnet.modules.codebook_init import Codebook_init
    latent_dim = 256
    # cod = Codebook(3000,
    #                latent_dim,
    #                0.2,
    #                False,
    #                True)
    cod = Codebook_topk(3000,
                     latent_dim=latent_dim,
                     beta_c=0.2,
                     legacy=False,
                     init_ema=None,
                     top_k=3)
    
    
    # codebook_init = Codebook_init('/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/finetune/kmeans_30000.pkl')
    # pretrained_tensor = codebook_init.codebook_emd
    # # cod = Codebook_EMA(3000,
    # #                    latent_dim,
    # #                    0.2,
    # #                    0.99,
    # #                    init_ema=pretrained_tensor)
    # cod = Codebook(size=3000,
    #                 latent_dim=256,
    #                 beta_c=0.2,
    #                 autoencoder=False,
    #                 legacy=False,
    #                 init_ema=pretrained_tensor)
    print( 'The architecture is'+'\n{}'.format(
        summary(cod,(20,latent_dim,2,2,2)) 
        ))
    
    # a,b,c = cod(torch.randn(20,latent_dim,2,2,2))


