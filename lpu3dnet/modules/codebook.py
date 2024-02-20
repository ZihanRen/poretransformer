#%%
from torch import nn
import torch
from torchinfo import summary
import torch.nn.functional as F

class Codebook(nn.Module):
    def __init__(
            self,
            size, # codebook size
            latent_dim, # feature dimension in z
            beta_c, # commitment loss
            autoencoder, # if True, it is autoencoder
            legacy, # if True, it is legacy
            init_ema = None    # pretrained codebook

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

        # reshape z to (#_Vectors, la5tent_dim) - this include all batch!
        # you have to permute first
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
        # reshape z_q back to original input vector z
        z_q = z_q.permute(0,4,1,2,3)

        return z_q, loss, (perplexity,min_encodings,min_encoding_indices)

    def get_codebook_entry(self, indices, vector_target): # shape is the target vector
        
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.size).to(indices)
        min_encodings.scatter_(1, indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if vector_target is not None:
            vector_target = vector_target.permute(0,2,3,4,1).contiguous()
            z_q = z_q.view(vector_target.shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0,4,1,2,3).contiguous()

        return z_q



#%%
if __name__ == "__main__":
    from lpu3dnet.modules.codebook_init import Codebook_init
    latent_dim = 256
    cod = Codebook(3000,
                   latent_dim,
                   0.2,
                   False,
                   True)
    

    ## do some testing
    # input latent vector
    a = torch.randn(20,latent_dim,3,3,3)
    z_qt, loss, info = cod(a)
    indices = info[2]
    z_test = cod.get_codebook_entry(indices,a) # z_test should be the same as z_qt

    # test indices
    # each image has 8 tokens
    # we have total 20 images
    # (1,160) -> (20,8)
    # each (1,8) tokens should be used to query codebook vector
    # the queried codebook vector should be the same as queried by z_test
    indices_batch = indices.view(20,-1)
    # get the specific codebook vector
    image_idx = 4
    idx = indices_batch[image_idx]

    query_idx = idx[:,None]
    target_output_shape = a[image_idx][None,::]  
    z_select = cod.get_codebook_entry(query_idx,target_output_shape)

    z_select_test = z_test[image_idx][None,::]


    






    


    # z_vec = cod.get_codebook_entry(torch.tensor([0,1,2,3,4,5,6,7,8,9]),(10,latent_dim,1,1,1))
    # z_vec = cod.get_codebook_entry(torch.tensor([0,1,2,3,4,5,6,7,8,9]),shape=None)
    # print(z_vec.shape)
    # cod = Codebook_topk(3000,
    #                  latent_dim=latent_dim,
    #                  beta_c=0.2,
    #                  legacy=False,
    #                  init_ema=None,
    #                  top_k=3)
    
    
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
    # print( 'The architecture is'+'\n{}'.format(
    #     summary(cod,(20,latent_dim,2,2,2)) 
    #     ))
    
    # a,b,c = cod(torch.randn(20,latent_dim,2,2,2))



# %%
