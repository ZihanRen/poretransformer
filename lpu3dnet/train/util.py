#%%
from lpu3dnet.train import dataset_transformer_cond
import hydra
import torch
from torch.utils.data import Dataset, DataLoader
from lpu3dnet.frame import vqgan
import os

t = 8*27
b = 1
cond1 = torch.rand(b, 8 , 3)
seq_per_patch = t // cond1.size(1) 

cond1_rep = cond1.repeat(1, 1, seq_per_patch).view(b, -1, cond1.size(2))


# %%
cond1 = torch.tensor([[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]]])  # shape: (8, 1, 1)
seq_per_patch = 3
cond1_rep = cond1.repeat(1, 1, seq_per_patch).view(8, -1, 1)
# %%
