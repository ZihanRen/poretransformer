#%%
from torch.nn import functional as F
import torch
logits = torch.rand(10, 216, 3000)
target = torch.randint(0, 3000, (10, 216))
F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1),ignore_index=-1)
# %%
