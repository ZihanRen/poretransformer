#%%
import torch
# parameters
p_keep = 0.9
vocab_size = 100
sos_token = 101

img_tokens = torch.randint(0, vocab_size, (2, 10))
cond = torch.rand(2,8,4)
sos_tokens = torch.ones(img_tokens.shape[0], 1) * sos_token
sos_tokens = sos_tokens.long()



mask = torch.bernoulli(
    p_keep * torch.ones(
    img_tokens.shape)
    )

mask = mask.round().to(dtype=torch.int64)

random_indices = torch.randint_like(
    img_tokens,
    vocab_size
)

perturbed_indices = mask * img_tokens + (1 - mask) * random_indices
            
# %%
