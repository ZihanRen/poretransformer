#%%
from torch.nn import functional as F
import torch

# test whether conditional vector is expandable
cond_dim = 2
cond = torch.rand(1, 8, cond_dim) # conditional vector
b, num_img_patch, features_num = cond.shape
seq_per_patch = 27  # Number of tokens corresponding to each image patch
generated_tokens = 0
idx = torch.randint(0, 3000, (10, 27-10))
max_new_tokens = 20


while generated_tokens < max_new_tokens:
    # Determine the conditional vectors for each segment of the sequence based on its length
    seq_length = idx.size(1) + generated_tokens  # Current total sequence length including generated tokens
    extended_cond = torch.zeros((b, seq_length, cond_dim), device=idx.device)
    
    # Fill in the conditional information for each segment
    for i in range(num_img_patch):
        start_idx = i * seq_per_patch
        end_idx = start_idx + seq_per_patch
        # Use min to ensure we don't go beyond the current sequence length
        end_idx = min(end_idx, seq_length)
        if start_idx < seq_length:
            extended_cond[:, start_idx:end_idx, :] = cond[:, i, :].unsqueeze(1).repeat(1, end_idx - start_idx, 1)
        else:
            break  # Exit the loop if we've covered the entire sequence length
    
    generated_tokens += 1

#%%

current_cond = cond[:, current_patch_index, :].unsqueeze(1).repeat(
    1,
    idx.size(1),
    1
    )  # Shape: (b, t, features_num)




# %%

while generated_tokens < max_new_tokens:
    current_patch_index = (idx.size(1) - 1) // seq_per_patch  # Determine current patch index based on the length of idx
    current_patch_index = min(current_patch_index, num_img_patch - 1)  # Ensure it doesn't exceed the number of patches
    
    # Extract current conditional vector for the appropriate patch
    current_cond = cond[:, current_patch_index, :].unsqueeze(1).repeat(1, idx.size(1), 1)  # Shape: (b, t, features_num)

    # Concatenate idx and current_cond along the last dimension for input
    idx_cond = torch.cat([self.transformer.wte(idx), current_cond], dim=-1)  # Adjust dimensionality as needed

    # Forward the model to get logits
    logits = self(idx_cond)[0]  # Assuming the model returns logits as the first element
    logits = logits[:, -1, :] / temperature  # Scale logits by temperature

    # Optionally crop logits to top_k options
    if top_k is not None:
        logits = self.top_k_logits(logits, top_k)