#%%
from lpu3dnet.train import dataset_transformer_cond
import torch
import torch
from lpu3dnet.frame import vqgan
from lpu3dnet.frame import transformer
from lpu3dnet.train import dataset_transformer_cond
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
import time
from datetime import timedelta
import hydra



def attention_window(sub_window_size,token_vector):
    
    
    sub_token_list = []
    expansion_features = 64
    left_idx = 0
    # window_size = 8 - initialize for sos token
    right_idx = sub_window_size-1
    window_size = 8

    while right_idx <= window_size:
        left_idx_expand = left_idx*expansion_features
        right_idx_expand = right_idx*expansion_features
        # extract sub_token but leave space for sos token
        sub_token = token_vector[:,left_idx_expand:right_idx_expand]
        sub_token_list.append(sub_token)

        if right_idx == sub_window_size-1:   
            # update right idx but not left idx
            right_idx += 1

        else:
            left_idx += 1
            right_idx += 1
    
    return sub_token_list



with hydra.initialize(config_path="../config/ex11"):
    cfg_vqgan = hydra.compose(config_name="vqgan")
    cfg_transformer = hydra.compose(config_name="transformer")
    cfg_dataset = hydra.compose(config_name="dataset")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load data
train_dataset = dataset_transformer_cond.Dataset_transformer(
                cfg_dataset,
                device=device)


train_data_loader = DataLoader(
                    train_dataset,
                    batch_size=cfg_transformer.train.batch_size,
                    shuffle=True,
                    drop_last=False
                    )


#%%
for i, data_obj in enumerate(train_data_loader):
    tokens, cond = data_obj[0], data_obj[1]
    sub_window = 4
    token_window_list = attention_window(sub_window,tokens)
    cond_window_list = attention_window(sub_window,cond)
    
    if i == 1:
        break
    # for sub_token in attn_list:
    #     print(f'sub token shape is {sub_token.size()}')



#%% validate:
# 1st sequence of tokens should be equal to token[:,64*3]
assert torch.allclose(token_window_list[0], tokens[:,:64*3])
assert torch.allclose(token_window_list[1], tokens[:,:64*4])
assert torch.allclose(token_window_list[2], tokens[:,64*1:64*5])
assert torch.allclose(token_window_list[3], tokens[:,64*2:64*6])


assert torch.allclose(cond_window_list[0], cond[:,:64*3])
assert torch.allclose(cond_window_list[1], cond[:,:64*4])
assert torch.allclose(cond_window_list[2], cond[:,64*1:64*5])
assert torch.allclose(cond_window_list[3], cond[:,64*2:64*6])




# %%
