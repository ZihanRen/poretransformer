#%%
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import hydra
import glob

def extract_index(filename):
    # Extracts the numeric index from a filename
    return int(filename.split('_')[-1].split('.')[0])


class Dataset_transformer(Dataset):
  
  def __init__(
      self,
      cfg,
      device
      ):
    # ct_idx: subvolumes that are sampled from main volume ct idx

    self.device = device
    if cfg.target == 'validation':
        self.ct_idx = cfg.ct_idx_val
    elif cfg.target == 'train':
        self.ct_idx = cfg.ct_idx
    else:
        raise ValueError('Invalid target value in config file. Must be either "train" or "validation"')
    
    self.img_tokens_path = cfg.PATH.image_tokens
    self.img_cond_path = cfg.PATH.image_tokens_cond

    # get training data PATH
    self.img_token_PATH = []
    self.img_cond_PATH = []

    # Collecting paths for tokens and conditions
    self.token_files = []
    self.cond_files = []

    
    for idx in self.ct_idx:
        token_files_ct = glob.glob(os.path.join(self.img_tokens_path, f'ct_{idx}', 'tokens_*.pt'))
        cond_files_ct = glob.glob(os.path.join(self.img_cond_path, f'ct_{idx}', 'cond_*.pt'))
        self.token_files.extend(sorted(token_files_ct, key=extract_index))
        self.cond_files.extend(sorted(cond_files_ct, key=extract_index))
    assert len(self.token_files) == len(self.cond_files) 

  def __len__(self):
    return len(self.token_files)

  def reshape(self, token,cond):
     features_num = token.size(-1)
     patch_num = cond.size(0)
     token_flatten = token.view(1,-1)
     cond_patch_list = [cond[i].unsqueeze(0).expand(features_num, -1) for i in range(patch_num)]
     cond_flatten = torch.cat(cond_patch_list, dim=0)
     return token_flatten, cond_flatten
     

  def __getitem__(self, index):
    token_path = self.token_files[index]
    cond_path = self.cond_files[index]
    
    token = torch.load(token_path).to(self.device)
    cond = torch.load(cond_path).to(self.device)
    token_flat, cond_flat = self.reshape(token,cond)
    
    # return token_flat[0], cond_flat.float()
    return token_flat[0], cond_flat.float()

  def print_file_counts(self):

    for ct_index in self.ct_idx:
        token_count = len(os.listdir(os.path.join(self.img_tokens_path, f'ct_{ct_index}')))
        cond_count = len(os.listdir(os.path.join(self.img_cond_path, f'ct_{ct_index}')))
        print(f'CT Index: {ct_index}, Tokens: {token_count}, Conditions: {cond_count}')    

  def print_parameters(self):
      print("training CT index is {}".format(self.ct_idx))
      print("training image tokens root path is {}".format(self.img_tokens_path))
      print("training image conditions root path is {}".format(self.img_cond_path))
      print("Total number of training data samples is {}".format(len(self)))


if __name__ == "__main__":
  experiment_idx = 11

  @hydra.main(
    config_path=f"/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/config/ex{experiment_idx}",
    config_name="dataset",
    version_base='1.2')
  def main(cfg):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_vqgan = Dataset_transformer(cfg,device)
    data_vqgan.print_parameters()
    data_vqgan.print_file_counts()


    train_data_loader = DataLoader(data_vqgan,batch_size=16,shuffle=True)

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


    for i, data_obj in enumerate(train_data_loader):
      tokens, cond = data_obj[0], data_obj[1]

      # attn_list = attention_window(4,tokens)
      # for sub_token in attn_list:
      #    print(f'sub token shape is {sub_token.size()}')



      if i == 1:
        break
      print(f'Batch number:{i}')
      print(f'input tokens shape is {tokens.size()}, dtype is {tokens.dtype}')
      print(f'input cond shape is {cond.size()}, dtype is {cond.dtype}')
      print(2*'\n')
    
  main()


# %%
