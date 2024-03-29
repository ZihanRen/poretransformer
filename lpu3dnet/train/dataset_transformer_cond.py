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
    self.ct_idx = cfg.ct_idx
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

  def __getitem__(self, index):
    token_path = self.token_files[index]
    cond_path = self.cond_files[index]
    
    token = torch.load(token_path).to(self.device)
    cond = torch.load(cond_path).to(self.device)
    
    return token[0], cond.float()

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
  experiment_idx = 7

  @hydra.main(
    config_path=f"/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/config/ex{experiment_idx}",
    config_name="dataset",
    version_base='1.2')
  def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_vqgan = Dataset_transformer(cfg,device)
    data_vqgan.print_parameters()
    data_vqgan.print_file_counts()
    # print(data_vqgan.token_files[:10])
    # print('\n')
    # print(data_vqgan.cond_files[:10])
    # print('\n')

    # print(data_vqgan.token_files[-10:])
    # print('\n')

    # print(data_vqgan.cond_files[-10:])
    # print('\n')



    train_data_loader = DataLoader(data_vqgan,batch_size=16,shuffle=True)

    for i, data_obj in enumerate(train_data_loader):
      tokens, cond = data_obj[0], data_obj[1]
      if i == 3:
        break
      print(f'Batch number:{i}')
      print(f'input tokens shape is {tokens.size()}, dtype is {tokens.dtype}')
      print(f'input cond shape is {cond.size()}, dtype is {cond.dtype}')
      print(2*'\n')
    
  main()


# %%
