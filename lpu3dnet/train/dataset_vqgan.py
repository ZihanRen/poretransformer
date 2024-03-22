#%%
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tifffile import imread
import hydra

class Dataset_vqgan(Dataset):
  def __init__(
      self,
      cfg,
      ):
    # ct_idx: subvolumes that are sampled from main volume ct idx

    self.root_PATH = cfg.PATH.sub_vol
    self.ct_idx = cfg.ct_idx
    self.transform = None

    # get training data PATH
    self.img_PATH = []
    for idx in self.ct_idx:
      PATH_folder = os.path.join(self.root_PATH,f'ct_{idx}')
      for img_name in os.listdir(PATH_folder):
        self.img_PATH.append(os.path.join(PATH_folder,img_name))


  @staticmethod
  def tif_to_np(f_name):
      '''
      convert tif to numpy
      '''
      img = imread(f_name)
      img = img.astype('float32')/255
      return img>0.5

  def __len__(self):
    return len(self.img_PATH)

  def __getitem__(self,index):
    
    # load images
    image = Dataset_vqgan.tif_to_np(self.img_PATH[index])

    # convert images and features into tensor
    image = np.repeat(image[np.newaxis,:,:],1,axis=0)
    image_t = torch.from_numpy(image).float()

    if self.transform is not None:
      image_t = self.transform(image_t)
   
    return image_t

  def img_name(self):
    return (os.listdir(self.img_PATH))

  def print_parameters(self):
      print("training CT index is {}".format(self.ct_idx))
      print("training data root path is {}".format(self.root_PATH))

0
if __name__ == "__main__":
  experiment_idx = 7

  @hydra.main(
    config_path=f"/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/config/ex{experiment_idx}",
    config_name="dataset",
    version_base='1.2')
  def main(cfg):

    data_vqgan = Dataset_vqgan(cfg)
    data_vqgan.print_parameters()

    print("Total number of training data samples is {}".format(len(data_vqgan)))
    train_data_loader = DataLoader(data_vqgan,batch_size=16,shuffle=True)

    for i, img in enumerate(train_data_loader):
      if i == 3:
        break
      print(f'Batch number:{i}')
      print(f'input img shape is {img.size()}, dtype is {img.dtype}')
      print(2*'\n')
    
  main()


# %%
