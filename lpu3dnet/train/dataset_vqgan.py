import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tifffile import imread
from lpu3dnet import init_yaml

class Dataset_vqgan(Dataset):
  def __init__(self,ct_idx=[2,3,4,5],transform=None):
    # ct_idx: subvolumes that are sampled from main volume ct idx

    self.root_PATH = init_yaml.PATH['img_path']['sub_vol']
    self.ct_idx = ct_idx
    self.transform = transform

    # get training data PATH
    self.img_PATH = []
    for idx in ct_idx:
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

  def ct_idx(self):
      print(self.ct_idx)


if __name__ == "__main__":

  data_vqgan = Dataset_vqgan()
  print("Total number of training data samples is {}".format(len(data_vqgan)))
  train_data_loader = DataLoader(data_vqgan,batch_size=20,shuffle=True)


  for i, img in enumerate(train_data_loader):
    if i == 3:
      break
    print(f'Batch number:{i}')
    print(f'input img shape is {img.size()}, dtype is {img.dtype}')
    print(2*'\n')

