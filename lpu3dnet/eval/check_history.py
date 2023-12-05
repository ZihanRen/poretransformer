#%% load training loss
from matplotlib import pyplot as plt
from tifffile import imread
import numpy as np
import pickle
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
import os
# load pytorch model
import torch
from lpu3dnet.frame import vqgan
import random
import torch.nn.functional as F

def read_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def tif_to_np(f_name):
    '''
    convert tif to numpy
    '''
    img = imread(f_name)
    img = img.astype('float32')/255
    return img>0.5


# initialize configuration parameters for a specific experiment
experiment = 'ex6'
initialize(config_path=f"../config/{experiment}")
cfg = compose(config_name="vqgan")
root_path = os.path.join(cfg.checkpoints.PATH, cfg.experiment)

#%%
# get the global path
PATH = cfg.data.PATH.sub_vol

def get_img_list(idx_list,ct_idx):
    # input: ct image idx
    # output: list of images
    img_list = []
    for idx in idx_list:
        img = tif_to_np(f'{PATH}/ct_{ct_idx}/{idx}.tif')
        img_list.append(img)
    return img_list

def img_list_to_np(img_list):
    # input: list of images
    # output: numpy array of images
    image = np.stack(img_list,axis=0)
    return image

def idx_to_matrix(ct_idx,img_idx_list):
    # input: ct image idx, list of image idx
    # output: numpy array of images
    img_list = get_img_list(img_idx_list,ct_idx)
    img_matrix = img_list_to_np(img_list)
    img_matrix = img_matrix[:,np.newaxis,...]
    img_tensor = torch.from_numpy(img_matrix).float()

    return img_tensor,img_matrix

model_vqgan = vqgan.VQGAN(cfg)
#%% generate random img idx
random.seed(1340)
# Generate a list of unique random integers
num_elements = 50
min_value = 0
max_value = 7900
ct_idx = 0
test_idx = 1
start_epoch = 0
end_epoch = 45

from cpgan.ooppnm import img_process
img_prc = img_process.Image_process()

random_img_idx = random.sample(
    range(min_value, max_value + 1), num_elements
    )


val_loss = []
epoch_list = []
test_loss = []

for epoch in range(start_epoch,end_epoch+5,5):
    PATH_model = os.path.join(root_path,f'vqgan_epoch_{epoch}.pth')
    model_vqgan.load_state_dict(
        torch.load(
                PATH_model,
                map_location=torch.device('cpu')
                )
        )

    model_vqgan.eval()



    img_tensor,img_matrix = idx_to_matrix(ct_idx,random_img_idx)

    with torch.no_grad():
        decode_img,codebook_indice,_ = model_vqgan(img_tensor)
        decode_img = img_prc.clean_img(decode_img)
        decode_img = decode_img[:,np.newaxis,...]
    
    rec_loss = np.mean((decode_img - img_matrix) ** 2)
    val_loss.append(rec_loss)

    img_tensor,img_matrix = idx_to_matrix(test_idx,random_img_idx)

    with torch.no_grad():
        decode_img,codebook_indice,_ = model_vqgan(img_tensor)
        decode_img = img_prc.clean_img(decode_img)
        decode_img = decode_img[:,np.newaxis,...]
    
    rec_loss = np.mean((decode_img - img_matrix) ** 2)
    test_loss.append(rec_loss)
    epoch_list.append(epoch)

# save epoch and val loss into pickle
save_obj = {'epoch':epoch_list,'val_loss':val_loss,'test_loss':test_loss}
with open(f'{root_path}/val_loss_ct{ct_idx}.pkl', 'wb') as f:
    pickle.dump(save_obj, f)