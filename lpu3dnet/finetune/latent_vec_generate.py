# generate pretrained codebook for finetuning
#%%
from matplotlib import pyplot as plt
from tifffile import imread
import numpy as np
import pickle
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
import os
import torch
from lpu3dnet.frame import vqgan

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

def get_img_fnames(ct_idx):
    # input: ct image idx
    # output: list of image idx
    img_fnames = os.listdir(f'{PATH}/ct_{ct_idx}')
    return img_fnames


def get_img_list(ct_idx):
    # input: ct image idx
    # output: list of images
    img_list = []
    for idx in ct_idx:
        f_names = get_img_fnames(idx)
        for f_name in f_names:
            path_img = os.path.join(f'{PATH}/ct_{idx}',f_name)
            img = tif_to_np(path_img)
            img_list.append(img)
    return img_list

def img_list_to_np(img_list):
    # input: list of images
    # output: numpy array of images
    image = np.stack(img_list,axis=0)
    return image

def idx_to_matrix(ct_idx):
    # input: ct image idx, list of image idx
    # output: numpy array of images
    img_list = get_img_list(ct_idx)
    img_matrix = img_list_to_np(img_list)
    img_matrix = img_matrix[:,np.newaxis,...]
    img_tensor = torch.from_numpy(img_matrix).float()

    return img_tensor,img_matrix,img_list

def tensor_to_np(tensor):
    # input: tensor
    # output: numpy array
    return tensor.detach().numpy()

# initialize configuration parameters for a specific experiment
experiment = 'ex1'
initialize(config_path=f"../config/{experiment}")
cfg = compose(config_name="vqgan")
root_path = os.path.join(cfg.checkpoints.PATH, cfg.experiment)
epoch = 120
# get the global path
PATH = cfg.data.PATH.sub_vol

# load the model
model_vqgan = vqgan.VQGAN(cfg)
PATH_model = os.path.join(root_path,f'vqgan_epoch_{epoch}.pth')
model_vqgan.load_state_dict(
    torch.load(
            PATH_model,
            map_location=torch.device('cpu')
               )
    )

model_vqgan.eval()

# encode training images
ct_idx = cfg.data.ct_idx
img_tensor,_,_ = idx_to_matrix(ct_idx)
z_list = []
infer_batch_size = 16
num_loop = img_tensor.shape[0]//infer_batch_size


#%%
with torch.no_grad():
    for i in range(num_loop):
        sample_tensor = img_tensor[i*infer_batch_size:(i+1)*infer_batch_size,...]
        z_sample = model_vqgan.encode(sample_tensor)
        z_sample = tensor_to_np(z_sample)
        z_list.append(z_sample)

z_list = np.concatenate(z_list,axis=0)

# save to npy file
np.save('latent_matrix.npy',z_list)