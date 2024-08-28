#%%
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
from cpgan.ooppnm import img_process

def generate_unique_list(n, range_min=0, range_max=3000):

    if n > (range_max - range_min):
        raise ValueError(
            "The range is too small for the requested number of unique elements."
            )

    return random.sample(range(range_min, range_max), n)

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

def tensor_to_np(tensor):
    # input: tensor
    # output: numpy array
    return tensor.detach().numpy()


# initialize configuration parameters for a specific experiment
experiment = 'ex6'
initialize(config_path=f"../config/{experiment}")
cfg = compose(config_name="vqgan")

root_path = os.path.join(cfg.checkpoints.PATH, cfg.experiment)

# get the global path
PATH = cfg.data.PATH.sub_vol
root_path = os.path.join(cfg.checkpoints.PATH, cfg.experiment)
epoch = 10

model_vqgan = vqgan.VQGAN(cfg)
PATH_model = os.path.join(root_path,f'vqgan_epoch_{epoch}.pth')
model_vqgan.load_state_dict(
    torch.load(
            PATH_model,
            map_location=torch.device('cpu')
               )
    )

model_vqgan.eval()



# %%

codebook = model_vqgan.codebook

batch = 1
latent_dim = 256
feature_dim = 4
num_vectors = batch * (feature_dim**3)


vec = codebook.get_codebook_entry(
    torch.tensor(generate_unique_list(num_vectors)),
        shape=(
        batch,
        latent_dim,
        feature_dim,
        feature_dim,
        feature_dim)
        )

random_img = model_vqgan.decode(vec)
img_prc = img_process.Image_process()
random_img = img_prc.clean_img(random_img)

# visualize those random images
f = plt.figure()
plt.imshow(random_img[0,0,...],cmap='gray')
plt.axis('off')
# %% Similar porosity vs different porosity
ct_idx = 1

img_idx = [0,100,200,300,400,500]

img_tensor,img_matrix = idx_to_matrix(ct_idx,img_idx)

with torch.no_grad():
    decode_img,info,_ = model_vqgan(img_tensor)
    decode_img = img_prc.clean_img(decode_img)

# %%
phi_cal = []
for i in range(6):
    phi_cal.append(img_prc.phi(decode_img[i,0,...]))
# %% generate extracted features
batch = 6
vec = codebook.get_codebook_entry(
        info[2].view(-1),
        shape=(
        batch,
        latent_dim,
        feature_dim,
        feature_dim,
        feature_dim)
        )

similar_idx = [1,5]
diff_idx = 0


# %%
diff_high_1 = ((vec[similar_idx[0]] - vec[diff_idx]) ** 2).sum()
diff_high_2 = ((vec[similar_idx[1]] - vec[diff_idx]) ** 2).sum()
diff_low = ((vec[similar_idx[0]] - vec[similar_idx[1]]) ** 2).sum()

# %%
