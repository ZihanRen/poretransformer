#%%
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread
from skimage.filters import threshold_multiotsu
import os
import porespy as ps
from tifffile import imwrite
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
import torch
from matplotlib import pyplot as plt
from tifffile import imread
import numpy as np
import pickle
from hydra.experimental import compose, initialize
import os
import torch
from lpu3dnet.frame import vqgan
from lpu3dnet.frame import transformer

def clean_img(img_t):
    '''
    2  3^3 median image filter
    3 Otsu binary segmentation
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/median.htm
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html
    '''

    thresholds = threshold_multiotsu(img_t,classes=2)
    # from the threshold we separate regions
    img_t = np.digitize(img_t, bins=thresholds)

    return img_t


def img_crop(img,crop_size):
    x_max = img.shape[0]
    y_max = img.shape[1]
    z_max = img.shape[2]

    sec = img[
        crop_size:x_max-crop_size,
        crop_size:y_max-crop_size,
        crop_size:z_max-crop_size
        ]
    return sec


def np_to_tif(img,f_name):
    '''
    convert numpy to tif
    '''

    img_save = (img * 255).astype('uint8')
    # Save the 3D array as a 3D tif
    imwrite(f_name, img_save)

def tif_to_np(f_name):
    '''
    convert tif to numpy
    '''
    img = imread(f_name)
    img = img.astype('float32')/255
    return img>0.5

initialize(config_path=f"../config/ex7")
cfg_vqgan = compose(config_name="vqgan")
cfg_transformer = compose(config_name="transformer")
cfg_dataset = compose(config_name="dataset")
# %%
epoch_vqgan = 25
epoch_transformer = 170
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = os.path.join(cfg_dataset.checkpoints.PATH, cfg_dataset.experiment)

vqgan_path = os.path.join(root_path,f'vqgan_epoch_{epoch_vqgan}.pth')
transformer_path = os.path.join(root_path,'transformer',f'transformer_epoch_{epoch_transformer}.pth')

model_vqgan = vqgan.VQGAN(cfg_vqgan)
model_transformer = transformer.Transformer(cfg_transformer)

model_vqgan.load_checkpoint(vqgan_path)
model_transformer.load_checkpoint(transformer_path)

model_vqgan = model_vqgan.to(device)
model_transformer = model_transformer.to(device)

model_transformer.eval()
model_vqgan.eval()

img_list = []
for i in range(6):
    img_list.append(
        tif_to_np(
            os.path.join(
                cfg_dataset.PATH.main_vol,
                f'main_{i}.tif'
                )
                  )
        ) 

#%%
def crop_img(img,vol_dim,sub_size=64):
    return img[
                :sub_size*vol_dim,
               :sub_size*vol_dim,
               :sub_size*vol_dim
               ]
img_0 = crop_img(img_list[0],3)
img_1 = crop_img(img_list[1],3) # ground truth
img_2 = crop_img(img_list[2],3)

img_gen = np.load('data_ref/1_pred.npy')

#%% kr/k simulation
from cpgan.ooppnm import pnm_sim_old

def simulation_phys(img):
    data_pnm = pnm_sim_old.Pnm_sim(im=img)
    data_pnm.network_extract()
    if data_pnm.error == 1:
        raise ValueError('Error in network extraction')
    data_pnm.init_physics()
    data_pnm.get_absolute_perm()
    data_pnm.invasion_percolation()
    data_pnm.kr_simulation()
    data_pnm.close_ws()
    return data_pnm.data_tmp
generate_df = simulation_phys(img_gen)

compare_df_list = [simulation_phys(img_0), 
                   simulation_phys(img_1), # ground truth
                   simulation_phys(img_2)
                   ]

with open('data_ref/generate_df.pickle', 'wb') as file:
    pickle.dump(generate_df, file)

with open('data_ref/compare_df_list.pickle', 'wb') as file:
    pickle.dump(compare_df_list, file)


#%%