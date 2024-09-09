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
from cpgan.ooppnm import pnm_sim_old
import porespy as ps


def simulation_phys(img):
    data_pnm = pnm_sim_old.Pnm_sim(im=img)
    data_pnm.network_extract()
    if data_pnm.error == 1:
        print('Error in network extraction')
        return None
        # raise ValueError('Error in network extraction')
        
    data_pnm.init_physics()
    data_pnm.get_absolute_perm()
    data_pnm.invasion_percolation()
    data_pnm.kr_simulation()
    data_pnm.close_ws()
    return data_pnm.data_tmp


def porosity_simulation(img):
    phi = ps.metrics.porosity(img)
    return phi


# read pickle file fiven ct and vol_dim
def load_data_gen(epoch_transformer, vol_dim):
    file_path = f'ex12/epoch_{epoch_transformer}/img_gen_vol_{vol_dim}.pkl'
    with open(file_path, 'rb') as file:
        img_fake = pickle.load(file)

    file_path = f'ex12/epoch_{epoch_transformer}/img_real_vol_{vol_dim}.pkl'
    with open(file_path, 'rb') as file:
        img_real = pickle.load(file)
    
    return img_real, img_fake


volume_dim = 3

for epoch_transformer in [50]:

    img_real, img_fake = load_data_gen(epoch_transformer, volume_dim)


    phys_results = {}
    phys_results['real'] = []
    phys_results['fake'] = []

    for ct_idx in range(6):
        for sample_idx in range(len(img_real[ct_idx])):
            img_real_sample = img_real[ct_idx][sample_idx]
            img_real_sample = img_real_sample[:192, :192, :192]
            img_fake_sample = img_fake[ct_idx][sample_idx]
            img_fake_sample = img_fake_sample[:192, :192, :192]
            # phys_results['real'].append(simulation_phys(img_real_sample))
            # phys_results['fake'].append(simulation_phys(img_fake_sample))
            phys_results['real'].append(porosity_simulation(img_real_sample))
            phys_results['fake'].append(porosity_simulation(img_fake_sample))


    with open(f'ex12/epoch_{epoch_transformer}/phys_result_{volume_dim}_192_porosity.pkl', 'wb') as file:
        pickle.dump(phys_results, file)