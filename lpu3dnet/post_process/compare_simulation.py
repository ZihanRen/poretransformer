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


initialize(config_path=f"../config/ex12")
cfg_dataset = compose(config_name="dataset")


# read pickle file fiven ct and vol_dim
def load_data(ct_idx, vol_dim, root_dir):
    file_path = f'{root_dir}/sample_{ct_idx}/img_gen_vol_{vol_dim}.pkl'
    with open(file_path, 'rb') as file:
        img_results = pickle.load(file)
    return img_results


# gloabl variables
volume_dim = 3
root_dir = 'db'



#%%
for ct_idx in [0,1,2,3,4,5]:
    img_data = load_data(ct_idx, volume_dim, root_dir)

    sim_results_per_ct = {}
    sim_results_per_ct['compare'] = []
    # simulating comparing samples
    for compare_samples in img_data['compare']:
        # compare_phys = simulation_phys(compare_samples)
        compare_phys = simulation_phys(compare_samples)
        sim_results_per_ct['compare'].append(compare_phys)


    # simulate each sample of real and generate
    for sample_idx in img_data.keys():
        # ignore compare key
        if sample_idx == 'compare':
            continue
        # simulate real
        sim_results_per_ct[sample_idx] = {}
        # real_phys = simulation_phys(img_data[sample_idx]['original'][:128,:128,:128])
        real_phys = simulation_phys(img_data[sample_idx]['original'])
        sim_results_per_ct[sample_idx]['original'] = real_phys

        # simulate generated
        sim_results_per_ct[sample_idx]['generate'] = []
        for gen_sample in img_data[sample_idx]['generate']:
            # gen_phys = simulation_phys(gen_sample[:128,:128,:128])
            gen_phys = simulation_phys(gen_sample)
            sim_results_per_ct[sample_idx]['generate'].append(gen_phys)


    with open(f'{root_dir}/sample_{ct_idx}/phys_results_{volume_dim}.pickle', 'wb') as file:
        pickle.dump(sim_results_per_ct, file)