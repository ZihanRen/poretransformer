from lpu3dnet.inference import block_generation_singlecond
import numpy as np
from hydra.experimental import compose, initialize
import torch
import pickle
import torch
from cpgan.ooppnm import pnm_sim_old

# initialize configuration parameters for a specific experiment
experiment = 'ex12'
initialize(config_path=f"../config/{experiment}")
cfg_vqgan = compose(config_name="vqgan")
cfg_transformer = compose(config_name="transformer")
cfg_dataset = compose(config_name="dataset")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
volume_dim = 6

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

def get_volume_shape(ds_spatial):
    max_i, max_j, max_k = 0, 0, 0
    for ijk in ds_spatial.keys():
        i, j, k = ijk
        max_i = max(max_i, i)
        max_j = max(max_j, j)
        max_k = max(max_k, k)
    return max_i + 1, max_j + 1, max_k + 1

def assemble_volume(ds_spatial):
    volume_shape = get_volume_shape(ds_spatial)
    volume = np.zeros((volume_shape[0] * 64, volume_shape[1] * 64, volume_shape[2] * 64))

    for ijk, data in ds_spatial.items():
        i, j, k = ijk
        image = data['img']
        volume[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64] = image

    return volume

result = {}

for cons_phi in [0.15,0.2]:

    block_generator = block_generation_singlecond.Block_generator_stochastic(
        cfg_dataset,
        cfg_vqgan,
        cfg_transformer,
        epoch_vqgan=25,
        epoch_transformer=50,
        device = device,
        volume_dimension=volume_dim,
        constant_phi=cons_phi
        )

    block_generator.generate_block(repeat=4)

    volume = assemble_volume(block_generator.ds_spatial)

    phys_homo = simulation_phys(volume)
    result[cons_phi] = {}
    result[cons_phi]['phys'] = phys_homo
    result[cons_phi]['img'] = volume



# save to pickle
with open(f'phys_results_homo.pickle', 'wb') as file:
    pickle.dump(result, file)