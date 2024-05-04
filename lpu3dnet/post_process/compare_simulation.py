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

initialize(config_path=f"../config/ex10")
cfg_vqgan = compose(config_name="vqgan")
cfg_transformer = compose(config_name="transformer")
cfg_dataset = compose(config_name="dataset")





#%%

#TODO : random cropping
# TODO ramdom realizations
def crop_img(img,vol_dim,sub_size=64):
    return img[
                :sub_size*vol_dim,
               :sub_size*vol_dim,
               :sub_size*vol_dim
               ]

# read pickle file 
volume_dim = 3

for sample_idx in range(6):
    compare_idx = [x for x in range(6) if x != sample_idx]

    # get generated volume and original volume
    with open(
        f'data_ref/sample_{sample_idx}/img_output_sample_{sample_idx}_vol_{volume_dim}.pkl',
        'rb'
        ) as file:
        img_cluster = pickle.load(file)

    img_compare_list = []

    for i in compare_idx:
        img_compare_list.append(
            tif_to_np(
                os.path.join(
                    cfg_dataset.PATH.main_vol,
                    f'main_{i}.tif'
                    )
                    )
            ) 

    img_crop_compare_list = [crop_img(x,volume_dim) for x in img_compare_list]

    
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

    pred_df = simulation_phys(img_cluster['generated'])
    real_df = simulation_phys(img_cluster['original'])
    compare_df_list = [simulation_phys(x) for x in img_crop_compare_list]

    df_results = {'prediction':pred_df,
                'real':real_df,
                'compare':compare_df_list
                }


    with open(f'data_ref/sample_{sample_idx}/df_results_{volume_dim}.pickle', 'wb') as file:
        pickle.dump(df_results, file)