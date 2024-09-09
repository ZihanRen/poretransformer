import numpy as np
import numpy as np
from tifffile import imread
import os
from hydra.experimental import compose, initialize
import torch
from tifffile import imread
import numpy as np
from hydra.experimental import compose, initialize
import os
import torch
from lpu3dnet.inference import block_generation_singlecond
from lpu3dnet.inference import block_generation

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
    volume_original = np.zeros((volume_shape[0] * 64, volume_shape[1] * 64, volume_shape[2] * 64))

    for ijk, data in ds_spatial.items():
        i, j, k = ijk
        image = data['img']
        image_original = data['img_original']
        volume[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64] = image
        volume_original[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64] = image_original

    return volume,volume_original

def tif_to_np(f_name):
    '''
    convert tif to numpy
    '''
    img = imread(f_name)
    img = img.astype('float32')/255
    return img>0.5



def generate_compare_img(img_idx=1,volume_dimension=3,epoch_transformer=250):

    initialize(config_path=f"../config/ex12")
    cfg_vqgan = compose(config_name="vqgan")
    cfg_transformer = compose(config_name="transformer")
    cfg_dataset = compose(config_name="dataset")

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

    img_idx = 1
    volume_dimension = 3
    img_sample = img_list[img_idx]
    img_sample = img_sample[:volume_dimension*64,
                            :volume_dimension*64,
                            :volume_dimension*64]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_generator = block_generation_singlecond.Block_generator_compare(
        cfg_dataset,
        cfg_vqgan,
        cfg_transformer,
        epoch_vqgan=25,
        epoch_transformer=epoch_transformer,
        device = device,
        img=img_sample,
        volume_dimension=volume_dimension
        )

    block_generator.generate_block()


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
        volume_original = np.zeros((volume_shape[0] * 64, volume_shape[1] * 64, volume_shape[2] * 64))

        for ijk, data in ds_spatial.items():
            i, j, k = ijk
            image = data['img']
            image_original = data['img_original']
            volume[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64] = image
            volume_original[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64] = image_original

        return volume,volume_original

    volume,_ = assemble_volume(block_generator.ds_spatial)

    return volume




def generate_compare_img_morecond(img_idx=1,volume_dimension=3,epoch_transformer=250):

    initialize(config_path=f"../config/ex11")
    cfg_vqgan = compose(config_name="vqgan")
    cfg_transformer = compose(config_name="transformer")
    cfg_dataset = compose(config_name="dataset")

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

    img_idx = 1
    volume_dimension = 3
    img_sample = img_list[img_idx]
    img_sample = img_sample[:volume_dimension*64,
                            :volume_dimension*64,
                            :volume_dimension*64]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_generator = block_generation.Block_generator_compare(
        cfg_dataset,
        cfg_vqgan,
        cfg_transformer,
        epoch_vqgan=25,
        epoch_transformer=epoch_transformer,
        device = device,
        img=img_sample,
        volume_dimension=volume_dimension
        )

    block_generator.generate_block()


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
        volume_original = np.zeros((volume_shape[0] * 64, volume_shape[1] * 64, volume_shape[2] * 64))

        for ijk, data in ds_spatial.items():
            i, j, k = ijk
            image = data['img']
            image_original = data['img_original']
            volume[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64] = image
            volume_original[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64] = image_original

        return volume,volume_original

    volume,_ = assemble_volume(block_generator.ds_spatial)

    return volume