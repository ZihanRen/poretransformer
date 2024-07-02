#%% import numpy as np
from tifffile import imread
import os
from hydra.experimental import compose, initialize
import torch
from tifffile import imread
import numpy as np
import pickle
from hydra.experimental import compose, initialize
import os
import torch
from lpu3dnet.post_process.util import *
# from lpu3dnet.inference import block_generation
from lpu3dnet.inference import block_generation_singlecond as block_generation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tif_to_np(f_name):
    '''
    convert tif to numpy
    '''
    img = imread(f_name)
    img = img.astype('float32')/255
    return img>0.5

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


# setting initial parameters
volume_dimension = 4
epoch_vqgan = 25
epoch_transformer = 80



# number of samples per ct
num_samples = 8


# transformer generated per sample
realization_num = 30


#%% random sampling

def sample_subvolumes(original_img, num_meta_volumes, num_samples):
    # Calculate the dimensions of the subvolume
    subvolume_dim = 64 * num_meta_volumes
    
    # Check if the subvolume dimensions are larger than the original image dimensions
    if any(subvolume_dim > dim for dim in original_img.shape):
        raise ValueError("The subvolume dimensions exceed the original image dimensions.")
    
    subvolumes = []  # List to hold the subvolumes
    for _ in range(num_samples):
        # Determine the maximum starting index for each dimension to fit the subvolume
        max_start_i = original_img.shape[0] - subvolume_dim
        max_start_j = original_img.shape[1] - subvolume_dim
        max_start_k = original_img.shape[2] - subvolume_dim
        
        # Randomly select a starting index for the subvolume within the allowable range
        start_i = np.random.randint(0, max_start_i + 1)
        start_j = np.random.randint(0, max_start_j + 1)
        start_k = np.random.randint(0, max_start_k + 1)
        
        # Extract the subvolume and add to the list
        subvolume = original_img[start_i:start_i + subvolume_dim,
                                 start_j:start_j + subvolume_dim,
                                 start_k:start_k + subvolume_dim]
        subvolumes.append(subvolume)
    
    return subvolumes


def generate_compare_list(img_list, ct_idx,num_samples_per_ct=4):
    compare_list = []
    for i in range(6):
        if i == ct_idx:
            continue
        current_list = sample_subvolumes(img_list[ct_idx], volume_dimension, num_samples_per_ct)
        compare_list.extend(current_list)
    return compare_list

#%% wrapper

def generate_imgs_given_ctidx(ct_idx,num_samples=4,realization_num=4):

    samples_list = sample_subvolumes(img_list[ct_idx], volume_dimension, num_samples)
    results = {}

    for sample_idx in range(len(samples_list)):

        img_sample = samples_list[sample_idx]
        results[sample_idx] = {}

        results[sample_idx]['original'] = img_sample
        results[sample_idx]['generate'] = []
        
        for _ in range(realization_num):

            block_generator = block_generation.Block_generator_compare(
            cfg_dataset,
            cfg_vqgan,
            cfg_transformer,
            epoch_vqgan=epoch_vqgan,
            epoch_transformer=epoch_transformer,
            device = device,
            img=img_sample,
            volume_dimension=volume_dimension
            )

            block_generator.generate_block()
            volume,_ = assemble_volume(block_generator.ds_spatial)
            results[sample_idx]['generate'].append(volume)

    # the naming should include sample number and volume dimension
    # generate compare data:
    compare_data = generate_compare_list(img_list, ct_idx,num_samples_per_ct=4)
    results['compare'] = compare_data
    os.makedirs(f'db/sample_{ct_idx}',exist_ok=True)
    with open(f'db/sample_{ct_idx}/img_gen_vol_{volume_dimension}.pkl', 'wb') as f:
        pickle.dump(results, f)



for ct_idx in range(6):
    generate_imgs_given_ctidx(
        ct_idx,
        num_samples=num_samples,
        realization_num=realization_num
        )