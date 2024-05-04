#%% import numpy as np
from tifffile import imread
import os
from hydra.experimental import compose, initialize
import torch
from matplotlib import pyplot as plt
from tifffile import imread
import numpy as np
import pickle
from hydra.experimental import compose, initialize
import os
import torch
from lpu3dnet.post_process.util import *



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

volume_dimension = 3

for sample_idx in range(len(img_list)):

    
    img_sample = img_list[sample_idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_transformer,model_vqgan = load_models(
        cfg_dataset,
        cfg_transformer,
        cfg_vqgan,
        device,
        epoch_transformer=170,
        epoch_vqgan=25
        )


    sampling_par = {
        'temperature': 1,
        'top_k': 4
    }

    generator = Generate_Spatial_Ds(
        model_transformer,
        model_vqgan,
        volume_dimension=3,
        img_sample=img_sample,
        device=device,
        sampling_par=sampling_par,
        with_original=True
        )


    ds_spatial, volume, volume_original = generator.main()
    phi_list,phi_gen_list = generator.compare_porosity()
    img_output = {'generated': volume, 'original': volume_original}
    
    os.makedirs(f'data_ref/sample_{sample_idx}',exist_ok=True)
    with open(f'data_ref/sample_{sample_idx}/img_output_sample_{sample_idx}_vol_{volume_dimension}.pkl', 'wb') as f:
        pickle.dump(img_output, f)



# %%
# f = plt.figure()
# plt.imshow(volume[:,40,:], cmap='gray')
# f = plt.figure()
# plt.imshow(volume_original[:,0,:], cmap='gray')
# # %%

# f = plt.figure()
# plt.scatter(phi_list, phi_gen_list,s=10,c='b')
# plt.plot([0,0.6],[0,0.6],c='r')
# plt.xlim([0,0.5])
# plt.ylim([0,0.5])
# plt.xlabel('Original Phi')
# plt.ylabel('Decoded Phi')
# plt.show()
# %%
