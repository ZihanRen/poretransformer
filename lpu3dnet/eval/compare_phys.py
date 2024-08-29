#%% evaluate VQGAN
# from matplotlib import pyplot as plt
# from tifffile import imread
# import numpy as np
# import pickle
# from hydra.experimental import compose, initialize
# from omegaconf import OmegaConf
# import os
# from cpgan.ooppnm import pnm_sim_old


# def read_pickle(file_name):
#     with open(file_name, 'rb') as f:
#         return pickle.load(f)

# def tif_to_np(f_name):
#     '''
#     convert tif to numpy
#     '''
#     img = imread(f_name)
#     img = img.astype('float32')/255
#     return img>0.5


# # initialize configuration parameters for a specific experiment
# experiment = 'ex11'
# initialize(config_path=f"../config/{experiment}")
# cfg_vqgan = compose(config_name="vqgan")
# cfg_transformer = compose(config_name="transformer")
# cfg_dataset = compose(config_name="dataset")

# # load pytorch model
# import torch
# from lpu3dnet.frame import vqgan

# # get the global path
# PATH = cfg_dataset.PATH.sub_vol

# def get_img_list(idx_list,ct_idx):
#     # input: ct image idx
#     # output: list of images
#     img_list = []
#     for idx in idx_list:
#         img = tif_to_np(f'{PATH}/ct_{ct_idx}/{idx}.tif')
#         img_list.append(img)
#     return img_list

# def img_list_to_np(img_list):
#     # input: list of images
#     # output: numpy array of images
#     image = np.stack(img_list,axis=0)
#     return image

# def idx_to_matrix(ct_idx,img_idx_list):
#     # input: ct image idx, list of image idx
#     # output: numpy array of images
#     img_list = get_img_list(img_idx_list,ct_idx)
#     img_matrix = img_list_to_np(img_list)
#     img_matrix = img_matrix[:,np.newaxis,...]
#     img_tensor = torch.from_numpy(img_matrix).float()

#     return img_tensor,img_matrix


# def tensor_to_np(tensor):
#     # input: tensor
#     # output: numpy array
#     return tensor.detach().numpy()

# root_path = os.path.join(cfg_dataset.checkpoints.PATH, cfg_dataset.experiment)
# epoch = 25

# model_vqgan = vqgan.VQGAN(cfg_vqgan)
# PATH_model = os.path.join(root_path,f'vqgan_epoch_{epoch}.pth')
# model_vqgan.load_state_dict(
#     torch.load(
#             PATH_model,
#             map_location=torch.device('cpu')
#                )
#     )

# model_vqgan.eval()


# num_samples_per_ct = 50
# real_img_list = []
# generate_img_list = []

# for ct_idx in range(6):

#     # randomly generate a list of image idx in the range of (0,8000)
#     img_idx = np.random.randint(0,8000-1,num_samples_per_ct)
#     # img_idx = [0,100,200,300,400,500]

#     img_tensor,img_matrix = idx_to_matrix(ct_idx,img_idx)

#     from cpgan.ooppnm import img_process
#     img_prc = img_process.Image_process()

#     with torch.no_grad():
#         decode_img,info,_ = model_vqgan(img_tensor)
#         decode_img = img_prc.clean_img(decode_img)
#     # insert a dimension in the channel

#     decode_img = decode_img[:,np.newaxis,...]
#     generate_img_list.append(decode_img)
#     real_img_list.append(img_matrix)


# # concatenate list elements to form a numpy array - at 1st dimension
# real_img = np.concatenate(real_img_list,axis=0)
# generate_img = np.concatenate(generate_img_list,axis=0)
# # %%
# # simulate pnm on real and generate images
# def simulation_phys(img):
#     data_pnm = pnm_sim_old.Pnm_sim(im=img)
#     data_pnm.network_extract()
#     if data_pnm.error == 1:
#         print('Error in network extraction')
#         return None
#         # raise ValueError('Error in network extraction')
        
#     data_pnm.init_physics()
#     data_pnm.get_absolute_perm()
#     data_pnm.invasion_percolation()
#     data_pnm.kr_simulation()
#     data_pnm.close_ws()
#     return data_pnm.data_tmp

# real_phys_list = []
# generate_phys_list = []
# for i in range(len(real_img)):
#     try:
#         real_phys = simulation_phys(real_img[i][0])
#         generate_phys = simulation_phys(generate_img[i][0])
#     except:
#         print(f"Error in simulation {i}")
#         real_phys = None
#         generate_phys = None
#         continue
        
#     real_phys_list.append(real_phys)
#     generate_phys_list.append(generate_phys)

# # save the results with pickle
# with open('db/real_phys.pkl','wb') as f:
#     pickle.dump(real_phys_list,f)

# with open('db/generate_phys.pkl','wb') as f:
#     pickle.dump(generate_phys_list,f)


#%% transformer sampling validation
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



# read pickle file fiven ct and vol_dim
def load_data(ct_idx, vol_dim, root_dir):
    file_path = f'{root_dir}/sample_{ct_idx}/img_gen_vol_{vol_dim}.pkl'
    with open(file_path, 'rb') as file:
        img_results = pickle.load(file)
    return img_results


# gloabl variables
volume_dim = 4
root_dir = '../post_process/db'
img_crop_list_all = []


crop_dim = 3


for ct_idx in range(6):

    img_data = load_data(ct_idx, volume_dim, root_dir)

    def crop_img(img_tmp):
        img_crop_list = []
        for i in range(crop_dim):
            for j in range(crop_dim):
                for k in range(crop_dim):
                    img_crop = img_tmp[i*64:(i+1)*64,j*64:(j+1)*64,k*64:(k+1)*64]
                    img_crop_list.append(img_crop)
        return img_crop_list


    # append all imgs to the list
    for i in range(4):
        img_generate = img_data[i]['generate']
        for img_large in img_generate:
            img_tmp = crop_img(img_large)
            img_crop_list_all.extend(img_tmp)




# %%
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

generate_phys_list = []

for i in range(len(img_crop_list_all)):
    if i > 400:
        break
    try:
        generate_phys = simulation_phys(img_crop_list_all[i])
    except:
        print(f"Error in simulation {i}")
        real_phys = None
        generate_phys = None
        continue
        
    generate_phys_list.append(generate_phys)

with open('db/generate_phys_transformer.pkl','wb') as f:
    pickle.dump(generate_phys_list,f)