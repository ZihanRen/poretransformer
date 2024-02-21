#%% load training loss
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
import torch.nn.functional as F
from lpu3dnet.frame import transformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# initialize configuration parameters for a specific experiment
experiment = 'ex6'
initialize(config_path=f"../config/{experiment}")
cfg_vqgan = compose(config_name="vqgan")
cfg_transformer = compose(config_name="transformer")
PATH = cfg_vqgan.data.PATH.sub_vol


root_path = os.path.join(cfg_vqgan.checkpoints.PATH, cfg_vqgan.experiment)

#%% load the models
# get the global path
# model_vqgan = vqgan.VQGAN(cfg_vqgan)

# #%% validation loss of VQGAN
# # generate random img idx
# random.seed(1340)
# # Generate a list of unique random integers
# num_elements = 50
# min_value = 0
# max_value = 7900
# ct_idx = 0
# test_idx = 1
# start_epoch = 0
# end_epoch = 45

# from cpgan.ooppnm import img_process
# img_prc = img_process.Image_process()

# random_img_idx = random.sample(
#     range(min_value, max_value + 1), num_elements
#     )


# val_loss = []
# epoch_list = []
# test_loss = []

# for epoch in range(start_epoch,end_epoch+5,5):
#     PATH_model = os.path.join(root_path,f'vqgan_epoch_{epoch}.pth')
#     model_vqgan.load_state_dict(
#         torch.load(
#                 PATH_model,
#                 map_location=torch.device('cpu')
#                 )
#         )

#     model_vqgan.eval()



#     img_tensor,img_matrix = idx_to_matrix(ct_idx,random_img_idx)

#     with torch.no_grad():
#         decode_img,codebook_indice,_ = model_vqgan(img_tensor)
#         decode_img = img_prc.clean_img(decode_img)
#         decode_img = decode_img[:,np.newaxis,...]
    
#     rec_loss = np.mean((decode_img - img_matrix) ** 2)
#     val_loss.append(rec_loss)

#     img_tensor,img_matrix = idx_to_matrix(test_idx,random_img_idx)

#     with torch.no_grad():
#         decode_img,codebook_indice,_ = model_vqgan(img_tensor)
#         decode_img = img_prc.clean_img(decode_img)
#         decode_img = decode_img[:,np.newaxis,...]
    
#     rec_loss = np.mean((decode_img - img_matrix) ** 2)
#     test_loss.append(rec_loss)
#     epoch_list.append(epoch)

# # save epoch and val loss into pickle
# save_obj = {'epoch':epoch_list,'val_loss':val_loss,'test_loss':test_loss}
# with open(f'{root_path}/val_loss_ct{ct_idx}.pkl', 'wb') as f:
#     pickle.dump(save_obj, f)


#%% validation loss of transformer
# load transformer model
val_loss = []
epoch_list = []
test_loss = []

def load_vqgan(PATH_model,cfg_vqgan,device):
    model_vqgan = vqgan.VQGAN(cfg_vqgan)
    model_vqgan.load_state_dict(
        torch.load(
                PATH_model,
                device
                )
        )
    model_vqgan.to(device)
    model_vqgan.eval()

    return model_vqgan

def load_transformer(PATH_model,cfg_transformer,device):
    model_transformer = transformer.Transformer(cfg_transformer)
    model_transformer.load_state_dict(
        torch.load(
                PATH_model,
                device
                )
        )
    model_transformer.to(device)
    model_transformer.eval()

    return model_transformer


def perturb_idx(cfg_transformer,img_tokens,device,sos_token=3000):

    sos_tokens = torch.ones(img_tokens.shape[0], 1) * sos_token
    sos_tokens = sos_tokens.long().to(device)

    mask = torch.bernoulli(
        cfg_transformer.train.p_keep * torch.ones(
        img_tokens.shape, device=device)
        )

    mask = mask.round().to(dtype=torch.int64)

    random_indices = torch.randint_like(
        img_tokens,
        cfg_transformer.architecture.vocab_size
                                        )

    perturbed_indices = mask * img_tokens + (1 - mask) * random_indices
    perturbed_indices = torch.cat((sos_tokens, perturbed_indices), dim=1)

    target = img_tokens
    perturbed_indices = perturbed_indices[:, :-1]

    return perturbed_indices, target

random.seed(1340)
# Generate a list of unique random integers
num_elements = 50
min_value = 0
max_value = 7900
ct_idx = 0
test_idx = 1
start_epoch = 0
end_epoch = 55

from cpgan.ooppnm import img_process
img_prc = img_process.Image_process()

random_img_idx = random.sample(
    range(min_value, max_value + 1), num_elements
    )


val_loss = []
epoch_list = []
test_loss = []
transformer_path = os.path.join(root_path, 'transformer')

PATH_model = os.path.join(root_path,f'vqgan_epoch_10.pth')
model_vqgan = load_vqgan(PATH_model,cfg_vqgan,device)

for epoch in range(start_epoch,end_epoch+5,5):
    transformer_path_epoch = os.path.join(transformer_path,f'transformer_epoch_{epoch}.pth')

    model_transformer = load_transformer(transformer_path_epoch,cfg_transformer,device)
    img_tensor,img_matrix = idx_to_matrix(ct_idx,random_img_idx)

    with torch.no_grad():
        img_tokens = model_vqgan.gen_img_tokens(img_tensor.to(device))
        perturb,target = perturb_idx(cfg_transformer,img_tokens,device)
        _,loss = model_transformer(perturb,target)
        val_loss.append(loss.item())
        epoch_list.append(epoch)
    

    img_tensor,img_matrix = idx_to_matrix(test_idx,random_img_idx)

    with torch.no_grad():
        img_tokens = model_vqgan.gen_img_tokens(img_tensor.to(device))
        perturb,target = perturb_idx(cfg_transformer,img_tokens,device)
        _,loss = model_transformer(perturb,target)
        test_loss.append(loss.item())

    
# save epoch and val loss into pickle
save_obj = {'epoch':epoch_list,'val_loss':val_loss,'test_loss':test_loss}
with open(f'{root_path}/transformer_val_loss_ct{ct_idx}_{test_idx}.pkl', 'wb') as f:
    pickle.dump(save_obj, f)


# %%
