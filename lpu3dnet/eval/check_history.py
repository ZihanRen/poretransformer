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
experiment = 'ex7'
initialize(config_path=f"../config/{experiment}")
cfg_vqgan = compose(config_name="vqgan")
cfg_transformer = compose(config_name="transformer")
cfg_dataset = compose(config_name="dataset")
PATH = cfg_dataset.PATH.sub_vol


root_path = os.path.join(cfg_dataset.checkpoints.PATH, cfg_vqgan.experiment)

#%% load the models
# get the global path
model_vqgan = vqgan.VQGAN(cfg_vqgan)

# #%% validation loss of VQGAN
# # generate random img idx
random.seed(1340)
# Generate a list of unique random integers
num_elements = 50
min_value = 0
max_value = 7900
ct_idx = 0
test_idx = 1
start_epoch = 0
end_epoch = 20

from cpgan.ooppnm import img_process
img_prc = img_process.Image_process()

random_img_idx = random.sample(
    range(min_value, max_value + 1), num_elements
    )


val_loss = []
epoch_list = []
test_loss = []

for epoch in range(start_epoch,end_epoch+5,5):
    PATH_model = os.path.join(root_path,f'vqgan_epoch_{epoch}.pth')
    model_vqgan.load_state_dict(
        torch.load(
                PATH_model,
                map_location=torch.device('cpu')
                )
        )

    model_vqgan.eval()



    img_tensor,img_matrix = idx_to_matrix(ct_idx,random_img_idx)

    with torch.no_grad():
        decode_img,codebook_indice,_ = model_vqgan(img_tensor)
        decode_img = img_prc.clean_img(decode_img)
        decode_img = decode_img[:,np.newaxis,...]
    
    rec_loss = np.mean((decode_img - img_matrix) ** 2)
    val_loss.append(rec_loss)

    img_tensor,img_matrix = idx_to_matrix(test_idx,random_img_idx)

    with torch.no_grad():
        decode_img,codebook_indice,_ = model_vqgan(img_tensor)
        decode_img = img_prc.clean_img(decode_img)
        decode_img = decode_img[:,np.newaxis,...]
    
    rec_loss = np.mean((decode_img - img_matrix) ** 2)
    test_loss.append(rec_loss)
    epoch_list.append(epoch)

# save epoch and val loss into pickle
save_obj = {'epoch':epoch_list,'val_loss':val_loss,'test_loss':test_loss}
with open(f'{root_path}/val_loss_ct{ct_idx}.pkl', 'wb') as f:
    pickle.dump(save_obj, f)


#%% validation loss of transformer
# load transformer model

# import torch
# import os
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import hydra
# import glob

# def extract_index(filename):
#     # Extracts the numeric index from a filename
#     return int(filename.split('_')[-1].split('.')[0])


# class Dataset_transformer(Dataset):
    
  

#     def __init__(
#         self,
#         cfg,
#         device
#         ):
#         self.device = device
#         self.ct_idx = cfg.ct_idx
#         self.img_tokens_path = cfg.PATH.image_tokens
#         self.img_cond_path = cfg.PATH.image_tokens_cond

#         self.img_token_PATH = []
#         self.img_cond_PATH = []

#         self.token_files = []
#         self.cond_files = []

#         for idx in self.ct_idx:
#             token_files_ct = glob.glob(os.path.join(self.img_tokens_path, f'ct_{idx}', 'tokens_*.pt'))
#             cond_files_ct = glob.glob(os.path.join(self.img_cond_path, f'ct_{idx}', 'cond_*.pt'))
#             self.token_files.extend(sorted(token_files_ct, key=extract_index))
#             self.cond_files.extend(sorted(cond_files_ct, key=extract_index))
#         assert len(self.token_files) == len(self.cond_files) 

#     def sample_data(self, num_samples):
#             # Ensure num_samples does not exceed the dataset size for ct_0 and ct_1
#             max_samples = min([len(os.listdir(os.path.join(self.img_tokens_path, 'ct_0'))),
#                             len(os.listdir(os.path.join(self.img_tokens_path, 'ct_1')))])
#             assert num_samples <= max_samples, "num_samples exceeds the available dataset size."

#             # Sample unique indices for ct_0
#             sampled_indices_ct_0 = np.random.choice(range(max_samples), size=num_samples, replace=False)

#             # Use the same indices for ct_1 to maintain alignment
#             sampled_indices_ct_1 = sampled_indices_ct_0.copy()

#             # Collect conditional vectors and tokens for the sampled indices from ct_0 and ct_1
#             sampled_cond_ct_0, sampled_tokens_ct_0 = [], []
#             sampled_cond_ct_1, sampled_tokens_ct_1 = [], []

#             for idx in sampled_indices_ct_0:
#                 cond_path_ct_0 = os.path.join(self.img_cond_path, f'ct_0', f'cond_{idx}.pt')
#                 token_path_ct_0 = os.path.join(self.img_tokens_path, f'ct_0', f'tokens_{idx}.pt')
#                 sampled_cond_ct_0.append(torch.load(cond_path_ct_0).float().to(self.device))
#                 sampled_tokens_ct_0.append(torch.load(token_path_ct_0).to(self.device))

#             for idx in sampled_indices_ct_1:
#                 cond_path_ct_1 = os.path.join(self.img_cond_path, f'ct_1', f'cond_{idx}.pt')
#                 token_path_ct_1 = os.path.join(self.img_tokens_path, f'ct_1', f'tokens_{idx}.pt')
#                 sampled_cond_ct_1.append(torch.load(cond_path_ct_1).float().to(self.device))
#                 sampled_tokens_ct_1.append(torch.load(token_path_ct_1).to(self.device))
            
#             sampled_tokens_ct_0 = torch.stack(sampled_tokens_ct_0)
#             sampled_cond_ct_0 = torch.stack(sampled_cond_ct_0)
#             sampled_tokens_ct_1 = torch.stack(sampled_tokens_ct_1)
#             sampled_cond_ct_1 = torch.stack(sampled_cond_ct_1)

#             return [sampled_tokens_ct_0,sampled_cond_ct_0,sampled_tokens_ct_1,sampled_cond_ct_1]

#     def __len__(self):
#         return len(self.token_files)

#     def __getitem__(self, index):

#         token_path = self.token_files[index]
#         cond_path = self.cond_files[index]

#         token = torch.load(token_path).to(self.device)
#         cond = torch.load(cond_path).to(self.device)

#         return token[0], cond.float()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = Dataset_transformer(cfg_dataset,device)


# #%%

# val_loss = []
# epoch_list = []
# test_loss = []

# def load_transformer(PATH_model,cfg_transformer,device):
#     model_transformer = transformer.Transformer(cfg_transformer)
#     model_transformer.load_state_dict(
#         torch.load(
#                 PATH_model,
#                 device
#                 )
#         )
#     model_transformer.to(device)
#     model_transformer.eval()

#     return model_transformer


# def perturb_idx(cfg_transformer,img_tokens,device,sos_token=3000):

#     sos_tokens = torch.ones(img_tokens.shape[0], 1) * sos_token
#     sos_tokens = sos_tokens.long().to(device)

#     mask = torch.bernoulli(
#         cfg_transformer.train.p_keep * torch.ones(
#         img_tokens.shape, device=device)
#         )

#     mask = mask.round().to(dtype=torch.int64)

#     random_indices = torch.randint_like(
#         img_tokens,
#         cfg_transformer.architecture.vocab_size
#                                         )

#     perturbed_indices = mask * img_tokens + (1 - mask) * random_indices
#     perturbed_indices = torch.cat((sos_tokens, perturbed_indices), dim=1)

#     target = img_tokens
#     perturbed_indices = perturbed_indices[:, :-1]

#     return perturbed_indices

# random.seed(1340)
# # Generate a list of unique random integers
# num_elements = 50
# start_epoch = 0
# end_epoch = 150
# batch = 50


# val_loss = []
# epoch_list = []
# test_loss = []
# transformer_path = os.path.join(root_path, 'transformer')


# for epoch in range(start_epoch,end_epoch+5,5):
#     epoch_list.append(epoch)
#     transformer_path_epoch = os.path.join(transformer_path,f'transformer_epoch_{epoch}.pth')

#     model_transformer = load_transformer(transformer_path_epoch,cfg_transformer,device)
#     # sample data
#     tensor_list = dataset.sample_data(batch)
#     val_token, val_cond, test_token, test_cond = tensor_list
#     # reshape validation tokens
#     val_token = val_token[:,0,:]
#     test_token = test_token[:,0,:]

#     with torch.no_grad():
#         # calculate validation loss
#         val_perturb = perturb_idx(cfg_transformer,val_token,device)
#         test_perturb = perturb_idx(cfg_transformer,test_token,device)
#         logits_val = model_transformer(val_perturb,val_cond)
#         logits_test = model_transformer(test_perturb,test_cond)

#         loss_val = model_transformer.loss_func(logits_val,val_token)
#         loss_test = model_transformer.loss_func(logits_test,test_token)

#         val_loss.append(loss_val.item())
#         test_loss.append(loss_test.item())


# # save epoch and val loss into pickle
# save_obj = {'epoch':epoch_list,'val_loss':val_loss,'test_loss':test_loss}

# with open(f'{root_path}/transformer_val_test_loss.pkl', 'wb') as f:
#     pickle.dump(save_obj, f)


# %%
