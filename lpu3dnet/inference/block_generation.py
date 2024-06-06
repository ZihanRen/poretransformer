# modules to generate 3D blocks using GPT based transformer
from matplotlib import pyplot as plt
from tifffile import imread
import numpy as np
import pickle
from hydra.experimental import compose, initialize
import os
import torch
from lpu3dnet.frame import vqgan
from lpu3dnet.frame import transformer

# some general preprocessing function
from cpgan.ooppnm import img_process
img_prc = img_process.Image_process()


class Block_generator_stochastic:
    def __init__(self,
                 cfg_dataset,
                 cfg_vqgan,
                 cfg_transformer,
                 epoch_vqgan,
                 epoch_transformer,
                 device,
                 total_features=64,
                 patch_num=8,
                 volume_dimension = 3):
        
        # model initialization
        self.epoch_vqgan = epoch_vqgan
        self.epoch_transformer = epoch_transformer



        self.device = device
        root_path = os.path.join(cfg_dataset.checkpoints.PATH, cfg_dataset.experiment)

        vqgan_path = os.path.join(root_path,f'vqgan_epoch_{self.epoch_vqgan}.pth')
        transformer_path = os.path.join(root_path,'transformer',f'transformer_epoch_{self.epoch_transformer}.pth')

        self.model_vqgan = vqgan.VQGAN(cfg_vqgan)
        self.model_transformer = transformer.Transformer(cfg_transformer)

        self.model_vqgan.load_checkpoint(vqgan_path)
        self.model_transformer.load_checkpoint(transformer_path)

        self.model_vqgan = self.model_vqgan.to(device)
        self.model_transformer = self.model_transformer.to(device)

        self.model_transformer.eval()
        self.model_vqgan.eval()

        self.total_features = total_features
        self.patch_num = patch_num

        self.sos_token = cfg_transformer.train.sos_token
        self.sos_tokens = torch.ones(1, self.total_features) * self.sos_token
        self.sos_tokens = self.sos_tokens.long().to(self.device)

        # some results 
        self.ds_spatial = None
        self.windows_idx = None

        self.window_size = 2 # sliding window size is 2x2x2 # VERY CERTAIN TODO
        self.volume_dimension = volume_dimension 

        # initialize data structure
        self.generate_sliding_windows()
        self.init_ds_spatial_info()

  
    

    def generate_sliding_windows(self, window_size=2):
        """
        Generates coordinates for sliding windows in a 3D cubic volume.
        Args:
        - volume_dimension: The size of the cubic volume
        - window_size: The size of the sliding attention window (default is 2 for a 2x2x2 window)
        
        Returns:
        - A list of lists, where each inner list contains tuples of (i, j, k) coordinates
        for all points within a window.
        e.g. (3,3,3) ijk here are absolute coordinates of the window
        """

        self.windows_idx = []

        # Traverse the 3D volume
        for i in range(self.volume_dimension - window_size + 1):
            for j in range(self.volume_dimension - window_size + 1):
                for k in range(self.volume_dimension - window_size + 1):
                    # Initialize the current window's list of coordinates
                    current_window = []
                    
                    # Populate the current window with coordinates
                    for di in range(window_size):
                        for dj in range(window_size):
                            for dk in range(window_size):
                                current_window.append((i+di, j+dj, k+dk))
                    
                    # Add the current window's coordinates to the main list
                    self.windows_idx.append(current_window)

    
    def init_ds_spatial_info(self,phi_small=0.05,phi_large=0.35):
        self.ds_spatial = {}
        for i in range(self.volume_dimension):
            for j in range(self.volume_dimension):
                for k in range(self.volume_dimension):
                    # store spatial info
                    self.ds_spatial[(i, j, k)] = {}
                    phi_gen = torch.rand(1) * (phi_large - phi_small) + phi_small
                    self.ds_spatial[(i, j, k)]['phi'] = phi_gen.item()
                    self.ds_spatial[(i, j, k)]['token'] = None
                    self.ds_spatial[(i, j, k)]['z'] = None
                    self.ds_spatial[(i, j, k)]['cond'] = None
                    self.ds_spatial[(i, j, k)]['img'] = None
                    self.ds_spatial[(i, j, k)]['phi_gen'] = None
        

    def expand_cond_single(self,cond_base):
        cond_flatten = cond_base.unsqueeze(0).expand(self.total_features, -1)
        return cond_flatten.unsqueeze(0).float()


    def gen_img_from_z(self,z):
        with torch.no_grad():
            img = self.model_vqgan.decode(z)
            img = img_prc.clean_img(img)[0]
        return img
    
    def generate_token(self,token_input,cond_input,top_k,temperature):

        token_nxt = self.model_transformer.model.sample(
            token_input,
            cond_input,
            temperature=temperature,
            top_k=top_k,
            features_num=self.total_features
            )
        
        return token_nxt


    def init_with_dummy(self,repeat=2,top_k=12,temperature=1):
        '''
        repeat 1 time, no dummy at all
        repeat 2 times, the second generation (conditioned to original) will be the first tokens
        repeat 3 times, the third generation will be the first token
        '''

        # initialize data structure
        # self.windows_idx = self.generate_sliding_windows()
        # self.ds_spatial = self.init_ds_spatial_info()


        # set up constant cond info
        phi = self.ds_spatial[(0,0,0)]['phi']
        cond_vec = torch.tensor([phi, 0, 0, 0]).to(self.device)
        cond_vec = self.expand_cond_single(cond_vec)
        cond_list = []
        token_list = [self.sos_tokens]
        # dummy generation
        for _ in range(repeat):
            cond_list.append(cond_vec)
            cond_input = torch.cat(cond_list,dim=1)
            token_input = torch.cat(token_list,dim=1)
            token_nxt = self.generate_token(token_input,cond_input,top_k=top_k,temperature=temperature)
            token_list.append(token_nxt)
        
        z_current = self.model_vqgan.tokens_to_z(token_list[-1],total_features_vec_num=self.total_features)
        self.ds_spatial[(0,0,0)]['cond'] = cond_vec
        self.ds_spatial[(0,0,0)]['z'] = z_current
        self.ds_spatial[(0,0,0)]['token'] = token_list[-1]
        self.ds_spatial[(0,0,0)]['img'] = self.gen_img_from_z(z_current)
        self.ds_spatial[(0,0,0)]['phi_gen'] = img_prc.phi(self.ds_spatial[(0,0,0)]['img'])
        
        # return accumulative dummy tokens and cond info - ignoring the last one (which is the first token)
        return token_list[:-1],cond_list[:-1]


    def generate_block_per_window(self,slide_window_idx,first_window,temperature=1,top_k=4,repeat=2):

        # input: slide window idx: list of ijk in certain attention window
        # output: updated ds_spatial
        # token and condition list initialization
        if first_window:
            # len(cond_list) + 1 = len(token_list)
            token_list, cond_list = self.init_with_dummy(repeat=repeat)
        else:
            token_list = [self.sos_tokens]
            cond_list = []

        
        flat_idx = 0

        # create normalized ijk condition vector
        for i in range(self.window_size):
            for j in range(self.window_size):
                for k in range(self.window_size):
                    # get absolute
                    abs_ijk = slide_window_idx[flat_idx]
                    

                    if self.ds_spatial[abs_ijk]['token'] is not None:
                        cond_vec = self.ds_spatial[abs_ijk]['cond'].clone()
                        token_current = self.ds_spatial[abs_ijk]['token'].clone()
                        cond_list.append(cond_vec)
                        token_list.append(token_current)
                        flat_idx += 1
                        continue # no need for inference
                    

                    
                    phi = self.ds_spatial[abs_ijk]['phi']
                    # generate conditional informatino
                    cond_vec = torch.tensor([phi, i, j, k]).to(self.device)
                    cond_vec = self.expand_cond_single(cond_vec)
                    
                    # add current conditioanl vector
                    cond_list.append(cond_vec)
                    cond_input = torch.cat(cond_list[-self.patch_num:], dim=1)

                    # aggregate previous tokens
                    token_input = torch.cat(token_list[-self.patch_num:], dim=1)
                    token_nxt = self.model_transformer.model.sample(
                                                                    token_input,
                                                                    cond_input,
                                                                    temperature=temperature,
                                                                    top_k=top_k,
                                                                    features_num=self.total_features
                                                                    )
                    token_list.append(token_nxt)

                    # update spatial ds
                    self.ds_spatial[abs_ijk]['token'] = token_nxt
                    self.ds_spatial[abs_ijk]['cond'] = cond_vec
                    z_current = self.model_vqgan.tokens_to_z(token_nxt,total_features_vec_num=self.total_features)
                    self.ds_spatial[abs_ijk]['z'] = z_current
                    self.ds_spatial[abs_ijk]['img'] = self.gen_img_from_z(z_current)
                    self.ds_spatial[abs_ijk]['phi_gen'] = img_prc.phi(self.ds_spatial[abs_ijk]['img'])
                    
                    flat_idx += 1
    
    def generate_block(self,first_window=True,temperature=1,top_k=4,repeat=2):
        # temperature and top_k are not designed for dummy sampling
        # only for general transformer generation
        first_window = True
        for idx in range(len(self.windows_idx)):
            slide_window_idx = self.windows_idx[idx]
            self.generate_block_per_window(slide_window_idx,first_window,temperature=temperature,top_k=top_k,repeat=repeat)
            first_window = False


    
