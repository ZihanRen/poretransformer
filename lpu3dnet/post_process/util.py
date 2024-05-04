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
from lpu3dnet.frame import vqgan
from lpu3dnet.frame import transformer
from cpgan.ooppnm import img_process


#TODO: img_prc and clean image

# initialize image processing modules
img_prc = img_process.Image_process()


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



def load_models(
        cfg_dataset,
        cfg_transformer,
        cfg_vqgan,device,
        epoch_transformer,
        epoch_vqgan=25):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_path = os.path.join(cfg_dataset.checkpoints.PATH, cfg_dataset.experiment)

    vqgan_path = os.path.join(root_path,f'vqgan_epoch_{epoch_vqgan}.pth')
    transformer_path = os.path.join(root_path,'transformer',f'transformer_epoch_{epoch_transformer}.pth')

    model_vqgan = vqgan.VQGAN(cfg_vqgan)
    model_transformer = transformer.Transformer(cfg_transformer)

    model_vqgan.load_checkpoint(vqgan_path)
    model_transformer.load_checkpoint(transformer_path)

    model_vqgan = model_vqgan.to(device)
    model_transformer = model_transformer.to(device)

    model_transformer.eval()
    model_vqgan.eval()

    return model_transformer,model_vqgan


def expand_cond_single(cond_base,features_num=27):
    '''
    expandsion of conditional vectors to make them compatitable with transformer inputs
    '''
    cond_flatten = cond_base.unsqueeze(0).expand(features_num, -1)
    return cond_flatten.unsqueeze(0).float()


def gen_img_from_z(z,model_vqgan):
    with torch.no_grad():
        img = model_vqgan.decode(z)
        img = img_prc.clean_img(img)[0]
    return img


def add_noise_to_cond(cond,device):
    b,seq_len,_ = cond.shape
    noise = torch.randn(b, seq_len, 1).to(device)
    cond = torch.cat([cond, noise], dim=-1)
    return cond.float()



class Generate_Spatial_Ds():
    def __init__(self,
                 model_transformer,model_vqgan,
                 volume_dimension,img_sample,
                 device,
                 sampling_par,
                 features_num=27,
                 add_noise=False,
                 sos_token=3000,
                 with_original=True
                 ):
        self.device = device
        self.img_sample = img_sample # original image sample 
        self.model_transformer = model_transformer
        self.model_vqgan = model_vqgan
        self.volume_dimension = volume_dimension
        self.sub_window_size = 2
        self.sample_par = sampling_par # {top_k:2,temperature:1.0}
        self.add_noise = add_noise
        self.features_num = features_num
        self.sos_token = sos_token
        self.sos_tokens = torch.ones(1, self.features_num) * self.sos_token
        self.sos_tokens = self.sos_tokens.long().to(self.device)
        self.with_original = with_original

        
        self.ds_spatial = {}
        self.init_ds_spatial_info(self.volume_dimension)
    

    def init_ds_spatial_info(self,volume_dimension):
        
        for i in range(volume_dimension):
            for j in range(volume_dimension):
                for k in range(volume_dimension):
                    # store spatial info
                    self.ds_spatial[(i, j, k)] = {}
                    self.ds_spatial[(i, j, k)]['phi'] = None
                    self.ds_spatial[(i, j, k)]['token'] = None
                    self.ds_spatial[(i, j, k)]['z'] = None
                    self.ds_spatial[(i, j, k)]['cond'] = None
                    self.ds_spatial[(i, j, k)]['img'] = None
    

    def patchify_img(self):
        # segment images into 64^3 cubic voxels into spatial grid and save them to ds_spatial data structure
        # only segment the left up corner
        # TODO: random segmentation is necessary

        for i in range(self.volume_dimension):
            for j in range(self.volume_dimension):
                for k in range(self.volume_dimension):
                    subset_image = self.img_sample[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64]
                    key_idx = (i, j, k)
                    self.ds_spatial[key_idx]['img_original'] = subset_image
                    porosity = ps.metrics.porosity(subset_image)
                    if porosity < 5e-2:
                        porosity = 0.1
                    self.ds_spatial[key_idx]['phi'] = porosity

    def generate_sliding_windows(self):
        """
        Generates coordinates for sliding windows in a 3D cubic volume.
        Args:
        - volume_dimension: The size of the cubic volume
        - window_size: The size of the sliding window (default is 2 for a 2x2x2 window)
        Returns:
        - A list of lists, where each inner list contains tuples of (i, j, k) coordinates
        for all points within a window.
        """

        windows = []
        volume_dimension = self.volume_dimension
        window_size = self.sub_window_size

        # Traverse the 3D volume
        for i in range(volume_dimension - window_size + 1):
            for j in range(volume_dimension - window_size + 1):
                for k in range(volume_dimension - window_size + 1):
                    # Initialize the current window's list of coordinates
                    current_window = []
                    
                    # Populate the current window with coordinates
                    for di in range(window_size):
                        for dj in range(window_size):
                            for dk in range(window_size):
                                current_window.append((i+di, j+dj, k+dk))
                    
                    # Add the current window's coordinates to the main list
                    windows.append(current_window)

        return windows
    
    def generate_ds_start(self):
        
        self.patchify_img()
        phi = self.ds_spatial[(0,0,0)]['phi']
        cond_vec = torch.tensor([phi, 0, 0, 0]).to(self.device)
        cond_vec = expand_cond_single(
            cond_vec,
            features_num=self.features_num
            )
        if self.add_noise:
            cond_vec = add_noise_to_cond(cond_vec,self.device)


        if self.with_original:
            img_origin = self.ds_spatial[(0,0,0)]['img_original']
            img_origin = np.expand_dims(img_origin, axis=0)
            img_origin = np.expand_dims(img_origin, axis=0)
            img_origin = torch.from_numpy(img_origin).to(self.device).float()
            token_nxt = self.model_vqgan.gen_img_tokens(img_origin).to(self.device)
        else:
            token_nxt = self.model_transformer.model.sample(
                self.sos_tokens,
                cond_vec,top_k=self.sample_par['top_k'],
                features_num=self.features_num,
                temperature=self.sample_par['temperature']
                ).to(self.device)

        z_current = self.model_vqgan.tokens_to_z(
            token_nxt,
            total_features_vec_num=self.features_num
            )

        self.ds_spatial[(0,0,0)]['cond'] = cond_vec
        self.ds_spatial[(0,0,0)]['z'] = z_current
        self.ds_spatial[(0,0,0)]['token'] = token_nxt
        self.ds_spatial[(0,0,0)]['img'] = gen_img_from_z(
            z_current,
            self.model_vqgan
            )
    

    def generate_ds_end(self):
        windows_idx = self.generate_sliding_windows()

        for slide_window_idx in windows_idx:
            flat_idx = 0
            # initialize aggregated vector in each window
            cond_window = []
            token_list = [self.sos_tokens]
            # create normalized ijk condition vector
            for i in range(self.sub_window_size):
                for j in range(self.sub_window_size):
                    for k in range(self.sub_window_size):
                        # get absolute
                        abs_ijk = slide_window_idx[flat_idx]
                    
                        if self.ds_spatial[abs_ijk]['token'] is not None:
                            cond_vec = self.ds_spatial[abs_ijk]['cond'].clone()
                            token_current = self.ds_spatial[abs_ijk]['token'].clone().to(self.device)
                            cond_window.append(cond_vec)
                            token_list.append(token_current)
                            flat_idx += 1
                            continue # no need for inference
                        
                        phi = self.ds_spatial[abs_ijk]['phi']
                        # generate conditional informatino
                        cond_vec = torch.tensor([phi, i, j, k]).to(self.device)
                        cond_vec = expand_cond_single(cond_vec,features_num=self.features_num)
                        if self.add_noise:
                            cond_vec = add_noise_to_cond(cond_vec,self.device)
                        cond_window.append(cond_vec)
                        cond_input = torch.cat(cond_window, dim=1)

                        # aggregate previous tokens
                        token_input = torch.cat(token_list, dim=1)
                        token_nxt = self.model_transformer.model.sample(
                            token_input,
                            cond_input,
                            top_k=self.sample_par['top_k'],
                            features_num=self.features_num,
                            temperature=self.sample_par['temperature']
                            )
                        
                        token_list.append(token_nxt)

                        # update spatial ds
                        self.ds_spatial[abs_ijk]['token'] = token_nxt
                        self.ds_spatial[abs_ijk]['cond'] = cond_vec
                        z_current = self.model_vqgan.tokens_to_z(token_nxt,total_features_vec_num=self.features_num)
                        self.ds_spatial[abs_ijk]['z'] = z_current
                        self.ds_spatial[abs_ijk]['img'] = gen_img_from_z(z_current,
                                                                         model_vqgan=self.model_vqgan
                                                                         )
                        
                        flat_idx += 1
    
    def main(self):
        self.generate_ds_start()
        self.generate_ds_end()
        volume,volume_original = assemble_volume(self.ds_spatial)
        return self.ds_spatial,volume,volume_original

    def compare_porosity(self):
        phi_list = []
        phi_gen_list = []

        for keys in self.ds_spatial.keys():
            phi_list.append(self.ds_spatial[keys]['phi'])
            img = self.ds_spatial[keys]['img']
            phi_gen = img_prc.phi(img)
            phi_gen_list.append(phi_gen)
        
        return phi_list,phi_gen_list








