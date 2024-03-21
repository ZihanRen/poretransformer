
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tifffile import imread
import hydra
import porespy as ps
from lpu3dnet.frame import vqgan



class Dataset_vqgan(Dataset):
    def __init__(
        self,
        cfg_vqgan,
        cfg_transformer,
        device
    ):
        # cfg: configuration of VQGAN
        self.root_PATH = cfg_vqgan.data.PATH.sub_vol_large
        self.ct_idx = cfg_vqgan.data.ct_idx
        self.transform = None
        self.device = device

        self.pretrained_vqgan_epoch = cfg_transformer.train.pretrained_vqgan_epoch

        self.vqgan_path = os.path.join(
            cfg_transformer.checkpoints.PATH,
            cfg_transformer.experiment
            )

        # initialize VQGAN model
        self.vqgan = vqgan.VQGAN(cfg_vqgan).to(device=self.device)
        self.vqgan.load_state_dict(
            torch.load(
                os.path.join(self.vqgan_path, f'vqgan_epoch_{self.pretrained_vqgan_epoch}.pth'),
                map_location=torch.device(self.device)
                )
        )
        self.vqgan.to(device=self.device)
        self.vqgan.eval()

        # get training data PATH
        self.img_PATH = []
        for idx in self.ct_idx:
            PATH_folder = os.path.join(self.root_PATH, f'ct_{idx}')
            for img_name in os.listdir(PATH_folder):
                self.img_PATH.append(os.path.join(PATH_folder, img_name))

    @staticmethod
    def tif_to_np(f_name):
        '''
        convert tif to numpy
        '''
        img = imread(f_name)
        img = img.astype('float32') / 255
        return img > 0.5

    def get_img_tokens(self, img):
        with torch.no_grad():
            img_tokens = self.vqgan.gen_img_tokens(img)
        return img_tokens

    def __len__(self):
        return len(self.img_PATH)

    def __getitem__(self, index):
        # load images
        image = Dataset_vqgan.tif_to_np(self.img_PATH[index])

        # segment images into 64^3 cubic voxels in 2*2*2 spatial grid
        tokens_patch_list = []
        phi_list = []
        i_list = []
        j_list = []
        k_list = []

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    subset_image = image[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64]
                    phi_list.append(ps.metrics.porosity(subset_image))
                    subset_image = np.expand_dims(subset_image, axis=(0,1))
                    subset_image = torch.from_numpy(subset_image).float().to(self.device)
                    subset_tokens = self.get_img_tokens(subset_image)
                    tokens_patch_list.append(subset_tokens)
                    i_list.append(i)
                    j_list.append(j)
                    k_list.append(k)

        tokens_all = torch.cat(tokens_patch_list, dim=1).to(self.device)
        phi_tensor = torch.tensor(phi_list).view(-1,1).to(self.device)
        i_tensor = torch.tensor(i_list).view(-1,1).to(self.device)
        j_tensor = torch.tensor(j_list).view(-1,1).to(self.device)
        k_tensor = torch.tensor(k_list).view(-1,1).to(self.device)

        # concatenate all the conditional information
        cond = torch.cat([phi_tensor,i_tensor,j_tensor,k_tensor], dim=1)

        return tokens_all[0],cond

    def img_name(self):
        return os.listdir(self.img_PATH)

    def print_parameters(self):
        print("training CT index is {}".format(self.ct_idx))
        print("training data root path is {}".format(self.root_PATH))




if __name__ == "__main__":
    experiment_idx = 7
    with hydra.initialize(config_path=f"../config/ex{experiment_idx}"):
        cfg_vqgan = hydra.compose(config_name="vqgan")
        cfg_transformer = hydra.compose(config_name="transformer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    data_vqgan = Dataset_vqgan(cfg_vqgan,cfg_transformer,device)
    data_vqgan.print_parameters()

    print("Total number of training data samples is {}".format(len(data_vqgan)))
    train_data_loader = DataLoader(data_vqgan, batch_size=16, shuffle=True)

    for i, data in enumerate(train_data_loader):
        if i == 3:
            break
        print(f'Batch number: {i}')
        tokens = data[0]
        cond = data[1]
    
        print(f'Batch size: {tokens.shape[0]}')
        print(f'Tokens shape: {tokens.shape}')
        print(f'conditional vector shape: {cond.shape}')


