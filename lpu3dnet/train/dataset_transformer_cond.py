
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tifffile import imread
import hydra
import porespy as ps

class Dataset_vqgan(Dataset):
    def __init__(
        self,
        cfg,
    ):
        # cfg: configuration of VQGAN
        self.root_PATH = cfg.data.PATH.sub_vol_large
        self.ct_idx = cfg.data.ct_idx
        self.transform = None

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

    def __len__(self):
        return len(self.img_PATH)

    def __getitem__(self, index):
        # load images
        image = Dataset_vqgan.tif_to_np(self.img_PATH[index])

        # segment images into 64^3 cubic voxels in 2*2*2 spatial grid
        segmented_images = {}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    subset_image = image[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64]
                    segmented_images[(i, j, k)] = {
                        'image': subset_image,
                        'porosity': ps.metrics.porosity(subset_image)
                    }

        return segmented_images

    def img_name(self):
        return os.listdir(self.img_PATH)

    def print_parameters(self):
        print("training CT index is {}".format(self.ct_idx))
        print("training data root path is {}".format(self.root_PATH))




if __name__ == "__main__":
    experiment_idx = 7

    @hydra.main(
        config_path=f"/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/config/ex{experiment_idx}",
        config_name="vqgan",
        version_base='1.2')
    def main(cfg):
        data_vqgan = Dataset_vqgan(cfg)
        data_vqgan.print_parameters()

        print("Total number of training data samples is {}".format(len(data_vqgan)))
        train_data_loader = DataLoader(data_vqgan, batch_size=16, shuffle=True)

        for i, segmented_images in enumerate(train_data_loader):
            if i == 3:
                break
            print(f'Batch number: {i}')
            for (x, y, z), data in segmented_images.items():
                print(f'Spatial coordinates: ({x}, {y}, {z})')
                print(f'Subset image shape: {data["image"].shape}')
                print(f'Porosity: {data["porosity"]}')
            print(2 * '\n')

    main()