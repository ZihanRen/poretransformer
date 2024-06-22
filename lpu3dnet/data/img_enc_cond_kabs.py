import os
import torch
import numpy as np
from tifffile import imread
import porespy as ps
from lpu3dnet.frame import vqgan
import hydra
import shutil
from cpgan.ooppnm import pnm_sim_old


from cpgan.ooppnm import img_process
img_prc = img_process.Image_process()

def get_kabs(img):
    data_pnm = pnm_sim_old.Pnm_sim(im=img)
    data_pnm.network_extract()
    if data_pnm.error == 1:
        print('Error in network extraction')
        return None
        # raise ValueError('Error in network extraction')
        
    data_pnm.init_physics()
    data_pnm.get_absolute_perm()

    return data_pnm.data_tmp['kabs']



class ImageTokensGenerator:
    def __init__(self, cfg_vqgan, cfg_dataset, device,vqgan_epoch):
        self.device = device
        self.root_PATH = cfg_dataset.PATH.sub_vol_large
        self.ct_idx = cfg_dataset.ct_idx
        self.img_tokens_path = cfg_dataset.PATH.image_tokens
        self.img_cond_path = cfg_dataset.PATH.image_tokens_cond

        self.vqgan = vqgan.VQGAN(cfg_vqgan).to(device)
        self.pretrained_vqgan_epoch = vqgan_epoch
        self.vqgan_path = os.path.join(
            cfg_dataset.checkpoints.PATH,
            cfg_dataset.experiment
            )
        
        self.vqgan.load_state_dict(
            torch.load(
                os.path.join(self.vqgan_path, f'vqgan_epoch_{self.pretrained_vqgan_epoch}.pth'),
                map_location=torch.device(self.device)
                )
        )
        self.vqgan.to(self.device)
        self.vqgan.eval()

    def gen_img_from_z(self,z):
        with torch.no_grad():
            img = self.vqgan.decode(z)
            img = img_prc.clean_img(img)
        return img
    
    def empty_folders(self):
        def delete_directory_contents(directory_path):
            if os.path.exists(directory_path) and os.path.isdir(directory_path):
                for filename in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                            print(f"Deleted file: {file_path}")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            print(f"Deleted directory: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
            else:
                print(f"'{directory_path}' is not a valid directory")

        # Empty the specific directories
        delete_directory_contents(self.img_tokens_path)
        delete_directory_contents(self.img_cond_path)


    @staticmethod
    def tif_to_np(f_name):
        img = imread(f_name)
        img = img.astype('float32') / 255
        return img > 0.5

    def get_img_tokens(self, img):
        with torch.no_grad():
            img_tokens = self.vqgan.gen_img_tokens(img)
        return img_tokens

    def get_patch_tokens_cond(self, img):
        # segment images into 64^3 cubic voxels in 2*2*2 spatial grid
        tokens_patch_list = []
        phi_list = []
        i_list = []
        j_list = []
        k_list = []
        kabs_list = []

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    subset_image = img[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64]
                    subset_image = np.expand_dims(subset_image, axis=(0,1))
                    subset_image = torch.from_numpy(subset_image).float().to(self.device)
                    with torch.no_grad():
                        subset_tokens = self.get_img_tokens(subset_image)
                        z_current = self.vqgan.tokens_to_z(subset_tokens,total_features_vec_num=64)
                        img_gen = self.gen_img_from_z(z_current)
                        phi_list.append(ps.metrics.porosity(img_gen))

                        try:
                            kabs_tmp = get_kabs(img_gen[0])
                            if kabs_tmp is None:
                                return None, None, True
                            kabs_list.append(kabs_tmp)
                        except:
                            print('Error in calculating kabs')
                            ignore = True
                            return None, None, True


                    tokens_patch_list.append(subset_tokens)
                    i_list.append(i)
                    j_list.append(j)
                    k_list.append(k)

        tokens_all = torch.cat(tokens_patch_list, dim=0).to(self.device)
        phi_tensor = torch.tensor(phi_list).view(-1,1).to(self.device)
        i_tensor = torch.tensor(i_list).view(-1,1).to(self.device)
        j_tensor = torch.tensor(j_list).view(-1,1).to(self.device)
        k_tensor = torch.tensor(k_list).view(-1,1).to(self.device)
        kabs_tensor = torch.tensor(kabs_list).view(-1,1).to(self.device)

        # concatenate all the conditional information
        cond = torch.cat([phi_tensor,kabs_tensor,i_tensor,j_tensor,k_tensor], dim=1)
        return tokens_all, cond, False

    def generate_and_save_tokens_cond(self):
        for ct_idx in [0,1,2,3,4,5]:
            print(f'generating tokens and conditional vector for ct_{ct_idx}')
            ct_folder = os.path.join(self.root_PATH, f'ct_{ct_idx}')
            tokens_path = os.path.join(self.img_tokens_path, f'ct_{ct_idx}')
            cond_path = os.path.join(self.img_cond_path, f'ct_{ct_idx}')

            os.makedirs(tokens_path, exist_ok=True)
            os.makedirs(cond_path, exist_ok=True)

            for idx, img_name in enumerate(os.listdir(ct_folder)):
                base_name = img_name.split('.')[0]
                img_path = os.path.join(ct_folder, img_name)
                image = self.tif_to_np(img_path)

                # Process your image here to generate `tokens_all` and `cond`
                tokens_all, cond, ignore = self.get_patch_tokens_cond(image)
                
                # handle exception
                if ignore:
                    continue

                tokens_all = tokens_all.cpu()
                cond = cond.cpu()
                # Save tokens and conditional vectors
                torch.save(tokens_all, os.path.join(tokens_path, f'tokens_{base_name}.pt'))
                torch.save(cond, os.path.join(cond_path, f'cond_{base_name}.pt'))



import time

# Example Usage
if __name__ == "__main__":
    start_time = time.time()
    # Assuming cfg_vqgan and cfg_dataset are defined as in your script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_idx = 13
    with hydra.initialize(config_path=f"../config/ex{experiment_idx}"):
        cfg_vqgan = hydra.compose(config_name="vqgan")
        cfg_dataset = hydra.compose(config_name="dataset")
    generator = ImageTokensGenerator(cfg_vqgan,cfg_dataset, device,vqgan_epoch=25) # TODO should replace 25 with 10
    generator.empty_folders()
    generator.generate_and_save_tokens_cond()
    end_time = time.time()  # Record end time
    total_time_seconds = end_time - start_time  # Calculate total time taken
    total_time_hours = total_time_seconds / 3600  # Convert seconds to hours    
    print(f"Total time taken to generate dataset: {total_time_hours:.2f} hours")

