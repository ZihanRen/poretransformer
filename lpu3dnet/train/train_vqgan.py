
import torch
from lpu3dnet.frame import vqgan
from lpu3dnet.modules import discriminator
from lpu3dnet.train import dataset_vqgan
from lpu3dnet.modules.components import gradient_penalty
import os
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
import time
from datetime import timedelta
import hydra


def save_to_pkl(my_list, file_path):
    """
    Save a Python list to a .pkl file.

    Parameters:
    - my_list (list): The list you want to save.
    - file_path (str): The path where you want to save the .pkl file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(my_list, f)


class TrainVQGAN:
    def __init__(
            self,
            cfg_dataset,
            cfg_vqgan,
            device,
            ):
        # use self.cfg mainly on modules
        self.cfg_dataset = cfg_dataset
        self.cfg_vqgan = cfg_vqgan
        # use self.cfg mainly on modules
        self.disc_factor = cfg_vqgan.train.disc_factor
        self.device = device
        self.load_model = cfg_vqgan.train.load_model
        self.pretrained_model_epoch = cfg_vqgan.train.pretrained_model_epoch

        if self.load_model:
            root_path = os.path.join(cfg_dataset.checkpoints.PATH, cfg_dataset.experiment)
            self.vqgan = vqgan.VQGAN(cfg_vqgan).to(device=self.device)
            PATH_model = os.path.join(root_path,f'vqgan_epoch_{self.pretrained_model_epoch}.pth')
            self.vqgan.load_state_dict(
                torch.load(
                        PATH_model,
                        map_location=torch.device(self.device)
                        )
                )
        else:
            self.vqgan = vqgan.VQGAN(cfg_vqgan).to(device=self.device)
        



        self.lr_vqgan = cfg_vqgan.train.lr_vqgan
        if self.load_model:
            
            self.lr_vqgan = cfg_vqgan.train.lr_vqgan/1.5
        self.lr_disc = cfg_vqgan.train.lr_disc
        self.beta1 = cfg_vqgan.train.beta1
        self.beta2 = cfg_vqgan.train.beta2
        self.epochs = cfg_vqgan.train.epochs
        self.threshold = cfg_vqgan.train.disc_start
        self.codebook_weight_increase_per_epoch = cfg_vqgan.train.codebook_weight_increase_per_epoch
        self.drop_last = cfg_vqgan.train.drop_last
        self.batch_size = cfg_vqgan.train.batch_size
        self.checkpoints_path = cfg_dataset.checkpoints.PATH
        self.max_weight_q_loss = cfg_vqgan.train.max_weight_q_loss
        
        self.opt_vq = self.configure_optimizers()

        self.training_losses = {}


    def get_path(self,epoch):
        model_path = os.path.join(
                    self.PATH,
                    f"vqgan_epoch_{epoch}.pth"
        )
        return model_path

    def configure_optimizers(self):
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=self.lr_vqgan, eps=1e-08, betas=(self.beta1, self.beta2)
        )

        return opt_vq


    def prepare_training(self,removed_files=False):

            # initialize object to track training losses
            self.training_losses['q_loss'] = []
            self.training_losses['rec_loss'] = []
            self.training_losses['d_loss'] = []
            self.training_losses['g_loss'] = []
            self.training_losses['total_loss'] = []
            self.training_losses['total_loss_per_epoch'] = []
            self.training_losses['perplexity'] = []
            self.training_losses['time'] = []
            self.training_losses['gp'] = []
            self.training_losses['l2_loss'] = []

            def remove_all_files_in_directory(directory):
                """Removes all files in the specified directory."""
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            # initialize training folders
            self.PATH =  os.path.join(
                                    self.checkpoints_path,
                                    self.cfg_dataset.experiment
                                    )
            
            os.makedirs(self.PATH, 
                        exist_ok=True)
            if removed_files:
                # clear all previous files in this folder starting from new models
                remove_all_files_in_directory(self.PATH)
        

    def train(self):
        
        print("Training VQGAN:")
        self.vqgan.save_checkpoint(self.get_path(-1))
        start_time = time.time()

        train_dataset = dataset_vqgan.Dataset_vqgan(self.cfg_dataset)
        
        train_data_loader = DataLoader(
                            train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=self.drop_last
                            )
        
        steps_per_epoch = len(train_data_loader)
        weight_increase_per_step = self.codebook_weight_increase_per_epoch/steps_per_epoch

        for epoch in range(self.epochs):

            if self.load_model:
                epoch += self.pretrained_model_epoch
            
            trian_loss_per_epoch = 0
            
            with tqdm(
                train_data_loader,
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.epochs}"
                ) as pbar:

                for i, imgs in enumerate(pbar):
                    imgs = imgs.to(device=self.device)

                    # get decoded image and embedding loss
                    decoded_images, codebook_info, q_loss = self.vqgan(imgs)
                    perplexity = codebook_info[0]

                    q_loss = q_loss or 0 # if q_loss is None, set it to 0

                    # reconstruction loss #TODO: check if this is correct
                    rec_loss = F.mse_loss(imgs,decoded_images)

                    # delay the update of codebook at the beginning
                    weight_q_loss = min(weight_increase_per_step * (epoch*steps_per_epoch+i),self.max_weight_q_loss)
                    q_loss = weight_q_loss * q_loss

                    # # add L2 regularization to the codebook
                    # l2_reg = self.vqgan.l2_reg()
                    # l2_loss = self.cfg.train.l2_reg_weight * l2_reg
                    vq_loss =  q_loss + rec_loss
                    
                    self.opt_vq.zero_grad()
                    vq_loss.backward()
                    self.opt_vq.step()

                    # calculate training loss and print out
                    train_loss = vq_loss.clone()
                    pbar.set_description(f"Epoch {epoch} at Step: {i+1}/{steps_per_epoch}")
                    pbar.set_postfix(Loss=train_loss.item())
                    trian_loss_per_epoch += train_loss.item()

                    # check if tensor
                    if isinstance(q_loss, torch.Tensor):
                        self.training_losses['q_loss'].append(q_loss.item())
                    else:
                        self.training_losses['q_loss'].append(q_loss)
                    
                    self.training_losses['rec_loss'].append(rec_loss.item())
                    # self.training_losses['l2_loss'].append(l2_loss.item())
                    self.training_losses['total_loss'].append(train_loss.item())
                    if not self.cfg_vqgan.architecture.codebook.autoencoder:
                        self.training_losses['perplexity'].append(perplexity.item())
            
            # save progress per epoch
            if epoch % 5 == 0:
                end_time = time.time()
                duration = timedelta(seconds=end_time-start_time)
                print(f'Training took {duration} seconds in total for now')
                self.training_losses['total_loss_per_epoch'].append(trian_loss_per_epoch/steps_per_epoch)
                self.training_losses['time'].append(duration)
                
                loss_path = os.path.join(self.PATH,f"training_losses_epoch_{epoch}.pkl")
                self.vqgan.save_checkpoint(self.get_path(epoch))
                save_to_pkl(self.training_losses, loss_path)




if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = 7

    with hydra.initialize(config_path=f"../config/ex{exp}"):
        cfg_vqgan = hydra.compose(config_name="vqgan")
        cfg_dataset = hydra.compose(config_name="dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = TrainVQGAN(device=device,cfg_vqgan=cfg_vqgan,cfg_dataset=cfg_dataset)
    train.prepare_training(removed_files=True)
    train.train()
