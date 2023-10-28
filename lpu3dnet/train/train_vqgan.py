
import torch
from lpu3dnet.frame import vqgan
from lpu3dnet.modules import discriminator
from lpu3dnet.train import dataset_vqgan
import os
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
import time
from datetime import timedelta
from lpu3dnet.init_yaml import config_vqgan as config
import argparse
import shutil


parser = argparse.ArgumentParser(description='Experiment setup')
parser.add_argument("--ex",type=int, required=True, help="set up experiment idx")
args = parser.parse_args()


def save_to_pkl(my_list, file_path):
    """
    Save a Python list to a .pkl file.

    Parameters:
    - my_list (list): The list you want to save.
    - file_path (str): The path where you want to save the .pkl file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(my_list, f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainVQGAN:
    def __init__(
            self,
            device = device,
            lr_vqgan=config['train']['lr_vqgan'],
            lr_disc=config['train']['lr_disc'],
            beta1=config['train']['beta1'],
            beta2=config['train']['beta2'],
            disc_factor=config['train']['disc_factor'],
            threshold=config['train']['disc_start'],
            experiment_idx = args.ex,
            epochs = config['train']['epochs'],
            w_embed = config['train']['w_embed'],
            codebook_weight_increase_per_epoch=config['train']['codebook_weight_increase_per_epoch']
            ):
        
        self.disc_factor = disc_factor
        self.device = device
        self.vqgan = vqgan.VQGAN().to(device=self.device)
        self.discriminator = discriminator.Discriminator().to(device=self.device)
        self.lr_vqgan = lr_vqgan
        self.lr_disc = lr_disc
        self.beta1 = beta1
        self.beta2 = beta2
        self.experiment_idx = experiment_idx
        self.epochs = epochs
        self.threshold = threshold
        self.w_embed = w_embed
        self.codebook_weight_increase_per_epoch = codebook_weight_increase_per_epoch
        
        self.opt_vq, self.opt_disc = self.configure_optimizers()

        self.training_losses = {}

        

    def configure_optimizers(self):
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=self.lr_vqgan, eps=1e-08, betas=(self.beta1, self.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self.lr_disc, eps=1e-08, betas=(self.beta1, self.beta2))

        return opt_vq, opt_disc

    def prepare_training(self):

        # initialize object to track training losses
        self.training_losses['q_loss'] = []
        self.training_losses['rec_loss'] = []
        self.training_losses['d_loss'] = []
        self.training_losses['g_loss'] = []
        self.training_losses['total_loss'] = []
        self.training_losses['total_loss_per_epoch'] = []
        self.training_losses['time'] = []

        def remove_all_files_in_directory(directory):
            """Removes all files in the specified directory."""
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # initialize training folders
        self.PATH =  os.path.join( config['checkpoints']['PATH'],f'ex{self.experiment_idx}' )
        os.makedirs(self.PATH, exist_ok=True)
        # clear all previous files in this folder
        remove_all_files_in_directory(self.PATH)
        

    def train(self):

        print("Training VQGAN:")
        start_time = time.time()

        train_dataset = dataset_vqgan.Dataset_vqgan()
        train_data_loader = DataLoader(train_dataset,batch_size=config['train']['epochs'],shuffle=True)
        steps_per_epoch = len(train_data_loader)

        for epoch in range(self.epochs):
            

            trian_loss_per_epoch = 0
            with tqdm(
                train_data_loader,
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.epochs}"
                ) as pbar:

                for i, imgs in enumerate(pbar):
                    imgs = imgs.to(device=self.device)

                    # get decoded image and embedding loss
                    decoded_images, _, q_loss = self.vqgan(imgs)
                    
                    # get discriminator values
                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    # calculate loss parameter for GAN
                    disc_factor = self.vqgan.adopt_weight(
                        self.disc_factor, 
                        epoch*steps_per_epoch+i, 
                        threshold=self.threshold)

                    # reconstruction loss #TODO: check if this is correct
                    rec_loss = F.mse_loss(imgs,decoded_images)

                    # generator loss
                    g_loss = (-torch.mean(disc_fake)) * disc_factor

                    # embedding loss + discriminator loss (loss for not fooling discriminator)
                    vq_loss = self.w_embed * q_loss + rec_loss + g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))

                    # discriminator loss - loss for being fooled by generator
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    
                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    # calculate training loss and print out
                    train_loss = vq_loss + gan_loss
                    pbar.set_description(f"Step: {i+1}/{steps_per_epoch}")
                    pbar.set_postfix(Loss=train_loss.item())
                    trian_loss_per_epoch += train_loss.item()

                    # save losses per step
                    self.training_losses['q_loss'].append(q_loss.item())
                    self.training_losses['rec_loss'].append(rec_loss.item())
                    self.training_losses['d_loss'].append(gan_loss.item())
                    self.training_losses['g_loss'].append(g_loss.item())
                    self.training_losses['total_loss'].append(train_loss.item())
            
            # save progress per epoch
            end_time = time.time()
            duration = timedelta(seconds=end_time-start_time)
            print(f'Training took {duration} seconds in total for now')
            self.training_losses['total_loss_per_epoch'].append(trian_loss_per_epoch/steps_per_epoch)
            self.training_losses['time'].append(duration)

            model_path = os.path.join(self.PATH,f"vqgan_epoch_{epoch+1}.pth")
            loss_path = os.path.join(self.PATH,f"training_losses_epoch_{epoch+1}.pkl")

            
            torch.save(self.vqgan.state_dict(), model_path)
            save_to_pkl(self.training_losses, loss_path)

    def train_nogan(self):
        

        print("Training VQGAN:")
        start_time = time.time()

        train_dataset = dataset_vqgan.Dataset_vqgan()
        train_data_loader = DataLoader(train_dataset,batch_size=20,shuffle=True)
        steps_per_epoch = len(train_data_loader)
        weight_increase_per_step = self.codebook_weight_increase_per_epoch/steps_per_epoch

        for epoch in range(self.epochs):
            

            trian_loss_per_epoch = 0
            with tqdm(
                train_data_loader,
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.epochs}"
                ) as pbar:

                for i, imgs in enumerate(pbar):
                    imgs = imgs.to(device=self.device)

                    # get decoded image and embedding loss
                    decoded_images, _, q_loss = self.vqgan(imgs)
                    q_loss = q_loss or 0

                    # reconstruction loss #TODO: check if this is correct
                    rec_loss = F.mse_loss(imgs,decoded_images)

                    # delay the update of codebook at the beginning
                    weight_q_loss = min(weight_increase_per_step * (epoch*steps_per_epoch+i),1)
                    q_loss = weight_q_loss * q_loss

                    vq_loss =  q_loss + rec_loss
                    
                    self.opt_vq.zero_grad()
                    vq_loss.backward()
                    self.opt_vq.step()

                    # calculate training loss and print out
                    train_loss = vq_loss
                    pbar.set_description(f"Epoch {epoch} at Step: {i+1}/{steps_per_epoch}")
                    pbar.set_postfix(Loss=train_loss.item())
                    trian_loss_per_epoch += train_loss.item()

                    # save losses per step
                    if q_loss != 0:
                        self.training_losses['q_loss'].append((q_loss).item())
                    
                    self.training_losses['rec_loss'].append(rec_loss.item())
                    self.training_losses['total_loss'].append(train_loss.item())
            
            # save progress per epoch
            end_time = time.time()
            duration = timedelta(seconds=end_time-start_time)
            print(f'Training took {duration} seconds in total for now')
            self.training_losses['total_loss_per_epoch'].append(trian_loss_per_epoch/steps_per_epoch)
            self.training_losses['time'].append(duration)
            
            model_path = os.path.join(self.PATH,f"vqgan_epoch_{epoch+1}.pth")
            loss_path = os.path.join(self.PATH,f"training_losses_epoch_{epoch+1}.pkl")
            
            torch.save(self.vqgan.state_dict(), model_path)
            save_to_pkl(self.training_losses, loss_path)



if __name__ == "__main__":

    train = TrainVQGAN()
    train.prepare_training()
    
    # backup configuration parameters to checkpoint folder
    shutil.copy(
        '../lpu_vqgan.yaml',
        os.path.join(train.PATH,f'lpu_vqgan_{train.experiment_idx}.yaml')
        )
    
    train.train_nogan()
