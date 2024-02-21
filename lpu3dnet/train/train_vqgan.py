
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
            cfg,
            device,
            ):
        # use self.cfg mainly on modules
        self.cfg = cfg
        self.disc_factor = cfg.train.disc_factor
        self.device = device
        self.load_model = cfg.train.load_model
        self.pretrained_model_epoch = cfg.train.pretrained_model_epoch

        if self.load_model:
            root_path = os.path.join(cfg.checkpoints.PATH, cfg.experiment)
            self.vqgan = vqgan.VQGAN(cfg).to(device=self.device)
            PATH_model = os.path.join(root_path,f'vqgan_epoch_{self.pretrained_model_epoch}.pth')
            self.vqgan.load_state_dict(
                torch.load(
                        PATH_model,
                        map_location=torch.device(self.device)
                        )
                )
        else:
            self.vqgan = vqgan.VQGAN(self.cfg).to(device=self.device)
        
        self.discriminator = discriminator.Discriminator(
                        image_channels = self.cfg.architecture.discriminator.img_channels,
                        num_filters_last = self.cfg.architecture.discriminator.init_filters_num,
                        n_layers = self.cfg.architecture.discriminator.num_layers
                        ).to(device=self.device)

        self.lr_vqgan = cfg.train.lr_vqgan
        if self.load_model:
            self.lr_vqgan = cfg.train.lr_vqgan/1.5
        self.lr_disc = cfg.train.lr_disc
        self.beta1 = cfg.train.beta1
        self.beta2 = cfg.train.beta2
        self.epochs = cfg.train.epochs
        self.threshold = cfg.train.disc_start
        self.w_embed = cfg.train.w_embed
        self.codebook_weight_increase_per_epoch = cfg.train.codebook_weight_increase_per_epoch
        self.drop_last = cfg.train.drop_last
        self.batch_size = cfg.train.batch_size
        self.checkpoints_path = cfg.checkpoints.PATH
        self.g_lambda = cfg.train.g_lambda
        self.max_weight_q_loss = cfg.train.max_weight_q_loss
        
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

        def remove_all_files_in_directory(directory):
            """Removes all files in the specified directory."""
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # initialize training folders
        self.PATH =  os.path.join(
                                  self.checkpoints_path,
                                  self.cfg.experiment
                                  )
        
        os.makedirs(self.PATH, 
                    exist_ok=True)
        if removed_files:
            # clear all previous files in this folder starting from new models
            remove_all_files_in_directory(self.PATH)
        

    def train(self):

        print("Training VQGAN:")
        start_time = time.time()

        train_dataset = dataset_vqgan.Dataset_vqgan(self.cfg)
        
        train_data_loader = DataLoader(
                            train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=self.drop_last
                            )
        
        steps_per_epoch = len(train_data_loader)
        weight_increase_per_step = self.codebook_weight_increase_per_epoch/steps_per_epoch

        for epoch in range(self.epochs):
            # update new epoch here
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
                    # calculate gradident penalty
                    # get discriminator values
                    # calculate loss parameter for GAN
                    disc_factor = self.vqgan.adopt_weight(
                        self.disc_factor, 
                        epoch*steps_per_epoch+i, 
                        threshold=self.threshold)

                    ############################### Train Discriminator ##################################
                    self.discriminator.zero_grad()
                    gp = gradient_penalty(
                        self.discriminator,
                        imgs,
                        decoded_images,
                        device=self.device)
                    gp = self.g_lambda * gp
                    # you need to backpropagate gradident penalty loss twice
                    gp.backward(retain_graph=True)

                    disc_fake = self.discriminator(decoded_images.detach())
                    disc_real = self.discriminator(imgs)
                    d_loss = -(torch.mean(disc_real) - torch.mean(disc_fake)) 
                    d_loss = disc_factor * d_loss
                    d_loss.backward()
                    self.opt_disc.step()

                    ############################### Train Generator ##################################
                    # reconstruction loss #TODO: check better options
                    self.vqgan.zero_grad()
                    disc_fake = self.discriminator(decoded_images)
                    rec_loss = F.mse_loss(imgs,decoded_images)
                    g_loss = (-torch.mean(disc_fake)) * disc_factor
                    # gradually increase weight of embedding loss
                    weight_q_loss = min(weight_increase_per_step * (epoch*steps_per_epoch+i),1)
                    q_loss = weight_q_loss * q_loss
                    # embedding loss + discriminator loss (loss for not fooling discriminator)
                    vq_loss = q_loss + rec_loss + g_loss
                    vq_loss.backward()
                    self.opt_vq.step()


                    ############################## Tracking Losses ####################################
                    # calculate training loss and print out
                    train_loss = vq_loss + d_loss

                    pbar.set_description(f"Epoch {epoch} at Step: {i+1}/{steps_per_epoch}")
                    pbar.set_postfix(Loss=train_loss.item())

                    trian_loss_per_epoch += train_loss.item()

                    # check if tensor
                    if isinstance(q_loss, torch.Tensor):
                        self.training_losses['q_loss'].append(q_loss.item())
                    else:
                        self.training_losses['q_loss'].append(q_loss)

                    # save losses per step
                    self.training_losses['rec_loss'].append(rec_loss.item())
                    self.training_losses['gp'].append(gp.item())

                    self.training_losses['d_loss'].append(d_loss.item())
                    self.training_losses['g_loss'].append(g_loss.item())
                    self.training_losses['total_loss'].append(train_loss.item())
                    self.training_losses['perplexity'].append(perplexity.item())
            
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
        self.vqgan.save_checkpoint(-1)
        start_time = time.time()

        train_dataset = dataset_vqgan.Dataset_vqgan(self.cfg)
        
        train_data_loader = DataLoader(
                            train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=self.drop_last
                            )
        
        steps_per_epoch = len(train_data_loader)
        weight_increase_per_step = self.codebook_weight_increase_per_epoch/steps_per_epoch

        for epoch in range(self.epochs):
            # whether freeze or not
            # if epoch == 10:
            #     self.vqgan.freeze_decoder()
            #     self.vqgan.freeze_encoder()
            
            # if epoch == 10:
            #     self.vqgan.unfreeze_encoder()

            # if epoch == 20:
            #     self.vqgan.unfreeze_decoder()

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
                    if not self.cfg.architecture.codebook.autoencoder:
                        perplexity = codebook_info[0]

                    q_loss = q_loss or 0 # if q_loss is None, set it to 0

                    # reconstruction loss #TODO: check if this is correct
                    rec_loss = F.mse_loss(imgs,decoded_images)

                    # delay the update of codebook at the beginning
                    weight_q_loss = min(weight_increase_per_step * (epoch*steps_per_epoch+i),self.max_weight_q_loss)
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

                    # check if tensor
                    if isinstance(q_loss, torch.Tensor):
                        self.training_losses['q_loss'].append(q_loss.item())
                    else:
                        self.training_losses['q_loss'].append(q_loss)
                    
                    self.training_losses['rec_loss'].append(rec_loss.item())
                    self.training_losses['total_loss'].append(train_loss.item())
                    if not self.cfg.architecture.codebook.autoencoder:
                        self.training_losses['perplexity'].append(perplexity.item())
            
            # save progress per epoch
            if epoch % 5 == 0:
                end_time = time.time()
                duration = timedelta(seconds=end_time-start_time)
                print(f'Training took {duration} seconds in total for now')
                self.training_losses['total_loss_per_epoch'].append(trian_loss_per_epoch/steps_per_epoch)
                self.training_losses['time'].append(duration)
                
                loss_path = os.path.join(self.PATH,f"training_losses_epoch_{epoch}.pkl")
                self.vqgan.save_checkpoint(epoch)
                save_to_pkl(self.training_losses, loss_path)



if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = 7

    @hydra.main(
        config_path=f"/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/config/ex{exp}",
        config_name="vqgan",
        version_base='1.2')
    def main(cfg):
        train = TrainVQGAN(device=device,cfg=cfg)
        train.prepare_training(removed_files=True)
        train.train_nogan()

    main()
