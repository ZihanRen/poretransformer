import torch
from lpu3dnet.frame import vqgan
from lpu3dnet.modules import discriminator
from lpu3dnet.train import dataset_vqgan
import os
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lpu3dnet import init_yaml
import pickle

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
latent_dim = 256

disc_factor = 0.2 # TODO: change this to a better value
threshold = 1000 # TODO: change this to a better value
epochs = 100 # TODO: change this to a better value
batch_size = 20 # TODO: change this to a better value

class TrainVQGAN:
    def __init__(self,device = device, learning_rate=5e-4,beta1=0.9,beta2=0.999,disc_factor=disc_factor):
        self.disc_factor = disc_factor
        self.device = device
        self.vqgan = vqgan.VQGAN().to(device=self.device)
        self.discriminator = discriminator.Discriminator().to(device=self.device)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.opt_vq, self.opt_disc = self.configure_optimizers()

        self.training_losses = {}
        # it's like the generator loss
        self.training_losses['vq_loss'] = []
        self.training_losses['rec_loss'] = []
        self.training_losses['d_loss'] = []
        self.training_losses['g_loss'] = []
        self.training_losses['total_loss'] = []
        self.training_losses['total_loss_per_epoch'] = []
        

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(self.beta1, self.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(self.beta1, self.beta2))

        return opt_vq, opt_disc

    def prepare_training(self,idx):

        def remove_all_files_in_directory(directory):
            """Removes all files in the specified directory."""
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # initialize training folders
        self.PATH =  os.path.join( init_yaml.PATH['checkpoints'],f'ex{idx}' )
        os.makedirs(self.PATH, exist_ok=True)
        # clear all previous files in this folder
        remove_all_files_in_directory(self.PATH)
        

    def train(self):

        print("Training VQGAN:")

        train_dataset = dataset_vqgan.Dataset_vqgan()
        train_data_loader = DataLoader(train_dataset,batch_size=20,shuffle=True)
        steps_per_epoch = len(train_data_loader)

        for epoch in range(epochs):
            trian_loss_per_epoch = 0
            with tqdm(
                train_data_loader,
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{epochs}"
                ) as pbar:

                for i, imgs in enumerate(pbar):
                    imgs = imgs.to(device=self.device)

                    # get decoded image and embedding loss
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    # reconstruction loss #TODO: check if this is correct
                    rec_loss = F.mse_loss(imgs,decoded_images)

                    # embedding loss + discriminator loss (loss for not fooling discriminator)
                    vq_loss = + q_loss + rec_loss

                    
                    self.opt_vq.zero_grad()
                    vq_loss.backward()

                    self.opt_vq.step()

                    # calculate training loss and print out
                    train_loss = vq_loss
                    pbar.set_description(f"Step: {i+1}/{steps_per_epoch}")
                    pbar.set_postfix(Loss=train_loss.item())
                    trian_loss_per_epoch += train_loss.item()

                    # save losses per step
                    self.training_losses['vq_loss'].append(vq_loss.item())
                    self.training_losses['rec_loss'].append(rec_loss.item())
                    self.training_losses['total_loss'].append(train_loss.item())
            
            # save progress per epoch
            self.training_losses['total_loss_per_epoch'].append(trian_loss_per_epoch/steps_per_epoch)

            model_path = os.path.join(self.PATH,f"vqgan_epoch_{epoch+1}.pth")
            loss_path = os.path.join(self.PATH,f"training_losses_epoch_{epoch+1}.pkl")

            
            torch.save(self.vqgan.state_dict(), model_path)
            save_to_pkl(self.training_losses, loss_path)

if __name__ == "__main__":
    experiment_idx = 1
    train = TrainVQGAN()
    train.prepare_training(experiment_idx)
    train.train()