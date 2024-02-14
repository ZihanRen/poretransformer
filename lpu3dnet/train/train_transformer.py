#%%
import torch
from lpu3dnet.frame import vqgan
from lpu3dnet.frame import transformer
from lpu3dnet.train import dataset_vqgan
import os
from tqdm import tqdm
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


class TrainTransformer:
    def __init__(
            self,
            cfg_vqgan,
            cfg_transformer,
            device,
            ):
        # use self.cfg mainly on modules - vqgan & transformer
        self.cfg_vqgan = cfg_vqgan
        self.cfg_transformer = cfg_transformer

        # decompose transformer config
        self.pretrained_transformer_epoch = self.cfg_transformer.train.pretrained_transformer_epoch
        self.pretrained_vqgan_epoch = self.cfg_transformer.train.pretrained_vqgan_epoch
        self.root_path = os.path.join(self.cfg_transformer.checkpoints.PATH, self.cfg_transformer.experiment)
        self.transformer_path = os.path.join(self.root_path, 'transformer')

        self.device = device


        # load pretrained VQGAN model
        self.vqgan = vqgan.VQGAN(self.cfg_vqgan).to(device=self.device)
        self.vqgan.load_state_dict(
            torch.load(
                os.path.join(self.root_path, f'vqgan_epoch_{self.pretrained_vqgan_epoch}.pth'),
                map_location=torch.device(self.device)
                )
            )
        # freeze VQGAN parameters
        self.vqgan.eval()

        # model initialization - whether to load model or initialzie the model
        if self.cfg_transformer.train.load_model:
            self.transformer = transformer.Transformer(self.cfg_transformer).to(device=self.device)

            PATH_model = os.path.join(
                self.transformer_path,
                f'transformer_epoch_{self.pretrained_model_epoch}.pth'
                )
            
            self.transformer.load_state_dict(
                torch.load(
                        PATH_model,
                        map_location=torch.device(self.device)
                        )
                )
        else:
            self.transformer = transformer.Transformer(self.cfg_transformer).to(device=self.device)
        
        # initializew optimizer
        self.opt = self.transformer.configure_optimizers(device=self.device)

        # intialize training losses tracking
        self.training_losses = {
            'total_loss': [],
            'total_loss_per_epoch': [],
            'time': []
        }


    def prepare_training(self,clear_all=False):

        def remove_all_files_in_directory(directory):
            """Removes all files in the specified directory."""
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)


        print(
            "Training history of transformer will be saved in: ", 
              self.transformer_path
              )
        
        os.makedirs(self.transformer_path, 
                    exist_ok=True)
        
        if clear_all:
            # clear all previous files in this folder starting from new models
            remove_all_files_in_directory(self.transformer_path)
        

    def train(self):
        
        sos_tokens = self.cfg_transformer.train.sos_token


        print("Training transformer:")
        start_time = time.time()
        # using the same dataloader as VQGAN training process
        train_dataset = dataset_vqgan.Dataset_vqgan(self.cfg_vqgan)

        
        train_data_loader = DataLoader(
                            train_dataset,
                            batch_size=self.cfg_transformer.train.batch_size, #TODO: change to cfg
                            shuffle=True,
                            drop_last=False
                            )
        
        steps_per_epoch = len(train_data_loader)

        for epoch in range(self.cfg_transformer.train.epochs):

            trian_loss_per_epoch = 0

            with tqdm(
                train_data_loader,
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.cfg_transformer.train.epochs}"
                ) as pbar:

                for i, imgs in enumerate(pbar):
                    
                    imgs = imgs.to(device=self.device)
                    # get sampled image tokens
                    with torch.no_grad():
                        img_tokens = self.vqgan.gen_img_tokens(imgs)
                    
                    # train transformer
                    sos_tokens = torch.ones(img_tokens.shape[0], 1) * sos_tokens
                    sos_tokens = sos_tokens.long().to(self.device)

                    mask = torch.bernoulli(
                        self.cfg_transformer.train.p_keep * torch.ones(
                        img_tokens.shape, device=self.device)
                        )
                    
                    mask = mask.round().to(dtype=torch.int64)

                    random_indices = torch.randint_like(
                        img_tokens,
                        self.cfg_transformer.architecture.vocab_size
                                                        )
                    
                    perturbed_indices = mask * img_tokens + (1 - mask) * random_indices
                    perturbed_indices = torch.cat((sos_tokens, perturbed_indices), dim=1)

                    target = img_tokens
                    print(target.shape)
                    print(perturbed_indices.shape)

                    logits, loss = self.transformer(perturbed_indices, target)
                    loss.backward()
                    self.opt.step()

                    pbar.set_description(f"Epoch {epoch} at Step: {i+1}/{steps_per_epoch}")
                    pbar.set_postfix(Loss=loss.item())

                    trian_loss_per_epoch += loss.item()

                    # save losses per step
                    self.training_losses['loss'].append(loss.item())
            
            # save progress per epoch
            end_time = time.time()
            duration = timedelta(seconds=end_time-start_time)
            print(f'Training took {duration} seconds in total for now')
            self.training_losses['total_loss_per_epoch'].append(trian_loss_per_epoch/steps_per_epoch)
            self.training_losses['time'].append(duration)

            model_path = os.path.join(
                self.transformer_path,
                f"transformer_epoch_{epoch+1}.pth"
                )
            
            loss_path = os.path.join(
                self.transformer_path,
                f"transformer_losses_epoch_{epoch+1}.pkl"
                )

            
            torch.save(self.transformer.state_dict(), model_path)
            save_to_pkl(self.training_losses, loss_path)


#%%
if __name__ == "__main__":
    with hydra.initialize(config_path="../config/ex6"):
        cfg_vqgan = hydra.compose(config_name="vqgan")
        cfg_transformer = hydra.compose(config_name="transformer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transformer = TrainTransformer(cfg_vqgan, cfg_transformer, device)
    train_transformer.prepare_training(clear_all=True)
    train_transformer.train()

    # train_transformer.train()
# %%
