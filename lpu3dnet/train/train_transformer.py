#%%
import torch
from lpu3dnet.frame import vqgan
from lpu3dnet.frame import transformer
from lpu3dnet.train import dataset_transformer_cond
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
            cfg_dataset,
            device,
            ):
        
        # use self.cfg mainly on modules - vqgan & transformer
        self.cfg_vqgan = cfg_vqgan
        self.cfg_transformer = cfg_transformer
        self.cfg_dataset = cfg_dataset

        # decompose transformer config
        self.pretrained_vqgan_epoch = self.cfg_transformer.train.pretrained_vqgan_epoch
        self.root_path = os.path.join(self.cfg_dataset.checkpoints.PATH, self.cfg_dataset.experiment)
        self.transformer_path = os.path.join(self.root_path, 'transformer')
        self.device = device

        self.transformer = transformer.Transformer(self.cfg_transformer).to(device=self.device)
        
        # initializew optimizer
        self.opt = self.transformer.configure_optimizers(device=self.device)

        # intialize training losses tracking
        self.training_losses = {
            'total_loss': [],
            'mse_loss_last': [],
            'mse_loss_all': [],
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
        
        sos_token = self.cfg_transformer.train.sos_token
        
        
        print("Training transformer:")
        start_time = time.time()
        # using the same dataloader as VQGAN training process
        train_dataset = dataset_transformer_cond.Dataset_transformer(
                        self.cfg_dataset,
                        device=self.device)
        
        
        train_data_loader = DataLoader(
                            train_dataset,
                            batch_size=self.cfg_transformer.train.batch_size,
                            shuffle=True,
                            drop_last=False
                            )
        
        steps_per_epoch = len(train_data_loader)

        for epoch in range(self.cfg_transformer.train.epochs):

            train_loss_per_epoch = 0

            with tqdm(
                train_data_loader,
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.cfg_transformer.train.epochs}"
                ) as pbar:

                for i, data_obj in enumerate(pbar):
                    img_tokens,cond  = data_obj[0], data_obj[1]
                    # only select the first dimension of cond - porosity
                    cond = cond[:,:,0].unsqueeze(2)
                    
                    # train transformer
                    sos_tokens = torch.ones(img_tokens.shape[0], self.cfg_transformer.architecture.features_num) * sos_token
                    sos_tokens = sos_tokens.long().to(self.device)

                    input_tokens = torch.cat((sos_tokens, img_tokens), dim=1)

                    input_tokens = input_tokens[:,:-self.cfg_transformer.architecture.features_num]
                    target = img_tokens.clone()

                    logits = self.transformer(idx=input_tokens, cond=cond)
                    loss_last = self.transformer.loss_func_last(logits, target)
                    loss_all = self.transformer.loss_func_all(logits, target)
                    loss = loss_last * 0.05 + loss_all * 0.95
                    self.opt.zero_grad()
                    loss.backward() 
                    self.opt.step()

                    pbar.set_description(f"Epoch {epoch} at Step: {i+1}/{steps_per_epoch}")
                    pbar.set_postfix(Loss=loss.item())

                    train_loss_per_epoch += loss.item()

                    # save losses per step
                    self.training_losses['total_loss'].append(loss.item())
                    self.training_losses['mse_loss_last'].append(loss_last.item())
                    self.training_losses['mse_loss_all'].append(loss_all.item())

            
            # save progress per epoch
            
            self.training_losses['total_loss_per_epoch'].append(train_loss_per_epoch/steps_per_epoch)

            if epoch % 5 == 0:
                
                end_time = time.time()
                duration = timedelta(seconds=end_time-start_time)
                print(f'Training took {duration} seconds in total for now')
                self.training_losses['time'].append(duration)



                model_path = os.path.join(
                    self.transformer_path,
                    f"transformer_epoch_{epoch}.pth"
                    )
                
                loss_path = os.path.join(
                    self.transformer_path,
                    f"transformer_losses_epoch_{epoch}.pkl"
                    )

                
                torch.save(self.transformer.state_dict(), model_path)
                save_to_pkl(self.training_losses, loss_path)


#%%
if __name__ == "__main__":
    with hydra.initialize(config_path="../config"):
        cfg_vqgan = hydra.compose(config_name="vqgan")
        cfg_transformer = hydra.compose(config_name="transformer")
        cfg_dataset = hydra.compose(config_name="dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # used for debug
    train_transformer = TrainTransformer(cfg_vqgan,cfg_transformer,cfg_dataset,device)
    train_transformer.prepare_training(clear_all=True)
    train_transformer.train()

# %%
