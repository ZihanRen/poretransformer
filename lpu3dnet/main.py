import hydra
import torch
from train.train_transformer import TrainTransformer
from train.train_vqgan import TrainVQGAN
from omegaconf import OmegaConf

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def train_transformer():
    with hydra.initialize(config_path="config"):
        # print all the config files

        cfg_vqgan = hydra.compose(config_name="vqgan")
        cfg_transformer = hydra.compose(config_name="transformer")
        cfg_dataset = hydra.compose(config_name="dataset")
        print(OmegaConf.to_yaml(cfg_vqgan))
        print(OmegaConf.to_yaml(cfg_transformer))
        print(OmegaConf.to_yaml(cfg_dataset))

    # device = torch.device("cpu") # used for debug
    train_transformer = TrainTransformer(cfg_vqgan,cfg_transformer,cfg_dataset,device)
    train_transformer.prepare_training(clear_all=True)
    train_transformer.train()


def train_vqgan():
    
    with hydra.initialize(config_path=f"config"):
        cfg_vqgan = hydra.compose(config_name="vqgan")
        cfg_dataset = hydra.compose(config_name="dataset")

    train = TrainVQGAN(device=device,cfg_vqgan=cfg_vqgan,cfg_dataset=cfg_dataset)
    train.prepare_training(removed_files=True)
    train.train()


if __name__ == "__main__":
    train_vqgan()
    # train_transformer()
