import torch
import torch.nn as nn
import os
from lpu3dnet.frame.pretrained_architectures import Pretrained_Architecture
from lpu3dnet.modules.codebook_init import Codebook_init
from omegaconf import OmegaConf
import yaml
import lpu3dnet.config.pretrained as pretrained_config

class Pretrained_Models(nn.Module):
    def __init__(self):
        super(Pretrained_Models, self).__init__()
        # load pretrained encoder and decoder
        config_path = os.path.join(
            pretrained_config.__path__[0],
            'pretrained.yaml'
            )
            

        with open(config_path,'r') as file:
            yaml_cfg = yaml.safe_load(file)
        cfg = OmegaConf.create(yaml_cfg)

        
        root_path = os.path.join(cfg.checkpoints.PATH, cfg.experiment)
        # Load the state of the model_vqgan from the specified path
        epoch = cfg.epoch
        PATH_model = os.path.join(root_path, f'vqgan_epoch_{epoch}.pth')
        model_vqgan = Pretrained_Architecture(cfg)
        model_vqgan.load_state_dict(
            torch.load(
                    PATH_model
                    )
                    )
        
        model_vqgan.load_state_dict(torch.load(PATH_model))

        self.encoder = type(model_vqgan.encoder)(
            image_channels=cfg.architecture.encoder.img_channels,
            latent_dim=cfg.architecture.encoder.latent_dim,
            num_groups=cfg.architecture.encoder.num_groups,
            num_res_blocks=cfg.architecture.encoder.num_res_blocks,
            channels=cfg.architecture.encoder.channels
        )

        self.decoder = type(model_vqgan.decoder)(            
            image_channels=cfg.architecture.decoder.img_channels,
            latent_dim=cfg.architecture.decoder.latent_dim,
            num_groups=cfg.architecture.decoder.num_groups,
            num_res_blocks=cfg.architecture.decoder.num_res_blocks,
            channels=cfg.architecture.decoder.channels
            )

        self.encoder.load_state_dict(model_vqgan.encoder.state_dict())
        self.decoder.load_state_dict(model_vqgan.decoder.state_dict())


        # load pretrained codebook tensor
        codebook_init = Codebook_init(cfg.pretrained_codebook.path)
        pretrained_tensor = codebook_init.codebook_emd
        self.pretrained_codebook = pretrained_tensor

if __name__ == "__main__":
    model = Pretrained_Models()
    print(model.encoder)
    print(model.decoder)
    print(model.pretrained_codebook.shape)