from lpu3dnet.modules import decoder,encoder,codebook
import torch
from torch import nn
from torchinfo import summary
import hydra
from omegaconf import OmegaConf
from lpu3dnet.frame.pretrained_models import Pretrained_Models
import os

class VQGAN(nn.Module):
    def __init__(self,cfg):
        super(VQGAN, self).__init__()
        self.cfg = cfg
        self.save_path = os.path.join(
            self.cfg.checkpoints.PATH,
            self.cfg.experiment
            )
        
        os.makedirs(self.save_path, 
                    exist_ok=True)

        if self.cfg.pretrained:
            self.encoder = Pretrained_Models().encoder
            self.decoder = Pretrained_Models().decoder
            pretrained_codebook = Pretrained_Models().pretrained_codebook
        else:
            self.encoder = encoder.Encoder(
                image_channels=cfg.architecture.encoder.img_channels,
                latent_dim=cfg.architecture.encoder.latent_dim,
                num_groups=cfg.architecture.encoder.num_groups,
                num_res_blocks=cfg.architecture.encoder.num_res_blocks,
                channels=cfg.architecture.encoder.channels
            )

            self.decoder = decoder.Decoder(
                image_channels=cfg.architecture.decoder.img_channels,
                latent_dim=cfg.architecture.decoder.latent_dim,
                num_groups=cfg.architecture.decoder.num_groups,
                num_res_blocks=cfg.architecture.decoder.num_res_blocks,
                channels=cfg.architecture.decoder.channels
            )
            pretrained_codebook = None
        
        if cfg.usecodebook_ema:
            self.codebook = codebook.Codebook_EMA(
                size=cfg.architecture.codebookEMA.size,
                latent_dim=cfg.architecture.codebookEMA.latent_dim,
                beta_c=cfg.architecture.codebookEMA.beta_c,
                decay=cfg.architecture.codebookEMA.decay,
                init_ema=pretrained_codebook
            )        
        else:
            self.codebook = codebook.Codebook(
                size=cfg.architecture.codebook.size,
                latent_dim=cfg.architecture.codebook.latent_dim,
                beta_c=cfg.architecture.codebook.beta_c,
                autoencoder=cfg.architecture.codebook.autoencoder,
                legacy=cfg.architecture.codebook.legacy,
                init_ema=pretrained_codebook
            )
        
        if self.cfg.usecodebook_topk:
            self.codebook = codebook.Codebook_topk(
                size=cfg.architecture.codebook_topk.size,
                latent_dim=cfg.architecture.codebook_topk.latent_dim,
                beta_c=cfg.architecture.codebook_topk.beta_c,
                top_k=cfg.architecture.codebook_topk.top_k,
            )

        # conv layer before codebook
        self.quant_conv = nn.Conv3d(
            cfg.architecture.codebook.latent_dim,
            cfg.architecture.codebook.latent_dim,
            1)

        # conv layer after codebook
        self.post_quant_conv = nn.Conv3d(
            cfg.architecture.codebook.latent_dim,
            cfg.architecture.codebook.latent_dim,
            1)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        if not self.cfg.architecture.codebook.autoencoder:
            codebook_mapping, q_loss, (perplexity,min_encodings,codebook_indices)  = \
                self.codebook(quant_conv_encoded_images)
        else:
            codebook_mapping, a, b  = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        if not self.cfg.architecture.codebook.autoencoder:
            return decoded_images, (perplexity,min_encodings,codebook_indices), q_loss
        else:
            return decoded_images, b, a

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        if self.cfg.architecture.codebook.autoencoder:
            codebook_mapping, _, _  = self.codebook(quant_conv_encoded_images)
            return codebook_mapping

        codebook_mapping, q_loss, (perplexity,min_encodings,codebook_indices)  = \
              self.codebook(quant_conv_encoded_images)
        
        return codebook_mapping, (perplexity,min_encodings,codebook_indices), q_loss

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def freeze_codebook(self):
        for param in self.codebook.parameters():
            param.requires_grad = False
        
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True

    def unfreeze_codebook(self):
        for param in self.codebook.parameters():
            param.requires_grad = True

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images
        

    # make the discriminate later for the gan loss
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

    def save_checkpoint(self,epoch):
        save_model_path = os.path.join(     
            self.save_path,
            f'vqgan_epoch_{epoch}.pth'
            )
        torch.save(self.state_dict(), save_model_path)


if __name__ == "__main__":
    experiment_idx = 5
    @hydra.main(
    config_path=f"/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/config/ex{experiment_idx}",
    config_name="vqgan",
    version_base='1.2')

    def main(cfg):
        print(OmegaConf.to_yaml(cfg))
        model = VQGAN(cfg)
        summary(model, input_size=(20,1,64,64,64))
        model.save_checkpoint(0)
    
    main()