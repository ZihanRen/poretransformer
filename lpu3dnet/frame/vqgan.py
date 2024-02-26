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


        self.encoder = encoder.Encoder(
            image_channels=cfg.architecture.encoder.img_channels,
            latent_dim=cfg.architecture.encoder.latent_dim,
            num_groups=cfg.architecture.encoder.num_groups,
            num_res_blocks=cfg.architecture.encoder.num_res_blocks,
            channels=cfg.architecture.encoder.channels,
            decrease_features=cfg.architecture.encoder.decrease_features
        )

        self.decoder = decoder.Decoder(
            image_channels=cfg.architecture.decoder.img_channels,
            latent_dim=cfg.architecture.decoder.latent_dim,
            num_groups=cfg.architecture.decoder.num_groups,
            num_res_blocks=cfg.architecture.decoder.num_res_blocks,
            channels=cfg.architecture.decoder.channels,
            decrease_features=cfg.architecture.decoder.decrease_features
        )
        pretrained_codebook = None
        
        self.codebook = codebook.Codebook(
            size=cfg.architecture.codebook.size,
            latent_dim=cfg.architecture.codebook.latent_dim,
            beta_c=cfg.architecture.codebook.beta_c,
            autoencoder=cfg.architecture.codebook.autoencoder,
            legacy=cfg.architecture.codebook.legacy,
            init_ema=pretrained_codebook
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

    def l2_reg(self):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param,2)
        return l2_reg

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

    @torch.no_grad()
    def gen_img_tokens(self,img):
        batch_num = img.shape[0]
        _,info,_ = self.encode(img)
        indices = info[2]   
        indices = indices.view(batch_num,-1) # reshape to (b,t)
        return indices

    @torch.no_grad()
    def tokens_to_z(self,indices,total_features_vec_num):
        # indices shape should be (b,t)
        # b: batch size
        # t: total number of tokens
        # target output shape should be the basis vector generated by codebook/encoder

        # flat the indices
        import math
        indices_flat = indices.view(-1)[:,None]
        # get the target z vector shape
        batch_num = indices.shape[0]
        latent_dim = self.cfg.architecture.codebook.latent_dim
        each_feature_vec_num = math.ceil(total_features_vec_num ** (1/3)) # xyz
    
        target_output_shape = (
            batch_num,
            latent_dim,
            each_feature_vec_num,
            each_feature_vec_num,
            each_feature_vec_num
            )
        
        target_tensor = torch.zeros(target_output_shape)
        z_q = self.codebook.get_codebook_entry(indices_flat,target_tensor)

        return z_q

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
    experiment_idx = 7
    @hydra.main(
    config_path=f"/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/config/ex{experiment_idx}",
    config_name="vqgan",
    version_base='1.2')

    def main(cfg):
        # print(OmegaConf.to_yaml(cfg))
        model = VQGAN(cfg)
        summary(model, input_size=(20,1,64,64,64))
        # model.save_checkpoint(0)
    
    main()