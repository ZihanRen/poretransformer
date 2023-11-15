from lpu3dnet.modules import decoder,encoder,codebook
import torch
from torch import nn
from torchinfo import summary
import hydra
from omegaconf import OmegaConf


class Pretrained_Architecture(nn.Module):
    def __init__(self,cfg):
        super(Pretrained_Architecture, self).__init__()
        self.cfg = cfg
        
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
            return decoded_images, a, b

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        if self.cfg.architecture.codebook.autoencoder:
            codebook_mapping, _, _  = self.codebook(quant_conv_encoded_images)
            return codebook_mapping

        codebook_mapping, q_loss, (perplexity,min_encodings,codebook_indices)  = \
              self.codebook(quant_conv_encoded_images)
        
        return codebook_mapping, (perplexity,min_encodings,codebook_indices), q_loss

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


if __name__ == "__main__":
    @hydra.main(
    config_path=f"/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/config/pretrained",
    config_name="pretrained",
    version_base='1.2')

    def main(cfg):
        print(OmegaConf.to_yaml(cfg))
        model = Pretrained_Architecture(cfg)
        summary(model, input_size=(20,1,64,64,64))
    
    main()