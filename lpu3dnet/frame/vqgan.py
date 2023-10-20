#%%
from lpu3dnet.modules import decoder,encoder,codebook
import torch 
from torch import nn 
from torchinfo import summary

class VQGAN(nn.Module):
    def __init__(self,latent_dim=256):
        super(VQGAN, self).__init__()
        self.encoder = encoder.Encoder()
        self.decoder = decoder.Decoder()
        self.codebook = codebook.Codebook()

        # conv layer before codebook
        self.quant_conv = nn.Conv3d(latent_dim, latent_dim, 1)
        self.post_quant_conv = nn.Conv3d(latent_dim, latent_dim, 1)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

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
    model = VQGAN()
    summary(model, input_size=(20,1,64,64,64))