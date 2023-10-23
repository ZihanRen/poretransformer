## training log

### Experiment 1
* VQVQE + GAN. WGAN & CNN encoder/decoder and codebook 
* GAN - w loss; no gradident penalty; discriminator start learning after 1000 steps; discriminator update and generator update is in the same training loop;
* VQGAN model is not learning or converging. Image quality is low.
* learning rate - 1e-4


#### Diagonostics
* adjust learning rate
* make sure VQGAN start reconstructing reasonable image before adding adversarial loss. Train VQGAN first
* There maybe some mistakes in GAN reconstruction loss
* Consider normalizing data before feeding to NNs?
* Separate discriminator update and generator update - making discriminator learn slower than VQ-VAE
* adding penalty to GAN loss
* increase training epoch??


### Experiment 2
* Run VQGAN individually to see whether VQGAN can sucessfully reconstruct the image
* check loss function of VQGAN - commitment loss and quantization loss
* adjust learning rate to be larger

### Experiment 3
* 
