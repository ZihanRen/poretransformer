## training log

### Experiment 1
* autoencoder without codebook
* Results are okay - finally see reasonable reconstruction iamge


### Experiment 2
* Run VQGAN individually to see whether VQGAN can sucessfully reconstruct the image
* check loss function of VQGAN - commitment loss and embedding loss (check changing of commitment loss and embedding loss)
* adjust learning rate?

#### Dignostics
Results are not good and Reconstruction loss is high. Potential diagnostics:
* weight initialization
* balance embedding loss and reconstruction loss - give more weights on reconstruction loss. The codebook loss is ignored


#### Improvement no.1
* add more weight to codebook loss. Current training process is mainly rewarding reconstruction loss
#### Result
* Terrible. Codebook loss quickly converge; Somehow NNs fail to learn the representation of codebook


#### Improvement no.2
* Increase size of codebook (1000->10000)

#### Result
* It doesn't work

#### Improvement no.3
* give codebook loss 0.01 weight - damping the effect of updating the codebook

#### Improvement no.4
* gradually increase weight for codebook as training epochs increase - start with very small weight

#### Improvement no.5
* Take a new codebook approach - Karpathy gambling





### Experiment 3

VQVAE + GAN series of experiments

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



### Experiment 4
* use codebook ema series of experiments
* cannot converge - rec_Loss around 0.04



### Experiment 5
* pretrained autoencoder series of experiments




### Experiment 6 
* add more features basis vectors (expand 2->4 in each dimension so total is 64)
* add more dataset (17000 ish -> 32000)
