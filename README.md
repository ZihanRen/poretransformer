# LatentPoreUpscale3DNet (LPU3DNet)

This repository contains the implementation of a two-stage modeling framework for spatial upscaling and arbitrary-size 3D porous media reconstruction, as described in our paper ["Constrained Transformer-Based Porous Media Generation to Spatial Distribution of Rock Properties"](https://arxiv.org/abs/2410.21462).

## Overview

LPU3DNet addresses key challenges in pore-scale modeling of rock images:
- Accounting for spatial distribution of rock properties that influence flow and transport characteristics
- Generating structures above the representative elementary volume (REV) scale for transport properties

Our approach combines:
1. A Vector Quantized Variational Autoencoder (VQVAE) to compress and quantize sub-volume training images into low-dimensional tokens
2. A transformer model to spatially assemble these tokens into larger images following a specific spatial order

![VQVAE Workflow](figures/VQGAN_workflow.png)
![Transformer Workflow](figures/transformer_workflow.jpg)

## Results

Our model generates 3D porous media that match spatial distribution of rock properties and accurately model transport properties including permeability and multiphase flow relative permeability.

![Synthetic 3D Results](figures/synthetic_3d_sections.png)
![General Workflow in Reservoir](figures/general_workflow_in_reservoir.png)

## Repository Structure

- **modules/**: Core model implementation files
  - VQVAE components (encoder.py, decoder.py, codebook.py, etc.)
  - Transformer implementation (nanogpt.py, minigpt.py)
- **config/**: Configuration files for model hyperparameters
- **train/**: Training scripts for VQVAE and transformer models
- **inference/**: Scripts for model inference
- **eval/**: Evaluation utilities
- **post_process/**: Post-processing utilities and analysis notebooks
- **frame/**: Framework utilities

## Usage

### Environment Setup

Please refer to the dependency files in the repository:
- `req-dl.txt`: Deep learning dependencies
- `req-pnm.txt` / `req-pnm-new.txt`: Porous media simulation dependencies

### Training

Train the VQVAE model:
```python
python main.py
```

Train the transformer model (uncomment the relevant line in main.py):
```python
# Modify main.py to call train_transformer() instead of train_vqgan()
python main.py
```

## ⚠️ Important Notes

1. **Post-Processing**: The `post_process` folder contains notebooks and scripts for analysis but is still under refinement. The packages and dependencies are highly specialized and require specific installation instructions. Feel free to review the notebook contents, but running them directly is not recommended at this time.

2. **Configuration**: The `config` folder contains hyperparameter settings for the neural networks. Complete hyperparameter details will be fully open-sourced once the manuscript is published in an accepted journal.

## Citation

If you use this code in your research, please cite our paper:
```
@article{ren2024constrained,
  title={Constrained Transformer-Based Porous Media Generation to Spatial Distribution of Rock Properties},
  author={Ren, Zihan and Srinivasan, Sanjay and Crandall, Dustin},
  journal={arXiv preprint arXiv:2410.21462},
  year={2024}
}
``` 