import hydra
from omegaconf import OmegaConf

experiment_idx = 6
@hydra.main(
config_path=f"/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/config/ex{experiment_idx}",
config_name="transformer",
version_base='1.2')

def main(cfg):
    a = cfg.architecture
    print(a.vocab_size)
    b = cfg.train
    print('---------------------------------')
    print(a)
    print(type(a))
    print('---------------------------------')
    print(b)
    print(type(b)) 

main()