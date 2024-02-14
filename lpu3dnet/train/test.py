import hydra
from omegaconf import OmegaConf
from hydra import initialize, compose

config_path = f"../config/ex{6}"

with initialize(config_path=config_path):
    cfg_vqgan = compose(config_name='vqgan')
    cfg_transformer = compose(config_name="transformer")



def main(cfg_vqgan, cfg_transformer):
    print(OmegaConf.to_yaml(cfg_vqgan))
    print(OmegaConf.to_yaml(cfg_transformer))

main(cfg_vqgan, cfg_transformer)
