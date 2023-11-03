
from os import path
import yaml
import lpu3dnet

with open(path.join(path.dirname(lpu3dnet.__file__),"lpu_vqgan.yaml"),'r') as f:
    config_vqgan = yaml.safe_load(f)

