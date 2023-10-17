
from os import path
import yaml
import lpu3dnet

with open(path.join(path.dirname(lpu3dnet.__file__),"config.yaml"),'r') as f:
    PATH = yaml.safe_load(f)

