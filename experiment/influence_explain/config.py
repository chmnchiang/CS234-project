from base_config import BaseConfig

class Config(BaseConfig):
    gamma = 0.9
    max_step = 200
    device = 'cuda'
    zero_initialize = True
    lr = 0.002
