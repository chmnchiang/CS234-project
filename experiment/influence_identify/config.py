from base_config import BaseConfig

class Config(BaseConfig):
    gamma = 0.9
    max_step = 300
    zero_initialize = True
    lr = 0.001
    eval_n = 20

    trans_n = 400
    trans_picked = 100

