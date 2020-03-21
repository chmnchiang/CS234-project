from base_config import BaseConfig

class Config(BaseConfig):
    gamma = 0.9
    max_step = 300
    zero_initialize = True
    lr = 0.01
    eval_n = 20

    episodes_n = 16
    episodes_picked = 4
    shapley_iter = 15

