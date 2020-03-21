from config import set_config
from enum import Enum
from importlib import import_module

class Experiments(Enum):
    shapley_explain = 'shapley_explain'
    influence_explain = 'influence_explain'
    shapley_identify = 'shapley_identify'

    def __str__(self):
        return self.value
    

    def run(self, arg):
        module = import_module(f'.{self.value}', __name__)
        module.main(arg)
