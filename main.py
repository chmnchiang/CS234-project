from argparse import ArgumentParser
import random
import numpy
import torch
from experiment import Experiments

parser = ArgumentParser()
parser.add_argument('experiment', type=Experiments, choices=list(Experiments))
parser.add_argument('-s', '--seed', type=int, help='set the seed', default=0)
parser.add_argument('-a', '--arg', type=str, help='additional argument')

opts = parser.parse_args()
if opts.seed != -1:
    seed = opts.seed
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

opts.experiment.run(opts.arg)
