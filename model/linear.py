import torch
from config import Config


class LinearModel(torch.nn.Module):

    def __init__(self, state_shape, action_n):
        super().__init__()

        x = torch.ones(1, *state_shape)
        x = x.flatten(start_dim=1)
        self.model = torch.nn.Linear(x.size()[1], action_n, bias=False)

        if Config.zero_initialize:
            self.reset()


    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.model(x)
        return x


    def reset(self):
        if Config.zero_initialize:
            torch.nn.init.zeros_(self.model.weight)
        else:
            self.model.reset_parameters()
