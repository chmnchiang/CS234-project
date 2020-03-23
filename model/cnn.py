import torch
from config import Config

class ConvModel(torch.nn.Module):

    def __init__(self, state_shape, action_n):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, 
                            out_channels=5,
                            kernel_size=(5, 5)),
            torch.nn.MaxPool2d(kernel_size=(4, 4)), 
            torch.nn.Conv2d(in_channels=5, 
                            out_channels=5,
                            kernel_size=(2, 2)),
            torch.nn.MaxPool2d(kernel_size=(2, 2)), 
        )

        # self.conv = torch.nn.Sequential(
            # torch.nn.Conv2d(in_channels=3, 
                            # out_channels=5,
                            # kernel_size=(3, 3),
                            # padding=1),
            # torch.nn.MaxPool2d(kernel_size=(3, 3)), 
            # torch.nn.Conv2d(in_channels=5, 
                            # out_channels=5,
                            # kernel_size=(2, 2)),
            # torch.nn.MaxPool2d(kernel_size=(2, 2)), 
        # )


        x = torch.zeros((1, *state_shape))
        x = self.conv(x)
        x = x.flatten(start_dim=1)

        hidden = x.shape[1]
        print('hidden =', hidden)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, action_n),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x

    def reset(self):
        # pass
        for params in self.model.parameters():
            torch.nn.init.zeros_(params)

