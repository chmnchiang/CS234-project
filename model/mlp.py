import torch

class SimpleMLP(torch.nn.Module):

    def __init__(self, state_shape, action_n):
        super().__init__()

        hidden = 60

        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_shape[0], hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, action_n),
        )

    def forward(self, x):
        x = self.model(x)
        return x
