import torch
from torch.utils.data import Dataset

TENSOR_TYPES = (
    torch.Tensor,
    torch.LongTensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
)

def history_to_tensors(history, device=None):
    tensors = tuple(
        tensor_type(x)
        for x, tensor_type in zip(zip(*history), TENSOR_TYPES)
    )
    if device is not None:
        tensors = tuple(x.to(device) for x in tensors)
    return tensors


class QLearningData(Dataset):

    def __init__(self, history):
        self.tensors = history_to_tensors(history)


    def __len__(self):
        return len(self.tensors[0])


    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)


class QLearningBatch:
    def __init__(self, data):
        state, action, reward, done, next_state = zip(*data)
        (self.state, 
         self.action, 
         self.reward,
         self.done,
         self.next_state) = (
             torch.stack(t) for t in (state, action, reward, done, next_state)
         )

    def pin_memory(self):
        self.state = self.state.pin_memory()
        self.action = self.action.pin_memory()
        self.reward = self.reward.pin_memory()
        self.done = self.done.pin_memory()
        self.next_state = self.next_state.pin_memory()
        return self

    def to(self, device):
        self.state = self.state.to(device)
        self.action = self.action.to(device)
        self.reward = self.reward.to(device)
        self.done = self.done.to(device)
        self.next_state = self.next_state.to(device)
        return self

    def __iter__(self):
        return (self.state, self.action, self.reward, self.done, self.next_state).__iter__()
