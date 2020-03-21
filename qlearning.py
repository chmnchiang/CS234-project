import numpy as np
import torch
from data import QLearningData, QLearningBatch, history_to_tensors
from config import Config
from torch.utils.data import DataLoader
from utils import eprint
from tqdm import tqdm

class OfflineQLearning:

    def __init__(self, env, model_class, device=Config.device):

        self.env = env
        self.state_shape = self.env.observation_space.shape
        self.action_n = self.env.action_space.n
        self.device = device

        self.model = model_class(
            self.state_shape, self.action_n).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.lr)

        self.gamma = Config.gamma
        

    def Q_value(self, state, action):

        if type(state) != torch.Tensor:
            state = torch.Tensor(state)

        if action is not None and type(action) != torch.Tensor:
            action = torch.LongTensor(action)

        Q_all = (self.model_fixed if use_fixed else self.model)(state)
        if action is not None:
            Q = torch.gather(Q_all, 1, action.view(-1, 1)).view(-1)
        else:
            Q = torch.max(Q_all, dim=1)[0]

        return Q


    def Q_values(self, state):

        if type(state) != torch.Tensor:
            state = torch.Tensor(state)

        state = state.to(self.device)

        expanded = False

        if len(state.size()) != len(self.state_shape) + 1:
            expanded = True
            state = state.unsqueeze(0)

        Qs = self.model(state)

        if expanded:
            Qs = Qs.squeeze(0)

        return Qs


    def Q_value(self, state, action):

        Qs = self.Q_values(state)

        if len(Qs.size()) == 1:
            Q = Qs[action]
        else:
            if type(action) != torch.Tensor:
                action = torch.LongTensor(action)

            Q = Qs.gather(1, action.view(-1, 1)).view(-1)

        return Q


    def Q_loss(self, batch):

        state, action, reward, done, next_state = batch

        Q = self.Q_value(state, action)
        Q_target = (
            reward +
            (1. - done) * self.gamma *
            torch.max(self.Q_values(next_state), dim=1)[0]
        ).detach()
        # print(Q.shape, Q_target.shape, (Q-Q_target).shape)

        loss = torch.mean((Q - Q_target)**2)

        return loss


    def train_with(self, history, epoch_n, evaluate=False):

        if not history:
            if evaluate:
                return self.eval()
            return

        dataset = QLearningData(history)
        data_loader = DataLoader(
            dataset,
            batch_size=Config.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=QLearningBatch,
        )

        eval_per_n = (epoch_n + 9) // 10

        for it in range(epoch_n):

            self.train_epoch(data_loader)

            if evaluate and (it+1) % eval_per_n == 0:
                ev = self.eval()
                print(f'Reward = {ev}')
                z = np.ones(16)


        if evaluate:
            ev = self.eval()
            return ev


    def train_epoch(self, data):

        losses = []

        for batch in data:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self.Q_loss(batch)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

        return np.mean(losses)


    def eval(self, n_iter=Config.eval_n, render=False):
        import time

        self.env.seed(np.random.randint(10**9))
        max_step = Config.max_step

        rewards = []

        for it in range(n_iter):

            state = self.env.reset()

            if render:
                self.env.render()

            done = False
            total_reward = 0.0

            for it in range(max_step):

                action = torch.argmax(self.Q_values(state)).item()
                state, reward, done, _ = self.env.step(action)

                if render:
                    self.env.render()
                    time.sleep(0.05)

                total_reward += reward

                if done or it >= max_step:
                    break

            rewards.append(total_reward)

        return np.mean(rewards)


    def grad_params(self, var, **kwargs):

        params = list(self.model.parameters())
        grad = torch.cat([
            x.flatten() for x in
            torch.autograd.grad(var, params, **kwargs)
        ])

        return grad


    def grad_all_loss(self, history):

        history = history_to_tensors(history, device=self.device)
        loss = self.Q_loss(history)
        grad = self.grad_params(loss)

        return grad.detach().cpu().numpy()
    

    def grad_single_loss(self, trans):

        return self.grad_all_loss([trans])


    def hessian(self, history):

        batch = history_to_tensors(history, device=self.device)
        loss = self.Q_loss(batch)
        grad = self.grad_params(loss, create_graph=True)
        
        hessian = []
        for g in tqdm(grad):
            grad2 = self.grad_params(g, retain_graph=True).cpu().numpy()
            hessian.append(grad2)

        hessian = np.array(hessian)
        return hessian


    def Q_grad(self, state, action):

        Q = self.Q_value(state, action)
        grad = self.grad_params(Q)

        return grad.detach().cpu().numpy()


    def save(self, path):
        torch.save(self.model.state_dict(), f'./saved_model/{path}')


    def load(self, path):
        self.model.load_state_dict(torch.load(f'./saved_model/{path}'))
