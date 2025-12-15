import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.replay_buffer import ReplayBuffer


class DQN(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class IndependentDQNAgent:

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 5000,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma

        self.q = DQN(obs_dim, act_dim)
        self.q_target = DQN(obs_dim, act_dim)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.replay = ReplayBuffer()
        self.steps = 0

    def select_action(self, obs: np.ndarray):
        self.steps += 1
        self.eps = max(self.eps_end, self.eps - (1.0 / self.eps_decay))

        if np.random.rand() < self.eps:
            return np.random.randint(self.act_dim)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch_size=64):
        if len(self.replay) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)

        states_t = torch.tensor(states)
        actions_t = torch.tensor(actions).long()
        rewards_t = torch.tensor(rewards)
        next_states_t = torch.tensor(next_states)
        dones_t = torch.tensor(dones)

        q_vals = self.q(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        next_q_vals = self.q_target(next_states_t).max(dim=1)[0]
        target = rewards_t + self.gamma * next_q_vals * (1 - dones_t)

        loss = torch.nn.functional.mse_loss(q_vals, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update
        for param, target_param in zip(self.q.parameters(), self.q_target.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
