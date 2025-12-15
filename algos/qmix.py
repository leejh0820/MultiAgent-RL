import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Transition:
    obs: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    next_state: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: List[Transition] = []
        self.ptr = 0

    def push(self, tr: Transition):
        if len(self.data) < self.capacity:
            self.data.append(tr)
        else:
            self.data[self.ptr] = tr
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.data, batch_size)
        obs = np.stack([b.obs for b in batch], axis=0)
        state = np.stack([b.state for b in batch], axis=0)
        actions = np.stack([b.actions for b in batch], axis=0)
        rewards = np.stack([b.rewards for b in batch], axis=0)
        next_obs = np.stack([b.next_obs for b in batch], axis=0)
        next_state = np.stack([b.next_state for b in batch], axis=0)
        dones = np.stack([b.dones for b in batch], axis=0)
        return Transition(obs, state, actions, rewards, next_obs, next_state, dones)

    def __len__(self):
        return len(self.data)


class AgentQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        return self.net(obs)


class QMixer(nn.Module):

    def __init__(self, n_agents: int, state_dim: int, hidden: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden = hidden

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_agents * hidden),
        )
        self.hyper_b1 = nn.Linear(state_dim, hidden)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden * 1)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

        self.v = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, q_agents: torch.Tensor, state: torch.Tensor) -> torch.Tensor:

        B = q_agents.size(0)

        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.hidden)
        b1 = self.hyper_b1(state).view(B, 1, self.hidden)

        q_agents = q_agents.view(B, 1, self.n_agents)
        hidden = F.elu(torch.bmm(q_agents, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(B, self.hidden, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.view(B, 1)

        return q_tot


class QMIX:
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)

        self.agent = AgentQNet(obs_dim, n_actions).to(self.device)
        self.target_agent = AgentQNet(obs_dim, n_actions).to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())

        self.mixer = QMixer(n_agents, state_dim).to(self.device)
        self.target_mixer = QMixer(n_agents, state_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.opt = torch.optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()), lr=lr
        )

    @torch.no_grad()
    def act(self, obs_n: np.ndarray, eps: float) -> np.ndarray:

        if np.random.rand() < eps:
            return np.random.randint(
                0, self.n_actions, size=(self.n_agents,), dtype=np.int64
            )

        obs = torch.tensor(
            obs_n, dtype=torch.float32, device=self.device
        )  # (n, obs_dim)
        q = self.agent(obs)  # (n, n_actions)
        a = torch.argmax(q, dim=-1).detach().cpu().numpy().astype(np.int64)
        return a

    def update(self, batch: Transition, tau: float = 0.01):
        B = batch.obs.shape[0]

        obs = torch.tensor(batch.obs, dtype=torch.float32, device=self.device)
        state = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(
            batch.next_state, dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(batch.dones, dtype=torch.float32, device=self.device)

        q_all = self.agent(obs.view(B * self.n_agents, self.obs_dim)).view(
            B, self.n_agents, self.n_actions
        )
        q_taken = torch.gather(q_all, dim=2, index=actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            tq_all = self.target_agent(
                next_obs.view(B * self.n_agents, self.obs_dim)
            ).view(B, self.n_agents, self.n_actions)
            tq_max = torch.max(tq_all, dim=2).values

            r_team = rewards.mean(dim=1)
            done_team = torch.max(dones, dim=1).values

            q_tot_target = self.target_mixer(tq_max, next_state).squeeze(-1)
            y = r_team + self.gamma * (1.0 - done_team) * q_tot_target

        q_tot = self.mixer(q_taken, state).squeeze(-1)
        loss = F.mse_loss(q_tot, y)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.parameters()) + list(self.mixer.parameters()), 10.0
        )
        self.opt.step()

        # soft update target
        with torch.no_grad():
            for p, tp in zip(self.agent.parameters(), self.target_agent.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
            for p, tp in zip(self.mixer.parameters(), self.target_mixer.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        return float(loss.item())

    def save(self, path: str):
        torch.save(
            {"agent": self.agent.state_dict(), "mixer": self.mixer.state_dict()},
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(ckpt["agent"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
