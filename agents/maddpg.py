from __future__ import annotations
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.marl_replay_buffer import MultiAgentReplayBuffer


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, total_obs_dim: int, total_act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs_all, act_all):
        """
        obs_all: (B, total_obs_dim)
        act_all: (B, total_act_dim)
        """
        x = torch.cat([obs_all, act_all], dim=-1)
        return self.net(x)


class OUNoise:

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            self.size
        )
        self.state = self.state + dx
        return self.state


class MADDPGAgent:
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        act_dim: int,
        n_agents: int,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        device: str = "cpu",
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device

        total_obs_dim = obs_dim * n_agents
        total_act_dim = act_dim * n_agents

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor_target = Actor(obs_dim, act_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(total_obs_dim, total_act_dim).to(device)
        self.critic_target = Critic(total_obs_dim, total_act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.noise = OUNoise(act_dim)

    def get_action(
        self, obs: np.ndarray, noise_scale: float = 0.1, explore: bool = True
    ):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            act = self.actor(obs_t).cpu().numpy()[0]

        if explore:
            act = act + noise_scale * self.noise.sample()

        # 박살나지 않게 더 타이트하게 클립
        act = np.clip(act, 1e-6, 1.0 - 1e-6)
        return act

    def soft_update(self, source: nn.Module, target: nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                (1.0 - self.tau) * target_param.data + self.tau * param.data
            )


class MADDPG:

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        device: str = "cpu",
        buffer_capacity: int = 200000,
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.agents: List[MADDPGAgent] = [
            MADDPGAgent(
                agent_id=i,
                obs_dim=obs_dim,
                act_dim=act_dim,
                n_agents=n_agents,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                device=device,
            )
            for i in range(n_agents)
        ]

        self.replay = MultiAgentReplayBuffer(capacity=buffer_capacity)
        self.gamma = gamma

    def select_actions(
        self, obs_dict: Dict[str, np.ndarray], agent_ids: List[str], explore=True
    ):

        actions = {}
        for idx, aid in enumerate(agent_ids):
            obs = obs_dict[aid]
            act = self.agents[idx].get_action(obs, explore=explore)
            actions[aid] = act
        return actions

    def push_transition(
        self,
        obs_all: np.ndarray,
        acts_all: np.ndarray,
        rews_all: np.ndarray,
        next_obs_all: np.ndarray,
        dones_all: np.ndarray,
    ):
        self.replay.push(obs_all, acts_all, rews_all, next_obs_all, dones_all)

    def update(self, batch_size: int = 64, min_buffer: int = 5000):

        if len(self.replay) < max(batch_size, min_buffer):
            return

        (
            obs_all,
            acts_all,
            rews_all,
            next_obs_all,
            dones_all,
        ) = self.replay.sample(batch_size)
        # shapes:
        # obs_all: (B, n_agents, obs_dim)
        # acts_all: (B, n_agents, act_dim)

        B = obs_all.shape[0]
        device = self.device

        obs_all_t = torch.tensor(obs_all, dtype=torch.float32, device=device)
        acts_all_t = torch.tensor(acts_all, dtype=torch.float32, device=device)
        rews_all_t = torch.tensor(rews_all, dtype=torch.float32, device=device)
        next_obs_all_t = torch.tensor(next_obs_all, dtype=torch.float32, device=device)
        dones_all_t = torch.tensor(dones_all, dtype=torch.float32, device=device)

        # flatten for critic: (B, n_agents*dim)
        obs_all_flat = obs_all_t.view(B, -1)
        acts_all_flat = acts_all_t.view(B, -1)
        next_obs_all_flat = next_obs_all_t.view(B, -1)

        rew_global = rews_all_t.mean(dim=1)  # (B,)

        with torch.no_grad():

            next_acts_list = []
            for i, agent in enumerate(self.agents):
                next_obs_i = next_obs_all_t[:, i, :]
                next_act_i = agent.actor_target(next_obs_i)
                next_acts_list.append(next_act_i)
            next_acts_all = torch.stack(next_acts_list, dim=1)  # (B, n_agents, act_dim)
            next_acts_all_flat = next_acts_all.view(B, -1)

        for i, agent in enumerate(self.agents):
            # ----- Critic update -----
            done_i = dones_all_t[:, i]  # (B,)

            # current Q
            q_vals = agent.critic(obs_all_flat, acts_all_flat).squeeze(-1)

            # target Q
            with torch.no_grad():
                target_q = agent.critic_target(
                    next_obs_all_flat, next_acts_all_flat
                ).squeeze(-1)
                y = rew_global + agent.gamma * target_q * (1.0 - done_i)

            critic_loss = nn.MSELoss()(q_vals, y)

            agent.critic_opt.zero_grad()
            critic_loss.backward()

            nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=0.5)
            agent.critic_opt.step()

            # ----- Actor update -----
            curr_acts_list = []
            for j, other_agent in enumerate(self.agents):
                obs_j = obs_all_t[:, j, :]
                if j == i:
                    act_j = other_agent.actor(obs_j)
                else:
                    act_j = other_agent.actor(obs_j).detach()
                curr_acts_list.append(act_j)

            curr_acts_all = torch.stack(curr_acts_list, dim=1)
            curr_acts_all_flat = curr_acts_all.view(B, -1)

            actor_loss = -agent.critic(obs_all_flat, curr_acts_all_flat).mean()

            agent.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=0.5)
            agent.actor_opt.step()

            # ----- Soft update -----
            agent.soft_update(agent.actor, agent.actor_target)
            agent.soft_update(agent.critic, agent.critic_target)

    # def update(self, batch_size: int = 64):
    #     if len(self.replay) < batch_size:
    #         return

    #     (
    #         obs_all,
    #         acts_all,
    #         rews_all,
    #         next_obs_all,
    #         dones_all,
    #     ) = self.replay.sample(batch_size)
    #     # shapes:
    #     # obs_all: (B, n_agents, obs_dim)
    #     # acts_all: (B, n_agents, act_dim)

    #     B = obs_all.shape[0]
    #     device = self.device

    #     obs_all_t = torch.tensor(obs_all, dtype=torch.float32, device=device)
    #     acts_all_t = torch.tensor(acts_all, dtype=torch.float32, device=device)
    #     rews_all_t = torch.tensor(rews_all, dtype=torch.float32, device=device)
    #     next_obs_all_t = torch.tensor(next_obs_all, dtype=torch.float32, device=device)
    #     dones_all_t = torch.tensor(dones_all, dtype=torch.float32, device=device)

    #     # flatten for critic: (B, n_agents*dim)
    #     obs_all_flat = obs_all_t.view(B, -1)
    #     acts_all_flat = acts_all_t.view(B, -1)
    #     next_obs_all_flat = next_obs_all_t.view(B, -1)

    #     with torch.no_grad():
    #         # target actions for next state
    #         next_acts_list = []
    #         for i, agent in enumerate(self.agents):
    #             next_obs_i = next_obs_all_t[:, i, :]
    #             next_act_i = agent.actor_target(next_obs_i)
    #             next_acts_list.append(next_act_i)
    #         next_acts_all = torch.stack(next_acts_list, dim=1)  # (B, n_agents, act_dim)
    #         next_acts_all_flat = next_acts_all.view(B, -1)

    #     for i, agent in enumerate(self.agents):
    #         # ----- Critic update -----
    #         rew_i = rews_all_t[:, i]  # (B,)
    #         done_i = dones_all_t[:, i]  # (B,)

    #         # current Q
    #         q_vals = agent.critic(obs_all_flat, acts_all_flat).squeeze(-1)

    #         # target Q
    #         with torch.no_grad():
    #             target_q = agent.critic_target(
    #                 next_obs_all_flat, next_acts_all_flat
    #             ).squeeze(-1)
    #             y = rew_i + agent.gamma * target_q * (1.0 - done_i)

    #         critic_loss = nn.MSELoss()(q_vals, y)

    #         agent.critic_opt.zero_grad()
    #         critic_loss.backward()
    #         agent.critic_opt.step()

    #         # ----- Actor update -----
    #         # 다른 에이전트 액션은 고정, i번째 agent의 액션만 actor로부터 얻기
    #         curr_acts_list = []
    #         for j, other_agent in enumerate(self.agents):
    #             obs_j = obs_all_t[:, j, :]
    #             if j == i:
    #                 act_j = other_agent.actor(obs_j)
    #             else:
    #                 # detach to avoid backprop through other actors
    #                 act_j = other_agent.actor(obs_j).detach()
    #             curr_acts_list.append(act_j)

    #         curr_acts_all = torch.stack(curr_acts_list, dim=1)
    #         curr_acts_all_flat = curr_acts_all.view(B, -1)

    #         actor_loss = -agent.critic(obs_all_flat, curr_acts_all_flat).mean()

    #         agent.actor_opt.zero_grad()
    #         actor_loss.backward()
    #         agent.actor_opt.step()

    #         # ----- Soft update -----
    #         agent.soft_update(agent.actor, agent.actor_target)
    #         agent.soft_update(agent.critic, agent.critic_target)
