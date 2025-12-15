import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class ImprovedMADDPG:
    def __init__(self, actors, critics, lr=1e-3, gamma=0.95, tau=0.01):
        self.actors = actors
        self.critics = critics
        self.target_actors = deepcopy(actors)
        self.target_critics = deepcopy(critics)

        self.opt_actor = [torch.optim.Adam(a.parameters(), lr=lr) for a in actors]
        self.opt_critic = [torch.optim.Adam(c.parameters(), lr=lr) for c in critics]

        self.gamma = gamma
        self.tau = tau

        # Gaussian noise parameters
        self.noise_std_init = 0.3
        self.noise_std_min = 0.05
        self.noise_decay = 0.995

    def act(self, obs, noise=True):
        actions = []
        for i, actor in enumerate(self.actors):
            a = actor(obs[i])
            if noise:
                std = max(self.noise_std_min, self.noise_std_init)
                a = a + torch.randn_like(a) * std
            actions.append(a.clamp(0, 1))
        self.noise_std_init *= self.noise_decay
        return actions

    def update(self, transitions):
        obs, actions, rewards, next_obs, dones = transitions

        batch_size = rewards[0].shape[0]

        # --- Update critics ---
        for i in range(len(self.critics)):
            with torch.no_grad():

                next_actions = []
                for j in range(len(self.target_actors)):
                    na = self.target_actors[j](next_obs[j])
                    na = na + torch.clamp(torch.randn_like(na) * 0.1, -0.2, 0.2)
                    next_actions.append(na.clamp(0, 1))

                q_next = self.target_critics[i](
                    torch.cat(next_obs, dim=1), torch.cat(next_actions, dim=1)
                )
                target_q = rewards[i] + self.gamma * q_next * (1 - dones[i])

            q = self.critics[i](torch.cat(obs, dim=1), torch.cat(actions, dim=1))

            critic_loss = F.mse_loss(q, target_q)

            self.opt_critic[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.opt_critic[i].step()

        # --- Update actors ---
        for i in range(len(self.actors)):
            pred_actions = [self.actors[j](obs[j]) for j in range(len(self.actors))]
            actor_loss = -self.critics[i](
                torch.cat(obs, dim=1), torch.cat(pred_actions, dim=1)
            ).mean()

            self.opt_actor[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.opt_actor[i].step()

        # --- Soft update ---
        for i in range(len(self.actors)):
            for tp, p in zip(
                self.target_actors[i].parameters(), self.actors[i].parameters()
            ):
                tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)

        for i in range(len(self.critics)):
            for tp, p in zip(
                self.target_critics[i].parameters(), self.critics[i].parameters()
            ):
                tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
