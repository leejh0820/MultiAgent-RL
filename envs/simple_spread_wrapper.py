from __future__ import annotations
from typing import Dict, Any, Tuple, List

import numpy as np
from pettingzoo.mpe import simple_spread_v3


import numpy as np
from pettingzoo.mpe import simple_spread_v3


class SimpleSpreadEnv:
    def __init__(
        self,
        n_agents: int = 3,
        local_ratio: float = 0.5,
        max_cycles: int = 25,
        render_mode=None,
    ):
        self.n_agents = n_agents
        self.num_agents = n_agents
        self.local_ratio = local_ratio
        self.max_cycles = max_cycles
        self.render_mode = render_mode

        self.env = simple_spread_v3.parallel_env(
            N=n_agents,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=True,
            render_mode=render_mode,
        )
        self.env.reset()

        self.agents = list(self.env.agents)
        self.agent_ids = self.agents

        first_agent = self.agents[0]
        self.observation_space = self.env.observation_space(first_agent)
        self.action_space = self.env.action_space(first_agent)

    def reset(self, seed: int | None = None):
        obs, infos = self.env.reset(seed=seed)
        return obs

    def step(self, actions):
        return self.env.step(actions)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

        self.env.close()

    def get_global_state(self):

        return self.env.state()
