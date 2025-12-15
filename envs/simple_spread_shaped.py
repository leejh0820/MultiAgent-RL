import numpy as np
from typing import Dict, Any, Tuple

from .simple_spread_wrapper import SimpleSpreadEnv


class SimpleSpreadShapedEnv(SimpleSpreadEnv):

    def __init__(
        self,
        n_agents: int = 3,
        max_cycles: int = 25,
        local_ratio: float = 0.5,
        team_reward_coef: float = 0.1,
        render_mode=None,
    ):
        super().__init__(
            n_agents=n_agents,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        self.team_reward_coef = team_reward_coef

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:

        obs, rewards, terms, truncs, infos = super().step(actions)

        reward_vals = np.array(list(rewards.values()), dtype=np.float32)
        team_mean = float(reward_vals.mean())

        shaped_rewards: Dict[str, float] = {}
        for agent_id, r in rewards.items():
            shaped = r + self.team_reward_coef * team_mean
            shaped_rewards[agent_id] = float(shaped)

        return obs, shaped_rewards, terms, truncs, infos
