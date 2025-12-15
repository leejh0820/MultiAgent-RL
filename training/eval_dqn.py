import time
import numpy as np
import imageio
import matplotlib.pyplot as plt

from envs.simple_spread_wrapper import SimpleSpreadEnv
from agents.dqn_independent import IndependentDQNAgent
import torch


EPISODES = 3
MAX_STEPS = 25
N_AGENTS = 3
SAVE_PATH = "experiments/dqn_play.gif"


def make_discrete_actions(env):
    act_dim = env.action_space.shape[0]
    actions = []

    def vec(left, right, down, up, comm=0.0):
        v = np.zeros(act_dim, dtype=np.float32)
        v[0] = left
        v[1] = right
        v[2] = down
        v[3] = up
        if act_dim > 4:
            v[4] = comm
        return v

    actions.append(vec(0.0, 0.0, 0.0, 0.0))  # stay
    actions.append(vec(0.0, 1.0, 0.0, 0.0))  # right
    actions.append(vec(1.0, 0.0, 0.0, 0.0))  # left
    actions.append(vec(0.0, 0.0, 0.0, 1.0))  # up
    actions.append(vec(0.0, 0.0, 1.0, 0.0))  # down

    return np.stack(actions, axis=0)


def save_frame(env, frames):
    frame = env.render()
    frames.append(frame)


def main():
    env = SimpleSpreadEnv(render_mode="rgb_array", n_agents=N_AGENTS)

    agents_ids = env.agents
    obs_dim = env.observation_space.shape[0]
    discrete_actions = make_discrete_actions(env)
    n_actions = discrete_actions.shape[0]

    agents = {
        agent_id: IndependentDQNAgent(obs_dim, n_actions) for agent_id in agents_ids
    }

    # 학습된 가중치 로드
    for agent_id in agents_ids:
        agents[agent_id].q.load_state_dict(
            torch.load(f"experiments/{agent_id}_dqn.pth")
        )

    frames = []

    for ep in range(EPISODES):
        obs = env.reset(seed=ep)
        done = {agent_id: False for agent_id in agents_ids}

        for t in range(MAX_STEPS):
            actions_cont = {}

            for agent_id in agents_ids:
                if done[agent_id]:
                    continue
                a_idx = agents[agent_id].select_action(obs[agent_id])
                actions_cont[agent_id] = discrete_actions[a_idx]

            save_frame(env, frames)
            obs, rewards, terms, truncs, infos = env.step(actions_cont)

            for agent_id in agents_ids:
                if terms.get(agent_id, False) or truncs.get(agent_id, False):
                    done[agent_id] = True

            if all(done.values()):
                break

    imageio.mimsave(SAVE_PATH, frames, fps=10)
    env.close()
    print("GIF saved to:", SAVE_PATH)


if __name__ == "__main__":
    main()
