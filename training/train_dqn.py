import os
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio

from envs.simple_spread_wrapper import SimpleSpreadEnv
from agents.dqn_independent import IndependentDQNAgent


NUM_EPISODES = 500
MAX_STEPS = 25
BATCH_SIZE = 64
LR = 1e-3
GAMMA = 0.95

N_AGENTS = 3

SAVE_DIR = "experiments"
os.makedirs(SAVE_DIR, exist_ok=True)


def make_discrete_actions(env):
    """
    continuous action space -> 우리가 쓸 discrete action set 정의
    simple_spread_v3의 연속 action dim: 5
    [left, right, down, up, comm] in [0, 1]
    """
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


def evaluate_and_save_gif(agents, discrete_actions, gif_path, episodes=3):

    env_eval = SimpleSpreadEnv(render_mode="rgb_array", n_agents=N_AGENTS)
    agents_ids = env_eval.agents
    frames = []

    for ep in range(episodes):
        obs = env_eval.reset(seed=100 + ep)
        done = {aid: False for aid in agents_ids}

        for t in range(MAX_STEPS):
            actions_cont = {}

            for aid in agents_ids:
                if done[aid]:
                    continue

                obs_t = torch.tensor(obs[aid], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_vals = agents[aid].q(obs_t)
                a_idx = int(torch.argmax(q_vals, dim=1).item())
                actions_cont[aid] = discrete_actions[a_idx]

            frame = env_eval.render()
            frames.append(frame)

            obs, rewards, terms, truncs, infos = env_eval.step(actions_cont)

            for aid in agents_ids:
                if terms.get(aid, False) or truncs.get(aid, False):
                    done[aid] = True

            if all(done.values()):
                break

    imageio.mimsave(gif_path, frames, fps=10)
    env_eval.close()
    print("Saved GIF to:", gif_path)


def train():
    env = SimpleSpreadEnv(render_mode=None, n_agents=N_AGENTS)

    agents_ids = env.agents
    obs_dim = env.observation_space.shape[0]

    discrete_actions = make_discrete_actions(env)
    n_actions = discrete_actions.shape[0]

    agents = {
        agent_id: IndependentDQNAgent(
            obs_dim=obs_dim,
            act_dim=n_actions,
            lr=LR,
            gamma=GAMMA,
        )
        for agent_id in agents_ids
    }

    episode_rewards = []

    for ep in range(NUM_EPISODES):
        obs = env.reset(seed=ep)
        done = {agent_id: False for agent_id in agents_ids}
        ep_reward = defaultdict(float)

        for t in range(MAX_STEPS):
            actions_continuous = {}
            actions_idx = {}

            for agent_id in agents_ids:
                if done[agent_id]:
                    continue

                obs_agent = obs[agent_id]
                a_idx = agents[agent_id].select_action(obs_agent)
                actions_idx[agent_id] = a_idx
                actions_continuous[agent_id] = discrete_actions[a_idx]

            next_obs, rewards, terminations, truncations, infos = env.step(
                actions_continuous
            )

            for agent_id in agents_ids:
                if done[agent_id]:
                    continue

                o = obs[agent_id]
                r = rewards.get(agent_id, 0.0)
                ep_reward[agent_id] += r

                no = next_obs.get(agent_id, np.zeros_like(o))
                d = bool(
                    terminations.get(agent_id, False)
                    or truncations.get(agent_id, False)
                )

                a_idx = actions_idx[agent_id]

                agents[agent_id].replay.push(o, a_idx, r, no, d)
                agents[agent_id].update(BATCH_SIZE)

                if d:
                    done[agent_id] = True

            obs = next_obs

            if all(done.values()):
                break

        mean_ep_reward = np.mean([ep_reward[a] for a in agents_ids])
        episode_rewards.append(mean_ep_reward)

        if (ep + 1) % 10 == 0:
            print(
                f"[Episode {ep+1}/{NUM_EPISODES}] "
                f"mean_reward={mean_ep_reward:.3f}, "
                f"eps(example)={agents[agents_ids[0]].eps:.3f}"
            )

    env.close()

    rewards_arr = np.array(episode_rewards, dtype=np.float32)
    np.save(os.path.join(SAVE_DIR, "rewards_dqn_independent.npy"), rewards_arr)

    plt.figure()
    plt.plot(rewards_arr)
    plt.xlabel("Episode")
    plt.ylabel("Mean episode reward (over agents)")
    plt.title("Independent DQN on simple_spread")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "rewards_dqn_independent.png"))
    plt.close()

    for agent_id, agent in agents.items():
        save_path = os.path.join(SAVE_DIR, f"{agent_id}_dqn.pth")
        torch.save(agent.q.state_dict(), save_path)
        print(f"Saved checkpoint for {agent_id} to {save_path}")

    gif_path = os.path.join(SAVE_DIR, "dqn_play.gif")
    evaluate_and_save_gif(agents, discrete_actions, gif_path, episodes=3)

    print("Training finished. Rewards, models, and GIF saved to:", SAVE_DIR)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    t0 = time.time()
    train()
    print(f"Total time: {time.time() - t0:.1f} sec")
