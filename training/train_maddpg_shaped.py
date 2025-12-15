import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

from envs.simple_spread_shaped import SimpleSpreadShapedEnv as SimpleSpreadEnv

from agents.maddpg import MADDPG


NUM_EPISODES = 600
MAX_STEPS = 25
BATCH_SIZE = 64
N_AGENTS = 3

SAVE_DIR = "experiments_maddpg"
os.makedirs("experiments_maddpg_shaped", exist_ok=True)


def make_action_space(env):
    act_dim = env.action_space.shape[0]
    return act_dim


def evaluate_and_save_gif(maddpg, gif_path, episodes=3):
    env = SimpleSpreadEnv(render_mode="rgb_array", n_agents=N_AGENTS)
    agent_ids = env.agents
    frames = []

    for ep in range(episodes):
        obs = env.reset(seed=100 + ep)
        done = {aid: False for aid in agent_ids}

        for t in range(MAX_STEPS):

            acts_dict = maddpg.select_actions(obs, agent_ids, explore=False)

            frame = env.render()
            frames.append(frame)

            next_obs, rewards, terms, truncs, infos = env.step(acts_dict)

            for aid in agent_ids:
                if terms.get(aid, False) or truncs.get(aid, False):
                    done[aid] = True

            obs = next_obs
            if all(done.values()):
                break

    imageio.mimsave(gif_path, frames, fps=10)
    env.close()
    print(f"[GIF SAVED] {gif_path}")


def train():
    env = SimpleSpreadEnv(render_mode=None, n_agents=N_AGENTS)
    agent_ids = env.agents

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    maddpg = MADDPG(
        n_agents=N_AGENTS,
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.95,
        tau=0.01,
        device="cpu",
        buffer_capacity=200000,
    )

    episode_rewards = []

    for ep in range(NUM_EPISODES):
        obs = env.reset(seed=ep)
        done = {aid: False for aid in agent_ids}
        ep_rews = np.zeros(N_AGENTS, dtype=np.float32)

        for t in range(MAX_STEPS):

            acts_dict = maddpg.select_actions(obs, agent_ids, explore=True)
            low = env.action_space.low
            high = env.action_space.high
            for aid in agent_ids:
                a = np.asarray(acts_dict[aid], dtype=np.float32)

                a = np.clip(a, low + 1e-6, high - 1e-6)
                acts_dict[aid] = a

            obs_all = np.array([obs[aid] for aid in agent_ids])
            acts_all = np.array([acts_dict[aid] for aid in agent_ids])

            next_obs, rewards, terms, truncs, infos = env.step(acts_dict)

            rews_all = np.array([rewards.get(aid, 0.0) for aid in agent_ids])
            dones_all = np.array(
                [
                    bool(terms.get(aid, False) or truncs.get(aid, False))
                    for aid in agent_ids
                ]
            )

            next_obs_all = np.array(
                [next_obs.get(aid, np.zeros_like(obs[aid])) for aid in agent_ids]
            )

            ep_rews += rews_all

            maddpg.push_transition(
                obs_all=obs_all,
                acts_all=acts_all,
                rews_all=rews_all,
                next_obs_all=next_obs_all,
                dones_all=dones_all,
            )

            maddpg.update(batch_size=BATCH_SIZE)

            obs = next_obs

            if all(dones_all):
                break

        mean_rew = np.mean(ep_rews)
        episode_rewards.append(mean_rew)

        if (ep + 1) % 10 == 0:
            print(f"[Episode {ep+1}/{NUM_EPISODES}] mean_reward={mean_rew:.3f}")

    env.close()

    rewards_arr = np.array(episode_rewards)
    # np.save(os.path.join(SAVE_DIR, "rewards_maddpg.npy"), rewards_arr)
    np.save(
        "experiments_maddpg_shaped/rewards_maddpg_shaped.npy", np.array(episode_rewards)
    )

    plt.figure()
    plt.plot(rewards_arr)
    plt.title("MADDPG on simple_spread")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (sum over agents)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "rewards_maddpg.png"))
    plt.close()

    gif_path = os.path.join(SAVE_DIR, "maddpg_play.gif")
    evaluate_and_save_gif(maddpg, gif_path)

    print(f"[DONE] Rewards curve + GIF saved to {SAVE_DIR}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    t0 = time.time()
    train()
    print(f"Total Time: {time.time() - t0:.1f} sec")
