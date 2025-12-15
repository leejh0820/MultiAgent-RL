import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from envs.simple_spread_shaped import SimpleSpreadShapedEnv
from agents.maddpg import MADDPG

N_AGENTS = 3
MAX_STEPS = 25
ROLLS = 30
DEVICE = "cpu"

SAVE_DIR = "experiments_maddpg_final"
CKPT_PATH = os.path.join(SAVE_DIR, "maddpg_last.pth")
OUT_DIR = os.path.join(SAVE_DIR, "viz")
os.makedirs(OUT_DIR, exist_ok=True)


def infer_dims(env):
    obs = env.reset()
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, _ = obs
    any_agent = list(obs.keys())[0]
    obs_dim = int(np.array(obs[any_agent]).shape[0])

    act_dim = None
    if hasattr(env, "action_space") and isinstance(env.action_space, dict):
        act_dim = int(env.action_space[any_agent].shape[0])
    elif hasattr(env, "action_space") and callable(getattr(env, "action_space")):
        act_dim = int(env.action_space(any_agent).shape[0])
    elif hasattr(env, "action_space"):
        act_dim = int(env.action_space.shape[0])
    if act_dim is None:
        raise RuntimeError("Could not infer act_dim from env.action_space")
    return obs_dim, act_dim


def load_ckpt(algo, ckpt):
    for i, ag in enumerate(algo.agents):
        blob = ckpt["agents"][i]
        ag.actor.load_state_dict(blob["actor"])
        ag.critic.load_state_dict(blob["critic"])
    print("✅ Loaded ckpt into algo")


def main():
    env = SimpleSpreadShapedEnv(n_agents=N_AGENTS)
    obs_dim, act_dim = infer_dims(env)

    algo = MADDPG(
        n_agents=N_AGENTS,
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.95,
        tau=0.01,
        device=DEVICE,
        buffer_capacity=1,
    )
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    load_ckpt(algo, ckpt)

    all_actions = []

    for _ in range(ROLLS):
        obs = env.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, _ = obs
        agent_ids = list(obs.keys())

        for t in range(MAX_STEPS):
            actions_raw = algo.select_actions(obs, agent_ids)
            actions = (
                actions_raw
                if isinstance(actions_raw, dict)
                else {agent_ids[i]: actions_raw[i] for i in range(len(agent_ids))}
            )
            a = np.stack(
                [actions[aid] for aid in agent_ids], axis=0
            )  # (n_agents, act_dim)
            all_actions.append(a)

            next_obs, rewards, terms, truncs, infos = env.step(
                {
                    aid: np.clip(actions[aid], 0, 1).astype(np.float32)
                    for aid in agent_ids
                }
            )
            obs = next_obs

    all_actions = np.array(all_actions, dtype=np.float32)  # (T, n_agents, act_dim)
    flat = all_actions.reshape(-1)

    plt.figure()
    plt.hist(flat, bins=50)
    plt.title("Action Distribution (all agents, all dims)")
    plt.xlabel("action value")
    plt.ylabel("count")
    out_png = os.path.join(OUT_DIR, "action_hist.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"✅ saved {out_png}")


if __name__ == "__main__":
    main()
