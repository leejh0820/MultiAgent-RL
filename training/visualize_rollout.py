import os
import numpy as np
import torch


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except:
    import imageio

from envs.simple_spread_shaped import SimpleSpreadShapedEnv
from agents.maddpg import MADDPG

N_AGENTS = 3
MAX_STEPS = 25
DEVICE = "cpu"

SAVE_DIR = "experiments_maddpg_final"
CKPT_PATH = os.path.join(SAVE_DIR, "maddpg_last.pth")
OUT_DIR = os.path.join(SAVE_DIR, "viz")
os.makedirs(OUT_DIR, exist_ok=True)


FIG_W, FIG_H = 5, 5
DPI = 120
FPS = 6


X_LIM = (-1.2, 1.2)
Y_LIM = (-1.2, 1.2)


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


def extract_positions_from_obs(obs, agent_ids):

    agent_pos = []
    for aid in agent_ids:
        o = np.array(obs[aid], dtype=np.float32)
        pos = o[2:4]
        agent_pos.append(pos)
    return np.stack(agent_pos, axis=0)


def _new_fixed_figure(title: str):

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.12)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_aspect("equal", adjustable="box")

    return fig, ax


def _fig_to_rgb_array(fig):

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)[..., :3]  # RGB
    return img


def rollout_once(algo, env, max_steps=MAX_STEPS, make_gif=True, tag="rollout"):
    obs = env.reset()
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, _ = obs

    agent_ids = list(obs.keys())

    traj = {aid: [] for aid in agent_ids}
    frames = []
    total_return = 0.0

    for t in range(max_steps):

        actions_raw = algo.select_actions(obs, agent_ids)

        actions = (
            actions_raw
            if isinstance(actions_raw, dict)
            else {agent_ids[i]: actions_raw[i] for i in range(len(agent_ids))}
        )

        for aid in actions:
            actions[aid] = np.clip(actions[aid], 0.0, 1.0).astype(np.float32)

        next_obs, rewards, terms, truncs, infos = env.step(actions)
        total_return += float(sum(rewards.values()))
        obs = next_obs

        agent_pos = extract_positions_from_obs(obs, agent_ids)
        lm_pos = None

        for i, aid in enumerate(agent_ids):
            traj[aid].append(agent_pos[i])

        if make_gif:
            fig, ax = _new_fixed_figure(f"{tag}  t={t}  return={total_return:.1f}")

            # landmark (있으면)
            if lm_pos is not None:
                ax.scatter(lm_pos[:, 0], lm_pos[:, 1], marker="x", s=80)

            # agents
            for i, aid in enumerate(agent_ids):
                p = agent_pos[i]
                ax.scatter([p[0]], [p[1]], s=90)
                ax.text(p[0] + 0.03, p[1] + 0.03, str(aid), fontsize=10)

            img = _fig_to_rgb_array(fig)
            frames.append(img)
            plt.close(fig)

    fig, ax = _new_fixed_figure(f"{tag} trajectory (return={total_return:.2f})")

    if lm_pos is not None:
        ax.scatter(lm_pos[:, 0], lm_pos[:, 1], marker="x", s=80)

    for aid in agent_ids:
        pts = np.stack(traj[aid], axis=0)
        ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=3, linewidth=1.5)
        ax.text(pts[0, 0], pts[0, 1], f"start {aid}", fontsize=9)

    out_png = os.path.join(OUT_DIR, f"{tag}_traj.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"✅ saved {out_png}")

    if make_gif and len(frames) > 0:
        out_gif = os.path.join(OUT_DIR, f"{tag}.gif")
        imageio.mimsave(out_gif, frames, fps=FPS)
        print(f"✅ saved {out_gif}")

    return total_return


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

    returns = []
    for k in range(10):
        r = rollout_once(algo, env, make_gif=False, tag=f"tmp_{k}")
        returns.append((r, k))

    returns.sort(key=lambda x: x[0])
    worst_r, worst_k = returns[0]
    best_r, best_k = returns[-1]
    print(f"worst: {worst_r:.3f} (k={worst_k}), best: {best_r:.3f} (k={best_k})")

    rollout_once(algo, env, make_gif=True, tag=f"best_return_{best_r:.1f}")
    rollout_once(algo, env, make_gif=True, tag=f"worst_return_{worst_r:.1f}")


if __name__ == "__main__":
    main()
