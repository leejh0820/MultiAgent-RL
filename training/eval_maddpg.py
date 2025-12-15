import os
import numpy as np
import torch

from envs.simple_spread_shaped import SimpleSpreadShapedEnv
from agents.maddpg import MADDPG

N_AGENTS = 3
DEVICE = "cpu"

SAVE_DIR = "experiments_maddpg_final"
CKPT_PATH = os.path.join(SAVE_DIR, "maddpg_last.pth")

EVAL_EPISODES = 20
MAX_STEPS = 25


def _infer_dims(env):
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


def load_ckpt_into_algo(algo, ckpt):
    assert "agents" in ckpt and len(ckpt["agents"]) == len(algo.agents)
    for i, ag in enumerate(algo.agents):
        blob = ckpt["agents"][i]

        ag.actor.load_state_dict(blob["actor"])
        ag.critic.load_state_dict(blob["critic"])

        if (
            "target_actor" in blob
            and blob["target_actor"] is not None
            and hasattr(ag, "target_actor")
        ):
            ag.target_actor.load_state_dict(blob["target_actor"])
        if (
            "target_critic" in blob
            and blob["target_critic"] is not None
            and hasattr(ag, "target_critic")
        ):
            ag.target_critic.load_state_dict(blob["target_critic"])

    print("✅ Loaded ckpt into algo")


def main():
    env = SimpleSpreadShapedEnv(n_agents=N_AGENTS)
    obs_dim, act_dim = _infer_dims(env)

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
    load_ckpt_into_algo(algo, ckpt)

    returns = []
    for ep in range(EVAL_EPISODES):
        obs = env.reset()
        ep_return = 0.0

        for step in range(MAX_STEPS):
            agent_ids = list(obs.keys())
            actions_raw = algo.select_actions(obs, agent_ids)

            if isinstance(actions_raw, dict):
                actions = actions_raw
            else:
                actions = {agent_ids[i]: actions_raw[i] for i in range(len(agent_ids))}

            for aid in actions:
                actions[aid] = np.clip(actions[aid], 0.0, 1.0).astype(np.float32)

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            obs = next_obs
            ep_return += sum(rewards.values())

        returns.append(ep_return)
        print(f"[EVAL {ep+1}/{EVAL_EPISODES}] return = {ep_return:.3f}")

    returns = np.array(returns, dtype=np.float32)
    print("\n==== Eval Summary ====")
    print(
        f"mean = {returns.mean():.3f}, std = {returns.std():.3f}, min = {returns.min():.3f}, max = {returns.max():.3f}"
    )
    import json
    from datetime import datetime

    metrics = {
        "eval_episodes": int(EVAL_EPISODES),
        "max_steps": int(MAX_STEPS),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Saved metrics: {out_path}")


if __name__ == "__main__":
    main()
