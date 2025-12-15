import os
import numpy as np
import torch

from envs.simple_spread_shaped import SimpleSpreadShapedEnv
from agents.maddpg import MADDPG

# =========================
# Hyperparameters
# =========================
EPISODES = 800
MAX_STEPS = 25

ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
GAMMA = 0.95
TAU = 0.01

NOISE_START = 0.3
NOISE_END = 0.05
NOISE_DECAY = 0.995

BATCH_SIZE = 1024
REPLAY_SIZE = 200_000

N_AGENTS = 3
DEVICE = torch.device("cpu")

SAVE_DIR = "experiments_maddpg_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Environment
# =========================
env = SimpleSpreadShapedEnv(n_agents=N_AGENTS)


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
# ===========================================


# =========================
# Algorithm
# =========================
algo = MADDPG(
    n_agents=N_AGENTS,
    obs_dim=obs_dim,
    act_dim=act_dim,
    lr_actor=ACTOR_LR,
    lr_critic=CRITIC_LR,
    gamma=GAMMA,
    tau=TAU,
    device="cpu",
    buffer_capacity=REPLAY_SIZE,
)


# =========================
# Training Loop
# =========================
reward_history = []

noise_scale = NOISE_START

print("🚀 Start MADDPG PBRS Training")

for ep in range(1, EPISODES + 1):
    obs = env.reset()
    ep_reward = 0.0

    for step in range(MAX_STEPS):

        agent_ids = list(obs.keys())

        actions_raw = algo.select_actions(obs, agent_ids, noise_scale)

        if isinstance(actions_raw, dict):
            actions = actions_raw
        else:
            actions = {agent_ids[i]: actions_raw[i] for i in range(len(agent_ids))}

        for aid in actions:
            actions[aid] = np.clip(actions[aid], 0.0, 1.0).astype(np.float32)

        next_obs, rewards, terms, truncs, infos = env.step(actions)
        dones = {k: (terms[k] or truncs[k]) for k in terms}

        agent_ids = list(obs.keys())

        obs_all = np.stack(
            [np.array(obs[aid], dtype=np.float32) for aid in agent_ids], axis=0
        )

        next_obs_all = np.stack(
            [np.array(next_obs[aid], dtype=np.float32) for aid in agent_ids], axis=0
        )

        acts_all = np.stack(
            [np.array(actions[aid], dtype=np.float32) for aid in agent_ids], axis=0
        )

        rews_all = np.array([rewards[aid] for aid in agent_ids], dtype=np.float32)

        dones_all = np.array([dones[aid] for aid in agent_ids], dtype=np.float32)

        algo.push_transition(obs_all, acts_all, rews_all, next_obs_all, dones_all)

        if step > BATCH_SIZE:
            algo.update(BATCH_SIZE)

        obs = next_obs
        ep_reward += sum(rewards.values())

    reward_history.append(ep_reward)

    noise_scale = max(NOISE_END, noise_scale * NOISE_DECAY)

    if ep % 50 == 0:
        mean_50 = np.mean(reward_history[-50:])
        print(f"[EP {ep}/{EPISODES}] mean_reward(last50) = {mean_50:.3f}")


import os
import numpy as np
import torch

from envs.simple_spread_shaped import SimpleSpreadShapedEnv
from agents.maddpg import MADDPG

# =========================
# Hyperparameters
# =========================
EPISODES = 800
MAX_STEPS = 25

ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
GAMMA = 0.95
TAU = 0.01

NOISE_START = 0.3
NOISE_END = 0.05
NOISE_DECAY = 0.995

BATCH_SIZE = 1024
REPLAY_SIZE = 200_000

N_AGENTS = 3
DEVICE = torch.device("cpu")

SAVE_DIR = "experiments_maddpg_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Environment
# =========================
env = SimpleSpreadShapedEnv(n_agents=N_AGENTS)


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
# ===========================================


# =========================
# Algorithm
# =========================
algo = MADDPG(
    n_agents=N_AGENTS,
    obs_dim=obs_dim,
    act_dim=act_dim,
    lr_actor=ACTOR_LR,
    lr_critic=CRITIC_LR,
    gamma=GAMMA,
    tau=TAU,
    device="cpu",
    buffer_capacity=REPLAY_SIZE,
)


# =========================
# Training Loop
# =========================
reward_history = []

noise_scale = NOISE_START

print("🚀 Start MADDPG PBRS Training")

for ep in range(1, EPISODES + 1):
    obs = env.reset()
    ep_reward = 0.0

    for step in range(MAX_STEPS):
        # actions: dict(agent_id -> action)
        agent_ids = list(obs.keys())

        actions_raw = algo.select_actions(obs, agent_ids, noise_scale)

        if isinstance(actions_raw, dict):
            actions = actions_raw
        else:
            actions = {agent_ids[i]: actions_raw[i] for i in range(len(agent_ids))}

        next_obs, rewards, terms, truncs, infos = env.step(actions)
        dones = {k: (terms[k] or truncs[k]) for k in terms}

        agent_ids = list(obs.keys())

        obs_all = np.stack(
            [np.array(obs[aid], dtype=np.float32) for aid in agent_ids], axis=0
        )

        # next_obs: dict -> (n_agents, obs_dim)
        next_obs_all = np.stack(
            [np.array(next_obs[aid], dtype=np.float32) for aid in agent_ids], axis=0
        )

        # actions: dict -> (n_agents, act_dim)
        acts_all = np.stack(
            [np.array(actions[aid], dtype=np.float32) for aid in agent_ids], axis=0
        )

        # rewards: dict -> (n_agents,)
        rews_all = np.array([rewards[aid] for aid in agent_ids], dtype=np.float32)

        # dones: dict -> (n_agents,)
        dones_all = np.array([dones[aid] for aid in agent_ids], dtype=np.float32)

        algo.push_transition(obs_all, acts_all, rews_all, next_obs_all, dones_all)

        if step > BATCH_SIZE:
            algo.update(BATCH_SIZE)

        obs = next_obs
        ep_reward += sum(rewards.values())

    reward_history.append(ep_reward)

    noise_scale = max(NOISE_END, noise_scale * NOISE_DECAY)

    if ep % 50 == 0:
        mean_50 = np.mean(reward_history[-50:])
        print(f"[EP {ep}/{EPISODES}] mean_reward(last50) = {mean_50:.3f}")


save_path = os.path.join(SAVE_DIR, "maddpg_last.pth")

ckpt = {"agents": []}

save_path = os.path.join(SAVE_DIR, "maddpg_last.pth")

ckpt = {"agents": []}

for ag in algo.agents:
    blob = {
        "actor": ag.actor.state_dict(),
        "critic": ag.critic.state_dict(),
    }

    if hasattr(ag, "target_actor"):
        blob["target_actor"] = ag.target_actor.state_dict()
    if hasattr(ag, "target_critic"):
        blob["target_critic"] = ag.target_critic.state_dict()

    if "target_actor" not in blob and hasattr(ag, "actor_target"):
        blob["target_actor"] = ag.actor_target.state_dict()
    if "target_critic" not in blob and hasattr(ag, "critic_target"):
        blob["target_critic"] = ag.critic_target.state_dict()

    ckpt["agents"].append(blob)

torch.save(ckpt, save_path)
print(f"✅ Saved: {save_path}")


torch.save(ckpt, save_path)

np.save(
    os.path.join(SAVE_DIR, "rewards.npy"), np.array(reward_history, dtype=np.float32)
)

print(f"✅ Saved: {save_path} + rewards.npy")

print(f"  - {SAVE_DIR}/maddpg_last.pth")
print(f"  - {SAVE_DIR}/rewards.npy")
