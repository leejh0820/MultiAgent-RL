import os
import numpy as np
import torch

from envs.simple_spread_wrapper import SimpleSpreadEnv
from algos.qmix import QMIX, ReplayBuffer, Transition


N_AGENTS = 3
N_ACTIONS = 5
MAX_STEPS = 25

EPISODES = 800
BUFFER_SIZE = 50_000
BATCH_SIZE = 256
START_LEARN = 2_000
UPDATE_EVERY = 2
GAMMA = 0.99
LR = 1e-4

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

DEVICE = "cpu"


def onehot_action(a: int) -> np.ndarray:
    v = np.zeros((N_ACTIONS,), dtype=np.float32)
    v[a] = 1.0
    return v


def pack_actions(actions_int: np.ndarray):

    return {f"agent_{i}": onehot_action(int(actions_int[i])) for i in range(N_AGENTS)}


def collect_obs_state(obs_dict, env: SimpleSpreadEnv):

    obs_n = np.stack([obs_dict[f"agent_{i}"] for i in range(N_AGENTS)], axis=0)

    state = obs_n.reshape(-1).copy()
    return obs_n, state


def main():
    os.makedirs("experiments_qmix", exist_ok=True)

    env = SimpleSpreadEnv(n_agents=N_AGENTS, max_cycles=MAX_STEPS, render_mode=None)
    obs_dict = env.reset()

    obs_n, state = collect_obs_state(obs_dict, env)
    obs_dim = obs_n.shape[1]
    state_dim = state.shape[0]

    algo = QMIX(
        n_agents=N_AGENTS,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=N_ACTIONS,
        lr=LR,
        gamma=GAMMA,
        device=DEVICE,
    )

    buf = ReplayBuffer(BUFFER_SIZE)

    eps = EPS_START
    rewards_curve = []
    losses = []

    total_steps = 0

    for ep in range(1, EPISODES + 1):
        obs_dict = env.reset()
        obs_n, state = collect_obs_state(obs_dict, env)

        ep_team_reward = 0.0

        for t in range(MAX_STEPS):
            actions_int = algo.act(obs_n, eps)
            acts_dict = pack_actions(actions_int)

            next_obs_dict, rewards, terms, truncs, infos = env.step(acts_dict)
            next_obs_n, next_state = collect_obs_state(next_obs_dict, env)

            r_n = np.array(
                [rewards[f"agent_{i}"] for i in range(N_AGENTS)], dtype=np.float32
            )
            d_n = np.array(
                [terms[f"agent_{i}"] or truncs[f"agent_{i}"] for i in range(N_AGENTS)],
                dtype=np.bool_,
            )

            buf.push(
                Transition(
                    obs=obs_n,
                    state=state,
                    actions=actions_int.astype(np.int64),
                    rewards=r_n,
                    next_obs=next_obs_n,
                    next_state=next_state,
                    dones=d_n,
                )
            )

            obs_n, state = next_obs_n, next_state
            ep_team_reward += float(r_n.sum())
            total_steps += 1

            if len(buf) >= START_LEARN and (total_steps % UPDATE_EVERY == 0):
                batch = buf.sample(BATCH_SIZE)
                loss = algo.update(batch, tau=0.005)
                losses.append(loss)

            if d_n.all():
                break

        rewards_curve.append(ep_team_reward)
        eps = max(EPS_END, eps * EPS_DECAY)

        if ep % 50 == 0:
            mean50 = np.mean(rewards_curve[-50:])
            print(
                f"[EP {ep}/{EPISODES}] mean_team_reward(last50)={mean50:.3f}, eps={eps:.3f}"
            )

        if ep % 200 == 0:
            algo.save("experiments_qmix/qmix.pth")

    algo.save("experiments_qmix/qmix.pth")
    np.save(
        "experiments_qmix/rewards_qmix.npy", np.array(rewards_curve, dtype=np.float32)
    )
    if losses:
        np.save("experiments_qmix/losses_qmix.npy", np.array(losses, dtype=np.float32))

    print("Saved: experiments_qmix/qmix.pth, rewards_qmix.npy")


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
