import numpy as np
from envs.simple_spread_wrapper import SimpleSpreadEnv
from algos.qmix import QMIX

N_AGENTS = 3
N_ACTIONS = 5
MAX_STEPS = 25


def onehot_action(a: int) -> np.ndarray:
    v = np.zeros((N_ACTIONS,), dtype=np.float32)
    v[a] = 1.0
    return v


def pack_actions(actions_int: np.ndarray):
    return {f"agent_{i}": onehot_action(int(actions_int[i])) for i in range(N_AGENTS)}


def collect_obs_state(obs_dict):
    obs_n = np.stack([obs_dict[f"agent_{i}"] for i in range(N_AGENTS)], axis=0)
    state = obs_n.reshape(-1).copy()
    return obs_n, state


def main():
    env = SimpleSpreadEnv(n_agents=N_AGENTS, max_cycles=MAX_STEPS, render_mode="human")
    obs_dict = env.reset()
    obs_n, state = collect_obs_state(obs_dict)
    obs_dim = obs_n.shape[1]
    state_dim = state.shape[0]

    algo = QMIX(
        n_agents=N_AGENTS,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=N_ACTIONS,
        device="cpu",
    )
    algo.load("experiments_qmix/qmix.pth")

    for ep in range(5):
        obs_dict = env.reset()
        obs_n, state = collect_obs_state(obs_dict)
        team_r = 0.0

        for t in range(MAX_STEPS):
            actions = algo.act(obs_n, eps=0.0)
            obs_dict, rewards, terms, truncs, infos = env.step(pack_actions(actions))
            obs_n, state = collect_obs_state(obs_dict)
            team_r += sum(rewards.values())
            if all(terms.values()) or all(truncs.values()):
                break

        print(f"[Eval EP {ep}] team_reward={team_r:.3f}")

    env.close()


if __name__ == "__main__":
    main()
