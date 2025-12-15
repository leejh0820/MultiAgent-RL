import numpy as np
from envs.simple_spread_shaped import SimpleSpreadShapedEnv

N_AGENTS = 3
EVAL_EPISODES = 20
MAX_STEPS = 25


def main():
    env = SimpleSpreadShapedEnv(n_agents=N_AGENTS)
    returns = []

    for ep in range(EVAL_EPISODES):
        obs = env.reset()
        ep_return = 0.0

        for _ in range(MAX_STEPS):
            agent_ids = list(obs.keys())
            actions = {}
            for aid in agent_ids:

                actions[aid] = np.random.uniform(0.0, 1.0, size=(5,)).astype(np.float32)

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            obs = next_obs
            ep_return += sum(rewards.values())

        returns.append(ep_return)
        print(f"[RANDOM {ep+1}/{EVAL_EPISODES}] return = {ep_return:.3f}")

    returns = np.array(returns, dtype=np.float32)
    print("\n==== Random Eval Summary ====")
    print(
        f"mean = {returns.mean():.3f}, std = {returns.std():.3f}, min = {returns.min():.3f}, max = {returns.max():.3f}"
    )


if __name__ == "__main__":
    main()
