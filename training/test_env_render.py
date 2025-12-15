import time
import numpy as np

from envs.simple_spread_wrapper import SimpleSpreadEnv


def main():
    env = SimpleSpreadEnv(render_mode="human", n_agents=3)

    agents = env.agents

    for episode in range(3):
        print(f"Episode {episode}")
        obs = env.reset(seed=episode)

        done = {agent: False for agent in agents}

        for t in range(50):
            actions = {}

            for agent in agents:
                if done[agent]:
                    continue

                act_dim = env.action_space.shape[0]
                actions[agent] = np.random.uniform(-1.0, 1.0, size=(act_dim,)).astype(
                    "float32"
                )

            obs, rewards, terms, truncs, infos = env.step(actions)

            env.render()

            for agent in agents:
                if terms.get(agent, False) or truncs.get(agent, False):
                    done[agent] = True

            if all(done.values()):
                break

            time.sleep(0.05)

    env.close()


if __name__ == "__main__":
    main()
