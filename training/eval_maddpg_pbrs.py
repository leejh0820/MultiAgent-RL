import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import imageio
import numpy as np

try:
    from envs.simple_spread_shaped import SimpleSpreadShapedEnv as SimpleSpreadPBRSEnv
except ModuleNotFoundError:
    from envs.simple_spread_wrapper import SimpleSpreadEnv as SimpleSpreadPBRSEnv


from envs.simple_spread_shaped import SimpleSpreadShapedEnv as SimpleSpreadPBRSEnv
from agents.maddpg import MADDPG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--render", type=str, default=None)
    parser.add_argument("--save_gif", type=str, default=None)
    args = parser.parse_args()

    env = SimpleSpreadPBRSEnv(render_mode=args.render)
    obs_dict = env.reset()

    n_agents = env.n_agents
    obs_dim = len(obs_dict["agent_0"])
    act_dim = env.action_space.shape[0]

    algo = MADDPG(n_agents, obs_dim, act_dim)
    algo.load(args.ckpt)

    frames = []

    obs = obs_dict
    for step in range(25):
        actions = algo.act(obs, noise=0.0)

        if args.render == "human":
            env.render()

        if args.save_gif:
            frame = env.render(mode="rgb_array")
            frames.append(frame)

        obs, rewards, terms, truncs, infos = env.step(actions)

        if all(terms.values()):
            break

    if args.save_gif:
        imageio.mimsave(args.save_gif, frames, fps=10)
        print(f"Saved gif to {args.save_gif}")


if __name__ == "__main__":
    main()
