import os
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():

    dqn_path = os.path.join("experiments", "rewards_dqn_independent.npy")
    maddpg_path = os.path.join("experiments_maddpg", "rewards_maddpg.npy")

    dqn_rewards = np.load(dqn_path)
    maddpg_rewards = np.load(maddpg_path)

    dqn_smooth = moving_average(dqn_rewards, window=10)
    maddpg_smooth = moving_average(maddpg_rewards, window=10)

    plt.figure(figsize=(8, 5))

    plt.plot(dqn_smooth, label="Independent DQN (smoothed)", alpha=0.8)
    plt.plot(maddpg_smooth, label="MADDPG (smoothed)", alpha=0.8)

    plt.xlabel("Episode")
    plt.ylabel("Mean episode reward")
    plt.title("Independent DQN vs MADDPG on simple_spread")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("experiments_plots", exist_ok=True)
    out_path = os.path.join("experiments_plots", "dqn_vs_maddpg.png")
    plt.savefig(out_path)
    plt.close()

    print("Saved comparison plot to:", out_path)


if __name__ == "__main__":
    main()
