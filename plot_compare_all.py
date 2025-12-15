import os
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    dqn = np.load("experiments/rewards_dqn_independent.npy")
    maddpg = np.load("experiments_maddpg/rewards_maddpg.npy")
    maddpg_shaped = np.load("experiments_maddpg_shaped/rewards_maddpg_shaped.npy")

    dqn_s = moving_average(dqn, 10)
    maddpg_s = moving_average(maddpg, 10)
    shaped_s = moving_average(maddpg_shaped, 10)

    plt.figure(figsize=(8, 5))
    plt.plot(dqn_s, label="Independent DQN", alpha=0.7)
    plt.plot(maddpg_s, label="MADDPG (baseline)", alpha=0.7)
    plt.plot(shaped_s, label="MADDPG (reward-shaped)", alpha=0.7)

    plt.xlabel("Episode")
    plt.ylabel("Mean episode reward (sum over agents)")
    plt.title("DQN vs MADDPG vs MADDPG (reward-shaped)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("experiments_plots", exist_ok=True)
    out_path = "experiments_plots/dqn_vs_maddpg_vs_shaped.png"
    plt.savefig(out_path)
    plt.close()

    print("Saved plot:", out_path)


if __name__ == "__main__":
    main()
