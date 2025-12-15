import os
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "experiments_maddpg_final"
rewards_path = os.path.join(SAVE_DIR, "rewards.npy")
out_png = os.path.join(SAVE_DIR, "rewards_curve.png")

rewards = np.load(rewards_path)


window = 50
if len(rewards) >= window:
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
else:
    ma = None

plt.figure()
plt.plot(rewards, label="episode reward")
if ma is not None:
    plt.plot(
        np.arange(window - 1, window - 1 + len(ma)), ma, label=f"moving avg ({window})"
    )
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=200)
print(f"✅ Saved plot: {out_png}")
