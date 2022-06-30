import matplotlib.pyplot as plt
import os
import numpy as np

reward_mdp = np.load("./hist/rewards.npz")
reward_mdp = reward_mdp.f.rewards
reward_rl = np.load("./hist/rewards_rl.npz")
reward_rl = reward_rl.f.rewards
reward_opt = np.load("./hist/rewards_opt.npz")
reward_opt = reward_opt.f.rewards
n_steps = 1000
ep_number = 99 # Equivalent to the 999 since we start at 900

# Cumulative reward episode 999
w, h = plt.figaspect(0.6)
fig = plt.figure(figsize=(w, h))
plt.xlabel("Step (n)", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.grid()
plt.plot(np.arange(n_steps), np.cumsum(reward_mdp[ep_number]), label="MDP")
plt.plot(np.arange(n_steps), np.cumsum(reward_rl[ep_number]), label="RL")
plt.plot(np.arange(n_steps), np.cumsum(reward_opt[ep_number]), label="Opt")
fig.tight_layout()
plt.xticks(fontsize=12)
plt.legend(fontsize=12)
os.makedirs("./results", exist_ok=True)
fig.savefig(
	"./results/reward_ep999.pdf",
	# bbox_inches="tight",
	pad_inches=0,
	format="pdf",
	dpi=1000,
)
# plt.show()
plt.close()


# Histogram
w, h = plt.figaspect(0.6)
fig = plt.figure(figsize=(w, h))
plt.xlabel("Step (n)", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.grid()
plt.hist(np.sum(reward_mdp, axis=1), label="MDP", alpha=0.5)
plt.hist(np.sum(reward_rl, axis=1), label="RL", alpha=0.5)
plt.hist(np.sum(reward_opt, axis=1), label="Opt", alpha=0.5)
fig.tight_layout()
plt.xticks(fontsize=12)
plt.legend(fontsize=12)
os.makedirs("./results", exist_ok=True)
fig.savefig(
	"./results/histogram.pdf",
	# bbox_inches="tight",
	pad_inches=0,
	format="pdf",
	dpi=1000,
)
# plt.show()
plt.close()