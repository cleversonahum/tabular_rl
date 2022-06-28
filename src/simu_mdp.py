import numpy as np

from UserSchedulingEnv import UserSchedulingEnv

episodes = [999]
n_steps = 1000
se = np.load("./src/spec_eff_matrix.npz")
se = se.f.spec_eff_matrix

rewards = np.zeros((len(episodes), n_steps))
for n_episode, episode in enumerate(episodes):
	file = np.load("./src/ep{}.npz".format(episode))
	for step in np.arange(n_steps):
		pos_ues = (file.f.ue1[step], file.f.ue2[step])
		# Here the agent should apply its action using UEs movement
		reward = 0 # Reward in a given step determined by the agent
		rewards[n_episode, step] = reward
