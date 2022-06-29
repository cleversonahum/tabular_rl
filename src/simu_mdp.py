import numpy as np

from FiniteMDP import FiniteMDP
from UserSchedulingEnv import UserSchedulingEnv

episodes = [999]
n_steps = 1000
se = np.load("./src/spec_eff_matrix.npz")
se = se.f.spec_eff_matrix
env = UserSchedulingEnv()
mdp = FiniteMDP(env)
rewards = np.zeros((len(episodes), n_steps))

shouldPrintAll = True
state_values, iteration = mdp.compute_optimal_state_values()
if shouldPrintAll:
	print('Optimum states, iteration = ', iteration, ' state_values = ', np.round(state_values, 1))

optimal_action_values, iteration = mdp.compute_optimal_action_values()
if shouldPrintAll:
	print('Optimum actions, iteration = ', iteration, ' action_values = ', np.round(optimal_action_values, 1))

optimal_policy = mdp.convert_action_values_into_policy(optimal_action_values)
if shouldPrintAll:
	print('policy = ', optimal_policy)
	mdp.prettyPrintValues(optimal_policy, env.stateGivenIndexList, env.actionGivenIndexList)

for n_episode, episode in enumerate(episodes):
	file = np.load("./src/ep{}.npz".format(episode))
	for step in np.arange(n_steps):
		pos_ues = (file.f.ue1[step], file.f.ue2[step])
		# Here the agent should apply its action using UEs movement
		reward = 0 # Reward in a given step determined by the agent
		rewards[n_episode, step] = reward
