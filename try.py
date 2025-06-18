import numpy as np


M = 5
n = 2
tau = 1e3
constant = n / (M * tau)
rest_rewards = [0.061523, 0.02124, 0.041016, 0.040039, 0.047607]
_chosen_reward=0.048096
_reject_reward=0.00766

g_chosen = constant * sum([np.log((np.exp(tau*_chosen_reward) + np.exp(tau*r))) for r in rest_rewards])
g_reject = constant * sum([np.log((np.exp(tau*_reject_reward) + np.exp(tau*r))) for r in rest_rewards])

print(g_chosen, g_reject)