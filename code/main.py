from gridworld import *
from gridgenerator import *
from gridrender import *
import gridrender as gui
import numpy as np
import time
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from numpy import linalg as LA


gamma = 0.95
size = [15, 15]
puddle_location=[[3,5,7,9,11], [3,5,7,9,11]]
N_round = 10 # nb rounds
N_psi_learning = 1000 # nb rounds in Psi Learning in the same environment
N_policy_evaluation = 50 # nb rounds in policy evaluation (accuracy)
epsilon = 0.15 # exploration/exploitation for psi-learning

# The generator of random grid
grid_gen = GridGenerator(size=size, puddle_location=puddle_location)

# Initialize Psi
psi = np.zeros((size[0]*size[1], 4, 4 + 2*len(grid_gen.get_possible_puddle_location())))

# Array saving an evaluation of the policy at each round
policy_evaluation = []
rewards_psi = [0]
rewards_q = [0]


for i in tqdm(range(N_round)):
    # set seed
    seed = 2 * i + 10009283
    # Create a new grid
    env, w_true = grid_gen.create_Grid()
    # env.activate_render()
    np.random.seed(seed)
    # Learn Psi
    psi, policy, reward, w_stock = grid_gen.psi_learning(env, psi, epsilon, N_psi_learning, render=True)
    rewards_psi += reward

    np.random.seed(seed)
    # Learn Q
    q, pol, reward = grid_gen.q_learning(env, epsilon, N_psi_learning, render=True)
    rewards_q += reward
    # evaluate the policy according to Psi
    # policy_evaluation.append(env.evaluate_policy(policy, N_policy_evaluation, render=True))

# Show the last environment
# env.activate_render()
# render_policy(env, policy)
# state =14
# env.step(state, policy[state])
# render_policy(env, policy)

# policy_evaluation = np.array(policy_evaluation)
rewards_psi = np.array(rewards_psi)
rewards_psi = np.cumsum(rewards_psi)

rewards_q = np.array(rewards_q)
rewards_q = np.cumsum(rewards_q)

plt.figure(1)
plt.plot(rewards_psi, label='Psi-learning')
plt.plot(rewards_q, label='Q-learning')
plt.legend()
plt.xlabel('Rounds')
plt.show()