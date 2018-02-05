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
N_round = 50           # nb rounds
N_learning = 200       # nb rounds in Psi and Q Learning in the same environment
epsilon = 0.3          # exploration/exploitation for learning
N_MC = 1
# Choose Seed to get different values

total_rewards_psi = np.zeros(N_round*N_learning + 1)
total_rewards_q = np.zeros(N_round*N_learning + 1)

for j in tqdm(range(N_MC)):

    seeds = [39*j + 2, 22* j +1, 28* j + 4, 12* j + 19]

    # The generator of random grid
    grid_gen = GridGenerator(size=size, puddle_location=puddle_location)

    # Initialize Psi
    psi = np.zeros((size[0]*size[1], 4, 4 + 2*len(grid_gen.get_possible_puddle_location())))

    # Array saving an evaluation of the policy at each round
    policy_evaluation = []
    rewards_psi = [0]
    rewards_q = [0]

    for i in tqdm(range(N_round)):
        # Set seed
        seed = seeds[0] * i + seeds[1]
        # Create a new grid
        np.random.seed(seed)
        env1, w_true = grid_gen.create_Grid()
        # Learn Psi
        psi, policy, reward, w_stock = grid_gen.psi_learning(env1,
            psi, epsilon, N_learning,
            view_end = False, render=True,
            a_seed = i * seeds[2]+2,
            b_seed= i * seeds[3])
        rewards_psi += reward

        np.random.seed(seed)
        env2, w_true = grid_gen.create_Grid()
        env1.exchange_window(env2)
        # Learn Q
        q, pol, reward = grid_gen.q_learning(env2,
            epsilon, N_learning,
            view_end = False,
            render=True,
            a_seed = i * seeds[2]+2,
            b_seed= i * seeds[3])
        rewards_q += reward


    # policy_evaluation = np.array(policy_evaluation)
    rewards_psi = np.array(rewards_psi)
    rewards_psi = np.cumsum(rewards_psi)
    # print(rewards_psi.shape)
    total_rewards_psi += rewards_psi

    rewards_q = np.array(rewards_q)
    rewards_q = np.cumsum(rewards_q)
    total_rewards_q += rewards_q
    # print(rewards_q.shape)

total_rewards_psi = total_rewards_psi/N_MC
total_rewards_q = total_rewards_q/N_MC

plt.figure(1)
plt.plot(total_rewards_psi, label='Psi-learning')
plt.plot(total_rewards_q, label='Q-learning')
plt.legend()
plt.xlabel('Rounds')
plt.show()