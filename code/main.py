from gridworld import *
from gridgenerator import *
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

# The generator of random grid
grid_gen = GridGenerator(size=size, puddle_location=puddle_location)

# Initialize Psi
psi = np.zeros((size[0]*size[1], 4, 4 + 2*len(grid_gen.get_possible_puddle_location())))

# Array saving an evaluation of the policy at each round
policy_evaluation = []
for i in tqdm(range(10)):
    # Create a new grid
    env, w_true = grid_gen.create_Grid()

    # Learn Psi
    psi, policy, V, w_stock = grid_gen.psi_learning(env, psi, 0.3, 100, render=False)

    # evaluate the policy according to Psi
    policy_evaluation.append(env.evaluate_policy(policy, 10000, render=False))

policy_evaluation = np.array(policy_evaluation)
plt.figure(1)
plt.plot(policy_evaluation, label='evaluation policy')
plt.legend()
plt.xlabel('Rounds')
# plt.ylabel('norm[ hat(w)-w ]')

plt.show()