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

grid_gen = GridGenerator(size=size, puddle_location=puddle_location)

# w is the reward vector
# it represent the possible reward in the setting
# the first 4 element represent the reward in each corner of the grid


# The psi learning
psi = np.zeros((size[0]*size[1], 4, 4 + 2*len(grid_gen.get_possible_puddle_location())))

for i in range(2):
    env, w_true = grid_gen.create_Grid()

    psi, policy, V, w_stock = grid_gen.psi_learning(env, psi, 0.3, 1000)

    # env.activate_render()
    state = 14
    # state, reward, absorb = env.step(state, policy[state])

    # q = []
    # for i1,i2 in enumerate(env.state_actions) :
    #     q.append([])
    #     for j1, j2 in enumerate(i2):
    #         q[i1].append(np.dot(psi[i1][j2], w_true))
    # gui.render_q(env, q)


    # gui.render_policy(env, policy)


    error = LA.norm(np.array(w_stock) - np.array(w_true), axis=1)

    plt.figure(i)
    plt.plot(error, label='Erreur de w')
    plt.legend()
    plt.xlabel('Rounds')
    plt.ylabel('norm[ hat(w)-w ]')

plt.show()