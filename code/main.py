from gridworld import *
from gridgenerator import *
import gridrender as gui
import numpy as np
import time
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from numpy import linalg as LA


size = [15, 15]
nb_puddle = 2
puddle_location=[[3,5,7,9,11], [3,5,7,9,11]]

grid_gen = GridGenerator(size=size, nb_puddle=nb_puddle, puddle_location=puddle_location)

# w is the reward vector
# it represent the possible reward in the setting
# the first 4 element represent the reward in each corner of the grid
grid2, w_true = grid_gen.create_Grid()
env = GridWorld(gamma=0.95, grid=grid2, render=False)

possible_puddle_location = grid_gen.get_possible_puddle_location()
sta_poss_puddle = []
for a,b in possible_puddle_location:
    # print("(a,b) : ({}, {})".format(a, b))
    sta_poss_puddle.append(env.coord2state[a,b])


def init_phi(grid_gen, env):
    '''
    Return phi :
    for each couple (state, action) which lead to a puddle or a goal
    (a reward not null), phi(state, action) is a one-hot vector where
    the one correspond to the index of the goal or the puddle
    '''
    size_r = env.n_rows
    size_c = env.n_cols
    possible_puddle_location = grid_gen.get_possible_puddle_location()
    nb_possible = len(sta_poss_puddle)
    phi = np.zeros((size_c*size_r, 4, 4 + 2*nb_possible))

    # Goal
    phi[1][2][0] = 1
    phi[size_c][3][0] = 1
    phi[size_c - 2][0][1] = 1
    phi[2*size_c - 1][3][1] = 1
    phi[size_c*(size_r - 2)][1][2] = 1
    phi[size_c*(size_r - 1) + 1][2][2] = 1
    phi[size_c*(size_r - 1) - 1][1][3] = 1
    phi[size_c*size_r - 2][0][3] = 1


    for idx,state in enumerate(sta_poss_puddle):
        # For the horizontal puddle
        phi[state - 2][0][4+idx] = 1
        phi[state + 2][2][4+idx] = 1
        phi[state + size_c][3][4+idx] = 1
        phi[state + size_c + 1][3][4+idx] = 1
        phi[state + size_c - 1][3][4+idx] = 1
        phi[state - size_c][1][4+idx] = 1
        phi[state - size_c + 1][1][4+idx] = 1
        phi[state - size_c - 1][1][4+idx] = 1

        # for the vertical puddle
        phi[state - 2*size_c][1][4+idx+nb_possible] = 1
        phi[state + 2*size_c][3][4+idx+nb_possible] = 1
        phi[state - 1][0][4+idx+nb_possible] = 1
        phi[state - size_c - 1][0][4+idx+nb_possible] = 1
        phi[state + size_c - 1][0][4+idx+nb_possible] = 1
        phi[state + 1][2][4+idx+nb_possible] = 1
        phi[state - size_c + 1][2][4+idx+nb_possible] = 1
        phi[state + size_c + 1][2][4+idx+nb_possible] = 1

    return phi

def psi_learning(env, epsilon, sta_poss_puddle, phi, N, Tmax = 50) :
    """
    Args:
            env (GridWorld): The grid where the algorithm will be applied
            init_policy (int): The initial policy
            epsilon (int) : exploration rate, the probability to take a random path
            N (int) : number of samples
            Tmax (int) : the limite of transitions (episodic)

    Returns:
            psi ([[float]]) : final successor features
            pol ([int]) : optimal policy according to the psi-learning
            V ([[float]]) : Values computed during the algorithm
            w_stock ([[float]]) : list successive value of w
    """
    gamma = env.gamma
    max_act = max(map(len, env.state_actions))
    sf_size = 4 + 2*len(sta_poss_puddle)
    psi = np.zeros((env.n_states, max_act, sf_size))

    # initialize a policy
    size = [env.n_rows, env.n_cols]
    pol = [1]*(size[1]) + ([0]*(size[1]-1) + [1])*(size[0]-2) + [0]*(size[1]-1) + [2]

    lrn_rate = 0.02
    t = 1
    alpha = []
    for i1,i2 in enumerate(env.state_actions) :
        alpha.append([])
        for j1, j2 in enumerate(i2):
            alpha[i1].append(0.5)

    # pol = init_policy

    V = []
    w = np.zeros(len(phi[0][0]))
    w_stock = []

    for n in tqdm(range(N)):
        state = env.reset()
        t_lim = 0
        absorbing = False
        while(not absorbing and t_lim < Tmax):

            greedy = np.random.rand(1) > epsilon

            action = pol[state] if greedy else np.random.choice(env.state_actions[state])

            # To update alpha
            idx_action = env.state_actions[state].index(action)

            q_tmp = []
            prev_state = state
            # prev_action = action
            state, reward, absorbing = env.step(state, action)

            # Compute the next expected Q-values
            for idx, new_action in enumerate(env.state_actions[state]):
                q_tmp.append(np.dot(psi[state][new_action], w))

            q_tmp = np.array(q_tmp)
            # Select the best action for the next step
            # idx_env = np.argmax(q_tmp)
            idx_env = np.random.choice(np.flatnonzero(q_tmp == q_tmp.max()))
            best_q = q_tmp[idx_env]

            # Update the policy
            pol[state] = env.state_actions[state][idx_env]

            # Update Psi, the successor feature
            TD_phi = phi[prev_state][action] + gamma*psi[state][pol[state]] - psi[prev_state][action]
            psi[prev_state][action] = psi[prev_state][action] + alpha[prev_state][idx_action] * TD_phi

            # Update w by gradient descent
            err = np.dot(phi[prev_state][action], w) - reward
            w = w - lrn_rate * phi[prev_state][action] * err / np.log(n+2) # smoothing convergence?

            # Update the value fonction
            v = []
            for i in np.arange(env.n_states):
                idx = env.state_actions[i].index(pol[i])
                v.append(np.dot(psi[i][pol[i]],w))
            V.append(v)

            alpha[prev_state][idx_action] = 1./((1/alpha[prev_state][idx_action]) + 1.)
            t_lim += 1

        w_stock.append(w)
    return psi, pol, V, w_stock


phi = init_phi(grid_gen, env)
# phi_nn = []
# for idx, phi_tmp in enumerate(phi):
#     for j in range(4):
#         if (LA.norm(phi_tmp[j])>1):
#             print("state : {}, action : {}, phi :".format(idx, j))
#             print(phi_tmp)

psi, policy, V, w_stock = psi_learning(env, 0.3, sta_poss_puddle, phi, 1000)

# # gui.render_policy(env, policy)

print("w hat")
print(w_stock[len(w_stock)-1])

print("w true")
print(w_true)

print("v final")
print(V[len(V)-1])

env.activate_render()
state = 14

state, reward, absorb = env.step(state, policy[state])

q = []
for i1,i2 in enumerate(env.state_actions) :
    q.append([])
    for j1, j2 in enumerate(i2):
        q[i1].append(np.dot(psi[i1][j2], w_true))

# gui.render_q(env, q)

gui.render_policy(env, policy)


error = LA.norm(np.array(w_stock) - np.array(w_true), axis=1)

plt.figure(1)
plt.plot(error, label='Erreur de w')
plt.legend()
plt.xlabel('Rounds')
plt.ylabel('norm[ hat(w)-w ]')
plt.show()
