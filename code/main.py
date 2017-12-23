from gridworld import *
from gridgenerator import *
import gridrender as gui
import numpy as np
import time
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from numpy import linalg as LA





grid1 = [
    ['', '', '', '', '', '', '', '', '', 1],
    ['', '', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', '', '']
]

size = [8, 8]
nb_puddle = 2
puddle_location=[[3,5], [3,5]]


grid_gen = GridGenerator(size=size, nb_puddle=nb_puddle, puddle_location=puddle_location)

grid2, w_true = grid_gen.create_Grid()
# print(w_true)

env = GridWorld(gamma=0.95, grid=grid2, render=False)

possible_puddle_location = grid_gen.get_possible_puddle_location()
sta_poss_puddle = []
for a,b in possible_puddle_location:
    # print("(a,b) : ({}, {})".format(a, b))
    sta_poss_puddle.append(env.coord2state[a,b])

# print(state_of_possible_puddle_location)

#env = GridWorld1
# env.activate_render()

# pol = [1]*15 + ([0]*14 + [1])*13 + [0]*14 + [2]
pol = [1]*7 + ([0]*6 + [1])*5 + [0]*6 + [2]

n = size[0]
pol = [1]*(n) + ([0]*(n-1) + [1])*(n-2) + [0]*(n-1) + [2]

pol_tmp = np.array(pol).reshape((n,n))



# gui.render_policy(env, pol)

# w is the reward vector
# it represent the possible reward in the setting
# the first 4 element represent the reward in each corner of the grid
# w = np.array([0] * 54)

max_act = max(map(len, env.state_actions))
#q_init = np.random.rand(env.n_states, max_act)
q_init = np.zeros((env.n_states, max_act))

# pol = [1]*7 + ([0]*6 + [1])*5 + [0]*6 + [2]
a_54 = 4 + 2*len(sta_poss_puddle)

psi_init = np.random.randn(size[0]*size[1], 4, a_54)

def init_phi(grid_gen, env):


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



def q_learning(env, init_policy, epsilon, init_q, init_psi, sta_poss_puddle, phi, N, Tmax = 50) :
    """
    Args:
            env (GridWorld): The grid where the algorithm will be applied
            init_policy (int): The initial policy
            epsilon (int) : exploration rate, the probability to take a random path
            N (int) : number of samples
            Tmax (int) : the limite of transitions (episodic)

    Returns:
            q_final ([[float]]) : final Q value
            policy ([int]) : optimal policy according to Q-learning
            V ([[float]]): Values computed during the algorithm
    """
    gamma = env.gamma
    lrn_rate = 0.1
    t = 0
    alpha = []
    for i1,i2 in enumerate(env.state_actions) :
        alpha.append([])
        for j1, j2 in enumerate(i2):
            alpha[i1].append(0.5)

    pol = init_policy
    q = init_q
    psi = init_psi
    V = []
    w = np.zeros(len(phi[0][0]))
    w_stock = []
    #for n in np.arange(N):
    for n in tqdm(range(N)):
        state = env.reset()
        t_lim = 0
        absorbing = False
        while(not absorbing and t_lim < Tmax):

            greedy = np.random.rand(1) > epsilon
            # print("ACTION")
            # print(len(env.state_actions))
            # print(state)
            action = pol[state] if greedy else np.random.choice(env.state_actions[state])
            idx_action = env.state_actions[state].index(action)

            q_tmp = []
            prev_state = state
            prev_action = action
            state, reward, absorbing = env.step(state, action)
            if(reward!=0):
                print("reward = {} with phi = ".format(reward))
                print(phi[prev_state, prev_action])

            for idx, new_action in enumerate(env.state_actions[state]):
                q_tmp.append(q[state][idx])

            idx_best = np.argmax(q_tmp)
            best_q = q_tmp[idx_best]
            pol[state] = env.state_actions[state][idx_best]

            # Update Q, the Q-function
            # TD_q = reward + gamma*best_q - q[prev_state][idx_action]
            # q[prev_state][idx_action] = q[prev_state][idx_action] + alpha[prev_state][idx_action] * TD_q

            # Update Psi, the successor feature
            TD = phi[state][action] + gamma*psi[state][idx_best] - psi[prev_state][idx_action]
            psi[prev_state][idx_action] = psi[prev_state][idx_action] + alpha[prev_state][idx_action] * TD

            err = np.dot(phi[state][action], w) - reward
            w = w - lrn_rate * phi[state][action] * err

            # Update Q
            q[prev_state][idx_action] = np.dot(psi[prev_state][idx_action], w)

            # Update the vaue fonction
            v = []
            for i in np.arange(env.n_states):
                idx = env.state_actions[i].index(pol[i])
                v.append(q[i][idx])
            V.append(v)

            alpha[prev_state][idx_action] = 1./((1/alpha[prev_state][idx_action]) + 1.)
            t += 1
            t_lim += 1
        w_stock.append(w)


    return q, pol, V, w_stock


phi = init_phi(grid_gen, env)
q_final, policy, V, w_stock = q_learning(env, pol, 0.4, q_init, psi_init, sta_poss_puddle, phi, 1000)

# gui.render_policy(env, policy)

env.activate_render()
state = 14

state, reward, absorb = env.step(state, policy[state])

gui.render_policy(env, policy)

print("w hat")
print(w_stock[len(w_stock)-1])

print("w true")
print(w_true)

error = LA.norm(np.array(w_stock) - np.array(w_true), axis=1)

plt.figure(1)
plt.plot(error)
plt.show()

# env.activate_render()
# print(policy)
# print(q_final)
# state = 14
# # gui.render_policy(env, policy)
# state, reward, absorb = GridWorld1.step(state, policy[state])
# gui.render_policy(env, policy)


# env.activate_render()

# state = 14


# gui.render_policy(env, policy)
# print(policy)

# gui.render_q(env, q)
