from gridworld import *
from gridgenerator import *
import gridrender as gui
import numpy as np
import time
import matplotlib.pyplot as plt
import time



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


grid_gen = GridGenerator()

grid2, w_true = grid_gen.create_Grid()
print(w_true)

GridWorld1 = GridWorld(gamma=0.95, grid=grid2, render=False)

env = GridWorld1

pol = [1]*15 + ([0]*14 + [1])*13 + [0]*14 + [2]
pol = [1]*7 + ([0]*6 + [1])*5 + [0]*6 + [2]
gui.render_policy(env, pol)

# w is the reward vector
# it represent the possible reward in the setting
# the first 4 element represent the reward in each corner of the grid

w = np.array([0] * 54)

max_act = max(map(len, env.state_actions))
q_q4 = np.random.rand(env.n_states, max_act)
#print(q)
# state = 0
# for i in np.arange(100):
#     state, reward, absorb = GridWorld1.step(state, pol[state])


def firstvisit_MC_evalutation(env, state, policy, n, Tlim=100):
    '''
    Implement the first visit Monte-Carlo algorithm
    Args:
            env (GridWorld): The grid where the algorithm will be applied
            state (int): The initial state
            policy (list of int) : the strategy to be evaluated
            n (int) : number of iteration
            Tmax (int) : the limite of iteration (episodic)

    Returns:
            mean_v (float): The Monte Carlos approximation of the value function
            stock_return ([float]): The matrix of the simulation
    '''
    assert state < env.n_states+1
    gamma = env.gamma

    stock_return = np.zeros(n)
    for i in np.arange(n):

        tmp_state = state
        r = 0
        t = 0
        absorbing = False
        while(not absorbing and t<Tlim):
            action = policy[tmp_state]
            tmp_state, reward, absorbing = env.step(tmp_state, action)
            r = r + (gamma**t) * reward
            t += 1

        stock_return[i] = r

    mean_v = np.mean(stock_return)
    return mean_v, stock_return

#mean, stock = firstvisit_MC_evalutation(env, 0, pol, 1000)
#print(mean)

def J(env, n, policy) :
    """
    Implement the first visit Monte Carlo for each initial state and compute the mean
    Args:
            env (GridWorld): The grid where the algorithm will be applied
            n (int): Number of iteration for each initial state
            policy (list of int) : the strategy to be evaluated
    Returns:
            mean_j (float): the mean of Monte Carlos approximations of the policy
    """
    J_n = np.zeros((env.n_states, n))
    for i in np.arange(env.n_states):
        V_n = np.array(firstvisit_MC_evalutation(env, i, policy, n)[1])
        tmps = np.cumsum(V_n)
        J_n[i, :] = tmps / np.arange(1,n+1)
    return np.mean(J_n, axis = 0)

def q_learning(env, init_policy, epsilon, init_q, N, Tmax = 100) :
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
    t = 0
    alpha = []
    for i1,i2 in enumerate(env.state_actions) :
        alpha.append([])
        for j1, j2 in enumerate(i2):
            alpha[i1].append(0.5)
    pol = init_policy
    q = init_q
    V = []
    for n in np.arange(N):
        state = env.reset()
        t_lim = 0
        absorbing = False
        while(not absorbing and t_lim < Tmax):

            greedy = np.random.rand(1) > epsilon
            action = pol[state] if greedy else np.random.choice(env.state_actions[state])
            idx_action = env.state_actions[state].index(action)

            q_tmp = []
            prev_state = state
            state, reward, absorbing = env.step(state, action)

            for idx, new_action in enumerate(env.state_actions[state]):
                q_tmp.append(q[state][idx])

            idx_best = np.argmax(q_tmp)
            pol[state] = env.state_actions[state][idx_best]
            best_q = np.max(np.array(q_tmp))

            TD = reward + gamma*best_q - q[prev_state][idx_action]

            # Update Q, the Q-function
            q[prev_state][idx_action] = q[prev_state][idx_action] + alpha[prev_state][idx_action] * TD

            # Update the vaue fonction
            v = []
            for i in np.arange(env.n_states):
                idx = env.state_actions[i].index(pol[i])
                v.append(q[i][idx])
            V.append(v)

            alpha[prev_state][idx_action] = 1/((1/alpha[prev_state][idx_action]) + 1)
            t += 1
            t_lim += 1


    return q, pol, V


q_final, policy, V = q_learning(env, pol, 0.1, q_q4, 5000)
gui.render_policy(env, policy)
print(policy)

env.activate_render()

state = 14
for i in np.arange(100):
    state, reward, absorb = GridWorld1.step(state, policy[state])

gui.render_policy(env, policy)
print(policy)

# gui.render_q(env, q)
