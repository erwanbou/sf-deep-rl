from gridworld import *
import gridrender as gui
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


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
GridWorld1 = GridWorld(gamma=0.95, grid=grid1)

env = GridWorld1



pol = [0]*10 + ([0]*9 + [1])*8 + [0]*9 + [2]
gui.render_policy(env, pol)

max_act = max(map(len, env.state_actions))
q = np.random.rand(env.n_states, max_act)
gui.render_q(env, q)
