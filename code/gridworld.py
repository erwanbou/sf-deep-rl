import numpy as np
from collections import namedtuple
import numbers
import gridrender as gui
from tkinter import Tk
import tkinter.font as tkfont
from tqdm import tqdm



MDP = namedtuple('MDP', 'S,A,P,R,gamma,d0')

class Policy:
    def __init__(self, env):
        self.size = [env.n_rows, env.n_cols]
        self.actions = []
        for i in range(env.n_states):
            self.actions.append(np.random.choice(env.state_actions[i]))

    def get_action(self, state):
        return self.actions[state]

    def update_action(self, state, action):
        self.actions[state] = action





class GridWorld:
    def __init__(self, gamma=0.95, grid=None, render=False, color = 'red'):
        self.grid = grid

        self.action_names = np.array(['right', 'down', 'left', 'up'])

        self.n_rows, self.n_cols = len(self.grid), max(map(len, self.grid))

        # Create a map to translate coordinates [r,c] to scalar index
        # (i.e., state) and vice-versa
        self.coord2state = np.empty_like(self.grid, dtype=np.int)
        self.n_states = 0
        self.state2coord = []
        for i in range(self.n_rows):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != 'x':
                    self.coord2state[i, j] = self.n_states
                    self.n_states += 1
                    self.state2coord.append([i, j])
                else:
                    self.coord2state[i, j] = -1

        # Compute the actions available in each state
        self.compute_available_actions()
        self.gamma = gamma
        self.proba_succ = 0.9
        self.render = render
        self.color = color

    def activate_render(self, color = 'red'):
        self.render = True
        self.color = color

    def deactivate_render(self):
        self.render = False

    def reset(self):
        """
        Returns:
            An initial state randomly drawn from
            the initial distribution
        """
        x_0 = np.random.randint(0, self.n_states)
        return x_0

    def step(self, state, action):
        """
        Args:
            state (int): the amount of good
            action (int): the action to be executed

        Returns:
            next_state (int): the state reached by performing the action
            reward (float): a scalar value representing the immediate reward
            absorb (boolean): True if the next_state is absorsing, False otherwise
        """
        r, c = self.state2coord[state]
        assert action in self.state_actions[state]
        if isinstance(self.grid[r][c], numbers.Number):
            return state, 0, True
        else:
            failed = np.random.rand(1) > self.proba_succ
            if action == 0:
                c = min(self.n_cols - 1, c + 1) if not failed else max(0, c - 1)
            elif action == 1:
                r = min(self.n_rows - 1, r + 1) if not failed else max(0, r - 1)
            elif action == 2:
                c = max(0, c - 1) if not failed else min(self.n_cols - 1, c + 1)
            elif action == 3:
                r = max(0, r - 1) if not failed else min(self.n_rows - 1, r + 1)

            if self.grid[r][c] == 'x':
                next_state = state
                r, c = self.state2coord[next_state]
            else:
                next_state = self.coord2state[r, c]
            if isinstance(self.grid[r][c], numbers.Number):
                reward = self.grid[r][c]
                absorb = True
                # if(reward == -1):
                #     absorb = False
                # else:
                #     absorb = True
            else:
                reward = 0.
                absorb = False

        if self.render:
            self.show(state, action, next_state, reward, color = self.color)

        return next_state, reward, absorb

    def show(self, state, action, next_state, reward, color = 'red'):
        dim = 40
        rows, cols = len(self.grid) + 0.5, max(map(len, self.grid))
        if not hasattr(self, 'window'):
            root = Tk()
            self.window = gui.GUI(root)

            self.window.config(width=cols * (dim + 12), height=rows * (dim + 12))
            my_font = tkfont.Font(family="Arial", size=32, weight="bold")
            for s in range(self.n_states):
                r, c = self.state2coord[s]
                x, y = 10 + c * (dim + 4), 10 + r * (dim + 4)
                if isinstance(self.grid[r][c], numbers.Number):
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                                               fill='blue', width=2)
                    self.window.create_text(x + dim / 2., y + dim / 2., text="{:.1f}".format(self.grid[r][c]),
                                            font=my_font, fill='white')
                else:
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                                               fill='white', width=2)
            self.window.pack()

        my_font = tkfont.Font(family="Arial", size=32, weight="bold")

        r0, c0 = self.state2coord[state]
        r0, c0 = 10 + c0 * (dim + 4), 10 + r0 * (dim + 4)
        x0, y0 = r0 + dim / 2., c0 + dim / 2.
        r1, c1 = self.state2coord[next_state]
        r1, c1 = 10 + c1 * (dim + 4), 10 + r1 * (dim + 4)
        x1, y1 = r1 + dim / 2., c1 + dim / 2.

        if hasattr(self, 'oval2'):
            # self.window.delete(self.line1)
            # self.window.delete(self.oval1)
            self.window.delete(self.oval2)
            self.window.delete(self.text1)
            self.window.delete(self.text2)

        # self.line1 = self.window.create_arc(x0, y0, x1, y1, dash=(3,5))
        # self.oval1 = self.window.create_oval(x0 - dim / 20., y0 - dim / 20., x0 + dim / 20., y0 + dim / 20., dash=(3,5))
        self.oval2 = self.window.create_oval(x1 - dim / 5., y1 - dim / 5., x1 + dim / 5., y1 + dim / 5., fill=color)
        self.text1 = self.window.create_text(dim, (rows - 0.25) * (dim + 12), font=my_font,
                                             text="r= {:.1f}".format(reward), anchor='center')
        self.text2 = self.window.create_text(2 * dim, (rows - 0.25) * (dim + 12), font=my_font,
                                             text="action: {}".format(self.action_names[action]), anchor='center')
        self.window.update()

    def matrix_representation(self):
        """
        Returns:
             A representation of the MDP in matrix form MDP(S, A_s, P, R, gamma) where
             - S is the number of states
             - A_s contains the list of action indices available in each state, i.e.,
                A_s[3] is a list representing the index of actions available in such state
             - P the transition matrix of dimension S x max{|A_s|} x S
             - R the reward matrix of dimension S x max{|A_s|}
        """
        if hasattr(self, 'P_mat'):
            return MDP(self.n_states, self.state_actions, self.P_mat, self.R_mat, self.gamma, self.d0)

        nstates = self.n_states
        nactions = max(map(len, self.state_actions))
        self.P_mat = np.inf * np.ones((nstates, nactions, nstates))
        self.R_mat = np.inf * np.ones((nstates, nactions))
        for s in range(nstates):
            r, c = self.state2coord[s]
            for a_idx, action in enumerate(self.state_actions[s]):
                self.P_mat[s, a_idx].fill(0.)
                if isinstance(self.grid[r][c], numbers.Number):
                    self.P_mat[s, a_idx, s] = 1.
                    self.R_mat[s, a_idx] = 0.
                else:
                    ns_succ, ns_fail = np.inf, np.inf
                    if action == 0:
                        ns_succ = self.coord2state[r, min(self.n_cols - 1, c + 1)]
                        ns_fail = self.coord2state[r, max(0, c - 1)]
                    elif action == 1:
                        ns_succ = self.coord2state[min(self.n_rows - 1, r + 1), c]
                        ns_fail = self.coord2state[max(0, r - 1), c]
                    elif action == 2:
                        ns_succ = self.coord2state[r, max(0, c - 1)]
                        ns_fail = self.coord2state[r, min(self.n_cols - 1, c + 1)]
                    elif action == 3:
                        ns_succ = self.coord2state[max(0, r - 1), c]
                        ns_fail = self.coord2state[min(self.n_rows - 1, r + 1), c]

                    x, y = self.state2coord[ns_fail]
                    if ns_fail == -1 or self.grid[x][y] == 'x':
                        ns_fail = s

                    self.P_mat[s, a_idx, ns_succ] = self.proba_succ
                    self.P_mat[s, a_idx, ns_fail] = 1. - self.proba_succ

                    x, y = self.state2coord[ns_fail]
                    x2, y2 = self.state2coord[ns_succ]
                    r_succ, r_fail = 0., 0.
                    if isinstance(self.grid[x][y], numbers.Number):
                        r_fail = self.grid[x][y]
                    if isinstance(self.grid[x2][y2], numbers.Number):
                        r_succ = self.grid[x2][y2]

                    self.R_mat[s, a_idx] = self.proba_succ * r_succ + (1 - self.proba_succ) * r_fail

        self.d0 = np.ones((nstates,)) / nstates

        return MDP(nstates, self.state_actions, self.P_mat, self.R_mat, self.gamma, self.d0)

    def compute_available_actions(self):
        # define available actions in each state
        # actions are indexed by: 0=right, 1=down, 2=left, 3=up
        self.state_actions = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                actions = [0, 1, 2, 3]
                if i == 0:
                    actions.remove(3)
                if j == self.n_cols - 1:
                    actions.remove(0)
                if i == self.n_rows - 1:
                    actions.remove(1)
                if j == 0:
                    actions.remove(2)

                self.state_actions.append(actions)


    def evaluate_policy(self, policy, N, render=True, Tmax=50):
        '''
        Evaluate a policy by Monte Carlo Method. It sample N trajectories
        with length less than Tmax and return the mean of the cumulative
        rewards with a discount factor gamma
        '''
        gamma = self.gamma
        Values = []
        N_state = self.n_states
        if(render):
            rang = tqdm(range(N_state), desc="Value evaluation")
        else:
            rang = range(N_state)
        for i in rang:
            value_by_state = []
            for j in range(N):
                absorbing = False
                t_lim = 0
                state = i
                s = 0
                while(not absorbing and t_lim<Tmax):
                    state, reward, absorbing = self.step(state, policy[state])
                    s += gamma**t_lim * reward
                    t_lim += 1
                value_by_state.append(s)
            value_by_state = np.array(value_by_state)
            Values.append(np.mean(value_by_state))
        Values = np.array(Values)
        return np.mean(Values)


