import numpy as np
import itertools
from gridworld import *
from tqdm import tqdm
from numpy import linalg as LA


class GridGenerator:

    def __init__(self, size=[15,15],
        puddle_location=[[3,5,7,9,11], [3,5,7,9,11]],
        # puddle_location=[[3,5], [3,5]],
        seed=None,
        gamma=0.95):

        if(seed!=None):
            np.random.seed(seed)

        self.seed = seed
        self.size = size
        self.puddle_location = puddle_location
        # all possible location of the puddle
        self.possible_puddle_location = list(itertools.product(puddle_location[0], puddle_location[1]))
        self.gamma = gamma
        self.init_phi()

    def create_Grid(self):
        '''
        Generate a random grid according to the parameter puddle_location and
        possible_puddle_location
        '''

        grid = [['' for i in np.arange(self.size[0])] for i in np.arange(self.size[1])]

        dim = 4 + len(self.puddle_location[0]) * len(self.puddle_location[1]) * 2
        w = np.array([0]*dim)

        idx_h = np.random.randint(len(self.possible_puddle_location))
        pos_puddle_horizontal = self.possible_puddle_location[idx_h]

        idx_v = np.random.randint(len(self.possible_puddle_location))
        pos_puddle_vertical = self.possible_puddle_location[idx_v]
        w[4+idx_h] = -1
        w[4+len(self.possible_puddle_location)+idx_v] = -1

        i_horz = pos_puddle_horizontal[0]
        j_horz = pos_puddle_horizontal[1]
        grid[i_horz][j_horz] = -1
        grid[i_horz][j_horz+1] = -1
        grid[i_horz][j_horz-1] = -1


        i_vert = pos_puddle_vertical[0]
        j_vert = pos_puddle_vertical[1]
        grid[i_vert][j_vert] = -1
        grid[i_vert+1][j_vert] = -1
        grid[i_vert-1][j_vert] = -1

        goal = np.random.randint(4)
        w[goal] = 1
        goal = {0:[0,0], 1:[0, self.size[0]-1], 2:[self.size[1]-1,0], 3:[self.size[1]-1, self.size[0]-1]}.get(goal)
        grid[goal[0]][goal[1]] = 1
        self.grid = GridWorld(gamma=self.gamma, grid=grid, render=False)

        return self.grid, w

    def get_possible_puddle_location(self):
        return self.possible_puddle_location

    def init_phi(self):
        '''
        Return phi :
        for each couple (state, action) which lead to a puddle or a goal
        (a reward not null), phi(state, action) is a one-hot vector where
        the one correspond to the index of the goal or the puddle
        '''
        size_r = self.size[0]
        size_c = self.size[1]

        nb_possible = len(self.possible_puddle_location)
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


        for idx,coord in enumerate(self.possible_puddle_location):
            # For the horizontal puddle
            state = coord[0]*size_c + coord[1]
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

        self.phi = phi
        return phi

    def psi_learning(self, env, psi, epsilon, N, lrn_rate = 0.02, Tmax = 50, render=True, view_end = True, a_seed =10, b_seed=302):
        """
        Args:
                env (GridWorld): The grid where the algorithm will be applied
                psi ([[floa]]): Successor feature
                epsilon (int) : exploration rate, the probability to take a random path
                N (int) : number of samples
                Tmax (int) : the limite of transitions (episodic)

        Returns:
                psi ([[float]]) : updated successor features
                pol (Policy object) : optimal policy according to the psi-learning
                V ([[float]]) : Values computed during the algorithm ??
                w_stock ([[float]]) : list successive value of w
        """
        phi = self.phi
        gamma = env.gamma
        # initialize a policy
        size = self.size
        # pol = [1]*(size[1]) + ([0]*(size[1]-1) + [1])*(size[0]-2) + [0]*(size[1]-1) + [2]
        pol = Policy(env)
        t = 1
        alpha = []
        for i1,i2 in enumerate(env.state_actions) :
            alpha.append([])
            for j1, j2 in enumerate(i2):
                alpha[i1].append(0.5)

        # Null or random initialiation? EB
        w = np.zeros(len(phi[0][0]))
        w_stock = []
        rewards = []

        if(render):
            rang = tqdm(range(N), desc="Psi-learning")
        else:
            rang = range(N)
        for n in rang:
            np.random.seed(a_seed * n + b_seed)
            state = env.reset()
            t_lim = 0
            absorbing = False
            # show the last episodes of a round
            if view_end == True:
                if(n == N-1):
                     env.activate_render(color = 'blue')
            while(not absorbing and t_lim < Tmax):

                greedy = np.random.rand(1) > epsilon

                action = pol.get_action(state) if greedy else np.random.choice(env.state_actions[state])

                # To update alpha
                idx_action = env.state_actions[state].index(action)

                q_tmp = []
                prev_state = state
                state, reward, absorbing = env.step(state, action)

                # Compute the next expected Q-values
                for idx, new_action in enumerate(env.state_actions[state]):
                    q_tmp.append(np.dot(psi[state][new_action], w))

                q_tmp = np.array(q_tmp)
                # Select the best action (random among the maximum) for the next step
                idx_env = np.random.choice(np.flatnonzero(q_tmp == q_tmp.max()))
                best_q = q_tmp[idx_env]

                # Update the policy
                pol.update_action(state, env.state_actions[state][idx_env])

                # Update Psi, the successor feature
                TD_phi = phi[prev_state][action] + gamma*psi[state][pol.get_action(state)] - psi[prev_state][action]
                psi[prev_state][action] = psi[prev_state][action] + alpha[prev_state][idx_action] * TD_phi

                # Update w by gradient descent
                err = np.dot(phi[prev_state][action], w) - reward
                w = w - lrn_rate * phi[prev_state][action] * err / np.log(n+2) # smoothing convergence?
                # the term np.log(n+2) allow to smooth the gradient descent. The smoothing
                # with O(1/n) is a bit too rude, where with a log-smoothing, the convergence is better. EB

                # Update the value fonction ??
                # v = []
                # for i in np.arange(env.n_states):
                #     idx = env.state_actions[i].index(pol[i])
                #     v.append(np.dot(psi[i][pol[i]],w))
                # V.append(v)

                alpha[prev_state][idx_action] = 1./((1/alpha[prev_state][idx_action]) + 1.)
                t_lim += 1
            rewards.append(reward * gamma**(t_lim-1))
            w_stock.append(w)
        if view_end == True:
            env.deactivate_render()
        return psi, pol, rewards, w_stock


    def q_learning(self, env, epsilon, N, Tmax = 50, render=False, view_end = True, a_seed =10, b_seed=302) :
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

        alpha = []
        for i1,i2 in enumerate(env.state_actions) :
            alpha.append([])
            for j1, j2 in enumerate(i2):
                alpha[i1].append(0.5)
        pol = Policy(env)
        q = np.zeros((env.n_states, 4))
        V = []
        rewards=[]
        if(render):
            rang = tqdm(range(N), desc="Q-learning")
        else:
            rang = range(N)
        for n in rang:
            np.random.seed(a_seed * n + b_seed)
            state = env.reset()
            t_lim = 0
            # show the last episodes of a round
            if view_end == True:
                if(n == N-1):
                     env.activate_render(color = 'red')
            absorbing = False
            while(not absorbing and t_lim < Tmax):

                greedy = np.random.rand(1) > epsilon
                action = pol.get_action(state) if greedy else np.random.choice(env.state_actions[state])
                idx_action = env.state_actions[state].index(action)

                q_tmp = []
                prev_state = state
                state, reward, absorbing = env.step(state, action)

                for idx, new_action in enumerate(env.state_actions[state]):
                    q_tmp.append(q[state][new_action])

                idx_best = np.argmax(q_tmp)
                pol.update_action(state, env.state_actions[state][idx_best])
                best_q = np.max(np.array(q_tmp))

                TD = reward + gamma*best_q - q[prev_state][idx_action]

                # Update Q, the Q-function
                q[prev_state][idx_action] = q[prev_state][idx_action] + alpha[prev_state][idx_action] * TD

                # Update the vaue fonction
                # v = []
                # for i in np.arange(env.n_states):
                #     idx = env.state_actions[i].index(pol[i])
                #     v.append(q[i][idx])
                # V.append(v)

                alpha[prev_state][idx_action] = 1/((1/alpha[prev_state][idx_action]) + 1)
                t_lim += 1
            rewards.append(reward * gamma**(t_lim-1))
        if view_end == True:
            env.deactivate_render()
        return q, pol, rewards
