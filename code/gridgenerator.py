import numpy as np
import itertools
from gridworld import *



class GridGenerator:

    def __init__(self, size=[7,7],
        nb_puddle=2,
        #puddle_location=[[3,5,7,9,11], [3,5,7,9,11]],
        puddle_location=[[3,5], [3,5]],
        seed=None):

        if(seed!=None):
            np.random.seed(seed)

        self.seed = seed
        self.size = size
        self.nb_puddle = nb_puddle
        self.puddle_location = puddle_location
        # all possible location of the puddle
        self.possible_puddle_location = list(itertools.product(puddle_location[0], puddle_location[1]))
        self.grid=None

    def create_Grid(self):

        grid = [['' for i in np.arange(self.size[0])] for i in np.arange(self.size[1])]

        dim = 4 + len(self.puddle_location[0]) * len(self.puddle_location[1]) * 2
        w = np.array([0]*dim)

        idx_h = np.random.randint(len(self.possible_puddle_location))
        pos_puddle_horizontal = self.possible_puddle_location[idx_h]
        print(pos_puddle_horizontal)
        print(self.puddle_location)

        idx_v = np.random.randint(len(self.possible_puddle_location))
        pos_puddle_vertical = self.possible_puddle_location[idx_v]
        w[4+idx_h] = -1
        w[4+len(self.possible_puddle_location)+idx_v] = -1
        # print("pos_puddle_horizontal {}".format(pos_puddle_horizontal))
        # print("pos_puddle_vertical {}".format(pos_puddle_vertical))

        # print("w = {}".format(w))
        # print("idx_h : {} et idx_v : {}".format(idx_h, idx_v))

        # pos_puddle_horizontal = [np.random.randint(len(self.puddle_location[0])), np.random.randint(len(self.puddle_location[1]))]
        # pos_puddle_vertical = [np.random.randint(len(self.puddle_location[0])), np.random.randint(len(self.puddle_location[1]))]

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



        # w[4 + len(self.puddle_location[0])*pos_puddle_horizontal[0] + pos_puddle_horizontal[1]] = -1
        # w[4 + len(self.puddle_location[0])*len(self.puddle_location[1]) + len(self.puddle_location[0])*pos_puddle_vertical[0] + pos_puddle_vertical[1]] = -1

        goal = np.random.randint(4)
        w[goal] = 1
        goal = {0:[0,0], 1:[0, self.size[0]-1], 2:[self.size[1]-1,0], 3:[self.size[1]-1, self.size[0]-1]}.get(goal)
        grid[goal[0]][goal[1]] = 1
        self.grid = grid

        return grid, w

    def get_possible_puddle_location(self):
        return self.possible_puddle_location

    # def get_phi(self):



