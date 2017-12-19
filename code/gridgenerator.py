import numpy as np
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


    def create_Grid(self):

        grid = [['' for i in np.arange(self.size[0])] for i in np.arange(self.size[1])]

        dim = 4 + len(self.puddle_location[0]) * len(self.puddle_location[1]) * 2
        w = np.array([0]*dim)

        pos_puddle1 = [np.random.randint(len(self.puddle_location[0])), np.random.randint(len(self.puddle_location[1]))]
        pos_puddle2 = [np.random.randint(len(self.puddle_location[0])), np.random.randint(len(self.puddle_location[1]))]
        grid[self.puddle_location[0][pos_puddle1[0]]][self.puddle_location[1][pos_puddle1[1]]] = -1
        grid[self.puddle_location[0][pos_puddle2[0]]][self.puddle_location[1][pos_puddle2[1]]] = -1

        w[4 + len(self.puddle_location[0])*pos_puddle1[0] + pos_puddle1[1]] = -1
        w[4 + len(self.puddle_location[0])*len(self.puddle_location[1]) + len(self.puddle_location[0])*pos_puddle2[0] + pos_puddle2[1]] = -1

        goal = np.random.randint(4)
        w[goal] = 1
        goal = {0:[0,0], 1:[0, self.size[0]-1], 2:[self.size[1]-1,0], 3:[self.size[1]-1, self.size[0]-1]}.get(goal)
        grid[goal[0]][goal[1]] = 1
        self.grid = grid



        return grid, w



