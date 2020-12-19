#-*- coding:utf-8 -*-

import random
import gym
import numpy as np
import pygame
from copy import copy
import time

class GameClass(gym.Env):
    def __init__(self):
        self.env=Board(4,4)
        self.action_space=gym.spaces.Discrete(4)
        self.env.new_tile()
        self.env.new_tile()
        sample=[]
        for i in range(4):
            t=[]
            for j in range(4):
                t.append(self.env.grid[i*4+j])
            sample.append(t)
        sample=np.array(sample)

        self.observation_space=gym.spaces.Box(low=0,high=100,shape=sample.shape)
        self.reward_range=[-1,10]

        pygame.init()
        self.screen=pygame.display.set_mode((400, 400))
        pygame.display.set_caption("2048 AI")
        self.font = pygame.font.Font(None, 50)
        self.TEST=False


    def reset(self):
        self.env=Board(4,4)
        self.env.new_tile()
        self.env.new_tile()


        state=[]
        for i in range(4):
            t=[]
            for j in range(4):
                t.append(self.env.grid[i*4+j])
            state.append(t)
        state=np.array(state)
        return state

    def step(self, action):
        act=["up","down","left","right"]
        old=copy(self.env.grid)


        numbers_old=16-self.env.grid.count(0)

        self.env.move(act[action])


        #報酬計算
        self.reward=0

        numbers=16-self.env.grid.count(0)
        self.reward+=(numbers_old-numbers)*2048 #消せば消すほど得点

        #端は倍率高めにして総和
        for i in range(4):
            for j in range(4):
                if i==0:
                    self.reward+=2**self.env.grid[i*4+j]

                    if j!=0 and self.env.grid[j]*2==self.env.grid[j-1]:
                        self.reward+=2048

                elif i==4:
                    self.reward+=2**(self.env.grid[i*4+j])/3
                else:
                    self.reward+=2**(self.env.grid[i*4+j])/2

        self.reward/=32768

        #報酬計算終わり!


        if old==self.env.grid:
            done=True
            if self.env.grid.count(0)!=0:
                self.reward=-1
        else:
            done=False
        
        if old!=self.env.grid:
            self.env.new_tile()

        state=[]
        for i in range(4):
            t=[]
            for j in range(4):
                t.append(self.env.grid[i*4+j])
            state.append(t)
        state=np.array(state)

        info={}
        info["max_value"]=max(self.env.grid)

        return state,self.reward,done,info


    def render(self, mode="Human"):
        self.screen.fill((255,255,255))

        text = self.font.render(str(2**max(self.env.grid)), True, (0,0,0))
        self.screen.blit(text, [50, 50])

        text = self.font.render("{:10f}".format(self.reward), True, (0,0,0))
        self.screen.blit(text, [50, 100])
        
        for i in range(4):
            for j in range(4):
                text = self.font.render(str(2**self.env.grid[i*4+j]), True, (0,0,0))
                self.screen.blit(text, [100+j*50, 150+i*50])

        pygame.display.update()
        if self.TEST:
            time.sleep(1/5)
        return 

    def auto(self):

        mx=0
        ans=1
        for i in range(4):
            base=copy(self)
            base.step(i)
            
            if(base.reward>mx):
                mx=base.reward
                ans=i

        return self.step(ans)
            
            


class Board:
    def __init__(self, _width, _height, grid=None):
        self.width = _width
        self.height = _height

        # The grid is a list of numbers representing the values of tiles. Each number 
        # corresponds to the power of two of the given tile, i.e. a value of 3 stored
        # in the grid corresponds to a game tile with the value 2 ^ 3, or 8. 
        #
        # If an argument is passed in for the grid, set it to that, otherwise default 
        # to an empty grid.
        if (grid == None):
            self.grid = []
            for _ in range(_width * _height):
                self.grid.append(0)
        else:
            self.grid = grid

        self.score = 0


    # @property
    # def score(self):
    #     return self._score

    
    # @property
    # def grid(self):
    #     return self._grid


    # @property
    # def width(self):
    #     return self._width


    # @property
    # def height(self):
    #     return self._height


    def new_tile(self):
        # Find the list indices of the free (value of 0) spaces in the grid
        candidates = []
        for i in range(len(self.grid)):
            if self.grid[i] == 0:
                candidates.append(i)

        if len(candidates) > 0:
            # Generate a value for the new tile, choosing a higher value 10% of the time.
            value = 1
            if random.randint(1, 10) == 10:
                value = 2

            # Choose a random grid index from the generated list and set its
            # value to the value generated before.
            self.grid[random.choice(candidates)] = value


    def __set(self, x, y, value):
        self.grid[y * self.height + x] = value


    def get(self, x, y):
        return self.grid[y * self.width + x]


    def __get_range(self, position, direction):
        r = []
        if direction == 'left' or direction == 'right':
            for x in range(self.width):
                r.append(self.get(x, position))
        else:
            for y in range(self.height):
                r.append(self.get(position, y))
        return r


    def __set_range(self, row, position, direction):
        if direction == 'left' or direction == 'right':
            for x in range(self.width):
                self.__set(x, position, row[x])
        else:
            for y in range(self.height):
                self.__set(position, y, row[y])


    def __compress(self, row, direction):
        new_row = []
        zeroes = []
        for value in row:
            if value != 0:
                new_row.append(value)
            else:
                zeroes.append(0)
        if direction == 'left' or direction == 'up':
            return new_row + zeroes
        else:
            return zeroes + new_row

    
    def __combine(self, row):
        for i in range(len(row) - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] += 1
                row[i + 1] = 0
                # Every time you combine two tiles, the value of the new tile is added to your score.
                self.score += 2 ** row[i]
        return row


    def move(self, direction):
        """
        Move in the specified direction.

        This will shift all tiles as far as possible in the given
        direction, combining adjacent tiles of the same value into a single tile double the value.

         This method takes a string argument that must be 'left', 'right', 'up', or 'down', and returns
        True if the grid has changed after the move or false otherwise.
        """
        # Make a copy of the current state of the grid to compare to later
        old_grid = []
        for value in self.grid:
            old_grid.append(value)
        ranges = {
            'left': self.height,
            'right': self.height,
            'up': self.width,
            'down': self.width
        }
        
        if direction in ranges:
            for i in range(ranges[direction]):
                r = self.__get_range(i, direction)
                r = self.__compress(r, direction)
                r = self.__combine(r)
                r = self.__compress(r, direction)
                self.__set_range(r, i, direction)

        # Return true if the current state is different than the old grid
        return not self.equal(old_grid)


    def equal(self, other_grid):
        if len(self.grid) != len(other_grid):
            return False

        for i in range(len(self.grid)):
            if self.grid[i] != other_grid[i]:
                return False

        return True


if __name__=="__main__":
    a=GameClass()
    while True:
        act=int(input())
        a.step(act)
        m=a.env.grid
        for i in range(4):
            for j in range(4):
                print("{} ".format(m[i*4+j]),end="")
            print()