import time
import os,sys
import numpy as np
from selenium.webdriver.common.keys import Keys



class game2048:
    cards = {"tile-" + str(2 ** i): 2 ** i for i in range(1, 12)}
    coordinates = {"tile-position-" + str(i) + '-' + str(j): (j - 1, i - 1) for i in range(1, 5) for j in range(1, 5)}
    UP = Keys.ARROW_UP
    LEFT = Keys.ARROW_LEFT
    DOWN = Keys.ARROW_DOWN
    RIGHT = Keys.ARROW_RIGHT
    keys = [UP, LEFT, DOWN, RIGHT]

def A(state):
    state = np.squeeze(state)
    legal = []
    temp = len(legal)
    identity = 0
    for i in range(1, 4):
        for j in range(0, 4):
            if (state[i, j] != identity and (state[i, j] == state[i - 1, j] or state[i - 1, j] == identity)):
                legal.append(game2048.UP)
                break
            else:
                pass
        if len(legal) == temp + 1:
            break
    temp = len(legal)
    for i in range(0, 4):
        for j in range(1, 4):
            if (state[i, j] != identity and (state[i, j] == state[i, j - 1] or state[i, j - 1] == identity)):
                legal.append(game2048.LEFT)
                break
            else:
                pass
        if len(legal) == temp + 1:
            break
    temp = len(legal)
    for i in range(0, 3):
        for j in range(0, 4):
            if (state[i, j] != identity and (state[i, j] == state[i + 1, j] or state[i + 1, j] == identity)):
                legal.append(game2048.DOWN)
                break
            else:
                pass
        if len(legal) == temp + 1:
            break
    temp = len(legal)
    for i in range(0, 4):
        for j in range(0, 3):
            if (state[i, j] != identity and (state[i, j] == state[i, j + 1] or state[i, j + 1] == identity)):
                legal.append(game2048.RIGHT)
                break
            else:
                pass
        if len(legal) == temp + 1:
            break
    return legal

class Table:
    def __init__(self):
        self.repr = {game2048.keys[i]:np.eye(4)[i] for i in range(0,4)}
        self.inv_repr =  {tuple(v): k for k, v in self.repr.items()}
    def encode(self,action):
        return self.repr[action]
    def decode(self,encoded_action):
        encoded_action = np.squeeze(encoded_action)
        encoded_action = tuple(encoded_action)
        return self.inv_repr[encoded_action]

def phi(s):
    return np.log2(s)