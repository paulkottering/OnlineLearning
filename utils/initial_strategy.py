import numpy as np
from numpy import random as rand

def initialize_game(Game,i):
    n = Game.n

    if i == "d":
        for k in range(n):
            Game.sample(k, k)

    if i == "o":
        n = Game.n
        k = Game.k
        zero = np.ones(k).astype(int)
        Game.sample(tuple(zero))

