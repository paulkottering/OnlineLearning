import numpy as np
from numpy import random as rand

def initialize_game(Game,i):
    n = Game.n

    if i == "d":
        for k in range(n):
            Game.sample(k, k)

    if i == "o":
        Game.sample(rand.random(n), rand.random(n))

