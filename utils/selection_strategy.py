import numpy as np

def select_index(Game,i):
    n = Game.n
    if i == "o":
        return np.unravel_index(np.argmax(Game.OptPhi, axis=None),(n,n))
    if i == "p":
        return np.unravel_index(np.argmax(Game.PesPhi, axis=None), (n, n))

