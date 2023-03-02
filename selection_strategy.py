import numpy as np

def select_index(Game,i):
    if i == "d":
        return descending_active(Game)
    if i == "ra":
        return random_active(Game)
    if i == "nr":
        return nearby_then_random(Game)
    if i == "r":
        return random_index(Game)
    if i == "or":
        return opt_then_random(Game)

def descending_active(game):

    reverse_ascending_order = np.argsort(game.OptPhi - game.PesPhi, axis=None)[::-1]

    for i in range(len(reverse_ascending_order)):
        index = reverse_ascending_order[i]
        row, col = np.unravel_index(index, game.OptPhi.shape)
        if game.OptPhi[row, col] > np.max(game.PesPhi):
            if np.isnan(game.OptPhi[row, col]):
                return row, col

    if np.any(np.isnan(game.KnownU1)):
        rand_ind1 = np.argwhere(np.isnan(game.KnownU1))[0][0]
        rand_ind2 = np.argwhere(np.isnan(game.KnownU1))[0][1]
        return rand_ind1, rand_ind2

def random_active(game):

    ind1, ind2 = np.unravel_index(np.argmax(game.OptPhi, axis=None), (game.n, game.n))

    if np.isnan(game.KnownU1[ind1, ind2]):
        return ind1, ind2

    nan_new_act = []
    new_act = []
    indices = []
    for index in np.ndindex((game.n,game.n)):
        indices.append(list(index))

    for ind in indices:
        if game.OptPhi[ind[0], ind[1]] >= np.max(game.PesPhi):
            if np.isnan(game.KnownU1[ind[0], ind[1]]):
                nan_new_act.append(ind)
            new_act.append(ind)

    if len(nan_new_act) > 1:
        nan_active_indices = np.array(nan_new_act)
        rand_active_ind = np.random.choice(range(len(nan_active_indices)), size=1, replace=False)

        rand_active_ind1 = nan_active_indices[rand_active_ind][0][0]
        rand_active_ind2 = nan_active_indices[rand_active_ind][0][1]
        return rand_active_ind1, rand_active_ind2

    if np.any(np.isnan(game.KnownU1)):
        rand_ind1 = np.argwhere(np.isnan(game.KnownU1))[0][0]
        rand_ind2 = np.argwhere(np.isnan(game.KnownU1))[0][1]
        return rand_ind1, rand_ind2

def nearby_then_random(game):

    ind1, ind2 = np.unravel_index(np.argmax(game.OptPhi, axis=None), (game.n, game.n))

    if np.isnan(game.KnownU1[ind1, ind2]):
        return ind1, ind2

    if 0 <= ind2-1 :
        if np.isnan(game.KnownU1[ind1,ind2-1]):
            return ind1,ind2-1

    if ind2+1 < game.n:
        if np.isnan(game.KnownU1[ind1,ind2+1]):
            return ind1,ind2+1

    if ind1+1 < game.n:
        if np.isnan(game.KnownU1[ind1+1, ind2]):
            return ind1+1, ind2

    if 0 <= ind1-1 :
        if np.isnan(game.KnownU1[ind1-1, ind2]):
            return ind1-1, ind2

    if np.any(np.isnan(game.KnownU1)):
        rand_ind1 = np.argwhere(np.isnan(game.KnownU1))[0][0]
        rand_ind2 = np.argwhere(np.isnan(game.KnownU1))[0][1]
        return rand_ind1, rand_ind2

def random_index(game):
    if np.any(np.isnan(game.KnownU1)):
        rand_ind1 = np.argwhere(np.isnan(game.KnownU1))[0][0]
        rand_ind2 = np.argwhere(np.isnan(game.KnownU1))[0][1]
        return rand_ind1, rand_ind2

def opt_then_random(game):
    ind1, ind2 = np.unravel_index(np.argmax(game.OptPhi, axis=None), (game.n, game.n))

    if np.isnan(game.KnownU1[ind1, ind2]):
        return ind1, ind2

    if np.any(np.isnan(game.KnownU1)):
        rand_ind1 = np.argwhere(np.isnan(game.KnownU1))[0][0]
        rand_ind2 = np.argwhere(np.isnan(game.KnownU1))[0][1]
        return rand_ind1, rand_ind2