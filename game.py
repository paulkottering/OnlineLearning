import numpy as np
from numpy import random as rand
from utils.updates import opt_pes_recalc_2, opt_pes_recalc, opt_pes_recalc_make
import matplotlib.pyplot as plt

class game():

    def __init__(self,n,k):

        shape = [n]*k
        #self.Potential = rand.randint(-12500, 12501, shape) / 100000
        self.Potential = np.zeros(shape)
        self.Potential[1,2,3] = 0.125

        unknown_utilitys = [np.zeros(shape) for i in range(k)]

        for p in range(k):
            for i in range(1,n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i-1] + [Ellipsis]

                unknown_utilitys[p][tuple(rel_slice)] = unknown_utilitys[p][tuple(prev_slice)] + self.Potential[tuple(rel_slice)] - self.Potential[tuple(prev_slice)]

        for p in range(k):
            unknown_utilitys[p] += 0.25

        self.UnknownUs = unknown_utilitys

        self.matrices = opt_pes_recalc_make(n, k)

        unknown_game, unknown_game_pes, diff = opt_pes_recalc_2(self.matrices, unknown_utilitys, unknown_utilitys, n,k)

        self.UnknownGame = unknown_game

        self.n = n
        self.k = k

        self.PhiMax = np.max(unknown_game)
        self.PhiMin = np.min(unknown_game)

        self.KnownUs = [np.full(shape, np.nan) for i in range(k)]

        self.OptPhi = np.ones(shape)*np.inf
        self.PesPhi = np.ones(shape)*-np.inf

        self.OptUs = [np.ones(shape) * np.max(unknown_utilitys) for i in range(k)]
        self.PesUs = [np.ones(shape) * np.min(unknown_utilitys) for i in range(k)]

        self.possible = np.zeros(shape)
        self.active = np.ones(shape)
        dtypes = [('f{}'.format(i), float) for i in range(self.k)]
        self.Heads = np.full(shape, np.nan, dtype=dtypes)
        self.Values = np.zeros(shape)

        self.number_samples = 0
        self.diff = []

        self.matrices = opt_pes_recalc_make(n, k)

    def sample(self,sample_tuple):
        if np.isnan(self.KnownUs[0][sample_tuple]):

            self.number_samples += 1

            # Sample
            for p in range(self.k):
                u_val = self.UnknownUs[p][sample_tuple]
                self.KnownUs[p][sample_tuple] = u_val
                self.OptUs[p][sample_tuple] = u_val
                self.PesUs[p][sample_tuple] = u_val

            # Update OptU1, PesU1, OptU2, PesU2
            for p in range(self.k):
                self.OptPesUUpdate(p, sample_tuple)

            # Update OptYP2,PesYP2
            self.OptPhi, self.PesPhi, diff = opt_pes_recalc_2(self.matrices, self.OptUs, self.PesUs, self.n, self.k)
            self.diff.append(diff)
            self.update_h_v(sample_tuple)

        plt.plot(self.diff)
        plt.show()

    def OptPesUUpdate(self,p,sample_tuple):

        slices = [sample_tuple[i] if i != p else slice(None) for i in range(self.k)]

        upper = self.OptUs[p][sample_tuple] + self.PhiMax - self.PhiMin
        lower = self.PesUs[p][sample_tuple] + self.PhiMin - self.PhiMax

        self.OptUs[p][tuple(slices)] = np.minimum(self.OptUs[p][tuple(slices)],upper)
        self.PesUs[p][tuple(slices)] = np.maximum(self.PesUs[p][tuple(slices)],lower)

    def check_bounds(self):

        if np.any(self.OptPhi < self.UnknownGame):
            print("OptPhi")

        if np.any(self.PesPhi > self.UnknownGame):
            print("PesPhi")

        for p in range(self.k):
            if np.any(self.OptUs[p] < self.UnknownUs[p]):
                print(p,"Opt")
            if np.any(self.PesUs[p] > self.UnknownUs[p]):
                print(p,"Pes")

    def check_end(self):
        if np.max(self.OptPhi) != np.max(self.UnknownGame):
            print("Opt Max != Potential Max")

    def update_active(self):
        mask = self.OptPhi < np.max(self.PesPhi)
        mask2 = self.Values < 0
        self.active[mask] = 0
        self.active[mask2] = 0

    def update_h_v(self, sample_tuple):
        dtypes = [('f{}'.format(i), float) for i in range(self.k)]
        heads = np.full(self.k,np.inf,dtype=dtypes)
        pot_new_val = np.ones(self.k) * np.nan

        for p in range(self.k):

            array = self.KnownUs[0][tuple(slice(None) if i == p else idx for i, idx in enumerate(sample_tuple))]

            i = 0
            cont = True
            while i < len(array) and cont:
                if ~np.isnan(array[i]) and (i != sample_tuple[p]):
                    ind_tuple = np.copy(sample_tuple)
                    ind_tuple[p] = i
                    ind_tuple = tuple(ind_tuple)
                    print(self.KnownUs[p][sample_tuple])
                    print(sample_tuple)
                    print(ind_tuple)
                    print(self.KnownUs[p][ind_tuple])
                    connection = self.KnownUs[p][sample_tuple] - self.KnownUs[p][ind_tuple]
                    val = self.Values[ind_tuple] + connection
                    heads[p] = self.Heads[ind_tuple]
                    pot_new_val[p] = val
                    cont = False
                i += 1

        u,indices = np.unique(heads,return_index=True)

        if u[-1][0] == np.inf:
            num_non_nan_unique = len(u)-1
            u = u[:-1]
            indices = indices[:-1]
        else:
            num_non_nan_unique = len(u)
        # Make new branch, val = 0, head = itself
        if num_non_nan_unique == 0:
            self.Heads[sample_tuple] = sample_tuple
            self.Values[sample_tuple] = 0

        # Add to branch, maybe updated all values if it is greater than 0, and update all heads in this case too
        if num_non_nan_unique == 1:
            new_val_p = indices[0]
            value = pot_new_val[new_val_p]
            if value <= 0:
                self.Heads[sample_tuple] = heads[new_val_p]
                self.Values[sample_tuple] = value
            if value > 0:

                old_head = heads[new_val_p]

                self.Heads[sample_tuple] = sample_tuple
                self.Values[sample_tuple] = 0

                mask = self.Heads == old_head
                self.Heads[mask] = sample_tuple
                self.Values[mask] -= value


        # Connect two branches suing new point. Updating heads, values and everything
        if num_non_nan_unique > 1:

            unique_indices = indices
            unique_heads = u
            unique_values = pot_new_val[unique_indices]

            min_new_val_p_in_unique = np.argmin(unique_values)
            new_val = unique_values[min_new_val_p_in_unique]

            if new_val > 0:
                self.Heads[sample_tuple] = sample_tuple
                self.Values[sample_tuple] = 0
                for i in range(len(unique_heads)):
                    if i != min_new_val_p_in_unique:
                        old_head = unique_heads[i]

                        mask = self.Heads == old_head
                        self.Heads[mask] = sample_tuple
                        self.Values[mask] -= unique_values[i]
            else:
                new_head = unique_heads[min_new_val_p_in_unique]
                self.Heads[sample_tuple] = new_head
                self.Values[sample_tuple] = unique_values[min_new_val_p_in_unique]

                for i in range(len(unique_heads)):
                    if i != min_new_val_p_in_unique:
                        old_head = unique_heads[i]

                        diff_in_val = unique_values[min_new_val_p_in_unique] - unique_values[i]

                        mask = self.Heads == old_head
                        self.Heads[mask] = new_head
                        self.Values[mask] += diff_in_val
