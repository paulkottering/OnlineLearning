import numpy as np
from numpy import random as rand
from utils.updates import opt_pes_recalc


class game():

    def __init__(self,n,k):

        shape = [n]*k
        self.Potential = rand.randint(-12500, 12501, shape) / 100000

        unknown_utilitys = [np.zeros(shape) for i in range(k)]

        for p in range(k):
            for i in range(1,n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i-1] + [Ellipsis]

                unknown_utilitys[p][tuple(rel_slice)] = unknown_utilitys[p][tuple(prev_slice)] + self.Potential[tuple(rel_slice)] - self.Potential[tuple(prev_slice)]

        for p in range(k):
            unknown_utilitys[p] += 0.25

        self.UnknownUs = unknown_utilitys

        Unknown_Game, Unknown_Game_Pes = opt_pes_recalc(unknown_utilitys,unknown_utilitys,n,k)

        self.UnknownGame = Unknown_Game
        print(Unknown_Game)
        self.n = n
        self.k = k

        self.PhiMax = np.max(Unknown_Game)
        self.PhiMin = np.min(Unknown_Game)

        self.KnownUs = [np.full(shape, np.nan) for i in range(k)]

        self.OptPhi = np.ones(shape)*np.inf
        self.PesPhi = np.ones(shape)*-np.inf

        self.OptUs = [np.ones(shape) * 0.5 for i in range(k)]
        self.PesUs = [np.ones(shape) * 0 for i in range(k)]

        self.number_samples = 0

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
            self.OptPhi, self.PesPhi = opt_pes_recalc(self.OptUs,self.PesUs,self.n,self.k)


    def OptPesUUpdate(self,p,sample_tuple):

        slices = [sample_tuple[i] if i != p else slice(None) for i in range(self.k)]

        upper = self.OptUs[p][sample_tuple] +self.PhiMax - self.PhiMin
        lower = self.PesUs[p][sample_tuple] +self.PhiMin - self.PhiMax

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

