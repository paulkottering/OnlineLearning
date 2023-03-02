import numpy as np
from numpy import random as rand
from utils.updates import OptPesYPRecalc,OptPesY1Recalc,OptPesY2Recalc


class game():

    def __init__(self,n):

        self.Potential = rand.randint(-12500, 12501, (n, n)) / 100000

        unknown_utility_two = np.zeros((n, n))
        unknown_utility_one = np.zeros((n, n))

        unknown_utility_two[:, -1] = 0
        unknown_utility_one[-1, :] = 0

        for i in range(n - 1):
            unknown_utility_two[:, -2 - i] = unknown_utility_two[:, -1 - i] + self.Potential[:, -2 - i] - self.Potential[:, -1 - i]
            unknown_utility_one[-2 - i, :] = unknown_utility_one[-1 - i, :] + self.Potential[-2 - i, :] - self.Potential[-1 - i]

        unknown_utility_one += 0.25
        unknown_utility_two += 0.25

        self.UnknownU1 = unknown_utility_one
        self.UnknownU2 = unknown_utility_two

        Phi = np.eye(n) - 1 / n * np.ones((n, n))
        Xi = 1 / n * np.ones((n, n))

        UnknownY1 = np.matmul(Phi, np.matmul(self.UnknownU1, Xi))
        UnknownY2 = np.matmul(Xi, np.matmul(self.UnknownU2, Phi))
        UnknownYP = np.matmul(Phi, np.matmul(self.UnknownU1, Phi))

        Unknown_Game = UnknownYP + UnknownY1 + UnknownY2

        self.UnknownGame = Unknown_Game
        self.n = n

        self.PhiMax = np.max(Unknown_Game)
        self.PhiMin = np.min(Unknown_Game)

        self.KnownU2 = np.full((n, n), np.nan)
        self.KnownU1 = np.full((n, n), np.nan)

        self.OptPhi = np.ones((n,n))*(self.PhiMax+2)
        self.PesPhi = np.ones((n,n))*(self.PhiMin-2)

        self.OptU1 = np.ones((n,n)) * 0.5
        self.PesU1 = np.ones((n, n)) * 0
        self.OptU2 = np.ones((n, n)) * 0.5
        self.PesU2 = np.ones((n, n)) * 0

        self.number_samples = 0

    def sample(self,i,j):

        if np.isnan(self.KnownU1[i, j]):

            self.number_samples += 1

            # Sample UnknownU1 and Unknown U2
            U1Val = self.UnknownU1[i, j]
            U2Val = self.UnknownU2[i, j]

            # Update KnownU1 and KnownU2
            self.KnownU1[i, j] = U1Val
            self.KnownU2[i, j] = U2Val

            # Update Samples in OptU1, PesU1, OptU2, PesU2
            self.OptU1[i, j] = U1Val
            self.PesU1[i, j] = U1Val

            self.OptU2[i, j] = U2Val
            self.PesU2[i, j] = U2Val

            # Update OptU1, PesU1, OptU2, PesU2
            self.OptPesU1Update(i,j)
            self.OptPesU2Update(i,j)

            # Update OptY1,PesY1
            OptY1, PesY1 = OptPesY1Recalc(self.OptU1,self.PesU1,self.n)
            OptY2, PesY2 = OptPesY2Recalc(self.OptU2,self.PesU2,self.n)

            # Recalculate OptYP1,PesYP1
            OptYP1, OptYP2 = OptPesYPRecalc(self.OptU1,self.PesU1,self.n)

            # Update OptYP2,PesYP2
            PesYP1, PesYP2 = OptPesYPRecalc(self.OptU2,self.PesU2,self.n)

            OptYP = np.minimum(OptYP1, OptYP2)
            PesYP = np.minimum(PesYP1, PesYP2)

            # Optimistic potential matrix estimate
            self.OptPhi = OptYP + np.array([OptY1] * self.n).T + np.array([OptY2] * self.n)

            # Pessimistic potential matrix estimate
            self.PesPhi = PesYP + np.array([PesY1] * self.n).T + np.array([PesY2] * self.n)

    def initial_samples(self):
        for i in range(self.n):
            self.sample(i,i)

    def OptPesU2Update(self,a,b):

        for j in range(self.n):
            if j != b:
                OptRange = np.minimum(self.PhiMax, self.OptPhi[a, j]) - np.maximum(self.PesPhi[a, b], self.PhiMin)
                PesRange = np.maximum(self.PhiMin, self.PesPhi[a, j]) - np.minimum(self.OptPhi[a, b], self.PhiMax)

                # if PesU2[a, b] + PhiMin - PhiMax > UnknownGame[a,j]:
                #     print(PesU2[a, b] + PhiMin - PhiMax - UnknownGame[a,j])
                #     print('Miss5')
                OptRange = self.PhiMax - self.PhiMin
                PesRange = - self.PhiMax + self.PhiMin

                self.OptU2[a, j] = np.minimum(self.OptU2[a, j], self.OptU2[a, b] + OptRange)
                self.PesU2[a, j] = np.maximum(self.PesU2[a, j], self.PesU2[a, b] + PesRange)

    def OptPesU1Update(self,a,b):

        for i in range(self.n):
            if i != a:
                OptRange = np.minimum(self.PhiMax, self.OptPhi[i, b]) - np.maximum(self.PesPhi[a, b], self.PhiMin)
                PesRange = np.maximum(self.PhiMin, self.PesPhi[i, b]) - np.minimum(self.OptPhi[a, b], self.PhiMax)

                OptRange = self.PhiMax - self.PhiMin
                PesRange = - self.PhiMax + self.PhiMin

                self.OptU1[i, b] = np.minimum(self.OptU1[i, b], self.OptU1[a, b] + OptRange)
                self.PesU1[i, b] = np.maximum(self.PesU1[i, b], self.PesU1[a, b] + PesRange)

    def check_bounds(self):

        if np.any(self.OptPhi < self.UnknownGame):
            print("OptPhi")

        if np.any(self.PesPhi > self.UnknownGame):
            print("PesPhi")

        if np.any(self.OptU1 < self.UnknownU1):
            print("OptU1")

        if np.any(self.PesU1 > self.UnknownU1):
            print("PesU1")

        if np.any(self.OptU2 < self.UnknownU2):
            print("OptU2")

        if np.any(self.PesU2 > self.UnknownU2):
            print("PesU2")
