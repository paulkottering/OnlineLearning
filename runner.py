import argparse
import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt

def parse_args():
    """
    Helper function to set up command line argument parsing
    """
    parser = argparse.ArgumentParser()
    # Set up arguments
    parser.add_argument("-s", "--strategynumber", default=5, type=int,
                        help = 'Number of Strategies for each player')
    parser.add_argument("-t", "--timesteps", default=100, type=int,
                        help='Number of timesteps')
    parser.add_argument("-k", "--optimismconstant", default=0.5, type=float,
                        help='Optimistic Constant')
    return parser.parse_args()

def OptPesYPInitCalc(KnownU, n, k):
    """
    Calculate the YP component
    """
    OptYP = np.ones((n, n)) *( ((n - 1) ** 2 / (n ** 2)) * k + ((n-1)**2) / (n ** 2) * k + (2*n-1)*((-n + 1) / (n ** 2)) * (0.5-k))

    PesYP = np.ones((n, n)) *( ((n - 1) ** 2 / (n ** 2)) * (0.5-k) + ((n-1)**2) / (n ** 2) * (0.5-k) + (2*n-2)*((-n + 1) / (n ** 2)) * k)

    # a = 0
    # for b in range(n):
    #     OptYP,PesYP = OptPesYPUpdate(OptYP,PesYP, a, b, KnownU[a,b],n,k)
    #
    # b = 0
    # for a in range(n - 1):
    #     OptYP,PesYP = OptPesYPUpdate(OptYP,PesYP, a+1, b, KnownU[a+1,b],n,k)

    return OptYP,PesYP

def OptPesY1InitCalc(KnownU, n, k):
    """
    Calculate the Optimistic Y1 Component
    """
    OptY1 = np.ones(n) * (((n - 1)*n / (n ** 2)) * k - (n - 1)*n / (n ** 2)*(0.5-k))
    PesY1 = np.ones(n) * (((n - 1)*n / (n ** 2)) *(0.5-k) - (n - 1)*n / (n ** 2)*k)

    # a = 0
    # for b in range(n):
    #     OptY1,PesY1 = OptPesY1Update(OptY1,PesY1, a, b, KnownU[a,b],n,k)
    #
    # b = 0
    # for a in range(n - 1):
    #     OptY1,PesY1 = OptPesY1Update(OptY1,PesY1, a+1, b, KnownU[a+1,b],n,k)

    return OptY1,PesY1

def OptPesY2InitCalc(KnownU, n, k):
    """
    Calculate the Optimistic Y1 Component
    """
    OptY2 = np.ones(n) * (((n - 1) * n / (n ** 2)) * k - (n - 1) * n / (n ** 2) * (0.5 - k))
    PesY2 = np.ones(n) * (((n - 1) * n / (n ** 2)) * (0.5 - k) - (n - 1) * n / (n ** 2) * k)

    # a = 0
    # for b in range(n):
    #     OptY2, PesY2 = OptPesY2Update(OptY2, PesY2, a, b, KnownU[a, b], n, k)
    #
    # b = 0
    # for a in range(n - 1):
    #     OptY2, PesY2 = OptPesY2Update(OptY2, PesY2, a + 1, b, KnownU[a + 1, b], n, k)

    return OptY2, PesY2

def OptPesYPUpdate(OptYP,PesYP, a, b, Val,n,k):

    # Update the YP matrix using the new sample
    for i in range(n):

        const1 = True if i == a else False
        for j in range(n):

            const2 = True if j == b else False

            if (not const1) and (not const2):
                OptYP[i, j] += (Val - k) / (n ** 2)
                PesYP[i, j] += (Val - (0.5 - k)) / (n ** 2)
            elif const1 and const2:
                OptYP[i, j] += (Val - k)*(n - 1) ** 2 / (n ** 2)
                PesYP[i, j] += (Val - (0.5 - k)) * (n - 1) ** 2 / (n ** 2)
            else:
                OptYP[i, j] += (Val - (0.5 - k)) * (-n + 1) / (n ** 2)
                PesYP[i, j] += (Val - k) * (-n + 1) / (n ** 2)
    return OptYP,PesYP

def OptPesY1Update(OptY1,PesY1, a, b, Val,n,k):

    # Update the Y1 matrix using the new sample
    for i in range(n):
        if i == a:
            OptY1[i] += (n - 1) / (n ** 2)*(Val - k)
            PesY1[i] += (n - 1) / (n ** 2)*(Val - (0.5-k))
        else:
            OptY1[i] += (- 1) / (n ** 2)*(Val - (0.5-k))
            PesY1[i] += (- 1) / (n ** 2)*(Val - k)
    return OptY1,PesY1

def OptPesY2Update(OptY2,PesY2, a, b, Val,n,k):
    # Update the Y2 matrix using the new sample
    for j in range(n):
        if j == b:
            OptY2[j] += (n - 1) / (n ** 2)*(Val - k)
            PesY2[j] += (n - 1) / (n ** 2)*(Val - (0.5-k))
        else:
            OptY2[j] += (- 1) / (n ** 2)*(Val - (0.5-k))
            PesY2[j] += (- 1) / (n ** 2)*(Val - k)
    return OptY2,PesY2

def FindNash(game):
    one_max_indices = np.argmax(game, axis=1)
    two_max_indices = np.argmax(game, axis=0)

    NashIndices = []

    n = len(one_max_indices)

    for i in range(n):
        if two_max_indices[one_max_indices[i]] == i:
            NashIndices.append([i,one_max_indices[i]])

    return NashIndices

def OptPesYPRecalc(OptU,PesU, n):

    OptYP = np.ones((n, n))
    PesYP = np.ones((n, n))

    OptMat = OptU
    ColSumOpt = np.matmul(np.ones(n),OptU)
    RowSumOpt = np.matmul(OptU,np.ones(n).T)
    OptSum = np.sum(OptU)

    PesMat =  PesU
    ColSumPes = np.matmul(np.ones(n), PesU)
    RowSumPes = np.matmul(PesU, np.ones(n).T)
    PesSum = np.sum(PesU)

    for i in range(n):
        for j in range(n):
            OtherOpt = (1)/(n**2)*(OptSum - ColSumOpt[j] - RowSumOpt[i] + OptMat[i,j])
            OtherPes = (1) / (n ** 2) * (PesSum - ColSumPes[j] - RowSumPes[i] + PesMat[i, j])

            OptYP[i,j] = ((n-1)**2)/(n**2)*OptMat[i,j] -(n-1)/(n**2)*(ColSumPes[j] - PesMat[i,j])  -(n-1)/(n**2)*(RowSumPes[i]- PesMat[i,j]) + OtherOpt
            PesYP[i, j] = ((n-1)**2)/(n**2)*PesMat[i, j] -(n-1)/(n**2)*(ColSumOpt[j] - OptMat[i,j]) -(n-1)/(n**2)*(RowSumOpt[i]- OptMat[i,j]) + OtherPes

    return OptYP, PesYP
def OptPesY1Recalc(OptU,PesU, n):

    OptY1 = np.ones(n)
    PesY1 = np.ones(n)

    RowSumOpt = np.matmul(OptU,np.ones(n).T)
    OptSum = np.sum(OptU)

    RowSumPes = np.matmul(PesU, np.ones(n).T)
    PesSum = np.sum(PesU)

    for i in range(n):
        OptY1[i] = (n-1)/(n**2)*RowSumOpt[i] - 1/(n**2) *(PesSum - RowSumPes[i])
        PesY1[i] = (n-1)/(n**2)*RowSumPes[i] - 1/(n**2) *(OptSum - RowSumOpt[i])

    return OptY1, PesY1
def OptPesY2Recalc(OptU,PesU, n):

    OptY2 = np.ones(n)
    PesY2 = np.ones(n)

    ColSumOpt = np.matmul(np.ones(n),OptU)
    OptSum = np.sum(OptU)

    ColSumPes = np.matmul(np.ones(n), PesU)
    PesSum = np.sum(PesU)

    for j in range(n):
        OptY2[j] = (n-1)/(n**2)*ColSumOpt[j] - 1/(n**2) *(PesSum - ColSumPes[j])
        PesY2[j] = (n-1)/(n**2)*ColSumPes[j] - 1/(n**2) *(OptSum - ColSumOpt[j])

    return OptY2, PesY2
def OptPesU2Update(OptU2,PesU2,a,b, n, PhiRange):

    for j in range(n):
        if j != b:
            OptU2[a, j] = np.minimum(OptU2[a, j], OptU2[a, b] + PhiRange)
            PesU2[a, j] = np.maximum(PesU2[a, j], PesU2[a, b] - PhiRange)

    return OptU2, PesU2
def OptPesU1Update(OptU1,PesU1,a,b, n, PhiRange):

    for i in range(n):
        if i != a:
            OptU1[i, b] = np.minimum(OptU1[i, b], OptU1[a, b] + PhiRange)
            PesU1[i, b] = np.maximum(PesU1[i, b], PesU1[a, b] - PhiRange)

    return OptU1,PesU1
def sample(i,j,matrices, n,k,PhiRange):

    UnknownU1 = matrices[0]
    UnknownU2 = matrices[1]

    KnownU1 = matrices[2]
    KnownU2 = matrices[3]

    OptYP1 = matrices[4]
    OptYP2 = matrices[5]

    PesYP1 = matrices[6]
    PesYP2 = matrices[7]

    OptY1 = matrices[8]
    OptY2 = matrices[9]

    PesY1 = matrices[10]
    PesY2 = matrices[11]

    OptU1 = matrices[12]
    PesU1 = matrices[13]

    OptU2 = matrices[14]
    PesU2 = matrices[15]

    if np.isnan(matrices[2][i,j]):
        # Sample UnknownU1 and Unknown U2
        U1Val = UnknownU1[i, j]
        U2Val = UnknownU2[i, j]

        # Update KnownU1 and KnownU2
        KnownU1[i, j] = U1Val
        KnownU2[i, j] = U2Val

        # Update Samples in OptU1, PesU1, OptU2, PesU2
        OptU1[i, j] = U1Val
        PesU1[i, j] = U1Val

        OptU2[i, j] = U2Val
        PesU2[i, j] = U2Val

        # Update OptU1, PesU1, OptU2, PesU2
        OptU1, PesU1 = OptPesU1Update(OptU1, PesU1, i, j, n, PhiRange)
        OptU2, PesU2 = OptPesU2Update(OptU2, PesU2, i, j, n, PhiRange)

        # Update OptY1,PesY1
        OptY1, PesY1 = OptPesY1Recalc(OptU1, PesU1, n)
        OptY2, PesY2 = OptPesY2Recalc(OptU2, PesU2, n)

        # Recalculate OptYP1,PesYP1
        OptYP1,PesYP1 = OptPesYPRecalc(OptU1, PesU1, n)

        # Update OptYP2,PesYP2
        OptYP2,PesYP2 = OptPesYPRecalc(OptU2, PesU2, n)

        #Update Potential Bounds
        matrices = [UnknownU1, UnknownU2, KnownU1, KnownU2, OptYP1, OptYP2, PesYP1, PesYP2, OptY1, OptY2, PesY1, PesY2,OptU1,PesU1,OptU2,PesU2]

    return matrices

def main(**kwargs):

    # Dimension of Problem
    n = kwargs.get("strategynumber")

    game = rand.randint(-12500, 12501, (n, n)) / 100000
    #game = np.ones((n, n)) * -0.1
    #game[5,19] = 0.125

    MaxPotential = np.max(game)
    MinPotential = np.min(game)

    NashIndices = FindNash(game)

    # Find max indices
    # Vmax1, Vmax2 = np.unravel_index(np.argmax(game, axis=None), game.shape)

    # Define the two unknown utility functions - rewards are between -0.25,0.25
    UnknownU2 = np.zeros((n, n))
    UnknownU1 = np.zeros((n, n))

    UnknownU2[:, -1] = 0
    UnknownU1[-1, :] = 0

    for i in range(n - 1):
        UnknownU2[:, -2 - i] = UnknownU2[:, -1 - i] + game[:, -2 - i] - game[:, -1 - i]
        UnknownU1[-2 - i, :] = UnknownU1[-1 - i, :] + game[-2 - i, :] - game[-1 - i]

    #Rewards are between 0 and 0.5
    UnknownU1 += 0.25
    UnknownU2 += 0.25

    # Calculate the components for comparison
    Phi = np.eye(n) - 1 / n * np.ones((n, n))
    Xi = 1 / n * np.ones((n, n))

    UnknownY1 = np.matmul(Phi, np.matmul(UnknownU1, Xi))
    UnknownY2 = np.matmul(Xi, np.matmul(UnknownU2, Phi))
    UnknownYP = np.matmul(Phi, np.matmul(UnknownU1, Phi))
    UnknownGame = UnknownYP + UnknownY1 + UnknownY2

    PhiRange = np.max(UnknownGame) - np.min(UnknownGame)

    # Prepare arrays for known samples
    KnownU2 = np.full((n, n), np.nan)
    KnownU1 = np.full((n, n), np.nan)

    t = 1

    Vs = []
    Percent = []
    Nash = []
    Gaps = []
    PercentBoundedPhi = []

    k = kwargs.get("optimismconstant")

    OptYP1,PesYP1 = OptPesYPInitCalc(KnownU1, n,k)
    OptYP2,PesYP2 = OptPesYPInitCalc(KnownU2, n,k)

    OptY1,PesY1 = OptPesY1InitCalc(KnownU2, n,k)
    OptY2,PesY2 = OptPesY2InitCalc(KnownU2, n,k)

    OptU1 = np.ones((n,n)) * 0.5
    PesU1 = np.ones((n, n)) * 0
    OptU2 = np.ones((n, n)) * 0.5
    PesU2 = np.ones((n, n)) * 0

    ## Loop Until we reach convergence

    t_max = kwargs.get("timesteps")

    active_indices = []
    for index in np.ndindex(game.shape):
        active_indices.append(list(index))

    matrices = [UnknownU1, UnknownU2, KnownU1, KnownU2, OptYP1, OptYP2, PesYP1, PesYP2, OptY1, OptY2, PesY1, PesY2,OptU1,PesU1,OptU2,PesU2]

    for i in range(n):
        matrices = sample(i, i, matrices, n, k,PhiRange)

    FullDiff = np.abs(UnknownGame - OptYP1 - np.array([OptY1] * n).T - np.array([OptY2] * n))

    while t<t_max:

        UnknownU1, UnknownU2, KnownU1, KnownU2, OptYP1, OptYP2, PesYP1, PesYP2, OptY1, OptY2, PesY1,PesY2,OptU1,PesU1,OptU2,PesU2 = matrices

        OptYP = np.minimum(OptYP1,OptYP2)
        PesYP = np.minimum(PesYP1,PesYP2)

        # Optimistic potential matrix estimate
        OptPhi = OptYP + np.array([OptY1] * n).T + np.array([OptY2] * n)

        # Pessimistic potential matrix estimate
        PesPhi = PesYP + np.array([PesY1] * n).T + np.array([PesY2] * n)

        # Find maximum of potential matrix estimate
        ind1, ind2 = np.unravel_index(np.argmax(OptPhi, axis=None),OptPhi.shape)

        #Max Index Potential Value
        Vs.append(UnknownGame[ind1, ind2])

        PesGap = np.count_nonzero(OptPhi < np.max(PesPhi)) / (n ** 2) * 100

        #Append gap
        Gaps.append(PesGap)

        #Check if point is a Nash Eq
        if [ind1,ind2] in NashIndices:
            Nash.append(MaxPotential)
        else:
            Nash.append(MinPotential)

        # Calculate the percentage of NaN values
        Percent.append(100-(np.count_nonzero(np.isnan(KnownU1)) / KnownU1.size) * 100)

        PercentBoundedPhi.append(np.count_nonzero(OptPhi < np.max(UnknownGame)) / (n ** 2) * 100)

        nan_new_act = []
        new_act = []
        indices = []
        for index in np.ndindex(game.shape):
            indices.append(list(index))

        for ind in indices:
            if OptPhi[ind[0],ind[1]] >= np.max(PesPhi):
                if np.isnan(matrices[2][ind[0],ind[1]]):
                    nan_new_act.append(ind)
                new_act.append(ind)

        if len(new_act) == 1:
            break

        if np.isnan(matrices[2][ind1, ind2]):
            matrices = sample(ind1, ind2, matrices, n, k,PhiRange)

        if len(nan_new_act) > 1:
            nan_active_indices = np.array(nan_new_act)
            rand_active_ind = np.random.choice(range(len(nan_active_indices)), size=1, replace=False)

            rand_active_ind1 = nan_active_indices[rand_active_ind][0][0]
            rand_active_ind2 = nan_active_indices[rand_active_ind][0][1]

            matrices = sample(rand_active_ind1, rand_active_ind2, matrices, n, k,PhiRange)
        elif np.any(np.isnan(matrices[2])):
            rand_ind1 = np.argwhere(np.isnan(matrices[2]))[0][0]
            rand_ind2 = np.argwhere(np.isnan(matrices[2]))[0][1]
            matrices = sample(rand_ind1, rand_ind2, matrices, n, k,PhiRange)

        # if np.any(np.isnan(matrices[2][ind1,:])):
        #     other2 = np.where(np.isnan(matrices[2][ind1,:]))[0][0]
        #     matrices = sample(ind1, other2, matrices, n, k,PhiRange)
        #
        # if np.any(np.isnan(matrices[2][:, ind2])):
        #     other1 = np.where(np.isnan(matrices[2][:, ind2]))[0][0]
        #     matrices = sample(other1, ind2, matrices, n, k,PhiRange)

        #Update time index
        t += 1

    print(np.sum(np.abs(UnknownGame - OptPhi)))

    # create a figure and axis object
    fig, ax1 = plt.subplots()
    fig.set_figwidth(15)
    # plot the first array using the left y-axis
    ax1.plot(Vs, color='red')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('True Potential Value of Optimistic Potential Estimate Maximum', color='red')
    ax1.axhline(y=np.max(UnknownGame), color='r', linestyle='--')
    ax1.set_ylim([np.min(UnknownGame), np.max(UnknownGame)+0.01])

    # create a twin axis object on the right side
    ax2 = ax1.twinx()

    # plot the second array using the right y-axis
    ax2.plot(Percent, color='blue',label = 'Percent of Utility Values Sampled')
    ax2.set_ylim([0,100])
    ax2.fill_between(range(len(Nash)), -5, 5, where=Nash > np.zeros(len(Nash)), color='green', alpha=0.5)
    # plot the second array using the right y-axis
    ax2.plot(Gaps, color='orange', label='Percentage of Strategy Profiles "Non-Active"')
    ax2.plot(PercentBoundedPhi, color='purple', label='Percentage of Optimistic Phi < Phi_max')
    ax2.set_ylabel('%')
    ax2.spines.right.set_position(("axes", 1.05))

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
              fancybox=True, shadow=True, ncol=5)

    # set the title of the plot
    plt.savefig("Figures/Test")

if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)