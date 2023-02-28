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
    """
    Update the YP matrix based on a new sample in the utility matrix

    Arguments:


    Returns:
    YP -- the updated YP matrix
    """

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

def sample(i,j,matrices, n, k):
    """
    Performs a sample of the UnknownU1 and updates all the relevant matrices.
    The joint strategy provided must not have been sample before.
    :param i: Strategy of Player 1
    :param j: Strategy of Player 2
    :matrices: List of matrices
    :return: Updated Matrices
    """
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

    if np.isnan(matrices[2][i,j]):
        # Sample UnknownU1 and Unknown U2
        U1Val = UnknownU1[i, j]
        U2Val = UnknownU2[i, j]

        # Update KnownU1 and KnownU2
        KnownU1[i, j] = U1Val
        KnownU2[i, j] = U2Val

        # Update OptU1, PesU1, OptU2, PesU2

        # Update OptY1,PesY1
        OptY1,PesY1 = OptPesY1Update(OptY1,PesY1, i, j, U1Val, n, k)

        # Update OptY2,PesY2
        OptY2,PesY2 = OptPesY2Update(OptY2,PesY2, i, j, U2Val, n, k)

        # Update OptYP1,PesYP1
        OptYP1,PesYP1 = OptPesYPUpdate(OptYP1,PesYP1, i, j, U1Val, n, k)

        # Update OptYP2,PesYP2
        OptYP2,PesYP2 = OptPesYPUpdate(OptYP2,PesYP2, i, j, U2Val, n, k)

        #Update Potential Bounds
        matrices = [UnknownU1, UnknownU2, KnownU1, KnownU2, OptYP1, OptYP2, PesYP1, PesYP2, OptY1, OptY2, PesY1, PesY2]

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

    # Prepare arrays for known samples
    KnownU2 = np.full((n, n), np.nan)
    KnownU1 = np.full((n, n), np.nan)

    t = 1

    Vs = []
    Percent = []
    Nash = []
    Gaps = []

    k = kwargs.get("optimismconstant")

    OptYP1,PesYP1 = OptPesYPInitCalc(KnownU1, n,k)
    OptYP2,PesYP2 = OptPesYPInitCalc(KnownU2, n,k)

    OptY1,PesY1 = OptPesY1InitCalc(KnownU2, n,k)
    OptY2,PesY2 = OptPesY2InitCalc(KnownU2, n,k)


    ## Loop Until we reach convergence

    t_max = kwargs.get("timesteps")

    active_indices = []
    for index in np.ndindex(game.shape):
        active_indices.append(list(index))


    matrices = [UnknownU1, UnknownU2, KnownU1, KnownU2, OptYP1, OptYP2, PesYP1, PesYP2, OptY1, OptY2, PesY1, PesY2]
    for i in range(n):
        matrices = sample(i, i, matrices, n, k)

    FullDiff = np.abs(UnknownGame - OptYP1 - np.array([OptY1] * n).T - np.array([OptY2] * n))

    while t<t_max:

        UnknownU1, UnknownU2, KnownU1, KnownU2, OptYP1, OptYP2, PesYP1, PesYP2, OptY1, OptY2, PesY1, PesY2 = matrices

        OptYP = OptYP1
        PesYP = PesYP1

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
        Percent.append((np.count_nonzero(np.isnan(KnownU1)) / KnownU1.size) * 100)

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

        if len(new_act) == 0:
            break

        if np.isnan(matrices[2][ind1, ind2]):
            matrices = sample(ind1, ind2, matrices, n, k)

        if len(nan_new_act) > 1:
            nan_active_indices = np.array(nan_new_act)
            rand_active_ind = np.random.choice(range(len(nan_active_indices)), size=1, replace=False)

            rand_active_ind1 = nan_active_indices[rand_active_ind][0][0]
            rand_active_ind2 = nan_active_indices[rand_active_ind][0][1]

            matrices = sample(rand_active_ind1, rand_active_ind2, matrices, n, k)

        if np.any(np.isnan(matrices[2])):
            rand_ind1 = np.argwhere(np.isnan(matrices[2]))[0][0]
            rand_ind2 = np.argwhere(np.isnan(matrices[2]))[0][1]
            matrices = sample(rand_ind1, rand_ind2, matrices, n, k)

        if np.any(np.isnan(matrices[2][ind1,:])):
            other2 = np.where(np.isnan(matrices[2][ind1,:]))[0][0]
            matrices = sample(ind1, other2, matrices, n, k)

        if np.any(np.isnan(matrices[2][:, ind2])):
            index = np.where(np.isnan(matrices[2][:, ind2]))[0]
            other1 = np.where(np.isnan(matrices[2][:, ind2]))[0][0]
            matrices = sample(other1, ind2, matrices, n, k)

        #Update time index
        t += 1

    # create a figure and axis object
    fig, ax1 = plt.subplots()
    fig.set_figwidth(15)
    # plot the first array using the left y-axis
    ax1.plot(Vs, color='red')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('True YP Value of Optimistic YP Estimate Maximum', color='red')
    ax1.axhline(y=np.max(UnknownGame), color='r', linestyle='--')

    ax1.plot(range(len(Nash)), Nash, color='green')

    # create a twin axis object on the right side
    ax2 = ax1.twinx()

    # plot the second array using the right y-axis
    ax2.plot(Percent, color='blue',label = 'Percent of Utility Values Sampled')
    ax2.set_ylabel('% of U Matrix Sampled', color='blue')
    ax2.set_ylim([0,100])

    # create a twin axis object on the right side
    ax3 = ax1.twinx()

    # plot the second array using the right y-axis
    ax3.plot(Gaps, color='orange', label='Gap')
    ax3.set_ylabel('% of YP Optimistic Estimates less than Pessimistic YP Maximum', color='orange')
    ax3.spines.right.set_position(("axes", 1.05))

    # set the title of the plot
    plt.savefig("Figures/Test")

if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)