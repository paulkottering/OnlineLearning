import numpy as np
import matplotlib.pyplot as plt


def plot_many(kwargs,cumulative_regret,game_mean):

    fig, ax1 = plt.subplots()
    fig.set_figwidth(15)
    # plot the first array using the left y-axis
    ax1.plot(np.mean(cumulative_regret, axis=0), color='red', label='Random Strategy')
    ax1.plot(game_mean*range(len(np.mean(cumulative_regret, axis=0))), color='blue', label='Strategy')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cumulative Regret', color='red')

    n = kwargs.get("dimension")
    k = kwargs.get("players")
    s = kwargs.get("solver")
    runs = kwargs.get("runs")
    regret = kwargs.get("regret")
    game = kwargs.get("game")
    title = "Number of Players: "+str(k)+"  Number of strategies per player:  "+str(n)+"  Algorithm:  "+ s
    plt.title(title)
    # set the title of the plot
    plt.savefig("Figures/"+str(game)+"_"+str(n)+str(k)+"_"+s+"_"+str(runs)+"_"+str(regret))
