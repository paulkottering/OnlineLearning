# Optimistic Normal Form Solver

This project contains an implementation of algorithms for learning in potential and congestion games with bandit feedbacl with a focus on sample-efficient learning and robustness. The solver supports various algorithms for learning in games, such as Nash-UCB II, NASH-CA, Optimistic Decomposition, and Exponential Weights with Annealing.


## Variables / Arguments Explanation

You can customize the experiment by providing various command-line arguments. The following table summarizes the available options:

| Short Name | Long Name    |                 Description                 |
|:-----------|:-------------|:-------------------------------------------:|
| `n`        | `dimension`  |       Number of strategies per player       |
| `k`        | `players`    |              Number of players              |
| `t`        | `timesteps`  |            Number of iterations             |
| `c`        | `constant`   |       Constant for UCB/LCB estimates        |
| `nl`       | `noise`      |      Gaussian noise standard deviation      |
| `r`        | `runs`       | Number of runs to average over (default 1)  |
| `a`        | `alpha`      | Alpha constant for direct optimistic method |
| `g`        | `game`       |            Type of game to play             |
| `s`        | `solver`     |             Which solver to use             |
|||

## Algorithms 

The repository supports various algorithms for learning in games. The following table lists the supported algorithms, their corresponding solver names, and the research papers they are based on:



| Full Name                          | solver name  | Hyperparameters                              |                                                        Paper                                                        |
|:-----------------------------------|:-------------|:---------------------------------------------|:-------------------------------------------------------------------------------------------------------------------:|
| Nash - UCB II                      | `nash_ucb`   | c (UCB constant)                             |                       Cui, Qiwen et al. “Learning in Congestion Games with Bandit Feedback.”                        |
| NASH-CA                            | `nash_ca`    | c (UCB constant)  , alpha  (Threshold)       | Song, Ziang et al. “When Can We Learn General-Sum Markov Games with a Large Number of Players Sample-Efficiently?”  |
| Optimistic Decomposition           | `optimistic` | c (UCB constant) , alpha  (greedy parameter) |                                                  Custom algorithm.                                                  |
| Exponential Weights with Annealing | `exp_weight` | c (UCB constant)                             |                      Héliou, Amélie et al. “Learning with Bandit Feedback in Potential Games.”                      |


## Example 


To run an experiment with 5 strategies per player, 4 players, 1000 iterations, a Gaussian noise standard deviation of 0.2, and using the Optimistic Decomposition solver, you would execute the following command:

````
python runner.py -n 5 -k 4 -t 1000 -nl 0.2 -s optimistic
````

The compare_plot script allows users to compare the cumulative regret of different simulation runs based on a varying parameter. 
The script takes as input fixed parameters and a varying parameter. 
It then filters the logs based on the fixed parameters and aggregates the cumulative regret of the remaining logs for each value of the varying parameter. 
Finally, the script generates a plot that shows the cumulative regret for each value of the varying parameter, with confidence intervals of one standard deviation, and lists the fixed parameter values on the right of the plot. 
The plot is saved in a folder called compare_figures. Users can also specify the width of the confidence intervals.
