# Optimistic Normal Form Solver

How to run experiment:
````
python multi_runner.py -n 8 -k 3 -t 600 -nl 0.1 r
````

## Variables / Arguments Explanation
| Short Name | Long Name      |                 Description                 |
|:-----------|:---------------|:-------------------------------------------:|
| `n`        | `dimension`    |       Number of strategies per player       |
| `k`        | `players`      |              Number of players              |
| `t`        | `timesteps`    |            Number of iterations             |
| `c`        | `ucb_constant` |       Constant for UCB/LCB estimates        |
| `nl`       | `noise`        |      Gaussian noise standard deviation      |
| `r`        | `runs`         | Number of runs to average over (default 1)  |
| `a`        | `alpha`        | Alpha constant for direct optimistic method |
| `g`        | `game`         |            Type of game to play             |
| `s`        | `solver`       |             Which solver to use             |
|||

## Algorithms 

| Full Name                          | solver name  |                                                       Paper                                                        |
|:-----------------------------------|:-------------|:------------------------------------------------------------------------------------------------------------------:|
| Nash - UCB II                      | `nash_ucb`   |                       Cui, Qiwen et al. “Learning in Congestion Games with Bandit Feedback.”                       |
| NASH-CA                            | `nash_ca`    | Song, Ziang et al. “When Can We Learn General-Sum Markov Games with a Large Number of Players Sample-Efficiently?” |
| Optimistic Decomposition           | `optimistic` |                                                 Custom algorithm.                                                  |
| Exponential Weights with Annealing | `exp_weight` |                     Héliou, Amélie et al. “Learning with Bandit Feedback in Potential Games.”                      |
| OGD - lb                           | `ogd_lb`     |          Liu, Wenting et al. “No-regret learning for repeated non-cooperative games with lossy bandits.”           |