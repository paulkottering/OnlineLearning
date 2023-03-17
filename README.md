# Optimistic Normal Form Solver

How to run experiment:
````
python multi_runner.py -n 8 -k 3 -t 600 -sa da
````

## Variables / Arguments Explanation
| Short Name | Long Name         |                Description                 |
|:-----------|:------------------|:------------------------------------------:|
| `n`        | `dimension`       |      Number of strategies per player       |
| `k`        | `players`         |             Number of players              |
| `t`        | `timesteps`       |            Number of iterations            |
| `sa`       | `sample_strategy` |      Sampling strategy (see choices)       |
| `si`       | `intial_strategy` |    How to initialize game (see choices)    |
| `r`        | `runs`            | Number of runs to average over (default 1) |
|||
