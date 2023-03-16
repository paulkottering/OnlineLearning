# Optimistic Normal Form Solver

How to run experiment:
````
python multi_runner.py -n 20 -t 600 -sa pa -si d -r 1
````

## Variables / Arguments Explanation
| Short Name | Long Name | Description |
| :------------ | :------------ |  :-----------: |
| `n` | `dimension` | Number of strategies per player |
| `t` | `timesteps` | Number of iterations |
| `sa` | `sample_strategy` |  Sampling strategy (see choices) |
| `si` | `intial_strategy` |  How to initialize game (see choices) |
| `r` | `runs` |  Number of runs to average over (default 1)  |
|||
