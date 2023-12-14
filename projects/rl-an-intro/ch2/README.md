# Chapter 2

## K-armed Bandit Testbed Implementation

#### Graphs for basic implementation with various parameters

- 10 arms
- $\epsilon$-greedy values: .1, .05, .01, 0
- averages over 2000 runs
- 1000 steps

![](k-armed-testbed-stat.png "K-armed Testbed, Basic")

- 10 arms
- $\epsilon$-greedy values: .1, .05, .01, 0
- averages over 2000 runs
- 1000 steps
- optimistic initial value of 5
- non-stationary update rule

![](k-armed-testbed-opt-init-non-stat.png "K-armed Testbed, Optimistic Initial Values")

- 10 arms
- UCB constant values: 1, 2, 5, 10
- averages over 2000 runs
- 1000 steps

![](k-armed-testbed-ucb.png "K-armed Testbed, Upper-Confidence-Bound")
