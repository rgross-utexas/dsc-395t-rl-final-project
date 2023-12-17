## Chapter 2

### K-armed Bandit Testbed Implementation

#### Graphs for basic implementation with various parameters

- 10 arms
- $\epsilon$-greedy values: .1, .05, .01, 0
- averages over 2000 runs
- 1000 steps

![](images/k-armed-testbed-stat.png "K-armed Testbed, Basic")

- 10 arms
- $\epsilon$-greedy values: .1, .05, .01, 0
- averages over 2000 runs
- 1000 steps
- optimistic initial value of 5
- non-stationary update rule

![](images/k-armed-testbed-opt-init-non-stat.png "K-armed Testbed, Optimistic Initial Values")

- 10 arms
- UCB constant values: 1, 2, 5, 10
- averages over 2000 runs
- 1000 steps

![](images/k-armed-testbed-ucb.png "K-armed Testbed, Upper-Confidence-Bound")

## Chapter 4

$p_\star$ for $4\times4$ deterministic GridWorld MDP:

```
[ F 0 0 0
  1 0 0 3
  1 0 2 3
  1 2 2 F]
```

State value prediction using $p_\star$ (above):

```
[ 0. -1. -2. -3.
 -1. -2. -3. -2.
 -2. -3. -2. -1.
 -3. -2. -1.  0.]
```

State-action value prediction using $p_\star$ (above):

```
[[ 0  0  0  0]
 [-1 -2 -3 -3]
 [-2 -3 -4 -4]
 [-3 -4 -4 -3]
 [-2 -1 -3 -3]
 [-2 -2 -4 -4]
 [-3 -3 -3 -3]
 [-4 -4 -3 -2]
 [-3 -2 -4 -4]
 [-3 -3 -3 -3]
 [-4 -4 -2 -2]
 [-3 -3 -2 -1]
 [-4 -3 -3 -4]
 [-4 -4 -2 -3]
 [-3 -3 -1 -2]
 [ 0  0  0  0]]
```


State value prediction from using a equiprobable random policy:

```
[  0.         -13.99893866 -19.99842728 -21.99824003
 -13.99893866 -17.99861452 -19.9984378  -19.99842728
 -19.99842728 -19.9984378  -17.99861452 -13.99893866
 -21.99824003 -19.99842728 -13.99893866   0.]
 ```

State-action value prediction from using a equiprobable random policy:

```
[[  0.           0.           0.           0.        ]
 [ -1.         -14.99887902 -20.99833891 -18.99853668]
 [-14.99887902 -20.99833891 -22.99814115 -20.99835003]
 [-20.99833891 -22.99814115 -22.99814115 -20.99833891]
 [-14.99887902  -1.         -18.99853668 -20.99833891]
 [-14.99887902 -14.99887902 -20.99835003 -20.99835003]
 [-18.99853668 -20.99833891 -20.99833891 -18.99853668]
 [-20.99835003 -22.99814115 -20.99833891 -14.99887902]
 [-20.99833891 -14.99887902 -20.99835003 -22.99814115]
 [-20.99833891 -18.99853668 -18.99853668 -20.99833891]
 [-20.99835003 -20.99835003 -14.99887902 -14.99887902]
 [-18.99853668 -20.99833891 -14.99887902  -1.        ]
 [-22.99814115 -20.99833891 -20.99833891 -22.99814115]
 [-22.99814115 -20.99835003 -14.99887902 -20.99833891]
 [-20.99833891 -18.99853668  -1.         -14.99887902]
 [  0.           0.           0.           0.        ]]
```