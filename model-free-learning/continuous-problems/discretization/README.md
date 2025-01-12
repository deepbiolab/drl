# Discretization

### Why Discretization is Needed

In reinforcement learning, model-free methods like Monte Carlo (MC) control and Temporal-Difference (TD) control perform efficiently in discrete state and action spaces. This is because they can represent the state-action value function $ Q(s, a) $ using lookup tables or dictionaries, and optimization tasks (like updating policies) can be performed through simple enumeration. However, these methods face significant challenges when applied to continuous state or action spaces. For instance, in continuous state spaces, a Q-table would require an infinite number of rows to represent all possible states, making it infeasible. Similarly, in continuous action spaces, algorithms like Q-learning require computing the $\max_a Q(s', a)$ at each time step, which becomes a complex optimization problem instead of a straightforward comparison. **Therefore, to make reinforcement learning algorithms applicable to continuous spaces, we need to discretize these spaces, transforming continuous problems into discrete ones and enabling the use of existing algorithms.**

---

### Methods of Discretization

Discretization involves dividing continuous spaces into a finite number of discrete regions, effectively converting continuous problems into discrete ones. For state spaces, this can be achieved by mapping continuous state values into discrete "grids" or "intervals." For example, in a two-dimensional physical space, continuous coordinates (e.g., $(x, y)$) can be divided into discrete grid points at regular intervals. Similarly, for action spaces, continuous action values can be divided into discrete sets, such as splitting angles into fixed increments like 90 degrees or 45 degrees. **However, uniform discretization is not always optimal. In some cases, non-uniform discretization is more effective, where grid sizes or intervals are dynamically adjusted based on the problem's characteristics**. Advanced methods like occupancy grids can further improve discretization by using finer divisions in important regions while avoiding unnecessary computation in less relevant areas. This allows for a more efficient representation and computation of value functions.

see details in `discretization.ipynb` to learn how to discretize continuous state spaces, to use tabular solution methods to solve complex tasks. 



## Observation Space

The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

| Num  | Observation                          | Min   | Max  | Unit         |
| ---- | ------------------------------------ | ----- | ---- | ------------ |
| 0    | position of the car along the x-axis | -1.2  | 0.6  | position (m) |
| 1    | velocity of the car                  | -0.07 | 0.07 | velocity (v) |

## Action Space

There are 3 discrete deterministic actions:

- 0: Accelerate to the left
- 1: Don’t accelerate
- 2: Accelerate to the right



### Resources

To learn about more advanced discretization approaches, refer to the following:

- Uther, W., and Veloso, M., 1998. [Tree Based Discretization for Continuous State Space Reinforcement Learning](http://www.cs.cmu.edu/~mmv/papers/will-aaai98.pdf). In _Proceedings of AAAI, 1998_, pp. 769-774.
- Munos, R. and Moore, A., 2002. [Variable Resolution Discretization in Optimal Control](https://link.springer.com/content/pdf/10.1023%2FA%3A1017992615625.pdf). In _Machine Learning_, 49(2), pp. 291-323.
