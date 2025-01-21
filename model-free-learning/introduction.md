# Reinforcement Learning: From Value-Based to Policy-Based Methods

The core goal of reinforcement learning is to learn the **optimal policy** through interaction with the environment. Intuitively, the first step is to estimate the **optimal action-value function**.

In **small state spaces**, this optimal value function can be represented using a **table**:  
- Each row corresponds to a state.  
- Each column corresponds to an action.  

Then, we can construct the optimal policy state by state using this table. For each state, we simply extract the corresponding row from the table and select the action with the **highest value** as the optimal action.

Model-free reinforcement learning methods are a class of algorithms that enable agents to learn optimal policies directly from interaction with the environment, without requiring an explicit model of the environment's dynamics. These methods can be broadly categorized into two main approaches: **Monte Carlo Methods** and **Temporal Difference (TD) Methods**.

## Estimating the Optimal Q-Table Using Monte Carlo Methods

[Monte Carlo methods](./discrete-state-problems/monte-carlo-methods/) rely on averaging complete episodes of experience to estimate value functions and improve policies. These methods are particularly useful in episodic environments where the agent can reach a terminal state.


### Limitations of Requiring Complete Episodes for Updates

Monte Carlo methods rely on episode-based learning, requiring complete episodes to compute returns, which can lead to slow convergence in environments with sparse rewards. These methods often employ epsilon-greedy policies to balance exploration and exploitation, and utilize techniques like incremental mean and constant-alpha updates to enhance computational efficiency. However, the dependency on complete episodes for updates remains a notable limitation.

## Stepwise Updates with Temporal Difference (TD) Methods

[Temporal Difference methods](./discrete-state-problems/temporal-difference-methods/) combine the strengths of Monte Carlo methods and dynamic programming. They update value functions based on partial episodes, making them more sample-efficient.

Temporal Difference (TD) methods are characterized by their ability to perform **online learning**, updating value functions after every time step, and leveraging **bootstrapping**, where estimates are updated using other estimates, enabling faster learning compared to Monte Carlo methods. These methods support both **on-policy learning** (e.g., SARSA) and **off-policy learning** (e.g., Q-Learning), making them versatile for various reinforcement learning paradigms. TD methods offer notable advantages, such as not requiring complete episodes and achieving faster convergence.

### Challenges with Large Continuous State Spaces

For environments with much **larger state spaces**, this approach is no longer feasible. For example, consider the **"CartPole" environment**:

- The goal is to teach an agent to keep a pole balanced by pushing a cart (either **to the left** or **to the right**).
- At each time step, the environment's state is represented by a **vector** containing four numbers:
  1. The cart's position.
  2. The cart's velocity.
  3. The pole's angle.
  4. The pole's angular velocity.

Since these numbers can take on **countless possible values**, the number of potential states becomes enormous. Without some form of **[discretization](./continuous-state-problems/)**, it is impossible to represent the optimal action-value function in a table. This is because it would require a row for every possible state, making the table far too large to be practical.

