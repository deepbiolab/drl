
# Reinforcement Learning: From Value-Based to Policy-Based Methods

The core goal of reinforcement learning is to learn the **optimal policy** through interaction with the environment. So far in **[model free learning](../model-free-learning/)**, where the first step is to estimate the **optimal action-value function**. 

In **small state spaces**, this optimal value function can be represented using a **table**:  
- Each row corresponds to a state.  
- Each column corresponds to an action.  

Then, we can construct the optimal policy state by state using this table. For each state, we simply extract the corresponding row from the table and select the action with the **highest value** as the optimal action.

## Challenges with Large State Spaces

For environments with much **larger state spaces**, this approach is no longer feasible. For example, consider the **"CartPole" environment**:

- The goal is to teach an agent to keep a pole balanced by pushing a cart (either **to the left** or **to the right**).
- At each time step, the environment's state is represented by a **vector** containing four numbers:
  1. The cart's position.
  2. The cart's velocity.
  3. The pole's angle.
  4. The pole's angular velocity.

Since these numbers can take on **countless possible values**, the number of potential states becomes enormous. Without some form of **discretization**, it is impossible to represent the optimal action-value function in a table. This is because it would require a row for every possible state, making the table far too large to be practical.

## Using Neural Networks for Large State Spaces

To address this issue, we explored how to use a **neural network** to represent the **optimal action-value function**, which forms the basis of the **[Value Based Methods](./value-iteration/), such as Deep Q Learning algorithm**.

- In this case, the neural network takes the **state of the environment** as input and outputs the **value of each possible action**.
- For example, in the "CartPole" environment, the possible actions are:
  - Pushing the cart **to the left**.
  - Pushing the cart **to the right**.

Similar to the case of using a table, we can easily determine the **best action** for any given state by selecting the action that **maximizes the output values** of the neural network.

## Moving Beyond Value Functions: Policy-Based Methods

Whether we use a **table** (for small state spaces) or a **neural network** (for large state spaces), we must first **estimate the optimal action-value function** before solving for the **optimal policy**.

However, the question now arises:  
**Can we directly find the optimal policy without estimating the value function?**

The answer is **yes**, and we can achieve this through a class of algorithms known as **[Policy-Based Methods](./policy-iteration/)**.

### Policy Networks for Both Continuous State/Action Spaces

First, let us consider how to estimate an optimal policy. Using the **CartPole** example, the agent has two possible actions: it can push the cart either to the left or to the right. At each time step, the agent selects one action from these two options.

We can construct a neural network to approximate the policy. This network takes the state as input and outputs the probabilities of the agent selecting each possible action. If there are two possible actions, the output layer will consist of two nodes. The agent uses this policy to interact with the environment by passing the most recent state into the network. The network outputs the action probabilities, and the agent samples from these probabilities to select an action. For instance, in a given state, the agent might have a **90%** probability of selecting "push left" and a **10%** probability of selecting "push right."

Our objective is to determine appropriate values for the network weights such that, for each state input to the network, it outputs an action probability distribution where the optimal action is most likely to be selected. This helps the agent achieve its goal of **maximizing the expected return**.

This is an iterative process. Initially, the weights are set to random values. As the agent interacts with the environment and learns how to better optimize its reward strategy, it adjusts these weights. Over time, as the weights are updated, the agent begins to select more appropriate actions for each state and eventually masters the task.

### Three Key Advantages of Policy-Based Methods

1. **Simplicity and Generalization**  
   - Policy-based methods directly address the problem of finding the optimal policy without relying on a value function as an intermediate step.  
   - They avoid storing unnecessary data and enable **generalization** across the state space, focusing on complex regions.

2. **Ability to Learn Stochastic Policies**  
   - Policy-based methods can learn true **stochastic policies**, which are essential for scenarios requiring randomness or handling **aliased states** (states that appear identical but are actually different).  
   - Unlike value-based methods with "ε-greedy" approaches, policy-based methods are more efficient and avoid oscillation or inefficient action selection.

3. **Suitability for Continuous Action Spaces**  
   - In continuous or high-dimensional action spaces, value-based methods require solving an optimization problem to select actions, which is computationally expensive.  
   - Policy-based methods map states directly to actions, significantly reducing computation time and making them ideal for complex scenarios.

