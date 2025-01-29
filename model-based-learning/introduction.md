
# Mobel Based Learning: From Value-Based to Policy-Based Methods and the Actor-Critic Framework

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

## Using Neural Networks for Large State Spaces: Value-Based Methods

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


## Combining Value and Policy: Actor-Critic Methods

While policy-based methods have the aforementioned advantages, they also suffer from certain drawbacks, such as **high variance** and **slow learning speed**. This is because policy-based methods typically rely solely on policy information without fully leveraging the value information from the environment.

To address these shortcomings, the **[Actor-Critic (AC) method](./actor-critic/)** was introduced. It combines the strengths of both value-based and policy-based methods, utilizing the advantages of each.

The Actor-Critic method consists of two main components:

- **Actor**: Responsible for learning and outputting the policy. Based on the current policy parameters, the actor decides which action to take in a given state.
- **Critic**: Responsible for evaluating the policy. It estimates the state-value function or action-value function to assess the quality of the current policy.

At each time step, the actor selects an action based on the policy, interacts with the environment, and receives feedback in the form of rewards. The critic evaluates this feedback using the estimated value function and provides guidance to update the actor's policy, steering it in a better direction.

### How Actor-Critic Methods Address Previous Drawbacks

- Reducing Variance
   - The critic provides an **estimate of the value function, which acts as a baseline for the policy gradient**. This significantly **reduces the variance of the gradient estimation**, improving the stability and efficiency of learning.

- Accelerating Learning Speed
   - The feedback from the critic allows the actor to **adjust its policy parameters more accurately**, accelerating the learning process.

- Stabilizing Policy Updates
   - Actor-Critic methods provide a framework for **smoother policy updates**, avoiding large parameter fluctuations caused by high variance.

### Applicability of Actor-Critic Methods

- Actor-Critic methods are particularly well-suited for handling **continuous and high-dimensional action spaces**, as the actor can directly output continuous actions.
- By combining value estimation with policy optimization, Actor-Critic methods excel in **solving complex tasks,** such as robotic control and game agent training.