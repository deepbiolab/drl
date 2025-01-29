
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

While policy-based methods have the aforementioned advantages, they also suffer from certain drawbacks, such as **high variance** due to their reliance on Monte-Carlo estimates of returns and gradients and **slow learning speed**. This is because policy-based methods typically rely solely on policy information without fully leveraging the value information from the environment.

To address these shortcomings, the **[Actor-Critic (AC) method](./actor-critic/)** was introduced. It combines the strengths of both value-based and policy-based methods, utilizing the advantages of each. 

The Actor-Critic method consists of two main components:

- **Actor**: Responsible for learning and outputting the policy. Based on the current policy parameters, the actor decides which action to take in a given state.
- **Critic**: Responsible for evaluating the policy. It estimates the state-value function or action-value function to assess the quality of the current policy.

At each time step, the actor selects an action based on the policy, interacts with the environment, and receives feedback in the form of rewards. The critic evaluates this feedback using the estimated value function and provides guidance to update the actor's policy, steering it in a better direction.

### How Actor-Critic Methods Address Previous Drawbacks

The term **"critic"** implies that **bias has been introduced**, as Monte-Carlo estimates are unbiased but high variance. If instead of using Monte-Carlo estimates to train baselines, we use **Temporal Difference (TD) estimates (low variance but high bias)**, then we can say we have a **critic**. 
> About Bias and Variance introduced in this [introduction](../model-free-learning/introduction.md)

While this introduces bias, it reduces variance, thus improving convergence properties and speeding up learning.  Furtherly, by introducing **function approximation**, such as neural networks, can help **mitigate the bias introduced by pure TD methods**. TD methods rely on bootstrapping, where estimates depend on current approximations, potentially leading to cumulative bias if the value function is inaccurate. Function approximation addresses this by leveraging global information across the state space, capturing broader relationships rather than relying solely on local updates. It generalizes from training data, enabling reasonable estimates for unseen states, smooths the value function to avoid extreme biases, and reduces the impact of noise by fitting patterns from large datasets. 

However, while function approximation reduces bias, it may introduce approximation errors if the model's capacity or training data is insufficient, highlighting a trade-off in practical applications.

In essence, actor-critic methods address these challenges by:
- Reducing Variance
- Accelerating Learning Speed
- Stabilizing Policy Updates


### Applicability of Actor-Critic Methods

- Actor-Critic methods are particularly well-suited for handling **continuous and high-dimensional action spaces**, as the actor can directly output continuous actions.
- By combining value estimation with policy optimization, Actor-Critic methods excel in **solving complex tasks,** such as robotic control and game agent training.


## Summary

| Aspect                  | Policy-Based Methods (Actor-only)                                                                           | Value-Based Methods (Critic-only)                                                                  | Actor-Critic Methods                                         |
|-------------------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| **Learning Approach**   | Adjust action probabilities based on overall match outcomes (wins or losses)                                | Make continual guesses (estimates) about the final score throughout the match                      | Combine policy adjustment with value estimation              |
| **Efficiency**          | Inefficient, requires a large amount of data and many repetitions to learn a useful policy                  | More efficient; guesses help in estimating situations and actions sooner                           | More efficient learning by using critic to speed up policy updates |
| **Variance**            | High variance due to reliance on complete match outcomes                                                    | Lower variance as guesses are more consistent over time                                            | Reduced variance compared to policy-based methods            |
| **Bias**                | Unbiased but can misinterpret good actions taken during losses                                              | Introduces bias through guesses, which may be inaccurate initially                                 | Balances bias and variance through combined learning         |
| **Sample Efficiency**   | Needs many samples; slow learning process                                                                   | Learns to estimate values with fewer samples                                                       | Requires fewer samples than policy-based methods             |
| **Advantages**          | Good at learning optimal actions after extensive experience                                                 | Effective at evaluating states and actions; helps in distinguishing good from bad situations       | More stable learning; combines strengths of both actor and critic |
| **Disadvantages**       | Slow to learn; may decrease probability of good actions taken during losses                                 | Guesses can be wrong due to lack of experience, introducing bias                                   | Requires careful tuning of both actor and critic components  |
| **Application**         | Useful when the primary goal is to learn optimal actions without immediate feedback on action quality       | Useful for evaluating the value of states and actions to guide decision-making                     | Widely used in practice for its balance of learning speed and stability |
