
# Deep Reinforcement Learning (DRL) Implementation

This repository contains implementations of various deep reinforcement learning algorithms, focusing on fundamental concepts and practical applications.

## Project Structure

> It is recommended to follow the material in the given order.

### Model Free Learning

#### Discrete Problems

##### Monte Carlo Methods
Implementation of Monte Carlo (MC) algorithms using the Blackjack environment as an example:

1. **[MC Prediction](model-free-learning/discrete-problems/monte-carlo-methods/monte_carlo_blackjack.ipynb)**
   - First-visit MC prediction for estimating action-value function
   - Policy evaluation with stochastic limit policy

2. **[MC Control with Incremental Mean](model-free-learning/discrete-problems/monte-carlo-methods/monte_carlo_blackjack.ipynb)**
   - GLIE (Greedy in the Limit with Infinite Exploration)
   - Epsilon-greedy policy implementation
   - Incremental mean updates

3. **[MC Control with Constant-alpha](model-free-learning/discrete-problems/monte-carlo-methods/monte_carlo_blackjack.ipynb)**
   - Fixed learning rate approach
   - Enhanced control over update process

##### Temporal Difference Methods
Implementation of TD algorithms on both Blackjack and CliffWalking environments:

1. **[SARSA (On-Policy TD Control)](model-free-learning/discrete-problems/temporal-difference-methods/temporal_difference_blackjack.ipynb)**
   - State-Action-Reward-State-Action
   - On-policy learning with epsilon-greedy exploration
   - Episode-based updates with TD(0)

2. **[Q-Learning (Off-Policy TD Control)](model-free-learning/discrete-problems/temporal-difference-methods/temporal_difference_blackjack.ipynb)**
   - Also known as SARSA-Max
   - Off-policy learning using maximum action values
   - Optimal action-value function approximation

3. **[Expected SARSA](model-free-learning/discrete-problems/temporal-difference-methods/temporal_difference_blackjack.ipynb)**
   - Extension of SARSA using expected values
   - More stable learning through action probability weighting
   - Combines benefits of SARSA and Q-Learning


#### Continuous Problems
##### Discretization

1. **[Q-Learning (Off-Policy TD Control)](model-free-learning/continuous-problems/discretization/discretization_mountaincar.ipynb)**
   - Q-Learning to the MountainCar environment using discretized state spaces
   - State space discretization through uniform grid representation for continuous variables
   - Exploration of the impact of discretization granularity on learning performance

2. **[Q-Learning (Off-Policy TD Control) with Tile Coding](model-free-learning/continuous-problems/discretization/tiling_discretization_acrobot.ipynb)**
   - Q-Learning applied to the Acrobot environment using tile coding for state space representation
   - Tile coding as a method to efficiently represent continuous state spaces by overlapping feature grids


### Model Based Learning

#### Value Based Iteration

##### Deep Q Networks Series

1. **[Deep Q Network with Experience Replay (DQN)](./model-based-learning/value-iteration/vanilla-dqn/dqn_lunarlander.ipynb)**
   - A neural network is used to approximate the Q-value function $Q(s, a)$.
   - Breaks the temporal correlation of samples by randomly sampling from a replay buffer.
   - Periodically updates the target network's parameters to reduce instability in target value estimation.
2. **[Double Deep Q Network with Experience Replay (DDQN)](./model-based-learning/value-iteration/variants-dqn/double_dqn_lunarlander.ipynb)**
   - Addresses the overestimation bias in vanilla DQN by decoupling action selection and evaluation.
   - This decoupling helps stabilize training and improves the accuracy of Q-value estimates.
3. **[Prioritized Double Deep Q Network (Prioritized DDQN)](./model-based-learning/value-iteration/variants-dqn/prioritized_ddqn_lunarlander.ipynb)**  
   - Enhances the efficiency of experience replay by prioritizing transitions with higher temporal-difference (TD) errors.  
   - Combines the stability of Double DQN with prioritized sampling to focus on more informative experiences.
4. **[Dueling Double Deep Q Network (Dueling DDQN)](./model-based-learning/value-iteration/variants-dqn/dueling_ddqn_lunarlander.ipynb)**
   - Introduces a new architecture that separates the estimation of **state value** $V(s)$ and **advantage function** $A(s, a)$
   - Improves learning efficiency by explicitly modeling the state value $V(s)$, which captures the overall "desirability" of actions 
   - Works particularly well in environments where some actions are redundant or where the state value $V(s)$ plays a dominant role in decision-making.


## Environments Brief in This Project

- **[Blackjack](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/blackjack.py)**: Classic card game environment for policy learning
- **[CliffWalking](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/cliffwalking.py)**: Grid-world navigation task with negative rewards and cliff hazards
- **[Taxi-v3](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/taxi.py)**: Grid-world transportation task where an agent learns to efficiently navigate, pick up and deliver passengers to designated locations while optimizing rewards.
- **[MountainCar](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/mountain_car.py)**: Continuous control task where an underpowered car must learn to build momentum by moving back and forth to overcome a steep hill and reach the goal position.
- **[Acrobot](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/acrobot.py)**: A two-link robotic arm environment where the goal is to swing the end of the second link above a target height by applying torque at the actuated joint. It challenges agents to solve nonlinear dynamics and coordinate the motion of linked components efficiently.
- **[LunarLander](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py)**: A physics-based environment where an agent controls a lunar lander to safely land on a designated pad. The task involves managing fuel consumption, balancing thrust, and handling the dynamics of gravity and inertia.

## Requirements

Create (and activate) a new environment with `Python 3.10` and `PyTorch 2.5.1`

- **Linux** or **Mac**: 

```bash
conda create -n DRL python=3.10
conda activate DRL
```


## Installation

1. Clone the repository:
```bash
git clone https://github.com/deepbiolab/drl.git
cd drl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Exmaple: Monte Carlo Methods

Run the Monte Carlo implementation:
```bash
cd monte-carlo-methods
python monte_carlo.py
```
Or explore the detailed notebook:

## Future Work

- Comprehensive implementations of fundamental RL algorithms
   - [x] [MC Control (Monte-Carlo Control)](http://incompleteideas.net/book/RLbook2020.pdf)
   - [x] [MC Control with Incremental Mean](http://incompleteideas.net/book/RLbook2020.pdf)
   - [x] [MC Control with Constant-alpha](http://incompleteideas.net/book/RLbook2020.pdf)
   - [x] [SARSA](http://incompleteideas.net/book/RLbook2020.pdf)
   - [x] [SARSA Max (Q-Learning)](http://incompleteideas.net/book/RLbook2020.pdf)
   - [x] [Expected SARSA](http://incompleteideas.net/book/RLbook2020.pdf)
   - [x] [Q-learning with Uniform Discretization](http://incompleteideas.net/book/RLbook2020.pdf)
   - [x] [Q-learning with Tile Coding Discretization](http://incompleteideas.net/book/RLbook2020.pdf)
   - [x] [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
   - [x] [DDQN](https://arxiv.org/pdf/1509.06461)
   - [x] [Prioritized DDQN](https://arxiv.org/pdf/1511.05952)
   - [x] [Dueling DDQN](https://arxiv.org/pdf/1511.06581)
   - [ ] [Rainbow](https://arxiv.org/pdf/1710.02298)
   - [ ] Hill Climbing
   - [ ] Cross Entropy Method
   - [ ] REINFORCE
   - [ ] A2C
   - [ ] [A3C](https://arxiv.org/pdf/1602.01783)
   - [ ] PPO
   - [ ] DDPG
   - [ ] MCTS, AlphaZero

    

