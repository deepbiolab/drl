
# Deep Reinforcement Learning (DRL) Implementation

This repository contains implementations of various deep reinforcement learning algorithms, focusing on fundamental concepts and practical applications.

## Project Structure

```
drl/
├── monte-carlo-methods/
│   ├── monte_carlo_blackjack.py         # Monte Carlo methods implementation
│   ├── monte_carlo_blackjack.ipynb      # Detailed notebook with illustrations
│   └── plot_utils.py                    # Plotting utilities for visualizing results
├── temporal-difference-methods/
│   ├── temporal_difference_blackjack.py         # TD methods implementation
│   ├── temporal_difference_blackjack.ipynb      # Detailed notebook with illustrations
│   └── plot_utils.py                            # Plotting utilities
└── requirements.txt                             # Project dependencies
```

## Implemented Algorithms

### Monte Carlo Methods
Implementation of Monte Carlo (MC) algorithms using the Blackjack environment as an example:

1. **MC Prediction**
   - First-visit MC prediction for estimating action-value function
   - Policy evaluation with stochastic limit policy

2. **MC Control with Incremental Mean**
   - GLIE (Greedy in the Limit with Infinite Exploration)
   - Epsilon-greedy policy implementation
   - Incremental mean updates

3. **MC Control with Constant-$α$**
   - Fixed learning rate approach
   - Enhanced control over update process

### Temporal Difference Methods
Implementation of TD algorithms using the Blackjack environment:

1. **SARSA (On-Policy TD Control)**
   - State-Action-Reward-State-Action
   - On-policy learning with epsilon-greedy exploration
   - Episode-based updates with TD(0)

2. **Q-Learning (Off-Policy TD Control)**
   - Also known as SARSA-Max
   - Off-policy learning using maximum action values
   - Optimal action-value function approximation

3. **Expected SARSA**
   - Extension of SARSA using expected values
   - More stable learning through action probability weighting
   - Combines benefits of SARSA and Q-Learning

## Requirements

```
gymnasium==1.0.0
ipython==8.12.3
matplotlib==3.10.0
numpy==2.2.1
plotly==5.24.1
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/drl.git
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

- Implementation of TD algorithms on CliffWalking problem

    

