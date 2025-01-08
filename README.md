
# Deep Reinforcement Learning (DRL) Implementation

This repository contains implementations of various deep reinforcement learning algorithms, focusing on fundamental concepts and practical applications.

## Project Structure

```
drl/
├── monte-carlo-methods/
│   ├── monte_carlo.py         # Concise version
│   ├── Monte_Carlo.ipynb      # Detailed version with illustration
│   └── plot_utils.py          # Plotting utilities for visualizing results
└── requirements.txt          # Project dependencies
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

## Requirements

```
python==3.10.x
gymnasium==1.0.0
matplotlib==3.10.0
numpy==2.2.1
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

- Implementation of additional algorithms

