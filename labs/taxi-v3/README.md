# Taxi-v3 Environment Solution using Q-Learning

## Project Overview
This project implements a Q-Learning solution for the Taxi-v3 environment from OpenAI Gym (maintained by Gymnasium). The goal is to train a taxi agent to navigate in a grid world, picking up and dropping off passengers at designated locations.

## Environment Description
The Taxi-v3 environment represents a 5x5 grid world where a taxi needs to navigate to pick up passengers from one location and drop them off at another. The state space consists of 500 possible states (25 taxi positions × 5 passenger locations × 4 destination locations).

### State Space
- Taxi location (25 possible positions)
- Passenger location (5 possible locations - 4 designated locations + taxi)
- Destination location (4 possible locations)

### Action Space
The agent can perform 6 actions:
- 0: Move South
- 1: Move North
- 2: Move East
- 3: Move West
- 4: Pickup passenger
- 5: Drop off passenger

### Rewards
- -1 per step
- +20 for successful drop-off
- -10 for illegal pickup/drop-off actions

## Implementation Details

### Project Structure
The repository contains three main Python files:
- `agent.py`: Contains the Q-Learning agent implementation
- `monitor.py`: Handles the environment interaction and performance monitoring
- `main.py`: Entry point for running experiments

### Agent Implementation
The agent uses Q-Learning with the following features:
- Epsilon-greedy exploration strategy with decay
- Learning rate (alpha) optimization
- Gamma (discount factor) set to 1.0
- Dynamic exploration-exploitation balance

## Running the Code

### Prerequisites

**Test on `Python 3.10.x`**

```bash
pip install gymnasium
pip install numpy
pip install matplotlib
```

### Basic Usage
To run a single training session:
```bash
python main.py
```

## Performance Metrics
- Episodes to convergence
- Final average reward
- Best average reward
- Stability of learning

## References
- [Original Paper](https://arxiv.org/pdf/cs/9905014.pdf)
- [OpenAI Gym Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/)
- [Q-Learning Algorithm](https://link.springer.com/article/10.1007/BF00992698)

