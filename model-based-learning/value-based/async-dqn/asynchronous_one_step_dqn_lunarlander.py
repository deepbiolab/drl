"""See explaination in notebook (asynchronous_one_step_dqn_lunarlander.ipynb)"""

import os
import time
import numpy as np
import gymnasium as gym
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
import random

device = torch.device("cpu")
print(f"Training on: {device}")


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=64, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.Q = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        actions = self.Q(state)
        return actions


class SharedAdam(optim.Adam):
    """Shared Adam optimizer for multiprocessing"""

    def __init__(self, params, lr=5e-4):
        """
        Initialize the shared Adam optimizer.
        Args:
            params: Parameters to optimize.
            lr (float): Learning rate.
        """
        super(SharedAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group["params"]:
                # Initialize optimizer state
                state = self.state[p]
                state["step"] = torch.tensor(0)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                # Share memory for multiprocessing
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
                state["step"].share_memory_()


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        Q_network,
        Q_target_network=None,
        optimizer=None,
        gamma=0.99,
        target_update_steps=1000,
        update_steps=5,
        seed=42,
        T=None,
        optimizer_lock=None,
    ):
        """
        Initialize the agent.
        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.
            Q_network (QNetwork): Main Q-Network.
            Q_target_network (QNetwork): Target Q-Network.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            gamma (float): Discount factor.
            target_update_steps (int): Steps between target network updates.
            update_steps (int): Steps between gradient application.
            seed (int): Random seed for reproducibility.
            T (multiprocessing.Value): Global step counter shared across processes.
            optimizer_lock (multiprocessing.Lock): Lock for synchronizing optimizer updates.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.Q = Q_network
        self.Q_target = Q_target_network
        self.optimizer = optimizer
        self.gamma = gamma
        self.target_update_steps = target_update_steps
        self.update_steps = (
            update_steps  # Number of steps after which to apply gradients
        )
        self.t_step = 0  # Time step counter for this thread
        self.T = T  # Shared global counter
        self.optimizer_lock = optimizer_lock
        self.seed = np.random.seed(seed)
        random.seed(seed)

        # Initialize accumulated gradients as None
        self.reset_gradients()

    def reset_gradients(self):
        """Reset accumulated gradients."""
        self.accumulated_grads = [torch.zeros_like(p) for p in self.Q.parameters()]

    def select_action(self, state, epsilon=0.0):
        """Selects an action using epsilon-greedy policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.Q.eval()
        with torch.no_grad():
            actions = self.Q(state)
        self.Q.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(actions.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """Processes a step and learns from the experience."""
        # Convert experience to tensors
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = torch.tensor(action, dtype=torch.int64).reshape(1, 1).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).reshape(1, 1).to(device)
        next_state = (
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        )
        done = torch.tensor(done, dtype=torch.float32).reshape(1, 1).to(device)

        # Perform learning step
        loss = self.compute_loss((state, action, reward, next_state, done))

        # Backpropagate loss and accumulate gradients
        self.Q.zero_grad()
        loss.backward()

        # Accumulate gradients
        with torch.no_grad():
            for acc_grad, param in zip(self.accumulated_grads, self.Q.parameters()):
                acc_grad += param.grad.clone()

        # Increment step counters
        self.t_step += 1
        with self.T.get_lock():
            self.T.value += 1

        # Check if it's time to apply accumulated gradients
        if self.t_step % self.update_steps == 0 or done.item():
            self.apply_gradients()
            self.reset_gradients()

        # Update target network
        if self.T.value % self.target_update_steps == 0:
            self.hard_update()

    def compute_loss(self, experience):
        """Computes the loss for a single experience tuple."""
        state, action, reward, next_state, done = experience

        # Compute TD target using the target network
        with torch.no_grad():
            Q_targets_next = torch.max(self.Q_target(next_state), dim=-1, keepdim=True)[
                0
            ]
            Q_targets = reward + (1 - done) * self.gamma * Q_targets_next

        # Compute expected Q values using the local network
        Q_expected = torch.gather(self.Q(state), dim=-1, index=action)

        # Compute loss (mean squared error)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def apply_gradients(self):
        """Apply accumulated gradients to the shared network."""
        with self.optimizer_lock:
            for param, acc_grad in zip(self.Q.parameters(), self.accumulated_grads):
                param.grad = acc_grad  # Set the accumulated gradients

            # Perform optimizer step
            self.optimizer.step()
            # Zero the parameter gradients (in case they weren't zeroed)
            self.optimizer.zero_grad()

    def hard_update(self):
        """Hard update: θ_target = θ"""
        with self.optimizer_lock:
            self.Q_target.load_state_dict(self.Q.state_dict())


def worker_dqn(
    rank,
    Q,
    Q_target,
    optimizer,
    T,
    max_score,
    env_name,
    num_episodes=2000,
    max_t=1000,
    window=100,
    optimizer_lock=None,
):
    """
    Function executed by each worker process.
    Args:
        rank (int): Process rank.
        Q (QNetwork): Shared Q-network.
        Q_target (QNetwork): Shared target Q-network.
        optimizer (SharedAdam): Shared optimizer.
        T (multiprocessing.Value): Global step counter.
        max_score (multiprocessing.Value): Global maximum score.
        env_name (str): Environment name.
        num_episodes (int): Number of training episodes.
        max_t (int): Maximum steps per episode.
        window (int): Window size for calculating average scores.
        optimizer_lock (multiprocessing.Lock): Lock for synchronizing optimizer updates.
    """
    # Create an environment instance for each worker
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize the agent with shared networks and optimizer
    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        Q_network=Q,
        Q_target_network=Q_target,
        optimizer=optimizer,
        gamma=0.995,
        target_update_steps=50,
        update_steps=10,
        seed=42 + rank,
        T=T,
        optimizer_lock=optimizer_lock,
    )

    epsilon = 1.0
    eps_min = 0.01
    eps_decay = 0.999

    scores_window = deque(maxlen=window)

    for i_episode in range(1, num_episodes + 1):
        with max_score.get_lock():
            if max_score.value >= 200.0:
                break

        state, _ = env.reset()
        total_reward = 0
        agent.t_step = 0
        agent.reset_gradients()
        done = False

        for t in range(max_t):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        # Decay epsilon
        epsilon = max(eps_min, eps_decay * epsilon)

        scores_window.append(total_reward)
        mean_score = np.mean(scores_window)

        if i_episode % 10 == 0:
            print(
                f"Process {rank}, Episode {i_episode}, Total Reward: {total_reward:.2f}, Average Score: {mean_score:.2f}"
            )

        if len(scores_window) >= window and mean_score >= 200.0:
            with max_score.get_lock():
                max_score.value = mean_score
                print(
                    f"\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {mean_score:.2f}"
                )
                torch.save(agent.Q.state_dict(), "checkpoint.pth")
                break

    env.close()


if __name__ == "__main__":
    start_time = time.time()

    os.environ["OMP_NUM_THREADS"] = "1"

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Environment name
    env_name = "LunarLander-v3"
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    # Initialize global networks
    global_Q = QNetwork(state_size, action_size, hidden_size=128).to(device)
    global_Q_target = QNetwork(state_size, action_size, hidden_size=128).to(device)
    global_Q.share_memory()
    global_Q_target.share_memory()

    # Initialize global optimizer
    global_optimizer = SharedAdam(global_Q.parameters(), lr=5e-4)

    # Global counter T
    global_T = mp.Value("i", 0)

    # Initialize global max score as a shared value
    global_max_score = mp.Value("d", -float("inf"))

    # Create optimizer lock
    optimizer_lock = mp.Lock()

    # Number of processes
    num_processes = 8
    num_episodes = 2000

    processes = []
    for rank in range(num_processes):
        p = mp.Process(
            target=worker_dqn,
            args=(
                rank,
                global_Q,
                global_Q_target,
                global_optimizer,
                global_T,
                global_max_score,
                env_name,
                num_episodes,
                1000,
                100,
                optimizer_lock,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"Process {p.pid} exited with code {p.exitcode}")

    tot_time = (time.time() - start_time) / 60
    print(f"Total Training Time: {tot_time:.2f}")
