import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import gymnasium as gym
from enum import Enum
from collections import deque
import ale_py

gym.register_envs(ale_py)

class Action(Enum):
    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3
    RIGHTFIRE = 4
    LEFTFIRE = 5

# Random action function using numpy broadcasting
def get_random_action(n):
    return np.random.choice([Action.RIGHTFIRE.value, Action.LEFTFIRE.value], size=n)

def perform_random_steps(env, nrand, parallel=False):
    """
    Perform a number of random steps in the environment to initialize the game.
    Supports both single and parallel environments.
    Args:
        env: The game environment (single or parallel)
        nrand (int): Number of random steps to perform
        parallel (bool): Whether the environment is parallel
    Returns:
        tuple: The last two frames after performing random steps
    """
    # Reset environment
    env.reset()
    # Get environment size (1 for single env, n for parallel envs)
    n = len(env.ps) if parallel else 1
    # Unified action definitions
    fire_action = np.full(n, Action.FIRE.value, dtype=np.int32)
    noop_action = np.full(n, Action.NOOP.value, dtype=np.int32)
    # Start the game with a FIRE action
    env.step(fire_action.item() if n == 1 else fire_action)
    # Initialize frames
    frames1, frames2 = None, None
    # Perform random steps
    for _ in range(nrand):
        # Get and format random action
        action = get_random_action(n)
        frames1, _, dones, *_ = env.step(action.item() if n == 1 else action)
        frames2, _, dones, *_ = env.step(noop_action.item() if n == 1 else noop_action)
        # Check termination condition
        if dones if n == 1 else dones.any():
            break
    return frames1, frames2

def preprocess(image, bkg_color=np.array([144, 72, 17])):
    """Preprocess a single frame - copied from a2c.ipynb"""
    cropped_image = image[34:-16, :]
    downsampled_image = cropped_image[::2, ::2]
    adjusted_image = downsampled_image - bkg_color
    grayscale_image = np.mean(adjusted_image, axis=-1)
    normalized_image = grayscale_image / 255.0
    return normalized_image

def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
    """Convert batch of frames to tensor - copied from a2c.ipynb"""
    batch_images = np.asarray(images)
    if len(batch_images.shape) < 5:
        batch_images = np.expand_dims(batch_images, 1)
    cropped_images = batch_images[:, :, 34:-16, :, :]
    downsampled_images = cropped_images[:, :, ::2, ::2, :]
    adjusted_images = downsampled_images - bkg_color
    grayscale_images = np.mean(adjusted_images, axis=-1)
    normalized_images = grayscale_images / 255.0
    batch_input = torch.from_numpy(normalized_images).float()
    batch_input = batch_input.permute(1, 0, 2, 3)
    return batch_input

# Network architectures remain the same
class SharedFeatureExtractor(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SharedFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(2, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.size = 7 * 7 * 64
        self.fc = nn.Linear(self.size, hidden_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc(x))
        return x

class ActorCriticNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super(ActorCriticNet, self).__init__()
        self.shared_extractor = SharedFeatureExtractor(hidden_dim=hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, 2)
        self.fc_critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared_extractor(x)
        logits = self.fc_actor(features)
        value = self.fc_critic(features)
        return logits, value

    def share_memory(self):
        self.shared_extractor.share_memory()
        self.fc_actor.share_memory()
        self.fc_critic.share_memory()

class Agent:
    """Interacts with and learns from the environment using A2C algorithm."""
    def __init__(
        self,
        network,
        optimizer=None,
        gamma=0.99,
        entropy_weight=0.01,
        value_loss_weight=1.0,
        n_steps=5,
        seed=42,
        T=None,  # 全局步数计数器（可在多进程共享）
        optimizer_lock=None,  # 用于保证 optimizer 更新的锁
    ):
        self.network = network
        self.optimizer = optimizer
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.value_loss_weight = value_loss_weight
        self.n_steps = n_steps
        self.seed = seed
        self.T = T  # 全局步数计数器
        self.optimizer_lock = optimizer_lock  # 确保optimizer更新的锁
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        self.states = deque(maxlen=n_steps)
        self.actions = deque(maxlen=n_steps)
        self.rewards = deque(maxlen=n_steps)
        self.values = deque(maxlen=n_steps)
        self.log_probs = deque(maxlen=n_steps)

        torch.manual_seed(seed)
        np.random.seed(seed)

    def get_action(self, frame1, frame2):
        """Choose action and return log_prob."""
        self.network.eval()
        state = preprocess_batch((frame1, frame2)).to(self.device)
        logits, value = self.network(state)
        probs = F.softmax(logits, dim=1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        self.network.train()
        return action.item(), log_prob.item(), value.item()

    def compute_loss(self, next_value, done):
        """Compute A2C loss."""
        R = torch.tensor(next_value, device=self.device, dtype=torch.float)
        if not done:
            R = R * self.gamma + next_value

        actor_loss = torch.tensor(0.0, device=self.device)
        critic_loss = torch.tensor(0.0, device=self.device)

        for i in reversed(range(len(self.rewards))):
            R = self.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            log_prob = torch.tensor(self.log_probs[i], device=self.device)
            actor_loss = actor_loss + (-log_prob * advantage.detach())
            critic_loss = critic_loss + (advantage**2)

        actor_loss = actor_loss / self.n_steps
        critic_loss = critic_loss / self.n_steps

        # Compute entropy loss
        state = torch.cat(list(self.states)).to(self.device)
        logits, _ = self.network(state)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs)).sum(1).mean()
        entropy_loss = -self.entropy_weight * entropy

        total_loss = actor_loss + critic_loss * self.value_loss_weight + entropy_loss
        return total_loss

    def learn(self, env, T, optimizer_lock):
        """Main learning loop for each worker."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        frame1, frame2 = perform_random_steps(env, nrand=5, parallel=False)
        done = False
        episode_reward = 0
        episode_length = 0

        while T.value < 1000000:  # 全局训练步数限制
            self.states.append(preprocess_batch((frame1, frame2)))
            action, log_prob, value = self.get_action(frame1, frame2)

            frame1, reward, done, *_ = env.step(Action.LEFTFIRE.value if action == 1 else Action.RIGHTFIRE.value)
            frame2, _, done, *_ = env.step(Action.NOOP.value)

            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)

            episode_reward += reward
            episode_length += 1

            if len(self.rewards) == self.n_steps or done:
                next_value = 0 if done else self.get_action(frame1, frame2)[2]  # 使用value估计
                loss = self.compute_loss(next_value, done)

                # 使用锁来确保只有一个进程更新全局网络
                with optimizer_lock:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
                    self.optimizer.step()

                # 清空缓存
                self.states.clear()
                self.actions.clear()
                self.rewards.clear()
                self.values.clear()
                self.log_probs.clear()

            # 更新全局步数计数器
            with T.get_lock():
                T.value += 1

            if done:
                print(f"Global step {T.value}, Episode reward: {episode_reward}, length: {episode_length}")
                frame1, frame2 = perform_random_steps(env, nrand=5, parallel=False)
                done = False
                episode_reward = 0
                episode_length = 0


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


def a3c(
    env_name="PongDeterministic-v4",
    hidden_dim=128,
    lr=1e-4,
    gamma=0.99,
    entropy_weight=0.001,
    value_loss_weight=1.0,
    n_steps=5,
    num_workers=8,
):
    """A3C training function."""
    # 创建全局网络
    global_network = ActorCriticNet(hidden_dim=hidden_dim)
    global_network.share_memory()  # 共享内存
    optimizer = SharedAdam(global_network.parameters(), lr=lr)
    # 创建全局计数器和锁
    T = mp.Value("i", 0)  # 共享的全局计数器
    optimizer_lock = mp.Lock()  # 确保optimizer更新的锁

    # 创建进程
    processes = []
    for rank in range(num_workers):
        env = gym.make(env_name)
        agent = Agent(
            network=global_network,
            optimizer=optimizer,
            gamma=gamma,
            entropy_weight=entropy_weight,
            value_loss_weight=value_loss_weight,
            n_steps=n_steps,
            seed=rank,  # 每个worker使用不同的随机种子
            T=T,
            optimizer_lock=optimizer_lock,
        )
        p = mp.Process(target=agent.learn, args=(env, T, optimizer_lock))
        p.start()
        processes.append(p)

    # 等待所有进程结束
    for p in processes:
        p.join()

    print("Training complete.")

if __name__ == "__main__":
    mp.set_start_method("spawn")  # 推荐使用spawn或forkserver
    a3c()
