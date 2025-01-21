import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from agent import Agent
from monitor import interact

def run_experiments(n_experiments=100, n_episodes=5000, window=100):
    all_avg_rewards = []
    all_best_rewards = []
    
    for i in range(n_experiments):
        print(f"\nExperiment {i+1}/{n_experiments}")
        env = gym.make("Taxi-v3")
        agent = Agent()
        avg_rewards, best_avg_reward = interact(env, agent, num_episodes=n_episodes, window=window)
        all_avg_rewards.append(list(avg_rewards))
        all_best_rewards.append(best_avg_reward)
        env.close()
    
    all_avg_rewards = np.array(all_avg_rewards)
    
    mean_rewards = np.mean(all_avg_rewards, axis=0)
    std_rewards = np.std(all_avg_rewards, axis=0)
    
    confidence = 0.95
    z = 1.96
    margin_of_error = z * (std_rewards / np.sqrt(n_experiments))
    
    plt.figure(figsize=(12, 6))
    episodes = np.arange(len(mean_rewards)) * window
    
    plt.plot(episodes, mean_rewards, label='Mean Reward', color='blue')
    plt.fill_between(episodes, 
                     mean_rewards - margin_of_error,
                     mean_rewards + margin_of_error,
                     alpha=0.2, color='blue')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Training Performance over {n_experiments} Experiments')
    plt.legend()
    plt.grid(True)
    
    final_mean = mean_rewards[-1]
    final_std = std_rewards[-1]
    best_mean = np.mean(all_best_rewards)
    best_std = np.std(all_best_rewards)
    
    stats_text = f'Final Average Reward: {final_mean:.3f} ± {final_std:.3f}\n' \
                 f'Best Average Reward: {best_mean:.3f} ± {best_std:.3f}'
    plt.text(0.02, 0.02, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='bottom')
    
    plt.savefig('training_performance.png')
    plt.close()
    
    print("\nTraining Statistics:")
    print(f"Final Average Reward: {final_mean:.3f} ± {final_std:.3f}")
    print(f"Best Average Reward: {best_mean:.3f} ± {best_std:.3f}")
    
    return mean_rewards, std_rewards, all_best_rewards

if __name__ == "__main__":
    mean_rewards, std_rewards, all_best_rewards = run_experiments(
        n_experiments=100,
        n_episodes=5000,
        window=100
    )