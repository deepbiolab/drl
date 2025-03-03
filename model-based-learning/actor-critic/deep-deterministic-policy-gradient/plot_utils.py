import pandas as pd
import matplotlib.pyplot as plt

def plot_scores(scores, actor_losses=None, critic_losses=None, rolling_window=100):
    """Plot scores and DDPG losses with rolling mean using specified window.
    
    Args:
        scores (list): Episode scores
        actor_losses (list, optional): Actor network losses
        critic_losses (list, optional): Critic network losses
        rolling_window (int): Window size for rolling mean calculation
    """
    # Create figure with 2 rows if we have losses to plot
    if actor_losses is not None or critic_losses is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 1])
    else:
        fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # Plot scores on top subplot
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color='tab:blue')
    ax1.plot(scores, color='tab:blue', alpha=0.3, label='Raw Scores')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    ax1.plot(rolling_mean, color='tab:blue', label=f'Rolling Mean (window={rolling_window})')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_title("Training Scores")
    
    # Plot losses if provided
    if actor_losses is not None or critic_losses is not None:
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        
        if actor_losses is not None:
            actor_mean = pd.Series(actor_losses).rolling(rolling_window).mean()
            ax2.plot(actor_losses, color='tab:green', alpha=0.2, label='Actor Loss')
            ax2.plot(actor_mean, color='tab:green', label='Actor Loss Mean')
            
        if critic_losses is not None:
            critic_mean = pd.Series(critic_losses).rolling(rolling_window).mean()
            ax2.plot(critic_losses, color='tab:red', alpha=0.2, label='Critic Loss')
            ax2.plot(critic_mean, color='tab:red', label='Critic Loss Mean')
            
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        ax2.set_title("Actor and Critic Losses")
    
    plt.tight_layout()
    return rolling_mean

