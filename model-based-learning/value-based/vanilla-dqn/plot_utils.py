import pandas as pd
import matplotlib.pyplot as plt

def plot_scores(scores, losses=None, rolling_window=100):
    """Plot scores and optional losses with rolling mean using specified window."""
    fig, ax1 = plt.subplots()
    
    # Plot scores on left y-axis
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color='tab:blue')
    ax1.plot(scores, color='tab:blue', alpha=0.3)
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    ax1.plot(rolling_mean, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Plot losses on right y-axis if provided
    if losses is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color='tab:orange')
        ax2.plot(losses, color='tab:orange', alpha=0.3)
        loss_mean = pd.Series(losses).rolling(rolling_window).mean()
        ax2.plot(loss_mean, color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Training Progress")
    fig.tight_layout()
    return rolling_mean