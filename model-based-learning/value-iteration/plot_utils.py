import pandas as pd
import matplotlib.pyplot as plt

def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.title("Scores")
    plt.plot(scores)
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    return rolling_mean