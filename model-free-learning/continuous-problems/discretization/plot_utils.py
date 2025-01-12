import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_samples(samples, discretized_samples, grid, low, high) -> None:
    """
    Visualize original and discretized samples on a 2D grid.
    Parameters:
    samples: Array of shape (n_samples, 2) containing original sample points
    discretized_samples: Array of shape (n_samples, 2) containing discretized sample indices
    grid: List of two arrays representing grid split points for x and y axes
    low: Optional, minimum values for axes [x_min, y_min]
    high: Optional, maximum values for axes [x_max, y_max]
    """
    # Input validation
    if samples.shape != discretized_samples.shape:
        raise ValueError("Shapes of samples and discretized_samples must match")
    if len(grid) != 2:
        raise ValueError("Grid must contain two arrays (split points for x and y axes)")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set grid
    for i, g in enumerate(grid):
        if i == 0:
            ax.xaxis.set_major_locator(plt.FixedLocator(g))
        else:
            ax.yaxis.set_major_locator(plt.FixedLocator(g))
    ax.grid(True)

    # Set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Use grid boundaries as ranges
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    # Calculate grid cell centers
    grid_extended = np.hstack(
        (
            np.array([low]).T,  # Add minimum boundary
            grid,  # Original grid points
            np.array([high]).T,  # Add maximum boundary
        )
    )

    # Calculate center of each grid cell
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2

    # Map discretized indices to grid center points
    locs = np.stack(
        [grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))]
    ).T

    # Plot sample points
    original = ax.plot(samples[:, 0], samples[:, 1], "o", label="Original", alpha=0.6)
    discretized = ax.plot(locs[:, 0], locs[:, 1], "s", label="Discretized", alpha=0.6)

    # Add connecting lines
    lines = mc.LineCollection(
        list(zip(samples, locs)), colors="orange", alpha=0.3, label="Mapping"
    )
    ax.add_collection(lines)

    # Set figure properties
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")

    ax.set_title("Sample Discretization Visualization")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.title("Scores")
    plt.plot(scores)
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    return rolling_mean


# Visualize the learned Q-table
def plot_q_table(q_table, state_grid):
    """Visualize max Q-value for each state and corresponding action."""
    state_size = tuple(len(splits) + 1 for splits in state_grid)
    state_value = np.zeros(state_size)
    policy = np.zeros(state_size)
    for state, action in q_table.items():
        state_value[state] = np.max(action)
        policy[state] = np.argmax(action)

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(state_value, cmap="jet")
    _ = fig.colorbar(cax)
    for x in range(state_value.shape[0]):
        for y in range(state_value.shape[1]):
            ax.text(
                x,
                y,
                policy[x, y],
                color="white",
                horizontalalignment="center",
                verticalalignment="center",
            )
    ax.grid(False)
    ax.set_title(f"Q-table, size: {state_size}")
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
