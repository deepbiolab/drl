import numpy as np
import pandas as pd
import matplotlib.collections as mc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


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


def visualize_tilings(tilings):
    """Plot each tiling as a grid."""
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    linestyles = ["-", "--", ":"]
    legend_lines = []

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, grid in enumerate(tilings):
        for x in grid[0]:
            l = ax.axvline(
                x=x,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                label=i,
            )
        for y in grid[1]:
            l = ax.axhline(
                y=y,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
            )
        legend_lines.append(l)
    ax.grid("off")
    ax.legend(
        legend_lines,
        ["Tiling #{}".format(t) for t in range(len(legend_lines))],
        facecolor="white",
        framealpha=0.9,
    )
    ax.set_title("Tilings")
    return ax  # return Axis object to draw on later, if needed


def visualize_encoded_samples(samples, encoded_samples, tilings, low=None, high=None):
    """Visualize samples by activating the respective tiles."""
    samples = np.array(samples)  # for ease of indexing

    # Show tiling grids
    ax = visualize_tilings(tilings)

    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Pre-render (invisible) samples to automatically set reasonable axis limits, 
        # and use them as (low, high)
        ax.plot(samples[:, 0], samples[:, 1], "o", alpha=0.0)
        low = [ax.get_xlim()[0], ax.get_ylim()[0]]
        high = [ax.get_xlim()[1], ax.get_ylim()[1]]

    # Map each encoded sample (which is really a list of indices) to 
    # the corresponding tiles it belongs to
    tilings_extended = [
        np.hstack((np.array([low]).T, grid, np.array([high]).T)) for grid in tilings
    ]  # add low and high ends
    tile_centers = [
        (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2
        for grid_extended in tilings_extended
    ]  # compute center of each tile
    tile_toplefts = [
        grid_extended[:, :-1] for grid_extended in tilings_extended
    ]  # compute topleft of each tile
    tile_bottomrights = [
        grid_extended[:, 1:] for grid_extended in tilings_extended
    ]  # compute bottomright of each tile

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for sample, encoded_sample in zip(samples, encoded_samples):
        for i, tile in enumerate(encoded_sample):
            # Shade the entire tile with a rectangle
            topleft = tile_toplefts[i][0][tile[0]], tile_toplefts[i][1][tile[1]]
            bottomright = (
                tile_bottomrights[i][0][tile[0]],
                tile_bottomrights[i][1][tile[1]],
            )
            ax.add_patch(
                Rectangle(
                    topleft,
                    bottomright[0] - topleft[0],
                    bottomright[1] - topleft[1],
                    color=colors[i],
                    alpha=0.33,
                )
            )

            # In case sample is outside tile bounds, it may not have been highlighted properly
            if any(sample < topleft) or any(sample > bottomright):
                # So plot a point in the center of the tile and draw a connecting line
                cx, cy = tile_centers[i][0][tile[0]], tile_centers[i][1][tile[1]]
                ax.add_line(Line2D([sample[0], cx], [sample[1], cy], color=colors[i]))
                ax.plot(cx, cy, "s", color=colors[i])

    # Finally, plot original samples
    ax.plot(samples[:, 0], samples[:, 1], "o", color="r")

    ax.margins(x=0, y=0)  # remove unnecessary margins
    ax.set_title("Tile-encoded samples")
    return ax
