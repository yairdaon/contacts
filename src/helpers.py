import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

## Important that theta is kept as the first entry.
JACOBIAN_COLS = ['theta', 'S1_0', 'I1_0', 'S2_0', 'I2_0']


def plot_G(G: pd.DataFrame, T: Optional[int] = None):
    """
    Plot S and I trajectories from the Jacobian computation.

    Parameters:
    -----------
    G : pd.DataFrame
        DataFrame with MultiIndex (t, j) containing columns 'S', 'I'
    T : int, optional
        Number of time steps. If None, inferred from data.

    Returns:
    --------
    matplotlib.figure.Figure
        Figure with two subplots showing S and I trajectories
    """
    if T is None:
        T = G.reset_index()['t'].max() + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract data for each region
    times = np.arange(T)

    # Region 1 data
    region1_data = G.loc[(slice(None), 1), :]
    S1_vals = region1_data['S'].values
    I1_vals = region1_data['I'].values

    # Region 2 data
    region2_data = G.loc[(slice(None), 2), :]
    S2_vals = region2_data['S'].values
    I2_vals = region2_data['I'].values

    # Plot Susceptible populations
    axes[0].plot(times, S1_vals, 'b-', linewidth=2, label='Region 1')
    axes[0].plot(times, S2_vals, 'r-', linewidth=2, label='Region 2')
    axes[0].set_xlabel('Time (weeks)')
    axes[0].set_ylabel('Susceptible fraction')
    axes[0].set_title('Susceptible Populations')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot Infected populations
    axes[1].plot(times, I1_vals, 'b-', linewidth=2, label='Region 1')
    axes[1].plot(times, I2_vals, 'r-', linewidth=2, label='Region 2')
    axes[1].set_xlabel('Time (weeks)')
    axes[1].set_ylabel('Infected fraction')
    axes[1].set_title('Infected Populations')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
