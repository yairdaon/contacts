import pandas as pd
from itertools import product
import numpy as np
from scipy.special import logit, expit
import matplotlib.pyplot as plt
from typing import Optional

## Important that theta is kept as the first entry.
JACOBIAN_COLS = ['theta', 'S1_0', 'I1_0', 'S2_0', 'I2_0']
YEAR_LENGTH = 365.25


def calc_t(date):
    """Convert datetime to continuous time where Jan 1 = integer."""
    return date.dt.year + (date.dt.dayofyear - 1) / YEAR_LENGTH


def a2s(x):
    return np.array2string(x, precision=3)


def fwd(x, lower, upper):
    """Transform values from [lower, upper] to the real line using scaled logit"""
    # Scale to [0, 1] then apply logit
    scaled = (x - lower) / (upper - lower)
    return logit(scaled)


def bckwd(y, lower, upper):
    """Transform values from real line to [lower, upper] using scaled expit"""
    # Apply expit then scale to [lower, upper]
    scaled = expit(y)
    return lower + (upper - lower) * scaled



def plot_G(G: pd.DataFrame, T: Optional[int] = None, theta: Optional[float] = None,
           phase: Optional[float] = None, amplitude: Optional[float] = None, phase2: Optional[float] = None):
    """
    This should fail, we'll need to fix phase, phase2
    Plot S and I trajectories from the Jacobian computation.

    Parameters:
    -----------
    G : pd.DataFrame
        DataFrame with MultiIndex (t, j) containing columns 'S', 'I'
    T : int, optional
        Number of time steps. If None, inferred from data.
    theta : float, optional
        Connectivity parameter (for title)
    phase : 
        Phase offset 
    amplitude : float, optional
        Seasonal amplitude (for title)
   
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

    # Add overall title with parameters
    title_parts = []
    if theta is not None:
        title_parts.append(f'θ={theta:.2e}')
    if phase is not None:
        if phase2 is not None and phase2 != phase:
            title_parts.append(f'φ₁={phase:.3f}, φ₂={phase2:.3f} rad')
        else:
            title_parts.append(f'φ={phase:.3f} rad')
    if amplitude is not None:
        title_parts.append(f'A={amplitude:.2f}')

    if title_parts:
        fig.suptitle(' | '.join(title_parts), fontsize=14, y=1.02)

    plt.tight_layout()
    return fig

