import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import traceback
import pdb
import os
import uuid
from typing import Callable, Dict, List
from scipy.linalg import cho_factor, cho_solve

from src import compute_g
from src.helper import JACOBIAN_COLS, plot_G

def compute_crlb(S0,
                 I0,
                 gamma: float,
                 theta: float,
                 T: int,
                 sigma: float,
                 beta0: float,
                 amplitude: float,
                 period: float,
                 noise: str = "mult",
                 pop_size: int = 1,
                 phase: float = 0.0,
                 phase2: float = None
                 ):
    """
    Compute Cramér-Rao Lower Bound for standard deviation of connectivity parameter theta.

    Parameters:
    -----------
    S1_0, S2_0, I1_0, I2_0 : float
        Initial conditions for susceptible and infected populations
    gamma : float
        Recovery rate
    theta : float
        Connectivity parameter
    T : int
        Number of time steps
    sigma : float
        Observation noise standard deviation
    beta0 : float
        Base transmission rate
    amplitude : float
        Seasonal amplitude
    period : float
        Seasonal period
    noise : str, default="mult"
        Noise model: "add" for additive, "mult" for multiplicative, "poisson" for Poisson
    pop_size : int, default=1
        Population scaling factor for Poisson model
    phase : float, default=0.0
        Phase offset for seasonal forcing (in radians).
        If phase2 is None, both regions use this phase.
    phase2 : float, optional
        Phase offset for region 2 (in radians).
        If None, region 2 uses the same phase as region 1.

    Returns:
    --------
    float
        the crlb for standard deviation of theta
    """
    
    df = compute_g.slow(
        S0=S0,
        I0=I0,
        gamma=gamma,
        theta=theta,
        T=T,
        beta0=beta0,
        amplitude=amplitude,
        period=period,
        phase=phase,
        phase2=phase2
    )
    
    mu = df['mu'].values
    G = df[JACOBIAN_COLS].values  # Shape: (2T, 5)
    
    if noise == "add":
        # Additive noise model: J = G^T G / sigma^2
        G /= sigma
        
    elif noise == "mult":
        # Multiplicative noise model: J = G^T W G / sigma^2
        # where W = diag((2*sigma^2 + 1) / (sigma^2 * mu_i(t)^2))

        # Compute weight matrix diagonal: (2*sigma^2 + 1) / (sigma^2 * mu_i(t)^2)
        sqrt_W = np.sqrt(2 * sigma**2 + 1) / sigma / mu
        
        # Apply weights: G_weighted = sqrt_weight * G (broadcasting)
        G = np.einsum('i, ij -> ij', sqrt_W, G)
                
    elif noise == "poisson":
        # Poisson noise model: J = G^T W G
        # where W = diag(pop_size / mu_i(t))

        # Compute weight matrix diagonal: N / mu_i(t)
        sqrt_W = np.sqrt(pop_size / mu)
        
        # Apply weights: G_weighted = sqrt_weight * G (broadcasting)
        G = np.einsum('i, ij -> ij', sqrt_W, G)

    elif noise == 'bin':
        sqrt_W = 1 /  mu  
        G = np.einsum('i, ij -> ij', sqrt_W, G)
        
    else:
        raise ValueError(f"Unknown noise model: {noise}. Use 'add', 'mult', or 'poisson'.") 

    # Fisher Information Matrix: J = G.T @ G since weights were
    # incorporated into G for each noise model.
    J = G.T @ G

    try:
        # Cholesky factorization LL^t = J
        L, low = cho_factor(J)

        # e1 = [1,0,0,0,0] since we have 5 variables. insert 1 where theta
        # is found in JACOBIAN_COLS
        e1 = np.zeros(J.shape[0])
        e1[0] = 1

        # find L^{-t}e1
        x = cho_solve((L, low), e1)

        ## CRLB for the standard deviation (!) of theta
        result = np.linalg.norm(x)

    except (ValueError, np.linalg.LinAlgError) as e:
        result = np.nan
        print(f"WARNING: Cholesky factorization failed: {e}")

    # Debug: save plot if result is NaN or inf
    if not np.isfinite(result):
        # Create debug directory if it doesn't exist
        debug_dir = 'crlb_res'
        os.makedirs(debug_dir, exist_ok=True)

        # Generate unique filename
        filename = os.path.join(debug_dir, f'{noise}_{uuid.uuid4().hex}.png')

        # Create and save plot with parameter information
        fig = plot_G(df, theta=theta, phase=phase, amplitude=amplitude, phase2=phase2)
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"WARNING: Non-finite CRLB ({result}). Debug plot saved to {filename}")

    return result
    
