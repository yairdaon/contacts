import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import traceback
import pdb
import os
import uuid
from typing import Callable, Dict, List
from scipy.linalg import cho_factor, cho_solve, solve

from src import compute_g
from src.diseases import flu
from src.helper import JACOBIAN_COLS, plot_G

def compute_crlb(S0,
                 I0,
                 model,
                 gamma: float,
                 theta: float,
                 T: int,
                 beta0: float,
                 amplitude: float,
                 period: float,
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
    beta0 : float
        Base transmission rate
    amplitude : float
        Seasonal amplitude
    period : float
        Seasonal period
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
    assert model in ('cross', 'contacts')
    G_fun = compute_g.cross if model == 'cross' else compute_g.contacts
    
    df = G_fun(
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
    if np.any(mu == 0):
        return np.inf
    
    ## Binomial noise
    rho = flu.rho
    sqrt_W = 1 / mu  
    G = np.einsum('i, ij -> ij', sqrt_W, G)
    
    # Fisher Information Matrix: J = G.T @ G 
    factor = ( 1+rho ) / ( 2*(1-rho) )
    J = factor * G.T @ G
        
    # Cholesky factorization LL^T = J (L is lower triangular)
    L = np.linalg.cholesky(J)
    
    # e1 = [1,0,0,0,0] since theta is first in JACOBIAN_COLS
    e1 = np.zeros(J.shape[0])
    e1[0] = 1
    
    # Solve Lw = e1 for w = L^{-1}e1
    # CRLB for variance is ||L^{-1}e1||^2 = [J^{-1}]_{11}
    w = np.linalg.solve(L, e1)
    
    # CRLB for the standard deviation of theta
    result = np.linalg.norm(w)
    return result
    

