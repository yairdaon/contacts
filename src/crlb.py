
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
from src.helper import JACOBIAN_COLS, plot_G

def compute_crlb(S0,
                 I0,
                 gamma: float,
                 theta: float,
                 Ts: np.ndarray,
                 beta0: float,
                 eps: float,
                 rho: float,
                 phase: np.ndarray
                 ):
    """
    Compute Cramér-Rao Lower Bound for standard deviation of connectivity parameter theta.

    Parameters:
    -----------
    S0 : np.ndarray
        Initial susceptible fractions per region
    I0 : np.ndarray
        Initial infected fractions per region
    gamma : float
        Recovery rate
    theta : float
        Connectivity parameter
    Ts : np.ndarray
        Array of time values
    beta0 : float
        Base transmission rate
    eps : float
        Seasonal amplitude
    rho : float
        Reporting rate
    phase : np.ndarray
        Phase offsets per region (in radians)

    Returns:
    --------
    float
        CRLB for variance of theta (i.e., [J^{-1}]_{11})
    """
    assert S0.shape == phase.shape
    assert I0.shape == phase.shape
    assert len(Ts.shape) == 1
    assert len(S0.shape) == 1
    df = compute_g.contacts(
        S0=S0,
        I0=I0,
        gamma=gamma,
        theta=theta,
        Ts=Ts,
        beta0=beta0,
        eps=eps,
        phase=phase
    )
    
    mu = df['mu'].values
    G = df[JACOBIAN_COLS].values  # Shape: (2T, 5)
    if np.any(mu == 0):
        return np.inf
    
    ## Binomial noise
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

    return w @ w
    

