
import numpy as np
from numpy.linalg import LinAlgError
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

JACOBIAN_COLS = ['S1_0', 'I1_0', 'S2_0', 'I2_0', 'theta']

def compute_precision(S0,
                      I0,
                      gamma: float,
                      theta: float,
                      Ts: np.ndarray,
                      beta0: float,
                      delta: float,
                      rho: float,
                      phase: np.ndarray,
                      N: np.ndarray
                      ):
    """`
    Compute the precision of the Cramér-Rao Lower Bound for theta.

    Parameters:
    -----------
    S0 : np.ndarray
        Initial susceptible counts per region
    I0 : np.ndarray
        Initial infected counts per region
    gamma : float
        Recovery rate
    theta : float
        Connectivity parameter
    Ts : np.ndarray
        Array of time values
    beta0 : float
        Base transmission rate
    delta : float
        Seasonal amplitude
    rho : float
        Reporting rate
    phase : np.ndarray
        Phase offsets per region (in radians)
    N : np.ndarray
        Population sizes per region

    Returns:
    --------
    float
        upper bound for the precision of theta
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
        delta=delta,
        phase=phase,
        N=N
    )
    
    mu = df['mu'].values
    G = df[JACOBIAN_COLS].values  # Shape: (2T, 5)

    # Ignore negligible incidence
    # valid = mu >= 1e-6
    # mu = mu[valid]
    # G = G[valid]

    # if len(mu) == 0:
    #     return np.inf

    # Per-observation Fisher weight w_i(t) = E[A_i(t)^2] for the
    # Gaussian approximation Y_i ~ N(rho mu, rho(1-rho) mu) with mu =
    # mu(phi): w_i = rho / ((1-rho) mu) + 1 / (2 mu^2)
    w = rho / ((1.0 - rho) * mu) + 1.0 / (2.0 * mu ** 2)
    sqrt_w = np.sqrt(w)
    G = np.einsum('i, ij -> ij', sqrt_w, G)
    
    _, R = np.linalg.qr(G)
    return R[-1,-1]**2
    
