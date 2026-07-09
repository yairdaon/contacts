
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

JACOBIAN_COLS = ['S1_0', 'I1_0', 'S2_0', 'I2_0', 'alpha', 'theta']

def compute_precision(S0,
                      I0,
                      gamma: float,
                      theta: float,
                      Ts: np.ndarray,
                      beta0: float,
                      delta: float,
                      rho: float,
                      phase: np.ndarray,
                      N: np.ndarray,
                      I_nat_pc,
                      alpha: float = 0.0,
                      k: float = 10.0
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
    np.ndarray of shape (2, 2)
        Trailing 2x2 upper-triangular block R_g of the R factor from the
        QR decomposition of sqrt(W) G, where G has columns ordered as
        (S1_0, I1_0, S2_0, I2_0, alpha, theta). This block encodes the
        per-season effective (alpha, theta) Fisher information via
        J_g = R_g^T R_g, but is returned in R_g form so the downstream
        aggregation can choose its scheme (sum of J_g, stacked QR, etc.).
        Returns a 2x2 zero matrix when no valid observations remain.
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
        N=N,
        I_nat_pc=I_nat_pc,
        alpha=alpha
    )

    mu = df['mu'].values
    G = df[JACOBIAN_COLS].values  # (2T, 6)

    valid = mu >= 1e-6
    mu = mu[valid]
    G = G[valid]

    if len(mu) == 0:
        return np.zeros((2, 2))

    # Per-observation Fisher weight w_i(t) = E[A_i(t)^2] under the
    # Gaussian pseudo-likelihood with Negative-Binomial variance:
    #   sigma^2 = rho mu + (rho mu)^2/k,
    #   a       = rho + 2 rho^2 mu/k,
    #   w       = rho^2/sigma^2 + a^2/(2 sigma^4).
    sim = rho * mu
    sigma2 = sim + sim ** 2 / k
    a = rho + 2.0 * rho ** 2 * mu / k
    w = rho ** 2 / sigma2 + a ** 2 / (2.0 * sigma2 ** 2)
    sqrt_w = np.sqrt(w)
    G = np.einsum('i, ij -> ij', sqrt_w, G)

    _, R = np.linalg.qr(G)
    return R[-2:, -2:]
    
