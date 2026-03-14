import numpy as np
from scipy.stats import nbinom, poisson
from scipy.special import xlogy

EPS = 1e-6

def gaussian(observed, simulated, rho):
    """
    Negative log-likelihood for Gaussian observation model.

    L = (1/2) Σ [log(2πσ²) + (Y - ρμ)² / σ²]

    where σ² = ρ(1-ρ)μ is the observation variance.

    Parameters:
    -----------
    observed : DataFrame
        Observed incidence data with 'incidence' column
    simulated : DataFrame
        Simulated incidence data with 'incidence' column
    rho : float
        Reporting rate (0 < rho <= 1)

    Returns:
    --------
    float
        Negative log-likelihood for minimization
    """
    obs = observed["incidence"].values

    # μ = true incidence, ρμ = expected observation
    mu = simulated["incidence"].values
    sim = mu * rho

    # σ² = ρ(1-ρ)μ
    sigma2 = sim * (1 - rho)

    # Negative log-likelihood: (1/2) Σ [log(2πσ²) + residual²/σ²]
    residual = obs - sim
    log_term = xlogy(sigma2, 2 * np.pi * sigma2)
    quad_term = residual ** 2 
    ret = np.sum( (log_term+quad_term) / (sigma2+EPS) ) / 2 
    return ret
