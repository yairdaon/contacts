import numpy as np
from scipy.stats import nbinom, poisson

def negbinom(observed, simulated, rho=1.0, theta=10.0):
    """Negative binomial log-likelihood with reporting rate (return negative for minimization)."""
    # Apply reporting rate to simulated mean
    mu = simulated["incidence"].values * rho + 1e-6
    k = theta
    loglik = nbinom.logpmf(observed["incidence"].values, n=k, p=k / (k + mu))
    return -np.sum(loglik)

def gaussian(observed, simulated, rho):
    """Gaussian likelihood with reporting rate (return negative log-likelihood)."""
    obs = observed["incidence"].values

    # Apply reporting rate to simulated data
    sim = simulated["incidence"].values * rho

    err = sim * (1-rho) + 1e-6
    ret = np.sum((obs - sim) ** 2 / 2 / err)
    assert not np.any(np.isnan(ret))
    return ret

def poisson(observed, simulated, rho):
    """
    Poisson observation model with reporting rate.
    
    Assumes: observed_incidence ~ Poisson(true_incidence * rho)
    
    Parameters:
    -----------
    observed : DataFrame with 'incidence' column
        Observed case counts (discrete)
    simulated : DataFrame with 'incidence' column  
        Predicted true incidence from SEIR model
    rho : float in (0, 1)
        Probability each case gets reported
        
    Returns:
    --------
    float : Negative log-likelihood (for minimization)
    """
    obs_counts = observed["incidence"].values.astype(int)
    true_incidence = simulated["incidence"].values
    
    # Predicted observed incidence (Poisson rate parameter)
    predicted_obs = true_incidence * rho
    
    # Add small epsilon to avoid numerical issues with zero rates
    predicted_obs = np.maximum(predicted_obs, 1e-10)
    
    # Poisson log-likelihood
    loglik = poisson.logpmf(obs_counts, mu=predicted_obs)
    
    # Handle any numerical issues (inf, nan)
    loglik = np.where(np.isfinite(loglik), loglik, -1e10)
    
    return -np.sum(loglik)  # Return negative for minimization

LOSSES = {
            "negbinom": negbinom,
            "gaussian": gaussian,
            "poisson_reporting": poisson,
        }
