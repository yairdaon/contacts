import numpy as np
from scipy.stats import nbinom, poisson

RHO = 0.8  # Default reporting rate

def negbinom(observed, simulated, rho=RHO, theta=10.0):
    """
    Negative binomial loss function with reporting rate correction.
    
    Models overdispersed count data with under-reporting. Returns negative
    log-likelihood for minimization in optimization routines.
    
    Parameters:
    -----------
    observed : DataFrame
        Observed incidence data with 'incidence' column
    simulated : DataFrame
        Simulated incidence data with 'incidence' column
    rho : float
        Reporting rate (0 < rho <= 1)
    theta : float
        Overdispersion parameter (higher = less overdispersed)
        
    Returns:
    --------
    float
        Negative log-likelihood for minimization
    """
    # Apply reporting rate to simulated mean
    mu = simulated["incidence"].values * rho + 1e-6
    k = theta
    loglik = nbinom.logpmf(observed["incidence"].values, n=k, p=k / (k + mu))
    return -np.sum(loglik)

def gaussian(observed, simulated, rho=RHO):
    """
    Gaussian loss function with heteroscedastic error model.
    
    Uses reporting rate to adjust both mean and variance of the 
    observation model. Error variance scales with predicted incidence.
    
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
        Weighted sum of squared errors for minimization
    """
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

# Dictionary mapping loss function names to implementations
LOSSES = {
    "negbinom": negbinom,
    "gaussian": gaussian, 
    "poisson_reporting": poisson,
}
