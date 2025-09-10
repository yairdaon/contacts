import numpy as np
from scipy.stats import nbinom


def least_squares_loss(observed, simulated):
    """Least squares on log-incidence."""
    obs = np.log1p(observed["incidence"].values)
    sim = np.log1p(simulated["incidence"].values)
    return np.sum((obs - sim) ** 2)

def negbinom_loss(observed, simulated, theta=10.0):
    """Negative binomial log-likelihood (return negative for minimization)."""
    mu = simulated["incidence"].values + 1e-6
    k = theta
    loglik = nbinom.logpmf(observed["incidence"].values, n=k, p=k / (k + mu))
    return -np.sum(loglik)

def gaussian_loss(observed, simulated, sigma=1.0):
    """Gaussian likelihood (return negative log-likelihood)."""
    obs = observed["incidence"].values
    sim = simulated["incidence"].values
    return np.sum((obs - sim) ** 2) / (2 * sigma ** 2)

LOSSES = {
            "lsq": least_squares_loss,
            "negbinom": negbinom_loss,
            "gaussian": gaussian_loss,
        }
