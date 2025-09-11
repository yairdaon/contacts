import numpy as np
import pandas as pd
from numba import jit
from numpy import sin, cos, pi, log, exp
import time

from CyRK import nbsolve_ivp

# Import from multi.py
from src.multi import calc_log_betas

# Days per year for seasonal forcing
PSI = 365


@jit(nopython=True, cache=True)
def log_der(t, y, mu, sigma, nu, beta0, eps, omega, contact_matrix, population):
    """
    Calculate derivatives for SEIR system in log space.
    
    Parameters:
    -----------
    t : float
        Current time in days
    y : array
        State vector [logS_0, logS_1, ..., logE_0, logE_1, ..., logI_0, logI_1, ...]
    mu : float
        Birth/death rate
    sigma : float
        Incubation rate (E -> I)
    nu : array
        Recovery rate (I -> R) per region
    beta0 : array
        Baseline transmission rate per region
    eps : array
        Seasonal amplitude per region
    omega : array
        Phase shift per region
    contact_matrix : 2D array
        Contact matrix between regions
    population : array
        Population per region
    
    Returns:
    --------
    dy : array
        Derivatives [dlogS/dt, dlogE/dt, dlogI/dt]
    """
    n_regions = len(population)
    
    # Extract state variables from flattened array
    logS = y[0:n_regions]
    logE = y[n_regions:2*n_regions]
    logI = y[2*n_regions:3*n_regions]
    
    # Calculate time-varying transmission rates
    log_betas = np.empty(n_regions)
    calc_log_betas(t, beta0, eps, omega, log_betas)
    
    # Force of infection calculation (copied from one_step)
    force_of_infection = contact_matrix @ (exp(logI) / population)
    
    # Calculate derivatives (copied from one_step)
    dlogS = mu * (population * exp(-logS) - 1) - exp(log_betas) * force_of_infection
    dlogE = exp(log_betas + logS - logE) * force_of_infection - sigma - mu
    dlogI = sigma * exp(logE - logI) - nu - mu
    
    # Return concatenated derivatives using pre-allocated array
    dy = np.empty(3 * n_regions)
    dy[0:n_regions] = dlogS
    dy[n_regions:2*n_regions] = dlogE
    dy[2*n_regions:3*n_regions] = dlogI
    return dy


def run_rk(S_init,
           E_init,
           I_init,
           dt_step=1,
           dt_output=7,
           n_weeks=20,
           beta0=0.28,
           sigma=1.0/3.0,
           mu=1/(70*365),
           nu=1.0/5.0,
           omega=0.0,
           eps=0.5,
           contact_matrix=None,
           population=None,
           start_date="1900-01-01"):
    """
    Run multi-region SEIR simulation using CyRK's nbsolve_ivp with RK method.
    
    Parameters:
    -----------
    dt_step : float
        Integration step size (days) for RK method
    dt_output : float  
        Output sampling interval (days), typically 7 for weekly output
    n_weeks : int
        Number of weeks to simulate
    
    Returns:
    --------
    DataFrame with results sampled at dt_output intervals, aligned with Euler method
    """
    n_regions = len(S_init)
    
    # Default population and contact matrix
    if population is None:
        population = np.ones(n_regions, dtype=np.float64)
    else:
        population = np.array(population, dtype=np.float64)
        
    if contact_matrix is None:
        contact_matrix = np.eye(n_regions, dtype=np.float64)
    else:
        contact_matrix = np.array(contact_matrix, dtype=np.float64)
    
    # Convert to arrays and ensure proper types
    omega = np.array(omega, dtype=np.float64) if np.isscalar(omega) else np.array(omega, dtype=np.float64)

    # Initial state vector in log space
    y0 = np.concatenate([
        log(S_init * population),
        log(E_init * population),
        log(I_init * population),
    ])
    
    t_span = (0.0, float(n_weeks * dt_output))
    t_eval = np.arange(0, n_weeks * dt_output, dt_step, dtype=np.float64)

    result = nbsolve_ivp(
        log_der,
        t_span,
        y0,
        args=(mu, sigma, nu, beta0, eps, omega, contact_matrix, population),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-15
    )
    assert result.success

    # Extract solution
    n_times = len(result.t)
    logS = result.y[0:n_regions, :].T
    logE = result.y[n_regions:2*n_regions, :].T
    logI = result.y[2*n_regions:3*n_regions, :].T
    
    # Convert back to regular space
    S = exp(logS)
    E = exp(logE)
    I = exp(logI)
    
    # Calculate instantaneous incidence rate (E -> I transitions per day)
    incidence_rate = sigma * E  # Cases per day at each time point


    # Convert to DataFrame with proper time index for rolling calculations
    time_full = pd.date_range(start=start_date, periods=n_times, freq=f'{dt_step}D')
    incidence_df = pd.DataFrame(incidence_rate, index=time_full)
    
    # Calculate cumulative incidence over rolling dt_output windows
    # Rolling sum over dt_output days, then multiply by dt_step to get total cases
    window_size = int(dt_output / dt_step)  # Number of dt_step intervals in dt_output window
    C_rolling = incidence_df.rolling(window=window_size, min_periods=1).sum() * dt_step
    
    # Set first time point to zero (no cases at t=0 by convention)
    C_rolling.iloc[0, :] = 0.0

    # Sample at dt_output intervals to align with Euler method
    # Create sampling indices: every dt_output/dt_step points
    sample_stride = int(dt_output / dt_step)
    sample_indices = np.arange(0, n_times, sample_stride)
    
    # If the last point doesn't align exactly, include it
    if sample_indices[-1] != n_times - 1:
        sample_indices = np.append(sample_indices, n_times - 1)
    
    # Limit to n_weeks points (to match Euler output)
    sample_indices = sample_indices[:n_weeks]
    
    # Sample all arrays at these indices
    S_sampled = S[sample_indices, :]
    E_sampled = E[sample_indices, :]
    I_sampled = I[sample_indices, :]
    C_sampled = C_rolling.iloc[sample_indices].values
    
    # Calculate time-varying transmission rates at sampled points
    F_sampled = np.zeros_like(S_sampled)
    log_betas = np.empty(n_regions)
    for i, idx in enumerate(sample_indices):
        t_current = result.t[idx]
        calc_log_betas(t_current, beta0, eps, omega, log_betas)
        F_sampled[i, :] = exp(log_betas)
    
    # Create DataFrames with proper time index aligned with Euler method
    n_sampled = len(sample_indices)
    time_index = pd.date_range(start=start_date, periods=n_sampled, freq=f'{dt_output}D')
    
    C_df = pd.DataFrame(C_sampled, index=time_index, columns=[f'C{i}' for i in range(n_regions)])
    F_df = pd.DataFrame(F_sampled, index=time_index, columns=[f'F{i}' for i in range(n_regions)])
    S_df = pd.DataFrame(S_sampled, index=time_index, columns=[f'S{i}' for i in range(n_regions)])
    E_df = pd.DataFrame(E_sampled, index=time_index, columns=[f'E{i}' for i in range(n_regions)])
    I_df = pd.DataFrame(I_sampled, index=time_index, columns=[f'I{i}' for i in range(n_regions)])
    
    # Combine all results
    df = pd.concat([C_df, F_df, S_df, E_df, I_df], axis=1)
    df.index.name = 'time'
    
    return df