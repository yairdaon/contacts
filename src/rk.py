import numpy as np
import pandas as pd
from numba import jit
from numpy import sin, cos, pi, log, exp
import time

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
    Integration step is dt_output (usually 7 days, i.e. weekly output).
    
    Parameters are the same as multi.py run() function, except dt_euler is removed.
    
    Returns:
    --------
    DataFrame with weekly results and timing information
    """
    try:
        from CyRK import nbsolve_ivp
    except ImportError:
        raise ImportError("CyRK package is required. Install with: pip install CyRK")
    
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
    beta0 = np.array(beta0, dtype=np.float64) if np.isscalar(beta0) else np.array(beta0, dtype=np.float64)
    omega = np.array(omega, dtype=np.float64) if np.isscalar(omega) else np.array(omega, dtype=np.float64)
    eps = np.array(eps, dtype=np.float64) if np.isscalar(eps) else np.array(eps, dtype=np.float64)
    nu = np.full(n_regions, nu, dtype=np.float64)
    
    # Convert initial conditions to absolute numbers
    S_init_abs = np.array(S_init, dtype=np.float64) * population
    E_init_abs = np.array(E_init, dtype=np.float64) * population
    I_init_abs = np.array(I_init, dtype=np.float64) * population
    
    # Initial state vector in log space
    y0 = np.concatenate([
        log(S_init_abs),
        log(E_init_abs),
        log(I_init_abs)
    ])
    
    # Time span and evaluation points (weekly: 7-day steps)
    t_span = (0.0, float(n_weeks * dt_output))
    t_eval = np.arange(0, n_weeks * dt_output, dt_output, dtype=np.float64)
    
    # Solve using CyRK
    start_time = time.time()
    
    result = nbsolve_ivp(
        log_der,
        t_span,
        y0,
        args=(mu, sigma, nu, beta0, eps, omega, contact_matrix, population),
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8
    )
    
    end_time = time.time()
    
    if not result.success:
        raise RuntimeError(f"Integration failed: {result.message}")
    
    # Extract solution
    n_times = len(result.t)
    logS = result.y[0:n_regions, :].T
    logE = result.y[n_regions:2*n_regions, :].T
    logI = result.y[2*n_regions:3*n_regions, :].T
    
    # Convert back to regular space
    S = exp(logS)
    E = exp(logE)
    I = exp(logI)
    
    # Calculate new cases (E -> I transitions) more accurately
    # New cases = integral of sigma * E over each week
    C = np.zeros_like(S)
    C[1:, :] = (E[:-1,:] + E[1:,:])/2 * sigma * dt_output
    # First week uses initial - No cases at t=0

    # Calculate time-varying transmission rates for output
    F = np.zeros_like(S)
    log_betas = np.empty(n_regions)
    for i in range(n_times):
        t_current = result.t[i]
        calc_log_betas(t_current, beta0, eps, omega, log_betas)
        F[i, :] = exp(log_betas)
    
    # Create DataFrames
    time_index = pd.date_range(start=start_date, periods=n_times, freq=f'{dt_output}D')
    
    C_df = pd.DataFrame(C, index=time_index, columns=[f'C{i}' for i in range(n_regions)])
    F_df = pd.DataFrame(F, index=time_index, columns=[f'F{i}' for i in range(n_regions)])
    S_df = pd.DataFrame(S, index=time_index, columns=[f'S{i}' for i in range(n_regions)])
    E_df = pd.DataFrame(E, index=time_index, columns=[f'E{i}' for i in range(n_regions)])
    I_df = pd.DataFrame(I, index=time_index, columns=[f'I{i}' for i in range(n_regions)])
    
    # Combine all results
    df = pd.concat([C_df, F_df, S_df, E_df, I_df], axis=1)
    df.index.name = 'time'
    
    return df, end_time - start_time


if __name__ == "__main__":
    # Simple test
    print("Testing RK integration...")

    # Simple 2-region test
    S_init = [0.99, 0.98]
    E_init = [0.005, 0.01]
    I_init = [0.005, 0.01]

    df, timing = run_rk(S_init, E_init, I_init, n_weeks=10)
    print(f"Integration completed in {timing:.4f} seconds")
    print("Sample output:")
    print(df.head())