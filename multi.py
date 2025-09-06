import pdb
import time

import numpy as np
import pandas as pd
from numba import jit, njit
from numba.core import types
from numpy import sin, cos, pi, log, exp, sqrt, ceil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("MacOSX")

# Days per year for seasonal forcing
PSI = 365

@jit(nopython=True)
def calc_log_betas(t,
                   beta0,
                   eps,
                   omega,
                   log_betas):
    """
    Calculate time-varying transmission rates with seasonal forcing.
    
    Parameters:
    -----------
    t : float
        Current time in days
    beta0 : array
        Baseline transmission rate per region (1/day)
    eps : array  
        Seasonal amplitude (0-1) per region
    omega : array
        Phase shift per region (fraction of year, 0=peak at t=0)
    log_betas : array
        Output array for log(beta(t)) values
    
    Notes:
    ------
    Seasonal pattern: β(t) = β₀ * (1 + ε * sin(2π(t/365 - ω)))
    where ω is phase as fraction of year (0.25 = peak in spring)
    """
    # Convert omega from fraction of year to phase in days
    log_betas[:] = log(beta0 * (1.0 + eps * sin(2.0 * pi * (t / PSI - omega))))
    

@jit(nopython=True)
def one_step(t,
             h,
             logS,
             logE,
             logI,
             C, ## Cases
             dlogS,
             dlogE,
             dlogI,
             dC,
             mu,
             sigma,
             nu,
             beta0,
             eps,
             omega,
             contact_matrix,
             population,
             log_betas):
    """Takes arrays logS, logE, logI and C (cases) at time
    t_init and modifies them inplace to have values of logS, logE,
    logI and CC (Cumulative Cases) at time t_init + h

    """

    calc_log_betas(t, beta0, eps, omega, log_betas)        

    # Vectorized computation: C @ (I / N) where I = exp(logI), N = population
    force_of_infection = contact_matrix @ (exp(logI) / population)  # Matrix multiplication: Σ_j (C_ij * I_j / N_j)
    
    # S -> E transition (infection) with contact matrix d(logS_i)/dt = -β_i * Σ_j (C_ij * I_j / N_j)
    dlogS[:] = mu * (population * exp(-logS) - 1) - exp(log_betas) * force_of_infection 
    dlogE[:] = exp(log_betas + logS - logE) * force_of_infection - sigma - mu 
    dlogI[:] = sigma * exp(logE - logI) - nu - mu 
    dC[:] = sigma * exp(logE) # E -> I transition (becoming infectious) / New cases (E -> I transition)

    # Apply derivatives
    dlogS[:] = dlogS * h
    dlogE[:] = dlogE * h 
    dlogI[:] = dlogI * h
    dC[:] = dC * h

    logS[:] += dlogS
    logE[:] += dlogE
    logI[:] += dlogI
    C[:] += dC

    
@jit(nopython=True)
def multi_seir(
        dt_euler,
        dt_output, 
        logSs,
        logEs,
        logIs,
        Cs,
        Ts,
        Fs,
        mu,
        nu,
        beta0,
        omega,
        eps,
        sigma,
        contact_matrix,
        population,
        dlogS,
        dlogE,
        dlogI,
        dC,
        log_betas):
    """Numba code to run the multi-region SEIR model. First line in
    all arrays is junk so truncate it.

    """

    ## Start at 1 cuz we have the first value as initial condition
    for iteration in range(1, logSs.shape[0]):
        t = (iteration-1) * dt_output
        out_t = iteration * dt_output


        calc_log_betas(out_t, beta0, eps, omega, log_betas)
        Fs[iteration, :] = exp(log_betas) ## Forcing
        Ts[iteration] = out_t ##  Time

        ## Get "initial conditions" for this output cycle
        logS = logSs[iteration-1, :].copy()
        logE = logEs[iteration-1, :].copy()
        logI = logIs[iteration-1, :].copy()
      
        ## Since C is a view, it accumulates new cases into the right index in Cs
        C = Cs[iteration,:] 
        C[:] = 0 
        
        while t < out_t:
            t_next = min(t + dt_euler, out_t)

            one_step(t,
                     t_next - t,
                     logS,
                     logE,
                     logI,
                     C,
                     dlogS,
                     dlogE,
                     dlogI,
                     dC,
                     mu,
                     sigma,
                     nu,
                     beta0,
                     eps,
                     omega,
                     contact_matrix,
                     population,
                     log_betas)

                     
            t = t_next


        logSs[iteration, :] = logS
        logEs[iteration, :] = logE
        logIs[iteration, :] = logI
                

def run(S_init,
        E_init,
        I_init,
        n_weeks=20,
        beta0=0.28,     # Flu: R₀≈1.4, recovery period ~5 days → β₀≈1.4/5=0.28/day
        sigma=1.0/3.0,  # Flu: incubation period ~3 days → σ=1/3/day  
        dt_output=7,    # Output every 7 days (weekly)
        dt_euler=5e-2,  # Euler step size (days)
        mu=1/(70*365),  # Birth/death rate: 70-year lifespan
        nu=1.0/5.0,     # Flu: infectious period ~5 days → ν=1/5/day
        omega=0.0,      # Phase: 0=winter peak, 0.25=spring, 0.5=summer, 0.75=fall
        eps=0.5,        # Seasonal amplitude: 50% variation (to be used in optimization)
        n_regions=2,
        contact_matrix=None,
        population=None,
        start_date="1900-01-01"):
    """
    Run multi-region SEIR simulation with seasonal forcing.
    
    Parameters:
    -----------
    S_init, E_init, I_init : array_like
        Initial conditions (absolute numbers, not fractions)
    n_weeks : int
        Simulation duration in weeks
    beta0 : float
        Baseline transmission rate (contacts/day). For flu: ~R₀/infectious_period
    sigma : float  
        Incubation rate E→I (1/day). For flu: ~1/3 day⁻¹
    nu : float
        Recovery rate I→R (1/day). For flu: ~1/5 day⁻¹  
    mu : float
        Birth/death rate (1/day). Typical: 1/(70*365)
    omega : float
        Seasonal phase (fraction of year). Should be constrained to [0,1]
        0=winter peak, 0.5=summer peak
    eps : float
        Seasonal amplitude (0-1). 0=no seasonality, 0.5=50% variation
    contact_matrix : 2D array, optional
        C[i,j] = relative contact rate from region j to region i.
        Dimensionless matrix in [0,1]. Units come from multiplication by beta0.
        Default: identity matrix (no inter-region transmission)
    population : array, optional
        Population size per region
        
    Returns:
    --------
    DataFrame with columns:
        - C{i}: Weekly incidence (new cases) in region i
        - F{i}: Transmission rate β(t) in region i  
        - S{i}, E{i}, I{i}: Compartment sizes in region i
    """
    
    logS = np.full((n_weeks, n_regions), np.nan)
    logE = np.full((n_weeks, n_regions), np.nan)
    logI = np.full((n_weeks, n_regions), np.nan)
    Cs = np.full((n_weeks, n_regions), np.nan)
    F = np.full((n_weeks, n_regions), np.nan)
    T = np.full(n_weeks, np.nan)

    logS[0, :] = log(S_init)
    logE[0, :] = log(E_init)
    logI[0, :] = log(I_init)
    Cs[0, :] = np.nan ## Initial count of new cases is C

    beta0 = np.full(n_regions, beta0)
    nu = np.full(n_regions, nu)
    omega = np.full(n_regions, omega)
    eps = np.full(n_regions, eps)
    log_betas = np.empty(n_regions)
    calc_log_betas(0,
                   beta0=beta0,
                   eps=eps,
                   omega=omega,
                   log_betas=log_betas)
    F[0, :] = exp(log_betas)

    # Default contact matrix: identity (no cross-region transmission)
    if contact_matrix is None:
        contact_matrix = np.eye(n_regions, dtype=np.float64)
    else:
        contact_matrix = np.array(contact_matrix, dtype=np.float64)
        assert contact_matrix.shape == (n_regions, n_regions), f"Contact matrix must be {n_regions}x{n_regions}"
    
    # Default population: all regions have population = 1.0
    if population is None:
        population = np.ones(n_regions, dtype=np.float64)
    else:
        population = np.array(population, dtype=np.float64)
        assert len(population) == n_regions, f"Population array must have {n_regions} elements"
  
    multi_seir(
        dt_euler=dt_euler,
        dt_output=dt_output,
        logSs=logS,
        logEs=logE,
        logIs=logI,
        Cs=Cs,
        Ts=T,
        Fs=F,
        mu=mu,
        nu=nu,
        beta0=beta0,
        omega=omega,
        eps=eps,
        sigma=sigma,
        contact_matrix=contact_matrix,
        population=population,
        dlogS=np.zeros(n_regions),
        dlogE=np.zeros(n_regions),
        dlogI=np.zeros(n_regions),
        dC=np.zeros(n_regions),
        log_betas=log_betas)

    C = pd.DataFrame(index=T, data=Cs, columns=[f'C{i}' for i in range(n_regions)])
    F = pd.DataFrame(index=T, data=F, columns=[f'F{i}' for i in range(n_regions)])
    S = pd.DataFrame(index=T, data=exp(logS), columns=[f'S{i}' for i in range(n_regions)])
    E = pd.DataFrame(index=T, data=exp(logE), columns=[f'E{i}' for i in range(n_regions)])
    I = pd.DataFrame(index=T, data=exp(logI), columns=[f'I{i}' for i in range(n_regions)])
    df = pd.concat([C, F, S, E, I], axis=1)
    assert not pd.isnull(df.iloc[1:,:].drop(C.columns, axis=1)).any().any()

    df.index = pd.date_range(start=start_date, periods=len(df), freq='7D')
    df.index.name = 'time'
    return df


def simulate(seed=43):
    
    # Set random seed for reproducible results
    np.random.seed(seed)

    # Implement random initial condition with small fraction infected/exposed
    total_pop = np.array([1e3, 1e4]) # np.array([1e6, 2e6])  # Normalized population
    
    # Small random fractions for initial conditions
    I_init = np.random.uniform(1e-6, 1e-4, size=2) * total_pop  # 0.0001% to 0.01% infected 
    E_init = np.random.uniform(1e-5, 1e-3, size=2) * total_pop # 0.001% to 0.1% exposed             
     
    # Susceptible = remaining population
    S_init = total_pop - I_init - E_init
    
    df = run(n_weeks=15,
             S_init=S_init,
             I_init=I_init,
             E_init=E_init)
    return df



def main():
    # Set dark mode style
    plt.style.use('dark_background')

    # Run simulation for 5 years with random initial conditions
    df = simulate()

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor('black')
    fig.suptitle('Two-Strain SEIR Model Simulation', fontsize=14, color='white')
    col1 = '#00FF7F'
    col2 = '#DA70D6'

    # Plot cases (incidence)
    ax = axes[0]
    ax.plot(df.index, df.C0, col1, label='Incidence 1', alpha=0.8, linewidth=2)  # Deep pink
    ax.plot(df.index, df.C1, col2, label='Incidence 2', alpha=0.8, linewidth=2)  # Dark orange
    ax.set_ylabel('Cases per Week', color='white')
    ax.legend()
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('black')
    # Plot transmission rates (seasonal forcing)
    ax = axes[1]
    ax.plot(df.index, df.F0, col1, label='β1(t)', alpha=0.8, linewidth=2)  # Spring green
    ax.plot(df.index, df.F1, col2, label='β2(t)', alpha=0.8, linewidth=2)  # Orchid
    ax.set_ylabel('β(t)', color='white')
    ax.set_xlabel('Time', color='white')
    ax.legend()
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('black')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        traceback.print_exc()  # Prints the full stack trace to stderr
        pdb.post_mortem()  # Starts debugger at the poi
