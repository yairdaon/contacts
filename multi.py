import pdb
import time

import numpy as np
import pandas as pd
from numba import jit, njit
from numba.core import types
from numpy import sin, cos, pi, log, exp, sqrt, ceil


@jit(nopython=True)
def calc_log_betas(t,
                   beta0,
                   eps,
                   psi,
                   omega,
                   log_betas):
    log_betas[:] = log(beta0 * (1.0 + eps * sin(2.0 * pi / psi * (t - omega * psi))))
    

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
             sigma_rate,
             nu,
             beta0,
             eps,
             psi,
             omega,
             log_betas):
    """Takes arrays logS, logE, logI and C (cases) at time
    t_init and modifies them inplace to have values of logS, logE,
    logI and CC (Cumulative Cases) at time t_init + h

    """

    # Zero out derivatives
    dlogS[:] = 0
    dlogE[:] = 0  
    dlogI[:] = 0
    dC[:] = 0
    
    calc_log_betas(t, beta0, eps, psi, omega, log_betas)        

    # Birth and death terms for S
    dlogS[:] += (exp(log(mu) - logS) - mu) * h  

    # S -> E transition (infection)
    infection_rate = exp(log_betas + logI)
    dlogS[:] -= infection_rate * h
    dlogE[:] += exp(log_betas + logS) * h
    
    # E -> I transition (becoming infectious)
    transition_rate = sigma_rate * exp(logE)
    dlogE[:] -= transition_rate * h
    dlogI[:] += transition_rate * h
        
    # I -> R transition (recovery)
    dlogI[:] -= (nu + mu) * h
        
    # New cases (E -> I transition)
    dC[:] += transition_rate * h

    # Apply derivatives
    logS[:] += dlogS
    logE[:] += dlogE
    logI[:] += dlogI
    C[:] += dC
    
@jit(nopython=True)
def multistrain_seir(
        dt_euler,
        dt_output, 
        n_regions, 
        logSs,
        logEs,
        logIs,
        Cs,
        Ts,
        Fs,
        mu,
        nu,
        beta0,
        psi,
        omega,
        eps,
        sigma_rate,
        dlogS,
        dlogE,
        dlogI,
        dC,
        log_betas):
    """Numba code to run the two-strain SEIR model. First line in all
    arrays is junk so truncate it."""


    ## Start at 1 cuz we have the first value as initial condition
    for iteration in range(1, logSs.shape[0]):
        t = (iteration-1) * dt_output
        out_t = iteration * dt_output


        calc_log_betas(out_t, beta0, eps, psi, omega, log_betas)
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
                     sigma_rate,
                     nu,
                     beta0,
                     eps,
                     psi,
                     omega,
                     log_betas)

                     
            t = t_next

        logSs[iteration, :] = logS
        logEs[iteration, :] = logE
        logIs[iteration, :] = logI

     

def run(S_init=None,
        E_init=None,
        I_init=None,
        n_weeks=20, ## 20 weeks per epidemic season,
        beta0=0.3,
        sigma_rate=0.5,  # Rate E->I (1/incubation period)
        dt_output=7,
        dt_euler=5e-2,
        mu=0,#1/30/365, ## birth death rate. assume const pop throughout season.
        nu=0.2,
        psi=365,
        omega=1,
        eps=0.1,
        n_regions=2):
    """Native python code to allocate arrays for and appropriately pack
    the results of the numba code above
   
    Parameters:
    - beta: float
        Transmission force
    - sigma_rate: float
        Rate of progression from Exposed to Infected (1/incubation_period).
    - dt_output: float
        Sampling time for output data.
    - dt_euler: float
        Time step for numerical integration using Eulers method.
    - mu: float, optional
        Natural mortality rate. Default is 1/30/365, so then the average life span is 30 years.
    - nu: float, optional
        Recovery rate. Default is 0.2, so average time of illness and infectivity is 5 time units.
    - psi: float, optional
        Duration of a cycle of the environmental driver in "time
        units". So psi=360 means a full cycle (i.e. year) constitutes
        of 360 time units (i.e. days). OTOH psi=1 means the time unit
        is a year.
    - omega: float, optional
        Rate of change of betas. Ignore in current simulation, but you can dig in and play with it.
    - eps: float
        Force of environmental driver effect on transmission
    - n_regions: int
        Number of regions in the simulation.
    - test_noise: np.array
        A noise array used for testing. User should ignore this.
    - S_init: array
        Initial number of susceptibles.
    - E_init: array
        Initial number of exposed individuals.
    - I_init: array
        Initial number of infected individuals.
   
    Returns:
    A pandas dataframe with the time series of susceptibles, infected, and possibly other states,
    depending on the implementation specifics.

    User should utilize C1 and C2 columns, which represent number of cases in the time before the sampling time. F1 and F2 the environmental drivers.
    """
    
    logS = np.full((n_weeks, n_regions), np.nan)
    logE = np.full((n_weeks, n_regions), np.nan)
    logI = np.full((n_weeks, n_regions), np.nan)
    Cs = np.full((n_weeks, n_regions), np.nan)
    F = np.full((n_weeks, n_regions), np.nan)
    T = np.full(n_weeks, np.nan)

    if S_init is None or E_init is None or I_init is None:
        # Implement random initial condition with small fraction infected/exposed
        total_pop = 1.0  # Normalized population
        
        # Small random fractions for initial conditions
        I_frac1 = np.random.uniform(1e-6, 1e-4)  # 0.0001% to 0.01% infected strain 1
        I_frac2 = np.random.uniform(1e-6, 1e-4)  # 0.0001% to 0.01% infected strain 2
        E_frac1 = np.random.uniform(1e-5, 1e-3)  # 0.001% to 0.1% exposed strain 1
        E_frac2 = np.random.uniform(1e-5, 1e-3)  # 0.001% to 0.1% exposed strain 2
        
        I_init = np.array([I_frac1 * total_pop, I_frac2 * total_pop])
        E_init = np.array([E_frac1 * total_pop, E_frac2 * total_pop])
        
        # Susceptible = remaining population
        S_init = np.array([total_pop - I_init[0] - E_init[0], 
                          total_pop - I_init[1] - E_init[1]])
    logS[0, :] = log(S_init)
    logE[0, :] = log(E_init)
    logI[0, :] = log(I_init)
    Cs[0, :] = np.nan ## Initial count of new cases is C

    beta0 = np.full(n_regions, beta0)
    nu = np.full(n_regions, nu)
    omega = np.full(n_regions, omega)
    eps = np.full(n_regions, eps)
  
    start = time.time()
    multistrain_seir(
        dt_euler=dt_euler,
        dt_output=dt_output,
        n_regions=n_regions,
        logSs=logS,
        logEs=logE,
        logIs=logI,
        Cs=Cs,
        Ts=T,
        Fs=F,
        mu=mu,
        nu=nu,
        beta0=beta0,
        psi=psi,
        omega=omega,
        eps=eps,
        sigma_rate=sigma_rate,
        dlogS=np.zeros(n_regions),
        dlogE=np.zeros(n_regions),
        dlogI=np.zeros(n_regions),
        dC=np.zeros(n_regions),
        log_betas=np.empty(n_regions))
    end = time.time()
    
    # print("Simulation run time", (end - start) / 60, "minutes", flush=True)
    # logI += np.random.randn(*logI.shape) * ona
    cols = ['logS1', 'logS2', 'logE1', 'logE2', 'logI1', 'logI2', 'C1', 'C2', 'F1', 'F2']
    data = np.hstack([logS, logE, logI, Cs, F])
    df = pd.DataFrame(index=T, data=data, columns=cols)
    
    df['S1'] = np.exp(df.logS1) 
    df['S2'] = np.exp(df.logS2)
    df['E1'] = np.exp(df.logE1) 
    df['E2'] = np.exp(df.logE2) 
    df['I1'] = np.exp(df.logI1) 
    df['I2'] = np.exp(df.logI2) 
    return df



def simulate():
    """Run simulation with default parameters"""
    df = run(n_weeks=100)
    df.index = pd.date_range(start='1900-01-01', periods=len(df), freq='7D')
    df.index.name = 'time'
    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Set dark mode style
    plt.style.use('dark_background')
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run simulation for 5 years with random initial conditions
    print("Running SEIR simulation with random initial conditions...")
    df = simulate()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.patch.set_facecolor('black')
    fig.suptitle('Two-Strain SEIR Model Simulation', fontsize=14, color='white')
    col1 ='#00FF7F'
    col2 = '#DA70D6'
    
    
    # Plot strain 1
    ax = axes[0]
    ax.plot(df.index, df.I1, col1, label='I1', alpha=0.8, linewidth=2)  # Tomato red
    ax.plot(df.index, df.I2, col2, label='I2', alpha=0.8, linewidth=2)  # Tomato red
    ax.set_title('Infecteds', color='white')
    ax.set_ylabel('Population Fraction', color='white')
    ax.legend()
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('black')
    
    # Plot cases (incidence)
    ax = axes[1]
    ax.plot(df.index, df.C1, col1, label='Incidence 1', alpha=0.8, linewidth=2)  # Deep pink
    ax.plot(df.index, df.C2, col2, label='Incidence 2', alpha=0.8, linewidth=2)  # Dark orange
    ax.set_title('Weekly Incidence (New Cases)', color='white')
    ax.set_ylabel('Cases per Week', color='white')
    ax.legend()
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('black')
    # Plot transmission rates (seasonal forcing)
    ax = axes[2]
    ax.plot(df.index, df.F1, col1, label='β1(t)', alpha=0.8, linewidth=2)  # Spring green
    ax.plot(df.index, df.F2, col2, label='β2(t)', alpha=0.8, linewidth=2)  # Orchid
    ax.set_title('Transmission Rates (Seasonal)', color='white')
    ax.set_ylabel('β(t)', color='white')
    ax.set_xlabel('Time', color='white')
    ax.legend()
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('black')
    
    
    plt.tight_layout()
    plt.show()
    
