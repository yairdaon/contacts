import pdb
import time

import numpy as np
import pandas as pd
from numba import jit, njit
from numba.core import types
from numpy import sin, cos, pi, log, exp, sqrt, ceil
import matplotlib.pyplot as plt


PSI=365
@jit(nopython=True)
def calc_log_betas(t,
                   beta0,
                   eps,
                   omega,
                   log_betas):
    log_betas[:] = log(beta0 * (1.0 + eps * sin(2.0 * pi / PSI * (t - omega * PSI))))
    

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
    force_of_infection = contact_matrix @ exp(logI) / population  # Matrix multiplication: Σ_j (C_ij * I_j / N_j)
    
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
        n_weeks=20, ## 20 weeks per epidemic season,
        beta0=0.45,
        sigma=0.5,  # Rate E->I (1/incubation period)
        dt_output=7,
        dt_euler=5e-2,
        mu= 0/30/365, ## birth death rate.
        nu=0.2,
        omega=1,
        eps=0.3,
        n_regions=2,
        contact_matrix=None,
        population=None):
    """Native python code to allocate arrays for and appropriately
    pack the results of the numba code above
   
    User should utilize C1 and C2 columns, which represent number of
    cases in the time before the sampling time. F1 and F2 the
    environmental drivers.

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
  
    start = time.time()
    multi_seir(
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
        omega=omega,
        eps=eps,
        sigma=sigma,
        contact_matrix=contact_matrix,
        population=population,
        dlogS=np.zeros(n_regions),
        dlogE=np.zeros(n_regions),
        dlogI=np.zeros(n_regions),
        dC=np.zeros(n_regions),
        log_betas=np.empty(n_regions))
    end = time.time()

    C = pd.DataFrame(index=T, data=Cs, columns=[f'C{i}' for i in range(n_regions)])
    F = pd.DataFrame(index=T, data=F, columns=[f'F{i}' for i in range(n_regions)])
    S = pd.DataFrame(index=T, data=exp(logS), columns=[f'S{i}' for i in range(n_regions)])
    E = pd.DataFrame(index=T, data=exp(logE), columns=[f'E{i}' for i in range(n_regions)])
    I = pd.DataFrame(index=T, data=exp(logI), columns=[f'I{i}' for i in range(n_regions)])
    df = pd.concat([C, F, S, E, I], axis=1)

    df.index = pd.date_range(start='1900-01-01', periods=len(df), freq='7D')
    df.index.name = 'time'
    return df


def simulate(seed=43):
    
    # Set random seed for reproducible results
    np.random.seed(seed)

    # Implement random initial condition with small fraction infected/exposed
    total_pop = 1.0  # Normalized population
    
    # Small random fractions for initial conditions
    I_init = np.random.uniform(1e-6, 1e-4, size=2) * total_pop  # 0.0001% to 0.01% infected 
    E_init = np.random.uniform(1e-5, 1e-3, size=2) * total_pop # 0.001% to 0.1% exposed             
     
    # Susceptible = remaining population
    S_init = total_pop - I_init - E_init
    
    df = run(n_weeks=100,
             S_init=S_init,
             I_init=I_init,
             E_init=E_init)
    return df

