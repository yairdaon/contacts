import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import traceback
import pdb
from typing import Callable, Dict, List
from scipy.linalg import cho_factor, cho_solve


## Important that theta is kept as the first entry.
JACOBIAN_COLS = ['theta', 'S1_0', 'I1_0', 'S2_0', 'I2_0']


def plot_G(G):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data for each region
    times = np.arange(T)
    
    # Region 1 data
    region1_data = G.loc[(slice(None), 1), :]
    S1_vals = region1_data['S'].values
    I1_vals = region1_data['I'].values
    
    # Region 2 data
    region2_data = G.loc[(slice(None), 2), :]
    S2_vals = region2_data['S'].values
    I2_vals = region2_data['I'].values
    
    # Plot Susceptible populations
    axes[0].plot(times, S1_vals, 'b-', linewidth=2, label='Region 1')
    axes[0].plot(times, S2_vals, 'r-', linewidth=2, label='Region 2')
    axes[0].set_xlabel('Time (weeks)')
    axes[0].set_ylabel('Susceptible fraction')
    axes[0].set_title('Susceptible Populations')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Infected populations
    axes[1].plot(times, I1_vals, 'b-', linewidth=2, label='Region 1')
    axes[1].plot(times, I2_vals, 'r-', linewidth=2, label='Region 2')
    axes[1].set_xlabel('Time (weeks)')
    axes[1].set_ylabel('Infected fraction')
    axes[1].set_title('Infected Populations')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
def compute_G(S1_0: float,
              S2_0: float,
              I1_0: float,
              I2_0: float,  # Initial conditions
              gamma: float,                       # Recovery rate
              theta: float,                       # Connectivity parameter
              T: int, # Number of time steps
              beta0: float,
              amplitude: float,
              period: float              
              ) -> pd.DataFrame:
    """
    Compute the Jacobian matrix G = ∂μ/∂φ for the two-region SIR model.
    
    This function tracks the partial derivatives of the mean observation vector μ
    with respect to all parameters φ = [θ, S₁(0), I₁(0), S₂(0), I₂(0)].
    
    Parameters:
    -----------
    S1_0, S2_0, I1_0, I2_0 : float
        Initial conditions for susceptible and infected populations in regions 1 and 2
    gamma : float
        Recovery rate
    theta : float 
        Connectivity parameter (off-diagonal elements of contact matrix)
    T : int
        Number of time steps to simulate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with MultiIndex (t, j) where:
        - t ∈ {0, 1, ..., T-1} (time steps)
        - j ∈ {1, 2} (regions)
        Columns: ['theta', 'S1_0', 'I1_0', 'S2_0', 'I2_0']
        Entry (t,j,param) = ∂μⱼ(t)/∂param
    """
    
    # Initialize arrays to store results
    G = []
    
    # Current state vectors
    S = np.array([S1_0, S2_0])  # [S1(0), S2(0)]
    I = np.array([I1_0, I2_0])  # [I1(0), I2(0)]
    
    # Connectivity matrix and its derivative
    C = np.array([[1-theta, theta], [theta, 1-theta]])
    Omega = np.array([[-1.0, 1.0], [1.0, -1.0]])  # ∂C/∂θ
    
    # Initialize sensitivity matrices (2x2 each) at t=0
    dS_dS0 = np.eye(2)        # ∂S(t)/∂S(0) = Id
    dI_dS0 = np.zeros((2, 2)) # ∂I(t)/∂S(0) = 0
    dS_dI0 = np.zeros((2, 2)) # ∂S(t)/∂I(0) = 0 
    dI_dI0 = np.eye(2)        # ∂I(t)/∂I(0) = Id 
    
    # Compute observations for T time steps (t = 0, 1, ..., T-1)
    for t in range(T):
        beta_t = beta0 * (1 + amplitude * np.sin(2 * np.pi * t / period))
        
        # Compute mean observations μ(t) = β(t) * [S₁(C·I)₁, S₂(C·I)₂]
        CI = C @ I  # Contact matrix times infected vector
        mu = beta_t * S * CI  # μ(t) ∈ ℝ²
        
        # Compute ∂μ(t)/∂θ = β(t) * S ∘ (Ω·I)
        dmu_dtheta = beta_t * S * (Omega @ I)
        
        # Compute ∂μ(t)/∂S(0) = β(t) * [∂S/∂S(0) ∘ (C·I) + S ∘ C·∂I/∂S(0)]
        dmu_dS0 = beta_t * (dS_dS0 * CI + S * (C @ dI_dS0))  # Shape (2,2)
        
        # Compute ∂μ(t)/∂I(0) = β(t) * [∂S/∂I(0) ∘ (C·I) + S ∘ C·∂I/∂I(0)]
        dmu_dI0 = beta_t * (dS_dI0 * CI + S * (C @ dI_dI0))  # Shape (2,2)
        
        # Store results for both regions (j=1,2)
        for j in range(2):
            row_data = {
                't': t,
                'j': j + 1,  # Regions numbered 1,2
                'S': S[j],   # Current susceptible population
                'I': I[j],   # Current infected population
                'mu': mu[j], # Mean observation μⱼ(t)
                'theta': dmu_dtheta[j],
                'S1_0': dmu_dS0[j, 0],  # ∂μⱼ(t)/∂S₁(0)
                'I1_0': dmu_dI0[j, 0],  # ∂μⱼ(t)/∂I₁(0)
                'S2_0': dmu_dS0[j, 1],  # ∂μⱼ(t)/∂S₂(0)
                'I2_0': dmu_dI0[j, 1]   # ∂μⱼ(t)/∂I₂(0)
            }
            G.append(row_data)
        
        # Update states for next time step (only if t < T-1)
        if t < T - 1:
            # Calculate next states using SIR dynamics
            mu = beta_t * S * CI  # mean observations μ(t)
            next_S = S - mu
            next_I = (1 - gamma) * I + mu
            
            # Calculate sensitivity updates (same as in main simulation)
            dMu_dS0 = CI * dS_dS0 + S * (C @ dI_dS0)
            dMu_dS0 *= beta_t
            
            dMu_dI0 = CI * dS_dI0 + S * (C @ dI_dI0)
            dMu_dI0 *= beta_t
            
            # Update sensitivity matrices
            next_dS_dS0 = dS_dS0 - dMu_dS0
            next_dI_dS0 = (1 - gamma) * dI_dS0 + dMu_dS0
            next_dS_dI0 = dS_dI0 - dMu_dI0
            next_dI_dI0 = (1 - gamma) * dI_dI0 + dMu_dI0
            
            # Update all variables
            S = next_S
            I = next_I
            dS_dS0 = next_dS_dS0
            dI_dS0 = next_dI_dS0
            dS_dI0 = next_dS_dI0
            dI_dI0 = next_dI_dI0
    
    # Create DataFrame with MultiIndex
    G = pd.DataFrame(G)
    G = G.set_index(['t', 'j'])
    
    return G


def compute_crlb(S1_0: float,
                 S2_0: float,
                 I1_0: float,
                 I2_0: float,
                 gamma: float,
                 theta: float,
                 T: int,
                 sigma: float,
                 beta0: float,
                 amplitude: float,
                 period: float,
                 noise: str = "mult"
                 ):
    """
    Compute Cramér-Rao Lower Bound for standard deviation of connectivity parameter theta.
    
    Parameters:
    -----------
    S1_0, S2_0, I1_0, I2_0 : float
        Initial conditions for susceptible and infected populations
    gamma : float
        Recovery rate
    theta : float
        Connectivity parameter
    T : int
        Number of time steps
    sigma : float
        Observation noise standard deviation
    beta0 : float
        Base transmission rate
    amplitude : float
        Seasonal amplitude
    period : float
        Seasonal period
    noise : str, default="add"
        Noise model: "add" for additive, "mult" for multiplicative
        
    Returns:
    --------
    float
        the crlb for standard deviation of theta
    """
    
    # Compute Jacobian matrix G.
    df = compute_G(
        S1_0=S1_0,
        S2_0=S2_0,
        I1_0=I1_0,
        I2_0=I2_0,
        gamma=gamma,
        theta=theta,
        T=T,
        beta0=beta0,
        amplitude=amplitude,
        period=period              
    )
    
    
    G = df[JACOBIAN_COLS].values  # Shape: (2T, 5)
    
    if noise == "add":
        # Additive noise model: J = G^T G / sigma^2
        G /= sigma
        
    elif noise == "mult":
        # Multiplicative noise model: J = G^T W G / sigma^2
        # where W = diag((2*sigma^2 + 1) / (sigma^2 * mu_i(t)^2))

        mu = df['mu'].values
        # Compute weight matrix diagonal: (2*sigma^2 + 1) / (sigma^2 * mu_i(t)^2)
        sqrt_W = np.sqrt(2 * sigma**2 + 1) / sigma / mu
        
        # Apply weights: G_weighted = sqrt_weight * G (broadcasting)
        G = np.einsum('i, ij -> ij', sqrt_W, G)
                
        
    else:
        raise ValueError(f"Unknown noise model: {noise}. Use 'add' or 'mult'.") 

    # Fisher Information Matrix: J = G.T @ G since weights were
    # incorporated into G for each noise model.
    J = G.T @ G

    assert np.all(J == J.T), "J not symmetric"
    
    # Cholesky factorization LL^t = J
    L, low = cho_factor(J) 

    # e1 = [1,0,0,0,0] since we have 5 variables. insert 1 where theta
    # is found in JACOBIAN_COLS
    e1 = np.zeros(J.shape[0])
    e1[0] = 1

    # find L^{-t}e1
    x = cho_solve((L, low), e1)

    ## CRLB for the standard deviation (!) of theta
    return np.linalg.norm(x)
