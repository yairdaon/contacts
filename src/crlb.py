import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import traceback
import pdb
import os
import uuid
from typing import Callable, Dict, List
from scipy.linalg import cho_factor, cho_solve

from src.helper import JACOBIAN_COLS, plot_G


def compute_G(S0: np.ndarray,
              I0: np.ndarray,
              gamma: float,
              theta: float,
              T: int,
              beta0: float,
              amplitude: float,
              period: float,
              phase: float = 0.0,
              phase2: float = None
              ) -> pd.DataFrame:
    """
    Compute the Jacobian matrix G = ∂μ/∂φ for the exponential discretization SIR model.

    This function uses the exponential formulation:
        S(t+1) = S(t) exp(-λ(t))
        I(t+1) = I(t) exp(-γ) + μ(t)
    where μ(t) = S(t)[1 - exp(-λ(t))] and λ(t) = β(t) C I(t)

    Parameters:
    -----------
    S0 : np.ndarray
        Initial susceptible populations (shape: n_regions)
    I0 : np.ndarray
        Initial infected populations (shape: n_regions)
    gamma : float
        Recovery rate parameter (not probability - can exceed 1)
    theta : float
        Connectivity parameter (off-diagonal elements of contact matrix)
    T : int
        Number of time steps to simulate
    beta0 : float
        Base transmission rate
    amplitude : float
        Seasonal amplitude
    period : float
        Seasonal period
    phase : float, default=0.0
        Phase offset for seasonal forcing (in radians). Applied to both regions if phase2 is None.
    phase2 : float, optional
        Phase offset for region 2 (in radians).
        If None, region 2 uses the same phase as region 1.
        β₁(t) = β₀(1 + A·sin(2πt/P + φ))
        β₂(t) = β₀(1 + A·sin(2πt/P + φ₂)) if phase2 is not None

    Returns:
    --------
    pd.DataFrame
        DataFrame with MultiIndex (t, j) where:
        - t ∈ {0, 1, ..., T-1} (time steps)
        - j ∈ {1, 2} (regions)
        Columns: ['S', 'I', 'mu', 'theta', 'S1_0', 'I1_0', 'S2_0', 'I2_0']
        Entry (t,j,param) = ∂μⱼ(t)/∂param
    """

    # Initialize arrays to store results
    G = []

    # Current state vectors
    S = S0.copy()  # [S1(0), S2(0), ...]
    I = I0.copy()  # [I1(0), I2(0), ...]
    n_regions = len(S)

    # Connectivity matrix and its derivative (2 regions only)
    C = np.array([[1-theta, theta], [theta, 1-theta]])
    Omega = np.array([[-1.0, 1.0], [1.0, -1.0]])  # ∂C/∂θ

    # Initialize sensitivity matrices (2x2 each) at t=0
    dS_dS0 = np.eye(2)        # ∂S(t)/∂S(0) = Id
    dI_dS0 = np.zeros((2, 2)) # ∂I(t)/∂S(0) = 0
    dS_dI0 = np.zeros((2, 2)) # ∂S(t)/∂I(0) = 0
    dI_dI0 = np.eye(2)        # ∂I(t)/∂I(0) = Id

    # Initialize sensitivity vectors for theta
    dS_dtheta = np.zeros(2)   # ∂S(t)/∂θ = 0 at t=0
    dI_dtheta = np.zeros(2)   # ∂I(t)/∂θ = 0 at t=0

    # Determine phase for region 2
    phase2 = phase if phase2 is None else phase2

    # Compute observations for T time steps (t = 0, 1, ..., T-1)
    for t in range(T):
        # Seasonal transmission rate (vector for each region)
        # β₁(t) = β₀(1 + A·sin(2πt/P + φ))
        # β₂(t) = β₀(1 + A·sin(2πt/P + φ₂))
        phase_vec = np.array([phase, phase2])
        beta_t = beta0 * (1 + amplitude * np.sin(2 * np.pi * t / period + phase_vec))
       
        # Force of infection: λ(t) = β(t) ∘ (C I(t))
        # Element-wise multiplication of beta_t with contact-weighted infections
        lambda_t = beta_t * (C @ I)

        # Mean incidence: μ(t) = S(t) [1 - exp(-λ(t))]
        mu = S * (1 - np.exp(-lambda_t))

        # === Compute ∂μ(t)/∂θ ===
        # ∂λ(t)/∂θ = β(t) [Ω I(t) + C ∂I(t)/∂θ]
        dlambda_dtheta = beta_t * (Omega @ I + C @ dI_dtheta)

        # ∂μ(t)/∂θ = ∂S(t)/∂θ [1 - exp(-λ(t))] + S(t) exp(-λ(t)) ∂λ(t)/∂θ
        dmu_dtheta = dS_dtheta * (1 - np.exp(-lambda_t)) + S * np.exp(-lambda_t) * dlambda_dtheta

        # === Compute ∂μ(t)/∂S(0) ===
        # ∂λ(t)/∂S(0) = β(t) ∘ C ∂I(t)/∂S(0)
        # Broadcasting: beta_t[:, None] has shape (2, 1), (C @ dI_dS0) has shape (2, 2)
        dlambda_dS0 = beta_t[:, None] * (C @ dI_dS0)  # Shape (2, 2)

        # ∂μ(t)/∂S(0) = ∂S(t)/∂S(0) ∘ [1 - exp(-λ(t))] + S(t) ∘ exp(-λ(t)) ∘ ∂λ(t)/∂S(0)
        # Broadcasting: (2,2) * (2,) -> each row gets multiplied
        dmu_dS0 = dS_dS0 * (1 - np.exp(-lambda_t)) + (S * np.exp(-lambda_t))[:, None] * dlambda_dS0

        # === Compute ∂μ(t)/∂I(0) ===
        # ∂λ(t)/∂I(0) = β(t) ∘ C ∂I(t)/∂I(0)
        # Broadcasting: beta_t[:, None] has shape (2, 1), (C @ dI_dI0) has shape (2, 2)
        dlambda_dI0 = beta_t[:, None] * (C @ dI_dI0)  # Shape (2, 2)

        # ∂μ(t)/∂I(0) = ∂S(t)/∂I(0) ∘ [1 - exp(-λ(t))] + S(t) ∘ exp(-λ(t)) ∘ ∂λ(t)/∂I(0)
        dmu_dI0 = dS_dI0 * (1 - np.exp(-lambda_t)) + (S * np.exp(-lambda_t))[:, None] * dlambda_dI0

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
            # === Update state variables ===
            # S(t+1) = S(t) exp(-λ(t))
            # I(t+1) = I(t) exp(-γ) + μ(t)
            next_S = S * np.exp(-lambda_t)
            next_I = I * np.exp(-gamma) + mu

            # === Update ∂S(t+1)/∂θ ===
            # ∂S(t+1)/∂θ = exp(-λ(t)) ∂S(t)/∂θ - S(t) exp(-λ(t)) ∂λ(t)/∂θ
            next_dS_dtheta = np.exp(-lambda_t) * dS_dtheta - S * np.exp(-lambda_t) * dlambda_dtheta

            # === Update ∂I(t+1)/∂θ ===
            # ∂I(t+1)/∂θ = exp(-γ) ∂I(t)/∂θ + ∂μ(t)/∂θ
            next_dI_dtheta = np.exp(-gamma) * dI_dtheta + dmu_dtheta

            # === Update ∂S(t+1)/∂S(0) ===
            # ∂S(t+1)/∂S(0) = diag(exp(-λ(t))) ∂S(t)/∂S(0) - diag(S(t) exp(-λ(t))) ∂λ(t)/∂S(0)
            exp_neg_lambda = np.exp(-lambda_t)
            next_dS_dS0 = exp_neg_lambda[:, None] * dS_dS0 - (S * exp_neg_lambda)[:, None] * dlambda_dS0

            # === Update ∂I(t+1)/∂S(0) ===
            # ∂I(t+1)/∂S(0) = exp(-γ) ∂I(t)/∂S(0) + ∂μ(t)/∂S(0)
            next_dI_dS0 = np.exp(-gamma) * dI_dS0 + dmu_dS0

            # === Update ∂S(t+1)/∂I(0) ===
            # ∂S(t+1)/∂I(0) = diag(exp(-λ(t))) ∂S(t)/∂I(0) - diag(S(t) exp(-λ(t))) ∂λ(t)/∂I(0)
            next_dS_dI0 = exp_neg_lambda[:, None] * dS_dI0 - (S * exp_neg_lambda)[:, None] * dlambda_dI0

            # === Update ∂I(t+1)/∂I(0) ===
            # ∂I(t+1)/∂I(0) = exp(-γ) ∂I(t)/∂I(0) + ∂μ(t)/∂I(0)
            next_dI_dI0 = np.exp(-gamma) * dI_dI0 + dmu_dI0

            # Update all variables
            S = next_S
            I = next_I
            dS_dtheta = next_dS_dtheta
            dI_dtheta = next_dI_dtheta
            dS_dS0 = next_dS_dS0
            dI_dS0 = next_dI_dS0
            dS_dI0 = next_dS_dI0
            dI_dI0 = next_dI_dI0

    # Create DataFrame with MultiIndex
    G = pd.DataFrame(G)
    G = G.set_index(['t', 'j'])

    return G



def compute_crlb(S0,
                 I0,
                 gamma: float,
                 theta: float,
                 T: int,
                 sigma: float,
                 beta0: float,
                 amplitude: float,
                 period: float,
                 noise: str = "mult",
                 pop_size: int = 1,
                 phase: float = 0.0,
                 phase2: float = None
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
    noise : str, default="mult"
        Noise model: "add" for additive, "mult" for multiplicative, "poisson" for Poisson
    pop_size : int, default=1
        Population scaling factor for Poisson model
    phase : float, default=0.0
        Phase offset for seasonal forcing (in radians).
        If phase2 is None, both regions use this phase.
    phase2 : float, optional
        Phase offset for region 2 (in radians).
        If None, region 2 uses the same phase as region 1.

    Returns:
    --------
    float
        the crlb for standard deviation of theta
    """
    
    df = compute_G(
        S0=S0,
        I0=I0,
        gamma=gamma,
        theta=theta,
        T=T,
        beta0=beta0,
        amplitude=amplitude,
        period=period,
        phase=phase,
        phase2=phase2
    )
    
    mu = df['mu'].values
    G = df[JACOBIAN_COLS].values  # Shape: (2T, 5)
    
    if noise == "add":
        # Additive noise model: J = G^T G / sigma^2
        G /= sigma
        
    elif noise == "mult":
        # Multiplicative noise model: J = G^T W G / sigma^2
        # where W = diag((2*sigma^2 + 1) / (sigma^2 * mu_i(t)^2))

        # Compute weight matrix diagonal: (2*sigma^2 + 1) / (sigma^2 * mu_i(t)^2)
        sqrt_W = np.sqrt(2 * sigma**2 + 1) / sigma / mu
        
        # Apply weights: G_weighted = sqrt_weight * G (broadcasting)
        G = np.einsum('i, ij -> ij', sqrt_W, G)
                
    elif noise == "poisson":
        # Poisson noise model: J = G^T W G
        # where W = diag(pop_size / mu_i(t))

        # Compute weight matrix diagonal: N / mu_i(t)
        sqrt_W = np.sqrt(pop_size / mu)
        
        # Apply weights: G_weighted = sqrt_weight * G (broadcasting)
        G = np.einsum('i, ij -> ij', sqrt_W, G)

    elif noise == 'bin':
        sqrt_W = 1 /  mu  
        G = np.einsum('i, ij -> ij', sqrt_W, G)
        
    else:
        raise ValueError(f"Unknown noise model: {noise}. Use 'add', 'mult', or 'poisson'.") 

    # Fisher Information Matrix: J = G.T @ G since weights were
    # incorporated into G for each noise model.
    J = G.T @ G

    try:
        # Cholesky factorization LL^t = J
        L, low = cho_factor(J)

        # e1 = [1,0,0,0,0] since we have 5 variables. insert 1 where theta
        # is found in JACOBIAN_COLS
        e1 = np.zeros(J.shape[0])
        e1[0] = 1

        # find L^{-t}e1
        x = cho_solve((L, low), e1)

        ## CRLB for the standard deviation (!) of theta
        result = np.linalg.norm(x)

    except (ValueError, np.linalg.LinAlgError) as e:
        result = np.nan
        print(f"WARNING: Cholesky factorization failed: {e}")

    # Debug: save plot if result is NaN or inf
    if not np.isfinite(result):
        # Create debug directory if it doesn't exist
        debug_dir = 'crlb_res'
        os.makedirs(debug_dir, exist_ok=True)

        # Generate unique filename
        filename = os.path.join(debug_dir, f'{noise}_{uuid.uuid4().hex}.png')

        # Create and save plot with parameter information
        fig = plot_G(df, theta=theta, phase=phase, amplitude=amplitude, phase2=phase2)
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"WARNING: Non-finite CRLB ({result}). Debug plot saved to {filename}")

    return result
    
