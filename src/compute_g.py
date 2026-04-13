import numpy as np
import pandas as pd


def contacts(S0: np.ndarray,
             I0: np.ndarray,
             gamma: float,
             theta: float,
             beta0: float,
             delta: float,
             Ts,
             phase,
             N: np.ndarray
             ) -> pd.DataFrame:
    """
    Compute the Jacobian matrix G = ∂μ/∂φ for the exponential discretization SIR model
    with force of infection: λ_i = β_i * (C @ I)_i / (C @ N)_i.

    All S, I, μ are in counts (not fractions).
    """

    G = []

    S = S0.copy()
    I = I0.copy()

    # Connectivity matrix and its derivative
    C = np.array([[1-theta, theta], [theta, 1-theta]])
    Omega = np.array([[-1.0, 1.0], [1.0, -1.0]])  # ∂C/∂θ

    # Effective population seen by each region: n_i = sum_j C_ij N_j
    n_eff = C @ N          # shape (2,)
    dn_eff_dtheta = Omega @ N  # ∂n_eff/∂θ, shape (2,)

    # Initialize sensitivities
    dS_dS0 = np.eye(2)
    dI_dS0 = np.zeros((2, 2))
    dS_dI0 = np.zeros((2, 2))
    dI_dI0 = np.eye(2)
    dS_dtheta = np.zeros(2)
    dI_dtheta = np.zeros(2)

    for t in Ts:
        beta_t = beta0 * (1 + delta * np.sin(2 * np.pi * t + phase))

        # FOI: λ_i = β_i * (C @ I)_i / n_eff_i
        CI = C @ I               # contact-weighted infected, shape (2,)
        lambda_t = beta_t * CI / n_eff

        mu = S * (1 - np.exp(-lambda_t))

        # === Sensitivities of λ w.r.t. θ ===
        # λ_i = β_i * (C @ I)_i / (C @ N)_i
        # ∂λ_i/∂θ = β_i * [ (Ω @ I + C @ ∂I/∂θ)_i / n_eff_i
        #                   - (C @ I)_i * (Ω @ N)_i / n_eff_i^2 ]
        dlambda_dtheta = beta_t * (
            (Omega @ I + C @ dI_dtheta) / n_eff
            - CI * dn_eff_dtheta / n_eff**2
        )

        dmu_dtheta = (dS_dtheta * (1 - np.exp(-lambda_t))
                      + S * np.exp(-lambda_t) * dlambda_dtheta)

        # === Sensitivities of λ w.r.t. S(0) ===
        # ∂λ_i/∂S_j(0) = β_i * (C @ ∂I/∂S(0))_i / n_eff_i
        # (no denominator derivative since N doesn't depend on S(0))
        dlambda_dS0 = beta_t[:, None] * (C @ dI_dS0) / n_eff[:, None]

        dmu_dS0 = (dS_dS0 * (1 - np.exp(-lambda_t))[:, None]
                   + (S * np.exp(-lambda_t))[:, None] * dlambda_dS0)

        # === Sensitivities of λ w.r.t. I(0) ===
        # Same structure as S(0): only numerator depends on I via ∂I/∂I(0)
        dlambda_dI0 = beta_t[:, None] * (C @ dI_dI0) / n_eff[:, None]

        dmu_dI0 = (dS_dI0 * (1 - np.exp(-lambda_t))[:, None]
                   + (S * np.exp(-lambda_t))[:, None] * dlambda_dI0)

        # Store results
        for j in range(2):
            G.append({
                't': t, 'j': j,
                'S': S[j], 'I': I[j], 'mu': mu[j],
                'theta': dmu_dtheta[j],
                'S1_0': dmu_dS0[j, 0], 'I1_0': dmu_dI0[j, 0],
                'S2_0': dmu_dS0[j, 1], 'I2_0': dmu_dI0[j, 1]
            })

        # Update states
        if t < Ts[-1]:
            exp_neg_lambda = np.exp(-lambda_t)
            next_S = S * exp_neg_lambda
            next_I = I * np.exp(-gamma) + mu

            next_dS_dtheta = exp_neg_lambda * dS_dtheta - S * exp_neg_lambda * dlambda_dtheta
            next_dI_dtheta = np.exp(-gamma) * dI_dtheta + dmu_dtheta

            next_dS_dS0 = exp_neg_lambda[:, None] * dS_dS0 - (S * exp_neg_lambda)[:, None] * dlambda_dS0
            next_dI_dS0 = np.exp(-gamma) * dI_dS0 + dmu_dS0

            next_dS_dI0 = exp_neg_lambda[:, None] * dS_dI0 - (S * exp_neg_lambda)[:, None] * dlambda_dI0
            next_dI_dI0 = np.exp(-gamma) * dI_dI0 + dmu_dI0

            S, I = next_S, next_I
            dS_dtheta, dI_dtheta = next_dS_dtheta, next_dI_dtheta
            dS_dS0, dI_dS0 = next_dS_dS0, next_dI_dS0
            dS_dI0, dI_dI0 = next_dS_dI0, next_dI_dI0

    G = pd.DataFrame(G)
    G = G.set_index(['t', 'j'])
    return G
