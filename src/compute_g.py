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
             N: np.ndarray,
             I_nat_pc=None,
             alpha: float = 0.0,
             ) -> pd.DataFrame:
    """
    Compute the Jacobian matrix G = ∂μ/∂φ for the exponential-discretization
    SIR model with force of infection:

        λ_i = β_i * [ (1 - α) * (C @ I)_i / (C @ N)_i  +  α * I_nat_pc(t) ]

    where I_nat_pc(t) is an exogenous per-capita national infection-rate
    estimate (loaded via load_national_driver). alpha=0 recovers the original
    model. I_nat_pc must have length len(Ts) when alpha > 0; if None or
    alpha == 0, the national term is treated as absent.

    All S, I, μ are in counts (not fractions).
    """
    G = []

    S = S0.copy()
    I = I0.copy()

    if I_nat_pc is None:
        I_nat_pc = np.zeros(len(Ts))
    else:
        I_nat_pc = np.asarray(I_nat_pc, dtype=float)
        assert I_nat_pc.shape[0] == len(Ts), \
            f"I_nat_pc length {I_nat_pc.shape[0]} != len(Ts) {len(Ts)}"

    # Connectivity matrix and its derivative
    C = np.array([[1-theta, theta], [theta, 1-theta]])
    Omega = np.array([[-1.0, 1.0], [1.0, -1.0]])  # ∂C/∂θ

    # Effective population seen by each region: n_i = sum_j C_ij N_j
    n_eff = C @ N               # shape (2,)
    dn_eff_dtheta = Omega @ N   # ∂n_eff/∂θ, shape (2,)

    # Initialize sensitivities. Six columns total: S1_0, I1_0, S2_0, I2_0, alpha, theta.
    dS_dS0 = np.eye(2)
    dI_dS0 = np.zeros((2, 2))
    dS_dI0 = np.zeros((2, 2))
    dI_dI0 = np.eye(2)
    dS_dtheta = np.zeros(2)
    dI_dtheta = np.zeros(2)
    dS_dalpha = np.zeros(2)
    dI_dalpha = np.zeros(2)

    for idx, t in enumerate(Ts):
        beta_t = beta0 * (1 + delta * np.sin(2 * np.pi * t + phase))

        # Local per-capita infected fraction (contact-weighted)
        CI = C @ I
        L = CI / n_eff                         # shape (2,) — local driver
        Nnat_t = I_nat_pc[idx]                 # scalar — national driver

        # FOI: λ_i = β_i * [(1-α) L_i + α * Nnat_t]
        lambda_t = beta_t * ((1.0 - alpha) * L + alpha * Nnat_t)

        mu = S * (1 - np.exp(-lambda_t))

        # === Sensitivities of λ w.r.t. θ (scaled by (1-α)) ===
        dL_dtheta = ((Omega @ I + C @ dI_dtheta) / n_eff
                     - CI * dn_eff_dtheta / n_eff**2)
        dlambda_dtheta = beta_t * (1.0 - alpha) * dL_dtheta

        # === Sensitivities of λ w.r.t. α ===
        # ∂λ_i/∂α = β_i * (Nnat_t - L_i) + β_i * (1-α) * ∂L_i/∂α, and
        # ∂L_i/∂α = (C @ ∂I/∂α)_i / n_eff_i  (via propagated dI/dalpha).
        dL_dalpha = (C @ dI_dalpha) / n_eff
        dlambda_dalpha = beta_t * (Nnat_t - L + (1.0 - alpha) * dL_dalpha)

        dmu_dtheta = (dS_dtheta * (1 - np.exp(-lambda_t))
                      + S * np.exp(-lambda_t) * dlambda_dtheta)
        dmu_dalpha = (dS_dalpha * (1 - np.exp(-lambda_t))
                      + S * np.exp(-lambda_t) * dlambda_dalpha)

        # === Sensitivities of λ w.r.t. S(0) and I(0) (scaled by (1-α)) ===
        dlambda_dS0 = beta_t[:, None] * (1.0 - alpha) * (C @ dI_dS0) / n_eff[:, None]
        dlambda_dI0 = beta_t[:, None] * (1.0 - alpha) * (C @ dI_dI0) / n_eff[:, None]

        dmu_dS0 = (dS_dS0 * (1 - np.exp(-lambda_t))[:, None]
                   + (S * np.exp(-lambda_t))[:, None] * dlambda_dS0)
        dmu_dI0 = (dS_dI0 * (1 - np.exp(-lambda_t))[:, None]
                   + (S * np.exp(-lambda_t))[:, None] * dlambda_dI0)

        for j in range(2):
            G.append({
                't': t, 'j': j,
                'S': S[j], 'I': I[j], 'mu': mu[j],
                'theta': dmu_dtheta[j],
                'alpha': dmu_dalpha[j],
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

            next_dS_dalpha = exp_neg_lambda * dS_dalpha - S * exp_neg_lambda * dlambda_dalpha
            next_dI_dalpha = np.exp(-gamma) * dI_dalpha + dmu_dalpha

            next_dS_dS0 = exp_neg_lambda[:, None] * dS_dS0 - (S * exp_neg_lambda)[:, None] * dlambda_dS0
            next_dI_dS0 = np.exp(-gamma) * dI_dS0 + dmu_dS0

            next_dS_dI0 = exp_neg_lambda[:, None] * dS_dI0 - (S * exp_neg_lambda)[:, None] * dlambda_dI0
            next_dI_dI0 = np.exp(-gamma) * dI_dI0 + dmu_dI0

            S, I = next_S, next_I
            dS_dtheta, dI_dtheta = next_dS_dtheta, next_dI_dtheta
            dS_dalpha, dI_dalpha = next_dS_dalpha, next_dI_dalpha
            dS_dS0, dI_dS0 = next_dS_dS0, next_dI_dS0
            dS_dI0, dI_dI0 = next_dS_dI0, next_dI_dI0

    G = pd.DataFrame(G)
    G = G.set_index(['t', 'j'])
    return G
