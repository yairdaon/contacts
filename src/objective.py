import numpy as np
from scipy.special import xlogy

from src.packer import Packer


class Objective:
    def __init__(self,
                 obs,
                 phase,
                 disease,
                 populations,
                 theta_upper=0.5):

        self.packer = Packer(disease=disease,
                             seasons=obs['season'].unique(),
                             regions=obs['region'].unique(),
                             all_Ts={season: np.sort(dd['t'].unique()) for season, dd in obs.groupby("season")},
                             populations=populations,
                             theta_upper=theta_upper)
        self.phase = phase
        self.obs = obs.sort_values(['season', 't', 'region']).reset_index(drop=True)
        self.disease = disease
        self.n_obs = len(self.obs)

    def compute_gradient(self, sim_df):
        """
        Compute gradient of the negative log-likelihood with respect to parameters.
        ∂L/∂φ = -Σ A_i(t) * ∂μ_i(t)/∂φ
        """
        n_seasons = self.packer.n_seasons
        n_regions = self.packer.n_regions
        M = n_seasons * n_regions
        rho = self.disease.rho

        grad = np.zeros(self.packer.n_params)
            

        mu = sim_df["mu"].values + 1e-6
        r = self.obs["incidence"].values - mu * rho

        A = -1 / (2 * mu) + r / ((1 - rho) * mu) + r ** 2 / (2 * rho * (1 - rho) * mu ** 2)
        dL_dmu = -A

        # theta gradient
        if self.packer.theta_upper > 0:
            grad[-1] = np.sum(dL_dmu * sim_df['theta'].values)

        # S_init and I_init gradients (per season), scaled by N for fraction variables
        for season_idx in range(n_seasons):
            mask = sim_df['season_idx'].values == season_idx
            season = self.packer.seasons[season_idx]
            N = np.array([self.packer.populations[(season, self.packer.regions[i])]
                          for i in range(n_regions)])

            grad[season_idx * n_regions + 0] = np.sum(dL_dmu[mask] * sim_df['S1_0'].values[mask]) * N[0]
            grad[season_idx * n_regions + 1] = np.sum(dL_dmu[mask] * sim_df['S2_0'].values[mask]) * N[1]

            grad[M + season_idx * n_regions + 0] = np.sum(dL_dmu[mask] * sim_df['I1_0'].values[mask]) * N[0]
            grad[M + season_idx * n_regions + 1] = np.sum(dL_dmu[mask] * sim_df['I2_0'].values[mask]) * N[1]

        return grad


    def __call__(self, x, grad=None):
        """Evaluate negative log-likelihood and gradient."""
        params = self.packer.unpack(x)
        simulated = self.packer.sim(params=params, phase=self.phase, disease=self.disease)

        obs = self.obs["incidence"].values
        mu = simulated["mu"].values
        sim = mu * self.disease.rho
        residual = obs - sim
        
        sigma2 = sim * (1 - self.disease.rho) + 1e-6
        log_term = xlogy(sigma2, 2 * np.pi * sigma2)
        out = np.sum((log_term + residual ** 2) / sigma2) / (2 * self.n_obs)

        if grad is not None and grad.size > 0:
            grad[:] = self.compute_gradient(simulated) / self.n_obs

        return out
