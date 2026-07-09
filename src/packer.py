import numpy as np
import pandas as pd
from numpy import exp, log
from pprint import pprint

from src import compute_g


class Packer:
    """
    Packs and unpacks parameter vectors for optimization.

    Parameters:
    -----------
    disease : object
        Disease parameters (gamma, beta0, delta, rho, n_weeks, step_size)
    nat_driver : dict {season -> np.array}
        Per-capita national infection-rate driver, one array per season.
        Passed into compute_g's FOI as the α-weighted national exposure term.
        Required (no default).
    theta_upper : float
        Upper bound on pairwise-coupling parameter θ. 0 pins θ at 0.
    alpha_upper : float
        Upper bound on national-driver mixing weight α. 0 pins α at 0.
    """
    def __init__(self,
                 disease,
                 nat_driver,
                 seasons=None,
                 regions=None,
                 all_Ts=None,
                 populations=None,
                 theta_upper=0.5,
                 alpha_upper=0.0):

        self.disease = disease
        self.regions = regions if regions is not None else ["HHS0", "HHS1"]
        self.seasons = seasons if seasons is not None else [1990, 1991, 1992]
        self.n_regions = len(self.regions)
        self.n_seasons = len(self.seasons)
        assert self.n_regions == 2 ## For this branch only
        self.region_dict = dict(zip(range(self.n_regions), self.regions))
        self.all_Ts = {season: season + np.arange(disease.n_weeks) * disease.step_size for season in self.seasons} if all_Ts is None else all_Ts

        self.populations = populations if populations is not None else {}
        self.nat_driver = nat_driver

        self.theta_upper = theta_upper
        self.alpha_upper = alpha_upper

        # Parameter vector layout: [S_init.ravel(), I_init.ravel(), alpha?, theta?]
        # α and θ each take a slot only if their upper bound is > 0.
        self.n_params = (2 * self.n_regions * self.n_seasons
                         + (self.alpha_upper > 0)
                         + (self.theta_upper > 0))


    def random_vector(self, seed=None):
        params = self.random_dict(seed=seed)
        vec = self.pack(params)
        return vec


    def random_dict(self, seed=None):
        np.random.seed(seed)

        out = {}
        if self.alpha_upper > 0:
            out["alpha"] = np.random.uniform(0.0, self.alpha_upper)
        if self.theta_upper > 0:
            out["theta"] = np.random.uniform(0.0, self.theta_upper)

        I_init = np.zeros((self.n_seasons, self.n_regions))
        S_init = np.zeros((self.n_seasons, self.n_regions))

        for s_idx, season in enumerate(self.seasons):
            for r_idx, region in enumerate(self.regions):
                N = self.populations[(season, region)]
                I_init[s_idx, r_idx] = np.random.uniform(*self.disease.ilim)
                S_init[s_idx, r_idx] = np.random.uniform(*self.disease.slim)

        out["I_init"] = I_init
        out["S_init"] = S_init
        return out


    def pack(self, params):
        parts = [params["S_init"].ravel(), params["I_init"].ravel()]
        if self.alpha_upper > 0:
            parts.append([params["alpha"]])
        if self.theta_upper > 0:
            parts.append([params["theta"]])
        return np.concatenate(parts)

    def unpack(self, flat):
        out = {}
        idx = 0
        M = self.n_seasons * self.n_regions

        s_flat = flat[idx:idx + M]; idx += M
        i_flat = flat[idx:idx + M]; idx += M

        out["S_init"] = s_flat.reshape(self.n_seasons, self.n_regions)
        out["I_init"] = i_flat.reshape(self.n_seasons, self.n_regions)

        if self.alpha_upper > 0:
            out["alpha"] = flat[idx]; idx += 1
        if self.theta_upper > 0:
            out["theta"] = flat[idx]; idx += 1

        assert idx == flat.size
        return out

    def sim(self, params, phase, disease):
        """Simulate and return incidence + Jacobian columns for gradient computation."""
        S_init = params['S_init']
        I_init = params['I_init']
        theta = params["theta"] if self.theta_upper > 0 else 0.0
        alpha = params["alpha"] if self.alpha_upper > 0 else 0.0

        results = []
        for season_idx, season in enumerate(self.seasons):
            N = np.array([self.populations[(season, self.regions[i])]
                          for i in range(self.n_regions)])

            S = S_init[season_idx, :] * N
            I = I_init[season_idx, :] * N

            df = compute_g.contacts(S0=S,
                                    I0=I,
                                    gamma=disease.gamma,
                                    theta=theta,
                                    Ts=self.all_Ts[season],
                                    beta0=disease.beta0,
                                    delta=disease.delta,
                                    phase=phase,
                                    N=N,
                                    I_nat_pc=self.nat_driver[season],
                                    alpha=alpha)

            df = df.reset_index()
            df['season_idx'] = season_idx
            df_long = df[['t', 'j', 'mu', 'season_idx',
                          'theta', 'alpha',
                          'S1_0', 'I1_0', 'S2_0', 'I2_0']].rename(
                columns={'j': 'region'})
            df_long["region"] = df_long["region"].astype(int).replace(self.region_dict)
            df_long["season"] = season
            df_long['mu'] = df_long['mu'] + 1
            assert np.all(df_long['mu'] >= 0), f"Non positive mu values in simulation output"
            results.append(df_long)

        res = pd.concat(results, ignore_index=True)
        res = res.sort_values(['season', 't', 'region']).reset_index(drop=True)
        return res
