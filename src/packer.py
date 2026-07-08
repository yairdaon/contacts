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
    seasons : list, optional
        List of season years (e.g., [2010, 2011, 2012])
    regions : list, optional
        List of region names (must have exactly 2 regions)
    all_Ts : dict, optional
        Dictionary mapping season -> array of time points
    populations : dict
        Dictionary mapping (season, region) -> population count (float).
        Required keys: all (season, region) pairs where season in seasons
        and region in regions.
        Example: {(2010, 'New York'): 19378102.0, (2010, 'California'): 37253956.0, ...}
    """
    def __init__(self,
                 disease,
                 seasons=None,
                 regions=None,
                 all_Ts=None,
                 populations=None,
                 theta_upper=0.5):

        self.disease = disease
        self.regions = regions if regions is not None else ["HHS0", "HHS1"]
        self.seasons = seasons if seasons is not None else [1990, 1991, 1992]
        self.n_regions = len(self.regions)
        self.n_seasons = len(self.seasons)
        assert self.n_regions == 2 ## For this branch only
        self.region_dict = dict(zip(range(self.n_regions), self.regions))
        self.all_Ts = {season: season + np.arange(disease.n_weeks) * disease.step_size for season in self.seasons} if all_Ts is None else all_Ts

        # populations: dict {(season, region): N_value}
        self.populations = populations if populations is not None else {}

        

        # Upper bound on theta; used by random_dict and by the optimizer bounds.
        # Set to 0.0 to fit the null model (theta pinned at 0), 0.5 for the
        # default diagonal-dominant coupled model, 1.0 to allow the high-coupling
        # regime.
        self.theta_upper = theta_upper


        # Count parameters: S, I init (2*n_regions*n_seasons), theta,
        self.n_params = 2 * self.n_regions * self.n_seasons + (self.theta_upper > 0)

            
    def random_vector(self, seed=None):
        params = self.random_dict(seed=seed)
        vec = self.pack(params)
        return vec

    
    def random_dict(self, seed=None):
        np.random.seed(seed)
      
        if self.theta_upper > 0:
            out = dict(theta=np.random.uniform(0.0, self.theta_upper))
        else:
            out = dict()

            
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
        parts = []

        S_init = params["S_init"]
        I_init = params["I_init"]
        parts.append(S_init.ravel())
        parts.append(I_init.ravel())
        if self.theta_upper > 0:
            parts.append([params["theta"]]) 

        flat = np.concatenate(parts)
        return flat

    def unpack(self, flat):
        out = {}
        idx = 0

        # Unpack individual S, I values - no transformations # , E (E_init = I_init)
        M = self.n_seasons * self.n_regions

        s_flat = flat[idx:idx + M]
        idx += M
        i_flat = flat[idx:idx + M]
        idx += M

        out["S_init"] = s_flat.reshape(self.n_seasons, self.n_regions)
        out["I_init"] = i_flat.reshape(self.n_seasons, self.n_regions)

        if self.theta_upper > 0:
            theta = flat[idx]
            out["theta"] = theta
            idx += 1

        assert idx == flat.size
        return out

    def sim(self, params, phase, disease):
        """Simulate and return incidence + Jacobian columns for gradient computation."""
        S_init = params['S_init']
        I_init = params['I_init']
        if self.theta_upper > 0:
            theta = params["theta"]
        else:
            theta = 0
            
        results = []
        for season_idx, season in enumerate(self.seasons):
            # Get populations for this season
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
                                    N=N)
            
            df = df.reset_index()
            df['season_idx'] = season_idx
            df_long = df[['t',
                          'j',
                          'mu',
                          'season_idx',
                          'theta',
                          'S1_0',
                          'I1_0',
                          'S2_0',
                          'I2_0']].rename(
                columns={'j': 'region'})
            df_long["region"] = df_long["region"].astype(int).replace(self.region_dict)
            df_long["season"] = season
            df_long['mu'] = df_long['mu'] + 1
            assert np.all(df_long['mu'] >= 0), f"Non positive mu values in simulation output"
            results.append(df_long)

        res = pd.concat(results, ignore_index=True)
        res = res.sort_values(['season', 't', 'region']).reset_index(drop=True)
        return res

