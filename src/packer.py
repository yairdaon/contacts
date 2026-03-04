import numpy as np
import pandas as pd
from numpy import exp, log
from pprint import pprint

from src import compute_g

SLIM = (0.8, 0.99)
ILIM = (1e-5, 1e-3)

class Packer:
    def __init__(self,
                 disease,
                 seasons=None,
                 regions=None,
                 all_Ts=None):

        self.disease = disease
        self.regions = regions if regions is not None else ["HHS0", "HHS1"]
        self.seasons = seasons if seasons is not None else [1990, 1991, 1992]
        self.n_regions = len(self.regions)
        self.n_seasons = len(self.seasons)
        assert self.n_regions == 2 ## For this branch only
        self.region_dict = dict(zip(range(self.n_regions), self.regions))
        self.all_Ts = {season: season + np.arange(disease.n_weeks) * disease.step_size for season in self.seasons} if all_Ts is None else all_Ts

        # Count parameters: S, I init (2*n_regions*n_seasons), theta,
        self.n_params = 2 * self.n_regions * self.n_seasons + 1 


    def random_vector(self, seed=None):
        params = self.random_dict(seed=seed)
        vec = self.pack(params)
        return vec

    
    def random_dict(self, seed=None):
        np.random.seed(seed)

        out = dict(
            theta=np.random.uniform(0.0, 0.5),  # Within [0, 0.5] bounds (max half population commuting)
        )
        out["I_init"] = np.random.uniform(*ILIM, size=(self.n_seasons, self.n_regions))
        out["S_init"] = np.random.uniform(*SLIM, size=(self.n_seasons, self.n_regions))
        return out

    def verify(self, flat):
        assert flat.shape == (self.n_params,), f"Packed vector shape {flat.shape} != ({self.n_params},)"
        assert np.all(flat >= 0)   
        tot = flat[:-1].reshape(2,-1).sum(axis=0)
        assert np.all(tot <= 1)
        

        
    def pack(self, params):
        parts = []

        S_init = params["S_init"]
        I_init = params["I_init"]
        parts.append(S_init.ravel())
        parts.append(I_init.ravel())
        parts.append([params["theta"]]) 

        flat = np.concatenate(parts)
        #self.verify(flat)
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

        theta = flat[idx]
        out["theta"] = theta
        idx += 1

        assert idx == flat.size
        #self.verify(flat)
        return out


    def sim(self, params, phase, disease):
        """Simulate and return incidence + Jacobian columns for gradient computation."""
        S_init = params['S_init']
        I_init = params['I_init']
        theta = params["theta"]

        results = []
        for season_idx, season in enumerate(self.seasons):
            S = S_init[season_idx, :]
            I = I_init[season_idx, :]

            df = compute_g.contacts(S0=S,
                                    I0=I,
                                    gamma=disease.gamma,
                                    theta=theta,
                                    Ts=self.all_Ts[season],
                                    beta0=disease.beta0,
                                    eps=disease.eps,
                                    phase=phase)
            
            df = df.reset_index()
            df['season_idx'] = season_idx
            df_long = df[['t', 'j', 'mu', 'season_idx', 'theta', 'S1_0', 'I1_0', 'S2_0', 'I2_0']].rename(
                columns={'j': 'region', 'mu': 'incidence'})
            df_long["region"] = df_long["region"].astype(int).replace(self.region_dict)
            df_long["season"] = season
            assert np.all(df_long['incidence'] >= 0), f"Negative incidence values in simulation output"
            results.append(df_long)

        res = pd.concat(results, ignore_index=True)
        return res

