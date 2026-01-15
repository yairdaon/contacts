import numpy as np
import pandas as pd
from numpy import exp, log
from pprint import pprint

SLIM = (0.8, 0.99)
ILIM = (1e-5, 1e-3)

class Packer:
    def __init__(self,
                 seasons=None,
                 regions=None,
                 seasonal_driver=True):

        self.regions = regions if regions is not None else ["HHS0", "HHS1"]
        self.seasons = seasons if seasons is not None else ["1900-01-01", "2000-01-01", "2100-01-01"] 
        self.n_regions = len(self.regions)
        self.n_seasons = len(self.seasons)
        assert self.n_regions == 2 ## For this branch only
        self.region_dict = dict(zip(range(self.n_regions), self.regions))

        # Count parameters: S, I init (2*n_regions*n_seasons), theta,
        self.n_params = 2 * self.n_regions * self.n_seasons + 1 
        self.seasonal_driver = seasonal_driver


    def random_vector(self, seed=None):
        params = self.random_dict(seed=seed)
        vec = self.pack(params)
        return vec

    
    def random_dict(self, seed=None):
        np.random.seed(seed)

        out = dict(
            theta=np.random.uniform(0.1, 0.9),  # Within [0.01, 0.99] bounds
        )
        out["I_init"] = np.random.uniform(*ILIM, size=(self.n_seasons, self.n_regions))
        out["S_init"] = np.random.uniform(*SLIM, size=(self.n_seasons, self.n_regions))
        return out
   
    
    def pack(self, params):
        parts = []

        S_init = params["S_init"]
        I_init = params["I_init"]
        parts.append(S_init.ravel())
        parts.append(I_init.ravel())
        parts.append([params["theta"]]) 

        flat = np.concatenate(parts)
        assert flat.shape == (self.n_params,), f"Packed vector shape {flat.shape} != ({self.n_params},)"
        return flat

    def unpack(self, flat):
        assert flat.shape == (self.n_params,)

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
        return out
