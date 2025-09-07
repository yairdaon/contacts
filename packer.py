import numpy as np
import pandas as pd
from scipy.special import logit, expit
from numpy import exp, log


class Packer:
    def __init__(self,
                 regions=None,
                 seasons=None):

        if regions is None:
            regions = ["HHS1", "HHS2"]
        if seasons is None:
            seasons = ["1900-01-01", "2000-01-01", "2100-01-01"]
        self.regions = regions
        self.n_regions = len(self.regions)
        self.region_dict = dict(zip(range(self.n_regions), regions)) 
        self.seasons = seasons
        self.n_seasons = len(self.seasons)

        # precompute upper-triangular indices (excluding diagonal)
        self.iu = np.triu_indices(self.n_regions, k=1)

        # compute parameter count directly
        self.n_params = 1 + len(self.iu[0]) + self.n_regions + 1 + 3 * self.n_regions * self.n_seasons

    def real2pop(self, params):
        """Generates fractions of population, not absolute numbers"""
        S_init = exp(params['S_init'])
        E_init = exp(params['E_init'])
        I_init = exp(params['I_init'])
        tot = S_init + E_init + I_init + 1  # eq R_init = 0
        params["S_init"] = S_init/tot
        params["E_init"] = E_init/tot
        params["I_init"] = I_init/tot
        self.verify(params)
        return params

    def pop2real(self, params):
        params["S_init"] = log(params['S_init'])
        params["E_init"] = log(params['E_init'])
        params["I_init"] = log(params['I_init'])
        return params

    def verify(self, params):
        """Verify that all parameters are in valid epidemiological ranges."""
        assert params['beta0'] > 0, "beta0 must be positive"
        assert np.all(0 <= params['c_vec']), "contact matrix elements must be > 0"
        assert np.all(params['c_vec'] < 1), "contact matrix elements must be < 1"
        assert 0 < params['eps'] < 1, "seasonal amplitude eps must be in (0,1)"
        
        # NEW: omega verification for seasonal phase (fraction of year)
        assert np.all(-0.5 <= params['omega']), "omega must be >= -0.5"
        assert np.all(params['omega'] <= 0.5), "omega must be <= 0.5"
        
        # Compartment fractions must be positive and < 1
        assert np.all(0 < params["S_init"]), "S_init must be positive"
        assert np.all(0 < params["E_init"]), "E_init must be positive"
        assert np.all(0 < params["I_init"]), "I_init must be positive"
        assert np.all(params["S_init"] < 1), "S_init must be < 1"
        assert np.all(params["E_init"] < 1), "E_init must be < 1"
        assert np.all(params["I_init"] < 1), "I_init must be < 1"
        
        # Total compartments must sum to < 1 (assuming R_init = 1 - S - E - I)
        tot = params["S_init"] + params["I_init"] + params["E_init"]
        assert np.all(tot < 1), "S + E + I must be < 1 (leaving room for R)"
        
        # Shape verification
        assert params["S_init"].shape == (self.n_seasons, self.n_regions)
        assert params["E_init"].shape == (self.n_seasons, self.n_regions)
        assert params["I_init"].shape == (self.n_seasons, self.n_regions)
        
        # No NaN values
        for key, value in params.items():
            assert not np.any(np.isnan(value)), f"NaN found in {key}"

        
    def pack(self, params):
        parts = []

        # beta0: single scalar
        parts.append([log(params["beta0"])])

        # c_vec: upper triangular (excluding diagonal) as vector
        parts.append(logit(params["c_vec"]))

        # omega: vector size n_reg
        parts.append(params["omega"])

        # eps: scalar
        parts.append([logit(params["eps"])])

        # S_init, I_init, E_init: each shape (n_seas, n_reg)
        for key in ["S_init", "I_init", "E_init"]:
            parts.append(params[key].ravel())

        flat = np.concatenate(parts)
        assert flat.shape == (self.n_params,)
        return flat

    
    # ---- unpack ----
    def unpack(self, flat):
        assert flat.shape == (self.n_params,)
        out = {}
        idx = 0

        # beta0
        out["beta0"] = exp(flat[idx])
        idx += 1

        # c_vec
        c_size = len(self.iu[0])
        c_vec = expit(flat[idx:idx+c_size])
        out["c_vec"] = c_vec
        idx += c_size

        # omega
        omega = flat[idx:idx+self.n_regions]
        out["omega"] = omega
        idx += self.n_regions

        # eps
        out["eps"] = expit(flat[idx])
        idx += 1

        # S_init, I_init, E_init
        for key in ["S_init", "I_init", "E_init"]:
            size = self.n_regions * self.n_seasons
            arr = flat[idx:idx+size].reshape(self.n_seasons, self.n_regions)
            out[key] = arr
            idx += size

        return out

    def c_vec_to_mat(self, c_vec):
        """Convert a compact c_vec representation into a full symmetric matrix with unit diagonal."""
        c_mat = np.eye(self.n_regions)
        c_mat[self.iu] = c_vec
        c_mat[(self.iu[1], self.iu[0])] = c_vec  # symmetry
        return c_mat

    def c_mat_to_vec(self, c_mat):
        return c_mat[self.iu]

    def random_vector(self, seed=None):
        """Generate a random packed vector in the transformed parameter space."""
        params = self.random_dict(seed=seed)
        vec = self.pack(self.pop2real(params))
        return vec

    def random_dict(self, seed=None):
        """Generate random parameters close to realistic flu values."""
        np.random.seed(seed)

        out = dict(
            beta0=np.random.uniform(0.2, 0.4),  # Flu range around 0.28
            c_vec=np.random.uniform(0.1, 0.9, size=self.c_vec_length),  # Avoid extremes
            omega=np.random.uniform(-0.25, 0.25, size=self.n_regions),  # Seasonal phase around winter
            eps=np.random.uniform(0.3, 0.7)  # Reasonable seasonal variation
        )

        # More realistic initial conditions (small but not tiny fractions)
        out["E_init"] = np.random.uniform(1e-5, 1e-3, size=self.n_seasons * self.n_regions).reshape(self.n_seasons, self.n_regions)
        out["I_init"] = np.random.uniform(1e-6, 1e-4, size=self.n_seasons * self.n_regions).reshape(self.n_seasons, self.n_regions)
        # S_init = most of population, ensuring S + E + I < 1
        out["S_init"] = np.random.uniform(0.95, 0.99, (self.n_seasons, self.n_regions)) - out['E_init'] - out['I_init']
        self.verify(out)
        return out


    @property
    def c_vec_length(self):
        return len(self.iu[0])


if __name__ == "__main__":
    # Tests have been moved to tests/test_packer.py
    print("Tests moved to tests/test_packer.py")
