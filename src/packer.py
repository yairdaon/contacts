import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint
from scipy.special import logit, expit
from numpy import exp, log

from src.helper import a2s

EPS = 1e-6
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
        # Order: S,E,I init (3*n_regions*n_seasons), beta0, c_vec, omega, eps, rho
        self.n_params = 3 * self.n_regions * self.n_seasons + 1 + len(self.iu[0]) + self.n_regions + 1 + 1

    def verify(self, params):
        if type(params) is np.ndarray:
            params = self.unpack(params)

        beta0 = params['beta0']
        c_vec = params['c_vec']
        eps = params['eps']
        rho = params['rho']
        # theta = params['theta']
        # omega = params['omega']
        S_init = params['S_init']
        E_init = params['E_init']
        I_init = params['I_init']

        assert beta0 > 0, f"beta0 == {beta0} must be positive"
        assert np.all(0 <= c_vec), f"contact matrix {a2s(c_vec)} must be >= 0"
        assert np.all(c_vec <= 1), f"contact matrix {a2s(c_vec)} must be <= 1"
        assert 0 <= eps < 1, f"eps {eps:.3f} must be in (0,1)"
        assert 0 < rho <= 1, f"reporting rate {rho:.3f} must be in (0,1]"
        #assert 0 < theta, f"overdispersion {theta:.3f} must be > 0"

        # omega verification for seasonal phase (fraction of year)
        # assert np.all(-0.5 <= params['omega']), "omega must be >= -0.5"
        # assert np.all(params['omega'] <= 0.5), "omega must be <= 0.5"

        # Compartment fractions must be positive and < 1
        assert np.all(0 < S_init), f"S_init must be positive. Min: {S_init.min():.6f}, values: {S_init.ravel()[:5]}"
        assert np.all(0 < E_init), f"E_init must be positive. Min: {E_init.min():.6f}, values: {E_init.ravel()[:5]}"
        assert np.all(0 < I_init), f"I_init must be positive. Min: {I_init.min():.6f}, values: {I_init.ravel()[:5]}"
        assert np.all(S_init < 1), f"S_init must be < 1. Max: {S_init.max():.6f}, values: {S_init.ravel()[:5]}"
        assert np.all(E_init < 1), f"E_init must be < 1. Max: {E_init.max():.6f}, values: {E_init.ravel()[:5]}"
        assert np.all(I_init < 1), f"I_init must be < 1. Max: {I_init.max():.6f}, values: {I_init.ravel()[:5]}"

        # Total compartments must sum to < 1 (assuming R_init = 1 - S - E - I)
        tot = S_init + I_init + E_init
        assert np.all(tot < 1), f"S + E + I must be < 1. Max sum: {tot.max():.6f}, violating indices: {np.where(tot >= 1)}"

        # Shape verification
        assert S_init.shape == (self.n_seasons, self.n_regions), f"S_init shape {S_init.shape} doesn't match expected ({self.n_seasons}, {self.n_regions})"
        assert E_init.shape == (self.n_seasons, self.n_regions), f"E_init shape {E_init.shape} doesn't match expected ({self.n_seasons}, {self.n_regions})"
        assert I_init.shape == (self.n_seasons, self.n_regions), f"I_init shape {I_init.shape} doesn't match expected ({self.n_seasons}, {self.n_regions})"

        # No NaN values
        for key, value in params.items():
            assert not np.any(np.isnan(value)), f"NaN found in {key}"

    def pack(self, params):
        parts = []

        S_init = params["S_init"]
        E_init = params["E_init"]
        I_init = params["I_init"]
        
        # Apply simplex transformation: f(x) = log(x / (1 - sum(x)))
        # Stack S, E, I along new axis: shape (n_seasons, n_regions, 3)
        SEI = np.stack([S_init, E_init, I_init], axis=2)
        
        # Apply transformation along the last axis
        sum_SEI = np.sum(SEI, axis=2, keepdims=True)  # shape (n_seasons, n_regions, 1)
        transformed = np.log(SEI / (1 - sum_SEI))  # shape (n_seasons, n_regions, 3)
        
        # Flatten and add to parts
        parts.append(transformed.ravel())

        parts.append([log(params["beta0"])])
        parts.append(logit(params["c_vec"]))  # c_vec: upper triangular (excluding diagonal) as vector
        parts.append(params["omega"] % 1)  # omega: vector size n_reg
        parts.append([logit(params["eps"])])  # eps: scalar
        parts.append([logit(params["rho"])])  # rho: scalar

        flat = np.concatenate(parts)
        assert flat.shape == (self.n_params,), f"Packed vector shape {flat.shape} != ({self.n_params},)"
        return flat

    def unpack(self, flat):
        err = f"Input vector shape {flat.shape} doesnt match expected ({self.n_params},)"
        assert flat.shape == (self.n_params,), err

        out = {}
        idx = 0

        # Unpack transformed SEI values
        sei_size = 3 * self.n_seasons * self.n_regions
        transformed_flat = flat[idx:idx + sei_size]
        idx += sei_size
        
        # Reshape to (n_seasons, n_regions, 3)
        transformed = transformed_flat.reshape(self.n_seasons, self.n_regions, 3)
        
        # Apply inverse transformation: g(y) = exp(y) / (1 + sum(exp(y)))
        exp_transformed = np.exp(transformed)  # shape (n_seasons, n_regions, 3)
        sum_exp = np.sum(exp_transformed, axis=2, keepdims=True)  # shape (n_seasons, n_regions, 1)
        SEI = exp_transformed / (1 + sum_exp)  # shape (n_seasons, n_regions, 3)
        
        # Extract S, E, I
        out["S_init"] = SEI[:, :, 0]
        out["E_init"] = SEI[:, :, 1] 
        out["I_init"] = SEI[:, :, 2]

        out["beta0"] = exp(flat[idx])
        idx += 1

        c_size = len(self.iu[0])
        c_vec = flat[idx:idx + c_size]
        out["c_vec"] = expit(c_vec)
        idx += c_size

        omega = flat[idx:idx + self.n_regions]
        out["omega"] = omega % 1
        idx += self.n_regions

        out["eps"] = expit(flat[idx])
        idx += 1

        out["rho"] = expit(flat[idx])
        idx += 1

        return out

    def c_vec_to_mat(self, c_vec):
        c_mat = np.eye(self.n_regions)
        c_mat[self.iu] = c_vec
        c_mat[(self.iu[1], self.iu[0])] = c_vec  # symmetry
        return c_mat

    def random_vector(self, seed=None):
        params = self.random_dict(seed=seed)
        vec = self.pack(params)
        return vec

    def random_dict(self, seed=None):
        np.random.seed(seed)

        out = dict(
            beta0=np.random.uniform(0.2, 0.4),  # Flu range around 0.28
            c_vec=np.random.uniform(0.1, 0.9, size=len(self.iu[0])),  # Avoid extremes
            eps=np.random.uniform(0.3, 0.7),  # Reasonable seasonal variation
            rho=np.random.uniform(0.1, 0.9),  # Reporting rate between 10% and 90%
            omega=np.random.uniform(0, 1, size=self.n_regions)  # omega in [0,1]
        )

        # More realistic initial conditions (small but not tiny fractions)
        out["E_init"] = np.random.uniform(1e-5, 1e-3, size=self.n_seasons * self.n_regions).reshape(self.n_seasons,
                                                                                                    self.n_regions)
        out["I_init"] = np.random.uniform(1e-6, 1e-4, size=self.n_seasons * self.n_regions).reshape(self.n_seasons,
                                                                                                    self.n_regions)
        out["S_init"] = np.random.uniform(0.7, 0.8, (self.n_seasons, self.n_regions)) - out['E_init'] - out['I_init']
        self.verify(out)
        return out