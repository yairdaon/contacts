import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint
from numpy import exp, log

from src.helper import a2s


class PackerR:
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

    def verify_params(self, params):
        """Verify that all parameters are in valid epidemiological ranges."""
        assert params['beta0'] > 0, f"beta0 == {params['beta0']} must be positive"
        assert np.all(0 <= params['c_vec']), "contact matrix " + a2s(params['c_vec'])
        assert np.all(params['c_vec'] < 1), "contact matrix " + a2s(params['c_vec'])
        assert 0 < params['eps'] < 1, f"eps {params['eps']:.3f}"
        assert 0 < params['rho'] <= 1, f"reporting rate {params['rho']:.3f}"
        
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
        assert params["S_init"].shape == (self.n_seasons, self.n_regions), f"S_init shape {params['S_init'].shape} doesn't match expected ({self.n_seasons}, {self.n_regions})"
        assert params["E_init"].shape == (self.n_seasons, self.n_regions), f"E_init shape {params['E_init'].shape} doesn't match expected ({self.n_seasons}, {self.n_regions})"
        assert params["I_init"].shape == (self.n_seasons, self.n_regions), f"I_init shape {params['I_init'].shape} doesn't match expected ({self.n_seasons}, {self.n_regions})"
        
        # No NaN values
        for key, value in params.items():
            assert not np.any(np.isnan(value)), f"NaN found in {key}"

        
    def pack(self, params):
        """Pack parameters into flat vector (no transformations). S,E,I init first."""
        parts = []

        # S_init, E_init, I_init: each shape (n_seas, n_reg) - PUT THESE FIRST
        for key in ["S_init", "E_init", "I_init"]:
            parts.append(params[key].ravel())

        # beta0: single scalar (no log transformation)
        parts.append([params["beta0"]])

        # c_vec: upper triangular (excluding diagonal) as vector (no logit)
        parts.append(params["c_vec"])

        # omega: vector size n_reg (no transformation)
        parts.append(params["omega"])

        # eps: scalar (no logit)
        parts.append([params["eps"]])
        
        # rho: scalar (no logit)
        parts.append([params["rho"]])

        flat = np.concatenate(parts)
        assert flat.shape == (self.n_params,), f"Packed vector shape {flat.shape} doesn't match expected ({self.n_params},)"
        return flat

    def unpack(self, flat):
        """Unpack flat vector into parameter dictionary (no transformations)."""
        assert flat.shape == (self.n_params,), f"Input vector shape {flat.shape} doesn't match expected ({self.n_params},)"
        out = {}
        idx = 0

        # S_init, E_init, I_init (no transformation) - THESE ARE FIRST
        for key in ["S_init", "E_init", "I_init"]:
            size = self.n_regions * self.n_seasons
            arr = flat[idx:idx+size].reshape(self.n_seasons, self.n_regions)
            out[key] = arr
            idx += size

        # beta0 (no exp)
        out["beta0"] = flat[idx]
        idx += 1

        # c_vec (no expit)
        c_size = len(self.iu[0])
        c_vec = flat[idx:idx+c_size]
        out["c_vec"] = c_vec
        idx += c_size

        # omega (no transformation)
        omega = flat[idx:idx+self.n_regions]
        out["omega"] = omega
        idx += self.n_regions

        # eps (no expit)
        out["eps"] = flat[idx]
        idx += 1
        
        # rho (no expit)
        out["rho"] = flat[idx]
        idx += 1

        return out

    def get_bounds(self):
        """Create scipy Bounds object for constrained optimization."""
        lower = []
        upper = []
        
        # S_init, E_init, I_init: (0, 1) for each - THESE ARE FIRST
        sei_size = 3 * self.n_regions * self.n_seasons
        lower.extend([1e-6] * sei_size)
        upper.extend([1.0 - 1e-6] * sei_size)
        
        # beta0: > 0
        lower.append(1e-6)  # Small positive number instead of 0
        upper.append(np.inf)
        
        # c_vec: [0, 1)
        c_size = len(self.iu[0])
        lower.extend([0.0] * c_size)
        upper.extend([1.0 - 1e-6] * c_size)  # Slightly less than 1
        
        # omega: no bounds (can be negative for phase)
        lower.extend([-np.inf] * self.n_regions)
        upper.extend([np.inf] * self.n_regions)
        
        # eps: (0, 1)
        lower.append(1e-6)
        upper.append(1.0 - 1e-6)
        
        # rho: (0, 1]
        lower.append(1e-6)
        upper.append(1.0)
        
        return Bounds(lower, upper)
    
    def get_linear_constraint(self):
        """Create vectorized LinearConstraint for S + E + I <= 1."""
        n_compartments = self.n_seasons * self.n_regions
        
        # Create constraint matrix - only affects the first 3*n_compartments elements (S,E,I)
        A = np.zeros((n_compartments, self.n_params))
        
        # For each season-region pair i, we want S_i + E_i + I_i <= 1
        for i in range(n_compartments):
            # S_i coefficient (S_init comes first)
            A[i, i] = 1.0
            # E_i coefficient (E_init comes after all S_init)
            A[i, i + n_compartments] = 1.0  
            # I_i coefficient (I_init comes after all S_init and E_init)
            A[i, i + 2*n_compartments] = 1.0
        
        return LinearConstraint(A, lb=0.0, ub=1.0 - 1e-6)

    def c_vec_to_mat(self, c_vec):
        """Convert a compact c_vec representation into a full symmetric matrix with unit diagonal."""
        c_mat = np.eye(self.n_regions)
        c_mat[self.iu] = c_vec
        c_mat[(self.iu[1], self.iu[0])] = c_vec  # symmetry
        return c_mat

    def c_mat_to_vec(self, c_mat):
        return c_mat[self.iu]

    def random_vector(self, seed=None):
        """Generate a random packed vector in the natural parameter space."""
        params = self.random_dict(seed=seed)
        vec = self.pack(params)
        return vec

    def random_dict(self, seed=None):
        """Generate random parameters close to realistic flu values."""
        np.random.seed(seed)

        out = dict(
            beta0=np.random.uniform(0.2, 0.4),  # Flu range around 0.28
            c_vec=np.random.uniform(0.1, 0.9, size=self.c_vec_length),  # Avoid extremes
            omega=np.random.uniform(-0.25, 0.25, size=self.n_regions),  # Seasonal phase around winter
            eps=np.random.uniform(0.3, 0.7),  # Reasonable seasonal variation
            rho=np.random.uniform(0.1, 0.9)  # Reporting rate between 10% and 90%
        )

        # More realistic initial conditions (small but not tiny fractions)
        out["E_init"] = np.random.uniform(1e-5, 1e-3, size=self.n_seasons * self.n_regions).reshape(self.n_seasons, self.n_regions)
        out["I_init"] = np.random.uniform(1e-6, 1e-4, size=self.n_seasons * self.n_regions).reshape(self.n_seasons, self.n_regions)
        # S_init = most of population, ensuring S + E + I < 1
        out["S_init"] = np.random.uniform(0.95, 0.99, (self.n_seasons, self.n_regions)) - out['E_init'] - out['I_init']
        self.verify_params(out)
        return out

    @property
    def c_vec_length(self):
        return len(self.iu[0])
