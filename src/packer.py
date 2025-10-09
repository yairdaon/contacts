import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint
from scipy.special import logit, expit
from numpy import exp, log

from src.helper import a2s, fwd, bckwd

SLIM = (0.8, 0.99)
ILIM = (1e-5, 1e-3)
ELIM = (1e-5, 1e-3)

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
        # Order: S, E. I init (3*n_regions*n_seasons), beta0, c_vec, eps, omega
        self.n_params = 3 * self.n_regions * self.n_seasons + 1 + len(self.iu[0]) + 1 + 1




    def verify(self, params):
        pass
        # if type(params) is np.ndarray:
        #     params = self.unpack(params)
        #
        # beta0 = params['beta0']
        # c_vec = params['c_vec']
        # eps = params['eps']
        # rho = params['rho']
        # # theta = params['theta']
        # # omega = params['omega']
        # S_init = params['S_init']
        # E_init = params['E_init']
        # I_init = params['I_init']
        #
        # assert beta0 >= 0, f"beta0 == {beta0} must be positive"
        # assert np.all(0 <= c_vec), f"contact matrix {a2s(c_vec)} must be >= 0"
        # assert np.all(c_vec < 1+EPS), f"contact matrix {a2s(c_vec)} must be <= 1"
        # assert 0 <= eps < 1, f"eps {eps:.3f} must be in (0,1)"
        # assert 0 < rho <= 1, f"reporting rate {rho:.3f} must be in (0,1]"
        # #assert 0 < theta, f"overdispersion {theta:.3f} must be > 0"
        #
        # # omega verification for seasonal phase (fraction of year)
        # # assert np.all(-0.5 <= params['omega']), "omega must be >= -0.5"
        # # assert np.all(params['omega'] <= 0.5), "omega must be <= 0.5"
        #
        # # Compartment fractions must be positive and < 1
        # assert np.all(-EPS < S_init), f"S_init must be positive. Min: {S_init.min():.6f}, values: {S_init.ravel()[:5]}"
        # assert np.all(-EPS < E_init), f"E_init must be positive. Min: {E_init.min():.6f}, values: {E_init.ravel()[:5]}"
        # assert np.all(-EPS < I_init), f"I_init must be positive. Min: {I_init.min():.6f}, values: {I_init.ravel()[:5]}"
        # assert np.all(S_init < 1+EPS), f"S_init must be < 1. Max: {S_init.max():.6f}, values: {S_init.ravel()[:5]}"
        # assert np.all(E_init < 1+EPS), f"E_init must be < 1. Max: {E_init.max():.6f}, values: {E_init.ravel()[:5]}"
        # assert np.all(I_init < 1+EPS), f"I_init must be < 1. Max: {I_init.max():.6f}, values: {I_init.ravel()[:5]}"
        #
        # # Total compartments must sum to < 1 (assuming R_init = 1 - S - E - I)
        # tot = S_init + I_init + E_init
        # assert np.all(tot < 1+EPS), f"S + E + I must be < 1. Max sum: {tot.max():.6f}, violating indices: {np.where(tot >= 1)}"
        #
        # # Shape verification
        # assert S_init.shape == (self.n_seasons, self.n_regions), f"S_init shape {S_init.shape} doesn't match expected ({self.n_seasons}, {self.n_regions})"
        # assert E_init.shape == (self.n_seasons, self.n_regions), f"E_init shape {E_init.shape} doesn't match expected ({self.n_seasons}, {self.n_regions})"
        # assert I_init.shape == (self.n_seasons, self.n_regions), f"I_init shape {I_init.shape} doesn't match expected ({self.n_seasons}, {self.n_regions})"
        #
        # # No NaN values
        # for key, value in params.items():
        #     assert not np.any(np.isnan(value)), f"NaN found in {key}"


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
            c_vec=np.random.uniform(0.1, 0.9, size=len(self.iu[0])),  # Within [0.01, 0.99] bounds
            eps=np.random.uniform(0.3, 0.7),  # Within [0.01, 0.99] bounds  
            omega=np.random.uniform(0, 1)  # omega scalar in [0,1]
        )

        # Initial conditions
        out["E_init"] = np.random.uniform(*ELIM, size=(self.n_seasons, self.n_regions))
        out["I_init"] = np.random.uniform(*ILIM, size=(self.n_seasons, self.n_regions))
        out["S_init"] = np.random.uniform(*SLIM, size=(self.n_seasons, self.n_regions))
        self.verify(out)
        return out


class Trans(Packer):

    def pack(self, params):
        parts = []

        S_init = params["S_init"]
        E_init = params["E_init"]
        I_init = params["I_init"]

        # Apply individual transformations with specific bounds
        # I: [1e-6, 0.05], S: [0.1, 1-2e-6] # , E: [1e-6, 0.05] (E_init = I_init)
        parts.append(fwd(S_init, *SLIM).ravel())
        parts.append(fwd(E_init, *ELIM).ravel())
        parts.append(fwd(I_init, *ILIM).ravel())

        parts.append([log(params["beta0"])])
        parts.append(fwd(params["c_vec"], 0.01, 0.99))  # c_vec: upper triangular (excluding diagonal) as vector
        parts.append([fwd(params["eps"], 0.01, 0.99)])  # eps: scalar
        parts.append([params["omega"] % 1])  # omega: scalar
        # parts.append([fwd(params["rho"], 0.01, 0.99)])  # rho: scalar - FIXED AT 0.8

        flat = np.concatenate(parts)
        assert flat.shape == (self.n_params,), f"Packed vector shape {flat.shape} != ({self.n_params},)"
        return flat

    def unpack(self, flat):
        err = f"Input vector shape {flat.shape} doesnt match expected ({self.n_params},)"
        assert flat.shape == (self.n_params,), err

        out = {}
        idx = 0

        # Unpack individual S, I values # , E (E_init = I_init)
        M = self.n_seasons * self.n_regions

        s_flat = flat[idx:idx + M]
        idx += M
        e_flat = flat[idx:idx + M]  # E_init = I_init
        idx += M
        i_flat = flat[idx:idx + M]
        idx += M

        # Apply inverse transformations with specific bounds
        out["S_init"] = bckwd(s_flat, *SLIM).reshape(self.n_seasons, self.n_regions)
        out["E_init"] = bckwd(e_flat, *ELIM).reshape(self.n_seasons, self.n_regions)
        out["I_init"] = bckwd(i_flat, *ILIM).reshape(self.n_seasons, self.n_regions)

        out["beta0"] = exp(flat[idx])
        idx += 1

        c_size = len(self.iu[0])
        c_vec = flat[idx:idx + c_size]
        out["c_vec"] = bckwd(c_vec, 0.01, 0.99)
        idx += c_size


        out["eps"] = bckwd(flat[idx], 0.01, 0.99)
        idx += 1

        # omega = flat[idx:idx + self.n_regions]  # omega now scalar
        out["omega"] = flat[idx] % 1
        idx += 1

        # out["rho"] = bckwd(flat[idx], 0.01, 0.99)  # FIXED AT 0.8
        # idx += 1

        assert idx == flat.size
        return out


class Straight(Packer):
    def pack(self, params):
        parts = []

        S_init = params["S_init"]
        E_init = params["E_init"]
        I_init = params["I_init"]

        # No transformations - direct packing
        parts.append(S_init.ravel())
        parts.append(E_init.ravel())
        parts.append(I_init.ravel())

        parts.append([params["beta0"]])
        parts.append(params["c_vec"])  # c_vec: upper triangular (excluding diagonal) as vector
        parts.append([params["eps"]])  # eps: scalar
        parts.append([params["omega"]])  # omega: scalar
        # parts.append([params["rho"]])  # rho: scalar - FIXED AT 0.8

        flat = np.concatenate(parts)
        assert flat.shape == (self.n_params,), f"Packed vector shape {flat.shape} != ({self.n_params},)"
        return flat

    def unpack(self, flat):
        err = f"Input vector shape {flat.shape} doesnt match expected ({self.n_params},)"
        assert flat.shape == (self.n_params,), err

        out = {}
        idx = 0

        # Unpack individual S, I values - no transformations # , E (E_init = I_init)
        M = self.n_seasons * self.n_regions

        s_flat = flat[idx:idx + M]
        idx += M
        e_flat = flat[idx:idx + M]
        idx += M
        i_flat = flat[idx:idx + M]
        idx += M

        # No inverse transformations
        out["S_init"] = s_flat.reshape(self.n_seasons, self.n_regions)
        out["E_init"] = e_flat.reshape(self.n_seasons, self.n_regions)
        out["I_init"] = i_flat.reshape(self.n_seasons, self.n_regions)

        out["beta0"] = flat[idx]
        idx += 1

        c_size = len(self.iu[0])
        c_vec = flat[idx:idx + c_size]
        out["c_vec"] = c_vec
        idx += c_size


        out["eps"] = flat[idx]
        idx += 1

        # omega = flat[idx:idx + self.n_regions]  # omega now scalar
        out["omega"] = flat[idx]
        idx += 1

        # out["rho"] = flat[idx]  # FIXED AT 0.8
        # idx += 1

        assert idx == flat.size
        return out
