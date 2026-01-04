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
    """
    Base class for parameter packing/unpacking in SEIR optimization.
    
    Handles conversion between epidemiological parameter dictionaries
    and flat optimization vectors, with support for multi-region and
    multi-season models.
    """
    
    def __init__(self,
                 regions=None,
                 seasons=None,
                 seasonal_driver=True):
        """
        Initialize parameter packer.
        
        Parameters:
        -----------
        regions : list, optional
            Region names (default: ["HHS1", "HHS2"])
        seasons : list, optional
            Season start dates (default: 3 seasons from 1900-2100)
        """

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
        self.seasonal_driver = seasonal_driver



    def verify(self, params):
        pass

    def c_vec_to_mat(self, c_vec):
        """
        Convert contact vector to symmetric contact matrix.
        
        Parameters:
        -----------
        c_vec : array
            Upper triangular contact coefficients (excluding diagonal)
            
        Returns:
        --------
        array
            Symmetric contact matrix with unit diagonal
        """
        c_mat = np.eye(self.n_regions)
        c_mat[self.iu] = c_vec
        c_mat[(self.iu[1], self.iu[0])] = c_vec  # symmetry
        return c_mat

    def random_vector(self, seed=None):
        """
        Generate random parameter vector for optimization.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        array
            Random parameter vector
        """
        params = self.random_dict(seed=seed)
        vec = self.pack(params)
        return vec

    def random_dict(self, seed=None):
        """
        Generate random parameter dictionary.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Random parameter dictionary with epidemiologically reasonable values
        """
        np.random.seed(seed)

        out = dict(
            beta0=np.random.uniform(0.2, 0.4),  # Flu range around 0.28
            c_vec=np.random.uniform(0.1, 0.9, size=len(self.iu[0])),  # Within [0.01, 0.99] bounds
            eps=np.random.uniform(0.3, 0.7) * self.seasonal_driver,  # Within [0.01, 0.99] bounds  
            omega=np.random.uniform(0, 1) * self.seasonal_driver  # omega scalar in [0,1]
        )

        if self.seasonal_driver:
            r = np.random.uniform(0,1,size=(4, self.n_seasons, self.n_regions))
            r = r / r.sum(axis=0)
            out["S_init"] = r[0, :, :]
            out["E_init"] = r[1, :, :]
            out["I_init"] = r[2, :, :]
            
        else:
            # Initial conditions
            out["E_init"] = np.random.uniform(*ELIM, size=(self.n_seasons, self.n_regions))
            out["I_init"] = np.random.uniform(*ILIM, size=(self.n_seasons, self.n_regions))
            out["S_init"] = np.random.uniform(*SLIM, size=(self.n_seasons, self.n_regions))
            

        
        self.verify(out)
        return out


class Trans(Packer):
    """
    Parameter packer with transformations for unconstrained optimization.
    
    Applies logit and log transformations to ensure parameters stay within
    valid bounds during optimization, enabling the use of unconstrained
    optimizers like L-BFGS-B.
    """

    def pack(self, params):
        """
        Pack parameters into flat vector with transformations.
        
        Parameters:
        -----------
        params : dict
            Parameter dictionary with epidemiological parameters
            
        Returns:
        --------
        array
            Flat transformed parameter vector for optimization
        """
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
        """
        Unpack flat vector into parameter dictionary with inverse transformations.
        
        Parameters:
        -----------
        flat : array
            Flat transformed parameter vector
            
        Returns:
        --------
        dict
            Parameter dictionary with epidemiological parameters
        """
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
    """
    Parameter packer without transformations for constrained optimization.
    
    Direct packing/unpacking without transformations, suitable for
    constrained optimizers that can handle parameter bounds explicitly.
    """
    
    def pack(self, params):
        """
        Pack parameters into flat vector without transformations.
        
        Parameters:
        -----------
        params : dict
            Parameter dictionary with epidemiological parameters
            
        Returns:
        --------
        array
            Flat parameter vector for constrained optimization
        """
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
        """
        Unpack flat vector into parameter dictionary without transformations.
        
        Parameters:
        -----------
        flat : array
            Flat parameter vector
            
        Returns:
        --------
        dict
            Parameter dictionary with epidemiological parameters
        """
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
