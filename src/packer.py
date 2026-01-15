import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint
from scipy.special import logit, expit
from numpy import exp, log
from pprint import pprint

from src.helper import a2s, fwd, bckwd

SLIM = (0.8, 0.99)
ILIM = (1e-5, 1e-3)

class Packer:
    """
    Base class for parameter packing/unpacking in SEIR optimization.
    
    Handles conversion between epidemiological parameter dictionaries
    and flat optimization vectors, with support for multi-region and
    multi-season models.
    """
    
    def __init__(self,
                 seasons=None,
                 regions=None,
                 seasonal_driver=True):
        """
        Initialize parameter packer.
        
        Parameters:
        -----------
        seasons : list, optional
            Season start dates (default: 3 seasons from 1900-2100)
        """

        self.regions = regions if regions is not None else ["HHS0", "HHS1"]
        self.seasons = seasons if seasons is not None else ["1900-01-01", "2000-01-01", "2100-01-01"] 
        self.n_regions = len(self.regions)
        self.n_seasons = len(self.seasons)
        assert self.n_regions == 2 ## For this branch only
        self.region_dict = dict(zip(range(self.n_regions), self.regions))



        # compute parameter count directly
        # Order: S, I init (2*n_regions*n_seasons), beta0, theta, eps, omega
        self.n_params = 2 * self.n_regions * self.n_seasons + 1 + 1 + 1 + 1
        self.seasonal_driver = seasonal_driver


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
            theta=np.random.uniform(0.1, 0.9),  # Within [0.01, 0.99] bounds
            eps=np.random.uniform(0.3, 0.7) * self.seasonal_driver,  # Within [0.01, 0.99] bounds  
            omega=np.random.uniform(0, 1) * self.seasonal_driver  # omega scalar in [0,1]
        )

        if self.seasonal_driver:
            r = np.random.uniform(0,1,size=(4, self.n_seasons, self.n_regions))
            r = r / r.sum(axis=0)
            out["S_init"] = r[0, :, :]
            out["I_init"] = r[2, :, :]
            
        else:
            # Initial conditions
            out["I_init"] = np.random.uniform(*ILIM, size=(self.n_seasons, self.n_regions))
            out["S_init"] = np.random.uniform(*SLIM, size=(self.n_seasons, self.n_regions))
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
        I_init = params["I_init"]

        # No transformations - direct packing
        parts.append(S_init.ravel())
        parts.append(I_init.ravel())

        parts.append([params["beta0"]])
        parts.append([params["theta"]]) 
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
        i_flat = flat[idx:idx + M]
        idx += M

        # No inverse transformations
        out["S_init"] = s_flat.reshape(self.n_seasons, self.n_regions)
        out["I_init"] = i_flat.reshape(self.n_seasons, self.n_regions)

        out["beta0"] = flat[idx]
        idx += 1

        theta = flat[idx]
        out["theta"] = theta
        idx += 1


        out["eps"] = flat[idx]
        idx += 1

        # omega = flat[idx:idx + self.n_regions]  # omega now scalar
        out["omega"] = flat[idx]
        idx += 1

        # out["rho"] = flat[idx]  # FIXED AT 0.8
        # idx += 1

        assert idx == flat.size
        return out
