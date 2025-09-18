import pdb
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import matplotlib

matplotlib.use("MacOSX")

from src.multi import run
from src.rk import run_rk
from src.packer_r import PackerR
from src.losses import LOSSES


class InverterR:
    """
    Inverter class using constrained optimization instead of parameter transformations.
    Uses scipy Bounds and LinearConstraint for optimization.
    """
    def __init__(self,
                 population,
                 n_weeks=26,
                 sigma=0.5,
                 dt_output=7,
                 mu=0 / (30 * 365),
                 nu=0.2,
                 loss='lsq',
                 integration='rk'):
        """population is a dataframe with columns [region, season, population]"""
        self.packer = PackerR(seasons=sorted(population.season.unique()),
                             regions=sorted(population.region.unique()))

        self.n_weeks = n_weeks
        self.sigma = sigma
        self.dt_output = dt_output
        self.mu = mu
        self.nu = nu
        self.integration = integration
        self.pops = {}
        for season in self.packer.seasons:
            pop = (population
                   .query("season == @season")
                   .set_index("region")
                   .loc[self.packer.regions, "population"]
                   .values)
            self.pops[season] = pop

        assert population.shape == (self.packer.n_seasons * self.packer.n_regions, 3), f"Population DataFrame shape {population.shape} doesn't match expected ({self.packer.n_seasons * self.packer.n_regions}, 3)"
        self.loss = LOSSES[loss]
        self.run_time = 0

    def sim(self,
            params=None):
        if params is None:
            params = self.packer.random_dict()
        self.packer.verify_params(params)
        S_init = params['S_init']
        E_init = params['E_init']  
        I_init = params['I_init']
        beta0 = params['beta0']
        omega = params['omega']
        eps = params['eps']
        c_mat = self.packer.c_vec_to_mat(params["c_vec"])

        results = []
        for season_idx, season in enumerate(self.packer.seasons):
            pop = self.pops[season]
            start = time.time()
            if self.integration == 'euler':
                df = run(S_init=S_init[season_idx, :],
                         E_init=E_init[season_idx, :],
                         I_init=I_init[season_idx, :],
                         n_weeks=self.n_weeks,
                         beta0=beta0,
                         sigma=self.sigma,
                         dt_output=self.dt_output,
                         dt_step=1e-2,
                         mu=self.mu,
                         nu=self.nu, omega=omega,
                         eps=eps,
                         contact_matrix=c_mat,
                         population=pop,
                         start_date=season).fillna(0)
            elif self.integration == 'rk':
                df = run_rk(S_init=S_init[season_idx, :],
                            E_init=E_init[season_idx, :],
                            I_init=I_init[season_idx, :],
                            dt_step=1,
                            dt_output=self.dt_output,
                            n_weeks=self.n_weeks,
                            beta0=beta0,
                            sigma=self.sigma,
                            mu=self.mu,
                            nu=self.nu,
                            omega=omega,
                            eps=eps,
                            contact_matrix=c_mat,
                            population=pop,
                            start_date=season)
            else:
                raise ValueError(f"Unknown integration method: {self.integration}. Use 'euler' or 'rk'")
            self.run_time += time.time() - start

            letter = "C"  ## If at any point wed like to look at infecteds instead
            df = df[[col for col in df.columns if letter in col]].reset_index(drop=False)
            assert np.all(df.drop("time", axis=1) >= 0), f"Negative values found in simulation output: {df[df < 0].dropna(how='all')}"
            df_long = df.melt(id_vars=["time"], var_name="region", value_name="incidence")
            df_long["region"] = (df_long["region"]
                                 .str.replace(letter, "")
                                 .astype(int)
                                 .replace(self.packer.region_dict)
                                 )
            df_long["season"] = season
            results.append(df_long)

        res = pd.concat(results, ignore_index=True)

        return res

    def fit(self,
            obs,
            method="L-BFGS-B",
            x0=None,
            seed=None,
            n_starts=10):
        
        # Start timing optimization
        opt_start_time = time.time()
        
        def objective(x):
            assert not np.isnan(x).any(), f"NaN values found in optimization vector: {x[np.isnan(x)]}"
            
            # Unpack parameters (no transformations needed)
            params = self.packer.unpack(x)

            # Run simulation with parameter dictionary
            kk = self.sim(params)
            assert np.all(kk.index == obs.index), f"Simulation and observation indices don't match. Sim: {len(kk)}, Obs: {len(obs)}"
            assert np.all(kk.isnull() == obs.isnull()), f"Simulation and observation null patterns don't match. Sim nulls: {kk.isnull().sum().sum()}, Obs nulls: {obs.isnull().sum().sum()}"
            out = self.loss(obs.dropna(), kk.dropna(), rho=params['rho'])
            assert not np.isnan(out), f"Loss function returned NaN. Loss value: {out}"
            return out

        # Generate starting points
        starts = []
        for start_idx in range(n_starts):
            if start_idx == 0 and x0 is not None:
                x_start = x0
            else:
                x_start = self.packer.random_vector(seed=(seed + start_idx) if seed is not None else None)
            starts.append(x_start)

        # Get bounds and constraints for constrained optimization
        bounds = self.packer.get_bounds()
        constraints = self.packer.get_linear_constraint()

        # Run multiple optimizations in parallel
        options = dict(maxiter=5000, disp=False)
        results = Parallel(n_jobs=-1)(
            delayed(minimize)(
                objective, 
                x0=x, 
                method=method, 
                bounds=bounds,
                constraints=constraints,
                options=options
            ) for x in starts
        )
        
        # Filter out failed optimizations and find best result
        successful_results = [res for res in results if res is not None and res.success]
        
        if not successful_results:
            raise RuntimeError("All optimization runs failed")
        
        # Select best result based on objective value
        best_result = min(successful_results, key=lambda r: r.fun)
        
        # Record total optimization time
        self.optimization_time = time.time() - opt_start_time
        
        print(f"Completed {len(successful_results)}/{n_starts} successful optimizations")
        print(f"Best objective: {best_result.fun:.3f}")
        print(f"Total optimization time: {self.optimization_time/60:.2f}s")

        self.params = self.packer.unpack(best_result.x)
        self.fun = best_result.fun
        return self