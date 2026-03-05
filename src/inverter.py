import copy
import matplotlib
import nlopt
import numpy as np
import pandas as pd
from datetime import timedelta
from joblib import delayed, Parallel
from scipy.optimize import minimize
from tqdm import tqdm
from pprint import pprint

from src import losses
from src.packer import Packer
from src import compute_g

class Objective:
    def __init__(self,
                 obs,
                 phase,
                 disease):

        self.packer = Packer(disease=disease,
                             seasons=obs['season'].unique(),
                             regions=obs['region'].unique(),
                             all_Ts={season: np.sort(dd['t'].unique()) for season, dd in obs.groupby("season")})
        self.phase = phase
        self.loss = losses.gaussian
        self.obs = obs
        self.disease = disease
        self.x_list = []
        self.out_list = []
        

    def compute_gradient(self, sim_df):
        """
        Compute gradient of negative log-likelihood with respect to parameters.

        From the paper, the score function (derivative of log-likelihood) is:
            ∂ℓ/∂φ = Σ A_i(t) * ∂μ_i(t)/∂φ

        where:
            A_i(t) = -1/(2μ) + r/((1-ρ)μ) + r²/(2ρ(1-ρ)μ²)

        For minimization of negative log-likelihood L = -ℓ:
            ∂L/∂φ = -Σ A_i(t) * ∂μ_i(t)/∂φ
        """
        n_seasons = self.packer.n_seasons
        n_regions = self.packer.n_regions
        M = n_seasons * n_regions

        # Initialize gradient vector: [S_init (M), I_init (M), theta (1)]
        grad = np.zeros(2 * M + 1)

        
        mu = sim_df["incidence"].values + 1e-6  # small constant for numerical stability

        # Residual: r = Y - ρμ
        r = self.obs["incidence"].values - mu * self.disease.rho

        # A_i(t) = ∂ℓ/∂μ = -1/(2μ) + r/((1-ρ)μ) + r²/(2ρ(1-ρ)μ²)
        A = -1 / (2 * mu) + r / ((1 - self.disease.rho) * mu) + r ** 2 / (2 * self.disease.rho * (1 - self.disease.rho) * mu ** 2)

        # ∂L/∂μ = -A (negative log-likelihood)
        dL_dmu = -A

        # Now accumulate gradients for each parameter
        # theta gradient (last element)
        grad[-1] = np.sum(dL_dmu * sim_df['theta'].values)

        # S_init and I_init gradients (per season)
        for season_idx in range(n_seasons):
            mask = sim_df['season_idx'].values == season_idx

            # S1_0, S2_0 -> indices in grad: season_idx * n_regions + region
            grad[season_idx * n_regions + 0] = np.sum(dL_dmu[mask] * sim_df['S1_0'].values[mask])
            grad[season_idx * n_regions + 1] = np.sum(dL_dmu[mask] * sim_df['S2_0'].values[mask])

            # I1_0, I2_0 -> indices in grad: M + season_idx * n_regions + region
            grad[M + season_idx * n_regions + 0] = np.sum(dL_dmu[mask] * sim_df['I1_0'].values[mask])
            grad[M + season_idx * n_regions + 1] = np.sum(dL_dmu[mask] * sim_df['I2_0'].values[mask])

        return grad

    
    def __call__(self, x, grad=None):
        """ x is a vector of parameters"""
        assert not np.isnan(x).any()

        params = self.packer.unpack(x)
        simulated = self.packer.sim(params=params, phase=self.phase, disease=self.disease)

        msg = f"Simulation and observation indices don't match. Sim: {len(simulated)}, Obs: {len(self.obs)}"
        assert np.all(simulated.index == self.obs.index), msg

        # Create mask for valid observations (not NaN in observed incidence)
        valid_mask = ~self.obs['incidence'].isna()
        sim_valid = simulated[valid_mask].copy()
        obs_valid = self.obs[valid_mask].copy()

        # Reweight to account for missing observations
        n_total = len(self.obs)
        n_valid = valid_mask.sum()
        weight = n_total / n_valid if n_valid > 0 else 1.0

        out = self.loss(obs_valid, sim_valid, rho=self.disease.rho) * weight
        assert not np.isnan(out), f"Loss is {out}"

        # Compute gradient if requested
        if grad is not None and grad.size > 0:
            computed_grad = self.compute_gradient(sim_valid) * weight
            grad[:] = computed_grad

        self.x_list.append(copy.deepcopy(x))
        self.out_list.append(out)
        return out


class Inverter:
    def __init__(self,
                 optimizer,
                 phase,
                 obs,
                 disease):

        self.objective = Objective(obs=obs,
                                   phase=phase,
                                   disease=disease)
        
        self.packer = self.objective.packer
        self.optimizer = optimizer
    
        
    def fit(self, n0=1, maxeval=None, n_jobs=-1):
        self.objective.x_list = []
        self.objective.out_list = []

        starts = []
        for i in range(n0):
            x0 = self.packer.random_vector()
            starts.append(x0)
            assert np.all(x0 <= 1)
            assert np.all(x0 >= 0)

        if n0 > 1 and n_jobs != 1:
            it = tqdm(starts)
            self.results = Parallel(n_jobs=n_jobs)(delayed(self.single_optimization)(x, maxeval) for x in it)
        else:
            # Sequential execution - avoids pickle issues with nlopt
            self.results = []
            for x0 in tqdm(starts):
                result = self.single_optimization(x0, maxeval)
                self.results.append(result)

        # Print any errors from failed runs
        errors = [res['err'] for res in self.results if res['err']]
        if errors:
            print(f"Errors ({len(errors)}):")
            for err in errors:
                print(f"  {err}")

        print(f"successes rate {sum(int(res['success']) for res in self.results)} / {len(self.results)}")
        best = min(self.results, key=lambda r: r['fun'])
        self.x = best['x']
        self.success = best['success']
        self.fun = best['fun']
   
        return self


    def single_optimization(self, x0, maxeval=None):
        objective = copy.deepcopy(self.objective)

        n_regions = self.packer.n_regions
        n_seasons = self.packer.n_seasons
        M = n_regions * n_seasons
        n = self.packer.n_params
        opt = nlopt.opt(self.optimizer, n)

        opt.set_xtol_rel(1e-9)
        # opt.set_xtol_abs(1e-6)
        #opt.set_ftol_rel(1e-6) 
        if maxeval is not None:
            opt.set_maxeval(maxeval)

        opt.set_min_objective(objective)
        # S_init and I_init bounded by [0, 1], theta bounded by [0, 0.5]
        opt.set_lower_bounds([0.]*n)
        opt.set_upper_bounds([1.]*(n-1) + [0.5])  # theta (last param) bounded by 0.5

        # Add simplex constraints: S[i] + I[i] <= 1 for each region-season combination
        # nlopt inequality constraint h(x) >= 0, so for S + I <= 1, we need 1 - S - I >= 0
        # For gradient-based methods, constraint function must also return gradient
        def make_constraint(idx):
            def constraint(x, grad):
                if grad.size > 0:
                    grad[:] = 0
                    grad[idx] = 1
                    grad[idx + M] = 1
                return x[idx] + x[idx + M] - 1
            return constraint

        for idx in range(M):
            opt.add_inequality_constraint(make_constraint(idx), 1e-8)

        x0 = self.packer.random_vector()
        try:
            x = opt.optimize(x0)
        except Exception as e:
            # Return failed result with objective at x0
            fun_x0 = objective(x0)
            return dict(x=x0, fun=fun_x0, success=False, x_list=[x0], out_list=[fun_x0], err=str(e))

        params = dict(x=x, fun=opt.last_optimum_value(), success=opt.last_optimize_result() == 4, err='')
        params['x_list'] = copy.deepcopy(objective.x_list)
        params['out_list'] = copy.deepcopy(objective.out_list)
        return params

