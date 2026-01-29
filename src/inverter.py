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
from src import flu


class Objective:
    def __init__(self,
                 population,
                 beta0,
                 gamma,
                 phase,
                 amplitude,
                 rho,
                 n_weeks=26,
                 seasonal_driver=True):
        self.packer = Packer(seasons=population['season'].unique(),
                             regions=population['region'].unique(),
                             seasonal_driver=seasonal_driver)
        self.n_weeks = n_weeks
        self.gamma = gamma
        self.beta0 = beta0
        self.phase = phase
        self.amplitude = amplitude
      
        self.pops = {}
        for season in self.packer.seasons:
            pop = (population
                   .query("season == @season")
                   .set_index("region")
                   .loc[self.packer.regions, "population"]
                   .values)
            self.pops[season] = pop
        self.population = population
        self.loss = losses.gaussian
        self.rho = rho

        msg =  f"Population shape {population.shape} != "
        msg += f" ({self.packer.n_seasons * self.packer.n_regions}, 3)"
        assert population.shape == (self.packer.n_seasons * self.packer.n_regions, 3), msg



    def sim(self, params):
        """Simulate and return incidence + Jacobian columns for gradient computation."""
        S_init = params['S_init']
        I_init = params['I_init']
        theta = params["theta"]

        results = []
        for season_idx, season in enumerate(self.packer.seasons):
            S = S_init[season_idx, :]
            I = I_init[season_idx, :]

            df = compute_g.slow(S0=S,
                                I0=I,
                                gamma=self.gamma,
                                theta=theta,
                                T=self.n_weeks,
                                beta0=self.beta0,
                                amplitude=self.amplitude,
                                period=53,
                                phase=self.phase[0],
                                phase2=self.phase[1])

            df = df.reset_index()
            df['time'] = df['t'] * timedelta(weeks=1)
            df['season_idx'] = season_idx
            df_long = df[['time', 'j', 'mu', 'season_idx', 'theta', 'S1_0', 'I1_0', 'S2_0', 'I2_0']].rename(
                columns={'j': 'region', 'mu': 'incidence'})
            df_long["region"] = df_long["region"].astype(int).replace(self.packer.region_dict)
            df_long["season"] = season
            assert np.all(df_long['incidence'] >= 0), f"Negative incidence values in simulation output"
            results.append(df_long)

        res = pd.concat(results, ignore_index=True)
        return res

    def compute_gradient(self, sim_df, obs_df):
        """
        Compute gradient of negative log-likelihood with respect to parameters.

        From the paper, the score function (derivative of log-likelihood) is:
            ∂ℓ/∂φ = Σ A_i(t) * ∂μ_i(t)/∂φ

        where:
            A_i(t) = -1/(2μ) + r/((1-ρ)μ) + r²/(2ρ(1-ρ)μ²)

        For minimization of negative log-likelihood L = -ℓ:
            ∂L/∂φ = -Σ A_i(t) * ∂μ_i(t)/∂φ
        """
        rho = self.rho
        n_seasons = self.packer.n_seasons
        n_regions = self.packer.n_regions
        M = n_seasons * n_regions

        # Initialize gradient vector: [S_init (M), I_init (M), theta (1)]
        grad = np.zeros(2 * M + 1)

        obs = obs_df["incidence"].values
        mu = sim_df["incidence"].values + 1e-6  # small constant for numerical stability

        # Residual: r = Y - ρμ
        r = obs - mu * rho

        # A_i(t) = ∂ℓ/∂μ = -1/(2μ) + r/((1-ρ)μ) + r²/(2ρ(1-ρ)μ²)
        A = -1 / (2 * mu) + r / ((1 - rho) * mu) + r ** 2 / (2 * rho * (1 - rho) * mu ** 2)

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

    def __call__(self, xx, grad=None):
        assert not np.isnan(xx).any()

        params = self.packer.unpack(xx)
        simulated = self.sim(params)

        msg = f"Simulation and observation indices don't match. Sim: {len(simulated)}, Obs: {len(self.obs)}"
        assert np.all(simulated.index == self.obs.index), msg

        msg = f"Nulls dont match. Sim: {simulated.isnull().sum().sum()}, Obs: {self.obs.isnull().sum().sum()}"
        assert np.all(simulated.isnull() == self.obs.isnull()), msg

        out = self.loss(self.obs.dropna(), simulated.dropna())
        assert not np.isnan(out), f"Loss is {out}"

        # Compute gradient if requested
        if grad is not None and grad.size > 0:
            computed_grad = self.compute_gradient(simulated.dropna(), self.obs.dropna())
            grad[:] = computed_grad

        self.x_list.append(copy.deepcopy(xx))
        self.out_list.append(out)
        return out



class Inverter:
    def __init__(self,
                 objective,
                 optimizer):
        
        self.objective = objective
        self.packer = objective.packer
        self.optimizer = optimizer

        
    def fit(self, n0=1, maxeval=None):
        self.objective.x_list = []
        self.objective.out_list = []
       
        starts = []
        for i in range(n0):
            x0 = self.packer.random_vector()
            starts.append(x0)
            assert np.all(x0 <= 1)
            assert np.all(x0 >= 0)

        if n0 > 1:
            it = tqdm(starts)
            self.results = Parallel(n_jobs=-1)(delayed(self.single_optimization)(x, maxeval) for x in it) 
        else:
            self.results = [self.single_optimization(starts[0], maxeval)]

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
        x = opt.optimize(x0)
        params = dict(x=x, fun=opt.last_optimum_value(), success=opt.last_optimize_result() == 4)
        params['x_list'] = copy.deepcopy(objective.x_list)
        params['out_list'] = copy.deepcopy(objective.out_list)
        return params

