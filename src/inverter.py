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

from src.losses import LOSSES
from src.packer import Packer
from src import compute_g

matplotlib.use("Agg")  # Use non-interactive backend for headless environments


class Objective:  
    def __init__(self,
                 population,
                 beta0,
                 gamma,
                 phase,
                 amplitude,
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
        self.loss = LOSSES['gaussian']

        msg =  f"Population shape {population.shape} != "
        msg += f" ({self.packer.n_seasons * self.packer.n_regions}, 3)"
        assert population.shape == (self.packer.n_seasons * self.packer.n_regions, 3), msg



    def sim(self, params):
        S_init = params['S_init']
        I_init = params['I_init']
        theta = params["theta"]

        results = []
        for season_idx, season in enumerate(self.packer.seasons):
            pop = self.pops[season]
            S = S_init[season_idx, :]
            I = I_init[season_idx, :]

            df = compute_g.slow(S0=S,
                                I0=I,
                                gamma=self.gamma,  # Recovery rate
                                theta=theta,
                                T=self.n_weeks,
                                beta0=self.beta0,
                                amplitude=self.amplitude,
                                period=53,  # 53 weeks per year
                                phase=self.phase[0],
                                phase2=self.phase[1])  # Same phase for both regions
            
            # compute_G returns DataFrame with MultiIndex (t, j) and column 'mu' for incidence
            df = df.reset_index()
            df['time'] = df['t'] * timedelta(weeks=1) # Convert week index to time
            df_long = df[['time', 'j', 'mu']].rename(columns={'j': 'region', 'mu': 'incidence'})
            df_long["region"] = df_long["region"].astype(int).replace(self.packer.region_dict)
            df_long["season"] = season
            assert np.all(df_long['incidence'] >= 0), f"Negative incidence values in simulation output"
            results.append(df_long)

        res = pd.concat(results, ignore_index=True)
        return res

    
    def __call__(self, xx, grad=None):
        assert not np.isnan(xx).any()
        
        params = self.packer.unpack(xx)
        kk = self.sim(params)
            
        msg = f"Simulation and observation indices don't match. Sim: {len(kk)}, Obs: {len(self.obs)}"
        assert np.all(kk.index == self.obs.index), msg

        msg = f"Nulls dont match. Sim: {kk.isnull().sum().sum()}, Obs: {self.obs.isnull().sum().sum()}"
        assert np.all(kk.isnull() == self.obs.isnull()), msg 

        out = self.loss(self.obs.dropna(), kk.dropna())
        assert not np.isnan(out), f"Loss is {out}"

        self.x_list.append(copy.deepcopy(xx))
        self.out_list.append(out)
        return out



class Inverter:
    def __init__(self,
                 objective):
                 
        self.objective = objective
        self.packer = objective.packer

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
            
        opt = nlopt.opt(nlopt.LN_COBYLA, n)
        opt.set_xtol_rel(1e-4)
        if maxeval is not None:
            opt.set_maxeval(maxeval)
                
        opt.set_min_objective(objective)
        # S_init and I_init bounded by [0, 1], theta bounded by [0, 0.5]
        opt.set_lower_bounds([0]*n)
        opt.set_upper_bounds([1]*(n-1) + [0.5])  # theta (last param) bounded by 0.5
            
        # Add simplex constraints: S[i] + I[i] <= 1 for each region-season combination
        # nlopt inequality constraint h(x) >= 0, so for S + I <= 1, we need 1 - S - I >= 0
        # Must use default argument (idx=idx) to capture value, not reference
        for idx in range(M):
            opt.add_inequality_constraint(lambda x, grad: x[idx] + x[idx + M] - 1, 1e-8)

        x0 = self.packer.random_vector()
        x = opt.optimize(x0)
        params = dict(x=x, fun=opt.last_optimum_value(), success=opt.last_optimize_result() == 4)
        params['x_list'] = copy.deepcopy(self.objective.x_list)
        params['out_list'] = copy.deepcopy(self.objective.out_list)
        return params

