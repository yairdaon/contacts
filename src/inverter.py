import copy
import matplotlib
import nlopt
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from scipy.optimize import minimize
from tqdm import tqdm

from src.losses import LOSSES
from src.multi import run_euler
from src.packer import Trans, Straight
from src.rk import run_rk

matplotlib.use("Agg")  # Use non-interactive backend for headless environments


class Objective:
    """
    Objective function for SEIR parameter inference.
    
    Encapsulates the forward simulation model, parameter packing/unpacking,
    and loss computation for optimization-based parameter estimation.
    """
    
    def __init__(self,
                 population,
                 transform,
                 method='rk',
                 loss='gaussian',
                 n_weeks=26,
                 sigma=0.5,
                 dt_output=7,
                 mu=0 / (30 * 365),
                 nu=0.2,
                 seasonal_driver=True):
        """
        Initialize objective function for parameter inference.
        
        Parameters:
        -----------
        population : DataFrame
            Population data with columns [region, season, population]
        transform : bool
            Whether to use parameter transformations (Trans vs Straight packer)
        method : str
            Integration method ('rk' or 'euler')
        loss : str
            Loss function name (see losses.py)
        n_weeks : int
            Simulation duration in weeks
        sigma : float
            E→I transition rate (1/day)
        dt_output : float
            Output timestep (days)
        mu : float
            Birth/death rate (1/day)
        nu : float
            I→R recovery rate (1/day)
        seasonal_driver : bool
            Whether to include seasonal forcing (default: True)
        """
        pack = Trans if transform else Straight
        self.packer = pack(seasons=population['season'].unique(),
                           regions=population['region'].unique(),
                           seasonal_driver=seasonal_driver)
        self.n_weeks = n_weeks
        self.sigma = sigma
        self.dt_output = dt_output
        self.mu = mu
        self.nu = nu
        self.pops = {}
        for season in self.packer.seasons:
            pop = (population
                   .query("season == @season")
                   .set_index("region")
                   .loc[self.packer.regions, "population"]
                   .values)
            self.pops[season] = pop
        self.population = population

        assert population.shape == (self.packer.n_seasons * self.packer.n_regions, 3), f"Population DataFrame shape {population.shape} doesn't match expected ({self.packer.n_seasons * self.packer.n_regions}, 3)"
        self.loss = LOSSES[loss]
        self.run_time = 0

        self.method = method
        if self.method == 'euler':
            self.run = run_euler
            self.dt_step = 0.05
        elif self.method == 'rk':
            self.run = run_rk
            self.dt_step = 1
        else:
            raise ValueError(f"Method {self.method} not implemented")

        
    def reset(self):
        """Reset optimization tracking lists."""
        self.x_list = []
        self.out_list = []

        
    def set_packer(self, packer):
        """Set parameter packer (Trans or Straight)."""
        self.packer = packer(seasons=sorted(self.population.season.unique()),
                             regions=sorted(self.population.region.unique()))

    def sim(self, params):
        """
        Run forward simulation with given parameters.
        
        Parameters:
        -----------
        params : dict
            Unpacked parameter dictionary from packer
            
        Returns:
        --------
        DataFrame
            Simulated incidence data with columns [time, region, season, incidence]
        """
        self.packer.verify(params)

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
            S = S_init[season_idx, :]
            E = E_init[season_idx, :]
            I = I_init[season_idx, :]

            df = self.run(S_init=S,
                        E_init=E,
                        I_init=I,
                        dt_step=self.dt_step,
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

            df = df[[col for col in df.columns if "C" in col]].reset_index(drop=False)
            assert np.all(df.drop("time", axis=1) >= 0), f"Negative values in simulation output: {df[df < 0].dropna(how='all')}"
            df_long = df.melt(id_vars=["time"], var_name="region", value_name="incidence")
            df_long["region"] = (df_long["region"]
                                 .str.replace("C", "")
                                 .astype(int)
                                 .replace(self.packer.region_dict)
                                 )
            df_long["season"] = season
            results.append(df_long)

        res = pd.concat(results, ignore_index=True)

        return res

    def __call__(self, xx, grad=None):
        """
        Evaluate objective function for optimization.
        
        Parameters:
        -----------
        xx : array
            Packed parameter vector
        grad : array, optional
            Gradient (not used, for nlopt compatibility)
            
        Returns:
        --------
        float
            Loss value (np.inf if simulation fails)
        """
        try:
            assert not np.isnan(xx).any(), f"NaN values found in optimization vector: {xx[np.isnan(xx)]}"

            # Unpack and transform parameters
            params = self.packer.unpack(xx)

            # Run simulation with parameter dictionary
            kk = self.sim(params)
            assert np.all(
                kk.index == self.obs.index), f"Simulation and observation indices don't match. Sim: {len(kk)}, Obs: {len(self.obs)}"
            assert np.all(
                kk.isnull() == self.obs.isnull()), f"Nulls dont match. Sim: {kk.isnull().sum().sum()}, Obs: {self.obs.isnull().sum().sum()}"
            out = self.loss(self.obs.dropna(), kk.dropna())  # Fixed rho, params.pop('rho'))  # , theta=params.pop('theta'))
            assert not np.isnan(out), f"Loss function returned NaN. Loss value: {out}"
        except Exception as e:
            #print(e)
            out = np.inf
        self.x_list.append(copy.deepcopy(xx))
        self.out_list.append(out)
        return out



class Inverter:
    """
    Parameter inference engine for SEIR models.
    
    Provides optimization-based parameter estimation with support for
    multiple optimizers (scipy, nlopt) and constraint handling.
    """
    
    def __init__(self,
                 objective,
                 optimizer='nlopt',
                 auglag=False):
        """
        Initialize parameter inference engine.
        
        Parameters:
        -----------
        objective : Objective
            Objective function instance
        optimizer : str
            Optimizer type ('scipy' or 'nlopt')
        auglag : bool
            Whether to use augmented Lagrangian method (nlopt only)
        """
        self.objective = objective
        self.packer = objective.packer
        self.optimizer = optimizer
        if self.optimizer == 'scipy':
            assert type(self.objective.packer) == Trans
        elif self.optimizer == 'nlopt':
            assert type(self.objective.packer) == Straight
        self.auglag = auglag

    def fit(self, seed=None, n0=1, maxeval=None):
        """
        Fit SEIR model parameters to observational data.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        n0 : int
            Number of optimization restarts
        maxeval : int, optional
            Maximum function evaluations
            
        Returns:
        --------
        self : Inverter
            Fitted inverter with optimal parameters in self.x
        """
        self.objective.reset()
        np.random.seed(seed)

        starts = []
        for i in range(n0):
            local_seed = seed + i if seed is not None else None
            x0 = self.packer.random_vector(seed=local_seed)
            starts.append(x0)
            if self.optimizer == 'nlopt':
                assert np.all(x0 <= 1)
                assert np.all(x0 >= 0)

        if n0 > 1:
            self.results = Parallel(n_jobs=-1)(delayed(self.single_optimization)(x, maxeval) for x in tqdm(starts))
        else:
            self.results = [self.single_optimization(starts[0], maxeval)]

        print(f"successes rate {sum(int(res['success']) for res in self.results)} / {len(self.results)}")
        best = min(self.results, key=lambda r: r['fun'])
        self.x = best['x']
        self.success = best['success']
        self.fun = best['fun']
   
        return self


    def single_optimization(self, x0, maxeval=None):
        """
        Perform single optimization run from given starting point.
        
        Parameters:
        -----------
        x0 : array
            Initial parameter vector
        maxeval : int, optional
            Maximum function evaluations
            
        Returns:
        --------
        dict
            Optimization result with keys: x, fun, success, x_list, out_list
        """
        objective = copy.deepcopy(self.objective)
        if self.optimizer == 'scipy':
            best = minimize(objective, x0=x0, method="SLSQP")#L-BFGS-B")
            params = dict(x=best.x, fun=best.fun, success=best.success)

        elif self.optimizer == 'nlopt':
            n_regions = self.packer.n_regions
            n_seasons = self .packer.n_seasons
            M = n_regions * n_seasons
            n = self.packer.n_params
            

            if self.auglag:
                opt = nlopt.opt(nlopt.AUGLAG, n)
                local_opt = nlopt.opt(nlopt.LD_SLSQP, n)  # or LD_SLSQP, LN_BOBYQA, LN_NEWUOA
                local_opt.set_xtol_rel(1e-6)
                local_opt.set_ftol_rel(1e-8)
                if maxeval is not None:
                    local_opt.set_maxeval(maxeval // 10)  # Limit evaluations per sub-problem
                opt.set_local_optimizer(local_opt)
                # Set tolerances for the outer augmented Lagrangian loop
                opt.set_xtol_rel(1e-4)
                opt.set_ftol_rel(1e-6)
                if maxeval is not None:
                    opt.set_maxeval(maxeval)
            else:
                opt = nlopt.opt(nlopt.LN_COBYLA, n)
                opt.set_xtol_rel(1e-4)
                if maxeval is not None:
                    opt.set_maxeval(maxeval)

            opt.set_min_objective(objective)
            opt.set_lower_bounds([0]*(n-1) + [-float('inf')])
            opt.set_upper_bounds([1]*(n-1) + [float('inf')])
            
            # Add your simplex constraints
            for idx in range(M):
                lower = lambda x, grad: -x[idx] - x[idx + M] #- x[idx + 2 * M]
                upper = lambda x, grad: -1 + x[idx] + x[idx + M] #+ x[idx + 2 * M]
                opt.add_inequality_constraint(lower, 1e-8)
                opt.add_inequality_constraint(upper, 1e-8)
                
            x0 = self.packer.random_vector()
            x = opt.optimize(x0)
            params = dict(x=x, fun=opt.last_optimum_value(), success=opt.last_optimize_result() == 4)
        params['x_list'] = copy.deepcopy(objective.x_list)
        params['out_list'] = copy.deepcopy(objective.out_list)
        return params
