import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from tqdm import tqdm

from src.losses import LOSSES
from src.packer import Packer
from src.rk import run_rk
from src.multi import run_euler

matplotlib.use("MacOSX")

RUNNER = run_rk
DT = 1
class Inverter:
    def __init__(self,
                 population,
                 loss='gaussian',
                 n_weeks=26,
                 sigma=0.5,
                 dt_output=7,
                 mu=0 / (30 * 365),
                 nu=0.2
                 ):

        """population is a dataframe with columns [region, season, population]"""
        self.packer = Packer(seasons=sorted(population.season.unique()),
                             regions=sorted(population.region.unique()))

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


        assert population.shape == (self.packer.n_seasons * self.packer.n_regions, 3), f"Population DataFrame shape {population.shape} doesn't match expected ({self.packer.n_seasons * self.packer.n_regions}, 3)"
        self.loss = LOSSES[loss]
        self.run_time = 0

    def sim(self, params):
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

            df = RUNNER(S_init=S,
                        E_init=E,
                        I_init=I,
                        dt_step=DT,
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

    def fit(self,
            obs,
            x0=None,
            seed=None,
            n0=1):

        np.random.seed(seed)

        def objective(x):
            assert not np.isnan(x).any(), f"NaN values found in optimization vector: {x[np.isnan(x)]}"

            # Unpack and transform parameters
            params = self.packer.unpack(x)

            # Run simulation with parameter dictionary
            kk = self.sim(params)
            assert np.all(kk.index == obs.index), f"Simulation and observation indices don't match. Sim: {len(kk)}, Obs: {len(obs)}"
            assert np.all(kk.isnull() == obs.isnull()), f"Nulls dont match. Sim: {kk.isnull().sum().sum()}, Obs: {obs.isnull().sum().sum()}"
            out = self.loss(obs.dropna(), kk.dropna(), rho=params.pop('rho'))#, theta=params.pop('theta'))
            assert not np.isnan(out), f"Loss function returned NaN. Loss value: {out}"
            return out

        starts = []
        for i in range(n0):
            local_seed = seed + i if seed is not None else None
            starts.append(self.packer.random_vector(seed=local_seed))
        if x0 is not None:
            starts.append(x0)

        if n0 > 1:
            results = Parallel(n_jobs=-1)(delayed(minimize)(objective, x0=x, method="L-BFGS-B") for x in tqdm(starts))
            print(f"successes rate {sum(int(res.success) for res in results)}  / {len(results)}")
            best = min(results, key=lambda r: r.fun)
        else:
            best = minimize(objective, x0=x0, method="L-BFGS-B")

        self.params = self.packer.unpack(best.x)
        self.fun = best.fun
        return self

