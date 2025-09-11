import pdb
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import matplotlib

from src.paralllel_inverter import runner

matplotlib.use("MacOSX")

from src.multi import run
from src.rk import run_rk
from src.packer import Packer
from src.losses import LOSSES


class Inverter:
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
        self.packer = Packer(seasons=sorted(population.season.unique()),
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

        # self.params = dict(n_weeks=self.n_weeks,
        #                    sigma=self.sigma,
        #                    dt_output=self.dt_output,
        #                    dt_step=self.dt_step,
        #                    mu=self.mu,
        #                    nu=self.nu)

        assert population.shape == (self.packer.n_seasons * self.packer.n_regions, 3)
        self.loss = LOSSES[loss]
        self.run_time = 0

    def sim(self,
            x=None):
        if x is None:
            x = self.packer.random_vector()
        xx = self.packer.unpack(x)
        xx = self.packer.real2pop(xx)
        self.packer.verify_params(xx)
        S_init = xx.pop('S_init')
        E_init = xx.pop('E_init')
        I_init = xx.pop('I_init')
        beta0 = xx['beta0']
        omega = xx['omega']
        eps = xx['eps']
        c_mat = self.packer.c_vec_to_mat(xx.pop("c_vec"))

        results = []
        for season_idx, season in enumerate(self.packer.seasons):
            pop = self.pops[season]
            start = time.time()
            if self.integration == 'euler':
                df = run(S_init=S_init[season_idx, :], E_init=E_init[season_idx, :], I_init=I_init[season_idx, :],
                         n_weeks=self.n_weeks, beta0=beta0, sigma=self.sigma, dt_output=self.dt_output, dt_step=1e-2,
                         mu=self.mu, nu=self.nu, omega=omega, eps=eps, contact_matrix=c_mat, population=pop,
                         start_date=season)
            elif self.integration == 'rk':
                df = run_rk(S_init=S_init[season_idx, :], E_init=E_init[season_idx, :], I_init=I_init[season_idx, :],
                            dt_step=1, dt_output=self.dt_output, n_weeks=self.n_weeks, beta0=beta0, sigma=self.sigma, mu=self.mu, nu=self.nu,
                            omega=omega, eps=eps, contact_matrix=c_mat, population=pop, start_date=season)
            else:
                raise ValueError(f"Unknown integration method: {self.integration}. Use 'euler' or 'rk'")
            self.run_time += time.time() - start

            letter = "C"  ## If at any point wed like to look at infecteds instead
            df = df[[col for col in df.columns if letter in col]].reset_index(drop=False)
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
            seed=None):
        def objective(x):
            assert not np.isnan(x).any()
            self.packer.verify_vector(x)
            kk = self.sim(x)
            assert np.all(kk.index == obs.index)
            assert np.all(kk.isnull() == obs.isnull())
            #df.sort_values(["season", "region", "time"]).set_index(["season", "region", "time"])
            out = self.loss(obs.dropna(), kk.dropna())
            assert not np.isnan(out)
            return out

        x0 = self.packer.random_vector(seed) if x0 is None else x0
        res = minimize(objective, x0, method=method)

        self.params = self.packer.real2pop(self.packer.unpack(res.x))
        self.fun = res.fun
        return self

