import pdb
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import matplotlib

matplotlib.use("MacOSX")

from multi import run
from packer import Packer
from losses import LOSSES


class Inverter:
    def __init__(self,
                 population,
                 n_weeks=26,
                 sigma=0.5,
                 dt_output=7,
                 dt_euler=1e-2,
                 mu=0 / (30 * 365),
                 nu=0.2,
                 loss='lsq'):
        """population is a dataframe with columns [region, season, population]"""
        self.packer = Packer(seasons=sorted(population.season.unique()),
                             regions=sorted(population.region.unique()))

        self.n_weeks = n_weeks
        self.sigma = sigma
        self.dt_output = dt_output
        self.dt_euler = dt_euler
        self.mu = mu
        self.nu = nu
        self.population = population

        assert population.shape == (self.packer.n_seasons * self.packer.n_regions, 3)
        self.loss = LOSSES[loss]
        self.run_time = 0

    def sim(self,
            x):
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

        start = time.time()
        results = []
        for season_idx, season in enumerate(self.packer.seasons):
            print(season_idx, end=' ')
            pop = (self.population
                   .query("season == @season")
                   .set_index("region")
                   .loc[self.packer.regions, "population"]
                   .values)

            df = run(
                S_init=S_init[season_idx, :],  # Now using fractions
                E_init=E_init[season_idx, :],
                I_init=I_init[season_idx, :],
                n_weeks=self.n_weeks,
                beta0=beta0,
                sigma=self.sigma,
                dt_output=self.dt_output,
                dt_euler=self.dt_euler,
                mu=self.mu,
                nu=self.nu,
                omega=omega,
                eps=eps,
                contact_matrix=c_mat,
                population=pop,
                start_date=season
            )
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
        end = time.time()
        self.run_time += end - start
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

