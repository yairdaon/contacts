import pdb
import time

import numpy as np
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.stats import nbinom
from joblib import Parallel, delayed

from multi import run
from packer import Packer

def least_squares_loss(observed, simulated):
    """Least squares on log-incidence."""
    obs = np.log1p(observed["incidence"].values)
    sim = np.log1p(simulated["incidence"].values)
    return np.sum((obs - sim) ** 2)

def negbinom_loss(observed, simulated, theta=10.0):
    """Negative binomial log-likelihood (return negative for minimization)."""
    mu = simulated["incidence"].values + 1e-6
    k = theta
    loglik = nbinom.logpmf(observed["incidence"].values, n=k, p=k / (k + mu))
    return -np.sum(loglik)

def gaussian_loss(observed, simulated, sigma=1.0):
    """Gaussian likelihood (return negative log-likelihood)."""
    obs = observed["incidence"].values
    sim = simulated["incidence"].values
    return np.sum((obs - sim) ** 2) / (2 * sigma ** 2)

LOSSES = {
            "lsq": least_squares_loss,
            "negbinom": negbinom_loss,
            "gaussian": gaussian_loss,
        }

def runner(season_idx, season, pop, params, xx,
           S_init, E_init, I_init):
    df = run(
        S_init=S_init[season_idx, :] * pop,
        E_init=E_init[season_idx, :] * pop,
        I_init=I_init[season_idx, :] * pop,
        **params,
        **xx,
        # n_weeks=self.n_weeks,
        # beta0=beta0,
        # sigma=self.sigma,
        # dt_output=self.dt_output,
        # dt_euler=self.dt_euler,
        # mu=self.mu,
        # nu=self.nu,
        # omega=omega,
        # eps=eps,
        # n_regions=self.packer.n_regions),
        # contact_matrix=c_mat,
        population=pop,
        start_date=season
    )
    df = df[[col for col in df.columns if "C" in col]].reset_index(drop=False)#f"C{i}" for i in range(self.packer.n_regions)]].reset_index(drop=False)
    df_long = df.melt(id_vars=["time"], var_name="region", value_name="incidence")
    df_long["region"] = df_long["region"].str.replace("C", "").astype(int)
    df_long["season"] = season
    return df_long


class Inverter:
    def __init__(self,
                 population,
                 n_weeks=26,
                 sigma=0.5,
                 dt_output=7,
                 dt_euler=5e-2,
                 mu=0/(30*365),
                 nu=0.2,
                 loss='lsq',
                 parallelize=False):

        """population is a dataframe with columns [region, season, population]"""
        self.packer = Packer(seasons=sorted(population.season.unique()),
                             regions=sorted(population.region.unique()))
        self.params=dict(
            n_weeks=n_weeks,
            sigma=sigma,
            dt_output=dt_output,
            dt_euler=dt_euler,
            mu=mu,
            nu=nu,
            n_regions=self.packer.n_regions
        )
        self.parallelize = parallelize
        # self.n_weeks = n_weeks
        # self.sigma = sigma
        # self.dt_output = dt_output
        # self.dt_euler = dt_euler
        # self.mu = mu
        # self.nu = nu
        self.population = population

        assert population.shape == (self.packer.n_seasons * self.packer.n_regions, 3)
        self.loss = LOSSES[loss]
        self.run_time = 0

    def makepop(self, season):
        pop = (self.population
               .query("season == @season")
               .set_index("region")
               .loc[self.packer.regions, "population"]
               .values)
        return pop
                                         
    def sim(self,
            x):

        xx = self.packer.unpack(x)
        xx = self.packer.real2pop(xx)
        self.packer.verify(xx)
        S_init = xx.pop('S_init')
        E_init = xx.pop('E_init')
        I_init = xx.pop('I_init')
        # beta0 = xx['beta0']
        # omega = xx['omega']
        # eps = xx['eps']
        xx['contact_matrix'] = self.packer.c_vec_to_mat(xx.pop("c_vec"))

        if self.parallelize:
            start = time.time()
            results = Parallel(n_jobs=-1)(delayed(runner)(
                season_idx=season_idx,
                season=season,
                pop=self.makepop(season),
                S_init=S_init,
                E_init=E_init,
                I_init=I_init,
                params=self.params,
                xx=xx) for season_idx, season in enumerate(self.packer.seasons))
            end = time.time()
            self.run_time = self.run_time + end - start

        else:
            start = time.time()
            results = []
            for season_idx, season in enumerate(self.packer.seasons):
                df_long = runner(season_idx=season_idx,
                                 season=season,
                                 pop=self.makepop(season),
                                 S_init=S_init,
                                 E_init=E_init,
                                 I_init=I_init,
                                 params=self.params,
                                 xx=xx)
                results.append(df_long)
            end = time.time()
            self.run_time = self.run_time + end - start
        res = pd.concat(results, ignore_index=True)
        res["region"] = res.region.replace(self.packer.region_dict)

        res = self.align(res)
        return res


    def align(self, df):
        return df.sort_values(["season", "region", "time"]).set_index(["season", "region", "time"])
    
    def fit(self,
            obs,
            method="L-BFGS-B",
            x0=None,
            seed=None):
        
        def objective(x):
            kk = self.sim(x)
            assert np.all(kk.index == obs.index)
            assert np.all(np.isnan(kk.values) == np.isnan(obs.values))
            out = self.loss(obs.dropna(), kk.dropna())
            assert not np.isnan(out)
            return out
        
        if x0 is None:
            x0 = self.packer.random_vector(seed)
        res = minimize(objective, x0, method=method)

        self.params = self.packer.real2pop(self.packer.unpack(res.x))
        self.fun = res.fun
        desc = "parallel" if self.parallelize else "serial"
        print(f"Run time {self.run_time/60:.3f} {desc}")
        return self
    

def main():
    start = time.time()
    seasons = [f"{year}-01-01" for year in range(1990, 1992)]
    regions = [f'HHS{i+1}' for i in range(3)]
    population = [{"season": s, "region": r, "population": np.random.randint(10**6)} for r, s in product(regions, seasons)]
    population = pd.DataFrame(population)
    
    inv = Inverter(population=population, n_weeks=4)
    x0 = inv.packer.random_vector()
    dd = inv.sim(x0)
    inv.fit(dd)
    end = time.time()
    print("Total:", (end-start)/60)

    
if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, pdb
        traceback.print_exc()  # Prints the full stack trace to stderr
        pdb.post_mortem()      # Starts debugger at the poi

