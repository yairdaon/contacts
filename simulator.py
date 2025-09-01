import numpy as np
import pandas as pd
from itertools import product

from multi import run
from packer import Packer


class Simulator:
    def __init__(self,
                 population,
                 n_weeks=26,
                 sigma=0.5,
                 dt_output=7,
                 dt_euler=5e-2,
                 mu=0/(30*365),
                 nu=0.2):

        """population is a dataframe with columns [region, season, population]"""
        
        self.n_weeks = n_weeks
        self.sigma = sigma
        self.dt_output = dt_output
        self.dt_euler = dt_euler
        self.mu = mu
        self.nu = nu
        self.population = population
        self.packer = Packer(seasons = population.season.value_counts().index,
                             regions = population.region.value_counts().index) 

        
    def __call__(self,
                 x):

        xx = self.packer.unpack(x)
        S_init = xx['S_init']
        E_init = xx['E_init']
        I_init = xx['I_init']
        beta0 = xx['beta0']
        omega = xx['omega']
        eps = xx['eps']
        c_mat = self.packer.c_vec_to_mat(xx['c_vec'])
        
        all_results = []
        for season_idx, season in enumerate(self.packer.seasons):
             
            df = run(
                S_init=S_init[season_idx],
                E_init=E_init[season_idx],
                I_init=I_init[season_idx],
                n_weeks=self.n_weeks,
                beta0=beta0,
                sigma=self.sigma,
                dt_output=self.dt_output,
                dt_euler=self.dt_euler,
                mu=self.mu,
                nu=self.nu,
                omega=omega,
                eps=eps,
                n_regions=len(self.packer.regions),
                contact_matrix=c_mat,
                population=self.population.query("season == @season").population,
                start_date=season
            )

            df = df[[f"C{i}" for i in range(self.packer.n_regions)]].reset_index(drop=False)
            
            # melt the dataframe: columns time, incidence, region, season
            df_long = df.melt(id_vars=["time"], var_name="region", value_name="incidence")
            df_long["region"] = df_long["region"].str.replace("C", "").astype(int)
            df_long["region"] = df_long.region.replace(self.packer.region_dict)
            df_long["season"] = season
            all_results.append(df_long)

        res = pd.concat(all_results, ignore_index=True)
        import pdb; pdb.set_trace()
        return res
        

def main():
    seasons = ["1990-01-01", "1991-01-02"]
    regions = ['HHS1', 'HHS2']
    packer = Packer(seasons=seasons, regions=regions)
    x = packer.random_packed(43)
    population = [{"season": s, "region": r, "population": 1.} for r, s in product(regions, seasons)]
    population = pd.DataFrame(population)
    
    sim = Simulator(population=population)
    sim(x)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb
        traceback.print_exc()  # Prints the full stack trace to stderr
        pdb.post_mortem()      # Starts debugger at the poi

