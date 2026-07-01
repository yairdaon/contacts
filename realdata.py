"""
Run inverse problem on real epidemic data for all state pairs, recording granular precision and optimization data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nlopt
import us
from itertools import combinations

from src.data_loader import load_real
from src.inverter import Inverter
from src.helpers import current
from src.flu import Mortality as flu
from joblib import Parallel, delayed
from tqdm import tqdm
OUTPUT_DIR = "outputs/states"

    
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("pix", exist_ok=True)

    seasons = list(range(2010, 2019)) + [2023, 2024, 2025]

    # Load population data once
    # Season Y starts Nov 1st of year (Y-1), so use July 1st population of year (Y-1)
    pop_df = pd.read_csv("data/pni_mortality/populations.csv", parse_dates=["date"])
    pop_df['season'] = pop_df['date'].dt.year + 1  # July 1st, Y -> season Y+1
    pop_df = pop_df[['season', 'state', 'population']].set_index(['season', 'state'])

    
    for state1, state2 in combinations(us.STATES, 2):
        if state1 == state2:
            continue
        
        # Ensure alphabetical order for filename and state assignment
        s1_abbr = state1.abbr
        s2_abbr = state2.abbr
        if s1_abbr > s2_abbr:
            state1, state2 = state2, state1
            s1_abbr, s2_abbr = s2_abbr, s1_abbr

        filename = f"{OUTPUT_DIR}/{s1_abbr}x{s2_abbr}.csv"
        regions = [state1.name, state2.name]
    
        # Load real data
        obs, phase = load_real(
            disease=flu,
            regions=regions,
            seasons=seasons,
            mortality_path="data/pni_mortality/excess_deaths.csv"
        )
        if obs.empty:
            print(f"No data for {s1_abbr}x{s2_abbr}")
            continue
        
        # Check if we have data for both regions in at least some seasons
        found_regions = obs['region'].unique()
        if len(found_regions) < 2:
            print(f"Insufficient regional data for {s1_abbr}x{s2_abbr}: found {found_regions}")
            continue

        # Build populations dictionary from MultiIndex dataframe
        populations = {}
        try:
            for season in seasons:
                for region in regions:
                    populations[(season, region)] = pop_df.loc[(season, region), 'population']
        except KeyError:
            print(f"Missing population data for {s1_abbr}x{s2_abbr}, skipping")
            continue
                

        print(f"\n\nInversion for {s1_abbr} X {s2_abbr}", current())
        inv = Inverter(
            phase=phase,
            obs=obs,
            disease=flu,
            populations=populations
        ).fit(n0=200, n_jobs=-1, fname=f"pix/rec_{s1_abbr}x{s2_abbr}")
        print("Finished inversion", current())

        # Save results for only the best fit
        rows = []
        fitted = inv.objective.packer.unpack(inv.x)
        for i, season in enumerate(seasons):
            row = {
                'state1': s1_abbr,
                'state2': s2_abbr,
                'season': season,
                'objective': inv.fun,
                'success': inv.success,
                'theta': fitted['theta'],
                'S1_0': fitted['S_init'][i, 0],
                'S2_0': fitted['S_init'][i, 1],
                'I1_0': fitted['I_init'][i, 0],
                'I2_0': fitted['I_init'][i, 1],
                'precision': inv.precisions[i],
                'status': inv.desc,
                'runtime': inv.runtime
              }
            rows.append(row)
        
        res = pd.DataFrame(rows)
        res.to_csv(filename, index=False)
        print(res.set_index(['state1', 'state2'], drop=True)[['season', 'theta', 'precision']])
        print(f"Ran {filename} at {current()}")

if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except:
    #     import sys, traceback, pdb
    #     _, _, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)
