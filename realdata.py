"""
Run inverse problem on real epidemic data for all state pairs, recording granular CRLB and optimization data.
"""

import os
import numpy as np
import pandas as pd
import nlopt
import us
from itertools import combinations

from src.helper import current
from src.data_loader import load_real
from src.inverter import Inverter
from src.crlb import compute_crlb
from src.flu import Mortality as flu
from joblib import Parallel, delayed
from tqdm import tqdm
OUTPUT_DIR = "outputs/states"

    
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    seasons = list(range(2009, 2019)) + [2022, 2023, 2024, 2025]    

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
        if os.path.exists(filename):
            continue



        regions = [state1.name, state2.name]
    
        # Load real data
        obs, phase = load_real(
            disease=flu,
            regions=regions,
            seasons=seasons
        )

        if obs.empty:
            print(f"No data for {s1_abbr}x{s2_abbr}")
            continue

        # Check if we have data for both regions in at least some seasons
        found_regions = obs['region'].unique()
        if len(found_regions) < 2:
            print(f"Insufficient regional data for {s1_abbr}x{s2_abbr}: found {found_regions}")
            continue

        print(f"\n\nInversion for {s1_abbr} X {s2_abbr}", current())
        inv = Inverter(
            optimizer=nlopt.LD_SLSQP,
            phase=phase,
            obs=obs,
            disease=flu
        ).fit(n0=5, maxeval=None, n_jobs=-3)
        print("Finished inversion", current())

        # Save results for only the best fit
        rows = []
        fitted = inv.objective.packer.unpack(inv.x)
        for i, season in enumerate(seasons):
            try:
                bound = compute_crlb(
                    S0=fitted['S_init'][i, :],
                    I0=fitted['I_init'][i, :],
                    gamma=flu.gamma,
                    theta=fitted['theta'],
                    Ts=inv.objective.packer.all_Ts[season],
                    beta0=flu.beta0,
                    eps=flu.eps,
                    rho=flu.rho,
                    phase=phase
                )
                err  = ''
            except Exception as e:
                bound = np.nan
                err = str(e)
                
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
                'crlb': bound,
                'error': err
              }
            rows.append(row)
            
        pd.DataFrame(rows).to_csv(filename, index=False)
        print(f"Saved {filename} at {current()}")


if __name__ == "__main__":
    main()
