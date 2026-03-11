"""
Run inverse problem on real epidemic data for all state pairs, recording granular CRLB and optimization data.
"""

import os
import numpy as np
import pandas as pd
import nlopt
import us
from itertools import combinations

from src.data_loader import load_real
from src.inverter import Inverter
from src.crlb import compute_crlb
from src.flu import Mortality as flu
from joblib import Parallel, delayed
from tqdm import tqdm

OUTPUT_DIR = "outputs/states"

def process_pair(state1, state2, seasons):
    # Ensure alphabetical order for filename and state assignment
    s1_abbr = state1.abbr
    s2_abbr = state2.abbr
    if s1_abbr > s2_abbr:
        state1, state2 = state2, state1
        s1_abbr, s2_abbr = s2_abbr, s1_abbr
    
    filename = f"{OUTPUT_DIR}/{s1_abbr}x{s2_abbr}.csv"
    if os.path.exists(filename):
        return

    regions = [state1.name, state2.name]
    
    # Load real data
    obs, phase = load_real(
        disease=flu,
        regions=regions,
        seasons=seasons
    )

    if obs.empty:
        print(f"No data for {s1_abbr}x{s2_abbr}")
        return

    # Check if we have data for both regions in at least some seasons
    found_regions = obs['region'].unique()
    if len(found_regions) < 2:
        print(f"Insufficient regional data for {s1_abbr}_{s2_abbr}: found {found_regions}")
        return

    print(f"Processing {s1_abbr}_{s2_abbr}...")

    try:
        print("Run inverse problem")
        n0 = 500
        inv = Inverter(
            optimizer=nlopt.LD_SLSQP,
            phase=phase,
            obs=obs,
            disease=flu
        ).fit(n0=n0, maxeval=None, n_jobs=-1)

        rows = []
        active_seasons = obs['season'].unique()
        
        print("Calculating CRLB for each season")
        def compute_single_row(run_idx, res, season):
            fitted = inv.packer.unpack(res['x'])
            theta = fitted['theta']
            obj = res['fun']
            success = res['success']
            
            season_idx = seasons.index(season)
            S0 = fitted['S_init'][season_idx, :]    
            I0 = fitted['I_init'][season_idx, :]
            Ts = np.sort(obs.query("season == @season")['t'].unique())
            
            row = {
                'state1': s1_abbr,
                'state2': s2_abbr,
                'run_idx': run_idx,
                'season': season,
                'theta': theta,
                'objective': obj,
                'success': success,
                'S1_0': S0[0],
                'S2_0': S0[1],
                'I1_0': I0[0],
                'I2_0': I0[1],
                'crlb': np.nan,
                'error': ''
            }
            
            try:
                crlb_val = compute_crlb(
                    S0=S0,
                    I0=I0,
                    gamma=flu.gamma,
                    theta=theta,
                    Ts=Ts,
                    beta0=flu.beta0,
                    eps=flu.eps,
                    rho=flu.rho,
                    phase=phase
                )
                row['crlb'] = crlb_val
            except Exception as e:
                row['error'] = str(e)
            return row

        tasks = [
            delayed(compute_single_row)(run_idx, res, season)
            for run_idx, res in enumerate(inv.results)
            for season in active_seasons
        ]
        rows = Parallel(n_jobs=-1)(tqdm(tasks, desc="CRLB Calculation", total=len(tasks)))

        # Save all rows for this pair
        df_results = pd.DataFrame(rows)
        df_results.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(df_results)} rows.")

    except Exception as e:
        print(f"Error processing {s1_abbr}_{s2_abbr}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    seasons = list(range(2009, 2018)) + [2022, 2023, 2024]
    all_states = us.STATES
    
    # Generate all unique pairs
    pairs = list(combinations(all_states, 2))
    print(f"Total pairs to process: {len(pairs)}")

    for s1, s2 in pairs:
        process_pair(s1, s2, seasons)

if __name__ == "__main__":
    main()
