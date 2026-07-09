"""
Run inverse problem on real epidemic data for all state pairs, recording granular precision and optimization data.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nlopt
import us
from itertools import combinations

from src.data_loader import load_real, load_national_driver
from src.inverter import Inverter
from src.helpers import current
from src.flu import Mortality as flu
from joblib import Parallel, delayed
from tqdm import tqdm
OUTPUT_DIR = "outputs/states"

    
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("pix", exist_ok=True)

    ##  the 2019 season starts at novembe 2019. it is excluded due to covid.
    seasons = list(range(2010, 2019)) + [2023, 2024, 2025] 

    # Load population data once
    # Season Y starts Nov 1st of year (Y-1), so use July 1st population of year (Y-1)
    pop_df = pd.read_csv("data/pni_mortality/populations.csv", parse_dates=["date"])
    pop_df['season'] = pop_df['date'].dt.year + 1  # July 1st, Y -> season Y+1
    pop_df = pop_df[['season', 'state', 'population']].set_index(['season', 'state'])

    # National-driver: per-capita national infection rate per (season, week).
    # Full US aggregate (all states). Used as the α-weighted exogenous FOI term.
    nat_driver = load_national_driver(
        mortality_path="data/pni_mortality/excess_deaths.csv",
        pop_path="data/pni_mortality/populations.csv",
        seasons=seasons,
        rho=flu.rho,
        n_weeks=flu.n_weeks,
    )

    pairs = list(combinations(us.STATES, 2))
    random.shuffle(pairs)

    for state1, state2 in pairs:
        if state1 == state2:
            continue
        
        # Ensure alphabetical order for filename and state assignment
        s1_abbr = state1.abbr
        s2_abbr = state2.abbr
        if s1_abbr > s2_abbr:
            state1, state2 = state2, state1
            s1_abbr, s2_abbr = s2_abbr, s1_abbr

        # Optional pilot filter: env var CONTACTS_PILOT_PAIRS="CAxNY,GAxOH,..."
        # limits the run to that comma-separated list of pair filenames.
        pilot = os.environ.get("CONTACTS_PILOT_PAIRS", "").strip()
        if pilot and f"{s1_abbr}x{s2_abbr}" not in set(pilot.split(",")):
            continue

        filename = f"{OUTPUT_DIR}/{s1_abbr}x{s2_abbr}.csv"
        # Resumability: skip pairs already written.  Move/back up outputs/states/
        # before running from scratch so this check doesn't stop a fresh run.
        if os.path.exists(filename):
            continue
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
                

        rows = []
        # SARIMA-mirroring 2x2: alpha_upper in {0, 1} x theta_upper in {0, 1}.
        # LRT of M_full vs M_alpha is the mechanistic analog of Thivierge's
        # conditional Granger test. Comparisons across the four fits exhibit
        # the paradox: θ alone looks significant, but vanishes once α is on.
        for k in [10.0]:
            for alpha_upper in [0.0, 1.0]:
                for theta_upper in [0.0, 1.0]:
                    print(f"\n\nInversion for {s1_abbr} X {s2_abbr} "
                          f"(alpha_upper={alpha_upper}, theta_upper={theta_upper})",
                          current())
                    inv = Inverter(
                        phase=phase,
                        obs=obs,
                        disease=flu,
                        populations=populations,
                        nat_driver=nat_driver,
                        theta_upper=theta_upper,
                        alpha_upper=alpha_upper,
                        k=k,
                    ).fit(n0=200, n_jobs=-1,
                          fname=f"pix/rec_{s1_abbr}x{s2_abbr}_a{alpha_upper}_t{theta_upper}")
                    print("Finished inversion", current())

                    fitted = inv.objective.packer.unpack(inv.x)
                    for i, season in enumerate(seasons):
                        rows.append({
                            'state1': s1_abbr,
                            'state2': s2_abbr,
                            'season': season,
                            'k': k,
                            'alpha_limit': alpha_upper,
                            'theta_limit': theta_upper,
                            'objective': inv.fun,
                            'log_likelihood': inv.log_likelihood,
                            'success': inv.success,
                            'alpha': fitted.get('alpha', 0.0),
                            'theta': fitted.get('theta', 0.0),
                            'S1_0': fitted['S_init'][i, 0],
                            'S2_0': fitted['S_init'][i, 1],
                            'I1_0': fitted['I_init'][i, 0],
                            'I2_0': fitted['I_init'][i, 1],
                            # Fit-level aggregated (alpha,theta) FIM and CRLB
                            # for theta (repeated across season rows).
                            'J_aa': inv.J_aa,
                            'J_at': inv.J_at,
                            'J_tt': inv.J_tt,
                            'crlb_theta': inv.crlb_theta,
                            'precision': inv.precision,
                            'status': inv.desc,
                            'runtime': inv.runtime,
                        })

        res = pd.DataFrame(rows)
        res.to_csv(filename, index=False)
        print(res.set_index(['state1', 'state2'], drop=True)[
            ['alpha_limit', 'theta_limit', 'season', 'alpha', 'theta',
             'log_likelihood', 'precision']
        ])
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
