"""
Run inverse problem on real epidemic data.
"""

import numpy as np
import pandas as pd
import nlopt

from src.data_loader import load_real
from src.inverter import Inverter
from src.crlb import compute_crlb
from src.flu import Mortality as flu

OUTPUT_DIR = "outputs"


def main():
    regions = ["California", "New York"]
    seasons = [2010, 2011, 2012]  # 2 seasons

    # Load real data
    obs, phase = load_real(
        disease=flu,
        regions=regions,
        seasons=seasons
    )

    print(f"Loaded {len(obs)} observations")
    print(f"Phase estimates: {np.degrees(phase)} degrees")
    print(f"Seasons: {obs['season'].unique()}")

    # Run inverse problem
    inv = Inverter(
        optimizer=nlopt.LD_SLSQP,
        phase=phase,
        obs=obs,
        disease=flu
    ).fit(n0=50, maxeval=None, n_jobs=-1)

    # Print results
    fitted = inv.packer.unpack(inv.x)
    theta = fitted['theta']
    print(f"\nFitted theta: {theta:.4f}")
    print(f"Final objective: {inv.fun:.4f}")
    print(f"Success: {inv.success}")
    
    # Compute CRLB for theta using fitted parameters
    # Fisher Information is additive across seasons: J_total = sum(J_i)
    # Since CRLB returns variance = 1/J, we have: 1/var_total = sum(1/var_i)
    n_regions = len(regions)
    precision = 0.0

    for season_idx, season in enumerate(seasons):
        
        # Get S0 and I0 for this season
        S0 = fitted['S_init'][season_idx, :]    
        I0 = fitted['I_init'][season_idx, :]
       
        # Get time array for this season
        Ts = np.sort(obs.query("season == @season")['t'].unique())
 
        # Compute CRLB (variance) for this season
        crlb = compute_crlb(
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

        # Accumulate Fisher information
        assert np.isfinite(crlb), f"Singular Fisher matrix season {season}, theta = {theta:.4f}"
        assert crlb > 0 
        precision += 1.0 / crlb
           
       

    variance = 1.0 / precision
    print(f"Combined CRLB std for theta: {np.sqrt(variance):.6e}")


if __name__ == "__main__":
    try:
        main()
    except:
        import sys, traceback, pdb
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
