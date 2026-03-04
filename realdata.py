"""
Run inverse problem on real epidemic data.
"""

import numpy as np
import pandas as pd
import nlopt

from src.data_loader import load_real
from src.inverter import Inverter
from src.flu import Mortality as flu

OUTPUT_DIR = "outputs"


def main():
    regions = ["California", "New York"]
    seasons = [2010, 2011]  # 2 seasons

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
    ).fit(n0=5, maxeval=None, n_jobs=-1)

    # Print results
    fitted = inv.packer.unpack(inv.x)
    print(f"\nFitted theta: {fitted['theta']:.4f}")
    print(f"Final objective: {inv.fun:.4f}")
    print(f"Success: {inv.success}")


if __name__ == "__main__":
    try:
        main()
    except:
        import sys, traceback, pdb
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
