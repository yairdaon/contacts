import time

import pytest
import numpy as np
import pandas as pd
import sys
import os
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverter import Inverter

NWEEKS = 20
EPS = 0.1 ## Allowed relative (!!) error

def makepop(n_regions=10, n_seasons=30):
    seasons = [f"{year}-01-01" for year in range(1990, 1990 + n_seasons)]
    regions = [f"HHS{region}" for region in range(n_regions)]

    # Setup: 1 season, 2 regions for simpler test
    population_data = []
    for season, region in product(seasons, regions):
        population_data.append({
            'season': season,
            'region': region,
            'population': 100
        })

    return pd.DataFrame(population_data)


def test_inverter_initialization():
    """Test that Inverter initializes correctly with population dataframe."""
    pop = makepop()

    # Test initialization
    inv = Inverter(population=pop, n_weeks=NWEEKS)

    assert inv.packer.n_seasons == pop.season.nunique()
    assert inv.packer.n_regions == pop.region.nunique()
    assert inv.n_weeks == NWEEKS
    assert pop.shape == (inv.packer.n_seasons * inv.packer.n_regions, 3)

    print("✓ Inverter initialization test passed")


def test_sim():
    """Test that Inverter.sim() produces valid output."""
    pop = makepop()
    inv = Inverter(population=pop, n_weeks=NWEEKS)

    total = 0
    # Generate random parameters and run simulation
    for i in range(100):
        x = inv.packer.random_vector(seed=i*23)
        start = time.time()
        results = inv.sim(x)
        total = total + time.time() - start

        # Check output format
        assert isinstance(results, pd.DataFrame)
        expected_cols = {'time', 'region', 'incidence', 'season'}
        assert set(results.columns) == expected_cols

        # Check data completeness
        expected_rows = pop.season.nunique() * pop.region.nunique() * inv.n_weeks
        assert len(results) == expected_rows

        # Check no NaN values in incidence
        for (region, season), dd in results.groupby(['region', 'season']):
            inc = dd.reset_index(drop=True).loc[1:, 'incidence']
            assert not inc.isna().any(), (region, season)

            # Check all incidence values     are non-negative
            assert (inc >= 0).all()

    print("Total", total)
    print("Sims", inv.run_time)


def test_inference(seed=43):
    """
    Test that Inverter can recover known parameters from synthetic data.
    This is the key test for parameter inference capability.
    """
    pop = makepop()

    # Create "true" parameters that we'll try to recover
    inv = Inverter(population=pop, n_weeks=NWEEKS)
    true = inv.packer.random_dict(seed=seed)

    # Pack true parameters
    x_true = inv.packer.pack(inv.packer.pop2real(true))
    assert not np.isnan(x_true).any()

    # Generate "observed" data using true parameters
    obs_data = inv.sim(x_true)

    print(f"Generated {len(obs_data)} observations")
    print(f"True parameters - beta0: {true['beta0']}, eps: {true['eps']}")

    # Fit model (with limited iterations for testing)
    inv.fit(obs=obs_data)

    # Compare inferred vs true parameters
    inferred_params = inv.params

    err_beta0 = abs(true['beta0'] - inferred_params['beta0']) / true['beta0']
    err_eps = abs(true['eps'] - inferred_params['eps']) / true['eps']
    err_omega = np.abs(true['omega'] - inferred_params['omega']) / true['omega']
    err_c = np.abs(true['c_vec'] - inferred_params['c_vec']) / true['c_vec']

    print("\nParameter Recovery Results:")
    print(f"  beta0 - True: {true['beta0']:.3f}, Inferred: {inferred_params['beta0']:.3f} err {err_beta0:.3f}")
    print(f"  eps   - True: {true['eps']:.3f}, Inferred: {inferred_params['eps']:.3f}, err {err_eps:.3f}")
    print(f"  omega - True: {true['omega']}, Inferred: {inferred_params['omega']}, err {err_omega:.3f}")
    print(f"  c - True: {true['c_vec']}, Inferred: {inferred_params['c_vec']}, err {err_c:.3f}")

    assert err_beta0 < EPS, err_beta0
    assert err_eps < EPS, err_eps
    assert np.all(err_omega < EPS), err_omega
    assert np.all(err_c < EPS), err_c

    # Test that final loss is finite and reasonable
    assert np.isfinite(inv.fun)
    assert inv.fun >= 0


def test_noisy(seed=43):
    """
    Test that Inverter can recover known parameters from synthetic data.
    This is the key test for parameter inference capability.
    """
    pop = makepop(n_seasons=15)

    # Create "true" parameters that we'll try to recover
    inv = Inverter(population=pop, n_weeks=NWEEKS)
    true = inv.packer.random_dict(seed=seed)

    # Pack true parameters
    x_true = inv.packer.pack(inv.packer.pop2real(true))
    assert not np.isnan(x_true).any()

    # Generate "observed" data using true parameters
    obs = inv.sim(x_true)
    obs['incidence'] = obs['incidence'] + np.random.normal(size=obs.shape[0]) * obs['incidence'] / 10


    print(f"Generated {len(obs)} observations")
    print(f"True parameters - beta0: {true['beta0']}, eps: {true['eps']}")

    # Fit model (with limited iterations for testing)
    inv.fit(obs=obs)

    # Compare inferred vs true parameters
    inferred_params = inv.params

    err_beta0 = abs(true['beta0'] - inferred_params['beta0']) / true['beta0']
    err_eps = abs(true['eps'] - inferred_params['eps']) / true['eps']
    err_omega = np.abs(true['omega'] - inferred_params['omega']) / true['omega']
    err_c = np.abs(true['c_vec'] - inferred_params['c_vec']) / true['c_vec']

    print("\nParameter Recovery Results:")
    print(f"  beta0 - True: {true['beta0']:.3f}, Inferred: {inferred_params['beta0']:.3f} err {err_beta0:.3f}")
    print(f"  eps   - True: {true['eps']:.3f}, Inferred: {inferred_params['eps']:.3f}, err {err_eps:.3f}")
    print(f"  omega - True: {true['omega']}, Inferred: {inferred_params['omega']}, err {err_omega:.3f}")
    print(f"  c - True: {true['c_vec']}, Inferred: {inferred_params['c_vec']}, err {err_c:.3f}")

    assert err_beta0 < EPS, err_beta0
    assert err_eps < EPS, err_eps
    assert np.all(err_omega < EPS), err_omega
    assert np.all(err_c < EPS), err_c

    # Test that final loss is finite and reasonable
    assert np.isfinite(inv.fun)
    assert inv.fun >= 0


if __name__ == "__main__":
    try:
        test_inverter_initialization()
        test_sim()
        test_inference()

    except:
        import traceback as tb
        import pdb
        tb.print_exc()
        pdb.post_mortem()
