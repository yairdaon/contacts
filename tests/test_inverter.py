import pytest
import numpy as np
import pandas as pd
import sys
import os
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverter import Inverter

NWEEKS = 20
def makepop():
    seasons = [f"{year}-01-01" for year in range(2020, 2028)]
    regions = ['Region1', 'Region2']

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

    # Generate random parameters and run simulation
    for i in range(100):
        x = inv.packer.random_vector(seed=i*23)
        results = inv.sim(x)

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


def test_parameter_inference(seed=43):
    """
    Test that Inverter can recover known parameters from synthetic data.
    This is the key test for parameter inference capability.
    """
    pop = makepop()

    # Create "true" parameters that we'll try to recover
    inv = Inverter(population=pop, n_weeks=15)
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

    print("\nParameter Recovery Results:")
    print(f"  beta0 - True: {true['beta0']:.3f}, Inferred: {inferred_params['beta0']:.3f}")
    print(f"  eps   - True: {true['eps']:.3f}, Inferred: {inferred_params['eps']:.3f}")
    print(f"  omega - True: {true['omega']}, Inferred: {inferred_params['omega']}")

    # Check if reasonably close (loose tolerances for test)
    assert abs(true['beta0'] - inferred_params['beta0']) / true['beta0'] < 0.1
    assert abs(true['eps'] - inferred_params['eps']) / true['eps'] < 0.1
    print("✓ Parameter inference test passed (parameters reasonably recovered)")

    # Test that final loss is finite and reasonable
    assert np.isfinite(inv.fun)
    assert inv.fun >= 0


#
# def test_contact_matrix_inference():
#     """Test specifically that we can infer contact matrix structure."""
#     print("Testing contact matrix inference...")
#
#     # 2 regions with known contact pattern
#     seasons = ['2020-01-01']
#     regions = ['Urban', 'Rural']
#     population_data = []
#     for season, region in product(seasons, regions):
#         population_data.append({
#             'season': season,
#             'region': region,
#             'population': 1e6
#         })
#
#     population_df = pd.DataFrame(population_data)
#     inv = Inverter(population=population_df, n_weeks=15)
#
#     # Create "true" contact matrix with strong asymmetry
#     # Urban affects Rural more than Rural affects Urban
#     true_c_vec = np.array([0.8])  # High contact from Rural to Urban
#     true_contact_matrix = inv.packer.c_vec_to_mat(true_c_vec)
#
#     print("True contact matrix:")
#     print(true_contact_matrix)
#
#     # Generate synthetic data with this contact pattern
#     true_params = inv.packer.random_dict()
#     true_params['c_vec'] = true_c_vec
#     true_params = inv.packer.real2pop(true_params.copy())
#
#     x_true = inv.packer.pack(true_params)
#     synthetic_data = inv.sim(x_true)
#
#     # Try to infer the contact matrix
#     try:
#         inv.fit(synthetic_data)
#         inferred_c_vec = inv.params['c_vec']
#         inferred_contact_matrix = inv.packer.c_vec_to_mat(inferred_c_vec)
#
#         print("Inferred contact matrix:")
#         print(inferred_contact_matrix)
#
#         # Check if structure is preserved (loose tolerance)
#         contact_error = abs(true_c_vec[0] - inferred_c_vec[0])
#         print(f"Contact parameter error: {contact_error:.3f}")
#
#         if contact_error < 0.3:
#             print("✓ Contact matrix inference test passed")
#         else:
#             print("⚠ Contact matrix inference shows large error (expected for complex fitting)")
#
#     except Exception as e:
#         print(f"⚠ Contact matrix inference completed with issues: {e}")
#
#     print("✓ Contact matrix inference framework test passed")
#

if __name__ == "__main__":
    try:
        test_inverter_initialization()
        test_sim()
        test_parameter_inference()
        # test_contact_matrix_inference()

    except:
        import traceback as tb
        import pdb
        tb.print_exc()
        pdb.post_mortem()