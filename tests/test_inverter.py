import pytest
import numpy as np
import pandas as pd
import sys
import os
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverter import Inverter


def test_inverter_initialization():
    """Test that Inverter initializes correctly with population dataframe."""
    # Create test population data
    seasons = ['2020-01-01', '2021-01-01']
    regions = ['A', 'B']
    population_data = []
    for season, region in product(seasons, regions):
        population_data.append({
            'season': season,
            'region': region,
            'population': np.random.randint(low=10 ** 5, high=10 ** 6)
        })

    population_df = pd.DataFrame(population_data)

    # Test initialization
    inv = Inverter(population=population_df, n_weeks=20)

    assert inv.packer.n_seasons == 2
    assert inv.packer.n_regions == 2
    assert inv.n_weeks == 20
    assert population_df.shape == (inv.packer.n_seasons * inv.packer.n_regions, 3)

    print("✓ Inverter initialization test passed")


def test_inverter_sim():
    """Test that Inverter.sim() produces valid output."""
    # Simple 1 season, 2 regions setup
    seasons = ['2020-01-01']
    regions = ['Region1', 'Region2']
    population_data = []
    for season, region in product(seasons, regions):
        population_data.append({
            'season': season,
            'region': region,
            'population': 1e6
        })

    population_df = pd.DataFrame(population_data)
    inv = Inverter(population=population_df, n_weeks=10)

    # Generate random parameters and run simulation
    x = inv.packer.random_vector(seed=123)
    results = inv.sim(x)

    # Check output format
    assert isinstance(results, pd.DataFrame)
    expected_cols = {'time', 'region', 'incidence', 'season'}
    assert set(results.columns) == expected_cols

    # Check data completeness
    expected_rows = len(seasons) * len(regions) * inv.n_weeks
    assert len(results) == expected_rows

    # Check no NaN values in incidence
    for (region, season), dd in results.groupby(['region', 'season']):
        inc = dd.reset_index(drop=True).loc[1:, 'incidence']
        assert not inc.isna().any(), (region, season)

        # Check all incidence values     are non-negative
        assert (inc >= 0).all()

    print("✓ Inverter.sim() test passed")


def test_parameter_inference(seed=43):
    """
    Test that Inverter can recover known parameters from synthetic data.
    This is the key test for parameter inference capability.
    """
    print("Running parameter inference test...")
    seasons = ['2020-01-01', '2021-01-01', '2022-01-01']
    regions = ['Region1', 'Region2']

    # Setup: 1 season, 2 regions for simpler test
    population_data = []
    for season, region in product(seasons, regions):
        population_data.append({
            'season': season,
            'region': region,
            'population': 1e5
        })

    population_df = pd.DataFrame(population_data)

    # Create "true" parameters that we'll try to recover
    inv = Inverter(population=population_df, n_weeks=32)
    true = inv.packer.random_dict(seed=seed)

    # Pack true parameters
    x_true = inv.packer.pack(inv.packer.pop2real(true))
    assert not np.isnan(x_true).any()

    # Generate "observed" data using true parameters
    obs_data = inv.sim(x_true)

    print(f"Generated {len(obs_data)} observations")
    print(f"True parameters - beta0: {true['beta0']}, eps: {true['eps']}")

    # Note: In practice you'd use more iterations and better starting points
    np.random.seed(42)  # For reproducible starting point

    x0 = x_true + np.random.randn(x_true.size) * 1e-1
    inv.packer.verify_vector(x0)

    # Fit model (with limited iterations for testing)
    inv.fit(obs=obs_data, x0=x0)

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
    print("Running Inverter tests...")

    test_inverter_initialization()
    test_inverter_sim()
    test_parameter_inference()
    # test_contact_matrix_inference()

    print("\nAll Inverter tests completed!")
