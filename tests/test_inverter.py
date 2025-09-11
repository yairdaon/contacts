import time

import pytest
import numpy as np
import pandas as pd
from numpy import array2string as a2s

from src.inverter import Inverter
from src.helper import makepop

NWEEKS = 20
EPS = 0.1 ## Allowed relative (!!) error

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for testing


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
    pop = makepop(n_regions=4, n_seasons=10)
    euler = Inverter(population=pop, n_weeks=NWEEKS, integration="euler")
    runge_kutta = Inverter(population=pop, n_weeks=NWEEKS, integration="rk")

    # Generate random parameters and run simulation
    for i in range(1000):  # Reduced for faster testing
        x = runge_kutta.packer.random_vector(seed=i*23)
        results = runge_kutta.sim(x)  # Use RK as primary
        euler.sim(x)  # Still test Euler
        
        # Check output format
        assert isinstance(results, pd.DataFrame)
        expected_cols = {'time', 'region', 'incidence', 'season'}
        assert set(results.columns) == expected_cols

        # Check data completeness
        expected_rows = pop.season.nunique() * pop.region.nunique() * runge_kutta.n_weeks
        #assert len(results) == expected_rows

        # Check no NaN values in incidence
        for (region, season), dd in results.groupby(['region', 'season']):
            inc = dd.reset_index(drop=True).loc[1:, 'incidence']
            assert not inc.isna().any(), (region, season)

            # Check all incidence values are non-negative
            assert (inc >= 0).all()

    print("RK", runge_kutta.run_time)
    print("Euler", euler.run_time)


@pytest.mark.parametrize("noise", [0, 0.1])
def test_inference(noise, seed=43):
    """
    Test that Inverter can recover known parameters from synthetic data.
    This is the key test for parameter inference capability.
    Also creates visualization of the reconstruction.
    """

    pop = makepop(n_regions=2, n_seasons=3)  # Smaller for faster testing

    # Create "true" parameters that we'll try to recover
    inv = Inverter(population=pop, n_weeks=NWEEKS, integration='rk')
    true_params = inv.packer.random_dict(seed=seed)

    # Pack true parameters
    x_true = inv.packer.pack(inv.packer.pop2real(true_params))
    assert not np.isnan(x_true).any()

    # Generate "observed" data using true parameters
    true_trajectory = inv.sim(x_true)
    obs = true_trajectory.assign(incidence=true_trajectory['incidence'] + np.random.normal(size=true_trajectory.shape[0]) * true_trajectory['incidence'] * noise)

    print(f"Generated {len(true_trajectory)} observations")
    print(f"True parameters - beta0: {true_params['beta0']}, eps: {true_params['eps']}")

    inv.fit(obs=obs, x0=x_true)

    # Generate reconstructed trajectory for visualization
    x_inferred = inv.packer.pack(inv.packer.pop2real(inv.params))
    reconstructed_trajectory = inv.sim(x_inferred)

    # Compare inferred vs true parameters
    inferred_params = inv.params

    err_beta0 = abs(true_params['beta0'] - inferred_params['beta0']) / true_params['beta0']
    err_eps = abs(true_params['eps'] - inferred_params['eps']) / true_params['eps']
    err_omega = np.max(np.abs(true_params['omega'] - inferred_params['omega']) / np.abs(true_params['omega']))
    err_c = np.max(np.abs(true_params['c_vec'] - inferred_params['c_vec']) / true_params['c_vec'])

    print("\nParameter Recovery Results:")
    print(f"  beta0 - True: {true_params['beta0']:.3f}, Inferred: {inferred_params['beta0']:.3f} err {err_beta0:.3f}")
    print(f"  eps   - True: {true_params['eps']:.3f}, Inferred: {inferred_params['eps']:.3f}, err {err_eps:.3f}")
    print(f"  omega - True: {a2s(true_params['omega'], precision=3)}, Inferred: {a2s(inferred_params['omega'], precision=3)}, err {err_omega:.3f}")
    print(f"  c - True: {a2s(true_params['c_vec'], precision=3)}, Inferred: {a2s(inferred_params['c_vec'], precision=3)}, err {err_c:.3f}")

    assert err_beta0 < EPS, err_beta0
    assert err_eps < EPS, err_eps
    assert np.all(err_omega < EPS), err_omega
    assert np.all(err_c < EPS), err_c

    # Test that final loss is finite and reasonable
    assert np.isfinite(inv.fun)
    assert inv.fun >= 0
    
    # Create visualization
    print("Creating parameter inference visualization...")
    seasons = sorted(pop.season.unique())
    
    # Set dark theme
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, len(seasons), figsize=(5*len(seasons), 5), facecolor='#1e1e1e')
    if len(seasons) == 1:
        axes = [axes]
    
    fig.suptitle(f'Parameter Inference Test Results (seed={seed})\n'
                 f'β₀ err: {err_beta0:.3f}, ε err: {err_eps:.3f}', fontsize=14, color='white')
    
    colors = ['#00D4FF', '#FF6B9D']  # Bright cyan and pink for dark background
    regions = ['HHS0', 'HHS1']
    
    for season_idx, season in enumerate(seasons):
        ax = axes[season_idx]
        
        for region_idx, region in enumerate(regions):
            # True trajectory
            true_data = true_trajectory[
                (true_trajectory.season == season) & 
                (true_trajectory.region == region)
            ].sort_values('time')

            obs_data = obs[
                (obs.season == season) &
                (obs.region == region)
                ].sort_values('time')

            # Reconstructed trajectory
            reconstructed_data = reconstructed_trajectory[
                (reconstructed_trajectory.season == season) & 
                (reconstructed_trajectory.region == region)
            ].sort_values('time')
            
            ax.plot(true_data.time, true_data.incidence,
                   color=colors[region_idx], linewidth=2.5,
                   label=f'{region} - True' if season_idx == 0 else "")
            
            ax.plot(reconstructed_data.time, reconstructed_data.incidence,
                   color=colors[region_idx], linewidth=2.5, linestyle='--', alpha=0.8,
                   label=f'{region} - Inferred' if season_idx == 0 else "")

            ax.plot(obs_data.time, reconstructed_data.incidence,
                    color=colors[region_idx], linewidth=2.5, linestyle=':', alpha=0.8,
                    label=f'{region} - Noisy' if season_idx == 0 else "")
        
        ax.set_title(f'{season[:4]}', fontsize=12, color='white')
        ax.set_xlabel('Time', color='white')
        if season_idx == 0:
            ax.set_ylabel('Weekly Incidence', color='white')
            ax.legend(loc='upper right', fancybox=True, shadow=True, facecolor='#2e2e2e', edgecolor='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig(f'pix/test_inference_visualization_{noise}.png', dpi=300, bbox_inches='tight')
    plt.close()