import time

import pytest
import numpy as np
import pandas as pd

from src.inverter import Inverter
from src.helper import makepop, a2s

NWEEKS = 28
EPS = 0.1 ## Allowed relative (!!) error

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for testing

@pytest.mark.parametrize("transform", [True, False])
def test_inverter_initialization(transform):
    """Test that Inverter initializes correctly with population dataframe."""
    pop = makepop()

    # Test initialization
    inv = Inverter(population=pop, n_weeks=NWEEKS, transform=transform)

    assert inv.packer.n_seasons == pop.season.nunique(), f"Packer seasons {inv.packer.n_seasons} != population seasons {pop.season.nunique()}"
    assert inv.packer.n_regions == pop.region.nunique(), f"Packer regions {inv.packer.n_regions} != population regions {pop.region.nunique()}"
    assert inv.n_weeks == NWEEKS, f"Inverter weeks {inv.n_weeks} != expected {NWEEKS}"
    assert pop.shape == (inv.packer.n_seasons * inv.packer.n_regions, 3), f"Population shape {pop.shape} != expected ({inv.packer.n_seasons * inv.packer.n_regions}, 3)"


@pytest.mark.parametrize("transform", [True, False])
def test_sim(transform):
    """Test that Inverter.sim() produces valid output."""
    pop = makepop(n_regions=10, n_seasons=30)
    inv = Inverter(population=pop, n_weeks=NWEEKS, transform=transform)

    # Generate random parameters and run simulation
    for i in range(10):
        params = inv.packer.random_dict()
        results = inv.sim(params)  # Use RK as primary

        # Check output format
        assert isinstance(results, pd.DataFrame), f"Simulation result is {type(results)}, expected DataFrame"
        expected_cols = {'time', 'region', 'incidence', 'season'}
        assert set(results.columns) == expected_cols, f"Result columns {set(results.columns)} != expected {expected_cols}"

        # Check data completeness
        expected_rows = pop.season.nunique() * pop.region.nunique() * inv.n_weeks
        assert len(results) == expected_rows, f"Result length {len(results)} != expected {expected_rows}"
        assert results.incidence.min() >= 0, f"Negative incidence found: min={results.incidence.min()}"


    print(f"Run time transform {transform}", inv.run_time)



@pytest.mark.parametrize("rho", [0.95, 0.8])
@pytest.mark.parametrize("cheat", [True, False])

@pytest.mark.parametrize("transform", [True, False])
def test_inference(rho, cheat, transform, seed=43):
    """
    Test that Inverter can recover known parameters from synthetic data.
    This is the key test for parameter inference capability.
    Also creates visualization of the reconstruction.
    """
    pop = makepop(n_regions=5, n_seasons=30)

    # Create "true" parameters that we'll try to recover
    inv = Inverter(population=pop, n_weeks=NWEEKS, transform=transform)
    true_params = inv.packer.random_dict(seed=seed)
    # Override rho with the parameterized value for testing
    true_params['rho'] = rho

    # Pack true parameters
    x_true = inv.packer.pack(inv.packer.pop2real(true_params))
    x0 = x_true if cheat else inv.packer.random_vector()
    assert not np.isnan(x_true).any(), f"NaN values found in true parameter vector"

    # Generate "observed" data using true parameters  
    initial_params = inv.packer.unpack(x0)
    initial_params = inv.packer.real2pop(initial_params)
    true_trajectory = inv.sim(initial_params)

    # Generate observed data using Poisson sampling with true reporting rate
    obs = true_trajectory.copy()
    obs['incidence'] = np.random.poisson(true_trajectory['incidence'] * true_params['rho'])

    print(f"Generated {len(true_trajectory)} observations")
    inv.fit(obs=obs, x0=x_true, n_starts=5)  # Use 5 starts for testing

    # Generate reconstructed trajectory for visualization
    reconstructed_trajectory = inv.sim(inv.params)

    # Compare inferred vs true parameters
    inferred_params = inv.params

    err_beta0 = abs(true_params['beta0'] - inferred_params['beta0']) / true_params['beta0']
    err_eps = abs(true_params['eps'] - inferred_params['eps']) / true_params['eps']
    err_rho = abs(true_params['rho'] - inferred_params['rho']) / true_params['rho']
    err_omega = np.max(np.abs(true_params['omega'] - inferred_params['omega']) / np.abs(true_params['omega']))
    err_c = np.max(np.abs(true_params['c_vec'] - inferred_params['c_vec']) / true_params['c_vec'])

    print(f"\nOptimization completed in {inv.optimization_time:.2f}s")
    print("Parameter Recovery Results:")
    print(f"  beta0  - True: {true_params['beta0']:.3f}, Inferred: {inferred_params['beta0']:.3f}, err: {err_beta0:.3f}")
    print(f"  eps    - True: {true_params['eps']:.3f}, Inferred: {inferred_params['eps']:.3f}, err: {err_eps:.3f}")
    print(f"  rho    - True: {true_params['rho']:.3f}, Inferred: {inferred_params['rho']:.3f}, err: {err_rho:.3f}")
    print(f"  omega  - True: {a2s(true_params['omega'])}, Inferred: {a2s(inferred_params['omega'])}, err: {err_omega:.3f}")
    print(f"  c      - True: {a2s(true_params['c_vec'])}, Inferred: {a2s(inferred_params['c_vec'])}, err: {err_c:.3f}")

    assert err_beta0 < EPS, f"err beta0 = {err_beta0:.3f}"
    assert err_eps < EPS, f"err eps {err_eps:.3f}"
    assert err_rho < EPS, f"err rho {err_rho:.3f}"
    assert np.all(err_omega < EPS), f"err omega {err_omega:.3f}"
    assert np.all(err_c < EPS), f"err c {err_c::.3f}"

    # Test that final loss is finite and reasonable
    assert np.isfinite(inv.fun), f"Final loss is not finite: {inv.fun}"
    assert inv.fun >= 0, f"Final loss is negative: {inv.fun}"
    
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
    plt.savefig(f'pix/test_inference_visualization_{method}_{rho}_{ic_noise}.png', dpi=300, box_inches='tight')
    plt.close()
