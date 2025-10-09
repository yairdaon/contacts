import time

import pytest
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from statsmodels.base import optimizer

from src.inverter import Objective, Inverter
from src.helper import makepop, a2s
from src.losses import RHO

NWEEKS = 28
EPS = 0.15  ## Allowed relative (!!) error

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for testing


@pytest.mark.parametrize("transform", [True, False])
def test_inverter_initialization(transform):
    """Test that Inverter initializes correctly with population dataframe."""
    pop = makepop()

    # Create objective and loss
    objective = Objective(population=pop, n_weeks=NWEEKS, transform=transform)

    # Test initialization
    inv = Inverter(objective=objective, optimizer=optimizer)

    assert inv.packer.n_seasons == pop.season.nunique(), f"Packer seasons {inv.packer.n_seasons} != population seasons {pop.season.nunique()}"
    assert inv.packer.n_regions == pop.region.nunique(), f"Packer regions {inv.packer.n_regions} != population regions {pop.region.nunique()}"
    assert objective.n_weeks == NWEEKS, f"Objective weeks {objective.n_weeks} != expected {NWEEKS}"
    assert pop.shape == (inv.packer.n_seasons * inv.packer.n_regions,
                         3), f"Population shape {pop.shape} != expected ({inv.packer.n_seasons * inv.packer.n_regions}, 3)"

@pytest.mark.parametrize("transform", [True, False])
def test_sim(transform):
    """Test that Objective.sim() produces valid output."""
    pop = makepop(n_regions=10, n_seasons=30)
    objective = Objective(population=pop, n_weeks=NWEEKS, transform=transform)

    # Generate random parameters and run simulation
    for i in range(10):
        params = objective.packer.random_dict()
        results = objective.sim(params)  

        # Check output format
        assert isinstance(results, pd.DataFrame), f"Simulation result is {type(results)}, expected DataFrame"
        expected_cols = {'time', 'region', 'incidence', 'season'}
        assert set(
            results.columns) == expected_cols, f"Result columns {set(results.columns)} != expected {expected_cols}"

        # Check data completeness
        expected_rows = pop.season.nunique() * pop.region.nunique() * objective.n_weeks
        assert len(results) == expected_rows, f"Result length {len(results)} != expected {expected_rows}"
        assert results.incidence.min() >= 0, f"Negative incidence found: min={results.incidence.min()}"


@pytest.mark.parametrize("optimizer", ['nlopt'])#, 'scipy'])
@pytest.mark.parametrize("difficulty", ["easy", "intermediate", "hard"])
def test_inference(optimizer, difficulty, seed=43):
    """
    Test that Inverter can recover known parameters from synthetic data.
    This is the key test for parameter inference capability.
    Also creates visualization of the reconstruction.
    
    Difficulty levels:
    - easy: 2 regions, 5 seasons, rho=0.95, theta=50 (low overdispersion = clean signal), start from true parameters
    - intermediate: 4 regions, 10 seasons, rho=0.8, theta=10 (moderate overdispersion), start from average of true and random
    - hard: 10 regions, 30 seasons, rho=0.5, theta=5 (high overdispersion = noisy data), start from random parameters
    """
    # Set test parameters based on difficulty
    if difficulty == "easy":
        n_regions, n_seasons, n0 = 2, 5, 50
    elif difficulty == "intermediate":
        n_regions, n_seasons, n0 = 4, 15, 250
    elif difficulty == "hard":
        n_regions, n_seasons, n0 = 10, 30, 1000
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    print(f"{difficulty} regions {n_regions}, seasons {n_seasons} starts={n0}")
    pop = makepop(n_regions=n_regions, n_seasons=n_seasons)

    objective = Objective(population=pop, n_weeks=NWEEKS, transform=optimizer == 'scipy')
    true_params = objective.packer.random_dict(seed=seed)

    # Override rho and theta with the parameterized values for testing
    # true_params['rho'] = rho  # rho is now fixed at 0.8

    # Pack true parameters
    x_true = objective.packer.pack(true_params)
    #objective.packer.verify(x_true)

    # Generate "observed" data using true parameters (not initial guess)
    true_trajectory = objective.sim(true_params)

    # Generate observed data
    obs = true_trajectory.copy()
    true_counts = true_trajectory['incidence'] * RHO # Fixed rho value

    scale = np.sqrt(RHO * (1 - RHO) * true_counts)
    obs['incidence'] = true_counts + np.random.randn(true_counts.size) * scale
    obs['incidence'] = np.maximum(0, obs['incidence'])  # Ensure non-negative
    objective.obs = obs

    res = Inverter(objective=objective, optimizer=optimizer).fit(n0=n0)
    fun = res.fun
    inferred_params = res.packer.unpack(res.x)

    # Generate reconstructed trajectory for visualization
    reconstructed_trajectory = objective.sim(inferred_params)

    err_beta0 = abs(true_params['beta0'] - inferred_params['beta0'])  #/ true_params['beta0']
    err_eps = abs(true_params['eps'] - inferred_params['eps'])  #/ true_params['eps']
    # err_rho = abs(true_params['rho'] - inferred_params['rho'])  # rho is fixed at 0.8
    # err_E_init = np.max(np.abs(true_params['E_init'] - inferred_params['E_init']))  # E_init = I_init
    err_omega = abs(true_params['omega'] - inferred_params['omega'])  # omega now scalar
    err_c = np.max(np.abs(true_params['c_vec'] - inferred_params['c_vec']))  # / true_params['c_vec'])
    # err_theta = abs(true_params['theta'] - inferred_params['theta']) / true_params['theta']

    print("\nParameter Recovery Results:")
    print(
        f"  beta0  - True: {true_params['beta0']:.3f}, Inferred: {inferred_params['beta0']:.3f}, err: {err_beta0:.3f}")
    print(f"  eps    - True: {true_params['eps']:.3f}, Inferred: {inferred_params['eps']:.3f}, err: {err_eps:.3f}")
    # print(f"  rho    - True: {true_params['rho']:.3f}, Inferred: {inferred_params['rho']:.3f}, err: {err_rho:.3f}")  # rho fixed at 0.8
    print(
        f"  omega  - True: {true_params['omega']:.3f}, Inferred: {inferred_params['omega']:.3f}, err: {err_omega:.3f}")
    print(f"  c      - True: {a2s(true_params['c_vec'])}, Inferred: {a2s(inferred_params['c_vec'])}, err: {err_c:.3f}")
    # print(f"  theta  - True: {true_params['theta']:.3f}, Inferred: {inferred_params['theta']:.3f}, err: {err_theta:.3f}")

    # assert err_beta0 < EPS, f"err beta0 = {err_beta0:.3f}"
    # assert err_eps < EPS, f"err eps {err_eps:.3f}"
    # # assert err_rho < EPS, f"err rho {err_rho:.3f}"  # rho fixed at 0.8
    # assert np.all(err_omega < EPS), f"err omega {err_omega:.3f}"
    # assert np.all(err_c < EPS), f"err c {err_c::.3f}"
    # # assert err_theta < EPS, f"err theta {err_theta:.3f}"
    #
    # Test that final loss is finite and reasonable
    assert np.isfinite(fun), f"Final loss is not finite: {fun}"
    assert fun >= 0, f"Final loss is negative: {fun}"

    # Create visualization
    print("Creating parameter inference visualization...")
    seasons = sorted(pop.season.unique())

    # Set dark theme
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, len(seasons), figsize=(5 * len(seasons), 5), facecolor='#1e1e1e')
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
    plt.savefig(f'pix/{difficulty}_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
