import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pytest

from src.multi import run
from src.rk import run_rk
from src.helper import makepop
from src.inverter import Inverter

matplotlib.use('Agg')  # Non-interactive backend for testing
EPS=0.07

def test_euler_vs_rk():
    """
    Test comparing Euler vs RK methods with visualization.
    """
    pop = makepop(n_regions=4, n_seasons=1).population.values
    S_init = [0.19, 0.68, 0.27, 0.96]
    E_init = [0.005, 0.01, 0.015, 0.02] 
    I_init = [0.005, 0.01, 0.015, 0.02]
    n_weeks = 15


    df_euler = run(S_init, E_init, I_init, n_weeks=n_weeks, dt_step=1e-2, population=pop)
    df_euler.fillna(0, inplace=True)
    df_rk = run_rk(S_init, E_init, I_init, n_weeks=n_weeks, dt_output=7, population=pop)

    # Verify both methods produced results
    assert isinstance(df_euler, pd.DataFrame)
    assert isinstance(df_rk, pd.DataFrame)
    assert df_euler.shape == df_rk.shape

    # Create comparison visualization
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor='#1e1e1e')
    fig.suptitle('Euler vs RK Methods: 4 Regions Comparison', fontsize=16, color='white')

    colors = ['#00D4FF', '#FF6B9D', '#00FF7F', '#FFD700']  # Bright colors for dark background
    region_names = ['Region 0', 'Region 1', 'Region 2', 'Region 3']

    # Plot 1: New Cases for all regions
    ax = axes[0, 0]
    for region in range(4):
        rk_col = f'C{region}'
        euler_col = f'C{region}'
        
        ax.plot(df_rk.index, df_rk[rk_col], 
               color=colors[region], linewidth=2, 
               label=f'RK - {region_names[region]}')
        
        ax.plot(df_euler.index, df_euler[euler_col], 
               color=colors[region], linestyle='--', linewidth=2, alpha=0.8,
               label=f'Euler - {region_names[region]}')

    ax.set_title('New Cases (C) - All Regions', color='white')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('New Cases', color='white')
    ax.legend(fancybox=True, shadow=True, facecolor='#2e2e2e', edgecolor='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')

    # Plot 2: Infected for all regions
    ax = axes[0, 1]
    for region in range(4):
        rk_col = f'I{region}'
        euler_col = f'I{region}'
        
        ax.plot(df_rk.index, df_rk[rk_col], 
               color=colors[region], linewidth=2, 
               label=f'RK - {region_names[region]}')
        
        ax.plot(df_euler.index, df_euler[euler_col], 
               color=colors[region], linestyle='--', linewidth=2, alpha=0.8,
               label=f'Euler - {region_names[region]}')

    ax.set_title('Infected (I) - All Regions', color='white')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Infected', color='white')
    ax.legend(fancybox=True, shadow=True, facecolor='#2e2e2e', edgecolor='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')

    # Plot 3: Susceptible for all regions
    ax = axes[1, 0]
    for region in range(4):
        rk_col = f'S{region}'
        euler_col = f'S{region}'
        
        ax.plot(df_rk.index, df_rk[rk_col], 
               color=colors[region], linewidth=2, 
               label=f'RK - {region_names[region]}')
        
        ax.plot(df_euler.index, df_euler[euler_col], 
               color=colors[region], linestyle='--', linewidth=2, alpha=0.8,
               label=f'Euler - {region_names[region]}')

    ax.set_title('Susceptible (S) - All Regions', color='white')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Susceptible', color='white')
    ax.legend(fancybox=True, shadow=True, facecolor='#2e2e2e', edgecolor='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')

    # Plot 4: Exposed for all regions
    ax = axes[1, 1]
    for region in range(4):
        rk_col = f'E{region}'
        euler_col = f'E{region}'
        
        ax.plot(df_rk.index, df_rk[rk_col], 
               color=colors[region], linewidth=2, 
               label=f'RK - {region_names[region]}')
        
        ax.plot(df_euler.index, df_euler[euler_col], 
               color=colors[region], linestyle='--', linewidth=2, alpha=0.8,
               label=f'Euler - {region_names[region]}')

    ax.set_title('Exposed (E) - All Regions', color='white')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Exposed', color='white')
    ax.legend(fancybox=True, shadow=True, facecolor='#2e2e2e', edgecolor='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig('pix/euler_vs_rk.png', dpi=300, bbox_inches='tight')
    err = np.max(np.abs(df_euler - df_rk) / (df_euler + 1e-15))
    assert err < EPS, f"rel err {err}"


@pytest.mark.skip(reason="Integration method comparison in Inverter needs debugging")
def test_integration_inverter():
    """Test that both integration methods work in Inverter class."""
    pop = makepop(n_regions=2, n_seasons=1)
    
    # Test RK integration (primary)
    inv_rk = Inverter(population=pop, n_weeks=15, integration='rk', dt_output=2)
    x0 = inv_rk.packer.random_vector(seed=2)
    result_rk = inv_rk.sim(x0)

    # Test Euler integration (for comparison)
    inv_euler = Inverter(population=pop, n_weeks=15, integration='euler', dt_output=2)
    x0 = inv_euler.packer.random_vector(seed=2)
    result_euler = inv_euler.sim(x0)#.fillna(0)

    # Verify both methods work
    assert isinstance(result_euler, pd.DataFrame)
    assert isinstance(result_rk, pd.DataFrame)
    assert len(result_euler) == len(result_rk)
    assert set(result_euler.columns) == set(result_rk.columns)
    assert np.all(result_euler.set_index(["season", "region", "time"]).index == result_euler.set_index(["season", "region", "time"]).index)

    # Create simple comparison plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1e1e1e')
    fig.suptitle('Integration Methods in Inverter Class', fontsize=14, color='white')
    
    seasons = sorted(pop.season.unique())
    colors = ['#00D4FF', '#FF6B9D']
    
    for season_idx, season in enumerate(seasons):
        ax = axes[season_idx]
        
        # Filter data for this season
        euler_season = result_euler[result_euler.season == season]
        rk_season = result_rk[result_rk.season == season]
        
        for region_idx, region in enumerate(['HHS0', 'HHS1']):
            euler_region = euler_season[euler_season.region == region].sort_values('time')
            rk_region = rk_season[rk_season.region == region].sort_values('time')


            ax.plot(euler_region.time, euler_region.incidence,
                   color=colors[region_idx], linewidth=2,
                   label=f'{region} - Euler' if season_idx == 0 else "")
            
            ax.plot(rk_region.time, rk_region.incidence,
                   color=colors[region_idx], linewidth=2, linestyle='--', alpha=0.8,
                   label=f'{region} - RK' if season_idx == 0 else "")
        
        ax.set_title(f'Season {season}', color='white')
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Incidence', color='white')
        if season_idx == 0:
            ax.legend(fancybox=True, shadow=True, facecolor='#2e2e2e', edgecolor='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig('pix/inverter_integration_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    re = result_euler.fillna(0).incidence.values
    rr = result_rk.incidence.values
    err = np.max(np.abs(re - rr) / (re + 1e-14))
    assert err < EPS
