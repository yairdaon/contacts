import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

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
    # Test parameters for 4 regions
    pop = makepop(n_regions=4, n_seasons=1)
    S_init = [0.99, 0.98, 0.97, 0.96]
    E_init = [0.005, 0.01, 0.015, 0.02] 
    I_init = [0.005, 0.01, 0.015, 0.02]
    n_weeks = 15

    # Test original Euler method
    start = time.time()
    df_euler, euler_time = run(S_init, E_init, I_init, n_weeks=n_weeks, dt_euler=1e-2, population=pop.population.values)
    total_euler_time = time.time() - start
    df_euler.fillna(0, inplace=True)

    # Test RK method
    start = time.time()
    df_rk, rk_time = run_rk(S_init, E_init, I_init, n_weeks=n_weeks, population=pop.population.values)
    total_rk_time = time.time() - start

    # print(f"Euler method: {euler_time:.4f}s (total: {total_euler_time:.4f}s)")
    # print(f"RK method: {rk_time:.4f}s (total: {total_rk_time:.4f}s)")
    # print(f"Speed ratio: {total_euler_time/total_rk_time:.2f}x")

    # Verify both methods produced results
    assert isinstance(df_euler, pd.DataFrame)
    assert isinstance(df_rk, pd.DataFrame)
    assert len(df_euler) == len(df_rk)
    
    # Test relative error between methods is small
    for region in range(4):
        rk_col = f'C{region}'
        euler_col = f'C{region}'
        
        # Calculate relative error: |rk - euler| / max(|euler|, 1e-10)
        rk_vals = df_rk[rk_col].values
        euler_vals = df_euler[euler_col].values
        
        # Avoid division by zero with small epsilon
        denominator = np.maximum(np.abs(euler_vals), 1e-10)
        relative_error = np.abs(rk_vals - euler_vals) / denominator
        
        max_error = np.max(relative_error)
        print(f"Region {region} max relative error: {max_error:.4f}")
        
        # Allow EPS maximum relative error
        assert max_error <= EPS, f"Region {region} relative error {max_error:.4f} exceeds 10%"

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


def test_integration_inverter():
    """Test that both integration methods work in Inverter class."""
    pop = makepop(n_regions=2, n_seasons=2)
    
    # Test RK integration (primary)
    inv_rk = Inverter(population=pop, n_weeks=15, integration='rk')
    x0 = inv_rk.packer.random_vector(seed=2)

    result_rk = inv_rk.sim(x0)
    rk_runtime = inv_rk.run_time
    
    # Test Euler integration (for comparison)
    inv_euler = Inverter(population=pop, n_weeks=15, integration='euler')
    result_euler = inv_euler.sim(x0)#.fillna(0)
    euler_runtime = inv_euler.run_time
    
    # Verify both methods work
    assert isinstance(result_euler, pd.DataFrame)
    assert isinstance(result_rk, pd.DataFrame)
    assert len(result_euler) == len(result_rk)
    assert set(result_euler.columns) == set(result_rk.columns)
    assert np.all(result_euler.set_index(["season", "region", "time"]).index == result_euler.set_index(["season", "region", "time"]).index)


    print(f"Inverter integration timing - Euler: {euler_runtime:.4f}s, RK: {rk_runtime:.4f}s")
    
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

    denominator = np.maximum(np.abs(result_euler.incidence.values), 1e-9)
    err = np.max(np.abs(result_euler.incidence.values - result_rk.incidence.values) / denominator)

    re = result_euler.query("season == '1991-01-01' and region == 'HHS1'")
    rr = result_rk.query("season == '1991-01-01' and region == 'HHS1'")
    assert err < EPS