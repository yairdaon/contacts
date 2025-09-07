import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
from itertools import product
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverter import Inverter

def visualize_packer_simulation():
    """
    Visualize SEIR simulation using Inverter.sim() with Packer-generated parameters.
    Two seasons, three regions, using seaborn for plotting.
    """
    # Set up population dataframe for Inverter
    seasons = ['2020-01-01', '2021-01-01'] 
    regions = ['Region_A', 'Region_B', 'Region_C']
    
    # Create population dataframe with similar but different sizes
    pop_sizes = [2e6, 8e5, 1.5e6]
    population_data = []
    for season, region in product(seasons, regions):
        region_idx = regions.index(region)
        population_data.append({
            'season': season,
            'region': region, 
            'population': pop_sizes[region_idx]
        })
    
    population_df = pd.DataFrame(population_data)
    print("Population setup:")
    print(population_df)
    
    # Create Inverter instance
    inv = Inverter(population=population_df, n_weeks=26)
    
    # Generate random parameter vector using packer
    x = inv.packer.random_vector(seed=42)
    
    # Print the parameters being used
    params = inv.packer.unpack(x)
    params_pop = inv.packer.real2pop(params.copy())
    print(f"\nUsing parameters:")
    print(f"  beta0: {params_pop['beta0']:.3f}")
    print(f"  eps: {params_pop['eps']:.3f}")
    print(f"  omega: {params_pop['omega']}")
    print(f"  Contact matrix:\n{inv.packer.c_vec_to_mat(params_pop['c_vec'])}")
    
    # Run simulation using Inverter.sim()
    print("\nRunning simulation using Inverter.sim()...")
    results_df = inv.sim(x)
    
    print(f"Results shape: {results_df.shape}")
    print("Results columns:", results_df.columns.tolist())
    print("Sample data:")
    print(results_df.head())
    
    # Create 2x3 subplot layout 
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    
    # Plot each region in separate column, seasons in rows
    for season_idx, season in enumerate(seasons):
        for region_idx, region in enumerate(regions):
            ax = axes[season_idx, region_idx]
            
            # Filter data for this season and region
            subset = results_df[
                (results_df['season'] == season) & 
                (results_df['region'] == region)
            ].copy()
            
            # Add week number for x-axis
            subset = subset.sort_values('time').reset_index(drop=True)
            subset['week'] = range(len(subset))
            
            # Plot using seaborn
            sns.lineplot(
                data=subset, 
                x='week', 
                y='incidence',
                ax=ax,
                linewidth=2.5,
                color=sns.color_palette("husl", len(regions))[region_idx]
            )
            
            ax.set_title(f'{region} - {season[:4]}')
            ax.set_xlabel('Week')
            if region_idx == 0:  # Only leftmost plots get y-label
                ax.set_ylabel('Weekly Incidence')
            else:
                ax.set_ylabel('')
            ax.grid(True, alpha=0.3)
    
    # Overall title and layout
    plt.suptitle(
        f'Multi-Region SEIR Simulations using Inverter.sim()\n' +
        f'β₀={params_pop["beta0"]:.3f}, ε={params_pop["eps"]:.3f}, ' +
        f'Populations: {[f"{p/1e6:.1f}M" for p in pop_sizes]}',
        fontsize=14
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    for season in seasons:
        season_data = results_df[results_df['season'] == season]
        total_by_time = season_data.groupby('time')['incidence'].sum()
        
        print(f"\n{season}:")
        print(f"  Peak weekly incidence: {total_by_time.max():.0f} cases")
        print(f"  Total cases: {total_by_time.sum():.0f} cases")
        
        for region in regions:
            region_data = season_data[season_data['region'] == region]
            region_total = region_data['incidence'].sum()
            print(f"  {region}: {region_total:.0f} cases")
    
    return results_df

if __name__ == "__main__":
    print("Testing Packer random parameters with Inverter.sim()...")
    print("Generating 2 seasons × 3 regions simulation...")
    
    df = visualize_packer_simulation()
    print("\nVisualization complete!")