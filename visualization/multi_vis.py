import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi import run
from packer import Packer

matplotlib.use("MacOSX")


def run_simulation(initial_scale=1.0,
                   seed=43):
    """
    Run a single SEIR simulation with specified parameters.
    
    Parameters:
    -----------
    initial_scale : float
        Scale factor for E_init and I_init (1.0 = normal, 0.1 = low, 10 = high)
    """
    packer = Packer(regions=["HHS1", "HHS2"], seasons=["1900-01-01"])
    params = packer.random_dict(seed=seed)

    # Set up population sizes
    total_pop = np.array([1e5, 2e5])
    
    # Create initial conditions with scaling (as fractions). No resistant / recovered
    E_init = params['E_init'][0, :] * initial_scale
    I_init = params['I_init'][0, :] * initial_scale
    S_init = 1 - E_init - I_init
    contact_matrix = packer.c_vec_to_mat(params['c_vec'])

    # Run simulation with fractions
    df = run(S_init=S_init,
             I_init=I_init,
             E_init=E_init,
             beta0=params['beta0'],
             eps=params['eps'],
             omega=params['omega'],
             contact_matrix=contact_matrix,
             population=total_pop
             )

    return df, params

def plot_scenarios(scenarios, title="SEIR Model Comparison"):
    """
    Plot multiple simulation scenarios in a grid.
    
    Parameters:
    -----------
    scenarios : list of dict
        Each dict should have 'name', 'data' (DataFrame), and optionally 'params'
    """
    n_scenarios = len(scenarios)
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, n_scenarios, figsize=(6*n_scenarios, 6))
    if n_scenarios == 1:
        axes = [axes]
        
    fig.patch.set_facecolor('black')
    fig.suptitle(title, fontsize=16, color='white')
    
    colors = ['#00FF7F', '#DA70D6']  # Green, Magenta
    
    for i, scenario in enumerate(scenarios):
        df = scenario['data']
        ax = axes[i]
        
        # Plot incidence
        ax.plot(df.index, df.C0, colors[0], label='Region 0', alpha=0.8, linewidth=2)
        ax.plot(df.index, df.C1, colors[1], label='Region 1', alpha=0.8, linewidth=2)
        
        ax.set_title(scenario['name'], color='white', fontsize=12)
        ax.set_xlabel('Time (days)', color='white')
        if i == 0:
            ax.set_ylabel('Cases per Day', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_facecolor('black')
        
        # Print stats
        total_cases = (df.C0 + df.C1).sum()
        peak_cases = (df.C0 + df.C1).max()
        print(f"{scenario['name']}: Peak={peak_cases:.1f}, Total={total_cases:.0f}")
    
    plt.tight_layout()
    plt.show()


def compare_initial_conditions():
    """Compare epidemics with different initial condition scales."""
    print("Comparing initial conditions with epidemic-capable parameters...")
    
    scenarios = []
    scales = [0.1, 1.0, 10.0]
    names = ["Low Initial (0.1x)", "Normal Initial (1x)", "High Initial (10x)"]
    
    for scale, name in zip(scales, names):
        df, params = run_simulation(initial_scale=scale, 
                                    seed=43)
        scenarios.append({'name': name, 'data': df, 'params': params})
    
    plot_scenarios(scenarios, "Effect of Initial Conditions")


def compare_transmission_rates():
    """Compare epidemics with different transmission rates."""
    print("Comparing transmission rates...")
    
    scenarios = []
    betas = [0.3, 0.6, 0.9]
    names = [f"β₀={beta}" for beta in betas]
    
    for beta, name in zip(betas, names):
        df, params = run_simulation(seed=43)
        scenarios.append({'name': name, 'data': df, 'params': params})
    
    plot_scenarios(scenarios, "Effect of Transmission Rate")




if __name__ == "__main__":
    try:
        print("Starting SEIR comparison visualization...")
        
        # Run the initial conditions comparison by default
        compare_initial_conditions()
        compare_transmission_rates()
        # Uncomment to run other comparisons:
        # compare_transmission_rates()
        # compare_seasonality()
        
        print("Visualization complete!")
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        pdb.post_mortem()