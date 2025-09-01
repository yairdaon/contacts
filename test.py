import numpy as np
from numba import jit, njit
from numba.core import types
from numpy import sin, cos, pi, log, exp, sqrt, ceil
from matplotlib import pyplot as plt

from multi import *
def main():
    # Set dark mode style
    plt.style.use('dark_background')
       
    # Run simulation for 5 years with random initial conditions
    df = simulate()  
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor('black')
    fig.suptitle('Two-Strain SEIR Model Simulation', fontsize=14, color='white')
    col1 ='#00FF7F'
    col2 = '#DA70D6'
    
    
    # Plot cases (incidence)
    ax = axes[0]
    ax.plot(df.index, df.C0, col1, label='Incidence 1', alpha=0.8, linewidth=2)  # Deep pink
    ax.plot(df.index, df.C1, col2, label='Incidence 2', alpha=0.8, linewidth=2)  # Dark orange
    ax.set_ylabel('Cases per Week', color='white')
    ax.legend()
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('black')
    # Plot transmission rates (seasonal forcing)
    ax = axes[1]
    ax.plot(df.index, df.F0, col1, label='β1(t)', alpha=0.8, linewidth=2)  # Spring green
    ax.plot(df.index, df.F1, col2, label='β2(t)', alpha=0.8, linewidth=2)  # Orchid
    ax.set_ylabel('β(t)', color='white')
    ax.set_xlabel('Time', color='white')
    ax.legend()
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('black')
    
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb
        traceback.print_exc()  # Prints the full stack trace to stderr
        pdb.post_mortem()      # Starts debugger at the poi
