"""
Load and process real epidemic data from Viboud et al.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


YEAR_LENGTH = 365.25


def main(state='California'):


    fname = f"data/viboud/{state}.csv"
    df = pd.read_csv(fname, parse_dates=["time"])
    df["t"] = df["time"].dt.year + (df["time"].dt.dayofyear - 1) / YEAR_LENGTH

    t = df["t"].values
    ah = df["AH"].values

    def neg_correlation(phi):
        signal = np.sin(2 * np.pi * t + phi)
        return -np.corrcoef(signal, ah)[0, 1]

    result = minimize_scalar(neg_correlation, bounds=(0, 2 * np.pi), method="bounded")
    phi_opt = result.x
    max_corr = -result.fun

    print(f"Optimal phi: {phi_opt:.4f} radians ({np.degrees(phi_opt):.2f} degrees)")
    print(f"Maximum correlation: {max_corr:.4f}")

    sinusoid = np.sin(2 * np.pi * t + phi_opt)
        
    # Scale sinusoid to match AH range (sinusoid has std = 1/sqrt(2), not 1)
    ah_mean, ah_std = ah.mean(), ah.std()
    sinusoid_scaled = ah_mean + ah_std * (sinusoid / sinusoid.std())

    
    # Plot only 3 years (156 weeks)
    n_samples = 5*52
    t_plot = t[200:200+n_samples]
    ah_plot = ah[200:200+n_samples]
    sin_plot = sinusoid_scaled[200:200+n_samples]

    plt.figure(figsize=(12, 5))
    plt.scatter(t_plot, ah_plot, label="AH", alpha=0.7, color='r')
    plt.plot(t_plot, sin_plot, label=f"sin(2πt + {phi_opt:.2f})", alpha=0.7, color='b')
    plt.xlabel("Time (years)")
    plt.ylabel("Absolute Humidity")
    plt.title(f"{state}: AH vs optimal sinusoid (corr = {max_corr:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"res/{state}_ah_sinusoid.png", dpi=150)
    plt.show()



    beta = beta0 * (1-eps * np.si(2 * np.pi * t + phi_opt))

if __name__ == "__main__":
    #main('California')
    main('New_York')
    
