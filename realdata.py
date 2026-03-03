"""
Load and process real epidemic data from Viboud et al.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import nlopt

from src import flu
from src.inverter import Objective, Inverter
from src.crlb import compute_crlb
import seaborn as sns

YEAR_LENGTH = 365.25


def calc_t(time):
    """Convert datetime to continuous time where Jan 1 = integer."""
    return time.dt.year + (time.dt.dayofyear - 1) / YEAR_LENGTH


def get_phi(state, plot=False):
    """Find phase shift phi such that sin(2πt + phi) maximally correlates with AH."""
    state_ = state.replace(" ","_")
    fname = f"data/viboud/{state_}.csv"
    df = pd.read_csv(fname, parse_dates=["time"])
    df["t"] = calc_t(df['time'])

    t = df["t"].values
    ah = -df["AH"].values

    def neg_correlation(phi):
        signal = np.sin(2 * np.pi * t + phi)
        return -np.corrcoef(signal, ah)[0, 1]

    result = minimize_scalar(neg_correlation, bounds=(0, 2 * np.pi), method="bounded")
    phi = result.x
    max_corr = -result.fun

    if plot:
        t_shifted = t + phi / (2 * np.pi)
        sinusoid = np.sin(2 * np.pi * t_shifted)

        # Scale sinusoid to match AH range
        ah_mean, ah_std = ah.mean(), ah.std()
        sinusoid_scaled = ah_mean + ah_std * sinusoid / sinusoid.std()

        # Plot 5 years
        n_samples = 5 * 52
        t_plot = t_shifted[200:200 + n_samples]
        ah_plot = ah[200:200 + n_samples]
        sin_plot = sinusoid_scaled[200:200 + n_samples]

        plt.figure(figsize=(12, 5))
        plt.scatter(t_plot, ah_plot, label="AH", alpha=0.7, color='r')
        plt.plot(t_plot, sin_plot, label="sin(2πt)", alpha=0.7, color='b')
        plt.xlabel("Time (years)")
        plt.ylabel("Absolute Humidity")
        plt.title(f"{state}: AH vs optimal sinusoid (corr = {max_corr:.3f})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return phi

    return population_data

def main():

    states = ["California" , "New York"]
    df = pd.read_csv("data/pni_mortality/output.csv", index_col=[1]).query("state in @states").sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df["t"] = calc_t(df["date"])

    phis = np.array([get_phi(state) for state in states])
    print(phis / 2 / np.pi * 360)
    
    df["t"] = df["t"] + np.mean(phis) / 2 / np.pi
    df["season"] = np.floor(df["t"]).astype(int)
    df['incidence'] = df.deaths / flu.ifr / df.population
    df['beta'] = flu.beta0 * (1 + flu.eps * np.sin(2 * np.pi * df['t']))
    df['beta'] = (df.beta - df.beta.mean()) / df.beta.std() * df.incidence.std() + df.incidence.mean() 

    population = []
    for (season, state), data in df.groupby(["season", "state"]):
        dd = data.iloc[:flu.nweeks].copy()
        population.append({
            'season': dd,
            'region': state,
            'population': dd.population.iloc[0]
        })
    


    g = sns.relplot(
        data=df,
        x="date",
        y="incidence",
        hue="season",
        row="state",
        kind="line",
        height=4,
        aspect=2.5,
        palette="tab10",
        legend=False
    )
    for ax, state in zip(g.axes.flat, states):
        state_df = df.query("state == @state")
        ax.plot(state_df["date"], state_df["beta"], color="black", alpha=0.5, linestyle="--", label="β(t)")
        ax.set_ylabel("β(t)", color="black")
        ax.tick_params(axis="y", labelcolor="black")
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    try:
        main()
    except:
        import sys, traceback, pdb
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

