"""
Data loading for real and synthetic epidemic data.

Time convention: November 1st of year Y maps to t = Y.0 (flu season start).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.helper import calc_t
from src.packer import Packer


def load_synthetic(disease, regions, seasons, theta, phase, S_init=None, I_init=None, add_noise=True):
    """
    Generate synthetic epidemic data using Packer.sim().

    Returns:
        obs: DataFrame with columns [t, region, season, season_idx, incidence, ...]
        true_params: dict with theta, S_init, I_init
    """
    packer = Packer(disease=disease, seasons=seasons, regions=regions)

    # Generate initial conditions if not provided
    if S_init is None or I_init is None:
        random_params = packer.random_dict()
        if S_init is None:
            S_init = random_params['S_init']
        if I_init is None:
            I_init = random_params['I_init']

    true_params = {'theta': theta, 'S_init': S_init, 'I_init': I_init}

    # Use Packer.sim to generate trajectory
    obs = packer.sim(true_params, phase, disease)

    if add_noise:
        true_counts = obs['incidence'] * disease.rho
        scale = np.sqrt(disease.rho * (1 - disease.rho) * true_counts)
        obs['incidence'] = true_counts + np.random.randn(len(true_counts)) * scale
        obs['incidence'] = np.maximum(1e-6, obs['incidence'])

    return obs, true_params


def estimate_phase(state, humidity_dir="data/viboud"):
    """Estimate phase offset from absolute humidity data."""
    state_file = state.replace(" ", "_")
    fname = f"{humidity_dir}/{state_file}.csv"

    try:
        df = pd.read_csv(fname, parse_dates=["time"])
    except FileNotFoundError:
        return 0.0

    df["t"] = calc_t(df["time"])
    t = df["t"].values
    ah = -df["AH"].values

    def neg_correlation(phi):
        signal = np.sin(2 * np.pi * t + phi)
        return -np.corrcoef(signal, ah)[0, 1]

    result = minimize_scalar(neg_correlation, bounds=(0, 2 * np.pi), method="bounded")
    return result.x


def load_real(disease, regions, seasons, mortality_path="data/pni_mortality/output.csv", humidity_dir="data/viboud"):
    """
    Load real epidemic data from CSV.

    Returns:
        obs: DataFrame with columns [t, region, season, season_idx, incidence]
        phase: array of phase offsets per region
    """
    # Load mortality data
    df = pd.read_csv(mortality_path, parse_dates=["date"])
    df = df[df["state"].isin(regions)].sort_values("date")

    # Compute phases
    phase = np.array([estimate_phase(r, humidity_dir) for r in regions])
    mean_phase = np.mean(phase)

    # Convert date to continuous time with Nov 1 = Y.0
    df["t"] = calc_t(df["date"]) + mean_phase / (2 * np.pi)
    df["season"] = np.floor(df["t"]).astype(int)
    df = df[df["season"].isin(seasons)]

    # Compute observed incidence (deaths per capita)
    df["incidence"] = df["deaths"] / df["population"]

    # Build observations
    results = []
    for season_idx, season in enumerate(seasons):
        season_data = df[df["season"] == season]
        for region in regions:
            region_data = season_data[season_data["state"] == region].head(disease.n_weeks).copy()
            region_data["season_idx"] = season_idx
            region_data["region"] = region
            results.append(region_data[["t", "region", "season", "season_idx", "incidence"]])

    obs = pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["t", "region", "season", "season_idx", "incidence"]
    )

    return obs, phase
