"""
Data loading for real and synthetic epidemic data.

Time convention: November 1st of year Y maps to t = Y.0 (flu season start).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.packer import Packer

YEAR_LENGTH = 365.25

def calc_t(date):
    """Convert datetime to continuous time where November 1st = integer.
    61 because november is 30 days and december is 31 days.
    """
    shifted = date + pd.Timedelta(days=61)
    return shifted.dt.year + (shifted.dt.dayofyear-1) / YEAR_LENGTH

def t_to_date(t):
    """Inverse of calc_t: convert continuous time back to datetime."""
    year = int(t)
    frac = t - year
    day_of_year = int(frac * YEAR_LENGTH) + 1
    dt = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day_of_year - 1)
    return dt - pd.Timedelta(days=61)

## We can check with
##  print(calc_t(pd.Series([pd.to_datetime("2009-11-01")])).iloc[0])
## prinnts 2010.0

def load_synthetic(disease, regions, seasons, theta, phase, populations, S_init=None, I_init=None, add_noise=True):
    """
    Generate synthetic epidemic data using Packer.sim().

    Returns:
        obs: DataFrame with columns [t, region, season, season_idx, incidence, ...]
        true_params: dict with theta, S_init, I_init
    """
    packer = Packer(disease=disease, seasons=seasons, regions=regions, populations=populations)

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

    mu = obs['mu']
    scale = np.sqrt(disease.rho * (1 - disease.rho) * mu) * int(add_noise)
    obs['incidence'] = mu * disease.rho + np.random.randn(len(mu)) * scale
    # obs['incidence'] = np.maximum(1e-6, obs['incidence'])

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


def load_national_driver(mortality_path, pop_path, seasons, rho, n_weeks,
                         exclude_regions=None):
    """
    Load per-capita national P&I mortality-derived infection-rate estimate
    for each season, aligned with the flu-season week grid.

    Returns:
      dict {season -> np.array of length n_weeks}
        with entries I_nat(t) / (rho * N_nat_season), i.e. the per-capita
        national incidence rate. This is the exogenous national driver term
        added to the FOI in the extended model.

    exclude_regions: optional list of state NAMES to leave out of the
        national aggregate (both numerator and denominator). If None, use
        all states.
    """
    pop_df = pd.read_csv(pop_path, parse_dates=["date"])
    pop_df["season"] = pop_df["date"].dt.year + 1

    mort_df = pd.read_csv(mortality_path, parse_dates=["date"])
    if exclude_regions:
        exclude = set(exclude_regions)
        mort_df = mort_df[~mort_df["state"].isin(exclude)]
        pop_df  = pop_df[~pop_df["state"].isin(exclude)]

    mort_df["t"] = calc_t(mort_df["date"])
    mort_df["season"] = np.floor(mort_df["t"]).astype(int)

    result = {}
    for season in seasons:
        pop_season = pop_df[pop_df["season"] == season]["population"].sum()
        if pop_season <= 0:
            continue
        agg = (mort_df[mort_df["season"] == season]
               .groupby("t")["deaths"].sum().sort_index())
        agg = agg.iloc[:n_weeks]
        if len(agg) < n_weeks:
            continue
        result[season] = agg.values / (rho * pop_season)
    return result


def load_real(disease, regions, seasons, mortality_path="data/pni_mortality/deaths.csv", humidity_dir="data/viboud"):
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

    # Convert date to continuous time with Nov 1 = Y.0
    df["t"] = calc_t(df["date"])
    df["season"] = np.floor(df["t"]).astype(int)
    df = df[df["season"].isin(seasons)]

    # Keep incidence as raw death counts
    df["incidence"] = df["deaths"]

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
