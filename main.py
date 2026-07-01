import numpy as np
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

from src.crlb import compute_precision
from src import flu

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_POP = 1e6  # Population size per region


def load_empirical_ic_distribution(bads=("AK", "HI", "AZ")):
    """Pool per-season, per-state fitted (S_i(0)/N_i, I_i(0)/N_i)
    from outputs/states/*.csv to form the empirical distribution of
    initial conditions.

    This is the only IC source used by the simulated CRLB analysis;
    the previously-used synthetic "similar" and "different" IC regimes
    have been dropped.
    """
    files = glob("outputs/states/*.csv")
    if not files:
        raise FileNotFoundError(
            "No real-data fit results in outputs/states/*.csv. "
            "Run realdata.py first to produce the empirical IC distribution."
        )
    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    res = res.query("state1 not in @bads and state2 not in @bads")

    S_vals = pd.concat([res["S1_0"], res["S2_0"]]).values
    I_vals = pd.concat([res["I1_0"], res["I2_0"]]).values

    S_vals = S_vals[np.isfinite(S_vals) & (S_vals > 0)]
    I_vals = I_vals[np.isfinite(I_vals) & (I_vals > 0)]
    return S_vals, I_vals


def compute_one_precision(theta, delta, gamma, beta0, rho, Ts, phase, S0_frac, I0_frac):
    """Compute Fisher information for one (theta, delta, phase) with
    pre-sampled IC fractions from the empirical distribution."""
    N = np.array([N_POP, N_POP])
    S0 = S0_frac * N
    I0 = I0_frac * N

    precision = compute_precision(
        S0=S0,
        I0=I0,
        gamma=gamma,
        theta=theta,
        Ts=Ts,
        beta0=beta0,
        delta=delta,
        rho=rho,
        phase=phase,
        N=N,
    )

    return {
        "theta": theta,
        "gamma": gamma,
        "delta": delta,
        "beta0": beta0,
        "S1_0": S0[0],
        "S2_0": S0[1],
        "I1_0": I0[0],
        "I2_0": I0[1],
        "precision": precision,
        "phase1": phase[0],
        "phase2": phase[1],
    }


def build_tasks(disease, thetas, deltas, Ts, S_vals, I_vals, n_reps):
    """Build task list with IC fractions pre-sampled independently per
    task and per region from the empirical distribution."""
    tasks = []
    for _ in range(n_reps):
        for delta in deltas:
            for phase2 in [0, np.pi]:
                for theta in thetas:
                    tasks.append(
                        {
                            "theta": theta,
                            "delta": delta,
                            "gamma": disease.gamma,
                            "beta0": disease.beta0,
                            "rho": disease.rho,
                            "Ts": Ts,
                            "phase": np.array([0, phase2]),
                            "S0_frac": np.random.choice(S_vals, size=2),
                            "I0_frac": np.random.choice(I_vals, size=2),
                        }
                    )
    return tasks


def main():
    disease = flu.Mortality

    S_vals, I_vals = load_empirical_ic_distribution()
    print(
        f"Loaded empirical IC distribution: {len(S_vals)} S fractions, "
        f"{len(I_vals)} I fractions, across all real-data state pairs."
    )

    N = 100
    thetas = 10 ** np.linspace(-4, -1, N, endpoint=True)
    deltas = np.linspace(0, 1, N, endpoint=False)
    Ts = 2000 + np.arange(disease.n_weeks) * disease.step_size
    n_reps = 500

    tasks = build_tasks(disease, thetas, deltas, Ts, S_vals, I_vals, n_reps)
    print(f"Total tasks: {len(tasks)}")

    results = Parallel(n_jobs=-3)(
        delayed(compute_one_precision)(**task) for task in tqdm(tasks)
    )

    df = pd.DataFrame(results)
    df["log_precision"] = np.log10(df.precision)
    df.to_csv(f"{OUTPUT_DIR}/precision.csv", index=False)
    print(f"Saved {len(df)} results to {OUTPUT_DIR}/precision.csv")


if __name__ == "__main__":
    main()
