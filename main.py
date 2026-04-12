import numpy as np
import sys
import traceback
import pdb
import os
import pandas as pd
from tqdm import tqdm
import plac
from joblib import Parallel, delayed

from src.crlb import compute_crlb
from src import flu

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_POP = 1e6  # Population size per region


def compute_one_crlb(theta, delta, gamma, beta0, rho, Ts, phase, ic_regime):
    """Compute CRLB for one parameter combination with random initial conditions.

    ic_regime: 'similar' or 'different'
        'similar' — both regions draw ICs from narrow ranges (slim_similar, ilim_similar)
        'different' — regions draw ICs independently from wide ranges (slim_different, ilim_different)
    """
    N = np.array([N_POP, N_POP])
    disease = flu.Mortality

    if ic_regime == 'similar':
        S0 = np.random.uniform(*disease.slim_similar, size=2) * N
        I0 = np.random.uniform(*disease.ilim_similar, size=2) * N
    else:  # 'different'
        S0 = np.random.uniform(*disease.slim_different, size=2) * N
        I0 = np.random.uniform(*disease.ilim_different, size=2) * N

    crlb = compute_crlb(
        S0=S0,
        I0=I0,
        gamma=gamma,
        theta=theta,
        Ts=Ts,
        beta0=beta0,
        delta=delta,
        rho=rho,
        phase=phase,
        N=N
    )

    return {
        'theta': theta,
        'gamma': gamma,
        'delta': delta,
        'beta0': beta0,
        'S1_0': S0[0],
        'S2_0': S0[1],
        'I1_0': I0[0],
        'I2_0': I0[1],
        'crlb': crlb,
        'phase1': phase[0],
        'phase2': phase[1],
        'ic_regime': ic_regime
    }


def build_tasks(disease, thetas, deltas, Ts, ic_regime, n_reps):
    """Build task list for a given IC regime."""
    tasks = []
    for delta in deltas:
        for phase2 in [0, np.pi]:
            for theta in thetas:
                tasks.append({
                    'theta': theta,
                    'delta': delta,
                    'gamma': disease.gamma,
                    'beta0': disease.beta0,
                    'rho': disease.rho,
                    'Ts': Ts,
                    'phase': np.array([0, phase2]),
                    'ic_regime': ic_regime
                })
    return tasks * n_reps


def main():
    disease = flu.Mortality

    N = 100
    thetas = 10 ** np.linspace(-4, -1, N, endpoint=True)
    deltas = np.linspace(0, 1, N, endpoint=False)
    Ts = 2000 + np.arange(disease.n_weeks) * disease.step_size
    n_reps = 500

    # Similar ICs (the standard case — synchronized epidemics with similar starting points)
    tasks_sim = build_tasks(disease, thetas, deltas, Ts, 'similar', n_reps)

    # Different ICs (the symmetry-broken case)
    tasks_dif = build_tasks(disease, thetas, deltas, Ts, 'different', n_reps)

    all_tasks = tasks_sim + tasks_dif
    print(f"Total tasks: {len(all_tasks)} ({len(tasks_sim)} similar + {len(tasks_dif)} different)")

    results = Parallel(n_jobs=-3)(
        delayed(compute_one_crlb)(**task) for task in tqdm(all_tasks)
    )

    df = pd.DataFrame(results)
    df['log_crlb'] = np.log10(df.crlb)
    df.to_csv(f"{OUTPUT_DIR}/crlb.csv", index=False)
    print(f"Saved {len(df)} results to {OUTPUT_DIR}/crlb.csv")


if __name__ == "__main__":
    main()
