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


def compute_one_crlb(theta, eps, gamma, beta0, rho, Ts, phase):
    """Compute CRLB for one parameter combination with random initial conditions."""
    # Random initial conditions
    log_base = np.random.uniform(3, 6)
    diffs = np.random.uniform(-1, 1, size=2)
    I0 = 10**(-log_base + diffs)
    S0 = 1 - I0

    crlb = compute_crlb(
        S0=S0,
        I0=I0,
        gamma=gamma,
        theta=theta,
        Ts=Ts,
        beta0=beta0,
        eps=eps,
        rho=rho,
        phase=phase
    )

    return {
        'theta': theta,
        'gamma': gamma,
        'eps': eps,
        'beta0': beta0,
        'S1_0': S0[0],
        'S2_0': S0[1],
        'I1_0': I0[0],
        'I2_0': I0[1],
        'crlb': crlb,
        'phase1': phase[0],
        'phase2': phase[1]
    }


@plac.annotations(
    n_runs=('Number of runs per parameter combination', 'option', 'r', int),
    n_jobs=('Number of parallel jobs', 'option', 'j', int)
)
def main(n_runs=200, n_jobs=-1):
    disease = flu.ILI

    beta0 = disease.beta0
    gamma = disease.gamma
    rho = disease.rho

    # Time array for one season
    season = 2000  # arbitrary reference season
    Ts = season + np.arange(disease.n_weeks) * disease.step_size

    N = 20
    thetas = 10 ** np.linspace(-4, -1, N, endpoint=True)
    epss = np.linspace(0, 1, N, endpoint=False)

    tasks = []
    for eps in epss:
        for phase2 in [0, np.pi]:
            for _ in range(n_runs):
                for theta in thetas:
                    tasks.append({
                        'theta': theta,
                        'eps': eps,
                        'gamma': gamma,
                        'beta0': beta0,
                        'rho': rho,
                        'Ts': Ts,
                        'phase': np.array([0, phase2])
                    })

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_crlb)(**task) for task in tqdm(tasks)
    )

    df = pd.DataFrame(results)
    df['log_crlb'] = np.log10(df.crlb)
    df.to_csv(f"{OUTPUT_DIR}/crlb.csv", index=False)
    print(f"Saved {len(df)} results to {OUTPUT_DIR}/crlb.csv")


if __name__ == "__main__":
    try:
        plac.call(main)
    except:
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
