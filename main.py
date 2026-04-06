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


def compute_one_crlb(theta, delta, gamma, beta0, rho, Ts, phase):
    """Compute CRLB for one parameter combination with random initial conditions."""
    N = np.array([N_POP, N_POP])

    log_base = np.random.uniform(3, 6)
    diffs = np.random.uniform(-1, 1, size=2)
    I0 = 10**(-log_base + diffs) * N
    S0 = (1 - 10**(-log_base + diffs)) * N

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
        'phase2': phase[1]
    }


def main():
    disease = flu.Mortality

    N = 200
    thetas = 10 ** np.linspace(-4, -1, N, endpoint=True)
    deltas = np.linspace(0, 1, N, endpoint=False)

    # Time array for one season.  2000 is an arbitrary reference season
    Ts = 4000 + np.arange(disease.n_weeks) * disease.step_size


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
                    'phase': np.array([0, phase2])
                })

    tasks = tasks  * 1000
    results = Parallel(n_jobs=-3)(
        delayed(compute_one_crlb)(**task) for task in tqdm(tasks)
    )

    df = pd.DataFrame(results)
    df['log_crlb'] = np.log10(df.crlb)
    df.to_csv(f"{OUTPUT_DIR}/crlb.csv", index=False)
    print(f"Saved {len(df)} results to {OUTPUT_DIR}/crlb.csv")


if __name__ == "__main__":
    main()
    # try:
    #     plac.call(main)
    # except:
    #     _, _, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)
