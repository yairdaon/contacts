import numpy as np
import sys
import traceback
import pdb
import os
import pandas as pd
import glob
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import IPython
import plac
from joblib import Parallel, delayed

from src.crlb import compute_crlb

crlb_res_dir = "crlb_res"
if os.path.exists(crlb_res_dir):
    for png_file in glob.glob(os.path.join(crlb_res_dir, "*.png")):
        os.remove(png_file)
        print(f"Removed {png_file}")
else:
    os.makedirs(crlb_res_dir)
    print(f"Created directory {crlb_res_dir}")
    

def compute_one_crlb(noise, theta, amplitude, gamma, R0, T, period,
                     sigma, pop_size, seed, phase=0.0, phase2=None):
    """Compute CRLB for one parameter combination."""
    # Set seed for reproducibility (unique per job)
    if seed is not None:
        np.random.seed(seed)

    # Calculate beta0 from R0 (measured at peak transmission)
    # beta0 = R0 * (1 - exp(-gamma)) / (1 + amplitude)
    # where (1 - exp(-gamma)) is the recovery probability per week
    beta0 = R0 * (1 - np.exp(-gamma)) / (1 + amplitude)

    # Random initial conditions
    I0 = 10**(-np.random.uniform(3, 4, size=2))  # Small initial outbreaks
    S0 = np.random.uniform(0.90, 1-I0)

    # Compute CRLB
    crlb = compute_crlb(
        S0=S0,
        I0=I0,
        gamma=gamma,
        theta=theta,
        T=T,
        sigma=sigma,
        beta0=beta0,
        amplitude=amplitude,
        period=period,
        noise=noise,
        pop_size=pop_size,
        phase=phase,
        phase2=phase2
    )

    # Store result
    result = {
        'theta': theta,
        'gamma': gamma,
        'amplitude': amplitude,
        'beta0': beta0,
        'S1_0': S0[0],
        'S2_0': S0[1],
        'I1_0': I0[0],
        'I2_0': I0[1],
        'T': T,
        'sigma': sigma,
        'period': period,
        'crlb': crlb,
        'noise': noise,
        'pop_size': pop_size,
        'R0': R0,
        'phase': phase,
        'phase2': phase2
    }

    return result


@plac.annotations(
    n_runs=('Number of runs per parameter combination', 'option', 'r', int, None, 'Number of runs'),
    seed=('Random seed', 'option', 's', int, None, 'Random seed'),
    n_jobs=('Number of parallel jobs', 'option', 'j', int, -1, 'Number of jobs')
)
def main(n_runs=3000, seed=None, n_jobs=-1):
    """CRLB analysis for epidemic connectivity."""

    np.random.seed(seed)

    # From keeling and rohani's boarding school influenza example
    R0 = 3.65

    # Recovery rate from calibration to continuous model
    gamma = 7/2.2 # 7 days == one week

    T = 25
    period = 53

    ## Only a single sigma cuz it just scales the CRLB
    sigma = 0.01

    pop_size = 10**3   

    tasks = []
    task_idx = 0
    for noise in ['bin']:#, 'mult',  'add', 'poisson']:
        for theta in [0, 5e-2]:
            for amplitude in [0, 0.7]:
                for phase in [0, np.pi]:
                    for run_idx in range(n_runs):
                        # Create unique seed for each task
                        seed_offset = seed + task_idx if seed is not None else None

                        tasks.append({
                            'noise': noise,
                            'theta': theta,
                            'amplitude': amplitude,
                            'gamma': gamma,
                            'R0': R0,
                            'T': T,
                            'period': period,
                            'sigma': sigma,
                            'pop_size': pop_size,
                            'seed': seed_offset,
                            'phase': phase,
                            'phase2': None  # Set to None for synchronized regions
                        })
                        task_idx += 1


    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_crlb)(**task) for task in tqdm(tasks)
    )
    
    df = pd.DataFrame(results)
    df['log_crlb'] = np.log10(df.crlb)
    df.to_csv("crlb.csv")
    

if __name__ == "__main__":
    try:
        plac.call(main)
    except:
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        
