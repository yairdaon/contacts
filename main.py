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
from src import flu

crlb_res_dir = "crlb_res"
if os.path.exists(crlb_res_dir):
    for png_file in glob.glob(os.path.join(crlb_res_dir, "*.png")):
        os.remove(png_file)
        print(f"Removed {png_file}")
else:
    os.makedirs(crlb_res_dir)
    print(f"Created directory {crlb_res_dir}")
    

def compute_one_crlb(theta,
                     amplitude,
                     gamma,
                     R0,
                     T,
                     period,
                     pop_size,
                     phase=0.0,
                     phase2=None):
    """Compute CRLB for one parameter combination."""

    # Calculate beta0 from R0 (measured at peak transmission)
    # beta0 = R0 * (1 - exp(-gamma)) / (1 + amplitude)
    # where (1 - exp(-gamma)) is the recovery probability per week
    beta0 = R0 * (1 - np.exp(-gamma)) / (1 + amplitude)

    # Random initial conditions
    log_base = np.random.uniform(3, 6)
    diffs = np.random.uniform(-0.5, 0.5, size=2)
    I0 = 10**(-log_base + diffs)
    S0 = 1-I0#np.random.uniform(0.90, 1-I0)

    # Compute CRLB
    crlb = compute_crlb(
        S0=S0,
        I0=I0,
        gamma=gamma,
        theta=theta,
        T=T,
        beta0=beta0,
        amplitude=amplitude,
        period=period,
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
        'period': period,
        'crlb': crlb,
        'pop_size': pop_size,
        'R0': R0,
        'phase': phase,
        'phase2': phase2
    }

    return result


@plac.annotations(
    n_runs=('Number of runs per parameter combination', 'option', 'r', int, None, 'Number of runs'),
    n_jobs=('Number of parallel jobs', 'option', 'j', int, -1, 'Number of jobs')
)
def main(n_runs=1000, n_jobs=-1):
    """CRLB analysis for epidemic connectivity."""

    # From keeling and rohani's boarding school influenza example
    R0 = flu.R0

    # Recovery rate from calibration to continuous model
    gamma = flu.gamma # 7 days == one week

    T = 25
    period = 53
    pop_size = 2e7
    N = 20
    thetas = 10 ** np.linspace(-4, -1, N, endpoint=True)
    amplitudes = np.linspace(0, 1, N, endpoint=False)
    tasks = []
    for theta in thetas: #[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        for amplitude in amplitudes:#np.arange(11) * 0.1:
            for phase2 in [0, np.pi]:
                for run_idx in range(n_runs):

                    tasks.append({
                        'theta': theta,
                        'amplitude': amplitude,
                        'gamma': gamma,
                        'R0': R0,
                        'T': T,
                        'period': period,
                        'pop_size': pop_size,
                        'phase': 0,
                        'phase2': phase2  # Set to None for synchronized regions
                    })
                    #task_idx += 1


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
        
