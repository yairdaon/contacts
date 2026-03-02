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

crlb_res_dir = "outputs"
if os.path.exists(crlb_res_dir):
    for png_file in glob.glob(os.path.join(crlb_res_dir, "*.png")):
        os.remove(png_file)
        print(f"Removed {png_file}")
else:
    os.makedirs(crlb_res_dir)
    print(f"Created directory {crlb_res_dir}")
    

def compute_one_crlb(model,
                     theta,
                     eps,
                     gamma,
                     beta0,
                     rho,
                     T,
                     period,
                     pop_size,
                     phase=0.0,
                     phase2=None):
    """Compute CRLB for one parameter combination."""
    # Random initial conditions
    log_base = np.random.uniform(3, 6)
    diffs = np.random.uniform(-1, 1, size=2)
    I0 = 10**(-log_base + diffs)
    S0 = 1-I0#np.random.uniform(0.90, 1-I0)

    # Compute CRLB
    crlb = compute_crlb(
        S0=S0,
        I0=I0,
        model=model,
        gamma=gamma,
        theta=theta,
        T=T,
        beta0=beta0,
        eps=eps,
        rho=rho,
        period=period,
        pop_size=pop_size,
        phase=phase,
        phase2=phase2
    )

    # Store result
    result = {
        'model': model,
        'theta': theta,
        'gamma': gamma,
        'eps': eps,
        'beta0': beta0,
        'S1_0': S0[0],
        'S2_0': S0[1],
        'I1_0': I0[0],
        'I2_0': I0[1],
        'T': T,
        'period': period,
        'crlb': crlb,
        'pop_size': pop_size,
        'beta0': beta0,
        'phase': phase,
        'phase2': phase2
    }

    return result


@plac.annotations(
    n_runs=('Number of runs per parameter combination', 'option', 'r', int, None, 'Number of runs'),
    n_jobs=('Number of parallel jobs', 'option', 'j', int, -1, 'Number of jobs')
)
def main(n_runs=200, n_jobs=-1, cross=False):
    """CRLB analysis for epidemic connectivity."""

    # From keeling and rohani's boarding school influenza example
    beta0 = flu.beta0

    # Recovery rate from calibration to continuous model
    gamma = flu.gamma # 7 days == one week

    T = 25
    period = 53
    pop_size = 2e7
    N = 20 
    thetas = 10 ** np.linspace(-4, -1, N, endpoint=True)
    epss = np.linspace(0, 1, N, endpoint=False)
    tasks = []
    for model in ['contacts']:
        for eps in epss:#np.arange(11) * 0.1:
            for phase2 in [0, np.pi]:
                for run_idx in range(n_runs):
                    for theta in thetas:
                        tasks.append({
                            'model': model,
                            'theta': theta,
                            'eps': eps,
                            'gamma': gamma,
                            'beta0': beta0,
                            'T': T,
                            'period': period,
                            'pop_size': pop_size,
                            'phase': 0,
                            'phase2': phase2,
                            'rho': flu.rho
                        })


    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_crlb)(**task) for task in tqdm(tasks)
    )
    
    df = pd.DataFrame(results)
    df['log_crlb'] = np.log10(df.crlb)
    df.to_csv(f"outputs/crlb.csv")
   
if __name__ == "__main__":
    try:
        plac.call(main)
    except:
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        
