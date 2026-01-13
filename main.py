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

from src.crlb import compute_crlb


@plac.annotations(
    n_runs=('Number of runs per parameter combination', 'option', 'r', int, None, 'Number of runs'),
    seed=('Random seed', 'option', 's', int, None, 'Random seed')
)
def main(n_runs=10, seed=None):
    """CRLB analysis for epidemic connectivity."""

    np.random.seed(seed)

    # From keeling and rohani's boarding school influenza example
    R0 = 3.65

    # Recovery rate from calibration to continuous model
    gamma = 1 - np.exp(-7/2.2) # 7 days == one week
    
    T = 25
    period = 53

    ## Only a single sigma cuz it just scales the CRLB
    sigma = 0.01
    
    pop_size = 10**3
    
    results = []
    for noise in ['mult',  'add', 'poisson']:
        for theta in [0, 0.05]:
            for amplitude in [0.3]:
                for cc in tqdm(range(n_runs), desc=f'{noise} noise θ={theta}'):
                    # Calculate beta0 from R0
                    beta0 = R0 * gamma / (1 + amplitude)

                    # Random initial conditions
                    I0 = 10**(-np.random.uniform(3, 7, size=2))  # Small initial outbreaks
                    S0 = np.random.uniform(0.90, 1-I0)

                    # Compute CRLB
                    crlb = compute_crlb(
                        S1_0=S0[0],
                        S2_0=S0[1],
                        I1_0=I0[0],
                        I2_0=I0[1],
                        gamma=gamma,
                        theta=theta,
                        T=T,
                        sigma=sigma,
                        beta0=beta0,
                        amplitude=amplitude,
                        period=period,
                        noise=noise,
                        pop_size=pop_size
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
                        'R0': R0
                    }

                    results.append(result)
    
    df = pd.DataFrame(results)
    #df = df[df.columns[df.nunique() > 1]]
    print(df.dropna().groupby(['theta', 'noise']).crlb.agg(['median', 'mean']))

    IPython.embed()

    

if __name__ == "__main__":
    #try:
    plac.call(main)
    # except:
    #     _, _, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)
        
