import numpy as np
import sys
import traceback
import pdb
import pickle
import uuid
import os
import pandas as pd
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

from src.crlb import compute_crlb


# Create output directory
OUTPUT_DIR = "crlb_res"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def runner(theta: float,
           gamma: float,
           amplitude: float,
           beta0: float, 
           S1_0: float,
           S2_0: float,
           I1_0: float,
           I2_0: float,
           T: int,
           sigma: float,
           period: int = 53,
           **kwargs):
    """Compute CRLB for given parameters."""
    
    try:
        # Compute CRLB
        crlb = compute_crlb(
            S1_0=S1_0,
            S2_0=S2_0,
            I1_0=I1_0,
            I2_0=I2_0,
            gamma=gamma,
            theta=theta,
            T=T,
            sigma=sigma,
            beta0=beta0,
            amplitude=amplitude,
            period=period,
        )
        ex = ''

        
    except Exception as e:
        # If CRLB computation fails, return NaN values
        crlb = np.nan
        ex = str(e)

    result = {
        'theta': theta,
        'gamma': gamma,
        'amplitude': amplitude,
        'beta0': beta0,
        'S1_0': S1_0,
        'S2_0': S2_0,
        'I1_0': I1_0,
        'I2_0': I2_0,
        'T': T,
        'sigma': sigma,
        'period': period,
        'crlb': crlb,
        'exception': ex,
        **kwargs
    }
    # Save result to pickle with random UUID filename
    fname = f"{OUTPUT_DIR}/{uuid.uuid4()}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(result, f)


def main(n_runs=200,
         seed=None):

    np.random.seed(seed)

    # From keeling and rohani's boarding school influenza example
    R0 = 3.65

    # Recovery rate from calibration to continuous model
    gamma = 1 - np.exp(-7/2.2) # 7 days == one week
    
    T = 25
    period = 53

    ## Only a single sigma cuz it just scales the CRLB
    sigma = 0.01 
    
    params = []
    for theta in [0,  0.01]:
        for amplitude in [0, 0.3, 0.8]:
            for _ in range(n_runs):
            
                # Calculate beta0 from R0
                beta0 = R0 * gamma / (1 + amplitude) / (1 + theta)

                # Random initial conditions
                I0 = 10**(-np.random.uniform(3, 7, size=2))  # Small initial outbreaks
                S0 = np.random.uniform(0.90, 1-I0)
        
                pp = {
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
                    'T': T,
                    'period': period,
                    'gamma': gamma,
                    'R0': R0,
                    'noise': 'mult'
                }
        
                params.append(pp)
    
    
    # Parallel execution
    Parallel(n_jobs=-1, verbose=1)(
        delayed(runner)(**pp) for pp in tqdm(params)
    )
       
    # Load all pickle files and create DataFrame
    
    results = []
    for fname in glob.glob(f"{OUTPUT_DIR}/*.pkl"):
        with open(fname, 'rb') as f:
            results.append(pickle.load(f))
    
    df = pd.DataFrame(results)
    
    # Add column for years required for CI to include zero
    # Formula: theta - 1.96 * crlb / sqrt(n) = 0
    # Solving for n: n = (1.96 * crlb / theta)^2
    # Convert from seasons to years (assuming ~25 weeks per season, ~2 seasons per year)

    
    # Save to CSV
    csv_filename = f"{OUTPUT_DIR}/crlb_results.csv"
    df.to_csv(csv_filename, index=False)
    

if __name__ == "__main__":
    try:
        main()
    except:
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
