import os
import numpy as np
import pickle
import pytest
import pandas as pd
import plac
import socket
from pprint import pprint

import nlopt

from src.helper import makepop, a2s
from src.inverter import Inverter, Objective
from src.diseases import flu

OUTPUT_DIR = os.path.expanduser("~/contacts/res")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main(sync,
         method='slsqp',
         model='cross'):
    
    theta = 0.05
    phase = np.zeros(2)
    grad = True
    if not sync:
        phase[1] = np.pi
    

    n_regions = 2
    n_seasons = 20
    n0 = 500  # No parallelization in debug mode
    maxeval = None
    if method == 'slsqp':
        optimizer = nlopt.LD_SLSQP
    elif method == 'mma':
        optimizer = nlopt.LD_MMA
    elif method == 'ccsaq':
        optimizer =  nlopt.LD_CCSAQ
    elif method == 'cobyla':
        optimizer = nlopt.LN_COBYLA
    else:
        raise ValueError("Invalid optimizer")

    
    #run = 'full' if full else 'debug'
    #oo = '' if grad else 'out'
    un = 'S' if sync else 'Uns' 
    print(f"\n\n{un}ynchronized with {method}")

    pop = makepop(n_regions=n_regions,
                  n_seasons=n_seasons)
    objective = Objective(model=model,
                          population=pop,
                          n_weeks=flu.nweeks,
                          gamma=flu.gamma,
                          beta0 = flu.beta0,
                          amplitude=flu.amplitude,
                          rho=flu.rho,
                          phase=phase)
    
    true = objective.packer.random_dict()
    true['theta'] = theta
    true_trajectory = objective.sim(true)

    # Generate observed data
    obs = true_trajectory.copy()
    true_counts = true_trajectory['incidence'] * flu.rho  # Fixed rho value
    scale = np.sqrt(flu.rho * (1 - flu.rho) * true_counts)
    obs['incidence'] = true_counts + np.random.randn(true_counts.size) * scale
    obs['incidence'] = np.maximum(1e-6, obs['incidence'])  # Ensure non-negative
    objective.obs = obs

    ## Solve inverse problem
    inv = Inverter(objective=objective, optimizer=optimizer).fit(n0=n0, maxeval=maxeval)

    ## Create CSV dataframe with optimization results
    # hostname = socket.gethostname().split('.')[0]  # short server name (e.g. dml12)
    flag = "" if sync else "un"
    #difficulty = 'full' if full else 'debug'
    #opt_type = 'grad' if grad else 'cobyla'
    fname = f'{OUTPUT_DIR}/{flag}sync_{method}.csv'
    
    # Prepare data list for DataFrame
    data_rows = []
    
    # Add true parameters row (chain_number = -1, step_number = 0)
    true_row = {'chain_number': -1, 'step_number': 0, 'fun': np.nan}
    true_row.update(true)
    
    # Flatten arrays in true_params for CSV storage
    for key, value in true.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 1:  # 1D array (like c_vec)
                for i, v in enumerate(value):
                    true_row[f'{key}_{i}'] = v
            elif value.ndim == 2:  # 2D array (like S_init, E_init, I_init)
                for season_idx in range(value.shape[0]):
                    for region_idx in range(value.shape[1]):
                        true_row[f'{key}_s{season_idx}_r{region_idx}'] = value[season_idx, region_idx]
            true_row.pop(key)  # Remove original array entry
    
    data_rows.append(true_row)
    
    # Add optimization chains
    for chain_idx, res in enumerate(inv.results):
        # Add optimization steps
        for step_idx, (x, obj_val) in enumerate(zip(res['x_list'], res['out_list'])):
            unpacked = inv.packer.unpack(x)
            row = {'chain_number': chain_idx, 'step_number': step_idx, 'fun': obj_val}
            
            # Add scalar parameters
            for key, value in unpacked.items():
                if not isinstance(value, np.ndarray):
                    row[key] = value
                elif isinstance(value, np.ndarray):
                    if value.ndim == 1:  # 1D array (like c_vec)
                        for i, v in enumerate(value):
                            row[f'{key}_{i}'] = v
                    elif value.ndim == 2:  # 2D array (like S_init, E_init, I_init)
                        for season_idx in range(value.shape[0]):
                            for region_idx in range(value.shape[1]):
                                row[f'{key}_s{season_idx}_r{region_idx}'] = value[season_idx, region_idx]
            
            data_rows.append(row)
        
        # Add optimal point (step_number = -1)
        optimal_unpacked = inv.packer.unpack(res['x'])
        opt_row = {'chain_number': chain_idx, 'step_number': -1, 'fun': res['fun']}
        
        # Add scalar parameters
        for key, value in optimal_unpacked.items():
            if not isinstance(value, np.ndarray):
                opt_row[key] = value
            elif isinstance(value, np.ndarray):
                if value.ndim == 1:  # 1D array (like c_vec)
                    for i, v in enumerate(value):
                        opt_row[f'{key}_{i}'] = v
                elif value.ndim == 2:  # 2D array (like S_init, E_init, I_init)
                    for season_idx in range(value.shape[0]):
                        for region_idx in range(value.shape[1]):
                            opt_row[f'{key}_s{season_idx}_r{region_idx}'] = value[season_idx, region_idx]
        
        data_rows.append(opt_row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data_rows)#.query("0 < chain_number <=14")
    df = df.assign(sync=sync, optimizer=method)#, likelihood=np.exp(-df.fun))
    # import seaborn as sns
    # from matplotlib import pyplot as plt
    # sns.relplot(hue='chain_number', x='fun', y='theta', data=df, kind='line')                                              
    # plt.show()


    # df['hostname'] = hostname
    df.to_csv(fname, index=False)
    print(f"Saved optimization results to {fname} with {len(df)} rows")

    
    # ll = -inv.fun ## Resulting log-likelihood
    # assert np.isfinite(ll), f"Log likelihood is not finite: {ll}"
    # assert ll >= 0, f"Final log-likelihood is negative: {ll}"
    

if __name__ == "__main__":
    try:
        for method in ['slsqp']:#, 'mma', 'ccsaq']:
            for sync in [True, False]:
                main(sync=sync, method=method)
    except:
        import sys, traceback, pdb
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

