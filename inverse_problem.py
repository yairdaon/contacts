import os
import numpy as np
import pickle
import pytest
import pandas as pd
import plac
import socket
from pprint import pprint

from src.helper import makepop, a2s
from src.inverter import Inverter, Objective
from src.losses import RHO
from tests.test_inverter import NWEEKS
from src import flu

OUTPUT_DIR = os.path.expanduser("~/contacts/res")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@plac.annotations(
    sync=('Synchronized seasonal driver', 'flag', 's'),
    full=('FULL, HARD (!!!) RUN', 'flag', 'f')
)


def main(sync,
         full):

    theta = 0.05
    phase = np.zeros(2)
    if not sync:
        phase[1] = np.pi
    

    n_regions = 2
    if full:
        n_seasons = 20
        n0 = 20
        maxeval =  None
    else:
        n_seasons = 10
        n0 = 1  # No parallelization in debug mode
        maxeval = None#5000

    pop = makepop(n_regions=n_regions,
                  n_seasons=n_seasons)
    objective = Objective(population=pop,
                          n_weeks=NWEEKS,
                          gamma=flu.gamma,
                          beta0 = flu.beta0,
                          amplitude=flu.amplitude,
                          phase=phase)
    
    true = objective.packer.random_dict()
    true['theta'] = theta
    true_trajectory = objective.sim(true)

    # Generate observed data
    obs = true_trajectory.copy()
    true_counts = true_trajectory['incidence'] * RHO  # Fixed rho value
    scale = np.sqrt(RHO * (1 - RHO) * true_counts)
    obs['incidence'] = true_counts + np.random.randn(true_counts.size) * scale
    obs['incidence'] = np.maximum(1e-6, obs['incidence'])  # Ensure non-negative
    objective.obs = obs

    ## Solve inverse problem
    inv = Inverter(objective=objective).fit(n0=n0, maxeval=maxeval)

    ## Create CSV dataframe with optimization results
    hostname = socket.gethostname().split('.')[0]  # short server name (e.g. dml12)
    flag = "" if sync else "un"
    difficulty = 'full' if full else 'debug'
    fname = f'{OUTPUT_DIR}/{difficulty}_{flag}sync_{hostname}.csv'
    
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
    df = pd.DataFrame(data_rows)
    df['hostname'] = hostname
    df.to_csv(fname, index=False)
    print(f"Saved optimization results to {fname} with {len(df)} rows")

    
    fun = inv.fun
    assert np.isfinite(fun), f"Final loss is not finite: {fun}"
    assert fun >= 0, f"Final loss is negative: {fun}"


if __name__ == "__main__":
    try:
        plac.call(main)
    except:
        import sys, traceback, pdb
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

