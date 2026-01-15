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


OUTPUT_DIR = os.path.expanduser("~/contacts/res")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@plac.annotations(
    seasonal_driver=('Enable seasonal driver', 'flag', 's'),
    #optimizer=('Optimizer to use: nlopt or scipy', 'option', 'o', str, ['nlopt', 'scipy']),
    difficulty=('Difficulty level', 'option', 'd', str, ['local', 'easy', 'inter', 'hard']),
    seed=('Random seed', 'option', 'r', int)
)


def main(seasonal_driver,
         optimizer='nlopt',
         difficulty='debug',
         seed=None):

    if difficulty == 'debug':
        n_regions, n_seasons, n0, maxeval = 2, 3, 10, None
    elif difficulty == "easy":
        n_regions, n_seasons, n0, maxeval = 2, 10, 10, 500
    elif difficulty == "inter":
        n_regions, n_seasons, n0, maxeval = 2, 15, 15, 1000
    elif difficulty == "hard":
        n_regions, n_seasons, n0, maxeval = 2, 20, 15, None
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    
    print(f"{difficulty} regions {n_regions}, seasons {n_seasons} starts={n0} seasonal_driver {seasonal_driver}")



    pop = makepop(n_regions=n_regions,
                  n_seasons=n_seasons)
    objective = Objective(population=pop,
                          n_weeks=NWEEKS,
                          gamma=7/2.2,  
                          transform=optimizer == 'scipy',
                          seasonal_driver=seasonal_driver)
    true_params = objective.packer.random_dict(seed=seed)
    true_params['theta'] = 0.05
    true_params['beta0'] = 0.5
    print(true_params['theta'])
    print(true_params['beta0'])
    
    # Pack true parameters
    x_true = objective.packer.pack(true_params)
   
    # Generate "observed" data using true parameters (not initial guess)
    true_trajectory = objective.sim(true_params)

    # Generate observed data
    obs = true_trajectory.copy()
    true_counts = true_trajectory['incidence'] * RHO  # Fixed rho value
    scale = np.sqrt(RHO * (1 - RHO) * true_counts)
    obs['incidence'] = true_counts + np.random.randn(true_counts.size) * scale
    obs['incidence'] = np.maximum(1e-6, obs['incidence'])  # Ensure non-negative
    objective.obs = obs

    ## Solve inverse problem
    inv = Inverter(objective=objective, optimizer=optimizer).fit(n0=n0, maxeval=maxeval)

    ## Create CSV dataframe with optimization results
    hostname = socket.gethostname().split('.')[0]  # short server name (e.g. dml12)
    flag = "" if seasonal_driver else "_not" 
    fname = f'{OUTPUT_DIR}/{difficulty}{flag}_seasonal_{hostname}.csv'
    
    # Prepare data list for DataFrame
    data_rows = []
    
    # Add true parameters row (chain_number = -1, step_number = 0)
    true_row = {'chain_number': -1, 'step_number': 0, 'fun': np.nan}
    true_row.update(true_params)
    
    # Flatten arrays in true_params for CSV storage
    for key, value in true_params.items():
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

