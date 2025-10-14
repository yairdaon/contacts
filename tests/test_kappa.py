import pytest
import numpy as np
import pandas as pd
from numpy import sin, cos, pi, log, exp

from src.inverter import Objective
from src.helper import makepop

NWEEKS = 28

@pytest.mark.parametrize("difficulty", ["easy", "intermediate", "hard"])
def test_kappa_condition_number(difficulty, seed=43):
    """
    Test to calculate the condition number of the matrix whose columns are I(t)/β(t).
    
    Difficulty levels:
    - easy: 2 regions, 5 seasons
    - intermediate: 5 regions, 15 seasons  
    - hard: 10 regions, 30 seasons
    """
    # Set test parameters based on difficulty (same as test_inference)
    if difficulty == "easy":
        n_regions, n_seasons = 2, 5
    elif difficulty == "intermediate":
        n_regions, n_seasons = 5, 15
    elif difficulty == "hard":
        n_regions, n_seasons = 10, 30
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    
    print(f"\n{difficulty} test: {n_regions} regions, {n_seasons} seasons")
    
    # Create population and objective
    pop = makepop(n_regions=n_regions, n_seasons=n_seasons)
    objective = Objective(population=pop, n_weeks=NWEEKS, transform=True)
    
    # Generate random parameters
    np.random.seed(seed)
    params = objective.packer.random_dict(seed=seed)
    
    # Get full simulation output including β(t) values
    S_init = params['S_init']
    E_init = params['E_init']
    I_init = params['I_init']
    beta0 = params['beta0']
    omega = params['omega']
    eps = params['eps']
    c_mat = objective.packer.c_vec_to_mat(params["c_vec"])
    
    # Create matrix whose columns are I(t)/β(t)
    kappa_matrix_list = []
    
    for season_idx, season in enumerate(objective.packer.seasons):
        pop = objective.pops[season]
        S = S_init[season_idx, :]
        E = E_init[season_idx, :]
        I = I_init[season_idx, :]

        # Get full simulation output with β(t) values
        df = objective.run(S_init=S,
                    E_init=E,
                    I_init=I,
                    dt_step=objective.dt_step,
                    dt_output=objective.dt_output,
                    n_weeks=objective.n_weeks,
                    beta0=beta0,
                    sigma=objective.sigma,
                    mu=objective.mu,
                    nu=objective.nu,
                    omega=omega,
                    eps=eps,
                    contact_matrix=c_mat,
                    population=pop,
                    start_date=season)
        
        # Extract I(t) and β(t) for each region
        for region_idx in range(len(objective.packer.regions)):
            # Get incidence I(t) and forcing β(t) for this region
            I_values = df[f'C{region_idx}'].values[1:]  # Skip first NaN value
            #beta_values = df[f'F{region_idx}'].values[1:]  # Skip first value
            
            # Calculate I(t)/β(t) column
            #I_over_beta = I_values / beta_values
            
            kappa_matrix_list.append(I_values)
    
    # Stack columns to form the matrix
    kappa_matrix = np.column_stack(kappa_matrix_list)
    
    # Calculate condition number
    condition_number = np.linalg.cond(kappa_matrix)
    
    print(f"Condition number of I(t) matrix: {condition_number:.6e}")
    print(f"Matrix shape: {kappa_matrix.shape}")
    
    # Basic sanity checks
    assert np.isfinite(condition_number), f"Condition number is not finite: {condition_number}"
    assert condition_number > 0, f"Condition number should be positive: {condition_number}"
