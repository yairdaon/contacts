import time

import pytest
import numpy as np
import pandas as pd
from itertools import cycle
from scipy.stats import nbinom
from statsmodels.base import optimizer

from src.inverter import Objective, Inverter
from src.helper import makepop, a2s
from src.losses import RHO

NWEEKS = 28
EPS = 0.15  ## Allowed relative (!!) error

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for testing


@pytest.mark.parametrize("transform", [True, False])
def test_inverter_initialization(transform):
    """Test that Inverter initializes correctly with population dataframe."""
    pop = makepop()

    # Create objective and loss
    objective = Objective(population=pop, n_weeks=NWEEKS, transform=transform)

    # Test initialization
    inv = Inverter(objective=objective, optimizer=optimizer)

    assert inv.packer.n_seasons == pop.season.nunique(), f"Packer seasons {inv.packer.n_seasons} != population seasons {pop.season.nunique()}"
    assert inv.packer.n_regions == pop.region.nunique(), f"Packer regions {inv.packer.n_regions} != population regions {pop.region.nunique()}"
    assert objective.n_weeks == NWEEKS, f"Objective weeks {objective.n_weeks} != expected {NWEEKS}"
    assert pop.shape == (inv.packer.n_seasons * inv.packer.n_regions,
                         3), f"Population shape {pop.shape} != expected ({inv.packer.n_seasons * inv.packer.n_regions}, 3)"

@pytest.mark.parametrize("transform", [True, False])
def test_sim(transform):
    """Test that Objective.sim() produces valid output."""
    pop = makepop(n_regions=10, n_seasons=30)
    objective = Objective(population=pop, n_weeks=NWEEKS, transform=transform)

    # Generate random parameters and run simulation
    for i in range(10):
        params = objective.packer.random_dict()
        results = objective.sim(params)  

        # Check output format
        assert isinstance(results, pd.DataFrame), f"Simulation result is {type(results)}, expected DataFrame"
        expected_cols = {'time', 'region', 'incidence', 'season'}
        assert set(
            results.columns) == expected_cols, f"Result columns {set(results.columns)} != expected {expected_cols}"

        # Check data completeness
        expected_rows = pop.season.nunique() * pop.region.nunique() * objective.n_weeks
        assert len(results) == expected_rows, f"Result length {len(results)} != expected {expected_rows}"
        assert results.incidence.min() >= 0, f"Negative incidence found: min={results.incidence.min()}"


