import pandas as pd
from itertools import product
import numpy as np


def a2s(x):
    return np.array2string(x, precision=3)


def makepop(n_regions=2, n_seasons=3):
    """
    Create a population DataFrame for testing with specified regions and seasons.
    
    Parameters:
    -----------
    n_regions : int
        Number of regions (default: 2)
    n_seasons : int  
        Number of seasons (default: 3)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns [season, region, population]
    """
    seasons = [f"{year}-01-01" for year in range(1990, 1990 + n_seasons)]
    regions = [f"HHS{region}" for region in range(n_regions)]

    population_data = []
    for season, region in product(seasons, regions):
        population_data.append({
            'season': season,
            'region': region,
            'population': 100
        })

    return pd.DataFrame(population_data)