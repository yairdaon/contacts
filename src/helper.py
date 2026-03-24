import pandas as pd
from itertools import product
import numpy as np
from scipy.special import logit, expit
import matplotlib.pyplot as plt
from typing import Optional
from datetime import datetime

## Important that theta is kept as the first entry.
JACOBIAN_COLS = ['theta', 'S1_0', 'I1_0', 'S2_0', 'I2_0']

def current():
    return f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

