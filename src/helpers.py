import numpy as np
from datetime import datetime


def std_from_precisions(precisions):
    """Aggregate per-season precisions (via CRLBs) into a single standard deviation bound.

    precisions: array-like of per-season precision values for theta.
    Returns: sqrt of the harmonic-mean aggregation (Eq. crlb_multi in the paper).
    """
    precisions = np.asarray(precisions, dtype=float)
    precisions = np.where(np.isfinite(precisions), precisions, 0)
    total = np.sum(precisions)
    return np.inf if total <= 0 else 1.0 / np.sqrt(total)



def current():
    return f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
"""  
┌──────┬─────────────────┬──────────────────────────────┐
│ Code │      Name       │           Meaning            │
├──────┼─────────────────┼──────────────────────────────┤
│ 1    │ SUCCESS         │ Generic success              │
├──────┼─────────────────┼──────────────────────────────┤
│ 2    │ STOPVAL_REACHED │ Hit target objective value   │
├──────┼─────────────────┼──────────────────────────────┤
│ 3    │ FTOL_REACHED    │ Hit ftol_rel or ftol_abs     │
├──────┼─────────────────┼──────────────────────────────┤
│ 4    │ XTOL_REACHED    │ Hit xtol_rel or xtol_abs     │
├──────┼─────────────────┼──────────────────────────────┤
│ 5    │ MAXEVAL_REACHED │ Hit max function evaluations │
├──────┼─────────────────┼──────────────────────────────┤
│ 6    │ MAXTIME_REACHED │ Hit max wall time            │
└──────┴─────────────────┴──────────────────────────────┘
Error codes (negative):
┌──────┬──────────────────┬──────────────────────────────────────────┐
│ Code │       Name       │                 Meaning                  │
├──────┼──────────────────┼──────────────────────────────────────────┤
│ -1   │ FAILURE          │ Generic failure                          │
├──────┼──────────────────┼──────────────────────────────────────────┤
│ -2   │ INVALID_ARGS     │ Bad arguments (e.g. lower > upper bound) │
├──────┼──────────────────┼──────────────────────────────────────────┤
│ -3   │ OUT_OF_MEMORY    │ Out of memory                            │
├──────┼──────────────────┼──────────────────────────────────────────┤
│ -4   │ ROUNDOFF_LIMITED │ Roundoff errors limited progress         │
├──────┼──────────────────┼──────────────────────────────────────────┤
│ -5   │ FORCED_STOP      │ User called force_stop()                 │
└──────┴──────────────────┴──────────────────────────────────────────┘
"""

CODES = {1: "Success",
         2: "STOPVAL_REACHED",
         3: "FTOL_REACHED",
         4: "XTOL_REACHED",
         5: "MAXEVAL_REACHED",
         6: "MAXTIME_REACHED"
         }
