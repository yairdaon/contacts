import math
import numpy as np


class Flu:
    # Ranges for optimizer starting points
    slim = (0.4, 0.95)
    ilim = (5e-5, 1e-2)

    # Similar ICs: both regions draw from these narrow ranges
    slim_similar = (0.84, 0.86)
    ilim_similar = (8e-5, 1.2e-4)

    # Different ICs: regions draw independently from these wide ranges
    slim_different = (0.5, 0.95)
    ilim_different = (1e-5, 1e-2)

    ## https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000316
    ## First line in Table 2
    R0_max = 3.52
    R0_min = 1.12
    D = 3.24 # mean infectious period

    # Recovery rate. assume T ~ Exp(1/3.24) then in one week recovery
    # probability is
    # p = \int_0^7 e^{t/3.24} / 3.24, dt = 1 - e^{-7/3.24} = 1 - e^{-2.160} \approx 0.885

    gamma = 7/D # 7 days == one week

    # For the discrete exponential SIR: R0 = beta / (1 - exp(-gamma))
    # So beta = R0 * (1 - exp(-gamma))
    beta_max = R0_max * (1 - np.exp(-gamma))
    beta_min = R0_min * (1 - np.exp(-gamma))
    beta0 = ( beta_max + beta_min ) / 2
    
    ## Since beta_max = beta0 (1+delta) and beta_min = beta0 (1-delta), we get
    delta = beta_max /beta0 - 1
    delta_ = 1 - beta_min/beta0 
    assert abs(delta-delta_) < 1e-15
    n_weeks = 20
    step_size = 7 / 365.25  # fraction of year per week

    def __repr__(self):
        attrs = {k: v for k, v in self.__class__.__dict__.items() if not k.startswith('_') and not callable(v)}
        # Also include attributes from parent classes that aren't overridden
        for base in self.__class__.__mro__:
            for k, v in base.__dict__.items():
                if not k.startswith('_') and not callable(v) and k not in attrs:
                    attrs[k] = v
        
        details = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(attrs.items()))
        return f"{self.__class__.__name__}({details})"

    def __str__(self):
        attrs = {k: v for k, v in self.__class__.__dict__.items() if not k.startswith('_') and not callable(v)}
        for base in self.__class__.__mro__:
            for k, v in base.__dict__.items():
                if not k.startswith('_') and not callable(v) and k not in attrs:
                    attrs[k] = v
                    
        header = f"--- {self.__class__.__name__} Parameters ---"
        lines = [header]
        for k, v in sorted(attrs.items()):
            val_str = f"{v:.4f}" if isinstance(v, float) else f"{v}"
            lines.append(f"{k:10} : {val_str}")
        return "\n".join(lines)

    
class Mortality(Flu):
    ## Infection fatality rate from first table of https://onlinelibrary.wiley.com/doi/10.1111/irv.12486
    rho = (4e3+2e4) / (9.2e6 + 3.56e7) ## Approximately half percent
    # print(f"IFR = {ifr*100:.3f}%")
    #ifr = 0.05 / 100

    
class ILI(Flu):
    ## Reporting rate for ILI?
    rho = 0.3

#print(Flu())
