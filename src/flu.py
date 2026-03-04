import math

## https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000316
## First line in Table 2
R0_max = 3.52
R0_min = 1.12
D = 3.24 # mean infectious period

# Recovery rate from calibration to continuous model
gamma = 7/D # 7 days == one week

beta_max = R0_max * gamma
beta_min = R0_min * gamma
beta0 = ( beta_max + beta_min ) / 2

## Since beta_max = beta0 (1+eps) and beta_min = beta0 (1-eps), we get
eps = beta_max /beta0 - 1
eps_ = 1 - beta_min/beta0 
assert abs(eps-eps_) < 1e-15


## Infection fatality rate from first table of https://onlinelibrary.wiley.com/doi/10.1111/irv.12486
ifr = (4e3+2e4) / (9.2e6 + 3.56e7) ## Approximately half percent
# print(f"IFR = {ifr*100:.3f}%")
#ifr = 0.05 / 100

## Reporting rate for ILI?
rho = 0.3
n_weeks = 25
