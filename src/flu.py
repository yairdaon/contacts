import math
R0 = 3.65

# Recovery rate from calibration to continuous model
gamma = 7/2.2 # 7 days == one week

amplitude = 0.7
# Calculate beta0 from R0 (measured at peak transmission)
# beta0 = R0 * (1 - exp(-gamma)) / (1 + amplitude)
# where (1 - exp(-gamma)) is the recovery probability per week
beta0 = R0 * (1 - math.exp(-gamma)) / (1 + amplitude)
