"""
Epidemiological parameters for childhood diseases.

Parameters are calibrated for discrete-time SIR models with weekly time steps.

References:
- Measles R0: Guerra et al. (2017) Lancet Infect Dis, https://pubmed.ncbi.nlm.nih.gov/28757186/
- Pertussis R0: Kretzschmar et al. (2010) PLoS Med, https://pmc.ncbi.nlm.nih.gov/articles/PMC2889930/
- Keeling & Rohani (2008) "Modeling Infectious Diseases"
"""

import math
from dataclasses import dataclass


@dataclass
class Disease:
    """Epidemiological parameters for a disease."""
    name: str
    R0: float                    # Basic reproduction number
    infectious_days: float       # Mean infectious period in days
    amplitude: float             # Seasonal forcing amplitude
    rho: float                   # Reporting rate
    nweeks: int                  # Default simulation length in weeks

    @property
    def gamma(self) -> float:
        """Weekly recovery rate (continuous-time equivalent)."""
        return 7.0 / self.infectious_days

    @property
    def beta0(self) -> float:
        """Baseline transmission rate, calibrated to R0 at peak."""
        # beta0 = R0 * (1 - exp(-gamma)) / (1 + amplitude)
        return self.R0 * (1 - math.exp(-self.gamma)) / (1 + self.amplitude)


# Influenza
# R0, infectious_days: Keeling & Rohani (2008) "Modeling Infectious Diseases"
#   boarding school example, Table 2.1
# amplitude: Keeling & Rohani (2008) Fig. 5.5
# rho: assumed
flu = Disease(
    name="Influenza",
    R0=3.65,                    # Keeling & Rohani (2008) Table 2.1
    infectious_days=2.2,        # Keeling & Rohani (2008) Table 2.1
    amplitude=0.7,              # Keeling & Rohani (2008) Fig. 5.5
    rho=0.3,                    # Assumed
    nweeks=25
)

# Measles
# R0: 12-18, Guerra et al. (2017) Lancet Infect Dis
#   https://pubmed.ncbi.nlm.nih.gov/28757186/
# infectious_days: 4 days before rash to 4 days after, CDC Pink Book
#   https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-13-measles.html
# amplitude: strong winter/spring peaks in temperate climates
# rho: higher than other diseases due to visible rash
measles = Disease(
    name="Measles",
    R0=15.0,                    # Guerra et al. (2017): range 12-18
    infectious_days=8.0,        # CDC: 4 days before to 4 days after rash
    amplitude=0.4,              # Winter/spring peaks in temperate climates
    rho=0.5,                    # Higher due to visible rash
    nweeks=25
)

# Pertussis (Whooping cough)
# R0: 12-17, Kretzschmar et al. (2010) PLoS Med
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC2889930/
# infectious_days: ~3 weeks after cough onset, CDC Clinical Overview
#   https://www.cdc.gov/pertussis/hcp/clinical-overview/
# amplitude: weaker seasonality than measles
# rho: lower reporting, early catarrhal stage resembles common cold
pertussis = Disease(
    name="Pertussis",
    R0=15.0,                    # Kretzschmar et al. (2010): range 12-17
    infectious_days=21.0,       # CDC: ~3 weeks infectious after cough onset
    amplitude=0.2,              # Weaker seasonality than measles
    rho=0.3,                    # Lower: early stage resembles cold
    nweeks=25
)
