"""Physical constants and numerical parameters for MOBIDIC simulation.

These parameters are for developers only and should not be exposed to end users.
They control numerical stability, physical constants, and internal simulation behavior.
"""

import numpy as np

# ============================================================================
# Simulation control parameters
# ============================================================================

F0_CONSTANT = 0.85  # Constant in fraction of time step without rain f0 (-)
N_SUBSTEP_RESERVOIR = 24  # Number of substeps for reservoir routing (-)
KC_FAO_DEFAULT = 1.0  # Default FAO crop coefficient (-). Not currently used.

# ============================================================================
# Numerical stability parameters
# ============================================================================

W_MIN = 1e-5  # Minimum water content (m)
FLUX_MIN = 1e-10  # Minimum flux to avoid division by zero (m/s)
HSOIL_MIN = 1e-3  # Minimum soil depth (m). Not currently used.
TOLERANCE = 1e-8  # General numerical tolerance (-). Not currently used.
MASS_BALANCE_RTOL = 1e-6  # Relative tolerance for mass balance checks (-). Not currently used.

# ============================================================================
# Physical constants
# ============================================================================

OMEGA_DAY = 2 * np.pi / 24 / 3600  # Earth's angular velocity (rad/s)
P_AIR = 1013.0  # Air pressure (hPa) (approximation, standard ~1013 hPa)

# Energy balance physical constants
STEFAN_BOLTZMANN = 5.6697e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
EMISS_AIR = 0.87  # Air emissivity (-)
EMISS_SOIL = 0.98  # Soil emissivity (-)
RHO_AIR = 1.225  # Air density (kg/m^3)
RHO_WATER = 1000.0  # Water density (kg/m^3)
LV = 2.5e6  # Latent heat of vaporization (J/kg)
CP_AIR = 1004.0  # Specific heat of air at constant pressure (J/K/kg)
