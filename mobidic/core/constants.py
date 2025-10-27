"""Physical constants and numerical parameters for MOBIDIC simulation.

These parameters are for developers only and should not be exposed to end users.
They control numerical stability, physical constants, and internal simulation behavior.
"""

import numpy as np

# ============================================================================
# Simulation control parameters
# ============================================================================

WARMUP_HOURS = 0  # Length of warmup period [hours]
FORECAST_LAG_SECONDS = 48 * 3600  # Forecast lag time [s]
N_SUBSTEP_RESERVOIR = 24  # Number of substeps for reservoir routing [-]
THIESSEN_ENABLED = False  # Use Thiessen polygons for interpolation (0 = disabled)
TEST_MODE = False  # Enable test/validation mode (0 = disabled)
KC_FAO_DEFAULT = 1.0  # Default FAO crop coefficient [-]
N_RIVERS = 0  # Number of rivers (legacy parameter, not currently used)
N_PERIOD = 0  # MODFLOW stress period (for future groundwater coupling)

# ============================================================================
# Numerical stability parameters
# ============================================================================

FLUX_MIN = 1e-10  # Minimum flux to avoid division by zero [m/s]
HSOIL_MIN = 1e-3  # Minimum soil depth [m]
W_MIN = 1e-5  # Minimum water content [m]
TOLERANCE = 1e-8  # General numerical tolerance [-]
MASS_BALANCE_RTOL = 1e-6  # Relative tolerance for mass balance checks [-]

# ============================================================================
# Physical constants
# ============================================================================

OMEGA_DAY = 2 * np.pi / 24 / 3600  # Earth's angular velocity [rad/s]
PAIR = 1000.0  # Air pressure [hPa] (approximation, standard ~1013 hPa)
