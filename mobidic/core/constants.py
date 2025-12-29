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
FLUX_MIN = 1e-10  # Minimum flux to avoid division by zero (m/s). Not currently used.
HSOIL_MIN = 1e-3  # Minimum soil depth (m). Not currently used.
TOLERANCE = 1e-8  # General numerical tolerance (-). Not currently used.
MASS_BALANCE_RTOL = 1e-6  # Relative tolerance for mass balance checks (-). Not currently used.

# ============================================================================
# Physical constants
# ============================================================================

OMEGA_DAY = 2 * np.pi / 24 / 3600  # Earth's angular velocity (rad/s). Not currently used.
P_AIR = 1013.0  # Air pressure (hPa) (approximation, standard ~1013 hPa). Not currently used.
