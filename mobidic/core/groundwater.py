"""Groundwater models for MOBIDIC hydrological simulation.

This module implements groundwater dynamics. Currently only the linear reservoir
model is implemented.

"""

import numpy as np


def groundwater_linear(
    h0: np.ndarray,
    kf: np.ndarray,
    recharge: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear groundwater reservoir.

    Approximates groundwater dynamics as a linear reservoir:

    $$
    q = k_f \\cdot h, \\quad dh/dt = R - q
    $$

    where $h$ is the groundwater head, $k_f$ is the aquifer conductivity,
    $R$ is the net recharge, and $q$ is the baseflow.

    The equation is solved analytically, and the average baseflow over the time step is computed.

    Args:
        h0: Initial groundwater head [m], 1D array
        kf: Aquifer conductivity [1/s], 1D array (same shape as h0)
        recharge: Net recharge [m/s] (percolation - capillary_rise - losses),
            1D array (same shape as h0)
        dt: Time step [s]

    Returns a tuple of:
        - $h$: Updated groundwater head [m]
        - $q$: Average baseflow over the time step [m/s]

    """
    h0 = np.asarray(h0, dtype=np.float64)
    kf = np.asarray(kf, dtype=np.float64)
    recharge = np.asarray(recharge, dtype=np.float64)

    # Guard against kf <= 0 (no aquifer): handle with direct integration
    active = kf > 0.0

    h = np.where(active, h0, h0 + recharge * dt)
    q = np.where(active, recharge, recharge)

    if not np.any(active):
        return h, q

    # Solve linear reservoir only on active cells
    kf_a = kf[active]
    h0_a = h0[active]
    R_a = recharge[active]

    exp_adt = np.exp(-kf_a * dt)
    one_minus_exp = 1.0 - exp_adt

    # Clamp recharge to prevent negative head (matching MATLAB groundwater_linear.m line 31)
    # Avoid division by zero when one_minus_exp is extremely small
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = np.where(
            one_minus_exp > 0.0,
            -0.5 * kf_a * h0_a * exp_adt / one_minus_exp,
            R_a,
        )
    Rn = np.maximum(R_a, lower)

    # Average baseflow over time step
    q_a = (kf_a * h0_a - Rn) / (kf_a * dt) * one_minus_exp + Rn

    # Updated head
    R_over_a = Rn / kf_a
    h_a = R_over_a + (h0_a - R_over_a) * exp_adt

    h[active] = h_a
    q[active] = q_a

    return h, q
