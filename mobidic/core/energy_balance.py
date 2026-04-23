"""Surface energy balance (1L scheme).

This module implements the analytical 1-layer surface energy balance.
The solver is an analytical Fourier decomposition
of the land-surface energy budget, splitting the timestep at the sunrise
and sunset boundaries.

The kernel (``energy_balance_1l``) is JIT-compiled with Numba; a pure
NumPy reference implementation (``_energy_balance_1l_numpy``) is kept
for testing.

"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from mobidic.core import constants as const

# Constants are saved at the module level for the JIT kernel
_SIGMA = const.STEFAN_BOLTZMANN
_EPS_AIR = const.EMISS_AIR
_EPS_SOI = const.EMISS_SOIL
_RHO_WATER = const.RHO_WATER
_RHO_AIR = const.RHO_AIR
_LV = const.LV
_CP_AIR = const.CP_AIR


# ---------------------------------------------------------------------------
# Auxiliary functions: solar position, diurnal cycle, saturation humidity
# ---------------------------------------------------------------------------


def solar_position(hour: float, day: int, lat: float, lon: float) -> tuple[float, float]:
    """Compute solar azimuth and elevation.

    Args:
        hour: Local time in decimal hours (0-24).
        day: Day of year (Julian day, 1-365).
        lat: Latitude [deg North].
        lon: Longitude [deg East].

    Returns:
        tuple[float, float]: Solar (azimuth, elevation) [deg].
    """
    delta = 23.45 * np.pi / 180.0 * np.cos(2.0 * np.pi / 365.0 * (172 - (day % 365)))
    dt1 = round(lon / 15.0) - lon / 15.0
    az = (hour + 12.0 - dt1) * 15.0
    az = az % 360.0
    sinalp = np.sin(delta) * np.sin(lat * np.pi / 180.0) + np.cos(delta) * np.cos(lat * np.pi / 180.0) * np.cos(
        az * np.pi / 180.0
    )
    el = 180.0 / np.pi * np.arcsin(sinalp)
    az = (180.0 + az) % 360.0
    return az, el


def solar_hours(lat: float, lon: float, jday: int) -> tuple[float, float]:
    """Compute sunrise and sunset hours via bisection on solar elevation.

    Args:
        lat: Latitude [deg North].
        lon: Longitude [deg East].
        jday: Day of year (1-365).

    Returns:
        tuple[float, float]: Sunrise (hrise) and sunset (hset) in decimal hours.
    """
    # Sunrise: solar elevation crosses zero in the morning (0-12)
    hpre, hpos = 0.0, 12.0
    hrise = 6.0
    while (hpos - hpre) > (1.0 / 60.0):
        hrise = 0.5 * (hpos + hpre)
        _, el = solar_position(hrise, jday, lat, lon)
        if el > 0:
            hpos = hrise
        else:
            hpre = hrise

    # Sunset: solar elevation crosses zero in the afternoon (12-24)
    hpre, hpos = 12.0, 24.0
    hset = 18.0
    while (hpos - hpre) > (1.0 / 60.0):
        hset = 0.5 * (hpos + hpre)
        _, el = solar_position(hset, jday, lat, lon)
        if el > 0:
            hpre = hset
        else:
            hpos = hset

    return hrise, hset


def diurnal_radiation_cycle(
    rs_avg: float | NDArray[np.float64],
    t_sunrise: float,
    t_sunset: float,
    mode: str = "average",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Decompose daily radiation into sinusoidal amplitude and constant.

    The diurnal radiation cycle is represented as ``A * sin(omega * t + phi) + C``
    where ``omega = 2*pi/86400`` and the phase ``phi = -pi/2``. This function
    returns ``(A, C)``.

    Args:
        rs_avg: Incoming shortwave radiation [W/m²]. Scalar or per-cell array.
        t_sunrise: Sunrise time [s] (seconds of day).
        t_sunset: Sunset time [s] (seconds of day).
        mode: "average" if ``rs_avg`` is the day-average radiation,
            "instant" if it is an instantaneous value at ``t_sunrise``.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: (amplitude, constant) arrays with the same shape as ``rs_avg``.
    """
    w = 2.0 * np.pi / 86400.0
    p_r = -np.pi / 2.0

    if mode == "average":
        dd = (np.cos(w * t_sunrise + p_r) - np.cos(w * t_sunset + p_r)) / w - np.sin(w * t_sunset + p_r) * (
            t_sunset - t_sunrise
        )
        amplitude = rs_avg * (t_sunset - t_sunrise) / dd
    elif mode == "instant":
        dd = np.sin(w * t_sunrise + p_r) - np.sin(w * t_sunset + p_r)
        amplitude = rs_avg / dd
    else:
        raise ValueError(f"Invalid mode '{mode}', expected 'average' or 'instant'")

    constant = -amplitude * np.sin(w * t_sunset + p_r)
    return amplitude, constant


def saturation_specific_humidity(
    T: NDArray[np.float64],
    P: float,
    dT: float | NDArray[np.float64] = 0.0,
) -> NDArray[np.float64]:
    """Saturation specific humidity (Magnus formula).

    Args:
        T: Temperature [K].
        P: Pressure [mb] (scalar).
        dT: Dew-point depression [K] (default 0).

    Returns:
        NDArray[np.float64]: Specific humidity [kg/kg].
    """
    ep = 0.622
    Tc = T - 273.15 - dT
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    # When dT is non-zero, apply the correction. dT=0 is the common case.
    if np.any(np.asarray(dT) != 0):
        es = es - (0.00066 * (1.0 + 0.00115 * Tc)) * P * dT
    return ep * es / (P - (1.0 - ep) * es)


# ---------------------------------------------------------------------------
# Core 1L solver
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, parallel=True)
def _energy_balance_1l_kernel(
    ff: float,
    a_tem: NDArray[np.float64],
    a_rad: NDArray[np.float64],
    p_tem: float,
    p_rad: float,
    c_tem: NDArray[np.float64],
    c_rad: NDArray[np.float64],
    td_ini: NDArray[np.float64],
    tm: NDArray[np.float64],
    u: NDArray[np.float64],
    pair: float,
    hair: NDArray[np.float64],
    step: float,
    ch: NDArray[np.float64],
    alb: NDArray[np.float64],
    kaps: float,
    nis: float,
    tcost: float,
    etrsuetp: NDArray[np.float64],
    tt_values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Numba-compiled per-cell kernel for the 1L surface energy balance."""
    n = tm.shape[0]
    ts_out = np.empty(n, dtype=np.float64)
    td_out = np.empty(n, dtype=np.float64)
    evp = np.empty(n, dtype=np.float64)

    sigma = _SIGMA
    eps_air = _EPS_AIR
    eps_soi = _EPS_SOI
    rhow = _RHO_WATER
    rhoair = _RHO_AIR
    lv = _LV
    cp = _CP_AIR

    dz = np.sqrt(nis / ff)
    alpha_param = np.sqrt(365.0)
    alphadz2 = (1.0 + alpha_param) * dz * dz
    kaps_dz = kaps / dz
    half_step = 0.5 * step

    # Phase trig (loop-invariant across cells AND time substeps)
    sin_p_rad = np.sin(p_rad)
    cos_p_rad = np.cos(p_rad)
    sin_p_tem = np.sin(p_tem)
    cos_p_tem = np.cos(p_tem)

    nt = tt_values.shape[0]
    w = ff

    for i in prange(n):
        tm_i = tm[i]
        c_tem_i = c_tem[i]
        c_rad_i = c_rad[i]
        a_tem_i = a_tem[i]
        a_rad_i = a_rad[i]
        td_ini_i = td_ini[i]
        u_i = u[i]
        ch_i = ch[i]
        alb_i = alb[i]
        hair_i = hair[i]
        etr_i = etrsuetp[i]

        tm_c = tm_i - 273.15
        tm_cx = tm_c + 243.5
        tm2 = tm_i * tm_i
        tm3 = tm2 * tm_i
        tm4 = tm3 * tm_i
        kh = ch_i * u_i
        rhoaircpkh = rhoair * cp * kh
        p622 = 0.622 * rhoair * lv * etr_i / pair * kh
        es4tm3 = 4.0 * sigma * eps_soi * tm3
        ea4tm3 = 4.0 * sigma * eps_air * tm3

        es1 = 6.112 * np.exp(17.67 * tm_c / tm_cx)
        es2 = 17.67 * es1 * (1.0 - tm_c / tm_cx) / tm_cx
        ea1 = es1 * hair_i
        ea2 = es2 * hair_i

        den = -kaps_dz - es4tm3 - rhoaircpkh - p622 * es2

        d0 = (
            2.0 * nis * (
                tcost / (alpha_param * alphadz2)
                - (p622 * tm_i * (es2 - ea2) + 3.0 * sigma * (eps_soi - eps_air) * tm4 + p622 * (ea1 - es1))
                / (den * alphadz2)
            )
        )
        d1 = -2.0 * nis * (1.0 - alb_i) / (den * alphadz2)
        d2 = -2.0 * nis * (p622 * ea2 + rhoaircpkh + ea4tm3) / (den * alphadz2)
        pp = 2.0 * nis * (-alpha_param * (kaps / (den * dz)) - (1.0 + alpha_param)) / (alpha_param * alphadz2)
        pp2 = pp * pp
        pp3 = pp2 * pp

        # Replicates the original ``(c_tem**2)**2`` (i.e. c_tem**4) term
        c_tem2 = c_tem_i * c_tem_i
        c_tem4 = c_tem2 * c_tem2

        variaroba = (
            c_rad_i * (1.0 - alb_i)
            + 3.0 * sigma * eps_soi * tm4
            + (-3.0 * eps_air * sigma + 4.0 * sigma * eps_air) * c_tem4
            + (rhoaircpkh + p622 * ea2) * c_tem_i
            + p622 * (ea1 - es1 + (es2 - ea2) * tm_i)
        )

        d1_s = d1 * a_rad_i
        d2_s = d2 * a_tem_i
        ppw = pp * w
        ppw2 = ppw * w
        denom_osc = pp3 + ppw2
        denom_ts = kaps_dz + es4tm3 + rhoaircpkh + p622 * es2
        d_const_td = (d1 * c_rad_i + d2 * c_tem_i) / pp
        ea_term = ea4tm3 + rhoaircpkh + p622 * ea2

        ts_i = 0.0
        td_i = 0.0
        evp_acc = 0.0
        evp0_prev = 0.0

        for k in range(nt):
            tt = tt_values[k]
            e_pt = np.exp(pp * tt)

            # Constant-part solution
            td1 = d_const_td * (e_pt - 1.0)
            ts1 = (kaps_dz * td1 + variaroba) / denom_ts

            # Sinusoidal-part solution (single mode in 1L scheme)
            wttp_r = w * tt + p_rad
            wttp_t = w * tt + p_tem
            sin_wttp_r = np.sin(wttp_r)
            cos_wttp_r = np.cos(wttp_r)
            sin_wttp_t = np.sin(wttp_t)
            cos_wttp_t = np.cos(wttp_t)

            td2 = (
                ppw * (d1_s * (e_pt * cos_p_rad - cos_wttp_r) + d2_s * (e_pt * cos_p_tem - cos_wttp_t))
                + (e_pt - 1.0) * d0 * (pp2 + w * w)
                + e_pt * (td_ini_i * (pp3 + ppw2) + pp2 * (d2_s * sin_p_tem + d1_s * sin_p_rad))
                - pp2 * (d2_s * sin_wttp_t + d1_s * sin_wttp_r)
            ) / denom_osc
            ts2 = (
                kaps_dz * td2
                + a_rad_i * sin_wttp_r * (1.0 - alb_i)
                + a_tem_i * sin_wttp_t * ea_term
            ) / denom_ts

            td_i = td1 + td2
            ts_i = ts1 + ts2

            # Inlined saturation_specific_humidity (Magnus, dT=0)
            ts_c = ts_i - 273.15
            es_s = 6.112 * np.exp(17.67 * ts_c / (ts_c + 243.5))
            qs_soil = 0.622 * es_s / (pair - 0.378 * es_s)

            tair_inst = a_tem_i * sin_wttp_t + c_tem_i
            ta_c = tair_inst - 273.15
            es_a = 6.112 * np.exp(17.67 * ta_c / (ta_c + 243.5))
            qs_air = 0.622 * es_a / (pair - 0.378 * es_a)

            evp0 = rhoair * lv * etr_i * kh * (qs_soil - qs_air * hair_i)
            evp_acc += (evp0_prev + evp0) * half_step
            evp0_prev = evp0

        evp_water = evp_acc / (rhow * lv)
        if evp_water > 0.0 and np.isfinite(evp_water):
            evp[i] = evp_water
        else:
            evp[i] = 0.0
        ts_out[i] = ts_i
        td_out[i] = td_i

    return ts_out, td_out, evp


def _broadcast_to_n(value: float | NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Broadcast a scalar or array input to a contiguous float64 array of length ``n``."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, arr.item(), dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"Expected scalar or array of shape ({n},), got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def energy_balance_1l(
    ff: float,
    a_tem: NDArray[np.float64],
    a_rad: NDArray[np.float64] | float,
    p_tem: float,
    p_rad: float,
    c_tem: NDArray[np.float64],
    c_rad: NDArray[np.float64] | float,
    td_ini: NDArray[np.float64],
    tm: NDArray[np.float64],
    u: NDArray[np.float64] | float,
    pair: float,
    hair: NDArray[np.float64],
    t_end: float,
    step: float,
    ch: NDArray[np.float64],
    alb: NDArray[np.float64],
    kaps: float,
    nis: float,
    tcost: float,
    etrsuetp: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Analytical Fourier 1-layer surface energy balance over a sub-period.

    Solves the constant + sinusoidal parts of the linearised surface energy
    balance analytically and integrates the evaporation flux via trapezoidal
    integration inside ``[0, t_end]`` with step ``step``. Numba-accelerated.

    Args:
        ff: Diurnal angular frequency [rad/s] (typically ``2*pi/86400``).
        a_tem: Air-temperature amplitude [K] (per cell).
        a_rad: Radiation amplitude [W/m²] (per cell or scalar 0 for night).
        p_tem: Air-temperature phase [rad] (scalar, already time-shifted).
        p_rad: Radiation phase [rad] (scalar, already time-shifted).
        c_tem: Constant part of air temperature [K] (per cell).
        c_rad: Constant part of radiation [W/m²] (per cell or 0 for night).
        td_ini: Initial deep-soil temperature [K] (per cell).
        tm: Mean air temperature used for linearisation [K] (per cell).
        u: Wind speed [m/s] (per cell, or 0 for night).
        pair: Air pressure [mb] (scalar).
        hair: Relative humidity [0-1] (per cell).
        t_end: Integration length of this sub-period [s].
        step: Integration step for evaporation [s].
        ch: Turbulent exchange coefficient for heat [-] (per cell).
        alb: Surface albedo [-] (per cell).
        kaps: Soil thermal conductivity [W/m/K] (scalar).
        nis: Soil thermal diffusivity [m²/s] (scalar).
        tcost: Deep ground (constant) temperature [K] (scalar).
        etrsuetp: Water-limited to energy-limited ET ratio [-] (scalar or per cell).

    Returns:
        ts (NDArray[np.float64]): Surface temperature at the end of the sub-period [K].
        td (NDArray[np.float64]): Deep-soil temperature at the end of the sub-period [K].
        evp (NDArray[np.float64]): Evaporation over the sub-period [m].
    """
    tm_arr = np.ascontiguousarray(tm, dtype=np.float64)
    n = tm_arr.shape[0]

    a_tem_arr = _broadcast_to_n(a_tem, n)
    a_rad_arr = _broadcast_to_n(a_rad, n)
    c_tem_arr = _broadcast_to_n(c_tem, n)
    c_rad_arr = _broadcast_to_n(c_rad, n)
    td_ini_arr = _broadcast_to_n(td_ini, n)
    u_arr = _broadcast_to_n(u, n)
    hair_arr = _broadcast_to_n(hair, n)
    ch_arr = _broadcast_to_n(ch, n)
    alb_arr = _broadcast_to_n(alb, n)
    etr_arr = _broadcast_to_n(etrsuetp, n)

    if step <= 0:
        tt_values = np.zeros(1, dtype=np.float64)
        step_safe = 0.0
    else:
        n_pts = int(np.floor(t_end / step + 1e-9)) + 1
        tt_values = np.arange(n_pts, dtype=np.float64) * step
        step_safe = float(step)

    return _energy_balance_1l_kernel(
        float(ff),
        a_tem_arr,
        a_rad_arr,
        float(p_tem),
        float(p_rad),
        c_tem_arr,
        c_rad_arr,
        td_ini_arr,
        tm_arr,
        u_arr,
        float(pair),
        hair_arr,
        step_safe,
        ch_arr,
        alb_arr,
        float(kaps),
        float(nis),
        float(tcost),
        etr_arr,
        tt_values,
    )


def _energy_balance_1l_numpy(
    ff: float,
    a_tem: NDArray[np.float64],
    a_rad: NDArray[np.float64] | float,
    p_tem: float,
    p_rad: float,
    c_tem: NDArray[np.float64],
    c_rad: NDArray[np.float64] | float,
    td_ini: NDArray[np.float64],
    tm: NDArray[np.float64],
    u: NDArray[np.float64] | float,
    pair: float,
    hair: NDArray[np.float64],
    t_end: float,
    step: float,
    ch: NDArray[np.float64],
    alb: NDArray[np.float64],
    kaps: float,
    nis: float,
    tcost: float,
    etrsuetp: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Pure-NumPy reference implementation of the 1L solver (used for regression tests)."""
    # Soil-depth scaling
    dz = np.sqrt(nis / ff)
    alpha_param = np.sqrt(365.0)  # dz_deep / dz

    # Physical constants
    sigma = const.STEFAN_BOLTZMANN
    eps_air = const.EMISS_AIR
    eps_soi = const.EMISS_SOIL
    rhow = const.RHO_WATER
    rhoair = const.RHO_AIR
    lv = const.LV
    cp = const.CP_AIR

    # Constant-part linearisation around Tm
    tm_c = tm - 273.15
    tm_cx = tm_c + 243.5
    tm3 = tm**3
    tm4 = tm3 * tm
    kh = ch * u
    rhoaircpkh = rhoair * cp * kh
    p622 = 0.622 * rhoair * lv * etrsuetp / pair * kh
    es4tm3 = 4.0 * sigma * eps_soi * tm3
    ea4tm3 = 4.0 * sigma * eps_air * tm3

    es1 = 6.112 * np.exp(17.67 * tm_c / tm_cx)
    es2 = 17.67 * es1 * (1.0 - tm_c / tm_cx) / tm_cx
    ea1 = es1 * hair
    ea2 = es2 * hair

    den = -kaps / dz - es4tm3 - rhoaircpkh - p622 * es2
    alphadz2 = (1.0 + alpha_param) * dz**2

    d0 = (
        2.0
        * nis
        * (
            tcost / (alpha_param * alphadz2)
            - (p622 * tm * (es2 - ea2) + 3.0 * sigma * (eps_soi - eps_air) * tm4 + p622 * (ea1 - es1))
            / (den * alphadz2)
        )
    )
    d1 = -2.0 * nis * (1.0 - alb) / (den * alphadz2)
    d2 = -2.0 * nis * (p622 * ea2 + rhoaircpkh + ea4tm3) / (den * alphadz2)
    pp = 2.0 * nis * (-alpha_param * (kaps / (den * dz)) - (1.0 + alpha_param)) / (alpha_param * alphadz2)
    pp2 = pp * pp
    pp3 = pp2 * pp

    # Precomputed terms that depend on constants only
    variaroba = (
        c_rad * (1.0 - alb)
        + 3.0 * sigma * eps_soi * tm4
        + (-3.0 * eps_air * sigma + 4.0 * sigma * eps_air) * (c_tem**2) ** 2
        + (rhoaircpkh + p622 * ea2) * c_tem
        + p622 * (ea1 - es1 + (es2 - ea2) * tm)
    )

    if step <= 0:
        tt_values = np.array([0.0])
    else:
        n_pts = int(np.floor(t_end / step + 1e-9)) + 1
        tt_values = np.arange(n_pts, dtype=np.float64) * step

    w = ff
    d1_s = d1 * a_rad
    d2_s = d2 * a_tem
    ppw = pp * w
    ppw2 = ppw * w
    denom_osc = pp3 + ppw2

    ts_out = np.zeros_like(tm)
    td_out = np.zeros_like(tm)
    evp = np.zeros_like(tm)
    evp0 = np.zeros_like(tm)

    for tt in tt_values:
        e_pt = np.exp(pp * tt)

        # Constant-part solution
        td1 = (d1 * c_rad + d2 * c_tem) / pp * (e_pt - 1.0)
        ts1 = (kaps / dz * td1 + variaroba) / (kaps / dz + es4tm3 + rhoaircpkh + p622 * es2)

        # Sinusoidal-part solution (single mode in 1L scheme)
        wttp_r = w * tt + p_rad
        wttp_t = w * tt + p_tem
        td2 = (
            ppw * (d1_s * (e_pt * np.cos(p_rad) - np.cos(wttp_r)) + d2_s * (e_pt * np.cos(p_tem) - np.cos(wttp_t)))
            + (e_pt - 1.0) * d0 * (pp2 + w * w)
            + e_pt * (td_ini * (pp3 + ppw2) + pp2 * (d2_s * np.sin(p_tem) + d1_s * np.sin(p_rad)))
            - pp2 * (d2_s * np.sin(wttp_t) + d1_s * np.sin(wttp_r))
        ) / denom_osc
        ts2 = (
            kaps / dz * td2
            + a_rad * np.sin(wttp_r) * (1.0 - alb)
            + a_tem * np.sin(wttp_t) * (ea4tm3 + rhoaircpkh + p622 * ea2)
        ) / (kaps / dz + es4tm3 + rhoaircpkh + p622 * es2)

        td_out = td1 + td2
        ts_out = ts1 + ts2

        evp_old = evp0
        tair_inst = a_tem * np.sin(wttp_t) + c_tem
        qs_soil = saturation_specific_humidity(ts_out, pair, 0.0)
        qs_air = saturation_specific_humidity(tair_inst, pair, 0.0)
        evp0 = rhoair * lv * etrsuetp * kh * (qs_soil - qs_air * hair)

        evp = evp + (evp_old + evp0) * (step / 2.0)

    # Convert energy to water depth and clean up
    evp = evp / (rhow * lv)
    evp = np.where((evp > 0.0) & np.isfinite(evp), evp, 0.0)

    return ts_out, td_out, evp


# ---------------------------------------------------------------------------
# Sub-stepping for one model timestep
# ---------------------------------------------------------------------------


# Fixed phase offsets
_P_RAD_BASE = -np.pi / 2.0
_P_TEM_BASE = -np.pi / 2.0 - np.pi / 6.0
_EVAP_SUBSTEP = 3600.0 * 2.0  # 2-hour sub-step for evaporation integration


def compute_energy_balance_1l(
    ts: NDArray[np.float64],
    td: NDArray[np.float64],
    td_rise: NDArray[np.float64],
    rs: NDArray[np.float64],
    u: NDArray[np.float64],
    tair_max: NDArray[np.float64],
    tair_min: NDArray[np.float64],
    qair: NDArray[np.float64],
    ch: NDArray[np.float64],
    alb: NDArray[np.float64],
    kaps: float,
    nis: float,
    tcost: float,
    pair: float,
    ctim_s: float,
    ftim_s: float,
    hrise_s: float,
    hset_s: float,
    etrsuetp: float | NDArray[np.float64],
    dt: float,
    reentry: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run the 1L surface energy balance for a single model timestep.

    Day and night sub-periods are solved separately
    (day uses radiation forcing, night does not).

    Args:
        ts: Surface temperature [K] at the beginning of the step (per cell).
        td: Deep-soil temperature [K] at the beginning of the step (per cell).
        td_rise: ``td`` state evaluated at sunrise [K]. Used as the starting
            deep temperature for the day-period re-computation. In the
            initial (``reentry=False``) pass the function updates this field;
            in re-entry it is read-only.
        rs: Incoming shortwave radiation [W/m²] (per cell).
        u: Wind speed [m/s] (per cell).
        tair_max: Maximum air temperature [K] (per cell).
        tair_min: Minimum air temperature [K] (per cell).
        qair: Relative humidity [0-1] (per cell).
        ch: Turbulent exchange coefficient [-] (per cell).
        alb: Surface albedo [-] (per cell).
        kaps: Soil thermal conductivity [W/m/K].
        nis: Soil thermal diffusivity [m²/s].
        tcost: Deep-soil constant temperature [K].
        pair: Air pressure [mb].
        ctim_s: Current-time seconds of day [s] (start of step).
        ftim_s: Future-time seconds of day [s] (end of step = ctim_s + dt).
        hrise_s: Sunrise time [s of day].
        hset_s: Sunset time [s of day].
        etrsuetp: Water-to-energy ET ratio [-]. ``1.0`` for the preliminary
            saturated pass; actual ``ETr/ETp`` for re-entry.
        dt: Model timestep [s].
        reentry: If True, skip the pre-sunrise night sub-period (physics is
            independent of ``etrsuetp``) and start day computation from
            ``td_rise`` rather than the current ``td``.

    Returns:
        ts_new (NDArray[np.float64]): Updated surface temperature [K].
        td_new (NDArray[np.float64]): Updated deep temperature [K].
        etp (NDArray[np.float64]): Potential evapotranspiration over the timestep [m].
        td_rise_new (NDArray[np.float64]): Updated ``td`` at sunrise (for later re-entry).
    """
    omega = 2.0 * np.pi / 86400.0

    # Decompose daily radiation into sinusoidal amplitude + constant.
    if dt > 86400.0 - 1.0:
        a_rad, c_rad = diurnal_radiation_cycle(rs, hrise_s, hset_s, "average")
    else:
        a_rad, c_rad = diurnal_radiation_cycle(rs, ctim_s, hset_s, "instant")

    # Air-temperature decomposition — constant over the day.
    a_tem = (tair_max - tair_min) / 2.0
    c_tem = (tair_max + tair_min) / 2.0  # tair_lin

    ts_out = ts.copy()
    td_out = td.copy()
    etp = np.zeros_like(ts)
    td_rise_out = td_rise.copy()

    def _night_call(
        td_in: NDArray[np.float64], t0: float, t1: float, kwargs: dict
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return energy_balance_1l(
            omega,
            a_tem,
            0.0,
            _P_TEM_BASE + omega * t0,
            _P_RAD_BASE + omega * t0,
            c_tem,
            0.0,
            td_in,
            c_tem,
            0.0,
            pair,
            qair,
            t1 - t0,
            t1 - t0,
            ch,
            alb,
            kaps,
            nis,
            tcost,
            kwargs["etrsuetp"],
        )

    def _day_call(
        td_in: NDArray[np.float64], t0: float, t1: float, evap_step: float, kwargs: dict
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return energy_balance_1l(
            omega,
            a_tem,
            a_rad,
            _P_TEM_BASE + omega * t0,
            _P_RAD_BASE + omega * t0,
            c_tem,
            c_rad,
            td_in,
            c_tem,
            u,
            pair,
            qair,
            t1 - t0,
            evap_step,
            ch,
            alb,
            kaps,
            nis,
            tcost,
            kwargs["etrsuetp"],
        )

    kw = {"etrsuetp": etrsuetp}

    if ctim_s < hrise_s:
        if ftim_s > hrise_s:
            # ctim < hrise < ftim  (step crosses sunrise)
            if not reentry:
                # Pre-sunrise night sub-period (independent of etrsuetp)
                ts_out, td_out, _ = _night_call(td_out, ctim_s, hrise_s, kw)
                td_rise_out = td_out.copy()
                day_start_td = td_out
            else:
                day_start_td = td_rise.copy()

            if ftim_s > hset_s:
                # ctim < hrise < hset < ftim  (full daylight inside step)
                ts_out, td_out, etp = _day_call(day_start_td, hrise_s, hset_s, _EVAP_SUBSTEP, kw)
                ts_out, td_out, _ = _night_call(td_out, hset_s, ftim_s, kw)
            else:
                # ctim < hrise < ftim < hset
                evap_step = min(ftim_s - hrise_s, _EVAP_SUBSTEP)
                ts_out, td_out, etp = _day_call(day_start_td, hrise_s, ftim_s, evap_step, kw)
        else:
            # ctim < ftim < hrise  (night only before sunrise)
            if reentry:
                # No radiation so etrsuetp has no impact — skip.
                return ts_out, td_out, etp, td_rise_out
            ts_out, td_out, _ = _night_call(td_out, ctim_s, ftim_s, kw)
    elif ctim_s < hset_s:
        if not reentry:
            td_rise_out = td_out.copy()
            day_start_td = td_out
        else:
            day_start_td = td_rise.copy()

        if ftim_s > hset_s:
            # hrise < ctim < hset < ftim  (step crosses sunset)
            evap_step = min(hset_s - ctim_s, _EVAP_SUBSTEP)
            ts_out, td_out, etp = _day_call(day_start_td, ctim_s, hset_s, evap_step, kw)
            ts_out, td_out, _ = _night_call(td_out, hset_s, ftim_s, kw)
        else:
            # hrise < ctim < ftim < hset  (daylight only)
            evap_step = min(ftim_s - ctim_s, _EVAP_SUBSTEP)
            ts_out, td_out, etp = _day_call(day_start_td, ctim_s, ftim_s, evap_step, kw)
    else:
        # hset < ctim < ftim  (night only after sunset)
        if reentry:
            return ts_out, td_out, etp, td_rise_out
        ts_out, td_out, _ = _night_call(td_out, ctim_s, ftim_s, kw)

    return ts_out, td_out, etp, td_rise_out
