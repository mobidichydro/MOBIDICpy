"""Derived metric computation for calibration objective functions.

Metrics are computed from simulated and observed time series and can be passed
to PEST++ as pseudo-observations for custom objective functions.

Includes custom implementations (nse, nse_log, pbias, peak_error) and the full
catalog of HydroErr metrics exposed via the ``he`` module
(see https://hydroerr.readthedocs.io/en/stable/list_of_metrics.html).
"""

import numpy as np
import HydroErr as he


######## CUSTOM METRIC IMPLEMENTATIONS ########


def nse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency.

    NSE = 1 - sum((sim - obs)^2) / sum((obs - mean(obs))^2)
    Perfect score: 1.0

    Args:
        simulated: Simulated values.
        observed: Observed values.

    Returns:
        NSE value (range: -inf to 1.0).
    """
    obs_mean = np.mean(observed)
    numerator = np.sum((simulated - observed) ** 2)
    denominator = np.sum((observed - obs_mean) ** 2)
    if denominator == 0:
        return np.nan
    return 1.0 - numerator / denominator


def nse_log(simulated: np.ndarray, observed: np.ndarray, eps: float = 1e-6) -> float:
    """NSE on log-transformed flows (emphasizes low flows).

    Args:
        simulated: Simulated values.
        observed: Observed values.
        eps: Small constant to avoid log(0).

    Returns:
        NSE of log-transformed values.
    """
    return nse(np.log(simulated + eps), np.log(observed + eps))


def pbias(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Percent bias as a fraction (not percentage).

    pbias = sum(sim - obs) / sum(obs)
    Perfect score: 0.0

    Args:
        simulated: Simulated values.
        observed: Observed values.

    Returns:
        Percent bias as fraction (e.g., 0.05 = 5% overestimation).
    """
    obs_sum = np.sum(observed)
    if obs_sum == 0:
        return np.nan
    return np.sum(simulated - observed) / obs_sum


def peak_error(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Relative peak error.

    peak_error = (max(sim) - max(obs)) / max(obs)
    Perfect score: 0.0

    Args:
        simulated: Simulated values.
        observed: Observed values.

    Returns:
        Relative peak error.
    """
    obs_peak = np.max(observed)
    if obs_peak == 0:
        return np.nan
    return (np.max(simulated) - obs_peak) / obs_peak


######## HYDROERR METRIC CATALOG ########
# Maps each HydroErr function name to its perfect-match target value.
_HYDROERR_TARGETS: dict[str, float] = {
    # Efficiency / agreement / correlation metrics (perfect = 1.0)
    "nse": 1.0,
    "nse_mod": 1.0,
    "nse_rel": 1.0,
    "kge_2009": 1.0,
    "kge_2012": 1.0,
    "lm_index": 1.0,
    "ve": 1.0,
    "d": 1.0,
    "d1": 1.0,
    "d1_p": 1.0,
    "dmod": 1.0,
    "dr": 1.0,
    "drel": 1.0,
    "r_squared": 1.0,
    "pearson_r": 1.0,
    "spearman_r": 1.0,
    "acc": 1.0,
    "mb_r": 1.0,
    "watt_m": 1.0,
    "g_mean_diff": 1.0,
    # Error metrics (perfect = 0.0)
    "rmse": 0.0,
    "mae": 0.0,
    "me": 0.0,
    "mse": 0.0,
    "mle": 0.0,
    "male": 0.0,
    "mape": 0.0,
    "mapd": 0.0,
    "maape": 0.0,
    "mase": 0.0,
    "mdae": 0.0,
    "mde": 0.0,
    "mdse": 0.0,
    "msle": 0.0,
    "rmsle": 0.0,
    "smape1": 0.0,
    "smape2": 0.0,
    "ed": 0.0,
    "ned": 0.0,
    "irmse": 0.0,
    "nrmse_iqr": 0.0,
    "nrmse_mean": 0.0,
    "nrmse_range": 0.0,
    "mean_var": 0.0,
    # Spectral metrics (perfect = 0.0)
    "sa": 0.0,
    "sc": 0.0,
    "sga": 0.0,
    "sid": 0.0,
    # H-series error metrics (perfect = 0.0)
    "h1_mhe": 0.0,
    "h1_mahe": 0.0,
    "h1_rmshe": 0.0,
    "h2_mhe": 0.0,
    "h2_mahe": 0.0,
    "h2_rmshe": 0.0,
    "h3_mhe": 0.0,
    "h3_mahe": 0.0,
    "h3_rmshe": 0.0,
    "h4_mhe": 0.0,
    "h4_mahe": 0.0,
    "h4_rmshe": 0.0,
    "h5_mhe": 0.0,
    "h5_mahe": 0.0,
    "h5_rmshe": 0.0,
    "h6_mhe": 0.0,
    "h6_mahe": 0.0,
    "h6_rmshe": 0.0,
    "h7_mhe": 0.0,
    "h7_mahe": 0.0,
    "h7_rmshe": 0.0,
    "h8_mhe": 0.0,
    "h8_mahe": 0.0,
    "h8_rmshe": 0.0,
    "h10_mhe": 0.0,
    "h10_mahe": 0.0,
    "h10_rmshe": 0.0,
}


# Registry mapping metric names to (function, target_value).
# Add custom implementations
METRIC_REGISTRY: dict[str, tuple] = {
    "nse": (nse, 1.0),
    "nse_log": (nse_log, 1.0),
    "pbias": (pbias, 0.0),
    "peak_error": (peak_error, 0.0),
    "kge": (he.kge_2012, 1.0),
}

# Add HydroErr metrics
for _name, _target in _HYDROERR_TARGETS.items():
    METRIC_REGISTRY.setdefault(_name, (getattr(he, _name), _target))


def compute_metrics(
    simulated: np.ndarray,
    observed: np.ndarray,
    metric_names: list[str],
) -> dict[str, float]:
    """Compute multiple metrics for a sim/obs pair.

    Args:
        simulated: Simulated values.
        observed: Observed values.
        metric_names: List of metric names to compute.

    Returns:
        Dict mapping metric name to computed value.
    """
    results = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{name}'. Available: {sorted(METRIC_REGISTRY.keys())}")
        func, _ = METRIC_REGISTRY[name]
        results[name] = func(simulated, observed)
    return results
