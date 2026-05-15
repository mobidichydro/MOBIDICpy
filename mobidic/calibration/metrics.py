"""Derived metric computation for calibration objective functions.

Metrics are computed from simulated and observed time series and can be passed
to PEST++ as pseudo-observations for custom objective functions.
"""

import numpy as np
import HydroErr as he


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


def rmse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Root Mean Square Error.

    Perfect score: 0.0

    Args:
        simulated: Simulated values.
        observed: Observed values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(np.mean((simulated - observed) ** 2)))


def kge(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Kling-Gupta Efficiency.

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    where r = correlation, alpha = std(sim)/std(obs), beta = mean(sim)/mean(obs)
    Perfect score: 1.0

    Args:
        simulated: Simulated values.
        observed: Observed values.

    Returns:
        KGE value (range: -inf to 1.0).
    """
    obs_mean = np.mean(observed)
    sim_mean = np.mean(simulated)
    obs_std = np.std(observed)
    sim_std = np.std(simulated)

    if obs_std == 0 or obs_mean == 0:
        return np.nan

    # Correlation
    r = np.corrcoef(simulated, observed)[0, 1]
    # Variability ratio
    alpha = sim_std / obs_std
    # Bias ratio
    beta = sim_mean / obs_mean

    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def kge_2012(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Kling-Gupta Efficiency (2012)"""
    return he.kge_2012(simulated, observed)

def mle(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Mean Logarith Error"""
    return he.mle(simulated, observed)



# Registry mapping metric names to (function, target_value)
METRIC_REGISTRY: dict[str, tuple] = {
    "nse": (nse, 1.0),
    "nse_log": (nse_log, 1.0),
    "pbias": (pbias, 0.0),
    "peak_error": (peak_error, 0.0),
    "rmse": (rmse, 0.0),
    "kge": (kge, 1.0),
    "kge_2012": (kge_2012, 1.0), 
    "mle": (mle, 0.0),   
}


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
