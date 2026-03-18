"""Load observed discharge data and align with simulation time steps."""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mobidic.calibration.config import ObservationGroup


def load_observations(
    obs_group: ObservationGroup,
    base_path: Path,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load observed data from CSV file for an observation group.

    Args:
        obs_group: Observation group configuration.
        base_path: Base directory for resolving relative file paths.
        start_date: Optional start date to filter observations.
        end_date: Optional end date to filter observations.

    Returns:
        DataFrame with 'time' (datetime) and 'value' columns, sorted by time.
    """
    obs_path = Path(obs_group.obs_file)
    if not obs_path.is_absolute():
        obs_path = base_path / obs_path

    if not obs_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    logger.info(f"Loading observations from: {obs_path}")

    df = pd.read_csv(obs_path)

    if obs_group.time_column not in df.columns:
        raise ValueError(f"Time column '{obs_group.time_column}' not found in {obs_path}")
    if obs_group.value_column not in df.columns:
        raise ValueError(f"Value column '{obs_group.value_column}' not found in {obs_path}")

    result = pd.DataFrame(
        {
            "time": pd.to_datetime(df[obs_group.time_column]),
            "value": pd.to_numeric(df[obs_group.value_column], errors="coerce"),
        }
    )
    result = result.dropna(subset=["value"]).sort_values("time").reset_index(drop=True)

    # Filter by calibration period
    if start_date is not None:
        result = result[result["time"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        result = result[result["time"] <= pd.Timestamp(end_date)]

    result = result.reset_index(drop=True)
    logger.info(f"Loaded {len(result)} observations for group '{obs_group.name}' (reach {obs_group.reach_id})")

    return result


def align_observations_to_simulation(
    obs_df: pd.DataFrame,
    sim_times: list | np.ndarray | pd.DatetimeIndex,
    tolerance_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align observed data to simulation time steps using nearest-neighbor matching.

    Args:
        obs_df: DataFrame with 'time' and 'value' columns.
        sim_times: Simulation time stamps.
        tolerance_seconds: Maximum allowed time difference in seconds for matching.
            If None, uses half the minimum simulation time step.

    Returns:
        Tuple of (sim_indices, obs_values, obs_times):
            - sim_indices: Indices into sim_times where observations match.
            - obs_values: Observed values at matched times.
            - obs_times: Observed times at matched times.
    """
    sim_times = pd.DatetimeIndex(sim_times)

    if tolerance_seconds is None:
        if len(sim_times) > 1:
            dt = (sim_times[1] - sim_times[0]).total_seconds()
            tolerance_seconds = dt / 2
        else:
            tolerance_seconds = 3600  # Default 1 hour

    tolerance = pd.Timedelta(seconds=tolerance_seconds)

    sim_indices = []
    obs_values = []
    obs_times_matched = []

    for _, row in obs_df.iterrows():
        obs_time = row["time"]
        # Find nearest simulation time
        diffs = np.abs(sim_times - obs_time)
        nearest_idx = diffs.argmin()

        if diffs[nearest_idx] <= tolerance:
            sim_indices.append(nearest_idx)
            obs_values.append(row["value"])
            obs_times_matched.append(obs_time)

    return np.array(sim_indices), np.array(obs_values), np.array(obs_times_matched)
