"""Report I/O for MOBIDIC simulations.

This module provides functions to save discharge time series and other
simulation outputs to Parquet format for efficient storage and analysis.
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger


def save_discharge_report(
    discharge_timeseries: np.ndarray,
    time_stamps: list[datetime],
    network: "GeoDataFrame",  # noqa: F821
    output_path: str | Path,
    reach_selection: str = "all",
    selected_reaches: list[int] | None = None,
    add_metadata: dict | None = None,
    output_format: str = "Parquet",
) -> None:
    """
    Save discharge time series to file (Parquet or CSV).

    Args:
        discharge_timeseries: 2D array of discharge values [m³/s], shape (n_timesteps, n_reaches)
        time_stamps: List of datetime objects for each time step
        network: River network GeoDataFrame with reach metadata
        output_path: Path to output file
        reach_selection: Reach selection mode: "all", "list", or "outlets"
        selected_reaches: List of reach IDs (mobidic_id) to include (used if reach_selection="list")
        add_metadata: Additional metadata to save (optional, saved as JSON in separate file)
        output_format: Output format: "Parquet" or "csv" (default: "Parquet")

    Examples:
        >>> from mobidic import Simulation
        >>> sim = Simulation(gisdata, forcing, config)
        >>> results = sim.run("2020-01-01", "2020-12-31")
        >>> from mobidic.io import save_discharge_report
        >>> save_discharge_report(
        ...     results.time_series["discharge"],
        ...     results.time_series["time"],
        ...     sim.network,
        ...     "Arno_discharge.parquet",
        ...     reach_selection="outlets",
        ...     output_format="Parquet"
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which reaches to include
    if reach_selection == "all":
        reach_indices = network["mobidic_id"].values
        logger.debug(f"Saving all {len(reach_indices)} reaches")
    elif reach_selection == "outlets":
        # Select reaches with no downstream reach (outlets)
        outlet_mask = pd.isna(network["downstream"])
        reach_indices = network.loc[outlet_mask, "mobidic_id"].values
        logger.debug(f"Saving {len(reach_indices)} outlet reaches")
    elif reach_selection == "list":
        if selected_reaches is None:
            raise ValueError("selected_reaches must be provided when reach_selection='list'")
        reach_indices = np.array(selected_reaches)
        logger.debug(f"Saving {len(reach_indices)} selected reaches")
    else:
        raise ValueError(f"Invalid reach_selection: {reach_selection}. Must be 'all', 'outlets', or 'list'")

    # Check that indices are valid
    max_reach_id = len(network) - 1
    if np.any(reach_indices > max_reach_id):
        invalid_ids = reach_indices[reach_indices > max_reach_id]
        raise ValueError(f"Invalid reach IDs: {invalid_ids}. Maximum reach ID is {max_reach_id}")

    # Extract discharge for selected reaches
    discharge_selected = discharge_timeseries[:, reach_indices.astype(int)]

    # Create DataFrame with time as index
    df = pd.DataFrame(
        discharge_selected,
        index=pd.DatetimeIndex(time_stamps, name="time"),
        columns=[f"reach_{rid:04d}" for rid in reach_indices],
    )

    # Add reach metadata as separate columns (optional, can be joined later with network data)
    # For now, keep it simple and just save discharge values

    # Save to file based on output_format
    if output_format.lower() == "csv":
        df.to_csv(output_path, index=True)
        logger.success(f"Discharge report saved to {output_path} (CSV format)")
    elif output_format.lower() == "parquet":
        df.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=True,
        )
        logger.success(f"Discharge report saved to {output_path} (Parquet format)")
    else:
        raise ValueError(f"Invalid output_format: {output_format}. Must be 'csv' or 'Parquet'")
    logger.debug(
        f"File size: {output_path.stat().st_size / 1024:.2f} KB, "
        f"Time steps: {len(time_stamps)}, "
        f"Reaches: {len(reach_indices)}"
    )

    # Save metadata to JSON file if provided
    if add_metadata:
        metadata_path = output_path.with_suffix(".json")
        import json

        metadata = {
            "reach_selection": reach_selection,
            "n_reaches": int(len(reach_indices)),
            "n_timesteps": int(len(time_stamps)),
            "start_time": time_stamps[0].isoformat(),
            "end_time": time_stamps[-1].isoformat(),
            "reach_ids": reach_indices.astype(int).tolist(),
            **add_metadata,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Metadata saved to {metadata_path}")


def load_discharge_report(input_path: str | Path) -> pd.DataFrame:
    """
    Load discharge time series from Parquet file.

    Args:
        input_path: Path to input Parquet file

    Returns:
        DataFrame with time as index and reach discharge as columns

    Raises:
        FileNotFoundError: If input file does not exist

    Examples:
        >>> from mobidic.io import load_discharge_report
        >>> df = load_discharge_report("Arno_discharge.parquet")
        >>> print(df.head())
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Report file not found: {input_path}")

    logger.info(f"Loading discharge report from Parquet: {input_path}")

    # Load from Parquet
    df = pd.read_parquet(input_path, engine="pyarrow")

    logger.success(f"Discharge report loaded: {len(df)} time steps, {len(df.columns)} reaches")

    return df


def save_lateral_inflow_report(
    lateral_inflow_timeseries: np.ndarray,
    time_stamps: list[datetime],
    network: "GeoDataFrame",  # noqa: F821
    output_path: str | Path,
    reach_selection: str = "all",
    selected_reaches: list[int] | None = None,
    output_format: str = "Parquet",
) -> None:
    """
    Save lateral inflow time series to file (Parquet or CSV).

    Similar to save_discharge_report but for lateral inflows.

    Args:
        lateral_inflow_timeseries: 2D array of lateral inflow [m³/s], shape (n_timesteps, n_reaches)
        time_stamps: List of datetime objects for each time step
        network: River network GeoDataFrame with reach metadata
        output_path: Path to output file
        reach_selection: Reach selection mode: "all", "list", or "outlets"
        selected_reaches: List of reach IDs to include (used if reach_selection="list")
        output_format: Output format: "Parquet" or "csv" (default: "Parquet")

    Examples:
        >>> save_lateral_inflow_report(
        ...     lateral_inflow_ts,
        ...     time_stamps,
        ...     network,
        ...     "lateral_inflow.parquet",
        ...     output_format="Parquet"
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use same logic as discharge report
    if reach_selection == "all":
        reach_indices = network["mobidic_id"].values
    elif reach_selection == "outlets":
        outlet_mask = pd.isna(network["downstream"])
        reach_indices = network.loc[outlet_mask, "mobidic_id"].values
    elif reach_selection == "list":
        if selected_reaches is None:
            raise ValueError("selected_reaches must be provided when reach_selection='list'")
        reach_indices = np.array(selected_reaches)
    else:
        raise ValueError(f"Invalid reach_selection: {reach_selection}")

    # Extract lateral inflow for selected reaches
    lateral_inflow_selected = lateral_inflow_timeseries[:, reach_indices.astype(int)]

    # Create DataFrame
    df = pd.DataFrame(
        lateral_inflow_selected,
        index=pd.DatetimeIndex(time_stamps, name="time"),
        columns=[f"reach_{rid:04d}" for rid in reach_indices],
    )

    # Save to file based on output_format
    if output_format.lower() == "csv":
        df.to_csv(output_path, index=True)
        logger.success(f"Lateral inflow report saved to {output_path} (CSV format)")
    elif output_format.lower() == "parquet":
        df.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=True,
        )
        logger.success(f"Lateral inflow report saved to {output_path} (Parquet format)")
    else:
        raise ValueError(f"Invalid output_format: {output_format}. Must be 'csv' or 'Parquet'")
