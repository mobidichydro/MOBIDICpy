"""
MOBIDIC Validation Python vs MATLAB discharge and lateral inflow results

This script compares discharge and lateral inflow outputs from the Python
implementation against the MATLAB reference implementation for the Arno River basin.

It is required to run 01a_run_example_Arno_daily.py first to generate the output files.

Usage:
    python examples/02-daily-balance-Arno-basin/01a_run_example_Arno_daily.py
    python examples/02-daily-balance-Arno-basin/01b_run_example_Arno_daily_plots.py

The script will:
1. Load Python output files (Parquet format)
2. Load MATLAB output files (CSV format)
3. Account for +1 offset in MATLAB reach IDs
4. Plot time series comparison for each reach
5. Calculate and display error metrics (RMSE, bias)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def calculate_metrics(reference, simulated):
    """Calculate performance metrics comparing two time series.

    Args:
        reference: Reference values (MATLAB)
        simulated: Simulated values (Python)

    Returns:
        dict: Dictionary with RMSE and bias metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(reference) | np.isnan(simulated))
    obs = reference[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return {"RMSE": np.nan, "bias": np.nan}

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((obs - sim) ** 2))

    # Bias
    bias = np.mean(sim - obs)

    return {"RMSE": rmse, "bias": bias}


def compare_variable(
    output_dir: Path,
    matlab_dir: Path,
    variable_name: str,
    python_pattern: str,
    matlab_filename: str,
    python_prefix: str,
    matlab_prefix: str,
    unit: str,
    variable_label: str,
):
    """Compare a variable (discharge or lateral inflow) between Python and MATLAB.

    Args:
        output_dir: Directory containing Python output files
        matlab_dir: Directory containing MATLAB output files
        variable_name: Name of variable (e.g., "Discharge", "Lateral Inflow")
        python_pattern: Glob pattern for Python files (e.g., "discharge*.parquet")
        matlab_filename: MATLAB CSV filename (e.g., "discharge.csv")
        python_prefix: Prefix for Python column names (e.g., "reach")
        matlab_prefix: Prefix for MATLAB column names (e.g., "Q_reach", "qL_reach")
        unit: Unit string (e.g., "m³/s")
        variable_label: Y-axis label (e.g., "Discharge (m³/s)")

    Returns:
        tuple: (mean_rmse, n_reaches, n_timesteps)
    """
    print("=" * 80)
    print(f"MOBIDIC - Python vs MATLAB {variable_name} Comparison")
    print("=" * 80)
    print()

    # Find Python output file
    parquet_files = list(output_dir.glob(python_pattern))
    if not parquet_files:
        raise FileNotFoundError(f"No {python_pattern} files found in {output_dir}")
    parquet_file = parquet_files[0]

    # MATLAB output file
    matlab_file = matlab_dir / matlab_filename

    if not matlab_file.exists():
        raise FileNotFoundError(f"MATLAB {variable_name} file not found: {matlab_file}")

    print(f"Python output: {parquet_file.name}")
    print(f"MATLAB output: {matlab_file.name}")
    print()

    # =========================================================================
    # Load data
    # =========================================================================
    print("Loading data...")

    # Load Python output (Parquet)
    df_python = pd.read_parquet(parquet_file)
    print(f"  Python data shape: {df_python.shape}")
    print(f"  Time range: {df_python.index[0]} to {df_python.index[-1]}")

    # Load MATLAB output (CSV)
    df_matlab = pd.read_csv(matlab_file)
    df_matlab["Time"] = pd.to_datetime(df_matlab["Time"], format="%d-%b-%Y %H:%M")
    df_matlab.set_index("Time", inplace=True)
    print(f"  MATLAB data shape: {df_matlab.shape}")
    print(f"  Time range: {df_matlab.index[0]} to {df_matlab.index[-1]}")
    print()

    # =========================================================================
    # Match reaches accounting for +1 offset
    # =========================================================================
    print("Matching reaches (MATLAB reach_id = Python reach_id + 1)...")

    # Extract reach IDs from column names
    # Python: reach_0313 -> 313
    # MATLAB: Q_reach_314 or qL_reach_314 -> 314
    python_reaches = {int(col.split("_")[1]): col for col in df_python.columns}
    matlab_reaches = {int(col.split("_")[-1]): col for col in df_matlab.columns}

    # Find matching reaches
    matched_reaches = []
    for python_id, python_col in python_reaches.items():
        matlab_id = python_id + 1  # MATLAB ID is Python ID + 1
        if matlab_id in matlab_reaches:
            matched_reaches.append(
                {
                    "python_id": python_id,
                    "matlab_id": matlab_id,
                    "python_col": python_col,
                    "matlab_col": matlab_reaches[matlab_id],
                }
            )

    print(f"  Found {len(matched_reaches)} matching reaches:")
    for match in matched_reaches:
        print(f"    Python {python_prefix}_{match['python_id']:04d} <-> MATLAB {matlab_prefix}_{match['matlab_id']}")
    print()

    if not matched_reaches:
        print("ERROR: No matching reaches found!")
        return None, 0, 0

    # =========================================================================
    # Align time series
    # =========================================================================
    print("Aligning time series...")

    # Find common time range
    start_time = max(df_python.index[0], df_matlab.index[0])
    end_time = min(df_python.index[-1], df_matlab.index[-1])

    # Align both dataframes to common time range
    df_python_aligned = df_python.loc[start_time:end_time]
    df_matlab_aligned = df_matlab.loc[start_time:end_time]

    # Resample to ensure exact time alignment (in case of floating point differences)
    common_index = df_python_aligned.index.intersection(df_matlab_aligned.index)
    df_python_aligned = df_python_aligned.loc[common_index]
    df_matlab_aligned = df_matlab_aligned.loc[common_index]

    print(f"  Common time range: {start_time} to {end_time}")
    print(f"  Number of time steps: {len(common_index)}")
    print()

    # =========================================================================
    # Calculate metrics
    # =========================================================================
    print("Calculating performance metrics...")
    print()

    metrics_summary = []
    for match in matched_reaches:
        python_series = df_python_aligned[match["python_col"]]
        matlab_series = df_matlab_aligned[match["matlab_col"]]

        metrics = calculate_metrics(matlab_series.values, python_series.values)
        metrics["reach_id"] = match["python_id"]
        metrics_summary.append(metrics)

        print(f"  Reach {match['python_id']:04d}:")
        print(f"    RMSE: {metrics['RMSE']:7.3f} {unit}")
        print(f"    Bias: {metrics['bias']:7.3f} {unit}")
        print()

    # Overall metrics
    all_rmse = [m["RMSE"] for m in metrics_summary if not np.isnan(m["RMSE"])]

    print("  Overall average:")
    print(f"    Mean RMSE: {np.mean(all_rmse):.3f} {unit}")
    print()

    # =========================================================================
    # Create plots
    # =========================================================================
    print("Creating plots...")

    n_reaches = len(matched_reaches)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 3 * n_reaches))
    gs = GridSpec(n_reaches, 2, figure=fig, width_ratios=[3, 1], hspace=0.3, wspace=0.3)

    fig.suptitle(
        f"MOBIDIC: Python vs MATLAB {variable_name} comparison - Arno River basin",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for i, match in enumerate(matched_reaches):
        python_series = df_python_aligned[match["python_col"]]
        matlab_series = df_matlab_aligned[match["matlab_col"]]
        time_index = df_python_aligned.index

        metrics = metrics_summary[i]

        # Time series plot
        ax_ts = fig.add_subplot(gs[i, 0])
        ax_ts.plot(time_index, matlab_series, "b-", linewidth=1.5, alpha=0.7, label="MATLAB (reference)")
        ax_ts.plot(time_index, python_series, "r--", linewidth=1.0, alpha=0.8, label="Python (MOBIDICpy)")

        # Only show x-axis label and rotation on the last plot
        if i == n_reaches - 1:
            ax_ts.set_xlabel("Time")
            ax_ts.tick_params(axis="x", rotation=45)
        else:
            ax_ts.tick_params(labelbottom=False)

        ax_ts.set_ylabel(variable_label)
        ax_ts.set_title(f"Reach {match['python_id']:04d} - Time Series " + f"(RMSE={metrics['RMSE']:.6f} {unit})")
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(loc="best")

        # Scatter plot
        ax_scatter = fig.add_subplot(gs[i, 1])
        ax_scatter.scatter(matlab_series, python_series, alpha=0.5, s=20, c="blue")

        # Add 1:1 line
        min_val = min(matlab_series.min(), python_series.min())
        max_val = max(matlab_series.max(), python_series.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, label="1:1 line")

        ax_scatter.set_xlabel(f"MATLAB {variable_label}")
        ax_scatter.set_ylabel(f"Python {variable_label}")
        ax_scatter.set_title(f"Reach {match['python_id']:04d} - Scatter")
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.legend(loc="best")

        # Make scatter plot square
        ax_scatter.set_aspect("equal", adjustable="box")

    plt.show()

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 80)
    print(f"{variable_name} Comparison Summary")
    print("=" * 80)
    print(f"Number of reaches compared: {len(matched_reaches)}")
    print(f"Time steps: {len(common_index)}")
    print(f"Mean RMSE: {np.mean(all_rmse):.6f} {unit}")
    print()

    return np.mean(all_rmse), len(matched_reaches), len(common_index)


def compare_observed_discharge(output_dir: Path, data_dir: Path, observed_file: str, mobidic_id: int):
    """Compare observed discharge with Python simulation for a specific reach.

    Args:
        output_dir: Directory containing Python output files
        data_dir: Directory containing observed data files
        observed_file: Filename of observed discharge data (Parquet)
        mobidic_id: MOBIDIC ID of the reach to compare

    Returns:
        dict: Performance metrics (RMSE, bias, NSE)
    """
    print()
    print("=" * 80)
    print(f"MOBIDIC - Observed vs simulated discharge (Reach {mobidic_id:04d})")
    print("=" * 80)
    print()

    # Load observed data
    observed_path = data_dir / observed_file
    if not observed_path.exists():
        raise FileNotFoundError(f"Observed discharge file not found: {observed_path}")

    df_observed = pd.read_parquet(observed_path)
    print(f"Observed data: {observed_file}")
    print(f"  Shape: {df_observed.shape}")
    print(f"  Time range: {df_observed['time'].min()} to {df_observed['time'].max()}")
    print()

    # Load Python discharge output
    parquet_files = list(output_dir.glob("discharge*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No discharge files found in {output_dir}")
    parquet_file = parquet_files[0]

    df_python = pd.read_parquet(parquet_file)
    print(f"Python output: {parquet_file.name}")
    print(f"  Shape: {df_python.shape}")
    print(f"  Time range: {df_python.index[0]} to {df_python.index[-1]}")
    print()

    # Find the reach column
    reach_col = f"reach_{mobidic_id:04d}"
    if reach_col not in df_python.columns:
        raise ValueError(
            f"Reach {reach_col} not found in Python output. Available reaches: {df_python.columns.tolist()}"
        )

    # Align time series
    print("Aligning time series...")
    df_observed["time"] = pd.to_datetime(df_observed["time"])
    df_observed.set_index("time", inplace=True)

    # Find common time range
    start_time = max(df_python.index[0], df_observed.index[0])
    end_time = min(df_python.index[-1], df_observed.index[-1])

    df_python_aligned = df_python.loc[start_time:end_time, reach_col]
    df_observed_aligned = df_observed.loc[start_time:end_time, "Q"]

    # Resample to ensure exact alignment
    common_index = df_python_aligned.index.intersection(df_observed_aligned.index)
    df_python_aligned = df_python_aligned.loc[common_index]
    df_observed_aligned = df_observed_aligned.loc[common_index]

    print(f"  Common time range: {start_time} to {end_time}")
    print(f"  Number of time steps: {len(common_index)}")
    print()

    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = calculate_metrics(df_observed_aligned.values, df_python_aligned.values)

    print(f"  RMSE: {metrics['RMSE']:.3f} m³/s")
    print(f"  Bias: {metrics['bias']:.3f} m³/s")
    print()

    # Create plots
    print("Creating plots...")
    plt.figure(figsize=(10, 4))

    plt.plot(common_index, df_observed_aligned, "b-", linewidth=1.5, alpha=0.7, label="Observed")
    plt.plot(common_index, df_python_aligned, "r--", linewidth=1.0, alpha=0.8, label="MOBIDICpy (uncalibrated)")

    plt.xlabel("Time")
    plt.ylabel("Discharge (m³/s)")
    plt.title(f"Observed vs MOBIDICpy - Reach {mobidic_id:04d} (Nave di Rosano)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    print("=" * 80)
    print("Observed vs MOBIDIC Discharge comparison summary")
    print("=" * 80)
    print(f"Reach: {mobidic_id:04d}")
    print(f"Time steps: {len(common_index)}")
    print(f"RMSE: {metrics['RMSE']:.3f} m³/s")
    print(f"Bias: {metrics['bias']:.3f} m³/s")
    print("=" * 80)

    return metrics


def main():
    """Main function to compare Python and MATLAB outputs."""

    # Define paths
    example_dir = Path(__file__).parent
    output_dir = example_dir / "output"
    matlab_dir = Path(__file__).parent.parent / "datasets" / "Arno" / "matlab" / "output" / "Arno_daily_balance_2017_2018"
    data_dir = Path(__file__).parent.parent / "datasets" / "Arno" / "data"

    # Compare discharge
    discharge_results = compare_variable(
        output_dir=output_dir,
        matlab_dir=matlab_dir,
        variable_name="Discharge",
        python_pattern="discharge*.parquet",
        matlab_filename="discharge.csv",
        python_prefix="reach",
        matlab_prefix="Q_reach",
        unit="m³/s",
        variable_label="Discharge (m³/s)",
    )

    # Compare lateral inflow
    lateral_results = compare_variable(
        output_dir=output_dir,
        matlab_dir=matlab_dir,
        variable_name="Lateral Inflow",
        python_pattern="lateral_inflow*.parquet",
        matlab_filename="lateral_inflow.csv",
        python_prefix="reach",
        matlab_prefix="qL_reach",
        unit="m³/s",
        variable_label="Lateral Inflow (m³/s)",
    )

    # Compare with observed discharge at Nave di Rosano (reach 278)
    observed_results = compare_observed_discharge(
        output_dir=output_dir,
        data_dir=data_dir,
        observed_file="Q_TOS01004659_2017_2018.parquet",
        mobidic_id=278,
    )

    # Overall summary
    print()
    print("=" * 80)
    print("OVERALL COMPARISON SUMMARY")
    print("=" * 80)
    if discharge_results[0] is not None:
        print(f"Discharge:      Mean RMSE = {discharge_results[0]:.6f} m³/s")
    if lateral_results[0] is not None:
        print(f"Lateral Inflow: Mean RMSE = {lateral_results[0]:.6f} m³/s")
    if observed_results is not None:
        print(f"Observed vs MOBIDICpy (Reach 278): RMSE = {observed_results['RMSE']:.3f} m³/s")
    print("=" * 80)


if __name__ == "__main__":
    main()
