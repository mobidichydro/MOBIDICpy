"""
MOBIDIC Example - Arno River Basin Simulation with Restart

This script demonstrates the simulation restart capability:
1. Run simulation from start to a middle point (e.g., 50% of period)
2. Save intermediate states
3. Load the last saved state from the first run
4. Restart simulation from the saved state and continue to the end
5. Compare results with a continuous run

The example shows how to:
- Set initial state from a previous simulation using set_initial_state()
- Continue simulations after interruption or for multi-stage modeling
- Verify that restarted simulations produce identical results to continuous runs

Usage:
    python examples/run_example_Arno_restart.py

    Options (modify script directly):
    - restart_fraction: Fraction of simulation period for first run (default: 0.5)
"""

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mobidic import (
    load_config,
    run_preprocessing,
    save_gisdata,
    save_network,
    load_gisdata,
    MeteoData,
    Simulation,
    configure_logger_from_config,
)

# Configuration
force_preprocessing = False  # Set to True to force re-running preprocessing
restart_fraction = 0.5  # Run first simulation to a portion of period, then restart

example_dir = Path(__file__).parent / "Arno"
config_file = example_dir / "Arno.yaml"

meteodata_mat_path = example_dir / "meteodata/meteodata.mat"


"""Run MOBIDIC with restart demonstration for Arno basin."""

print("=" * 80)
print("MOBIDIC - Arno River Basin Example with Restart")
print("=" * 80)
print()


# =========================================================================
# Step 1: Load Configuration
# =========================================================================
print("Step 1: Loading configuration...")
config = load_config(config_file)
print(f"  Configuration loaded: {config.basin.id}")
print(f"  Time step: {config.simulation.timestep} seconds")
print(f"  Routing method: {config.parameters.routing.method}")
print()

# Configure logging
configure_logger_from_config(config)


# =========================================================================
# Step 2: GIS Preprocessing
# =========================================================================
print("Step 2: GIS Preprocessing...")

if force_preprocessing or not config.paths.gisdata.exists() or not config.paths.network.exists():
    print("  Running GIS preprocessing...")

    # Run preprocessing
    gisdata = run_preprocessing(config)

    # Save preprocessed data
    print("  Saving preprocessed GIS data...")
    save_gisdata(gisdata, config.paths.gisdata)
    save_network(gisdata.network, config.paths.network, format="parquet")

    print(f"  [OK] GIS data saved to: {config.paths.gisdata}")
    print(f"  [OK] Network saved to: {config.paths.network}")
    print(f"  [OK] Grid size: {gisdata.metadata['shape']}")
    print(f"  [OK] Number of reaches: {len(gisdata.network)}")
else:
    print("  Loading preprocessed data (already exists)...")
    gisdata = load_gisdata(config.paths.gisdata, config.paths.network)
    print(f"  [OK] Loaded GIS data: {gisdata.metadata['shape']}")
    print(f"  [OK] Loaded network: {len(gisdata.network)} reaches")

print()

# =========================================================================
# Step 3: Meteorological Data Preparation
# =========================================================================
print("Step 3: Meteorological data preparation...")

if force_preprocessing or not config.paths.meteodata.exists():
    print("  Converting MATLAB .mat format to NetCDF...")

    # Load and convert meteorological data
    meteo_data = MeteoData.from_mat(meteodata_mat_path)

    # Save to NetCDF format
    meteo_data.to_netcdf(
        config.paths.meteodata,
        add_metadata={
            "basin": config.basin.id,
            "description": "Arno basin meteorological data",
        },
    )

    print(f"  [OK] Meteo data converted and saved to: {config.paths.meteodata}")
    print(f"  [OK] Variables: {meteo_data.variables}")
    print(f"  [OK] Period: {meteo_data.start_date} to {meteo_data.end_date}")
else:
    print(f"  Meteorological data already exists: {config.paths.meteodata}")
    print("  (Set force_preprocessing=True to reconvert)")

print()

# =========================================================================
# Step 4: Load Forcing Data
# =========================================================================
print("Step 4: Loading meteorological forcing...")

forcing = MeteoData.from_netcdf(config.paths.meteodata)
print("  [OK] Forcing data loaded")
print(f"  [OK] Time range: {forcing.start_date} to {forcing.end_date}")
print(f"  [OK] Available variables: {list(forcing.variables)}")
print()

# =========================================================================
# Step 5: Determine Simulation Periods
# =========================================================================
print("Step 5: Determining simulation periods...")

# Full simulation period
start_date = forcing.start_date
end_date = forcing.end_date

# Calculate restart point (e.g., 50% of simulation period)
# Round to nearest timestep to ensure clean split
total_duration = end_date - start_date
restart_point_raw = start_date + pd.Timedelta(seconds=total_duration.total_seconds() * restart_fraction)

# Round restart point to nearest timestep boundary
dt = config.simulation.timestep  # seconds
seconds_from_start = (restart_point_raw - start_date).total_seconds()
rounded_seconds = round(seconds_from_start / dt) * dt
restart_point = start_date + pd.Timedelta(seconds=rounded_seconds)

print(f"  Full period: {start_date} to {end_date}")
print(f"  Restart fraction: {restart_fraction * 100:.0f}%")
print(f"  Restart point (raw): {restart_point_raw}")
print(f"  Restart point (rounded): {restart_point}")
print(f"  First run: {start_date} to {restart_point}")
print(f"  Second run: {restart_point} to {end_date}")
print()

# =========================================================================
# Step 6: Run First Simulation (Start to Restart Point)
# =========================================================================
print("Step 6: Running first simulation (to restart point)...")
print()

# Create simulation object for first run
sim1 = Simulation(gisdata, forcing, config)

# Run first simulation
print(f"  Running simulation from {start_date} to {restart_point}...")
start_time_1 = time.time()
results_1 = sim1.run(
    start_date=start_date,
    end_date=restart_point,
)
end_time_1 = time.time()
elapsed_time_1 = end_time_1 - start_time_1

print()
print("  [OK] First simulation completed successfully!")
print(f"  Execution time: {elapsed_time_1:.2f} seconds ({elapsed_time_1 / 60:.2f} minutes)")
print()

# Check that states were saved (handle chunked files)
states_dir = Path(config.paths.states)
states_file = states_dir / "states.nc"
chunk_file = states_dir / "states_001.nc"
state_file = chunk_file if chunk_file.exists() else states_file

if not state_file.exists():
    raise FileNotFoundError(
        f"State file not found: {state_file}. Please ensure output_states is enabled in configuration."
    )
print(f"  [OK] State file found: {state_file}")
print()

# =========================================================================
# Step 7: Run Second Simulation (Restart from Saved State)
# =========================================================================
print("Step 7: Running second simulation (restart from saved state)...")
print()

# Create new simulation object for restart
sim2 = Simulation(gisdata, forcing, config)

# Load last state from first simulation
print(f"  Loading last state from: {state_file}")
sim2.set_initial_state(state_file=state_file, time_index=-1)
print()

# Run second simulation from restart point to end
print(f"  Running simulation from {restart_point} to {end_date}...")
start_time_2 = time.time()
results_2 = sim2.run(
    start_date=restart_point,
    end_date=end_date,
)
end_time_2 = time.time()
elapsed_time_2 = end_time_2 - start_time_2

print()
print("  [OK] Second simulation completed successfully!")
print(f"  Execution time: {elapsed_time_2:.2f} seconds ({elapsed_time_2 / 60:.2f} minutes)")
print()

# =========================================================================
# Step 8: Combine Results from Both Runs
# =========================================================================
print("Step 8: Combining results from both runs...")

# Concatenate discharge time series
# Skip the first timestep of the second run to avoid duplication at restart_point
# (The restart_point is the last timestep of run 1 and the first timestep of run 2)
discharge_combined = np.concatenate(
    [results_1.time_series["discharge"], results_2.time_series["discharge"][1:]], axis=0
)
time_combined = results_1.time_series["time"] + results_2.time_series["time"][1:]

print(f"  [OK] Combined time series: {len(time_combined)} timesteps")
print(f"  First run: {len(results_1.time_series['time'])} timesteps")
print(f"  Second run: {len(results_2.time_series['time'])} timesteps (first timestep skipped in merge)")
print()

# =========================================================================
# Step 9: Run Continuous Simulation for Comparison (Optional)
# =========================================================================
print("Step 9: Running continuous simulation for comparison...")
print()

# Create simulation object for continuous run
sim_continuous = Simulation(gisdata, forcing, config)

# Run continuous simulation
print(f"  Running continuous simulation from {start_date} to {end_date}...")
start_time_cont = time.time()
results_continuous = sim_continuous.run(
    start_date=start_date,
    end_date=end_date,
)
end_time_cont = time.time()
elapsed_time_cont = end_time_cont - start_time_cont

print()
print("  [OK] Continuous simulation completed successfully!")
print(f"  Execution time: {elapsed_time_cont:.2f} seconds ({elapsed_time_cont / 60:.2f} minutes)")
print()

# =========================================================================
# Step 10: Compare Results
# =========================================================================
print("Step 10: Comparing restarted vs continuous results...")

# Compare discharge time series
discharge_continuous = results_continuous.time_series["discharge"]
time_continuous = results_continuous.time_series["time"]

# Calculate differences
discharge_diff = np.abs(discharge_combined - discharge_continuous)
max_diff = np.max(discharge_diff)
mean_diff = np.mean(discharge_diff)
relative_error = mean_diff / (np.mean(discharge_continuous) + 1e-10)

print(f"  Maximum difference: {max_diff:.6e} m³/s")
print(f"  Mean difference: {mean_diff:.6e} m³/s")
print(f"  Relative error: {relative_error * 100:.6e} %")

if max_diff < 1e-6:
    print("  Results match within numerical precision!")
else:
    print(f"  Results differ by: {max_diff:.6e}")

print()

# =========================================================================
# Step 11: Visualize Results
# =========================================================================
print("Step 11: Plotting results...")

# Select reach for visualization
reach_id = 329

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle("MOBIDIC Simulation Restart Demonstration - Arno River Basin", fontsize=14, fontweight="bold")

# Plot 1: Continuous vs Restarted simulation at specific reach
axes[0].plot(time_continuous, discharge_continuous[:, reach_id], "b-", linewidth=2, label="Continuous run")
axes[0].plot(time_combined, discharge_combined[:, reach_id], "r--", linewidth=1.5, alpha=0.7, label="Restarted run")
axes[0].axvline(restart_point, color="green", linestyle=":", linewidth=2, label="Restart point")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Discharge (m³/s)")
axes[0].set_title(f"Hydrograph comparison at reach {reach_id}")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Difference between continuous and restarted
axes[1].plot(time_continuous, discharge_diff[:, reach_id], "k-", linewidth=1)
axes[1].axvline(restart_point, color="green", linestyle=":", linewidth=2, label="Restart point")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Absolute Difference (m³/s)")
axes[1].set_title(f"Difference between continuous and restarted runs at reach {reach_id}")
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_yscale("log")

# Plot 3: Network-wide statistics
q_mean_cont = np.mean(discharge_continuous, axis=1)
q_mean_comb = np.mean(discharge_combined, axis=1)

axes[2].plot(time_continuous, q_mean_cont, "b-", linewidth=2, label="Continuous run")
axes[2].plot(time_combined, q_mean_comb, "r--", linewidth=1.5, alpha=0.7, label="Restarted run")
axes[2].axvline(restart_point, color="green", linestyle=":", linewidth=2, label="Restart point")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Mean Discharge (m³/s)")
axes[2].set_title("Network-wide average discharge")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()

print()

# =========================================================================
# Summary Statistics
# =========================================================================
print("=" * 80)
print("Simulation Restart Summary")
print("=" * 80)
print(f"Basin: {config.basin.id}")
print(f"Full period: {start_date} to {end_date}")
print(f"Restart point: {restart_point} ({restart_fraction * 100:.0f}% of period)")
print()

print("Execution times:")
print(f"  First run: {elapsed_time_1:.2f} seconds ({elapsed_time_1 / 60:.2f} minutes)")
print(f"  Second run: {elapsed_time_2:.2f} seconds ({elapsed_time_2 / 60:.2f} minutes)")
print(
    f"  Total (restarted): {elapsed_time_1 + elapsed_time_2:.2f} seconds ({(elapsed_time_1 + elapsed_time_2) / 60:.2f} minutes)"
)
print(f"  Continuous: {elapsed_time_cont:.2f} seconds ({elapsed_time_cont / 60:.2f} minutes)")
print()

print("Result comparison:")
print(f"  Maximum difference: {max_diff:.6e} m³/s")
print(f"  Mean difference: {mean_diff:.6e} m³/s")
print(f"  Relative error: {relative_error * 100:.6e} %")
if max_diff < 1e-6:
    print("  Results match within numerical precision!")
else:
    print(f"  Results differ by {max_diff:.6e}")
print()

print("=" * 80)
print("Restart example completed successfully!")
print("=" * 80)
