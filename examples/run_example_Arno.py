"""
MOBIDIC Example - Arno River Basin Simulation

This script demonstrates the complete MOBIDIC workflow:
1. Load configuration
2. Run GIS preprocessing (or load preprocessed data)
3. Convert meteorological data to NetCDF format
4. Run hydrological simulation
5. Save results (states and discharge time series)
6. Visualize results

The example uses data from the Arno River basin in Tuscany, Italy.

Usage:
    python examples/run_example_Arno.py

    Options (modify script directly):
    - force_preprocessing: Set to True to rerun preprocessing even if data exists
"""

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

from mobidic import (
    load_config,
    run_preprocessing,
    save_gisdata,
    save_network,
    load_gisdata,
    MeteoData,
    Simulation,
    configure_logger,
)

# Configuration
force_preprocessing = False  # Set to True to force re-running preprocessing

example_dir = Path(__file__).parent / "Arno"
config_file = example_dir / "Arno.yaml"

meteodata_mat_path = example_dir / "meteodata/meteodata.mat"

debug_level = "DEBUG"  # Logging level


"""Run complete MOBIDIC workflow for Arno basin."""

print("=" * 80)
print("MOBIDIC - Arno River Basin Example")
print("=" * 80)
print()

# Configure logging
configure_logger(level=debug_level)


# =========================================================================
# Step 1: Load Configuration
# =========================================================================
print("Step 1: Loading configuration...")
config = load_config(config_file)
print(f"  Configuration loaded: {config.basin.id}")
print(f"  Time step: {config.simulation.timestep} seconds")
print(f"  Routing method: {config.parameters.routing.method}")
print()


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
# Step 5: Initialize and Run Simulation
# =========================================================================
print("Step 5: Running hydrological simulation...")

# Create simulation object
sim = Simulation(gisdata, forcing, config)

# Determine simulation period
start_date = forcing.start_date
# end_date = forcing.start_date + pd.Timedelta(hours=2) # For testing short runs
end_date = forcing.end_date

print(f"  Simulation period: {start_date} to {end_date}")
print()

# Run simulation
print("  Running simulation...")
start_time = time.time()
results = sim.run(
    start_date=start_date,
    end_date=end_date,
)
end_time = time.time()
elapsed_time = end_time - start_time

print()
print("  [OK] Simulation completed successfully!")
print(f"  Execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
print()

# =========================================================================
# Step 6: Results Summary
# =========================================================================
print("Step 6: Results summary...")
print()
print("  NOTE: Output files are automatically saved based on configuration:")
print(f"    - Discharge report: {config.output_report.discharge}")
print(f"    - Lateral inflow report: {config.output_report.lateral_inflow}")
print(f"    - Final state: {config.output_states_settings.output_states}")
print()

# Output files have been automatically saved to:
start_date_str = start_date.strftime("%Y%m%d")
end_date_str1 = end_date.strftime("%Y%m%d")

if config.output_report.discharge:
    discharge_file = config.paths.output / f"discharge_{start_date_str}_{end_date_str1}.parquet"
    print(f"  [OK] Discharge report saved to: {discharge_file}")

if config.output_report.lateral_inflow:
    lateral_inflow_file = config.paths.output / f"lateral_inflow_{start_date_str}_{end_date_str1}.parquet"
    print(f"  [OK] Lateral inflow report saved to: {lateral_inflow_file}")

if config.output_states_settings.output_states in ["final", "all"]:
    final_time = results.time_series["time"][-1]
    state_file = config.paths.states / f"state_{final_time.strftime('%Y%m%d_%H%M%S')}.nc"
    print(f"  [OK] Final state saved to: {state_file}")

print()

# =========================================================================
# Step 7: Visualize Results
# =========================================================================
print("Step 7: Plotting results...")

# Get discharge time series
discharge_ts = results.time_series["discharge"]
time_ts = results.time_series["time"]

# Use one reach for visualization
reach_id = 329

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("MOBIDIC Simulation Results - Arno River Basin", fontsize=14, fontweight="bold")

# Plot 1: Discharge hydrograph at outlet
axes[0].plot(time_ts, discharge_ts[:, reach_id], "b-", linewidth=2, label=f"Outlet (reach {reach_id})")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Discharge [m³/s]")
axes[0].set_title(f"Discharge Hydrograph at reach {reach_id}")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Network-wide discharge statistics
q_mean = np.mean(discharge_ts, axis=1)
q_max = np.max(discharge_ts, axis=1)
q_min = np.min(discharge_ts, axis=1)

axes[1].plot(time_ts, q_mean, "b-", linewidth=2, label="Mean")
axes[1].fill_between(time_ts, q_min, q_max, alpha=0.3, label="Range (min-max)")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Discharge [m³/s]")
axes[1].set_title("Network-Wide Discharge Statistics")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

print()

# =========================================================================
# Summary Statistics
# =========================================================================
print("=" * 80)
print("Simulation Summary")
print("=" * 80)
print(f"Basin: {config.basin.id}")
print(f"Period: {start_date} to {end_date}")
print(f"Time step: {config.simulation.timestep} seconds")
print(f"Number of time steps: {len(time_ts)}")
print(f"Execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
print(f"Grid size: {gisdata.metadata['shape'][0]} x {gisdata.metadata['shape'][1]}")
print(f"Number of reaches: {len(gisdata.network)}")
print()

print("Network-wide statistics:")
print(f"  Mean discharge: {np.mean(discharge_ts):.2f} m³/s")
print(f"  Max discharge:  {np.max(discharge_ts):.2f} m³/s")
print()

print("=" * 80)
print("Example completed successfully!")
print("=" * 80)
