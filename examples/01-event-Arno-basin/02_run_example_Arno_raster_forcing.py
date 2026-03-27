"""
MOBIDIC Example - Arno River Basin with raster forcing

This script demonstrates the comparison between:
1. Station-based meteorological forcing with spatial interpolation performed during simulation
2. Raster forcing data (pre-interpolated meteorological data)

The workflow is as follows:
1. Load configuration from Arno.yaml
2. GIS preprocessing (or load preprocessed data)
3. Convert meteorological data to NetCDF format
4. Run simulation with station data (outputs interpolated meteo data)
5. Run second simulation using the interpolated raster data as forcing
6. Compare discharge results from both simulations

The forcing raster input data has the same format as the exported interpolated meteorological data.

Usage:
    python examples/01-event-Arno-basin/02_run_example_Arno_raster_forcing.py
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
    MeteoRaster,
    Simulation,
    configure_logger_from_config,
)

# Configuration
force_preprocessing = False  # Set to True to force re-running preprocessing

config_file = Path(__file__).parent / "Arno.yaml"
meteodata_mat_path = Path(__file__).parent.parent / "datasets" / "Arno_event_Nov_2023" / "meteodata" / "meteodata.mat"

print("=" * 80)
print("MOBIDIC - Arno Basin: station vs raster forcing comparison")
print("=" * 80)
print()

# =========================================================================
# Step 1: Load Configuration
# =========================================================================
print("Step 1: Loading configuration...")
config = load_config(config_file)
print(f"  Configuration loaded: {config.basin.id}")
print(f"  Time step: {config.simulation.timestep} seconds")
print()

# Configure logging
configure_logger_from_config(config)

# =========================================================================
# Step 2: GIS Preprocessing
# =========================================================================
print("Step 2: GIS Preprocessing...")

if force_preprocessing or not config.paths.gisdata.exists() or not config.paths.network.exists():
    print("  Running GIS preprocessing...")
    gisdata = run_preprocessing(config)
    print("  Saving preprocessed GIS data...")
    save_gisdata(gisdata, config.paths.gisdata)
    save_network(gisdata.network, config.paths.network, format="parquet")
    print(f"  [OK] GIS data saved to: {config.paths.gisdata}")
    print(f"  [OK] Network saved to: {config.paths.network}")
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

    # Create meteodata folder if it doesn't exist
    config.paths.meteodata.parent.mkdir(parents=True, exist_ok=True)

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
# Step 4: Simulation 1 - Station-based forcing with interpolation output
# =========================================================================
print("Step 4: First simulation with station-based forcing...")
print("  (This will generate interpolated meteorological data)")
print()

# Modify config to enable meteo forcing data output
config.output_forcing_data.meteo_data = True
print("  Enabled meteo forcing data output")

# Load station-based forcing
print("  Loading station-based meteorological forcing...")
forcing_stations = MeteoData.from_netcdf(config.paths.meteodata)
print("  [OK] Forcing data loaded from stations")
print(f"  [OK] Time range: {forcing_stations.start_date} to {forcing_stations.end_date}")
print(f"  [OK] Available variables: {list(forcing_stations.variables)}")
print()

# Create simulation object for first run
sim1 = Simulation(gisdata, forcing_stations, config)

# Determine simulation period
start_date = forcing_stations.start_date
end_date = forcing_stations.end_date

print(f"  Simulation period: {start_date} to {end_date}")
print()

# Run first simulation
print("  Running simulation 1 (station-based forcing)...")
start_time = time.time()
results1 = sim1.run(
    start_date=start_date,
    end_date=end_date,
)
end_time = time.time()
elapsed_time1 = end_time - start_time

print()
print("  [OK] Simulation 1 completed successfully!")
print(f"  Execution time: {elapsed_time1:.2f} seconds ({elapsed_time1 / 60:.2f} minutes)")
print()

# Check that forcing data was created
interpolated_meteo_path = config.paths.output / "meteo_forcing.nc"
if interpolated_meteo_path.exists():
    print(f"  [OK] Meteo forcing data saved to: {interpolated_meteo_path}")
else:
    print(f"  [WARNING] Meteo forcing data not found at: {interpolated_meteo_path}")
    print("  Cannot proceed with second simulation.")
    exit(1)

print()

# =========================================================================
# Step 5: Simulation 2 - Raster-based forcing (pre-interpolated)
# =========================================================================
print("Step 5: Second simulation with raster-based forcing...")
print("  (Using pre-interpolated meteorological data)")
print()

# Disable meteo forcing data output for second run (not needed)
config.output_forcing_data.meteo_data = False

# Load raster-based forcing (the interpolated data from simulation 1)
print("  Loading raster-based meteorological forcing...")
forcing_raster = MeteoRaster.from_netcdf(interpolated_meteo_path)
print("  [OK] Forcing data loaded from interpolated raster")
print(f"  [OK] Time range: {forcing_raster.start_date} to {forcing_raster.end_date}")
print(f"  [OK] Available variables: {list(forcing_raster.variables)}")
print()

# Create simulation object for second run
sim2 = Simulation(gisdata, forcing_raster, config)

print(f"  Simulation period: {start_date} to {end_date}")
print()

# Run second simulation
print("  Running simulation 2 (raster-based forcing)...")
start_time = time.time()
results2 = sim2.run(
    start_date=start_date,
    end_date=end_date,
)
end_time = time.time()
elapsed_time2 = end_time - start_time

print()
print("  [OK] Simulation 2 completed successfully!")
print(f"  Execution time: {elapsed_time2:.2f} seconds ({elapsed_time2 / 60:.2f} minutes)")
print()

# =========================================================================
# Step 6: Compare Results
# =========================================================================
print("Step 6: Comparing results...")
print()

# Get discharge time series from both simulations
discharge_ts1 = results1.time_series["discharge"]
discharge_ts2 = results2.time_series["discharge"]
time_ts = results1.time_series["time"]

# Calculate differences
discharge_diff = discharge_ts2 - discharge_ts1
max_abs_diff = np.max(np.abs(discharge_diff))
mean_abs_diff = np.mean(np.abs(discharge_diff))
relative_error = np.max(np.abs(discharge_diff / (discharge_ts1 + 1e-6))) * 100

print("  Discharge comparison statistics:")
print(f"    Max absolute difference: {max_abs_diff:.6e} m³/s")
print(f"    Mean absolute difference: {mean_abs_diff:.6e} m³/s")
print(f"    Max relative error: {relative_error:.6e} %")
print()

# Performance comparison
speedup = elapsed_time1 / elapsed_time2
print("  Performance comparison:")
print(f"    Simulation 1 (station-based): {elapsed_time1:.2f} seconds")
print(f"    Simulation 2 (raster-based): {elapsed_time2:.2f} seconds")
print(f"    Speedup factor: {speedup:.2f}x")
print()

# =========================================================================
# Step 7: Visualize Results
# =========================================================================
print("Step 7: Plotting results...")

# Select reach for detailed comparison
reach_id = 329

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(
    "MOBIDIC Arno Basin: Station vs Raster Forcing Comparison",
    fontsize=14,
    fontweight="bold",
)

# Plot 1: Discharge hydrograph at specific reach - both simulations
axes[0, 0].plot(time_ts, discharge_ts1[:, reach_id], "b-", linewidth=2, label="Station-based", alpha=0.7)
axes[0, 0].plot(time_ts, discharge_ts2[:, reach_id], "r--", linewidth=2, label="Raster-based", alpha=0.7)
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Discharge (m³/s)")
axes[0, 0].set_title(f"Hydrograph at reach {reach_id}")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Discharge difference at specific reach
axes[0, 1].plot(time_ts, discharge_diff[:, reach_id], "g-", linewidth=1.5)
axes[0, 1].axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Discharge difference (m³/s)")
axes[0, 1].set_title(f"Difference at reach {reach_id} (Raster - Station)")
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Network-wide mean discharge
q_mean1 = np.mean(discharge_ts1, axis=1)
q_mean2 = np.mean(discharge_ts2, axis=1)
axes[1, 0].plot(time_ts, q_mean1, "b-", linewidth=2, label="Station-based", alpha=0.7)
axes[1, 0].plot(time_ts, q_mean2, "r--", linewidth=2, label="Raster-based", alpha=0.7)
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Mean Discharge (m³/s)")
axes[1, 0].set_title("Network-wide mean discharge")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Plot 4: Network-wide mean discharge difference
axes[1, 1].plot(time_ts, q_mean2 - q_mean1, "g-", linewidth=1.5)
axes[1, 1].axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
axes[1, 1].set_xlabel("Time")
axes[1, 1].set_ylabel("Mean discharge difference (m³/s)")
axes[1, 1].set_title("Network-wide mean difference (Raster - Station)")
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Max discharge across network
q_max1 = np.max(discharge_ts1, axis=1)
q_max2 = np.max(discharge_ts2, axis=1)
axes[2, 0].plot(time_ts, q_max1, "b-", linewidth=2, label="Station-based", alpha=0.7)
axes[2, 0].plot(time_ts, q_max2, "r--", linewidth=2, label="Raster-based", alpha=0.7)
axes[2, 0].set_xlabel("Time")
axes[2, 0].set_ylabel("Max Discharge (m³/s)")
axes[2, 0].set_title("Network-wide maximum discharge")
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend()

# Plot 6: Scatter plot - discharge correlation
sample_reach_ids = [0, 100, 200, 292, 313, 329, 514]  # Sample of reaches
colors = plt.cm.viridis(np.linspace(0, 1, len(sample_reach_ids)))

for i, rid in enumerate(sample_reach_ids):
    if rid < discharge_ts1.shape[1]:
        axes[2, 1].scatter(
            discharge_ts1[:, rid],
            discharge_ts2[:, rid],
            alpha=0.3,
            s=10,
            color=colors[i],
            label=f"Reach {rid}",
        )

# Add 1:1 line
min_val = min(discharge_ts1.min(), discharge_ts2.min())
max_val = max(discharge_ts1.max(), discharge_ts2.max())
axes[2, 1].plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.5)
axes[2, 1].set_xlabel("Station-based discharge (m³/s)")
axes[2, 1].set_ylabel("Raster-based discharge (m³/s)")
axes[2, 1].set_title("Discharge correlation (1:1 line)")
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.show()

print()

# =========================================================================
# Summary Statistics
# =========================================================================
print("=" * 80)
print("Comparison Summary")
print("=" * 80)
print(f"Basin: {config.basin.id}")
print(f"Period: {start_date} to {end_date}")
print(f"Time step: {config.simulation.timestep} seconds")
print(f"Number of time steps: {len(time_ts)}")
print(f"Number of reaches: {len(gisdata.network)}")
print()

print("Simulation execution times:")
print(f"  Station-based forcing: {elapsed_time1:.2f} seconds ({elapsed_time1 / 60:.2f} minutes)")
print(f"  Raster-based forcing:  {elapsed_time2:.2f} seconds ({elapsed_time2 / 60:.2f} minutes)")
print(f"  Speedup factor:        {speedup:.2f}x")
print()

print("Discharge comparison (across all reaches and timesteps):")
print(f"  Max absolute difference:  {max_abs_diff:.6e} m³/s")
print(f"  Mean absolute difference: {mean_abs_diff:.6e} m³/s")
print(f"  Max relative error:       {relative_error:.6e} %")
print()

print("Interpretation:")
if max_abs_diff < 1e-6:
    print("  OK Results are numerically identical (differences < 1e-6 m³/s)")
    print("  OK Raster forcing produces exact same results as station interpolation")
elif max_abs_diff < 1e-3:
    print("  OK Results are very close (differences < 0.001 m³/s)")
    print("  OK Minor numerical differences likely due to interpolation precision")
else:
    print("  Results show noticeable differences")

print()
print("=" * 80)
print("Example completed successfully!")
print("=" * 80)
