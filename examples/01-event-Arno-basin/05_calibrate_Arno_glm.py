"""Example: Gradient-based calibration of Arno basin using pestpp-glm (Gauss-Levenberg-Marquardt).

This script demonstrates the GLM calibration workflow:
1. Preprocess data, and convert meteorological forcing to NetCDF format
2. Set up PEST++ working directory
3. Run pestpp-glm
4. Plot results

Prerequisites:
    - Install calibration dependencies and PEST++ binaries:
        make install-calib
            or (manually)
        pip install mobidicpy[calibration] && get-pestpp :pyemu
    - Ensure pestpp-glm executable is added to PATH

Usage:
    python examples/01-event-Arno-basin/05_calibrate_Arno_glm.py

"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from mobidic.calibration import PestSetup, kge, nse
from mobidic.calibration.config import load_calibration_config
from mobidic import (
    load_config,
    save_gisdata,
    save_network,
    run_preprocessing,
    MeteoData,
    MeteoRaster,
    Simulation,
    configure_logger_from_config,
)


# Path to calibration configuration
calib_config_path = Path(__file__).parent / "Arno.calibration.yaml"

# Path to meteorological data in .mat format
meteodata_mat_path = Path(__file__).parent.parent / "datasets" / "Arno_event_Nov_2023" / "meteodata" / "meteodata.mat"

# Read calibration configuration
calib_config = load_calibration_config(calib_config_path)

# Path to main MOBIDIC config file
mobidic_config = Path(calib_config.mobidic_config)
config_file = mobidic_config if mobidic_config.is_absolute() else Path(__file__).parent / mobidic_config


# =========================================================================
# Step 1: Run preprocessing and convert meteo forcing to .nc
# =========================================================================
config = load_config(config_file)
gisdata = run_preprocessing(config)

# Save preprocessed data
print("  Saving preprocessed GIS data...")
save_gisdata(gisdata, config.paths.gisdata)
save_network(gisdata.network, config.paths.network, format="parquet")

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

# =========================================================================
# Step 2: Run PEST++ GLM calibration
# =========================================================================

# Initialize PestSetup and generate all PEST++ files
pest = PestSetup(calib_config_path)
working_dir = pest.setup()

print(f"PEST++ working directory created: {working_dir}")
print("Files generated:")
for f in sorted(working_dir.iterdir()):
    print(f"  {f.name}")

# Run pestpp-glm
results = pest.run(num_workers=4)

# Get optimal parameters
optimal = results.get_optimal_parameters()
print("\nOptimal parameters:")
for name, value in optimal.items():
    print(f"  {name}: {value:.6g}")

# Get objective function history
phi_history = results.get_objective_function_history()
if phi_history is not None:
    print("\nObjective function history:")
    print(phi_history.to_string(index=False))

# Get parameter sensitivities
sens = results.get_parameter_sensitivities()
if sens is not None:
    print("\nParameter sensitivities:")
    print(sens.to_string(index=False))

# =========================================================================
# Step 3: Run validation simulation with optimal parameters and plot results
# =========================================================================
print("\nRunning validation simulation with optimal parameters...")

config = load_config(config_file)

# Map PEST parameter names to YAML dot-paths
param_name_to_key = {p.name: p.parameter_key for p in pest.calib_config.parameters}
param_updates = {param_name_to_key[name]: val for name, val in optimal.items() if name in param_name_to_key}

# Configure logger
configure_logger_from_config(config)

# Update gisdata and load forcing
gisdata = run_preprocessing(config)
forcing = MeteoRaster.from_netcdf(pest.working_dir / "forcing_raster.nc")

# Load raster forcing generated during setup and run simulation
sim_start = forcing.start_date
sim_end = forcing.end_date
results_cal = Simulation(gisdata, forcing, config).run(sim_start, sim_end)

# --- Plot 1: Observed vs Simulated discharge ---
obs_group = pest.calib_config.observations[0]
df_obs = pd.read_csv(pest.base_path / obs_group.obs_file)
df_obs["time"] = pd.to_datetime(df_obs[obs_group.time_column])
df_obs = df_obs.set_index("time")[obs_group.value_column]

reach_ids = gisdata.network["mobidic_id"].values
reach_to_idx = {int(rid): i for i, rid in enumerate(reach_ids)}
col_idx = reach_to_idx[obs_group.reach_id]
sim_times = pd.DatetimeIndex(results_cal.time_series["time"])
df_sim = pd.Series(results_cal.time_series["discharge"][:, col_idx], index=sim_times)

common_index = df_sim.index.intersection(df_obs.index)
sim_aligned = df_sim.loc[common_index]
obs_aligned = df_obs.loc[common_index]

nse_val = nse(sim_aligned.values, obs_aligned.values)
kge_val = kge(sim_aligned.values, obs_aligned.values)

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(common_index, obs_aligned, "b-", linewidth=1.5, alpha=0.8, label="Observed")
ax1.plot(common_index, sim_aligned, "r--", linewidth=1.0, alpha=0.9, label="Simulated (optimal)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Discharge (m$^3$/s)")
ax1.set_title(f"Reach {obs_group.reach_id} - {obs_group.name}  |  NSE={nse_val:.3f}  KGE={kge_val:.3f}")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax1.tick_params(axis="x", rotation=45)
ax1.grid(True, alpha=0.3)
ax1.legend()
fig1.tight_layout()

# --- Plot 2: Objective function history ---
if phi_history is not None:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(phi_history["iteration"], phi_history["phi"], "o-", color="steelblue", linewidth=1.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Objective function ($\\phi$)")
    ax2.set_title("PEST++ GLM - Objective function history")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

plt.show()
