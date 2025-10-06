"""Demo script showing how to use the MOBIDIC configuration parser."""

import sys
from pathlib import Path

# Add parent directory to path for development (no install needed)
sys.path.insert(0, str(Path(__file__).parent.parent))

from mobidic.config import load_config

# Path to sample configuration
config_path = Path(__file__).parent / "sample_config.yaml"

# Load and validate configuration
print("Loading configuration from:", config_path)

try:
    config = load_config(config_path)
    print("Configuration loaded and validated successfully!")
except Exception as e:
    print("Error loading configuration:", e)
    sys.exit(1)

# Access configuration values in the same order as YAML file

# PROJECT DATA
print("\n" + "=" * 60)
print("Basin Information:")
print("=" * 60)
print(f"Basin ID: {config.basin.id}")
print(f"Parameter Set: {config.basin.paramset_id}")
print(f"Baricenter: ({config.basin.baricenter.lon}°E, {config.basin.baricenter.lat}°N)")

# FILE PATHS
print("\n" + "=" * 60)
print("File Paths:")
print("=" * 60)
print(f"Meteodata: {config.paths.meteodata}")
print(f"GIS Data: {config.paths.gisdata}")
print(f"States Directory: {config.paths.states}")
print(f"Output Directory: {config.paths.output}")

# VECTOR FILES
print("\n" + "=" * 60)
print("Vector Files:")
print("=" * 60)
print(f"River Network: {config.vector_files.river_network.shp}")
print(f"ID Field: {config.vector_files.river_network.id_field}")

# RASTER FILES
print("\n" + "=" * 60)
print("Raster Files:")
print("=" * 60)
print(f"DTM: {config.raster_files.dtm}")
print(f"Flow Direction: {config.raster_files.flow_dir}")
print(f"Flow Accumulation: {config.raster_files.flow_acc}")
print(f"Wc0: {config.raster_files.Wc0}")
print(f"Wg0: {config.raster_files.Wg0}")
print(f"ks: {config.raster_files.ks}")
if config.raster_files.kf:
    print(f"kf: {config.raster_files.kf}")
if config.raster_files.CH:
    print(f"CH: {config.raster_files.CH}")
if config.raster_files.Alb:
    print(f"Alb: {config.raster_files.Alb}")
if config.raster_files.Ma:
    print(f"Ma: {config.raster_files.Ma}")
if config.raster_files.Mf:
    print(f"Mf: {config.raster_files.Mf}")
if config.raster_files.gamma:
    print(f"gamma: {config.raster_files.gamma}")
if config.raster_files.kappa:
    print(f"kappa: {config.raster_files.kappa}")
if config.raster_files.beta:
    print(f"beta: {config.raster_files.beta}")
if config.raster_files.alpha:
    print(f"alpha: {config.raster_files.alpha}")

print("\n" + "=" * 60)
print("Raster Settings:")
print("=" * 60)
print(f"Flow Direction Type: {config.raster_settings.flow_dir_type}")

# GLOBAL LAND PARAMETERS
print("\n" + "=" * 60)
print("Soil Parameters:")
print("=" * 60)
print(f"Wc0: {config.parameters.soil.Wc0} mm")
print(f"Wg0: {config.parameters.soil.Wg0} mm")
print(f"ks: {config.parameters.soil.ks} mm/h")
if hasattr(config.parameters.soil, "ks_min") and config.parameters.soil.ks_min:
    print(f"ks_min: {config.parameters.soil.ks_min} mm/h")
if hasattr(config.parameters.soil, "ks_max") and config.parameters.soil.ks_max:
    print(f"ks_max: {config.parameters.soil.ks_max} mm/h")
print(f"kf: {config.parameters.soil.kf} m/s")
print(f"gamma (percolation): {config.parameters.soil.gamma} 1/s")
print(f"kappa (adsorption): {config.parameters.soil.kappa} 1/s")
print(f"beta (hypodermic): {config.parameters.soil.beta} 1/s")
print(f"alpha (hillslope): {config.parameters.soil.alpha} 1/s")

print("\n" + "=" * 60)
print("Energy Parameters:")
print("=" * 60)
print(f"Tconst: {config.parameters.energy.Tconst} K")
print(f"kaps: {config.parameters.energy.kaps} W/m/K")
print(f"nis: {config.parameters.energy.nis} m²/s")
print(f"CH: {config.parameters.energy.CH}")
print(f"Alb: {config.parameters.energy.Alb}")

print("\n" + "=" * 60)
print("Routing Parameters:")
print("=" * 60)
print(f"Method: {config.parameters.routing.method}")
print(f"wcel (wave celerity): {config.parameters.routing.wcel} m/s")
print(f"Br0: {config.parameters.routing.Br0} m")
print(f"NBr: {config.parameters.routing.NBr}")
print(f"n_Man (Manning coeff.): {config.parameters.routing.n_Man} s/m^(1/3)")

print("\n" + "=" * 60)
print("Groundwater Parameters:")
print("=" * 60)
print(f"Model: {config.parameters.groundwater.model}")
print(f"Global Loss: {config.parameters.groundwater.global_loss} m³/s")

if hasattr(config.parameters, "multipliers"):
    print("\n" + "=" * 60)
    print("Multipliers:")
    print("=" * 60)
    if hasattr(config.parameters.multipliers, "ks_factor"):
        print(f"ks_factor: {config.parameters.multipliers.ks_factor}")
    if hasattr(config.parameters.multipliers, "Wc_factor"):
        print(f"Wc_factor: {config.parameters.multipliers.Wc_factor}")
    if hasattr(config.parameters.multipliers, "Wg_factor"):
        print(f"Wg_factor: {config.parameters.multipliers.Wg_factor}")
    if hasattr(config.parameters.multipliers, "Wg_Wc_tr"):
        print(f"Wg_Wc_tr: {config.parameters.multipliers.Wg_Wc_tr}")
    if hasattr(config.parameters.multipliers, "CH_factor"):
        print(f"CH_factor: {config.parameters.multipliers.CH_factor}")
    if hasattr(config.parameters.multipliers, "cel_factor"):
        print(f"cel_factor: {config.parameters.multipliers.cel_factor}")
    if hasattr(config.parameters.multipliers, "chan_factor"):
        print(f"chan_factor: {config.parameters.multipliers.chan_factor}")

# INITIAL CONDITIONS
if hasattr(config, "initial_conditions"):
    print("\n" + "=" * 60)
    print("Initial Conditions:")
    print("=" * 60)
    if hasattr(config.initial_conditions, "Ws"):
        print(f"Ws (hillslope depth): {config.initial_conditions.Ws} m")
    if hasattr(config.initial_conditions, "Wcsat"):
        print(f"Wcsat (capillary saturation): {config.initial_conditions.Wcsat}")
    if hasattr(config.initial_conditions, "Wgsat"):
        print(f"Wgsat (gravitational saturation): {config.initial_conditions.Wgsat}")

# SIMULATION CONTROLS
print("\n" + "=" * 60)
print("Simulation Settings:")
print("=" * 60)
print(f"Realtime Mode: {config.simulation.realtime}")
print(f"Timestep: {config.simulation.timestep} seconds")
print(f"Resample Factor: {config.simulation.resample}")
print(f"Soil Scheme: {config.simulation.soil_scheme}")
print(f"Energy Balance: {config.simulation.energy_balance}")

# OUTPUT CONTROLS
print("\n" + "=" * 60)
print("Output States:")
print("=" * 60)
print(f"Discharge: {config.output_states.discharge}")
print(f"Reservoir States: {config.output_states.reservoir_states}")
print(f"Soil Capillary: {config.output_states.soil_capillary}")
print(f"Soil Gravitational: {config.output_states.soil_gravitational}")
print(f"Surface Temperature: {config.output_states.surface_temperature}")
print(f"Ground Temperature: {config.output_states.ground_temperature}")
print(f"Aquifer Head: {config.output_states.aquifer_head}")
print(f"ET/Precip: {config.output_states.et_prec}")

print("\n" + "=" * 60)
print("Output States Settings:")
print("=" * 60)
print(f"Output Format: {config.output_states_settings.output_format}")
if hasattr(config.output_states_settings, "output_interval") and config.output_states_settings.output_interval:
    print(f"Output Interval: {config.output_states_settings.output_interval} seconds")

if hasattr(config, "output_report"):
    print("\n" + "=" * 60)
    print("Output Report:")
    print("=" * 60)
    if hasattr(config.output_report, "discharge"):
        print(f"Discharge: {config.output_report.discharge}")
    if hasattr(config.output_report, "lateral_inflow"):
        print(f"Lateral Inflow: {config.output_report.lateral_inflow}")

print("\n" + "=" * 60)
print("Output Report Settings:")
print("=" * 60)
print(f"Output Format: {config.output_report_settings.output_format}")
if hasattr(config.output_report_settings, "report_interval") and config.output_report_settings.report_interval:
    print(f"Report Interval: {config.output_report_settings.report_interval} seconds")
print(f"Reach Selection: {config.output_report_settings.reach_selection}")
if config.output_report_settings.reach_selection == "file":
    print(f"Selection File: {config.output_report_settings.sel_file}")
if config.output_report_settings.reach_selection == "list":
    print(f"Selected Reaches: {config.output_report_settings.sel_list}")

# ADVANCED SETTINGS
if hasattr(config, "advanced"):
    print("\n" + "=" * 60)
    print("Advanced Settings:")
    print("=" * 60)
    if hasattr(config.advanced, "log_level"):
        print(f"Log Level: {config.advanced.log_level}")
    if hasattr(config.advanced, "log_file"):
        print(f"Log File: {config.advanced.log_file}")


# Demonstrate validation by creating config with invalid data
print("\n" + "=" * 60)
print("Demonstrating Validation:")
print("=" * 60)

# Create a test YAML with invalid timestep
test_config_path = Path(__file__).parent / "test_invalid.yaml"
test_config_path.write_text("""
basin:
  id: Test
  paramset_id: Test
  baricenter: {lon: 10.0, lat: 45.0}
paths:
  meteodata: test.nc
  gisdata: test.nc
  states: states/
  output: output/
vector_files:
  river_network: {shp: test.shp, id_field: ID}
raster_files:
  dtm: dtm.tif
  flow_dir: flowdir.tif
  flow_acc: flowacc.tif
  Wc0: wc0.tif
  Wg0: wg0.tif
  ks: ks.tif
raster_settings:
  flow_dir_type: Grass
parameters:
  soil: {Wc0: 100, Wg0: 50, ks: 1.0, kf: 1e-7, gamma: 1e-7, kappa: 1e-7, beta: 1e-6, alpha: 1e-5}
  energy: {Tconst: 290, kaps: 2.5, nis: 8e-7, CH: 1e-3, Alb: 0.2}
  routing: {method: Linear, wcel: 5.0, Br0: 1.0, NBr: 1.5, n_Man: 0.03}
  groundwater: {model: None}
simulation:
  realtime: 0
  timestep: -100
  resample: 1
  soil_scheme: Bucket
  energy_balance: None
output_states:
  discharge: true
  reservoir_states: true
  soil_capillary: true
  soil_gravitational: true
  surface_temperature: false
  ground_temperature: false
  aquifer_head: false
  et_prec: false
""")

try:
    invalid_config = load_config(test_config_path)
    print("ERROR: Negative timestep was accepted (should have been rejected!)")
except ValueError as e:
    print("[OK] Validation worked! Negative timestep rejected during config loading")
    print(f"     Error type: {type(e).__name__}")
finally:
    # Clean up test file
    if test_config_path.exists():
        test_config_path.unlink()

print("\n" + "=" * 60)
print("Configuration parser demo completed successfully!")
print("=" * 60)
