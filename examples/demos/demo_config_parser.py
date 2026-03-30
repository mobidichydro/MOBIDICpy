"""Demo script showing how to use the MOBIDIC configuration parser."""

import sys
from pathlib import Path

from loguru import logger

# Add parent directory to path for development (no install needed)
sys.path.insert(0, str(Path(__file__).parent.parent))

from mobidic.config import load_config
from mobidic.utils import configure_logger

# Configure logger using centralized configuration
configure_logger(level="INFO")

# Path to sample configuration
config_path = Path(__file__).parent.parent / "01-event-Arno-basin" / "Arno.yaml"

# Load and validate configuration
try:
    config = load_config(config_path)
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    sys.exit(1)

# Access configuration values in the same order as YAML file

# PROJECT DATA
logger.info("=" * 60)
logger.info("Basin Information:")
logger.info("=" * 60)
logger.info(f"Basin ID: {config.basin.id}")
logger.info(f"Parameter Set: {config.basin.paramset_id}")
logger.info(f"Baricenter: ({config.basin.baricenter.lon}°E, {config.basin.baricenter.lat}°N)")

# FILE PATHS
logger.info("")
logger.info("=" * 60)
logger.info("File Paths:")
logger.info("=" * 60)
logger.info(f"Meteodata: {config.paths.meteodata}")
logger.info(f"GIS Data: {config.paths.gisdata}")
logger.info(f"Network: {config.paths.network}")
logger.info(f"States Directory: {config.paths.states}")
logger.info(f"Output Directory: {config.paths.output}")

# VECTOR FILES
logger.info("")
logger.info("=" * 60)
logger.info("Vector Files:")
logger.info("=" * 60)
logger.info(f"River Network: {config.vector_files.river_network.shp}")

# RASTER FILES
logger.info("")
logger.info("=" * 60)
logger.info("Raster Files:")
logger.info("=" * 60)
logger.info(f"DTM: {config.raster_files.dtm}")
logger.info(f"Flow Direction: {config.raster_files.flow_dir}")
logger.info(f"Flow Accumulation: {config.raster_files.flow_acc}")
logger.info(f"Wc0: {config.raster_files.Wc0}")
logger.info(f"Wg0: {config.raster_files.Wg0}")
logger.info(f"ks: {config.raster_files.ks}")
if config.raster_files.kf:
    logger.info(f"kf: {config.raster_files.kf}")
if config.raster_files.CH:
    logger.info(f"CH: {config.raster_files.CH}")
if config.raster_files.Alb:
    logger.info(f"Alb: {config.raster_files.Alb}")
if config.raster_files.Ma:
    logger.info(f"Ma: {config.raster_files.Ma}")
if config.raster_files.Mf:
    logger.info(f"Mf: {config.raster_files.Mf}")
if config.raster_files.gamma:
    logger.info(f"gamma: {config.raster_files.gamma}")
if config.raster_files.kappa:
    logger.info(f"kappa: {config.raster_files.kappa}")
if config.raster_files.beta:
    logger.info(f"beta: {config.raster_files.beta}")
if config.raster_files.alpha:
    logger.info(f"alpha: {config.raster_files.alpha}")

logger.info("")
logger.info("=" * 60)
logger.info("Raster Settings:")
logger.info("=" * 60)
logger.info(f"Flow Direction Type: {config.raster_settings.flow_dir_type}")

# GLOBAL LAND PARAMETERS
logger.info("")
logger.info("=" * 60)
logger.info("Soil Parameters:")
logger.info("=" * 60)
logger.info(f"Wc0: {config.parameters.soil.Wc0} mm")
logger.info(f"Wg0: {config.parameters.soil.Wg0} mm")
logger.info(f"ks: {config.parameters.soil.ks} mm/h")
if hasattr(config.parameters.soil, "ks_min") and config.parameters.soil.ks_min:
    logger.info(f"ks_min: {config.parameters.soil.ks_min} mm/h")
if hasattr(config.parameters.soil, "ks_max") and config.parameters.soil.ks_max:
    logger.info(f"ks_max: {config.parameters.soil.ks_max} mm/h")
logger.info(f"kf: {config.parameters.soil.kf} m/s")
logger.info(f"gamma (percolation): {config.parameters.soil.gamma} 1/s")
logger.info(f"kappa (adsorption): {config.parameters.soil.kappa} 1/s")
logger.info(f"beta (hypodermic): {config.parameters.soil.beta} 1/s")
logger.info(f"alpha (hillslope): {config.parameters.soil.alpha} 1/s")

logger.info("")
logger.info("=" * 60)
logger.info("Energy Parameters:")
logger.info("=" * 60)
logger.info(f"Tconst: {config.parameters.energy.Tconst} K")
logger.info(f"kaps: {config.parameters.energy.kaps} W/m/K")
logger.info(f"nis: {config.parameters.energy.nis} m²/s")
logger.info(f"CH: {config.parameters.energy.CH}")
logger.info(f"Alb: {config.parameters.energy.Alb}")

logger.info("")
logger.info("=" * 60)
logger.info("Routing Parameters:")
logger.info("=" * 60)
logger.info(f"Method: {config.parameters.routing.method}")
logger.info(f"wcel (wave celerity): {config.parameters.routing.wcel} m/s")
logger.info(f"Br0: {config.parameters.routing.Br0} m")
logger.info(f"NBr: {config.parameters.routing.NBr}")
logger.info(f"n_Man (Manning coeff.): {config.parameters.routing.n_Man} s/m^(1/3)")

logger.info("")
logger.info("=" * 60)
logger.info("Groundwater Parameters:")
logger.info("=" * 60)
logger.info(f"Model: {config.parameters.groundwater.model}")
logger.info(f"Global Loss: {config.parameters.groundwater.global_loss} m³/s")

if hasattr(config.parameters, "multipliers"):
    logger.info("")
    logger.info("=" * 60)
    logger.info("Multipliers:")
    logger.info("=" * 60)
    if hasattr(config.parameters.multipliers, "ks_factor"):
        logger.info(f"ks_factor: {config.parameters.multipliers.ks_factor}")
    if hasattr(config.parameters.multipliers, "Wc_factor"):
        logger.info(f"Wc_factor: {config.parameters.multipliers.Wc_factor}")
    if hasattr(config.parameters.multipliers, "Wg_factor"):
        logger.info(f"Wg_factor: {config.parameters.multipliers.Wg_factor}")
    if hasattr(config.parameters.multipliers, "Wg_Wc_tr"):
        logger.info(f"Wg_Wc_tr: {config.parameters.multipliers.Wg_Wc_tr}")
    if hasattr(config.parameters.multipliers, "CH_factor"):
        logger.info(f"CH_factor: {config.parameters.multipliers.CH_factor}")
    if hasattr(config.parameters.multipliers, "cel_factor"):
        logger.info(f"cel_factor: {config.parameters.multipliers.cel_factor}")
    if hasattr(config.parameters.multipliers, "chan_factor"):
        logger.info(f"chan_factor: {config.parameters.multipliers.chan_factor}")

# INITIAL CONDITIONS
if hasattr(config, "initial_conditions"):
    logger.info("")
    logger.info("=" * 60)
    logger.info("Initial Conditions:")
    logger.info("=" * 60)
    if hasattr(config.initial_conditions, "Ws"):
        logger.info(f"Ws (hillslope depth): {config.initial_conditions.Ws} m")
    if hasattr(config.initial_conditions, "Wcsat"):
        logger.info(f"Wcsat (capillary saturation): {config.initial_conditions.Wcsat}")
    if hasattr(config.initial_conditions, "Wgsat"):
        logger.info(f"Wgsat (gravitational saturation): {config.initial_conditions.Wgsat}")

# SIMULATION CONTROLS
logger.info("")
logger.info("=" * 60)
logger.info("Simulation Settings:")
logger.info("=" * 60)
logger.info(f"Timestep: {config.simulation.timestep} seconds")
logger.info(f"Decimation Factor: {config.simulation.decimation}")
logger.info(f"Soil Scheme: {config.simulation.soil_scheme}")
logger.info(f"Energy Balance: {config.simulation.energy_balance}")

# OUTPUT CONTROLS
logger.info("")
logger.info("=" * 60)
logger.info("Output States:")
logger.info("=" * 60)
logger.info(f"Discharge: {config.output_states.discharge}")
logger.info(f"Reservoir States: {config.output_states.reservoir_states}")
logger.info(f"Soil Capillary: {config.output_states.soil_capillary}")
logger.info(f"Soil Gravitational: {config.output_states.soil_gravitational}")
logger.info(f"Soil Plant: {config.output_states.soil_plant}")
logger.info(f"Soil Surface: {config.output_states.soil_surface}")
logger.info(f"Surface Temperature: {config.output_states.surface_temperature}")
logger.info(f"Ground Temperature: {config.output_states.ground_temperature}")
logger.info(f"Aquifer Head: {config.output_states.aquifer_head}")
logger.info(f"Evapotranspiration: {config.output_states.evapotranspiration}")

logger.info("")
logger.info("=" * 60)
logger.info("Output States Settings:")
logger.info("=" * 60)
logger.info(f"Output Format: {config.output_states_settings.output_format}")
logger.info(f"Output States: {config.output_states_settings.output_states}")
if config.output_states_settings.output_interval:
    logger.info(f"Output Interval: {config.output_states_settings.output_interval} seconds")
if config.output_states_settings.output_list:
    logger.info(f"Output List: {config.output_states_settings.output_list}")
logger.info(f"Flushing: {config.output_states_settings.flushing}")
logger.info(f"Max File Size: {config.output_states_settings.max_file_size} MB")

if hasattr(config, "output_report"):
    logger.info("")
    logger.info("=" * 60)
    logger.info("Output Report:")
    logger.info("=" * 60)
    if hasattr(config.output_report, "discharge"):
        logger.info(f"Discharge: {config.output_report.discharge}")
    if hasattr(config.output_report, "lateral_inflow"):
        logger.info(f"Lateral Inflow: {config.output_report.lateral_inflow}")

logger.info("")
logger.info("=" * 60)
logger.info("Output Forcing Data:")
logger.info("=" * 60)
logger.info(f"Meteo Data: {config.output_forcing_data.meteo_data}")

logger.info("")
logger.info("=" * 60)
logger.info("Output Report Settings:")
logger.info("=" * 60)
logger.info(f"Output Format: {config.output_report_settings.output_format}")
if hasattr(config.output_report_settings, "report_interval") and config.output_report_settings.report_interval:
    logger.info(f"Report Interval: {config.output_report_settings.report_interval} seconds")
logger.info(f"Reach Selection: {config.output_report_settings.reach_selection}")
if config.output_report_settings.reach_selection == "file":
    logger.info(f"Selection File: {config.output_report_settings.sel_file}")
if config.output_report_settings.reach_selection == "list":
    logger.info(f"Selected Reaches: {config.output_report_settings.sel_list}")

# ADVANCED SETTINGS
if hasattr(config, "advanced"):
    logger.info("")
    logger.info("=" * 60)
    logger.info("Advanced Settings:")
    logger.info("=" * 60)
    if hasattr(config.advanced, "log_level"):
        logger.info(f"Log Level: {config.advanced.log_level}")
    if hasattr(config.advanced, "log_file"):
        logger.info(f"Log File: {config.advanced.log_file}")


# Demonstrate validation by creating config with invalid data
logger.info("")
logger.info("=" * 60)
logger.info("Demonstrating Validation:")
logger.info("=" * 60)

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
  timestep: -100
  decimation: 1
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
    logger.info("ERROR: Negative timestep was accepted (should have been rejected!)")
except ValueError as e:
    logger.info("[OK] Validation worked! Negative timestep rejected during config loading")
    logger.info(f"     Error type: {type(e).__name__}")
finally:
    # Clean up test file
    if test_config_path.exists():
        test_config_path.unlink()

logger.info("")
logger.info("=" * 60)
logger.info("Configuration parser demo completed successfully!")
logger.info("=" * 60)
