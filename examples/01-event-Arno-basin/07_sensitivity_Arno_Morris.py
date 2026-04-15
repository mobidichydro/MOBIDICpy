"""Example: Morris global sensitivity analysis of Arno basin using pestpp-sen.

This script demonstrates the sensitivity analysis workflow:
1. Preprocess data, and convert meteorological forcing to NetCDF format
2. Load configuration
3. Set up PEST++ for sensitivity analysis
4. Run pestpp-sen with Morris method
5. Analyze parameter sensitivities

Prerequisites:
    - Install calibration dependencies and PEST++ binaries:
        make install-calib
            or (manually)
        pip install mobidicpy[calibration] && get-pestpp :pyemu
    - Ensure pestpp-sen executable is added to PATH

Usage:
    python examples/01-event-Arno-basin/07_sensitivity_Arno_Morris.py
"""

from pathlib import Path

from mobidic.calibration import PestSetup, load_calibration_config
from mobidic.calibration.config import CalibrationConfig
from mobidic import (
    load_config,
    save_gisdata,
    save_network,
    run_preprocessing,
    MeteoData,
)

# Settings
case_name = "sen_morris"  # Custom case name for sensitivity analysis
gsa_morris_r = 10  # Number of samples (r) for Morris sensitivity analysis


# Path to calibration configuration
calib_config_path = Path(__file__).parent / "Arno.calibration.yaml"

# Path to meteorological data in .mat format
meteodata_mat_path = (
    Path(__file__).parent.parent / "datasets" / "Arno" / "matlab" / "meteodata" / "Arno_event_Nov_2023.mat"
)

# Load calibration configuration
cc = load_calibration_config(calib_config_path)

# Path to main MOBIDIC config file
mobidic_config = Path(cc.mobidic_config)
config_file = mobidic_config if mobidic_config.is_absolute() else calib_config_path.parent / mobidic_config

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
# Step 2: Run PEST++ Morris sensitivity analysis
# =========================================================================

# Override pest_tool for sensitivity analysis
cc_dict = cc.model_dump()
cc_dict["pest_tool"] = "sen"
cc_dict["pest_options"] = {"gsa_method": "morris", "gsa_morris_r": gsa_morris_r, "gsa_morris_obs_sen": False}
cc_dict["parallel"] = {"num_workers": None}  # Use all available CPUs for parallel runs
cc_dict["case_name"] = case_name  # Custom case name

cc_sen = CalibrationConfig(**cc_dict)

# Set up PEST++
pest = PestSetup(cc_sen, base_path=calib_config_path.parent)
working_dir = pest.setup()

print(f"PEST++ SEN working directory created: {working_dir}")
print(f"Parameters to analyze: {[p.name for p in cc_sen.parameters]}")

# Run pestpp-sen
results = pest.run()

# Get parameter sensitivities
sens = results.get_parameter_sensitivities()
if sens is not None:
    print("\nParameter sensitivity:")
    print(sens.to_string(index=False))
