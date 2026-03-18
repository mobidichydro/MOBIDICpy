"""Example: Morris sensitivity analysis of Arno basin using pestpp-sen.

This script demonstrates the sensitivity analysis workflow:
1. Load calibration configuration
2. Set up PEST++ for sensitivity analysis
3. Run pestpp-sen with Morris method
4. Analyze parameter rankings

Prerequisites:
    - Install calibration dependencies: pip install mobidic[calibration]
    - Download PEST++ binaries: python -c "import pyemu; pyemu.utils.get_pestpp()"
"""

from pathlib import Path

from mobidic.calibration import PestSetup, load_calibration_config
from mobidic.calibration.config import CalibrationConfig

# Load calibration config and switch to SEN
calib_config_path = Path(__file__).parent / "calibration.yaml"
cc = load_calibration_config(calib_config_path)

# Override pest_tool for sensitivity analysis
cc_dict = cc.model_dump()
cc_dict["pest_tool"] = "sen"
cc_dict["pest_options"] = {"gsa_method": "morris", "morris_r": 4, "morris_p": 5}

cc_sen = CalibrationConfig(**cc_dict)

# Set up PEST++
pest = PestSetup(cc_sen, base_path=calib_config_path.parent)
working_dir = pest.setup()

print(f"PEST++ SEN working directory created: {working_dir}")
print(f"Parameters to analyze: {[p.name for p in cc_sen.parameters]}")

# Run pestpp-sen (requires PEST++ binaries)
# Uncomment to actually run:
# results = pest.run(num_workers=4)
#
# # Get parameter sensitivities
# sens = results.get_parameter_sensitivities()
# if sens is not None:
#     print("\nParameter sensitivity ranking:")
#     print(sens.to_string(index=False))
