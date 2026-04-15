"""MOBIDIC: MATLAB -> Python translation
Script for checking the flow direction against the result from MATLAB
"""

import numpy as np
from pathlib import Path
from mobidic import (
    load_config,
    configure_logger,
    grid_to_matrix,
)
from mobidic.preprocessing import convert_to_mobidic_notation

# Path to configuration file
config_file = Path(__file__).parent.parent / "01-event-Arno-basin" / "Arno.yaml"

# Configuration
config = load_config(config_file)

# Read flow direction computed in matlab
zp = np.loadtxt(
    Path(__file__).parent.parent / "datasets" / "Arno" / "matlab" / "output" / "Arno_event_Nov_2023" / "zp.csv",
    delimiter=",",
)

# Configure logger
configure_logger(level="DEBUG")

# Step 1: Read flow direction and convert to MOBIDIC notation
print("\n" + "=" * 60)
print("STEP 1: Reading raster data")
print("=" * 60)

# Read flow direction
flow_dir_path = config.raster_files.flow_dir
if flow_dir_path:
    print(f"\nReading flow direction: {flow_dir_path}")
    flow_dir_result = grid_to_matrix(flow_dir_path)
    flow_dir = flow_dir_result["data"]
    print(f"  Grid shape: {flow_dir.shape}")
    print(f"  Resolution: {flow_dir_result['cellsize']} m")
else:
    raise ValueError("Flow direction raster not specified in configuration")

# Convert flow direction to MOBIDIC notation
flow_dir_mobidic = convert_to_mobidic_notation(flow_dir)

# Read flow direction from MATLAB
print("\n" + "=" * 60)
print("STEP 2: Comparing with MATLAB results")
print("=" * 60)
if zp.shape != flow_dir_mobidic.shape:
    raise ValueError("Shape mismatch between Python and MATLAB flow direction data")
difference = flow_dir_mobidic - zp
num_differences = np.count_nonzero(difference[~np.isnan(difference)])
if num_differences == 0:
    print("CHECK PASSED. All flow direction values match between Python and MATLAB.")
else:
    print(f"Number of differing cells: {num_differences}")
