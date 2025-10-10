"""MOBIDIC: MATLAB -> Python translation
Script for checking the compute hillslope cells against the MATLAB version.
"""

import numpy as np
from scipy.io import loadmat
from mobidic import load_config
from mobidic import process_river_network
from mobidic.preprocessing import (
    compute_hillslope_cells,
)

# Load configuration
config = load_config("examples/Arno/Arno.yaml")


# MATLAB ret structure import file .mat (matlab/Arno/gisdata/Arno_gisdata.mat)
mat_data = loadmat("matlab/example/gisdata/Arno_gisdata.mat")
ret = mat_data["ret"]

# Process river network
network = process_river_network(
    shapefile_path=config.vector_files.river_network.shp,
    join_single_tributaries=True,
    routing_params={
        "wcel": config.parameters.routing.wcel,
        "Br0": config.parameters.routing.Br0,
        "NBr": config.parameters.routing.NBr,
        "n_Man": config.parameters.routing.n_Man,
    },
)

# Compute hillslope cells
network = compute_hillslope_cells(
    network=network,
    flowdir_path=config.raster_files.flow_dir,
)

# Compare with MATLAB results hillslope_cells vs versanti
print("Comparing Python hillslope_cells with MATLAB versanti field...")
print(f"Number of reaches in Python network: {len(network)}")
print(f"Number of reaches in MATLAB ret: {ret.shape[0]}")

# Count matches and mismatches
total_matches = 0
total_mismatches = 0
mismatched_reaches = []

for i in range(min(len(network), ret.shape[0])):
    # Get MATLAB versanti (1-based linear indices)
    matlab_versanti = ret["versanti"][i, 0].flatten()

    # Get Python hillslope_cells (0-based linear indices)
    python_cells = network.iloc[i]["hillslope_cells"]

    # Convert Python cells to 1-based for comparison
    python_cells_1based = np.array(sorted(python_cells)) + 1 if len(python_cells) > 0 else np.array([])
    matlab_versanti_sorted = np.array(sorted(matlab_versanti))

    # Compare
    if np.array_equal(python_cells_1based, matlab_versanti_sorted):
        total_matches += 1
    else:
        total_mismatches += 1
        mismatched_reaches.append(i)

        # Print details for first few mismatches
        if total_mismatches <= 5:
            print(f"\nMismatch at reach {i} (mobidic_id={network.iloc[i]['mobidic_id']}):")
            print(
                f"  MATLAB versanti: {matlab_versanti_sorted[:5]}"
                f"{'...' if len(matlab_versanti_sorted) > 5 else ''} "
                f"(total: {len(matlab_versanti_sorted)})"
            )
            print(
                f"  Python cells+1: {python_cells_1based[:5]}"
                f"{'...' if len(python_cells_1based) > 5 else ''} "
                f"(total: {len(python_cells_1based)})"
            )

print(f"\n{'=' * 60}")
print("Results:")
print(f"  Matches:    {total_matches}/{len(network)} ({100 * total_matches / len(network):.2f}%)")
print(f"  Mismatches: {total_mismatches}/{len(network)} ({100 * total_mismatches / len(network):.2f}%)")

if total_mismatches == 0:
    print("\nSUCCESS: All hillslope cells match MATLAB versanti!")
else:
    print(f"\nFAILURE: {total_mismatches} reaches have mismatched hillslope cells")
    print(f"  Mismatched reach indices: {mismatched_reaches[:20]}{'...' if len(mismatched_reaches) > 20 else ''}")

    # Detailed comparison for first mismatch
    if len(mismatched_reaches) > 0:
        idx = mismatched_reaches[0]
        matlab_versanti = ret["versanti"][idx, 0].flatten()
        python_cells = network.iloc[idx]["hillslope_cells"]
        python_cells_1based = np.array(sorted(python_cells)) + 1 if len(python_cells) > 0 else np.array([])
        matlab_versanti_sorted = np.array(sorted(matlab_versanti))

        print(f"\nDetailed comparison for reach {idx}:")
        print(f"  MATLAB: {matlab_versanti_sorted}")
        print(f"  Python: {python_cells_1based}")

        # Find differences
        matlab_only = set(matlab_versanti_sorted) - set(python_cells_1based)
        python_only = set(python_cells_1based) - set(matlab_versanti_sorted)

        if matlab_only:
            print(f"  In MATLAB only: {sorted(matlab_only)}")
        if python_only:
            print(f"  In Python only: {sorted(python_only)}")
