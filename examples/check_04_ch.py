"""MOBIDIC: MATLAB -> Python translation
Script for checking the computed hill -> reach mapping against the MATLAB version.
"""

import numpy as np
from mobidic import (
    load_config,
    configure_logger,
    process_river_network,
    compute_hillslope_cells,
    map_hillslope_to_reach,
)
from scipy.io import loadmat

# Configuration
config = load_config("examples/Arno/Arno.yaml")

# MATLAB ch matrix import file .mat (matlab/Arno/gisdata/Arno_gisdata.mat)
mat_data = loadmat("matlab/example/gisdata/Arno_gisdata.mat")
ch = mat_data["ch"]

# Configure logger
configure_logger(level="INFO")

# Step 1: Process river network
print("=" * 60)
print("STEP 1: Processing river network")
print("=" * 60)

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

# Step 2: Compute hillslope cells
print("\n" + "=" * 60)
print("STEP 2: Computing hillslope cells")
print("=" * 60)

network = compute_hillslope_cells(
    network=network,
    grid_path=config.raster_files.flow_dir,
)

# Step 3: Map hillslope cells to reaches
print("\n" + "=" * 60)
print("STEP 3: Mapping hillslope cells to reaches")
print("=" * 60)

reach_map = map_hillslope_to_reach(
    network=network,
    flowdir_path=config.raster_files.flow_dir,
    flow_dir_type=config.raster_settings.flow_dir_type,
)

print(f"\nReach map shape: {reach_map.shape}")
print(f"Unique reaches assigned: {len(set(reach_map[~np.isnan(reach_map)]))}")
print(f"Cells assigned to reaches: {np.sum(~np.isnan(reach_map))}")
print(f"Unassigned cells: {np.sum(reach_map == -9999)}")

# Step 4: Compare with MATLAB results ch vs reach_map
print("\n" + "=" * 60)
print("STEP 4: Comparing Python vs MATLAB results")
print("=" * 60)

print(f"\nMATLAB ch shape: {ch.shape}")
print(f"Python reach_map shape: {reach_map.shape}")

# Check shape
if ch.shape != reach_map.shape:
    print("WARNING: Shape mismatch!")
    print(f"  MATLAB: {ch.shape}")
    print(f"  Python: {reach_map.shape}")
    print("\nCannot proceed with comparison - shapes must match")
    exit(1)

# Convert Python output to 1-based indexing to match MATLAB
# MATLAB uses 1-based reach indices, Python uses 0-based mobidic_id
reach_map_1based = reach_map.copy()
valid_mask = ~np.isnan(reach_map_1based) & (reach_map_1based != -9999)
reach_map_1based[valid_mask] = reach_map_1based[valid_mask] + 1

# Statistics
matlab_valid = np.isfinite(ch) & (ch != -9999)
python_valid = np.isfinite(reach_map_1based) & (reach_map_1based != -9999)

matlab_unassigned = ch == -9999
python_unassigned = reach_map_1based == -9999

print("\nMATLAB statistics:")
print(f"  Valid cells (assigned to reaches): {np.sum(matlab_valid)}")
print(f"  Unassigned cells (-9999): {np.sum(matlab_unassigned)}")
print(f"  NaN cells: {np.sum(np.isnan(ch))}")
print(f"  Unique reaches: {len(np.unique(ch[matlab_valid]))}")

print("\nPython statistics:")
print(f"  Valid cells (assigned to reaches): {np.sum(python_valid)}")
print(f"  Unassigned cells (-9999): {np.sum(python_unassigned)}")
print(f"  NaN cells: {np.sum(np.isnan(reach_map_1based))}")
print(f"  Unique reaches: {len(np.unique(reach_map_1based[python_valid]))}")

# Compare values
print("\n" + "=" * 60)
print("Cell-by-cell comparison")
print("=" * 60)

# Exact match (including NaN)
exact_match = np.allclose(ch, reach_map_1based, equal_nan=True, rtol=0, atol=0)
print(f"Exact match (all cells): {exact_match}")

if not exact_match:
    # Count differences
    # For NaN cells
    matlab_nan = np.isnan(ch)
    python_nan = np.isnan(reach_map_1based)
    nan_match = np.sum(matlab_nan == python_nan)
    nan_total = ch.size

    print(f"\nNaN mask match: {nan_match}/{nan_total} cells ({100 * nan_match / nan_total:.2f}%)")

    # For valid cells (not NaN)
    both_valid = matlab_valid & python_valid
    both_unassigned = matlab_unassigned & python_unassigned

    # Cells that are valid in both
    if np.sum(both_valid) > 0:
        matches_valid = np.sum(ch[both_valid] == reach_map_1based[both_valid])
        total_valid = np.sum(both_valid)
        print(f"\nValid cells match: {matches_valid}/{total_valid} ({100 * matches_valid / total_valid:.2f}%)")

    # Cells that are unassigned in both
    if np.sum(both_unassigned) > 0:
        print(f"Unassigned cells match: {np.sum(both_unassigned)}")

    # Find mismatches
    # Mask for cells that are both not NaN
    both_not_nan = ~np.isnan(ch) & ~np.isnan(reach_map_1based)
    mismatch_mask = both_not_nan & (ch != reach_map_1based)
    n_mismatches = np.sum(mismatch_mask)

    total_not_nan = np.sum(both_not_nan)
    print(f"\nMismatches (non-NaN cells): {n_mismatches}/{total_not_nan} ({100 * n_mismatches / total_not_nan:.4f}%)")

    if n_mismatches > 0:
        # Show first few mismatches
        mismatch_indices = np.where(mismatch_mask)
        n_show = min(10, n_mismatches)

        print(f"\nFirst {n_show} mismatches (row, col, MATLAB, Python):")
        for i in range(n_show):
            row = mismatch_indices[0][i]
            col = mismatch_indices[1][i]
            matlab_val = ch[row, col]
            python_val = reach_map_1based[row, col]
            print(f"  ({row:4d}, {col:4d}): MATLAB={matlab_val:6.0f}, Python={python_val:6.0f}")

        # Histogram of differences
        diffs = ch[mismatch_mask] - reach_map_1based[mismatch_mask]
        print("\nDifference statistics:")
        print(f"  Mean difference: {np.mean(diffs):.2f}")
        print(f"  Std difference: {np.std(diffs):.2f}")
        print(f"  Min difference: {np.min(diffs):.2f}")
        print(f"  Max difference: {np.max(diffs):.2f}")

        # Count by reach assignment differences
        # Cells where MATLAB assigned but Python didn't
        matlab_assigned_python_not = both_not_nan & matlab_valid & ~python_valid
        print(f"\nCells assigned in MATLAB but not Python: {np.sum(matlab_assigned_python_not)}")

        # Cells where Python assigned but MATLAB didn't
        python_assigned_matlab_not = both_not_nan & python_valid & ~matlab_valid
        print(f"Cells assigned in Python but not MATLAB: {np.sum(python_assigned_matlab_not)}")

        # Cells where both assigned but to different reaches
        both_assigned_diff = both_valid & (ch != reach_map_1based)
        print(f"Cells assigned to different reaches: {np.sum(both_assigned_diff)}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if exact_match:
    print("  SUCCESS: Python implementation matches MATLAB exactly!")
else:
    total_cells = ch.size
    matching_cells = np.sum((ch == reach_map_1based) | (np.isnan(ch) & np.isnan(reach_map_1based)))
    match_percentage = 100 * matching_cells / total_cells

    print(f"   PARTIAL MATCH: {matching_cells}/{total_cells} cells match ({match_percentage:.4f}%)")

    if match_percentage >= 99.9:
        print("   Very high agreement - differences likely due to edge cases or numerical precision")
    elif match_percentage >= 95.0:
        print("   Good agreement - investigate remaining differences")
    else:
        print("   Significant differences - investigation required")
