"""MOBIDIC: MATLAB -> Python translation
Script for checking the preprocessing against buildgis in MATLAB.
"""

from scipy.io import loadmat
import numpy as np
from pathlib import Path
from mobidic import load_config, run_preprocessing, GISData, configure_logger

# Path to configuration file
config_file = Path(__file__).parent.parent / "01-event-Arno-basin" / "Arno.yaml"

# Configure logging
configure_logger(level="INFO")

# Step 1: Load configuration
config = load_config(config_file)

# Step 2: Run preprocessing
gisdata = run_preprocessing(config)

# Step 3: Save preprocessed data
gisdata_path = Path(config.paths.gisdata)
network_path = Path(config.paths.network)

# Create output directories if needed
gisdata_path.parent.mkdir(parents=True, exist_ok=True)
network_path.parent.mkdir(parents=True, exist_ok=True)

gisdata.save(gisdata_path, network_path)

# Step 4: Load preprocessed data
loaded_gisdata = GISData.load(gisdata_path, network_path)

# Step 5: Compare with MATLAB buildgis output
print("\n" + "=" * 80)
print("STEP 5: Comparing Python vs MATLAB buildgis results")
print("=" * 80)

# Load MATLAB results
matlab_file = Path(__file__).parent.parent / "datasets" / "Arno" / "matlab" / "gisdata" / "Arno_gisdata.mat"
if not matlab_file.exists():
    print(f"\nERROR: MATLAB file not found: {matlab_file}")
    print("Please ensure the MATLAB reference data is available.")
    exit(1)

print(f"\nLoading MATLAB data from: {matlab_file}")
mat_data = loadmat(matlab_file)

# Variables to compare: (MATLAB name, Python name, description)
comparisons = [
    ("zz", "dtm", "Digital Terrain Model (elevation)"),
    ("zp", "flow_dir", "Flow Direction"),
    ("zr", "flow_acc", "Flow Accumulation"),
    ("ks", "ks", "Hydraulic Conductivity"),
    ("Wc0", "Wc0", "Capillary Water Holding Capacity"),
    ("Wg0", "Wg0", "Gravitational Water Holding Capacity"),
    ("ch", "hillslope_reach_map", "Hillslope-to-Reach Mapping"),
]

# Summary statistics
all_match = True
results = []

for matlab_var, python_var, description in comparisons:
    print(f"\n{'=' * 80}")
    print(f"Comparing {description}")
    print(f"  MATLAB variable: {matlab_var}")
    print(f"  Python variable: {python_var}")
    print(f"{'=' * 80}")

    # Get MATLAB data
    if matlab_var not in mat_data:
        print(f"  WARNING: MATLAB variable '{matlab_var}' not found in .mat file")
        results.append((description, False, "MATLAB variable not found"))
        all_match = False
        continue

    matlab_array = mat_data[matlab_var]

    # Get Python data
    if python_var == "hillslope_reach_map":
        python_array = loaded_gisdata.hillslope_reach_map.copy()
        # Convert Python 0-based indexing to MATLAB 1-based indexing
        # MATLAB uses 1-based reach indices, Python uses 0-based mobidic_id
        valid_mask = ~np.isnan(python_array) & (python_array != -9999)
        python_array[valid_mask] = python_array[valid_mask] + 1
    elif python_var in loaded_gisdata.grids:
        python_array = loaded_gisdata.grids[python_var]
    else:
        print(f"  WARNING: Python variable '{python_var}' not found")
        results.append((description, False, "Python variable not found"))
        all_match = False
        continue

    # Check shapes
    print("\nShapes:")
    print(f"  MATLAB: {matlab_array.shape}")
    print(f"  Python: {python_array.shape}")

    if matlab_array.shape != python_array.shape:
        print("  ERROR: Shape mismatch!")
        results.append((description, False, f"Shape mismatch: {matlab_array.shape} vs {python_array.shape}"))
        all_match = False
        continue

    # Statistics
    print("\nMATLAB statistics:")
    print(f"  Min:  {np.nanmin(matlab_array):.6f}")
    print(f"  Max:  {np.nanmax(matlab_array):.6f}")
    print(f"  Mean: {np.nanmean(matlab_array):.6f}")
    print(f"  Std:  {np.nanstd(matlab_array):.6f}")
    print(f"  NaN cells: {np.sum(np.isnan(matlab_array))}")

    print("\nPython statistics:")
    print(f"  Min:  {np.nanmin(python_array):.6f}")
    print(f"  Max:  {np.nanmax(python_array):.6f}")
    print(f"  Mean: {np.nanmean(python_array):.6f}")
    print(f"  Std:  {np.nanstd(python_array):.6f}")
    print(f"  NaN cells: {np.sum(np.isnan(python_array))}")

    # Comparison
    print("\nComparison:")

    # Exact match (including NaN handling)
    exact_match = np.allclose(matlab_array, python_array, equal_nan=True, rtol=1e-9, atol=1e-12)

    if exact_match:
        print("  EXACT MATCH (rtol=1e-9, atol=1e-12)")
        results.append((description, True, "Exact match"))
    else:
        # Check NaN mask match
        matlab_nan = np.isnan(matlab_array)
        python_nan = np.isnan(python_array)
        nan_match_pct = 100 * np.sum(matlab_nan == python_nan) / matlab_array.size
        print(f"  NaN mask match: {nan_match_pct:.4f}%")

        # For non-NaN values, compute differences
        both_valid = ~matlab_nan & ~python_nan

        if np.sum(both_valid) > 0:
            matlab_valid = matlab_array[both_valid]
            python_valid = python_array[both_valid]

            # Absolute differences
            abs_diff = np.abs(matlab_valid - python_valid)
            max_abs_diff = np.max(abs_diff)
            mean_abs_diff = np.mean(abs_diff)

            # Relative differences
            rel_diff = abs_diff / (np.abs(matlab_valid) + 1e-15)
            max_rel_diff = np.max(rel_diff)
            mean_rel_diff = np.mean(rel_diff)

            print(f"  Valid cells: {np.sum(both_valid)}")
            print(f"  Max absolute difference: {max_abs_diff:.6e}")
            print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
            print(f"  Max relative difference: {max_rel_diff:.6e}")
            print(f"  Mean relative difference: {mean_rel_diff:.6e}")

            # Check with looser tolerance
            close_match = np.allclose(matlab_array, python_array, equal_nan=True, rtol=1e-6, atol=1e-9)

            if close_match:
                print("  CLOSE MATCH (rtol=1e-6, atol=1e-9)")
                results.append((description, True, f"Close match (max diff: {max_abs_diff:.2e})"))
            else:
                # Count cells that match within tolerance
                matching_cells = np.sum(np.isclose(matlab_valid, python_valid, rtol=1e-6, atol=1e-9))
                match_pct = 100 * matching_cells / len(matlab_valid)
                print(f"  Matching cells (rtol=1e-6): {matching_cells}/{len(matlab_valid)} ({match_pct:.2f}%)")

                if match_pct >= 99.9:
                    print(f"  VERY HIGH AGREEMENT ({match_pct:.4f}%)")
                    results.append((description, True, f"Very high agreement ({match_pct:.4f}%)"))
                elif match_pct >= 95.0:
                    print(f"  HIGH AGREEMENT ({match_pct:.2f}%)")
                    results.append((description, False, f"High agreement ({match_pct:.2f}%), investigate differences"))
                    all_match = False
                else:
                    print(f"  SIGNIFICANT DIFFERENCES ({match_pct:.2f}%)")
                    results.append((description, False, f"Significant differences ({match_pct:.2f}%)"))
                    all_match = False

                # Show first few mismatches
                mismatch_mask = ~np.isclose(matlab_valid, python_valid, rtol=1e-6, atol=1e-9)
                if np.sum(mismatch_mask) > 0:
                    n_show = min(5, np.sum(mismatch_mask))
                    mismatch_idx = np.where(both_valid)[0][mismatch_mask][:n_show]

                    print(f"\n  First {n_show} mismatches:")
                    for idx in mismatch_idx:
                        row, col = np.unravel_index(idx, matlab_array.shape)
                        m_val = matlab_array[row, col]
                        p_val = python_array[row, col]
                        diff = abs(m_val - p_val)
                        print(f"    ({row:4d}, {col:4d}): MATLAB={m_val:.6f}, Python={p_val:.6f}, diff={diff:.6e}")

# Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

for description, match, message in results:
    status = "OK" if match else "--"
    print(f"  {status} {description}: {message}")
