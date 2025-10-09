""" MOBIDIC: MATLAB -> Python translation
Script for checking the network processing result from MATLAB ("ret" structure)
"""

import numpy as np
from scipy.io import loadmat
from mobidic import load_config
from mobidic import process_river_network

# Load configuration
config = load_config("examples/Arno/Arno.yaml")


# MATLAB ret structure import file .mat (matlab/Arno/gisdata/Arno_gisdata.mat)
mat_data = loadmat("matlab/example/gisdata/Arno_gisdata.mat")
ret = mat_data["ret"]

# Process river network
network = process_river_network(
    shapefile_path=config.vector_files.river_network.shp,
    join_single_tributaries=False,
    routing_params={
        "wcel": config.parameters.routing.wcel,
        "Br0": config.parameters.routing.Br0,
        "NBr": config.parameters.routing.NBr,
        "n_Man": config.parameters.routing.n_Man,
    },
)

# Extract MATLAB ret fields (MATLAB uses 1-based indexing, struct array)
ret_squeeze = ret.squeeze()
matlab_downstream = np.array([x[0] for x in ret_squeeze["ramovalle"]]).squeeze() - 1  # Convert to 0-based indexing
matlab_downstream[matlab_downstream < 0] = -1

matlab_strahler = np.array([x[0] for x in ret_squeeze["ordine"]]).squeeze()
matlab_calc_order = np.array([x[0] for x in ret_squeeze["ord_calc"]]).squeeze()

matlab_upstream = np.array([x[0] for x in ret_squeeze["ramimonte"]]).squeeze() - 1  # Convert to 0-based indexing
matlab_upstream_1 = matlab_upstream[:, 0]
matlab_upstream_2 = matlab_upstream[:, 1]

# Extract Python network fields
python_downstream = network["downstream"].values
python_strahler = network["strahler_order"].values
python_calc_order = network["calc_order"].values
python_upstream_1 = network["upstream_1"].values
python_upstream_2 = network["upstream_2"].values


# Cut the arrays to the minimum length
min_length = min(len(matlab_calc_order), len(python_calc_order))
matlab_calc_order = matlab_calc_order[:min_length]
python_calc_order = python_calc_order[:min_length]
matlab_downstream = matlab_downstream[:min_length]
python_downstream = python_downstream[:min_length]
matlab_upstream_1 = matlab_upstream_1[:min_length]
python_upstream_1 = python_upstream_1[:min_length]
matlab_upstream_2 = matlab_upstream_2[:min_length]
python_upstream_2 = python_upstream_2[:min_length]
matlab_strahler = matlab_strahler[:min_length]
python_strahler = python_strahler[:min_length]


# Compare upstream connectivity
print("\n=== Upstream connectivity ===")
print(f"MATLAB shape: {matlab_upstream_1.shape}")
print(f"Python shape: {python_upstream_1.shape}")
upstream_match_1 = np.allclose(matlab_upstream_1, python_upstream_1, equal_nan=True)
upstream_match_2 = np.allclose(matlab_upstream_2, python_upstream_2, equal_nan=True)
print(f"Match upstream_1: {upstream_match_1}")
print(f"Match upstream_2: {upstream_match_2}")
if not upstream_match_1:
    diff_mask = ~np.isclose(matlab_upstream_1, python_upstream_1, equal_nan=True)
    print(f"Differences in upstream_1: {np.sum(diff_mask)} reaches")
    print(f"First 5 diffs: MATLAB={matlab_upstream_1[diff_mask][:5]}, Python={python_upstream_1[diff_mask][:5]}")
    print(f"Indexes of differences (Python): {np.where(diff_mask)[0][:5]}")

if not upstream_match_2:
    diff_mask = ~np.isclose(matlab_upstream_2, python_upstream_2, equal_nan=True)
    print(f"Differences in upstream_2: {np.sum(diff_mask)} reaches")
    print(f"First 5 diffs: MATLAB={matlab_upstream_2[diff_mask][:5]}, Python={python_upstream_2[diff_mask][:5]}")
    print(f"Indexes of differences (Python): {np.where(diff_mask)[0][:5]}")

# Compare downstream connectivity
print("\n=== Downstream connectivity ===")
print(f"MATLAB shape: {matlab_downstream.shape}")
print(f"Python shape: {python_downstream.shape}")
downstream_match = np.allclose(matlab_downstream, python_downstream, equal_nan=True) # equal_nan=True include NaN in comparison
print(f"Match: {downstream_match}")
if not downstream_match:
    diff_mask = ~np.isclose(matlab_downstream, python_downstream, equal_nan=True)
    print(f"Differences: {np.sum(diff_mask)} reaches")
    print(f"First 5 diffs: MATLAB={matlab_downstream[diff_mask][:5]}, Python={python_downstream[diff_mask][:5]}")
    print(f"Indexes of differences (Python): {np.where(diff_mask)[0][:5]}")


# Compare Strahler order
print("\n=== Strahler order ===")
print(f"MATLAB shape: {matlab_strahler.shape}")
print(f"Python shape: {python_strahler.shape}")
strahler_match = np.allclose(matlab_strahler, python_strahler, equal_nan=True)
print(f"Match: {strahler_match}")
if not strahler_match:
    diff_mask = ~np.isclose(matlab_strahler, python_strahler, equal_nan=True)
    print(f"Differences: {np.sum(diff_mask)} reaches")
    print(f"First 5 diffs: MATLAB={matlab_strahler[diff_mask][:5]}, Python={python_strahler[diff_mask][:5]}")
    print(f"Indexes of differences (Python): {np.where(diff_mask)[0][:5]}")

# Compare calculation order
print("\n=== Calculation order ===")
print(f"MATLAB shape: {matlab_calc_order.shape}")
print(f"Python shape: {python_calc_order.shape}")
calc_order_match = np.allclose(matlab_calc_order, python_calc_order, equal_nan=True)
print(f"Match: {calc_order_match}")
if not calc_order_match:
    diff_mask = ~np.isclose(matlab_calc_order, python_calc_order, equal_nan=True)
    print(f"Differences: {np.sum(diff_mask)} reaches")
    print(f"First 5 diffs: MATLAB={matlab_calc_order[diff_mask][:5]}, Python={python_calc_order[diff_mask][:5]}")
    print(f"Indexes of differences (Python): {np.where(diff_mask)[0][:5]}")

# Summary
print("\n=== Summary ===")
print(f"All fields match: {downstream_match and strahler_match and calc_order_match}")
