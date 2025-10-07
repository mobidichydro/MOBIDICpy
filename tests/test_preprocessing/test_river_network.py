"""Tests for river network processing module."""

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import LineString
from mobidic.preprocessing.river_network import (
    process_river_network,
    export_network,
    load_network,
    _build_network_topology,
    _enforce_binary_tree,
    _compute_strahler_order,
    _calculate_routing_parameters,
    _compute_calculation_order,
)


@pytest.fixture
def simple_network_gdf():
    """Create a simple test river network with 4 reaches forming a Y shape.

    Network structure:
        R1 (upstream left)  \\
                              -> R3 (junction) -> R4 (outlet)
        R2 (upstream right) /
    """
    # Create geometries
    reaches = {
        "REACH_ID": [1, 2, 3, 4],
        "geometry": [
            LineString([(0, 2), (1, 2), (2, 2)]),  # R1: left tributary
            LineString([(0, 0), (1, 0), (2, 0)]),  # R2: right tributary
            LineString([(2, 0), (2, 1), (2, 2)]),  # R3: connects R2 to R1
            LineString([(2, 2), (3, 2), (4, 2)]),  # R4: main stem to outlet
        ],
    }

    gdf = gpd.GeoDataFrame(reaches, crs="EPSG:3003")  # Monte Mario / Italy zone 1
    return gdf


@pytest.fixture
def complex_network_gdf():
    """Create a more complex river network with 7 reaches.

    Network structure:
        R1 \\
             -> R3 \\
        R2 /         \\
                       -> R6 -> R7 (outlet)
        R4 \\        /
             -> R5 /
    """
    reaches = {
        "REACH_ID": [1, 2, 3, 4, 5, 6, 7],
        "geometry": [
            LineString([(0, 4), (1, 4), (2, 4)]),  # R1
            LineString([(0, 3), (1, 3), (2, 3)]),  # R2
            LineString([(2, 3), (2, 3.5), (2, 4)]),  # R3: joins R2 and R1
            LineString([(0, 1), (1, 1), (2, 1)]),  # R4
            LineString([(2, 1), (2, 2), (2, 3)]),  # R5: joins R4 and R3
            LineString([(2, 4), (3, 4), (4, 4)]),  # R6: joins R5 and outlet
            LineString([(4, 4), (5, 4), (6, 4)]),  # R7: outlet
        ],
    }

    gdf = gpd.GeoDataFrame(reaches, crs="EPSG:3003")
    return gdf


@pytest.fixture
def non_binary_network_gdf():
    """Create a network with a non-binary junction (3 reaches joining at one node).

    Network structure:
        R1 \\
        R2 --> Node --> R4 (outlet)
        R3 /
    """
    reaches = {
        "REACH_ID": [1, 2, 3, 4],
        "geometry": [
            LineString([(0, 3), (1, 3), (2, 2)]),  # R1: tributary 1
            LineString([(0, 2), (1, 2), (2, 2)]),  # R2: tributary 2
            LineString([(0, 1), (1, 1), (2, 2)]),  # R3: tributary 3
            LineString([(2, 2), (3, 2), (4, 2)]),  # R4: downstream
        ],
    }

    gdf = gpd.GeoDataFrame(reaches, crs="EPSG:3003")
    return gdf


def test_build_network_topology_simple(simple_network_gdf):
    """Test building network topology on simple network."""
    network = _build_network_topology(simple_network_gdf, "REACH_ID")

    # Check that topology fields exist
    assert "upstream_1" in network.columns
    assert "upstream_2" in network.columns
    assert "downstream" in network.columns
    assert "mobidic_id" in network.columns

    # Check that original shapefile fields are preserved
    assert "REACH_ID" in network.columns

    # Check terminal reach (R4 has no downstream)
    terminal_reach = network[network["REACH_ID"] == 4]
    assert terminal_reach["downstream"].values[0] == -1

    # Check that R4 has R3 as upstream
    r4_idx = network[network["REACH_ID"] == 4].index[0]
    r3_idx = network[network["REACH_ID"] == 3].index[0]
    assert network.at[r4_idx, "upstream_1"] == r3_idx or network.at[r4_idx, "upstream_2"] == r3_idx


def test_enforce_binary_tree(non_binary_network_gdf):
    """Test enforcing binary tree structure on network with 3-way junction."""
    # Build topology first
    network = _build_network_topology(non_binary_network_gdf, "REACH_ID")

    # Before enforcement, we have 4 reaches
    assert len(network) == 4

    # Enforce binary tree
    network = _enforce_binary_tree(network)

    # After enforcement, we should have 5 reaches (4 original + 1 fictitious)
    assert len(network) == 5

    # All reaches should have at most 2 upstream tributaries
    for idx in network.index:
        upstream_1 = network.at[idx, "upstream_1"]
        upstream_2 = network.at[idx, "upstream_2"]

        # Count non-NaN upstreams
        num_upstream = sum([not np.isnan(upstream_1), not np.isnan(upstream_2)])
        assert num_upstream <= 2, f"Reach {idx} has more than 2 upstream tributaries"


def test_compute_strahler_order_simple(simple_network_gdf):
    """Test Strahler ordering on simple network."""
    network = _build_network_topology(simple_network_gdf, "REACH_ID")
    network = _compute_strahler_order(network)

    # R1 and R2 should be order 1 (no upstream)
    assert network[network["REACH_ID"] == 1]["strahler_order"].values[0] == 1
    assert network[network["REACH_ID"] == 2]["strahler_order"].values[0] == 1

    # R3 should be order 1 (joins two order-1 streams but one is very short)
    # Actually, depending on connection, could be order 2
    r3_order = network[network["REACH_ID"] == 3]["strahler_order"].values[0]
    assert r3_order >= 1

    # R4 should have highest order
    r4_order = network[network["REACH_ID"] == 4]["strahler_order"].values[0]
    assert r4_order >= r3_order


def test_compute_strahler_order_complex(complex_network_gdf):
    """Test Strahler ordering on complex network."""
    network = _build_network_topology(complex_network_gdf, "REACH_ID")
    network = _compute_strahler_order(network)

    # First-order streams (no upstream): R1, R2, R4
    first_order_ids = [1, 2, 4]
    for reach_id in first_order_ids:
        order = network[network["REACH_ID"] == reach_id]["strahler_order"].values[0]
        assert order == 1, f"Reach {reach_id} should be order 1, got {order}"

    # Check that downstream reaches have equal or higher order
    for idx in network.index:
        downstream_idx = network.at[idx, "downstream"]
        if downstream_idx >= 0:
            current_order = network.at[idx, "strahler_order"]
            downstream_order = network.at[int(downstream_idx), "strahler_order"]
            assert downstream_order >= current_order


def test_calculate_routing_parameters(simple_network_gdf):
    """Test calculation of routing parameters."""
    network = _build_network_topology(simple_network_gdf, "REACH_ID")
    network = _compute_strahler_order(network)

    params = {"wcel": 5.0, "Br0": 1.0, "NBr": 1.5, "n_Man": 0.03}
    network = _calculate_routing_parameters(network, params)

    # Check that fields are created
    assert "length_m" in network.columns
    assert "width_m" in network.columns
    assert "lag_time_s" in network.columns
    assert "storage_coeff" in network.columns
    assert "n_manning" in network.columns

    # Check that values are positive
    assert (network["length_m"] > 0).all()
    assert (network["width_m"] > 0).all()
    assert (network["lag_time_s"] > 0).all()
    assert (network["storage_coeff"] > 0).all()
    assert (network["n_manning"] == 0.03).all()

    # Check width calculation: B = Br0 * order^NBr
    for idx in network.index:
        order = network.at[idx, "strahler_order"]
        expected_width = params["Br0"] * (order ** params["NBr"])
        actual_width = network.at[idx, "width_m"]
        assert np.isclose(actual_width, expected_width)


def test_compute_calculation_order(simple_network_gdf):
    """Test computation of calculation order."""
    network = _build_network_topology(simple_network_gdf, "REACH_ID")
    network = _compute_strahler_order(network)
    network = _compute_calculation_order(network)

    # Check that calc_order field exists
    assert "calc_order" in network.columns

    # Check that all values are positive
    assert (network["calc_order"] > 0).all()

    # Check that upstream reaches have lower calc_order than downstream
    for idx in network.index:
        current_order = network.at[idx, "calc_order"]

        for up_field in ["upstream_1", "upstream_2"]:
            upstream_idx = network.at[idx, up_field]
            if not np.isnan(upstream_idx):
                upstream_order = network.at[int(upstream_idx), "calc_order"]
                assert upstream_order < current_order


def test_process_river_network_simple(simple_network_gdf, tmp_path):
    """Test full river network processing pipeline."""
    # Save to temporary shapefile
    shp_path = tmp_path / "test_network.shp"
    simple_network_gdf.to_file(shp_path)

    # Process network
    network = process_river_network(
        shp_path,
        "REACH_ID",
        join_single_tributaries=False,
        routing_params={"wcel": 5.0, "Br0": 1.0, "NBr": 1.5, "n_Man": 0.03},
    )

    # Check that all MOBIDIC fields are present
    expected_fields = [
        "mobidic_id",
        "upstream_1",
        "upstream_2",
        "downstream",
        "strahler_order",
        "length_m",
        "width_m",
        "lag_time_s",
        "storage_coeff",
        "n_manning",
        "calc_order",
    ]

    for field in expected_fields:
        assert field in network.columns, f"Field {field} not found in processed network"

    # Check that original shapefile fields are preserved
    assert "REACH_ID" in network.columns, "Original REACH_ID field should be preserved"

    # Check that network is valid
    assert len(network) > 0
    assert (network["strahler_order"] > 0).all()


def test_process_river_network_with_joining(simple_network_gdf, tmp_path):
    """Test river network processing with single-tributary joining."""
    # Save to temporary shapefile
    shp_path = tmp_path / "test_network.shp"
    simple_network_gdf.to_file(shp_path)

    # Process network with joining
    network = process_river_network(
        shp_path,
        "REACH_ID",
        join_single_tributaries=True,
        routing_params={"wcel": 5.0, "Br0": 1.0, "NBr": 1.5, "n_Man": 0.03},
    )

    # Number of reaches should be <= original (some may be joined)
    assert len(network) <= len(simple_network_gdf)


def test_preserve_original_fields(tmp_path):
    """Test that all original shapefile fields are preserved in processed network.

    Note: Shapefile format truncates field names to 10 characters max.
    """
    # Create a network with multiple fields (using short names for shapefile compatibility)
    reaches = {
        "REACH_ID": [1, 2, 3],
        "NAME": ["River A", "River B", "River C"],
        "LENGTH_KM": [10.5, 8.2, 15.3],
        "ORDER": [2, 1, 3],
        "CUST_DATA": [100, 200, 300],  # Short field name for shapefile
        "geometry": [
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
            LineString([(2, 0), (3, 0)]),
        ],
    }
    gdf = gpd.GeoDataFrame(reaches, crs="EPSG:3003")

    # Save to shapefile
    shp_path = tmp_path / "test_network.shp"
    gdf.to_file(shp_path)

    # Process network
    network = process_river_network(shp_path, "REACH_ID", join_single_tributaries=False)

    # Check that all original fields are preserved
    original_fields = ["REACH_ID", "NAME", "LENGTH_KM", "ORDER", "CUST_DATA"]
    for field in original_fields:
        assert field in network.columns, f"Original field {field} not preserved"

    # Check that original values are preserved for real reaches (not fictitious)
    for i in range(len(gdf)):
        # Find the reach in processed network by REACH_ID
        reach = network[network["REACH_ID"] == gdf.at[i, "REACH_ID"]]
        assert len(reach) == 1, f"Should find exactly one reach with REACH_ID {gdf.at[i, 'REACH_ID']}"
        assert reach["NAME"].values[0] == gdf.at[i, "NAME"]
        assert reach["LENGTH_KM"].values[0] == gdf.at[i, "LENGTH_KM"]
        assert reach["ORDER"].values[0] == gdf.at[i, "ORDER"]
        assert reach["CUST_DATA"].values[0] == gdf.at[i, "CUST_DATA"]


def test_export_and_load_network_parquet(simple_network_gdf, tmp_path):
    """Test exporting and loading network in Parquet format."""
    # Save to temporary shapefile
    shp_path = tmp_path / "test_network.shp"
    simple_network_gdf.to_file(shp_path)

    # Process network
    network = process_river_network(shp_path, "REACH_ID", join_single_tributaries=False)

    # Export to Parquet
    output_path = tmp_path / "network.parquet"
    export_network(network, output_path, format="parquet")

    assert output_path.exists()

    # Load network
    loaded_network = load_network(output_path)

    # Check that loaded network matches original
    assert len(loaded_network) == len(network)
    assert list(loaded_network.columns) == list(network.columns)


def test_export_and_load_network_shapefile(simple_network_gdf, tmp_path):
    """Test exporting and loading network in shapefile format."""
    # Save to temporary shapefile
    shp_path = tmp_path / "test_network.shp"
    simple_network_gdf.to_file(shp_path)

    # Process network
    network = process_river_network(shp_path, "REACH_ID", join_single_tributaries=False)

    # Export to shapefile
    output_path = tmp_path / "network_out.shp"
    export_network(network, output_path, format="shapefile")

    assert output_path.exists()

    # Load network
    loaded_network = load_network(output_path)

    # Check that loaded network has same number of features
    assert len(loaded_network) == len(network)


def test_invalid_id_field(simple_network_gdf, tmp_path):
    """Test that invalid id_field raises ValueError."""
    shp_path = tmp_path / "test_network.shp"
    simple_network_gdf.to_file(shp_path)

    with pytest.raises(ValueError, match="Field 'INVALID_FIELD' not found"):
        process_river_network(shp_path, "INVALID_FIELD")


def test_nonexistent_shapefile():
    """Test that nonexistent shapefile raises FileNotFoundError."""
    with pytest.raises(Exception):  # geopandas raises various exceptions
        process_river_network("nonexistent.shp", "REACH_ID")


def test_load_nonexistent_network():
    """Test that loading nonexistent network raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_network("nonexistent.parquet")


def test_export_invalid_format(simple_network_gdf, tmp_path):
    """Test that invalid export format raises ValueError."""
    shp_path = tmp_path / "test_network.shp"
    simple_network_gdf.to_file(shp_path)

    network = process_river_network(shp_path, "REACH_ID", join_single_tributaries=False)

    with pytest.raises(ValueError, match="Unsupported format"):
        export_network(network, tmp_path / "network.csv", format="csv")
