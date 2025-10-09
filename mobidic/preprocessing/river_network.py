"""River network processing module for MOBIDIC.

This module provides functions to process river network shapefiles,
compute Strahler stream ordering, join single-tributary reaches,
and calculate routing parameters.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger
from pathlib import Path


def process_river_network(
    shapefile_path: str | Path,
    join_single_tributaries: bool = False,
    routing_params: dict | None = None,
) -> gpd.GeoDataFrame:
    """Process river network shapefile to create a complete network structure.

    This function reads a river network shapefile, builds the network topology,
    enforces binary tree structure (max 2 upstream tributaries per reach),
    computes Strahler stream ordering, optionally joins single-tributary reaches,
    and calculates routing parameters.

    Args:
        shapefile_path: Path to river network shapefile
        join_single_tributaries: If True, joins reaches with single tributaries (default: True)
        routing_params: Dictionary with routing parameters:
            - wcel: Wave celerity [m/s] (default: 5.0)
            - Br0: Width of 1st order streams [m] (default: 1.0)
            - NBr: Width-order exponent (default: 1.5)
            - n_Man: Manning coefficient [s/m^(1/3)] (default: 0.03)

    Returns:
        GeoDataFrame with processed network including topology and routing parameters

    Raises:
        FileNotFoundError: If shapefile does not exist
        ValueError: If id_field is not found in shapefile or network is invalid

    Examples:
        >>> network = process_river_network(
        ...     "river_network.shp",
        ...     "REACH_ID",
        ...     routing_params={"wcel": 3.0, "Br0": 1.0, "NBr": 1.5, "n_Man": 0.03}
        ... )
    """
    logger.info(f"Processing river network from {shapefile_path}")

    # Read shapefile
    gdf = gpd.read_file(shapefile_path)

    # Build network topology
    logger.debug("Building network topology")
    network = _build_network_topology(gdf)

    # Enforce binary tree structure (max 2 upstream tributaries per reach)
    logger.debug("Enforcing binary tree structure")
    network = _enforce_binary_tree(network)

    # Compute Strahler ordering
    logger.debug("Computing Strahler stream order")
    network = _compute_strahler_order(network)

    # Optionally join single-tributary reaches
    if join_single_tributaries:
        logger.debug("Joining single-tributary reaches")
        network = _join_single_tributaries(network)

    # Calculate routing parameters
    if routing_params is None:
        routing_params = {"wcel": 5.0, "Br0": 1.0, "NBr": 1.5, "n_Man": 0.03}

    logger.debug("Calculating routing parameters")
    network = _calculate_routing_parameters(network, routing_params)

    # Compute calculation order (topological order)
    logger.debug("Computing calculation order")
    network = _compute_calculation_order(network)

    logger.success(f"River network processed successfully: {len(network)} reaches")

    return network


def _build_network_topology(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Build network topology by detecting upstream/downstream connections.

    Args:
        gdf: GeoDataFrame with river network

    Returns:
        GeoDataFrame with added topology fields and all original shapefile fields preserved
    """
    # Create a copy to avoid modifying original (preserves all original fields)
    network = gdf.copy()

    # Create internal MOBIDIC ID (0-indexed) for topology mapping
    # All original shapefile fields are preserved in the network
    network["mobidic_id"] = np.arange(len(network))
    n_reaches = len(network)

    # Initialize topology fields
    network["upstream_1"] = np.nan
    network["upstream_2"] = np.nan
    network["downstream"] = -1
    network["strahler_order"] = 0

    # Extract start and end coordinates for each reach
    start_coords = []
    end_coords = []

    for geom in network.geometry:
        coords = np.array(geom.coords)
        start_coords.append(coords[0])
        end_coords.append(coords[-1])

    start_coords = np.array(start_coords)
    end_coords = np.array(end_coords)

    # Find connections by matching end/start nodes (within tolerance)
    tolerance = 0.05  # meters (adjust if needed)

    for i in range(n_reaches):
        # Find downstream reach (where end of i connects to start of another)
        distances = np.sqrt(np.sum((start_coords - end_coords[i]) ** 2, axis=1))
        downstream_candidates = np.where(distances < tolerance)[0]

        if len(downstream_candidates) > 0:
            downstream_idx = downstream_candidates[0]
            network.at[i, "downstream"] = downstream_idx

            # Update upstream connections in downstream reach
            if np.isnan(network.at[downstream_idx, "upstream_1"]):
                network.at[downstream_idx, "upstream_1"] = i
            elif np.isnan(network.at[downstream_idx, "upstream_2"]):
                network.at[downstream_idx, "upstream_2"] = i
        else:
            network.at[i, "downstream"] = -1  # Terminal reach

    # Log terminal reaches
    terminal_reaches = network[network["downstream"] == -1]["mobidic_id"].values
    logger.info(f"Found {len(terminal_reaches)} terminal reach(es): {terminal_reaches}")

    return network


def _enforce_binary_tree(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Enforce binary tree structure by adding fictitious reaches for non-binary junctions.

    This function identifies nodes with more than 2 upstream tributaries and adds
    short fictitious reaches (0.1m long) to convert them into binary junctions.
    This matches the behavior of MATLAB's bintree.m function.

    Fictitious reaches only contain MOBIDIC-specific fields (mobidic_id, geometry,
    upstream_1, upstream_2, downstream, strahler_order). Original shapefile fields
    are preserved only for real reaches.

    Args:
        network: GeoDataFrame with network topology

    Returns:
        GeoDataFrame with binary tree structure enforced
    """
    from shapely.geometry import LineString

    # Store list of original shapefile columns (before MOBIDIC processing)
    original_cols = [
        col
        for col in network.columns
        if col not in ["mobidic_id", "upstream_1", "upstream_2", "downstream", "strahler_order", "geometry"]
    ]

    # Find all unique downstream nodes (identified by end coordinates)
    end_coords = []
    for idx in network.index:
        geom = network.at[idx, "geometry"]
        coords = np.array(geom.coords)
        end_coords.append(coords[-1])

    end_coords = np.array(end_coords)

    # Group reaches by their downstream node (end coordinate)
    tolerance = 0.05
    fictitious_reaches = []
    max_mobidic_id = network["mobidic_id"].max()
    fictitious_id_counter = 1

    # Process each unique confluence node
    processed_nodes = set()

    for i in network.index:
        end_coord = end_coords[i]
        end_coord_tuple = tuple(end_coord)

        if end_coord_tuple in processed_nodes:
            continue

        # Find all reaches that end at this node (i.e., flow into it)
        distances = np.sqrt(np.sum((end_coords - end_coord) ** 2, axis=1))
        reaches_at_node = np.where(distances < tolerance)[0]

        # Find the downstream reach (the one that starts at this node)
        downstream_idx = network.at[i, "downstream"]

        if downstream_idx == -1:
            # This is a terminal node, skip
            processed_nodes.add(end_coord_tuple)
            continue

        # Count how many reaches flow into this node
        num_upstream = len(reaches_at_node)

        if num_upstream <= 2:
            # Binary or less, no action needed
            processed_nodes.add(end_coord_tuple)
            continue

        # We have more than 2 upstream reaches - need to add fictitious reaches
        logger.debug(
            f"Node at {end_coord} has {num_upstream} upstream reaches. "
            "Adding fictitious reaches to enforce binary tree."
        )

        # Keep the first 2 reaches connected to the original downstream node
        # Add fictitious reaches for the rest
        while num_upstream > 2:
            # Create a new fictitious node slightly offset from the confluence
            new_node_coord = end_coord + np.array([0.05, 0.05])

            # Create a fictitious reach from the new node to the original confluence
            fictitious_geom = LineString([new_node_coord, end_coord])

            # Create the fictitious reach entry
            fictitious_reach = {
                "mobidic_id": max_mobidic_id + fictitious_id_counter,
                "geometry": fictitious_geom,
                "upstream_1": np.nan,
                "upstream_2": np.nan,
                "downstream": downstream_idx,
                "strahler_order": 0,
            }

            fictitious_reaches.append(fictitious_reach)
            fictitious_id_idx = len(network) + len(fictitious_reaches) - 1

            # Redirect the last 2 upstream reaches to the new fictitious reach
            last_idx = reaches_at_node[-1]
            second_last_idx = reaches_at_node[-2]

            network.at[last_idx, "downstream"] = fictitious_id_idx
            network.at[second_last_idx, "downstream"] = fictitious_id_idx

            # Update the fictitious reach's upstream connections
            fictitious_reaches[-1]["upstream_1"] = last_idx
            fictitious_reaches[-1]["upstream_2"] = second_last_idx

            # Update downstream reach's upstream connections
            # The downstream could be in the network or in fictitious_reaches
            if downstream_idx < len(network):
                # Downstream is in the original network
                if network.at[downstream_idx, "upstream_1"] == last_idx:
                    network.at[downstream_idx, "upstream_1"] = fictitious_id_idx
                elif network.at[downstream_idx, "upstream_2"] == last_idx:
                    network.at[downstream_idx, "upstream_2"] = fictitious_id_idx

                if network.at[downstream_idx, "upstream_1"] == second_last_idx:
                    network.at[downstream_idx, "upstream_1"] = fictitious_id_idx
                elif network.at[downstream_idx, "upstream_2"] == second_last_idx:
                    network.at[downstream_idx, "upstream_2"] = fictitious_id_idx
            else:
                # Downstream is a fictitious reach
                fict_idx = downstream_idx - len(network)
                if fictitious_reaches[fict_idx]["upstream_1"] == last_idx:
                    fictitious_reaches[fict_idx]["upstream_1"] = fictitious_id_idx
                elif fictitious_reaches[fict_idx]["upstream_2"] == last_idx:
                    fictitious_reaches[fict_idx]["upstream_2"] = fictitious_id_idx

                if fictitious_reaches[fict_idx]["upstream_1"] == second_last_idx:
                    fictitious_reaches[fict_idx]["upstream_1"] = fictitious_id_idx
                elif fictitious_reaches[fict_idx]["upstream_2"] == second_last_idx:
                    fictitious_reaches[fict_idx]["upstream_2"] = fictitious_id_idx

            # Update for next iteration
            reaches_at_node = reaches_at_node[:-2]  # Remove the last 2
            downstream_idx = fictitious_id_idx  # The new downstream is the fictitious reach
            end_coord = new_node_coord  # The new confluence is the fictitious node
            num_upstream = len(reaches_at_node)
            fictitious_id_counter += 1

        processed_nodes.add(end_coord_tuple)

    # Add fictitious reaches to network
    if fictitious_reaches:
        fictitious_gdf = gpd.GeoDataFrame(fictitious_reaches, crs=network.crs)
        # Fill original shapefile columns with NaN for fictitious reaches
        for col in original_cols:
            if col not in fictitious_gdf.columns:
                fictitious_gdf[col] = np.nan
        network = gpd.GeoDataFrame(pd.concat([network, fictitious_gdf], ignore_index=True), crs=network.crs)
        logger.info(f"Added {len(fictitious_reaches)} fictitious reach(es) to enforce binary tree structure")

    return network


def _compute_strahler_order(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute Strahler stream order for each reach using recursive algorithm.

    The Strahler order is calculated as follows:
    - First-order streams are those with no tributaries
    - When two streams of order i join, the resulting stream is order i+1
    - When two streams of different order join, the resulting stream has the higher order

    Args:
        network: GeoDataFrame with network topology

    Returns:
        GeoDataFrame with strahler_order field populated
    """

    def _recursive_strahler(idx: int) -> int:
        """Recursively compute Strahler order for a reach."""
        if network.at[idx, "strahler_order"] > 0:
            return network.at[idx, "strahler_order"]

        upstream_1 = network.at[idx, "upstream_1"]
        upstream_2 = network.at[idx, "upstream_2"]

        # Base case: no upstream tributaries (first-order stream)
        if np.isnan(upstream_1) and np.isnan(upstream_2):
            network.at[idx, "strahler_order"] = 1
            return 1

        # Get orders of upstream reaches
        orders = []
        if not np.isnan(upstream_1):
            orders.append(_recursive_strahler(int(upstream_1)))
        if not np.isnan(upstream_2):
            orders.append(_recursive_strahler(int(upstream_2)))

        # Compute Strahler order
        if len(orders) == 0:
            order = 1
        elif len(orders) == 1:
            order = orders[0]
        else:  # Two tributaries
            if orders[0] == orders[1]:
                order = orders[0] + 1
            else:
                order = max(orders)

        network.at[idx, "strahler_order"] = order
        return order

    # Start from terminal reaches
    terminal_indices = network[network["downstream"] == -1].index

    for idx in terminal_indices:
        _recursive_strahler(idx)

    # Log order statistics
    order_counts = network["strahler_order"].value_counts().sort_index()
    logger.info(f"Strahler order distribution: {order_counts.to_dict()}")

    return network


def _join_single_tributaries(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Join reaches that have exactly one upstream tributary.

    This simplifies the network by merging linear sequences of reaches.

    Args:
        network: GeoDataFrame with network topology

    Returns:
        GeoDataFrame with simplified network
    """
    # Mark reaches to keep (those with order > 0)
    network["active"] = network["strahler_order"] > 0

    max_iterations = len(network)
    iteration = 0

    while iteration < max_iterations:
        # Find reaches with exactly one upstream tributary and order > 0
        has_one_upstream = (~np.isnan(network["upstream_1"])) & (np.isnan(network["upstream_2"])) & (network["active"])

        candidates = network[has_one_upstream].index

        if len(candidates) == 0:
            break

        # Process first candidate
        idx = candidates[0]
        upstream_idx = int(network.at[idx, "upstream_1"])

        # Deactivate upstream reach
        network.at[upstream_idx, "active"] = False

        # Merge upstream reach into current reach
        # Update upstream connections
        network.at[idx, "upstream_1"] = network.at[upstream_idx, "upstream_1"]
        network.at[idx, "upstream_2"] = network.at[upstream_idx, "upstream_2"]

        # Merge geometry (concatenate coordinates)
        upstream_geom = network.loc[upstream_idx, "geometry"]
        current_geom = network.loc[idx, "geometry"]

        upstream_coords = list(upstream_geom.coords)
        current_coords = list(current_geom.coords)
        merged_coords = upstream_coords + current_coords

        from shapely.geometry import LineString

        network.at[idx, "geometry"] = LineString(merged_coords)

        # Update downstream connections of upstream's upstream reaches
        for up_field in ["upstream_1", "upstream_2"]:
            up_up_idx = network.at[upstream_idx, up_field]
            if not np.isnan(up_up_idx):
                up_up_idx = int(up_up_idx)
                if network.at[up_up_idx, "downstream"] == upstream_idx:
                    network.at[up_up_idx, "downstream"] = idx

        iteration += 1

    # Filter to active reaches only and store original indices
    network_filtered = network[network["active"]].copy()
    old_indices = network_filtered.index.tolist()
    network_filtered = network_filtered.reset_index(drop=True)

    # Create mapping from old indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(old_indices)}

    for idx in network_filtered.index:
        # Remap downstream
        downstream = network_filtered.at[idx, "downstream"]
        if downstream >= 0 and downstream in old_to_new:
            network_filtered.at[idx, "downstream"] = old_to_new[downstream]
        elif downstream >= 0:
            network_filtered.at[idx, "downstream"] = -1  # Downstream was removed

        # Remap upstream
        for up_field in ["upstream_1", "upstream_2"]:
            upstream = network_filtered.at[idx, up_field]
            if not np.isnan(upstream):
                upstream = int(upstream)
                if upstream in old_to_new:
                    network_filtered.at[idx, up_field] = old_to_new[upstream]
                else:
                    network_filtered.at[idx, up_field] = np.nan

    network_filtered = network_filtered.drop(columns=["active"])

    logger.info(f"Joined single tributaries: {len(network)} -> {len(network_filtered)} reaches")

    return network_filtered


def _calculate_routing_parameters(network: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    """Calculate routing parameters for each reach.

    Args:
        network: GeoDataFrame with network topology
        params: Dictionary with routing parameters (wcel, Br0, NBr, n_Man)

    Returns:
        GeoDataFrame with added routing parameter fields
    """
    wcel = params["wcel"]
    Br0 = params["Br0"]
    NBr = params["NBr"]
    n_Man = params["n_Man"]

    # Calculate length from geometry
    network["length_m"] = network.geometry.length

    # Calculate channel width from Strahler order: B = Br0 * order^NBr
    network["width_m"] = Br0 * (network["strahler_order"] ** NBr)

    # Calculate lag time: tau = L / wcel
    network["lag_time_s"] = network["length_m"] / wcel

    # Calculate storage coefficient: K = 0.5 * exp(-L/10000)
    network["storage_coeff"] = 0.5 * np.exp(-network["length_m"] / 10000)

    # Store Manning coefficient
    network["n_manning"] = n_Man

    logger.debug(
        f"Routing parameters calculated: "
        f"length range [{network['length_m'].min():.1f}, {network['length_m'].max():.1f}] m, "
        f"width range [{network['width_m'].min():.2f}, {network['width_m'].max():.2f}] m"
    )

    return network


def _compute_calculation_order(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute calculation order (topological order) for routing.

    The calculation order determines the sequence in which reaches should be processed
    during routing, ensuring upstream reaches are computed before downstream ones.

    Args:
        network: GeoDataFrame with network topology

    Returns:
        GeoDataFrame with calc_order field
    """

    def _recursive_calc_order(idx: int) -> int:
        """Recursively compute calculation order for a reach."""
        if "calc_order" in network.columns and network.at[idx, "calc_order"] > 0:
            return network.at[idx, "calc_order"]

        upstream_1 = network.at[idx, "upstream_1"]
        upstream_2 = network.at[idx, "upstream_2"]

        # Base case: no upstream tributaries
        if np.isnan(upstream_1) and np.isnan(upstream_2):
            network.at[idx, "calc_order"] = 1
            return 1

        # Get calculation orders of upstream reaches
        orders = []
        if not np.isnan(upstream_1):
            orders.append(_recursive_calc_order(int(upstream_1)))
        if not np.isnan(upstream_2):
            orders.append(_recursive_calc_order(int(upstream_2)))

        # Calculation order is max upstream order + 1
        calc_order = max(orders) + 1 if orders else 1
        network.at[idx, "calc_order"] = calc_order
        return calc_order

    # Initialize calc_order column
    network["calc_order"] = 0

    # Start from terminal reaches
    terminal_indices = network[network["downstream"] == -1].index

    for idx in terminal_indices:
        _recursive_calc_order(idx)

    logger.debug(f"Calculation order range: [1, {network['calc_order'].max()}]")

    return network


def export_network(network: gpd.GeoDataFrame, output_path: str | Path, format: str = "parquet") -> None:
    """Export processed river network to file.

    Args:
        network: Processed river network GeoDataFrame
        output_path: Path to output file
        format: Output format, either 'parquet' (default) or 'shapefile'

    Raises:
        ValueError: If format is not supported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        network.to_parquet(output_path)
        logger.success(f"Network exported to GeoParquet: {output_path}")
    elif format == "shapefile":
        import warnings

        # Suppress shapefile field name truncation warnings (10 char limit is expected)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Column names longer than 10 characters.*")
            warnings.filterwarnings("ignore", message=".*Normalized/laundered field name.*")
            network.to_file(output_path)
        logger.success(f"Network exported to shapefile: {output_path}")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'shapefile'")


def load_network(network_path: str | Path) -> gpd.GeoDataFrame:
    """Load processed river network from file.

    Args:
        network_path: Path to network file (parquet or shapefile)

    Returns:
        GeoDataFrame with river network

    Raises:
        FileNotFoundError: If network file does not exist
    """
    network_path = Path(network_path)

    if not network_path.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    if network_path.suffix == ".parquet":
        network = gpd.read_parquet(network_path)
    else:
        network = gpd.read_file(network_path)

    logger.info(f"Network loaded from {network_path}: {len(network)} reaches")

    return network
