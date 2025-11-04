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
    join_single_tributaries: bool = True,
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
        ValueError: If network is invalid

    Examples:
        >>> network = process_river_network(
        ...     "river_network.shp",
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
    network["strahler_order"] = -1  # -1 indicates uncomputed (will be 1 or higher when computed)

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
    logger.info(f"Found {len(terminal_reaches)} terminal reach(es)")
    logger.debug(f"Terminal reach(es): {terminal_reaches}")

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

    # We need to rebuild the topology after adding fictitious reaches
    # First, clear the upstream connections (we'll rebuild them)
    network["upstream_1"] = np.nan
    network["upstream_2"] = np.nan

    # Extract end coordinates for each reach
    end_coords = []
    for idx in network.index:
        geom = network.at[idx, "geometry"]
        coords = np.array(geom.coords)
        end_coords.append(coords[-1])
    end_coords = np.array(end_coords)

    # Find unique downstream nodes (sorted to match MATLAB's unique() behavior)
    unique_downstream = network["downstream"].unique()
    unique_downstream = unique_downstream[unique_downstream >= 0]  # Exclude terminal reaches (-1)
    unique_downstream = np.sort(unique_downstream)  # Sort to match MATLAB unique() order

    max_mobidic_id = network["mobidic_id"].max()
    fictitious_id_counter = 1
    original_network_size = len(network)
    num_fictitious_added = 0
    max_iterations = len(network)  # Prevent infinite loops

    # Process each unique downstream node
    for downstream_node in unique_downstream:
        downstream_node = int(downstream_node)

        # Keep processing this node until it has <= 2 upstream reaches
        iteration = 0
        while iteration < max_iterations:
            # Find all reaches that flow into this downstream node
            reaches_to_node = network[network["downstream"] == downstream_node].index.tolist()
            num_upstream = len(reaches_to_node)

            # If binary or less, we're done with this node
            if num_upstream <= 2:
                break

            logger.debug(
                f"Downstream node {downstream_node} has {num_upstream} upstream reaches. "
                "Adding fictitious reach to enforce binary tree."
            )

            # Create new fictitious node ID
            new_fictitious_node_id = original_network_size + num_fictitious_added

            # Get the end coordinate of the LAST upstream reach
            last_reach_idx = reaches_to_node[-1]
            last_reach_end = end_coords[last_reach_idx]

            # Create fictitious reach vertices offset by ±0.05 from the last reach's end point
            fictitious_start = last_reach_end + np.array([0.05, 0.05])
            fictitious_end = last_reach_end - np.array([0.05, 0.05])

            # Create the fictitious reach geometry
            fictitious_geom = LineString([fictitious_start, fictitious_end])

            # Get the last 2 upstream reaches
            last_idx = reaches_to_node[-1]
            second_last_idx = reaches_to_node[-2]

            # Create the fictitious reach entry
            fictitious_data = {
                "mobidic_id": max_mobidic_id + fictitious_id_counter,
                "geometry": fictitious_geom,
                "upstream_1": np.nan,
                "upstream_2": np.nan,
                "downstream": downstream_node,
                "strahler_order": -1,
            }

            # Fill original shapefile columns with NaN
            for col in original_cols:
                fictitious_data[col] = np.nan

            # Add the fictitious reach to the network immediately
            # This is critical so it's available for the next iteration of the while loop
            new_row = gpd.GeoDataFrame([fictitious_data], crs=network.crs)
            for col in network.columns:
                if col not in new_row.columns:
                    new_row[col] = np.nan
            network = gpd.GeoDataFrame(pd.concat([network, new_row], ignore_index=True), crs=network.crs)

            # Redirect the last 2 upstream reaches to the new fictitious node
            network.at[last_idx, "downstream"] = new_fictitious_node_id
            network.at[second_last_idx, "downstream"] = new_fictitious_node_id

            # Update the end coordinate for the fictitious reach (for potential next iteration)
            end_coords = np.vstack([end_coords, fictitious_end])

            fictitious_id_counter += 1
            num_fictitious_added += 1

            # Loop continues to check if this node still has > 2 upstream reaches

    if num_fictitious_added > 0:
        logger.info(f"Added {num_fictitious_added} fictitious reach(es) to enforce binary tree structure")

    # Rebuild upstream connections from downstream connections
    network["upstream_1"] = np.nan
    network["upstream_2"] = np.nan

    for idx in network.index:
        downstream_idx = network.at[idx, "downstream"]
        if downstream_idx >= 0:
            downstream_idx = int(downstream_idx)
            # Add this reach to the downstream reach's upstream list
            if np.isnan(network.at[downstream_idx, "upstream_1"]):
                network.at[downstream_idx, "upstream_1"] = idx
            elif np.isnan(network.at[downstream_idx, "upstream_2"]):
                network.at[downstream_idx, "upstream_2"] = idx
            else:
                logger.error(
                    f"Reach {downstream_idx} has >2 upstream reaches after binary tree enforcement. "
                    f"This should not happen."
                )

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
        if network.at[idx, "strahler_order"] >= 1:
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

    # Handle disconnected components (reaches not reachable from any terminal reach)
    unprocessed = network[network["strahler_order"] == -1].index
    if len(unprocessed) > 0:
        logger.warning(
            f"Found {len(unprocessed)} reaches not connected to any terminal outlet. "
            "Processing as disconnected subnetworks."
        )
        for idx in unprocessed:
            if network.at[idx, "strahler_order"] == -1:  # Still unprocessed
                _recursive_strahler(idx)

    # Verify all reaches were processed
    still_unprocessed = network[network["strahler_order"] == -1]
    if len(still_unprocessed) > 0:
        logger.error(
            f"{len(still_unprocessed)} reaches still have unassigned Strahler order. "
            f"This may indicate circular references in the network topology."
        )

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
    # Mark reaches to keep (those with order >= 1)
    network["active"] = network["strahler_order"] >= 1

    max_iterations = len(network)  # Prevent infinite loops
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

    # Recreate mobidic_id indexing
    network_filtered["mobidic_id"] = network_filtered.index

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
        if "calc_order" in network.columns and network.at[idx, "calc_order"] >= 1:
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
    network["calc_order"] = -1  # -1 indicates uncomputed

    # Start from terminal reaches
    terminal_indices = network[network["downstream"] == -1].index

    for idx in terminal_indices:
        _recursive_calc_order(idx)

    # Handle disconnected components (reaches not reachable from any terminal reach)
    unprocessed = network[network["calc_order"] == -1].index
    if len(unprocessed) > 0:
        logger.warning(
            f"Found {len(unprocessed)} reaches not connected to any terminal outlet. "
            "Processing as disconnected subnetworks."
        )
        for idx in unprocessed:
            if network.at[idx, "calc_order"] == -1:  # Still unprocessed
                _recursive_calc_order(idx)

    # Verify all reaches were processed
    still_unprocessed = network[network["calc_order"] == -1]
    if len(still_unprocessed) > 0:
        logger.error(
            f"{len(still_unprocessed)} reaches still have unassigned calculation order. "
            f"This may indicate circular references in the network topology."
        )

    logger.debug(f"Calculation order range: [1, {network['calc_order'].max()}]")

    return network
