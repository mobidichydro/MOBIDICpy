"""
Routing module for MOBIDIC hydrological model.

This module implements hillslope and channel routing algorithms. The hillslope routing
accumulates lateral flow contributions from upslope cells, while channel routing
propagates water through the river network.

Translated from MATLAB:
    - hill_route.m -> hillslope_routing()
    - go_route_ord.m + calc_par_ord.m (LINEAR case) -> linear_channel_routing()

"""

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from loguru import logger


def hillslope_routing(
    lateral_flow: np.ndarray,
    flow_direction: np.ndarray,
    flow_dir_type: str = "Grass",
) -> np.ndarray:
    """
    Route lateral flow from upslope cells to downslope cells following flow direction.

    This function accumulates lateral flow contributions by traversing the flow direction
    grid. Each cell receives flow from all upslope cells that drain into it, following
    the D8 flow direction algorithm.

    Args:
        lateral_flow: 2D array of lateral flow from each cell [m³/s].
            Shape: (nrows, ncols). NaN values indicate no-data cells.
        flow_direction: 2D array of flow directions [dimensionless].
            Shape: (nrows, ncols). Uses either Grass (1-8) or Arc (power-of-2) notation.
        flow_dir_type: Flow direction notation, either "Grass" (1-8 coding) or
            "Arc" (1,2,4,8,16,32,64,128 coding). Default: "Grass".

    Returns:
        Accumulated flow array with same shape as lateral_flow [m³/s].
        Each cell contains its own lateral flow plus contributions from all upslope cells.

    Notes:
        Flow direction coding (D8):
            Grass notation:     Arc notation:
            7  6  5             64  128  32
            8  ·  4             16   ·    8
            1  2  3              1   2    4

    Examples:
        >>> # Simple 3x3 grid with center cell receiving flow from all neighbors
        >>> lateral_flow = np.ones((3, 3)) * 0.1  # 0.1 m³/s from each cell
        >>> flow_direction = np.array([[5, 6, 7],
        ...                            [4, 0, 8],  # center cell is outlet
        ...                            [3, 2, 1]])  # all cells drain to center
        >>> accumulated = hillslope_routing(lateral_flow, flow_direction, "Grass")
        >>> accumulated[1, 1]  # Center cell receives flow from all 8 neighbors + itself
        0.9
    """
    logger.debug(f"Starting hillslope routing with flow_dir_type={flow_dir_type}, grid shape={lateral_flow.shape}")

    # Validate inputs
    if lateral_flow.shape != flow_direction.shape:
        raise ValueError(
            f"lateral_flow shape {lateral_flow.shape} must match flow_direction shape {flow_direction.shape}"
        )

    if flow_dir_type not in ["Grass", "Arc"]:
        raise ValueError(f"flow_dir_type must be 'Grass' or 'Arc', got '{flow_dir_type}'")

    nrows, ncols = lateral_flow.shape

    # Initialize accumulated flow with lateral flow
    accumulated_flow = lateral_flow.copy()

    # Define flow direction mappings (direction -> row/col offsets)
    if flow_dir_type == "Grass":
        # Grass notation: 1-8, clockwise from bottom-left
        dir_map = {
            1: (1, -1),  # bottom-left
            2: (1, 0),  # bottom
            3: (1, 1),  # bottom-right
            4: (0, 1),  # right
            5: (-1, 1),  # top-right
            6: (-1, 0),  # top
            7: (-1, -1),  # top-left
            8: (0, -1),  # left
        }
    else:  # Arc notation
        # Arc notation: powers of 2
        dir_map = {
            1: (1, 0),  # bottom
            2: (1, 1),  # bottom-right
            4: (0, 1),  # right
            8: (-1, 1),  # top-right
            16: (-1, 0),  # top
            32: (-1, -1),  # top-left
            64: (0, -1),  # left
            128: (1, -1),  # bottom-left
        }

    # Build a reverse map: for each cell, track which cells flow into it
    # This requires scanning the entire grid
    inflow_cells = [[[] for _ in range(ncols)] for _ in range(nrows)]

    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(flow_direction[i, j]) or np.isnan(lateral_flow[i, j]):
                continue

            flow_dir = int(flow_direction[i, j])
            if flow_dir == 0:
                # Outlet cell (no downstream direction)
                continue

            if flow_dir not in dir_map:
                logger.warning(f"Invalid flow direction {flow_dir} at cell ({i}, {j}), skipping")
                continue

            # Find downstream cell
            di, dj = dir_map[flow_dir]
            down_i, down_j = i + di, j + dj

            # Check bounds
            if 0 <= down_i < nrows and 0 <= down_j < ncols:
                # Record that cell (i,j) flows into (down_i, down_j)
                inflow_cells[down_i][down_j].append((i, j))

    # Accumulate flow by traversing from headwaters to outlets
    # We need to process cells in topological order (upstream to downstream)
    # Since we don't have explicit ordering, we iterate until convergence

    # Create a processing order based on flow accumulation concept
    # Process cells with fewer upstream contributors first
    processed = np.zeros((nrows, ncols), dtype=bool)
    max_iterations = nrows * ncols  # Safety limit

    for iteration in range(max_iterations):
        any_processed = False

        for i in range(nrows):
            for j in range(ncols):
                if processed[i, j]:
                    continue

                if np.isnan(lateral_flow[i, j]):
                    processed[i, j] = True
                    continue

                # Check if all upstream cells have been processed
                upstream_cells = inflow_cells[i][j]
                if all(processed[ui, uj] for ui, uj in upstream_cells):
                    # Accumulate flow from upstream cells
                    for ui, uj in upstream_cells:
                        if not np.isnan(accumulated_flow[ui, uj]):
                            accumulated_flow[i, j] += accumulated_flow[ui, uj]

                    processed[i, j] = True
                    any_processed = True

        if not any_processed:
            # No more cells can be processed
            break

    if iteration == max_iterations - 1:
        logger.warning("Hillslope routing reached maximum iterations")

    logger.debug(f"Hillslope routing completed in {iteration + 1} iterations")

    return accumulated_flow


def linear_channel_routing(
    network: GeoDataFrame,
    discharge_initial: np.ndarray,
    lateral_inflow: np.ndarray,
    dt: float,
    storage_coeff: str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Route water through river network using linear reservoir method.

    The linear routing method models each reach as a simple reservoir with exponential
    recession. Water is routed from upstream reaches to downstream reaches following
    the network topology, with contributions from lateral inflows.

    The routing equation for each reach is:
        Q_out(t+dt) = C3 * Q_out(t) + C4 * qL

    Where:
        C3 = exp(-dt/K)  [recession coefficient]
        C4 = 1 - C3      [lateral inflow coefficient]
        K = storage coefficient [s]
        qL = lateral inflow + integrated upstream contributions [m³/s]

    Args:
        network: River network GeoDataFrame with columns:
            - mobidic_id: Internal reach ID (0-indexed)
            - upstream_1, upstream_2: Upstream reach IDs (NaN if none)
            - downstream: Downstream reach ID (NaN if outlet)
            - calc_order: Calculation order (lower values processed first)
            - storage_coeff or {storage_coeff}: Storage coefficient K [s]
        discharge_initial: Initial discharge for each reach [m³/s].
            Shape: (n_reaches,). Indexed by mobidic_id.
        lateral_inflow: Lateral inflow to each reach during this time step [m³/s].
            Shape: (n_reaches,). Indexed by mobidic_id.
        dt: Time step duration [s].
        storage_coeff: Name of column containing storage coefficient [s].
            If None, uses 'storage_coeff'. Can also specify custom column name.

    Returns:
        Tuple containing:
            - discharge_final: Discharge at end of time step [m³/s], shape (n_reaches,)
            - routing_state: Dictionary with routing diagnostics:
                - 'C3': Recession coefficients for each reach
                - 'C4': Lateral inflow coefficients for each reach
                - 'qL_total': Total inflow (lateral + upstream) for each reach [m³/s]

    Raises:
        ValueError: If network is missing required columns
        ValueError: If array shapes don't match network size

    Notes:
        - Reaches are processed in calc_order to ensure upstream reaches are
          computed before downstream reaches
        - For upstream contributions, the function computes the mean integral
          of the exponential decay: ∫[C3^t dt] over the time step
        - Negative discharges are clipped to zero with a warning

    Examples:
        >>> # Simple 2-reach network: reach 0 flows into reach 1
        >>> network = gpd.GeoDataFrame({
        ...     'mobidic_id': [0, 1],
        ...     'upstream_1': [np.nan, 0],
        ...     'upstream_2': [np.nan, np.nan],
        ...     'downstream': [1, np.nan],
        ...     'calc_order': [0, 1],
        ...     'storage_coeff': [3600.0, 7200.0],  # 1 hour and 2 hours
        ...     'geometry': [...]
        ... })
        >>> Q_init = np.array([10.0, 5.0])  # m³/s
        >>> qL = np.array([2.0, 1.0])  # m³/s lateral inflow
        >>> Q_final, state = linear_channel_routing(network, Q_init, qL, dt=900)
    """
    logger.debug(f"Starting linear channel routing for {len(network)} reaches, dt={dt}s")

    # Validate inputs
    required_cols = ["mobidic_id", "upstream_1", "upstream_2", "downstream", "calc_order"]
    missing_cols = [col for col in required_cols if col not in network.columns]
    if missing_cols:
        raise ValueError(f"Network missing required columns: {missing_cols}")

    # Get storage coefficient column
    if storage_coeff is None:
        storage_coeff = "storage_coeff"
    if storage_coeff not in network.columns:
        raise ValueError(f"Storage coefficient column '{storage_coeff}' not found in network")

    n_reaches = len(network)

    if len(discharge_initial) != n_reaches:
        raise ValueError(f"discharge_initial length {len(discharge_initial)} must match number of reaches {n_reaches}")

    if len(lateral_inflow) != n_reaches:
        raise ValueError(f"lateral_inflow length {len(lateral_inflow)} must match number of reaches {n_reaches}")

    if dt <= 0:
        raise ValueError(f"Time step dt must be positive, got {dt}")

    # Create mapping from mobidic_id to DataFrame index (for array indexing)
    # mobidic_ids may not be consecutive (e.g., after joining reaches), but DataFrame indices are always 0 to n-1
    mobidic_id_to_idx = {int(network.at[idx, "mobidic_id"]): idx for idx in network.index}

    # Initialize output arrays (indexed by DataFrame index, not mobidic_id)
    discharge_final = np.zeros(n_reaches)
    qL_total = lateral_inflow.copy()

    # Calculate routing coefficients for all reaches
    K = network[storage_coeff].values
    C3 = np.exp(-dt / K)  # Recession coefficient
    C4 = 1 - C3  # Lateral inflow coefficient

    # Sort reaches by calculation order
    network_sorted = network.sort_values("calc_order")

    # Route through network in topological order
    for _, reach in network_sorted.iterrows():
        mobidic_id = int(reach["mobidic_id"])
        ki = mobidic_id_to_idx[mobidic_id]  # Get DataFrame index for this mobidic_id

        # Start with lateral inflow
        qL_total[ki] = lateral_inflow[ki]

        # Add contributions from upstream reaches
        for upstream_col in ["upstream_1", "upstream_2"]:
            upstream_mobidic_id = reach[upstream_col]

            if pd.notna(upstream_mobidic_id):
                upstream_mobidic_id = int(upstream_mobidic_id)
                jj = mobidic_id_to_idx[upstream_mobidic_id]  # Get DataFrame index for upstream reach

                # Compute mean integral of upstream discharge over time step
                # Formula from MATLAB: Qx(jj)/C4(jj) + (Qx(jj)-Q(jj,t-1)*C4(jj))/log(C3(jj))
                # This integrates the exponential decay from upstream reach

                if C3[jj] == 1.0:
                    # Special case: no decay (K → ∞)
                    # Mean = Qx(jj)
                    mean_upstream = qL_total[jj]
                elif abs(C3[jj]) < 1e-10:
                    # Special case: instant decay (K → 0)
                    # Only lateral inflow contributes
                    mean_upstream = qL_total[jj] / C4[jj]
                else:
                    # General case: compute integral mean
                    # ∫[Q(t)] dt from 0 to dt where Q(t) = C3^(t/dt) * Q(0) + (1-C3^(t/dt)) * qL
                    # After integration and division by dt:
                    term1 = qL_total[jj] / C4[jj]
                    term2 = (qL_total[jj] - discharge_initial[jj] * C4[jj]) / np.log(C3[jj])
                    mean_upstream = term1 + term2

                qL_total[ki] += mean_upstream

        # Apply routing equation: Q_out(t+dt) = C3 * Q_out(t) + C4 * qL
        discharge_final[ki] = C3[ki] * discharge_initial[ki] + C4[ki] * qL_total[ki]

    # Check for negative discharges
    negative_mask = discharge_final < 0
    if np.any(negative_mask):
        n_negative = np.sum(negative_mask)
        min_discharge = np.min(discharge_final[negative_mask])
        logger.warning(
            f"Linear routing produced {n_negative} negative discharges "
            f"(min={min_discharge:.6f} m³/s). Clipping to zero."
        )
        discharge_final = np.maximum(discharge_final, 0.0)

    # Prepare routing state dictionary
    routing_state = {
        "C3": C3,
        "C4": C4,
        "qL_total": qL_total,
    }

    logger.success(
        f"Linear channel routing completed. "
        f"Discharge range: [{discharge_final.min():.3f}, {discharge_final.max():.3f}] m³/s"
    )

    return discharge_final, routing_state
