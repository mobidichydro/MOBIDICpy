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
    Route lateral flow ONE STEP from upslope cells to immediate downstream neighbors.

    This function performs ONE-STEP routing matching MATLAB's hill_route.m behavior.
    Each cell receives flow ONLY from its immediate upstream neighbors, NOT from all
    upslope cells. Water moves gradually cell-by-cell over multiple timesteps.

    CRITICAL: This is NOT cumulative routing! To move water from headwaters to outlets
    requires calling this function once per timestep for many timesteps.

    Args:
        lateral_flow: 2D array of lateral flow from each cell [m³/s].
            Shape: (nrows, ncols). NaN values indicate no-data cells.
        flow_direction: 2D array of flow directions [dimensionless].
            Shape: (nrows, ncols). Uses either Grass (1-8) or Arc (power-of-2) notation.
        flow_dir_type: Flow direction notation, either "Grass" (1-8 coding) or
            "Arc" (1,2,4,8,16,32,64,128 coding). Default: "Grass".

    Returns:
        Upstream contribution array with same shape as lateral_flow [m³/s].
        Each cell contains ONLY flow from immediate upstream neighbors (NOT its own flow).

    Notes:
        Flow direction coding (D8) - MOBIDIC convention from MATLAB stack8point.m:
            Grass/MOBIDIC notation:   Arc notation:
            7  6  5                   64  128  32
            8  ·  4                   16   ·    8
            1  2  3                    1   2    4

        Direction number → (row_offset, col_offset) in matrix coordinates:
            1: (-1, -1) northwest,  2: (-1, 0) north,    3: (-1, 1) northeast,
            4: (0, 1) east,         5: (1, 1) southeast, 6: (1, 0) south,
            7: (1, -1) southwest,   8: (0, -1) west

        Special values:
            0: Outlet cell (no downstream neighbor)
            -1: Basin outlet marker (set by preprocessor at cell with max flow accumulation)

    Examples:
        >>> # Simple 3x3 grid - all cells flow to center
        >>> lateral_flow = np.ones((3, 3)) * 0.1  # 0.1 m³/s from each cell
        >>> # Flow directions in MOBIDIC notation (all point toward center at [1,1])
        >>> flow_direction = np.array([[5, 6, 7],
        ...                            [4, 0, 8],  # center [1,1] is outlet (0 = no flow out)
        ...                            [3, 2, 1]])  # all 8 neighbors drain to center
        >>> upstream = hillslope_routing(lateral_flow, flow_direction, "Grass")
        >>> upstream[1, 1]  # Center receives flow from all 8 neighbors (one-step)
        0.8  # 8 neighbors x 0.1 m³/s each = 0.8 m³/s (does NOT include center's own 0.1)
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

    # Initialize upstream contribution to ZERO (matching MATLAB line 19: uf=0*fl)
    # MATLAB hill_route does NOT include cell's own flow - only upstream contributions
    upstream_contribution = np.zeros_like(lateral_flow)

    # Define flow direction mappings (direction -> row/col offsets)
    # MOBIDIC convention matches MATLAB stack8point.m (lines 8-9):
    # i8=[-1 -1 -1 0 1 1 1 0]
    # j8=[-1 0 1 1 1 0 -1 -1]
    # This is the convention stored in gisdata after preprocessing
    if flow_dir_type == "Grass":
        # MOBIDIC/Grass convention (gisdata contains MOBIDIC notation)
        dir_map = {
            1: (-1, -1),  # up-left (i8[0]=-1, j8[0]=-1)
            2: (-1, 0),  # up (i8[1]=-1, j8[1]=0)
            3: (-1, 1),  # up-right (i8[2]=-1, j8[2]=1)
            4: (0, 1),  # right (i8[3]=0, j8[3]=1)
            5: (1, 1),  # down-right (i8[4]=1, j8[4]=1)
            6: (1, 0),  # down (i8[5]=1, j8[5]=0)
            7: (1, -1),  # down-left (i8[6]=1, j8[6]=-1)
            8: (0, -1),  # left (i8[7]=0, j8[7]=-1)
        }
    else:  # Arc notation
        # Arc notation: powers of 2 (if used in original data)
        dir_map = {
            1: (1, 0),  # down
            2: (1, 1),  # down-right
            4: (0, 1),  # right
            8: (-1, 1),  # up-right
            16: (-1, 0),  # up
            32: (-1, -1),  # up-left
            64: (0, -1),  # left
            128: (1, -1),  # down-left
        }

    # MATLAB hill_route does ONE-STEP routing: each cell receives flow from immediate upstream neighbors
    # NOT cumulative routing! Water moves cell-by-cell over multiple timesteps.
    #
    # For each cell, find its upstream neighbors and add their flow
    # Matching MATLAB lines 20-35: uf(k1) = uf(k1) + fl(st(k1))

    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(flow_direction[i, j]) or np.isnan(lateral_flow[i, j]):
                continue

            flow_dir = int(flow_direction[i, j])
            if flow_dir == 0 or flow_dir == -1:
                # Outlet cell (no downstream direction)
                # -1 is special marker for basin outlet (matching MATLAB buildgis line 645: zp(ifoc)=-1)
                continue

            if flow_dir not in dir_map:
                logger.warning(f"Invalid flow direction {flow_dir} at cell ({i}, {j}), skipping")
                continue

            # Find downstream cell - cell (i,j) flows INTO (down_i, down_j)
            di, dj = dir_map[flow_dir]
            down_i, down_j = i + di, j + dj

            # Check bounds
            if 0 <= down_i < nrows and 0 <= down_j < ncols:
                # Add this cell's flow to its downstream neighbor (one-step routing)
                # Matching MATLAB: uf(downstream) += fl(upstream)
                if not np.isnan(lateral_flow[i, j]):
                    upstream_contribution[down_i, down_j] += lateral_flow[i, j]

    logger.debug("Hillslope routing completed (one-step)")

    return upstream_contribution


def linear_channel_routing(
    network: GeoDataFrame,
    discharge_initial: np.ndarray,
    lateral_inflow: np.ndarray,
    dt: float,
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
        K = lag_time_s [s] (lag time used as storage coefficient)
        qL = lateral inflow + integrated upstream contributions [m³/s]

    Args:
        network: River network GeoDataFrame with columns:
            - mobidic_id: Internal reach ID (0-indexed)
            - upstream_1, upstream_2: Upstream reach IDs (NaN if none)
            - downstream: Downstream reach ID (NaN if outlet)
            - calc_order: Calculation order (lower values processed first)
            - lag_time_s: Lag time [s] (used as storage coefficient K)
        discharge_initial: Initial discharge for each reach [m³/s].
            Shape: (n_reaches,). Indexed by mobidic_id.
        lateral_inflow: Lateral inflow to each reach during this time step [m³/s].
            Shape: (n_reaches,). Indexed by mobidic_id.
        dt: Time step duration [s].

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
        ...     'lag_time_s': [3600.0, 7200.0],  # 1 hour and 2 hours
        ...     'geometry': [...]
        ... })
        >>> Q_init = np.array([10.0, 5.0])  # m³/s
        >>> qL = np.array([2.0, 1.0])  # m³/s lateral inflow
        >>> Q_final, state = linear_channel_routing(network, Q_init, qL, dt=900)
    """
    logger.debug(f"Starting linear channel routing for {len(network)} reaches, dt={dt}s")

    # Validate inputs
    required_cols = ["mobidic_id", "upstream_1", "upstream_2", "downstream", "calc_order", "lag_time_s"]
    missing_cols = [col for col in required_cols if col not in network.columns]
    if missing_cols:
        raise ValueError(f"Network missing required columns: {missing_cols}")

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

    # Calculate routing coefficients for all reaches using lag_time_s as K
    K = network["lag_time_s"].values
    C3 = np.exp(-dt / K)  # Recession coefficient
    C4 = 1 - C3  # Lateral inflow coefficient

    # Sort reaches by calculation order
    network_sorted = network.sort_values("calc_order")

    # Route through network in topological order
    for _, reach in network_sorted.iterrows():
        mobidic_id = int(reach["mobidic_id"])
        ki = mobidic_id_to_idx[mobidic_id]  # Get DataFrame index for this mobidic_id

        # Count number of upstream reaches (nm in MATLAB)
        nm = 0
        if pd.notna(reach["upstream_1"]):
            nm += 1
        if pd.notna(reach["upstream_2"]):
            nm += 1

        # Start with lateral inflow (MATLAB line 66: Qx(ki) = qL(ki))
        qL_total[ki] = lateral_inflow[ki]

        # Add contributions from upstream reaches (MATLAB lines 67-74)
        for upstream_col in ["upstream_1", "upstream_2"]:
            upstream_mobidic_id = reach[upstream_col]

            if pd.notna(upstream_mobidic_id):
                upstream_mobidic_id = int(upstream_mobidic_id)
                jj = mobidic_id_to_idx[upstream_mobidic_id]  # Get DataFrame index for upstream reach

                # Compute mean integral of upstream discharge over time step
                # Formula from MATLAB go_route_ord.m line 70 (LINEAR routing):
                # mean_upstream = Qx(jj)/C4(jj) + (Qx(jj) - Q(jj,tt-1)*C4(jj))/log(C3(jj))
                # where Qx(jj) = C4(jj) * qL_total(jj) from previous reach computation
                # This integrates the exponential decay from upstream reach over the time step

                if C3[jj] == 1.0:
                    # Special case: no decay (K → ∞)
                    # Mean = Qx(jj) / C4(jj)
                    mean_upstream = qL_total[jj] / C4[jj]
                elif abs(C3[jj]) < 1e-10:
                    # Special case: instant decay (K → 0)
                    # Only lateral inflow contributes
                    mean_upstream = qL_total[jj] / C4[jj]
                else:
                    # General case: compute integral mean
                    # Exact translation from MATLAB go_route_ord.m line 70:
                    # Qx(ki) = Qx(ki) + Qx(jj)/C4(jj) + (Qx(jj) - Q(jj,tt-1)*C4(jj))/log(C3(jj))
                    # Note: qL_total[jj] in Python = Qx(jj) in MATLAB (already multiplied by C4)
                    mean_upstream = qL_total[jj] / C4[jj] + (qL_total[jj] - discharge_initial[jj] * C4[jj]) / np.log(
                        C3[jj]
                    )

                qL_total[ki] += mean_upstream

        # MATLAB line 75: Qx(ki) = Qx(ki) * C4(ki)
        qL_total[ki] = qL_total[ki] * C4[ki]

        # MATLAB line 79: Qpast = Q(ki, tt-1)
        Qpast = discharge_initial[ki]

        # Check if reach is too short (MATLAB lines 120-130)
        # For LINEAR routing this should never happen (nt is always 1, never NaN)
        # But we keep this check for robustness and consistency with MATLAB structure
        if np.isnan(K[ki]) or K[ki] <= 0:
            # Reach too short - flow passes directly through
            if nm > 0:
                # Sum upstream discharges (MATLAB line 122)
                upstream_sum = 0.0
                for upstream_col in ["upstream_1", "upstream_2"]:
                    upstream_mobidic_id = reach[upstream_col]
                    if pd.notna(upstream_mobidic_id):
                        upstream_mobidic_id = int(upstream_mobidic_id)
                        jj = mobidic_id_to_idx[upstream_mobidic_id]
                        upstream_sum += discharge_final[jj]
                discharge_final[ki] = lateral_inflow[ki] + upstream_sum
            else:
                # No upstream reaches (MATLAB line 124)
                discharge_final[ki] = lateral_inflow[ki]
        else:
            # Normal routing (MATLAB lines 150-154)
            # MATLAB line 151: QQ(1,1) = Qx(ki) + C3(ki) * Qpast(1)
            QQ = qL_total[ki] + C3[ki] * Qpast

            # MATLAB line 154: Q(ki,tt) = QQ(nx(ki), nt(ki))
            # For LINEAR, nx=1 and nt=1, so this is just QQ(1,1)
            discharge_final[ki] = QQ

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
