"""
Routing module for MOBIDIC hydrological model.

This module implements hillslope and channel routing algorithms. The hillslope routing
accumulates lateral flow contributions from upslope cells, while channel routing
propagates water through the river network.

It includes Numba-optimized functions for performance.

Translated from MATLAB:
    - hill_route.m -> hillslope_routing()
    - go_route_ord.m + calc_par_ord.m (LINEAR case) -> linear_channel_routing()

"""

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from loguru import logger
from numba import njit


@njit(cache=True, fastmath=True)
def _hillslope_routing_kernel(
    lateral_flow: np.ndarray,
    flow_direction: np.ndarray,
    upstream_contribution: np.ndarray,
    nrows: int,
    ncols: int,
) -> None:
    """
    Numba-compiled kernel for hillslope routing with MOBIDIC flow direction notation.

    This function modifies upstream_contribution in-place for performance.
    Compiled to machine code with Numba for speedup over pure Python loops.

    Args:
        lateral_flow: 2D array of lateral flow [m³/s]
        flow_direction: 2D array of flow directions (MOBIDIC 1-8 notation)
        upstream_contribution: 2D output array (modified in-place)
        nrows: Number of rows
        ncols: Number of columns
    """
    # Pre-computed direction offsets for MOBIDIC notation (1-8)
    # Matches MATLAB stack8point.m: i8=[-1 -1 -1 0 1 1 1 0], j8=[-1 0 1 1 1 0 -1 -1]
    dir_offsets_i = np.array([-1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int32)
    dir_offsets_j = np.array([-1, 0, 1, 1, 1, 0, -1, -1], dtype=np.int32)

    for i in range(nrows):
        for j in range(ncols):
            # Skip NaN cells
            if np.isnan(flow_direction[i, j]) or np.isnan(lateral_flow[i, j]):
                continue

            flow_dir = int(flow_direction[i, j])

            # Skip outlets (0, -1) and invalid directions
            if flow_dir <= 0 or flow_dir > 8:
                continue

            # Get downstream cell offset (flow_dir 1-8 maps to index 0-7)
            idx = flow_dir - 1
            di = dir_offsets_i[idx]
            dj = dir_offsets_j[idx]
            down_i = i + di
            down_j = j + dj

            # Check bounds and accumulate flow
            if 0 <= down_i < nrows and 0 <= down_j < ncols:
                upstream_contribution[down_i, down_j] += lateral_flow[i, j]


def hillslope_routing(
    lateral_flow: np.ndarray,
    flow_direction: np.ndarray,
) -> np.ndarray:
    """
    Route lateral flow ONE STEP from upslope cells to immediate downstream neighbors.

    This function performs ONE-STEP routing matching MATLAB's hill_route.m behavior.
    Each cell receives flow ONLY from its immediate upstream neighbors, NOT from all
    upslope cells. Water moves gradually cell-by-cell over multiple timesteps.

    CRITICAL: This is NOT cumulative routing! To move water from headwaters to outlets
    requires calling this function once per timestep for many timesteps.

    PERFORMANCE: Uses Numba JIT compilation for speedup over pure Python loops.

    Args:
        lateral_flow: 2D array of lateral flow from each cell [m³/s].
            Shape: (nrows, ncols). NaN values indicate no-data cells.
        flow_direction: 2D array of flow directions in MOBIDIC notation (1-8) [dimensionless].
            Shape: (nrows, ncols). Flow directions are standardized to MOBIDIC convention
            during preprocessing, regardless of original format.

    Returns:
        Upstream contribution array with same shape as lateral_flow [m³/s].
        Each cell contains ONLY flow from immediate upstream neighbors (NOT its own flow).

    Notes:
        Flow direction coding (D8) - MOBIDIC convention from MATLAB stack8point.m:
            MOBIDIC notation (1-8):
            7  6  5
            8  ·  4
            1  2  3

        Direction number -> (row_offset, col_offset) in matrix coordinates:
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
        >>> upstream = hillslope_routing(lateral_flow, flow_direction)
        >>> upstream[1, 1]  # Center receives flow from all 8 neighbors (one-step)
        0.8  # 8 neighbors x 0.1 m³/s each = 0.8 m³/s (does NOT include center's own 0.1)
    """
    logger.debug(f"Starting hillslope routing with grid shape={lateral_flow.shape}")

    # Validate inputs
    if lateral_flow.shape != flow_direction.shape:
        raise ValueError(
            f"lateral_flow shape {lateral_flow.shape} must match flow_direction shape {flow_direction.shape}"
        )

    nrows, ncols = lateral_flow.shape

    # Initialize upstream contribution to ZERO (matching MATLAB line 19: uf=0*fl)
    # MATLAB hill_route does NOT include cell's own flow - only upstream contributions
    upstream_contribution = np.zeros_like(lateral_flow)

    # Call Numba-compiled kernel for maximum performance
    # Flow direction is always in MOBIDIC format (1-8) after preprocessing
    _hillslope_routing_kernel(
        lateral_flow,
        flow_direction,
        upstream_contribution,
        nrows,
        ncols,
    )

    logger.debug("Hillslope routing completed (one-step)")

    return upstream_contribution


@njit(cache=True, fastmath=True)
def _linear_routing_kernel(
    discharge_initial: np.ndarray,
    lateral_inflow: np.ndarray,
    sorted_reach_idx: np.ndarray,
    upstream_1_idx: np.ndarray,
    upstream_2_idx: np.ndarray,
    n_upstream: np.ndarray,
    K: np.ndarray,
    C3: np.ndarray,
    C4: np.ndarray,
    discharge_final: np.ndarray,
    qL_total: np.ndarray,
) -> None:
    """
    Numba-compiled kernel for linear channel routing.

    This function modifies discharge_final and qL_total in-place for performance.
    Compiled to machine code with Numba for significant speedup over pure Python loops.

    Args:
        discharge_initial: Initial discharge for each reach [m³/s]
        lateral_inflow: Lateral inflow to each reach [m³/s]
        sorted_reach_idx: Reach indices sorted by calc_order
        upstream_1_idx: Index of first upstream reach (-1 if none)
        upstream_2_idx: Index of second upstream reach (-1 if none)
        n_upstream: Number of upstream reaches (0, 1, or 2)
        K: Storage coefficient (lag time) [s]
        C3: Recession coefficients
        C4: Lateral inflow coefficients
        discharge_final: Output discharge array (modified in-place)
        qL_total: Total inflow array (modified in-place)
    """
    # Route through network in topological order
    for ki in sorted_reach_idx:
        # Start with lateral inflow (MATLAB line 66: Qx(ki) = qL(ki))
        qL_total[ki] = lateral_inflow[ki]

        # Add contributions from upstream reaches (MATLAB lines 67-74)
        jj1 = upstream_1_idx[ki]
        if jj1 >= 0:
            # Compute mean integral of upstream discharge over time step
            if C3[jj1] == 1.0:
                # Special case: no decay (K → ∞)
                mean_upstream = qL_total[jj1] / C4[jj1]
            elif abs(C3[jj1]) < 1e-10:
                # Special case: instant decay (K → 0)
                mean_upstream = qL_total[jj1] / C4[jj1]
            else:
                # General case: compute integral mean
                mean_upstream = qL_total[jj1] / C4[jj1] + (qL_total[jj1] - discharge_initial[jj1] * C4[jj1]) / np.log(
                    C3[jj1]
                )
            qL_total[ki] += mean_upstream

        jj2 = upstream_2_idx[ki]
        if jj2 >= 0:
            # Compute mean integral of upstream discharge over time step
            if C3[jj2] == 1.0:
                # Special case: no decay (K → ∞)
                mean_upstream = qL_total[jj2] / C4[jj2]
            elif abs(C3[jj2]) < 1e-10:
                # Special case: instant decay (K → 0)
                mean_upstream = qL_total[jj2] / C4[jj2]
            else:
                # General case: compute integral mean
                mean_upstream = qL_total[jj2] / C4[jj2] + (qL_total[jj2] - discharge_initial[jj2] * C4[jj2]) / np.log(
                    C3[jj2]
                )
            qL_total[ki] += mean_upstream

        # MATLAB line 75: Qx(ki) = Qx(ki) * C4(ki)
        qL_total[ki] = qL_total[ki] * C4[ki]

        # MATLAB line 79: Qpast = Q(ki, tt-1)
        Qpast = discharge_initial[ki]

        # Check if reach is too short (MATLAB lines 120-130)
        if np.isnan(K[ki]) or K[ki] <= 0:
            # Reach too short - flow passes directly through
            if n_upstream[ki] > 0:
                # Sum upstream discharges (MATLAB line 122)
                upstream_sum = 0.0
                if jj1 >= 0:
                    upstream_sum += discharge_final[jj1]
                if jj2 >= 0:
                    upstream_sum += discharge_final[jj2]
                discharge_final[ki] = lateral_inflow[ki] + upstream_sum
            else:
                # No upstream reaches (MATLAB line 124)
                discharge_final[ki] = lateral_inflow[ki]
        else:
            # Normal routing (MATLAB lines 150-154)
            # MATLAB line 151: QQ(1,1) = Qx(ki) + C3(ki) * Qpast(1)
            QQ = qL_total[ki] + C3[ki] * Qpast
            # For LINEAR, nx=1 and nt=1, so this is just QQ(1,1)
            discharge_final[ki] = QQ


def linear_channel_routing(
    network: GeoDataFrame | dict,
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

    PERFORMANCE: Uses Numba JIT compilation for significant speedup over pure Python loops.

    Args:
        network: River network GeoDataFrame with columns:
            - mobidic_id: Internal reach ID (0-indexed)
            - upstream_1, upstream_2: Upstream reach IDs (NaN if none)
            - downstream: Downstream reach ID (NaN if outlet)
            - calc_order: Calculation order (lower values processed first)
            - lag_time_s: Lag time [s] (used as storage coefficient K)
            OR dictionary with pre-processed topology (from Simulation._preprocess_network_topology):
            - 'upstream_1_idx': numpy array of first upstream indices
            - 'upstream_2_idx': numpy array of second upstream indices
            - 'n_upstream': numpy array of upstream counts
            - 'sorted_reach_idx': numpy array of reach indices sorted by calc_order
            - 'K': numpy array of storage coefficients
            - 'n_reaches': number of reaches
        discharge_initial: Initial discharge for each reach [m³/s].
            Shape: (n_reaches,). Indexed by DataFrame index (not mobidic_id).
        lateral_inflow: Lateral inflow to each reach during this time step [m³/s].
            Shape: (n_reaches,). Indexed by DataFrame index (not mobidic_id).
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
    # Check if network is pre-processed dictionary (from Simulation class)
    if isinstance(network, dict):
        # Use pre-processed topology (fast path)
        logger.debug(f"Starting linear channel routing for {network['n_reaches']} reaches, dt={dt}s")

        n_reaches = network["n_reaches"]
        upstream_1_idx = network["upstream_1_idx"]
        upstream_2_idx = network["upstream_2_idx"]
        n_upstream = network["n_upstream"]
        sorted_reach_idx = network["sorted_reach_idx"]
        K = network["K"]

    else:
        # Process GeoDataFrame (slower path, for backwards compatibility)
        logger.debug(f"Starting linear channel routing for {len(network)} reaches, dt={dt}s")

        # Validate inputs
        required_cols = ["mobidic_id", "upstream_1", "upstream_2", "downstream", "calc_order", "lag_time_s"]
        missing_cols = [col for col in required_cols if col not in network.columns]
        if missing_cols:
            raise ValueError(f"Network missing required columns: {missing_cols}")

        n_reaches = len(network)

        # Create mapping from mobidic_id to DataFrame index
        mobidic_id_to_idx = {int(network.at[idx, "mobidic_id"]): idx for idx in network.index}

        # Pre-extract topology to numpy arrays (Strategy 2)
        upstream_1_idx = np.array(
            [mobidic_id_to_idx.get(int(uid), -1) if pd.notna(uid) else -1 for uid in network["upstream_1"]],
            dtype=np.int32,
        )
        upstream_2_idx = np.array(
            [mobidic_id_to_idx.get(int(uid), -1) if pd.notna(uid) else -1 for uid in network["upstream_2"]],
            dtype=np.int32,
        )
        n_upstream = np.array(
            [
                (1 if pd.notna(network.at[idx, "upstream_1"]) else 0)
                + (1 if pd.notna(network.at[idx, "upstream_2"]) else 0)
                for idx in network.index
            ],
            dtype=np.int32,
        )

        # Get sorted reach indices
        sorted_reach_idx = network.sort_values("calc_order").index.values.astype(np.int32)

        # Extract K (lag time as storage coefficient)
        K = network["lag_time_s"].values

    # Validate array sizes
    if len(discharge_initial) != n_reaches:
        raise ValueError(f"discharge_initial length {len(discharge_initial)} must match number of reaches {n_reaches}")

    if len(lateral_inflow) != n_reaches:
        raise ValueError(f"lateral_inflow length {len(lateral_inflow)} must match number of reaches {n_reaches}")

    if dt <= 0:
        raise ValueError(f"Time step dt must be positive, got {dt}")

    # Initialize output arrays
    discharge_final = np.zeros(n_reaches, dtype=np.float64)
    qL_total = np.zeros(n_reaches, dtype=np.float64)

    # Calculate routing coefficients for all reaches using lag_time_s as K
    C3 = np.exp(-dt / K)  # Recession coefficient
    C4 = 1 - C3  # Lateral inflow coefficient

    # Call Numba-compiled kernel for maximum performance
    _linear_routing_kernel(
        discharge_initial,
        lateral_inflow,
        sorted_reach_idx,
        upstream_1_idx,
        upstream_2_idx,
        n_upstream,
        K,
        C3,
        C4,
        discharge_final,
        qL_total,
    )

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
