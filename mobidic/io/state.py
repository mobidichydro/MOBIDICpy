"""State file I/O for MOBIDIC simulations.

This module provides functions to load simulation state variables
(soil water content, discharge, etc.) from NetCDF format and the StateWriter
class for saving states incrementally to a single file.
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from mobidic import __version__
from mobidic.utils.crs import crs_to_cf_attrs


def load_state(
    input_path: str | Path,
    network_size: int,
    time_index: int = -1,
    config: "MOBIDICConfig | None" = None,  # noqa: F821
    gisdata: "GISData | None" = None,  # noqa: F821
) -> tuple["SimulationState", datetime, dict]:  # noqa: F821
    """
    Load simulation state from NetCDF file.

    Supports both single-timestep and multi-timestep state files.
    For multi-timestep files, loads the specified time index (default: last timestep).
    Automatically detects chunk files (e.g., states_001.nc) when the base path
    doesn't exist.

    If a state variable is missing from the file, it will be initialized using
    the initial conditions from the configuration file (if config and gisdata are
    provided), otherwise grid variables will be initialized with zeros and network
    variables will be initialized with zeros.

    Args:
        input_path: Path to output NetCDF file (may be chunked as _001.nc, _002.nc, etc.)
        network_size: Expected number of reaches in network. Validates consistency between
                        the saved state and current model setup.
        time_index: Index of timestep to load (default: -1 = last timestep)
        config: Optional MOBIDIC configuration for initializing missing variables
        gisdata: Optional GIS data for initializing missing variables

    Returns:
        Tuple of (state, time, metadata) where:
            - state: SimulationState object
            - time: Simulation time from file
            - metadata: Grid metadata dictionary

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If file format is invalid

    Examples:
        >>> from mobidic.io import load_state
        >>> # Load last timestep
        >>> state, time, metadata = load_state("states.nc", 1235)
        >>> # Load first timestep with config (for missing variables)
        >>> state, time, metadata = load_state("states.nc", 1235, time_index=0,
        ...                                      config=config, gisdata=gisdata)
        >>> # Works with chunked files too
        >>> state, time, metadata = load_state("states.nc", 1235)  # Auto-loads states_001.nc
    """
    input_path = Path(input_path)

    # Check if the exact path exists, otherwise look for chunk files
    if not input_path.exists():
        # Try to find chunk files
        stem = input_path.stem
        suffix = input_path.suffix
        parent = input_path.parent

        # Look for first chunk
        first_chunk = parent / f"{stem}_001{suffix}"
        if first_chunk.exists():
            logger.info(f"Base file not found, using chunk file: {first_chunk}")
            input_path = first_chunk
        else:
            raise FileNotFoundError(f"State file not found: {input_path} (also checked for {first_chunk})")

    logger.info(f"Loading simulation state from NetCDF: {input_path}")

    # Load dataset
    ds = xr.open_dataset(input_path)

    # Check if time is a dimension or scalar coordinate
    has_time_dim = "time" in ds.dims
    if has_time_dim:
        num_times = len(ds.time)
        logger.debug(f"Multi-timestep file detected: {num_times} timesteps available")
        # Select the specified timestep
        ds = ds.isel(time=time_index)
        logger.info(f"Loading timestep {time_index} (out of {num_times})")

    # Get grid shape from coordinates (needed for initializing missing variables)
    nrows = len(ds.y)
    ncols = len(ds.x)

    # Check if we can use config-based initialization for missing variables
    can_use_config = config is not None and gisdata is not None

    # Extract state variables (conditionally, since they may not be present)
    # Grid variables: initialize from config if missing and config provided, otherwise use zeros
    if "Wc" in ds:
        wc = ds["Wc"].values
    else:
        if can_use_config:
            logger.warning("Wc not found in state file, initializing from config initial conditions")
            from mobidic.core import constants as const

            # Apply same preprocessing as in Simulation.__init__
            wc0 = gisdata.grids["Wc0"] * config.parameters.multipliers.Wc_factor
            wc0 = np.maximum(wc0, const.W_MIN)

            # Apply Wg_Wc_tr transition if specified
            Wg_Wc_tr = config.parameters.multipliers.Wg_Wc_tr
            if Wg_Wc_tr >= 0:
                wg0 = gisdata.grids["Wg0"] * config.parameters.multipliers.Wg_factor
                wg0 = np.maximum(wg0, const.W_MIN)
                wtot = wc0 + wg0
                wg0 = np.minimum(Wg_Wc_tr * wg0, wtot)
                wc0 = wtot - wg0

            # Apply initial saturation from config
            wcsat = config.initial_conditions.Wcsat
            wc = wc0 * wcsat

            # Set NaN outside domain
            flow_acc = gisdata.grids["flow_acc"]
            wc[np.isnan(flow_acc)] = np.nan
        else:
            logger.warning("Wc not found in state file, initializing with zeros")
            wc = np.zeros((nrows, ncols))

    if "Wg" in ds:
        wg = ds["Wg"].values
    else:
        if can_use_config:
            logger.warning("Wg not found in state file, initializing from config initial conditions")
            from mobidic.core import constants as const

            # Apply same preprocessing as in Simulation.__init__
            wc0 = gisdata.grids["Wc0"] * config.parameters.multipliers.Wc_factor
            wg0 = gisdata.grids["Wg0"] * config.parameters.multipliers.Wg_factor
            wc0 = np.maximum(wc0, const.W_MIN)
            wg0 = np.maximum(wg0, const.W_MIN)

            # Apply Wg_Wc_tr transition if specified
            Wg_Wc_tr = config.parameters.multipliers.Wg_Wc_tr
            if Wg_Wc_tr >= 0:
                wtot = wc0 + wg0
                wg0 = np.minimum(Wg_Wc_tr * wg0, wtot)
                wc0 = wtot - wg0

            # Apply initial saturation from config
            wgsat = config.initial_conditions.Wgsat
            wg = wg0 * wgsat

            # Set NaN outside domain
            flow_acc = gisdata.grids["flow_acc"]
            wg[np.isnan(flow_acc)] = np.nan
        else:
            logger.warning("Wg not found in state file, initializing with zeros")
            wg = np.zeros((nrows, ncols))

    if "Ws" in ds:
        ws = ds["Ws"].values
    else:
        if can_use_config:
            logger.warning("Ws not found in state file, initializing from config initial conditions")
            # Initialize surface water from config
            ws_init = config.initial_conditions.Ws
            ws = np.full((nrows, ncols), ws_init)

            # Set NaN outside domain
            flow_acc = gisdata.grids["flow_acc"]
            ws[np.isnan(flow_acc)] = np.nan
        else:
            logger.warning("Ws not found in state file, initializing with zeros")
            ws = np.zeros((nrows, ncols))

    # Plant water: optional, initialize with zeros if missing
    if "Wp" in ds:
        wp = ds["Wp"].values
    else:
        if can_use_config:
            logger.warning("Wp not found in state file, initializing from config initial conditions")
            # Initialize plant water (currently zeros)
            wp = np.zeros((nrows, ncols))

            # Set NaN outside domain
            flow_acc = gisdata.grids["flow_acc"]
            wp[np.isnan(flow_acc)] = np.nan
        else:
            logger.warning("Wp not found in state file, initializing with zeros")
            wp = np.zeros((nrows, ncols))

    # Groundwater head: optional, only present when Linear groundwater is active
    h = None
    gw_active = can_use_config and config.parameters.groundwater.model == "Linear"
    if "h" in ds:
        h = ds["h"].values
    elif gw_active:
        h_init = config.initial_conditions.groundwater_head
        logger.warning(
            f"h not found in state file, initializing to {h_init} m from initial_conditions.groundwater_head"
        )
        flow_acc = gisdata.grids["flow_acc"]
        h = np.where(np.isfinite(flow_acc), h_init, np.nan)

    # Surface and deep-soil temperatures: only present when energy balance is active
    ts = None
    td = None
    energy_active = can_use_config and config.simulation.energy_balance != "None"
    if "Ts" in ds:
        ts = ds["Ts"].values
    elif energy_active:
        tcost = config.parameters.energy.Tconst
        logger.warning(f"Ts not found in state file, initializing to Tconst={tcost} K")
        flow_acc = gisdata.grids["flow_acc"]
        ts = np.where(np.isfinite(flow_acc), tcost, np.nan)
    if "Td" in ds:
        td = ds["Td"].values
    elif energy_active:
        tcost = config.parameters.energy.Tconst
        logger.warning(f"Td not found in state file, initializing to Tconst={tcost} K")
        flow_acc = gisdata.grids["flow_acc"]
        td = np.where(np.isfinite(flow_acc), tcost, np.nan)

    # Network variables: initialize with zeros if missing
    if "discharge" in ds:
        discharge = ds["discharge"].values
        # For multi-timestep files, discharge might be 1D after isel
        # For single-timestep files, it's already 1D
        if discharge.ndim > 1:
            discharge = discharge.flatten()
        # Check array shapes
        if len(discharge) != network_size:
            logger.warning(
                f"Network size mismatch: expected {network_size}, got {len(discharge)}. "
                "This may cause issues if network has changed."
            )
    else:
        logger.warning("discharge not found in state file, initializing with zeros")
        discharge = np.zeros(network_size)

    if "lateral_inflow" in ds:
        lateral_inflow = ds["lateral_inflow"].values
        # For multi-timestep files, lateral_inflow might be 1D after isel
        if lateral_inflow.ndim > 1:
            lateral_inflow = lateral_inflow.flatten()
    else:
        logger.warning("lateral_inflow not found in state file, initializing with zeros")
        lateral_inflow = np.zeros(network_size)

    # Extract time
    time = pd.Timestamp(ds["time"].values).to_pydatetime()

    # Extract grid metadata
    resolution_x = float(ds.x.values[1] - ds.x.values[0]) if len(ds.x) > 1 else float(ds.attrs.get("resolution_x", 1.0))
    resolution_y = float(ds.y.values[1] - ds.y.values[0]) if len(ds.y) > 1 else float(ds.attrs.get("resolution_y", 1.0))
    xllcorner = float(ds.x.values[0])
    yllcorner = float(ds.y.values[0])
    crs = ds["crs"].attrs.get("crs_wkt", "") if "crs" in ds else ""

    metadata = {
        "shape": (nrows, ncols),
        "resolution": (resolution_x, resolution_y),
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "crs": crs,
    }

    # Close dataset
    ds.close()

    # Import here to avoid circular import
    from mobidic.core.simulation import SimulationState

    state = SimulationState(wc, wg, wp, ws, discharge, lateral_inflow, h=h, ts=ts, td=td)

    logger.success(f"State loaded: time={time.isoformat()}, grid={nrows}x{ncols}, reaches={len(discharge)}")

    return state, time, metadata


class StateWriter:
    """
    Incremental NetCDF state writer with buffering, flushing, and automatic chunking.

    This class manages writing simulation states to NetCDF files across multiple
    timesteps, with configurable memory buffering, periodic flushing to disk, and
    automatic file chunking when size limits are reached.

    File chunking occurs when the current file reaches max_file_size before a new flush.
    For effective chunking, flushing must be set to a positive integer (e.g., flush
    every N timesteps). If flushing=-1 (flush only at end), all data is written in
    one operation and chunking may not work as expected.

    Note: Files may slightly exceed max_file_size by up to one flush worth of data,
    since the size check occurs before writing each flush batch.

    Args:
        output_path: Path to output NetCDF file (will be created/overwritten)
        grid_metadata: Dictionary with grid metadata (shape, resolution, crs, etc.)
        network_size: Number of reaches in network
        output_states: Configuration object specifying which state variables to save
        flushing: Flush interval (positive int = every N steps, -1 = only at end)
        max_file_size: Maximum file size in MB before creating a new chunk (default: 500)
        add_metadata: Additional global attributes (optional)

    Examples:
        >>> # With automatic chunking at 500 MB
        >>> writer = StateWriter("states.nc", metadata, 1235, config.output_states,
        ...                      flushing=10, max_file_size=500)
        >>> for step in range(num_steps):
        ...     writer.append_state(state, current_time)
        >>> writer.close()
        >>> # This may create: states_001.nc, states_002.nc, states_003.nc, etc.
    """

    def __init__(
        self,
        output_path: str | Path,
        grid_metadata: dict,
        network_size: int,
        output_states: "OutputStates",  # noqa: F821
        flushing: int = -1,
        max_file_size: float = 500.0,
        add_metadata: dict | None = None,
        reservoir_size: int = 0,
    ):
        """Initialize the state writer and create the NetCDF file."""
        self.base_output_path = Path(output_path)
        self.base_output_path.parent.mkdir(parents=True, exist_ok=True)

        self.grid_metadata = grid_metadata
        self.network_size = network_size
        self.reservoir_size = reservoir_size
        self.output_states = output_states
        self.flushing = flushing
        self.max_file_size_bytes = max_file_size * 1024 * 1024  # Convert MB to bytes
        self.add_metadata = add_metadata or {}

        # Chunking state
        self.current_chunk = 1
        self.output_path = self._get_chunk_path(self.current_chunk)
        self.chunk_files_created = []

        # Remove existing chunk files if present
        self._remove_existing_chunks()

        # Get grid dimensions
        self.nrows, self.ncols = grid_metadata["shape"]
        resolution = grid_metadata["resolution"]
        xllcorner = grid_metadata["xllcorner"]
        yllcorner = grid_metadata["yllcorner"]

        # Create coordinate arrays
        self.x = xllcorner + np.arange(self.ncols) * resolution[0]
        self.y = yllcorner + np.arange(self.nrows) * resolution[1]
        self.reaches = np.arange(network_size)
        self.reservoirs = np.arange(reservoir_size) if reservoir_size > 0 else np.array([])

        # Initialize buffer for states
        self.buffer = []
        self.buffer_times = []
        self.step_count = 0

        # Create the NetCDF file structure
        self._initialize_file()

        logger.debug(
            f"StateWriter initialized: {self.output_path}, flushing={self.flushing}, "
            f"max_file_size={max_file_size:.1f} MB"
        )

    def _get_chunk_path(self, chunk_number: int) -> Path:
        """
        Generate path for a chunk file.

        Args:
            chunk_number: Chunk number (1-indexed)

        Returns:
            Path object with chunk suffix (e.g., states_001.nc)
        """
        stem = self.base_output_path.stem
        suffix = self.base_output_path.suffix
        parent = self.base_output_path.parent
        return parent / f"{stem}_{chunk_number:03d}{suffix}"

    def _remove_existing_chunks(self) -> None:
        """Remove any existing chunk files matching the base path pattern."""
        stem = self.base_output_path.stem
        suffix = self.base_output_path.suffix
        parent = self.base_output_path.parent

        # Find and remove all chunk files
        import glob

        pattern = str(parent / f"{stem}_[0-9][0-9][0-9]{suffix}")
        existing_chunks = glob.glob(pattern)

        for chunk_file in existing_chunks:
            chunk_path = Path(chunk_file)
            logger.info(f"Removing existing chunk file: {chunk_path}")
            chunk_path.unlink()

    def _check_and_rotate_file(self) -> None:
        """Check current file size and rotate to new chunk if needed."""
        if not self.output_path.exists():
            return

        file_size = self.output_path.stat().st_size

        if file_size >= self.max_file_size_bytes:
            # Current chunk exceeded size limit - rotate to new chunk
            file_size_mb = file_size / 1024 / 1024
            logger.info(
                f"File size limit reached ({file_size_mb:.1f} MB >= {self.max_file_size_bytes / 1024 / 1024:.1f} MB). "
                f"Creating new chunk file."
            )

            # Record the completed chunk
            self.chunk_files_created.append(self.output_path)

            # Increment chunk number and update path
            self.current_chunk += 1
            self.output_path = self._get_chunk_path(self.current_chunk)

            logger.info(f"Starting new chunk: {self.output_path}")

    def _initialize_file(self) -> None:
        """Create the NetCDF file with unlimited time dimension."""
        # Create coordinates dictionary
        coords = {
            "x": (["x"], self.x),
            "y": (["y"], self.y),
        }

        # Add reach coordinate if discharge is enabled
        if self.output_states.discharge:
            coords["reach"] = (["reach"], self.reaches)

        # Add reservoir coordinate if reservoir_states is enabled
        if self.output_states.reservoir_states and self.reservoir_size > 0:
            coords["reservoir"] = (["reservoir"], self.reservoirs)

        # Create empty dataset with coordinates only (no time yet)
        self.ds = xr.Dataset(coords=coords)

        # Add coordinate attributes
        self.ds["x"].attrs = {
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection",
            "units": "m",
            "axis": "X",
        }
        self.ds["y"].attrs = {
            "standard_name": "projection_y_coordinate",
            "long_name": "y coordinate of projection",
            "units": "m",
            "axis": "Y",
        }
        if "reach" in self.ds.coords:
            self.ds["reach"].attrs = {
                "long_name": "reach index",
                "description": "MOBIDIC reach ID (mobidic_id)",
                "units": "1",
            }
        if "reservoir" in self.ds.coords:
            self.ds["reservoir"].attrs = {
                "long_name": "reservoir index",
                "description": "Reservoir index (0-based)",
                "units": "1",
            }

        # Add grid mapping variable for CRS (CF-1.12 compliance)
        self.ds["crs"] = ([], 0)
        self.ds["crs"].attrs = crs_to_cf_attrs(self.grid_metadata.get("crs"))

        # Add global attributes
        self.ds.attrs["Conventions"] = "CF-1.12"
        self.ds.attrs["title"] = "MOBIDIC simulation states"
        self.ds.attrs["source"] = "MOBIDICpy simulation"
        self.ds.attrs["history"] = f"Created by MOBIDICpy version {__version__} at {datetime.now().isoformat()}"
        self.ds.attrs.update(self.add_metadata)

    def append_state(self, state: "SimulationState", time: datetime) -> None:  # noqa: F821
        """
        Add a state to the buffer and flush if necessary.

        Args:
            state: SimulationState object containing state variables
            time: Current simulation time
        """
        # Import here to avoid circular import
        from mobidic.core.simulation import SimulationState

        # Create a copy of the state to avoid storing references to mutable arrays
        # This is critical because the same state object is reused across timesteps
        # Copy reservoir states if present
        reservoir_states_copy = None
        if state.reservoir_states is not None:
            from copy import deepcopy

            reservoir_states_copy = deepcopy(state.reservoir_states)

        state_copy = SimulationState(
            wc=state.wc.copy(),
            wg=state.wg.copy(),
            wp=state.wp.copy() if state.wp is not None else None,
            ws=state.ws.copy(),
            discharge=state.discharge.copy(),
            lateral_inflow=state.lateral_inflow.copy(),
            reservoir_states=reservoir_states_copy,
            h=state.h.copy() if state.h is not None else None,
            ts=state.ts.copy() if state.ts is not None else None,
            td=state.td.copy() if state.td is not None else None,
            et=state.et.copy() if state.et is not None else None,
        )

        # Add to buffer
        self.buffer.append(state_copy)
        self.buffer_times.append(time)
        self.step_count += 1

        # Check if we need to flush
        should_flush = False
        if self.flushing > 0 and self.step_count % self.flushing == 0:
            should_flush = True

        if should_flush:
            self.flush()
            logger.debug(f"StateWriter flushed at step {self.step_count}")

    def flush(self) -> None:
        """Write buffered states to disk using efficient append mode."""
        if not self.buffer:
            return

        # Check if we need to rotate to a new chunk BEFORE writing
        # This prevents files from exceeding the size limit
        self._check_and_rotate_file()

        logger.debug(f"Flushing {len(self.buffer)} states to {self.output_path}")

        # Convert buffer to arrays with time dimension
        num_buffered = len(self.buffer)

        # Initialize data arrays for this flush
        data_vars = {}

        # Grid variables
        if self.output_states.soil_capillary:
            wc_data = np.array([s.wc for s in self.buffer])
            data_vars["Wc"] = (["time", "y", "x"], wc_data)

        if self.output_states.soil_gravitational:
            wg_data = np.array([s.wg for s in self.buffer])
            data_vars["Wg"] = (["time", "y", "x"], wg_data)

        if self.output_states.soil_plant:
            # Check if any state has wp (plant water)
            has_wp = any(s.wp is not None for s in self.buffer)
            if has_wp:
                wp_data = np.array(
                    [s.wp if s.wp is not None else np.full((self.nrows, self.ncols), np.nan) for s in self.buffer]
                )
                data_vars["Wp"] = (["time", "y", "x"], wp_data)

        if self.output_states.soil_surface:
            ws_data = np.array([s.ws for s in self.buffer])
            data_vars["Ws"] = (["time", "y", "x"], ws_data)

        if self.output_states.aquifer_head:
            has_h = any(s.h is not None for s in self.buffer)
            if has_h:
                h_data = np.array(
                    [s.h if s.h is not None else np.full((self.nrows, self.ncols), np.nan) for s in self.buffer]
                )
                data_vars["h"] = (["time", "y", "x"], h_data)

        if self.output_states.surface_temperature:
            has_ts = any(s.ts is not None for s in self.buffer)
            if has_ts:
                ts_data = np.array(
                    [s.ts if s.ts is not None else np.full((self.nrows, self.ncols), np.nan) for s in self.buffer]
                )
                data_vars["Ts"] = (["time", "y", "x"], ts_data)

        if self.output_states.ground_temperature:
            has_td = any(s.td is not None for s in self.buffer)
            if has_td:
                td_data = np.array(
                    [s.td if s.td is not None else np.full((self.nrows, self.ncols), np.nan) for s in self.buffer]
                )
                data_vars["Td"] = (["time", "y", "x"], td_data)

        if self.output_states.evapotranspiration:
            has_et = any(s.et is not None for s in self.buffer)
            if has_et:
                et_data = np.array(
                    [s.et if s.et is not None else np.full((self.nrows, self.ncols), np.nan) for s in self.buffer]
                )
                data_vars["ET"] = (["time", "y", "x"], et_data)

        # Network variables
        if self.output_states.discharge:
            discharge_data = np.array([s.discharge for s in self.buffer])
            lateral_inflow_data = np.array([s.lateral_inflow for s in self.buffer])
            data_vars["discharge"] = (["time", "reach"], discharge_data)
            data_vars["lateral_inflow"] = (["time", "reach"], lateral_inflow_data)

        # Reservoir variables
        if self.output_states.reservoir_states and self.reservoir_size > 0:
            # Check if any state has reservoir_states
            has_reservoirs = any(s.reservoir_states is not None for s in self.buffer)
            if has_reservoirs:
                # Extract reservoir data (volume, stage, inflow, outflow)
                res_volume_data = np.array([[r.volume for r in s.reservoir_states] for s in self.buffer])
                res_stage_data = np.array([[r.stage for r in s.reservoir_states] for s in self.buffer])
                res_inflow_data = np.array([[r.inflow for r in s.reservoir_states] for s in self.buffer])
                res_outflow_data = np.array([[r.outflow for r in s.reservoir_states] for s in self.buffer])

                data_vars["reservoir_volume"] = (["time", "reservoir"], res_volume_data)
                data_vars["reservoir_stage"] = (["time", "reservoir"], res_stage_data)
                data_vars["reservoir_inflow"] = (["time", "reservoir"], res_inflow_data)
                data_vars["reservoir_outflow"] = (["time", "reservoir"], res_outflow_data)

        # Create dataset for this flush
        flush_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": (["time"], self.buffer_times),
                "x": (["x"], self.x),
                "y": (["y"], self.y),
            },
        )

        # Add reach coordinate if discharge is enabled
        if self.output_states.discharge:
            flush_ds.coords["reach"] = (["reach"], self.reaches)

        # Add reservoir coordinate if reservoir_states is enabled
        if self.output_states.reservoir_states and self.reservoir_size > 0:
            flush_ds.coords["reservoir"] = (["reservoir"], self.reservoirs)

        # Add variable attributes
        if "Wc" in flush_ds:
            flush_ds["Wc"].attrs = {
                "long_name": "Capillary Water Content",
                "units": "m",
                "description": "Water held by capillary forces in soil",
                "grid_mapping": "crs",
            }

        if "Wg" in flush_ds:
            flush_ds["Wg"].attrs = {
                "long_name": "Gravitational Water Content",
                "units": "m",
                "description": "Drainable water in soil large pores",
                "grid_mapping": "crs",
            }

        if "Wp" in flush_ds:
            flush_ds["Wp"].attrs = {
                "long_name": "Plant/Canopy Water Content",
                "units": "m",
                "description": "Water intercepted by vegetation canopy",
                "grid_mapping": "crs",
            }

        if "Ws" in flush_ds:
            flush_ds["Ws"].attrs = {
                "long_name": "Surface Water Content",
                "units": "m",
                "description": "Water in surface depressions",
                "grid_mapping": "crs",
            }

        if "h" in flush_ds:
            flush_ds["h"].attrs = {
                "long_name": "Groundwater Head",
                "units": "m",
                "description": "Linear-reservoir groundwater head",
                "grid_mapping": "crs",
            }

        if "Ts" in flush_ds:
            flush_ds["Ts"].attrs = {
                "long_name": "Surface Temperature",
                "units": "K",
                "description": "Land surface temperature from energy balance",
                "grid_mapping": "crs",
            }

        if "Td" in flush_ds:
            flush_ds["Td"].attrs = {
                "long_name": "Deep Soil Temperature",
                "units": "K",
                "description": "Deep-soil temperature from energy balance",
                "grid_mapping": "crs",
            }

        if "ET" in flush_ds:
            flush_ds["ET"].attrs = {
                "long_name": "Actual Evapotranspiration Rate",
                "units": "m s-1",
                "description": "Actual evapotranspiration as bounded by soil water availability",
                "grid_mapping": "crs",
            }

        if "discharge" in flush_ds:
            flush_ds["discharge"].attrs = {
                "long_name": "River Discharge",
                "units": "m3 s-1",
                "description": "Discharge at downstream end of each reach",
            }

        if "lateral_inflow" in flush_ds:
            flush_ds["lateral_inflow"].attrs = {
                "long_name": "Lateral Inflow",
                "units": "m3 s-1",
                "description": "Lateral inflow from hillslope to each reach",
            }

        if "reservoir_volume" in flush_ds:
            flush_ds["reservoir_volume"].attrs = {
                "long_name": "Reservoir Volume",
                "units": "m3",
                "description": "Water volume stored in reservoir",
            }

        if "reservoir_stage" in flush_ds:
            flush_ds["reservoir_stage"].attrs = {
                "long_name": "Reservoir Stage",
                "units": "m",
                "description": "Water level/stage in reservoir",
            }

        if "reservoir_inflow" in flush_ds:
            flush_ds["reservoir_inflow"].attrs = {
                "long_name": "Reservoir Inflow",
                "units": "m3 s-1",
                "description": "Total inflow to reservoir from upstream reaches",
            }

        if "reservoir_outflow" in flush_ds:
            flush_ds["reservoir_outflow"].attrs = {
                "long_name": "Reservoir Outflow",
                "units": "m3 s-1",
                "description": "Total outflow from reservoir (release + withdrawals)",
            }

        # Add time attributes
        flush_ds["time"].attrs = {
            "long_name": "simulation time",
            "axis": "T",
        }

        # Add coordinate attributes
        flush_ds["x"].attrs = {
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection",
            "units": "m",
            "axis": "X",
        }
        flush_ds["y"].attrs = {
            "standard_name": "projection_y_coordinate",
            "long_name": "y coordinate of projection",
            "units": "m",
            "axis": "Y",
        }
        if "reach" in flush_ds.coords:
            flush_ds["reach"].attrs = {
                "long_name": "reach index",
                "description": "MOBIDIC reach ID (mobidic_id)",
                "units": "1",
            }

        # Determine write mode: first write or append
        is_first_write = not self.output_path.exists()

        if is_first_write:
            # First write - add CRS variable and global attributes
            flush_ds["crs"] = xr.DataArray(
                data=np.int32(0),  # Explicitly use int32 to avoid type issues
                dims=[],
                attrs=crs_to_cf_attrs(self.grid_metadata.get("crs")),
            )
            flush_ds.attrs["Conventions"] = "CF-1.12"
            flush_ds.attrs["title"] = "MOBIDIC simulation states"
            flush_ds.attrs["source"] = "MOBIDICpy simulation"
            flush_ds.attrs["history"] = f"Created by MOBIDICpy version {__version__} at {datetime.now().isoformat()}"
            flush_ds.attrs.update(self.add_metadata)

            # Write initial file with compression
            encoding = {var: {"zlib": True, "complevel": 4} for var in flush_ds.data_vars if var != "crs"}
            encoding["crs"] = {"dtype": "int32"}  # Explicitly set CRS dtype

            flush_ds.to_netcdf(self.output_path, encoding=encoding, engine="netcdf4", mode="w", unlimited_dims=["time"])
            flush_ds.close()
        else:
            # Append mode - use efficient NetCDF append via netCDF4 library
            # This avoids the slow read-concatenate-write cycle
            import netCDF4 as nc4

            # Open existing file in append mode
            with nc4.Dataset(self.output_path, "a") as nc_file:
                # Get current time dimension length
                time_dim_len = len(nc_file.dimensions["time"])

                # Extend time dimension
                new_time_len = time_dim_len + num_buffered

                # Append time values
                time_var = nc_file.variables["time"]
                for i, t in enumerate(self.buffer_times):
                    time_var[time_dim_len + i] = nc4.date2num(t, time_var.units, time_var.calendar)

                # Append data variables
                for var_name in flush_ds.data_vars:
                    if var_name == "crs":
                        continue  # Skip CRS - it's a scalar

                    nc_var = nc_file.variables[var_name]
                    data = flush_ds[var_name].values

                    # Append along time dimension
                    nc_var[time_dim_len:new_time_len, ...] = data

            flush_ds.close()

        # Clear buffer
        self.buffer = []
        self.buffer_times = []

    def close(self) -> None:
        """Flush any remaining buffered states and close the writer."""
        # Flush remaining buffer
        if self.buffer:
            self.flush()

        # Add the current (last) chunk to the list
        if self.output_path.exists():
            self.chunk_files_created.append(self.output_path)

        # Report summary
        total_chunks = len(self.chunk_files_created)
        if total_chunks == 1:
            logger.success(f"States file saved: {self.chunk_files_created[0]} ({self.step_count} states written)")
            logger.debug(f"File size: {self.chunk_files_created[0].stat().st_size / 1024 / 1024:.2f} MB")
        else:
            logger.success(f"States saved in {total_chunks} chunk files ({self.step_count} states total):")
            total_size_mb = 0
            for chunk_path in self.chunk_files_created:
                size_mb = chunk_path.stat().st_size / 1024 / 1024
                total_size_mb += size_mb
                logger.info(f"  {chunk_path.name}: {size_mb:.2f} MB")
            logger.debug(f"Total size: {total_size_mb:.2f} MB")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures file is closed."""
        self.close()
        return False
