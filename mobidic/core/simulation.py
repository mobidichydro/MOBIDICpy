"""Main simulation engine

This module implements the main time-stepping loop of MOBIDIC hydrological model.
It orchestrates the water balance calculations, routing, and I/O operations.

Translated from MATLAB: mobidic_sid.m (main simulation loop)
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Any
import time
import numpy as np
import pandas as pd
from loguru import logger

from mobidic import __version__
from mobidic.config import MOBIDICConfig
from mobidic.core import constants as const
from mobidic.preprocessing.meteo_preprocessing import MeteoData
from mobidic.preprocessing.meteo_raster import MeteoRaster
from mobidic.core.soil_water_balance import soil_mass_balance
from mobidic.core.reservoir import reservoir_routing
from mobidic.core.routing import hillslope_routing, linear_channel_routing
from mobidic.core.groundwater import groundwater_linear
from mobidic.core.energy_balance import compute_energy_balance_1l, solar_hours
from mobidic.core.interpolation import precipitation_interpolation, station_interpolation
from mobidic.core.pet import calculate_pet
from mobidic.core.crop_coefficients import compute_kc_grid, load_kc_clc_mapping
from mobidic.io import StateWriter

# Energy-balance forcing variables required when energy_balance != "None"
_ENERGY_VARIABLES = ["temperature_min", "temperature_max", "humidity", "wind_speed", "radiation"]


def _create_progress_bar(current: int, total: int, bar_length: int = 20) -> str:
    """Create a text-based progress bar.

    Args:
        current: Current step number (1-indexed)
        total: Total number of steps
        bar_length: Length of the progress bar in characters (default: 20)

    Returns:
        Progress bar string like "[=========>          ]"
    """
    progress = current / total
    filled = int(bar_length * progress)

    if filled == bar_length:
        # Complete bar
        bar = "=" * bar_length
    elif filled > 0:
        # Partial bar with arrow
        bar = "=" * (filled - 1) + ">" + " " * (bar_length - filled)
    else:
        # Empty bar
        bar = " " * bar_length

    return f"[{bar}]"


class SimulationState:
    """Container for simulation state variables."""

    def __init__(
        self,
        wc: np.ndarray,
        wg: np.ndarray,
        wp: np.ndarray | None,
        ws: np.ndarray,
        discharge: np.ndarray,
        lateral_inflow: np.ndarray,
        reservoir_states: list | None = None,
        h: np.ndarray | None = None,
        ts: np.ndarray | None = None,
        td: np.ndarray | None = None,
        et: np.ndarray | None = None,
    ):
        """Initialize simulation state.

        Args:
            wc: Capillary water content [m]
            wg: Gravitational water content [m]
            wp: Plant/canopy water content [m] (None to disable)
            ws: Surface water content [m]
            discharge: River discharge for each reach [m³/s]
            lateral_inflow: Lateral inflow to each reach [m³/s]
            reservoir_states: List of ReservoirState objects (None if no reservoirs)
            h: Groundwater head [m] (None if groundwater is disabled)
            ts: Surface temperature [K] (None when energy balance is disabled)
            td: Deep-soil temperature [K] (None when energy balance is disabled)
            et: Actual evapotranspiration rate [m/s] (None when ET output is disabled)
        """
        self.wc = wc
        self.wg = wg
        self.wp = wp
        self.ws = ws
        self.discharge = discharge
        self.lateral_inflow = lateral_inflow
        self.reservoir_states = reservoir_states
        self.h = h
        self.ts = ts
        self.td = td
        self.et = et


class SimulationResults:
    """Container for simulation results."""

    def __init__(self, config: MOBIDICConfig, simulation: "Simulation | None" = None):
        """Initialize results container.

        Args:
            config: MOBIDIC configuration
            simulation: Simulation object (needed for saving states/reports)
        """
        self.config = config
        self.simulation = simulation
        self.time_series = {}  # Store time series data
        self.final_state = None  # Store final state

    def save_report(
        self,
        output_path: str | Path,
        reach_selection: str = "all",
        selected_reaches: list[int] | None = None,
        reach_file: str | Path | None = None,
        add_metadata: dict[str, Any] | None = None,
        output_format: str = "Parquet",
    ) -> None:
        """Save discharge time series to file (Parquet or CSV).

        Args:
            output_path: Path to output file
            reach_selection: "all", "file", or "list"
            selected_reaches: List of reach IDs (if reach_selection="list")
            reach_file: Path to JSON file containing reach IDs (if reach_selection="file")
            add_metadata: Additional metadata to save (optional)
            output_format: Output format: "Parquet" or "csv" (default: "Parquet")
        """
        if "discharge" not in self.time_series:
            raise ValueError("No discharge data to save. Run simulation first.")

        if self.simulation is None:
            raise ValueError("Cannot save report without simulation object")

        from mobidic.io import save_discharge_report

        save_discharge_report(
            discharge_timeseries=self.time_series["discharge"],
            time_stamps=self.time_series["time"],
            network=self.simulation.network,
            output_path=output_path,
            reach_selection=reach_selection,
            selected_reaches=selected_reaches,
            reach_file=reach_file,
            add_metadata=add_metadata,
            output_format=output_format,
        )

    def save_lateral_inflow_report(
        self,
        output_path: str | Path,
        reach_selection: str = "all",
        selected_reaches: list[int] | None = None,
        reach_file: str | Path | None = None,
        output_format: str = "Parquet",
    ) -> None:
        """Save lateral inflow time series to file (Parquet or CSV).

        Args:
            output_path: Path to output file
            reach_selection: "all", "file", or "list"
            selected_reaches: List of reach IDs (if reach_selection="list")
            reach_file: Path to JSON file containing reach IDs (if reach_selection="file")
            output_format: Output format: "Parquet" or "csv" (default: "Parquet")
        """
        if "lateral_inflow" not in self.time_series:
            raise ValueError("No lateral inflow data to save. Run simulation first.")

        if self.simulation is None:
            raise ValueError("Cannot save lateral inflow report without simulation object")

        from mobidic.io import save_lateral_inflow_report

        save_lateral_inflow_report(
            lateral_inflow_timeseries=self.time_series["lateral_inflow"],
            time_stamps=self.time_series["time"],
            network=self.simulation.network,
            output_path=output_path,
            reach_selection=reach_selection,
            selected_reaches=selected_reaches,
            reach_file=reach_file,
            output_format=output_format,
        )


class Simulation:
    """MOBIDIC simulation engine.

    This class orchestrates the hydrological simulation, including:
    - Loading input data (GIS, meteorology)
    - Initializing state variables
    - Running the main time-stepping loop
    - Saving results

    Examples:
        >>> from mobidic import load_config, load_gisdata, Simulation, MeteoData
        >>> config = load_config("Arno.yaml")
        >>> gisdata = load_gisdata("Arno_gisdata.nc", "Arno_network.parquet")
        >>> forcing = MeteoData.from_netcdf("Arno_meteo.nc")
        >>> sim = Simulation(gisdata, forcing, config)
        >>> results = sim.run("2020-01-01", "2020-12-31")
        >>> results.save_report("Arno_discharge.parquet")
    """

    def __init__(
        self,
        gisdata: Any,  # GISData object
        forcing: MeteoData | MeteoRaster,
        config: MOBIDICConfig,
    ):
        """Initialize simulation.

        Args:
            gisdata: Preprocessed GIS data (from load_gisdata or run_preprocessing)
            forcing: Meteorological forcing data as MeteoData (stations) or MeteoRaster (gridded)
            config: MOBIDIC configuration
        """
        self.gisdata = gisdata
        self.forcing = forcing
        self.config = config

        # Detect forcing type and set up method
        # _raster_et_source: "et" if raster has actual ET, "pet" if raster has PET, None otherwise.
        # When set, energy balance is skipped and the raster variable is used directly.
        # "et" takes priority over "pet" when both are present.
        self._raster_et_source: str | None = None
        if isinstance(forcing, MeteoRaster):
            self.forcing_mode = "raster"
            logger.info("Using raster meteorological forcing (no interpolation)")
            # Validate grid alignment
            forcing.validate_grid_alignment(gisdata.metadata)
            # Set forcing extraction method
            self._get_forcing_fn = self._get_raster_forcing
            # No interpolation weights needed
            self._interpolation_weights = None
            # Detect precomputed ET/PET: when present, skip energy balance calculations
            if "et" in forcing.variables:
                self._raster_et_source = "et"
                logger.info("Precomputed ET found in raster forcing: energy balance will be skipped")
            elif "pet" in forcing.variables:
                self._raster_et_source = "pet"
                logger.info(
                    "Precomputed PET found in raster forcing: energy balance will be skipped (Kc will be applied)"
                )
        else:  # MeteoData
            self.forcing_mode = "station"
            logger.info("Using station meteorological forcing (with spatial interpolation)")
            # Set forcing extraction method
            self._get_forcing_fn = self._interpolate_forcing
            # Interpolation weights will be precomputed later

        # Extract grid metadata
        self.nrows, self.ncols = gisdata.metadata["shape"]
        self.resolution = gisdata.metadata["resolution"]
        self.xllcorner = gisdata.metadata["xllcorner"]
        self.yllcorner = gisdata.metadata["yllcorner"]

        # Flow direction type from config
        self.flow_dir_type = config.raster_settings.flow_dir_type

        # Extract grids from gisdata
        self.dtm = gisdata.grids["dtm"]
        self.flow_dir = gisdata.grids["flow_dir"]
        self.flow_acc = gisdata.grids["flow_acc"]
        self.wc0 = gisdata.grids["Wc0"]
        self.wg0 = gisdata.grids["Wg0"]
        self.ks = gisdata.grids["ks"]
        self.alpsur = gisdata.grids["alpsur"]
        self.hillslope_reach_map = gisdata.hillslope_reach_map

        # Preprocess Wg0, Wc0, Wp0, and Ws0
        self.wc0 = self.wc0 * self.config.parameters.multipliers.Wc_factor
        self.wg0 = self.wg0 * self.config.parameters.multipliers.Wg_factor
        self.wp0 = np.zeros((self.nrows, self.ncols))
        self.ws0 = np.zeros((self.nrows, self.ncols))

        # Apply minumum limits
        self.wc0 = np.maximum(self.wc0, const.W_MIN)
        self.wg0 = np.maximum(self.wg0, const.W_MIN)

        # Transition factor between gravitational and capillary storage
        Wg_Wc_tr = self.config.parameters.multipliers.Wg_Wc_tr
        if Wg_Wc_tr >= 0:
            wtot = self.wc0 + self.wg0
            self.wg0 = np.minimum(Wg_Wc_tr * self.wg0, wtot)
            self.wc0 = wtot - self.wg0

        # Optional grids
        # These may be None if not provided: use get() to avoid KeyError
        self.kf = gisdata.grids.get("kf")
        self.gamma = gisdata.grids.get("gamma")
        self.kappa = gisdata.grids.get("kappa")
        self.beta = gisdata.grids.get("beta")
        self.alpha = gisdata.grids.get("alpha")
        # Energy balance grids (optional rasters; fallback to scalar from config)
        self.ch_raster = gisdata.grids.get("CH")
        self.alb_raster = gisdata.grids.get("Alb")

        # Corine Land Cover grid (optional). When present, FAO Kc values are
        # looked up per CLC class and per month; otherwise the scalar default
        # Kc (parameters.soil.Kc) is applied uniformly.
        self.clc = gisdata.grids.get("CLC")
        kc_map_path = self.config.parameters.soil.Kc_CLC_map
        self.kc_default = float(self.config.parameters.soil.Kc)
        if self.clc is not None:
            self.kc_clc_mapping = load_kc_clc_mapping(kc_map_path)
            logger.info(
                f"FAO Kc enabled with {len(self.kc_clc_mapping)} CLC classes "
                f"(default Kc={self.kc_default} for unmapped / missing cells)"
            )
        else:
            self.kc_clc_mapping = None
            if kc_map_path is not None:
                logger.warning(
                    "parameters.soil.Kc_CLC_map is set but raster_files.CLC is not: "
                    "Kc will default uniformly to parameters.soil.Kc"
                )
        # Cache the most recently computed Kc grid to avoid rebuilding it when
        # the month does not change between timesteps.
        self._kc_grid_cache: tuple[int, np.ndarray] | None = None

        # Freatic aquifer mask (optional). When multiple positive classes are
        # present, enables multi-aquifer mode with per-class h averaging
        self.mf = gisdata.grids.get("Mf")
        self.aquifer_ids: np.ndarray | None = None
        if self.mf is not None:
            mf_arr = np.asarray(self.mf, dtype=float)
            valid = np.isfinite(mf_arr) & (mf_arr > 0)
            unique_ids = np.unique(mf_arr[valid])
            if unique_ids.size > 1:
                self.aquifer_ids = unique_ids
                logger.info(
                    f"Multi-aquifer mode: {unique_ids.size} classes detected in Mf "
                    f"(ids={unique_ids.tolist()}); groundwater head will be averaged within each class"
                )

        # River network
        self.network = gisdata.network

        # Reservoirs (optional)
        self.reservoirs = getattr(gisdata, "reservoirs", None)
        if self.reservoirs is not None:
            logger.info(f"Reservoirs loaded: {len(self.reservoirs)} reservoirs")
        else:
            logger.debug("No reservoirs in gisdata")

        # Time step
        self.dt = config.simulation.timestep

        # Prepare parameter grids
        self.param_grids = self._prepare_grids()

        # Preprocess and cache network topology for fast routing
        self._network_topology = self._preprocess_network_topology()

        # Pre-compute interpolation weights for meteorological forcing (station mode only).
        # Always include precipitation; add the energy variables when energy balance is active.
        if self.forcing_mode == "station":
            forcing_vars = ["precipitation"]
            if self.config.simulation.energy_balance != "None":
                forcing_vars.extend(_ENERGY_VARIABLES)
            self._interpolation_weights = self._precompute_interpolation_weights(forcing_vars)
        # else: self._interpolation_weights already set to None above

        # Time indices cache (will be populated in run() when simulation period is known)
        self._time_indices_cache = None

        # Initialize state
        self.state = None
        # True when the state was loaded from a file (i.e. set via set_initial_state).
        # When False at run() time, energy balance variables Ts/Td are overridden
        # with Tair_lin at the first timestep.
        self._state_was_loaded = False

        logger.info(
            f"Simulation initialized: grid={self.nrows}x{self.ncols}, "
            f"dt={self.dt}s, network={len(self.network)} reaches"
        )

    @property
    def _kc_is_active(self) -> bool:
        """True when Kc differs from 1.0 (CLC mapping present or scalar Kc != 1)."""
        return self.clc is not None or self.kc_default != 1.0

    def _initial_state(self) -> SimulationState:
        """Initialize the initial simulation state variables with initial conditions from configuration.

        Returns:
            Initial simulation state
        """
        logger.info("Initial simulation state")

        # Initialize soil water content from configuration
        wcsat = self.config.initial_conditions.Wcsat  # Fraction of capillary saturation
        wgsat = self.config.initial_conditions.Wgsat  # Fraction of gravitational saturation
        ws_init = self.config.initial_conditions.Ws  # Initial surface water depth [m]

        # Initial values
        wc = self.wc0.copy() * wcsat
        wg = self.wg0.copy() * wgsat
        ws = np.full((self.nrows, self.ncols), ws_init)
        wp = self.wp0.copy()  # Initial plant/canopy water content [m]

        # Set NaN outside domain (lines 298-300 in mobidic_sid.m)
        wc[np.isnan(self.flow_acc)] = np.nan
        wg[np.isnan(self.flow_acc)] = np.nan
        ws[np.isnan(self.flow_acc)] = np.nan
        wp[np.isnan(self.flow_acc)] = np.nan

        # Initialize river discharge and lateral inflow
        discharge = np.zeros(len(self.network))
        lateral_inflow = np.zeros(len(self.network))

        # Initialize groundwater head when Linear groundwater model is active
        h = None
        if self.config.parameters.groundwater.model == "Linear":
            h_init = self.config.initial_conditions.groundwater_head
            h = np.where(np.isfinite(self.dtm), h_init, np.nan)

        # Initialize surface and deep-soil temperatures when energy balance is active.
        ts = None
        td = None
        if self.config.simulation.energy_balance != "None":
            tcost = self.config.parameters.energy.Tconst
            ts = np.where(np.isfinite(self.dtm), tcost, np.nan)
            td = np.where(np.isfinite(self.dtm), tcost, np.nan)

        # Initialize reservoir states if reservoirs exist
        reservoir_states = None
        if self.reservoirs is not None:
            from mobidic.core.reservoir import ReservoirState

            reservoir_states = []
            for reservoir in self.reservoirs.reservoirs:
                # Initialize with configured initial volume
                state = ReservoirState(
                    volume=reservoir.initial_volume,
                    stage=0.0,  # Will be calculated in first timestep
                    inflow=0.0,
                    outflow=0.0,
                )
                reservoir_states.append(state)
            logger.info(f"Initialized {len(reservoir_states)} reservoir states")

        log_msg = (
            f"State initialized. Initial conditions (average): "
            f"Wc={np.nanmean(wc) * 1000:.1f} mm, "
            f"Wg={np.nanmean(wg) * 1000:.1f} mm, "
            f"Ws={np.nanmean(ws) * 1000:.1f} mm, "
            f"Wp={np.nanmean(wp) * 1000:.1f} mm"
        )
        if h is not None:
            log_msg += f", h={np.nanmean(h) * 1000:.1f} mm"
        logger.success(log_msg)

        return SimulationState(wc, wg, wp, ws, discharge, lateral_inflow, reservoir_states, h=h, ts=ts, td=td)

    def set_initial_state(
        self,
        state: SimulationState | None = None,
        state_file: str | Path | None = None,
        time_index: int = -1,
    ) -> None:
        """Set the initial simulation state from a previous simulation.

        This method allows restarting a simulation from a previously saved state,
        enabling warm starts, multi-stage simulations, or continuation after interruption.

        Args:
            state: SimulationState object to use as initial state. If provided, state_file is ignored.
            state_file: Path to NetCDF state file to load. Used if state is None.
            time_index: Time index to load from state file (default: -1 for last timestep).
                Only used when loading from state_file.

        Raises:
            ValueError: If neither state nor state_file is provided, or if state_file doesn't exist.

        Examples:
            >>> # Method 1: Set state directly from SimulationState object
            >>> sim.set_initial_state(state=previous_state)
            >>>
            >>> # Method 2: Load from state file (last timestep)
            >>> sim.set_initial_state(state_file="states.nc")
            >>>
            >>> # Method 3: Load specific timestep from state file
            >>> sim.set_initial_state(state_file="states.nc", time_index=10)
        """
        if state is not None:
            # Use provided state directly
            logger.info("Setting initial state from provided SimulationState object")
            self.state = state
            self._state_was_loaded = True
            logger.success(
                f"Initial state set. State contains: "
                f"Wc={np.nanmean(state.wc) * 1000:.1f} mm, "
                f"Wg={np.nanmean(state.wg) * 1000:.1f} mm, "
                f"Ws={np.nanmean(state.ws) * 1000:.1f} mm, "
                f"Q_mean={np.mean(state.discharge):.3f} m³/s"
            )
        elif state_file is not None:
            # Load from state file
            state_path = Path(state_file)
            if not state_path.exists():
                raise ValueError(f"State file not found: {state_path}")

            logger.info(f"Loading initial state from file: {state_path}")

            from mobidic.io import load_state

            loaded_state, state_time, metadata = load_state(
                input_path=state_path,
                network_size=len(self.network),
                time_index=time_index,
                config=self.config,
                gisdata=self.gisdata,
            )

            self.state = loaded_state
            self._state_was_loaded = True
            logger.success(
                f"Initial state loaded from {state_path} at time {state_time}. "
                f"State contains: "
                f"Wc={np.nanmean(loaded_state.wc) * 1000:.1f} mm, "
                f"Wg={np.nanmean(loaded_state.wg) * 1000:.1f} mm, "
                f"Ws={np.nanmean(loaded_state.ws) * 1000:.1f} mm, "
                f"Q_mean={np.mean(loaded_state.discharge):.3f} m³/s"
            )
        else:
            raise ValueError("Either 'state' or 'state_file' must be provided")

    def _interpolate_forcing(
        self,
        time: datetime,
        variable: str,
        weights_cache: dict[str, np.ndarray | None] | None = None,
        time_step_index: int | None = None,
    ) -> np.ndarray:
        """Interpolate meteorological forcing data to grid.

        Args:
            time: Current simulation time
            variable: Variable name ('precipitation', 'temperature_min', 'temperature_max', etc.)
            weights_cache: Optional pre-computed interpolation weights for performance.
                If None, weights will be computed on-the-fly.
            time_step_index: Optional time step index for accessing cached time indices.
                If None, time indices will be computed on-the-fly.

        Returns:
            2D grid of interpolated values
        """
        # Get station data for this variable
        if variable not in self.forcing.stations:
            raise ValueError(f"Variable '{variable}' not found in forcing data")

        var_stations = self.forcing.stations[variable]
        n_stations = len(var_stations)

        if n_stations == 0:
            raise ValueError(f"No stations found for variable '{variable}'")

        # Pre-allocate arrays (faster than list append)
        station_x = np.zeros(n_stations, dtype=np.float64)
        station_y = np.zeros(n_stations, dtype=np.float64)
        station_elevation = np.zeros(n_stations, dtype=np.float64)
        station_values = np.zeros(n_stations, dtype=np.float64)
        valid_mask = np.ones(n_stations, dtype=bool)

        # Use cached time indices if available
        if (
            self._time_indices_cache is not None
            and time_step_index is not None
            and variable in self._time_indices_cache
        ):
            time_indices_var = self._time_indices_cache[variable]

            if time_indices_var is not None:
                # Fast path: use pre-computed indices
                indices = time_indices_var[:, time_step_index]

                for i, station in enumerate(var_stations):
                    if len(station["time"]) == 0:
                        valid_mask[i] = False
                        continue

                    idx = indices[i]
                    station_x[i] = station["x"]
                    station_y[i] = station["y"]
                    station_elevation[i] = station["elevation"]
                    station_values[i] = station["data"][idx]
            else:
                # Fallback for variables with no cached indices (shouldn't happen)
                valid_mask[:] = False
        else:
            # Fallback: compute time indices on-the-fly (slower, used when cache is not available)
            target_time = pd.Timestamp(time)

            for i, station in enumerate(var_stations):
                if len(station["time"]) == 0:
                    valid_mask[i] = False
                    continue

                time_index = station["time"].get_indexer([target_time], method="nearest")[0]

                if time_index >= 0 and time_index < len(station["data"]):
                    station_x[i] = station["x"]
                    station_y[i] = station["y"]
                    station_elevation[i] = station["elevation"]
                    station_values[i] = station["data"][time_index]
                else:
                    valid_mask[i] = False

        # Filter out invalid stations
        if not valid_mask.any():
            raise ValueError(f"No valid data found for {variable} at time {time}")

        station_x = station_x[valid_mask]
        station_y = station_y[valid_mask]
        station_elevation = station_elevation[valid_mask]
        station_values = station_values[valid_mask]

        # Use single resolution value (assume square cells)
        resolution = self.resolution[0]

        # Choose interpolation method based on variable
        if variable == "precipitation":
            # Check if there's any rainfall before interpolating (MATLAB line 1241: if nansum(pp))
            # This optimization skips interpolation when all precipitation values are zero or NaN
            if np.nansum(station_values) == 0 or np.all(np.isnan(station_values)):
                # Return zero grid (matching MATLAB line 1256: p_i = Mzeros)
                grid_values = np.zeros_like(self.dtm)
            else:
                # Get precipitation interpolation method from config
                precip_interp = self.config.simulation.precipitation_interp

                if precip_interp == "Nearest":
                    # Use nearest neighbor for precipitation (MATLAB: pluviomap.m when thiessen=1)
                    grid_values = precipitation_interpolation(
                        station_x,
                        station_y,
                        station_values,
                        self.dtm,
                        self.xllcorner,
                        self.yllcorner,
                        resolution,
                    )
                else:  # "IDW"
                    # Use IDW without elevation correction for precipitation
                    # MATLAB mobidic_sid.m line 1246: uses tmww3 with expon=2
                    # tmww3 = tmww^3 where tmww = 1/dist^2, so tmww3 = 1/dist^6
                    # This gives stronger weight to nearby stations (power=6)
                    # Use cached weights if available
                    weights_matrix = weights_cache.get(variable) if weights_cache else None
                    grid_values = station_interpolation(
                        station_x,
                        station_y,
                        station_elevation,
                        station_values,
                        self.dtm,
                        self.xllcorner,
                        self.yllcorner,
                        resolution,
                        weights_matrix=weights_matrix,
                        apply_elevation_correction=False,  # switchregz=0 in MATLAB
                        power=6.0,  # tmww3 in MATLAB: (1/dist^2)^3 = 1/dist^6
                    )

                # Convert from mm/h to m/s
                # Station data is stored in mm/h (same as raster data)
                grid_values = grid_values / 1000.0 / 3600.0
        else:
            # Use IDW with per-variable settings matching MATLAB calc_forcing_day.m:
            # - Temperature (min/max): switchregz=1, expon=2 (elevation correction, power=2)
            # - Humidity:              switchregz=0, expon=2
            # - Wind speed:            switchregz=0, expon=0.5
            # - Radiation:             switchregz=0, expon=2
            if variable in ("temperature_min", "temperature_max"):
                apply_elevation = True
                power = 2.0
            elif variable == "wind_speed":
                apply_elevation = False
                power = 0.5
            else:
                apply_elevation = False
                power = 2.0

            weights_matrix = weights_cache.get(variable) if weights_cache else None
            grid_values = station_interpolation(
                station_x,
                station_y,
                station_elevation,
                station_values,
                self.dtm,
                self.xllcorner,
                self.yllcorner,
                resolution,
                weights_matrix=weights_matrix,
                apply_elevation_correction=apply_elevation,
                power=power,
            )

        return grid_values

    def _get_raster_forcing(
        self,
        time: datetime,
        variable: str,
        weights_cache: dict[str, np.ndarray | None] | None = None,
        time_step_index: int | None = None,
    ) -> np.ndarray:
        """Extract meteorological forcing from raster data.

        Args:
            time: Current simulation time
            variable: Variable name ('precipitation', 'temperature', etc.)
            weights_cache: Unused (for signature compatibility with _interpolate_forcing)
            time_step_index: Unused (for signature compatibility with _interpolate_forcing)

        Returns:
            2D grid in simulation units (m/s for precip, degC for temperature)
        """
        # Get grid from raster (returns values in file units: mm/h for precip, degC for temp)
        grid = self.forcing.get_timestep(time, variable)

        # Convert units to match _interpolate_forcing output
        if variable in ("precipitation", "pet", "et"):
            # Convert mm/h to m/s: divide by 1000 * 3600
            grid = grid / 1000.0 / 3600.0

        # Temperature and other variables are already in correct units (degC)
        return grid

    def _calculate_pet(self, time: datetime) -> np.ndarray:
        """Calculate potential evapotranspiration.

        Currently energy balance is not implemented -> uses MATLAB approach: constant 1 mm/day.
        MATLAB: etp = Mones/(1000*3600*24) [mobidic_sid.m line 332]

        Args:
            time: Current simulation time

        Returns:
            PET grid [m] over time step
        """
        # Use MATLAB approach: constant 1 mm/day when no energy balance
        pet = calculate_pet((self.nrows, self.ncols), self.dt, pet_rate_mm_day=1.0)

        return pet

    def _get_kc(self, time: datetime) -> np.ndarray | float:
        """Return the FAO Kc factor for the current timestep.

        Uses the CLC raster and the monthly Kc mapping when available,
        otherwise returns the scalar default Kc. The result is cached per
        month since the mapping only changes at month boundaries.
        """
        if self.clc is None or self.kc_clc_mapping is None:
            return self.kc_default

        month = time.month
        if self._kc_grid_cache is not None and self._kc_grid_cache[0] == month:
            return self._kc_grid_cache[1]

        kc = compute_kc_grid(self.clc, self.kc_clc_mapping, month, self.kc_default)
        self._kc_grid_cache = (month, kc)
        return kc

    def _prepare_grids(self) -> dict[str, np.ndarray]:
        """Prepare grids for simulation.

        Returns:
            Dictionary of parameter grids
        """
        # Get parameters from config
        params = self.config.parameters

        # Create grids from scalar parameters or raster files
        # If raster exists, use it; otherwise use scalar value
        param_grids = {}

        # Rainfall fraction f0: fraction of time step without rain [-]
        # Time-dependent parameter from mobidic_sid.m line 213
        f0_value = const.F0_CONSTANT * (
            1 - np.exp(-self.dt / (24 * 3600) * np.log(const.F0_CONSTANT / (const.F0_CONSTANT - 0.75)))
        )
        param_grids["f0"] = np.full((self.nrows, self.ncols), f0_value)
        param_grids["f0"][np.isnan(self.dtm)] = np.nan

        # Hydraulic conductivity [m/s]
        ks_factor = self.config.parameters.multipliers.ks_factor
        if self.ks is not None:
            param_grids["ks"] = self.ks * ks_factor
        else:
            param_grids["ks"] = np.full((self.nrows, self.ncols), params.soil.ks) * ks_factor
        param_grids["ks"] = np.maximum(param_grids["ks"], const.FLUX_MIN)
        param_grids["ks"][np.isnan(self.dtm)] = np.nan

        # Aquifer conductivity kf [1/s] (only used when groundwater is active)
        if self.kf is not None:
            param_grids["kf"] = self.kf
        else:
            param_grids["kf"] = np.full((self.nrows, self.ncols), params.soil.kf)

        # Flow coefficients
        if self.gamma is not None:
            param_grids["gamma"] = self.gamma
        else:
            param_grids["gamma"] = np.full((self.nrows, self.ncols), params.soil.gamma)

        if self.kappa is not None:
            param_grids["kappa"] = self.kappa
        else:
            param_grids["kappa"] = np.full((self.nrows, self.ncols), params.soil.kappa)

        if self.beta is not None:
            param_grids["beta"] = self.beta
        else:
            param_grids["beta"] = np.full((self.nrows, self.ncols), params.soil.beta)

        if self.alpha is not None:
            param_grids["alpha"] = self.alpha
        else:
            param_grids["alpha"] = np.full((self.nrows, self.ncols), params.soil.alpha)

        # Channelized flow fraction cha [-]
        # From buildgis_mysql_include.m line 655-656, then mobidic_sid.m line 775: cha^chafac
        param_grids["cha"] = self.flow_acc / np.nanmax(self.flow_acc)
        param_grids["cha"] = param_grids["cha"] ** self.config.parameters.multipliers.chan_factor

        # Surface alpha parameter alpsur
        param_grids["alpsur"] = self.alpsur * param_grids["alpha"]

        # Energy balance grids (only built when energy balance is active)
        if self.config.simulation.energy_balance != "None":
            ch_factor = self.config.parameters.multipliers.CH_factor
            if self.ch_raster is not None:
                param_grids["CH"] = self.ch_raster * ch_factor
            else:
                param_grids["CH"] = np.full((self.nrows, self.ncols), params.energy.CH) * ch_factor
            param_grids["CH"][np.isnan(self.dtm)] = np.nan

            if self.alb_raster is not None:
                param_grids["Alb"] = self.alb_raster.copy()
            else:
                param_grids["Alb"] = np.full((self.nrows, self.ncols), params.energy.Alb)
            param_grids["Alb"][np.isnan(self.dtm)] = np.nan

        return param_grids

    def _precompute_interpolation_weights(self, variables: list[str] | None = None) -> dict[str, np.ndarray | None]:
        """Pre-compute IDW interpolation weights for each meteorological variable.

        Weights depend only on station geometry, not on values, so they can be
        computed once and reused for all timesteps. This provides significant
        performance improvement for long simulations.

        Args:
            variables: List of variable names to compute weights for.
                If None, computes weights for all variables in forcing data.
                Examples: ["precipitation"], ["precipitation", "temperature_min"]

        Returns:
            Dictionary mapping variable names to weight matrices (3D arrays).
            For variables using nearest neighbor (precipitation with "nearest" method),
            the value is None since weights aren't needed.
        """
        from mobidic.core.interpolation import compute_idw_weights

        # Determine which variables to compute weights for
        if variables is None:
            # Default: compute weights for all variables
            variables_to_process = list(self.forcing.stations.keys())
            logger.debug("Pre-computing interpolation weights for all meteorological variables")
        else:
            # Validate that requested variables exist in forcing data
            available_vars = set(self.forcing.stations.keys())
            invalid_vars = [v for v in variables if v not in available_vars]
            if invalid_vars:
                raise ValueError(
                    f"Variables {invalid_vars} not found in forcing data. Available variables: {sorted(available_vars)}"
                )
            variables_to_process = variables
            logger.debug(f"Pre-computing interpolation weights for selected variables: {variables_to_process}")

        weights_cache = {}
        resolution = self.resolution[0]  # Use single resolution value

        for variable in variables_to_process:
            var_stations = self.forcing.stations[variable]

            if len(var_stations) == 0:
                logger.debug(f"No stations for {variable}, skipping weight computation")
                weights_cache[variable] = None
                continue

            # Get station coordinates (assuming they don't change over time)
            station_x = np.array([s["x"] for s in var_stations])
            station_y = np.array([s["y"] for s in var_stations])

            # Determine interpolation settings based on variable type
            if variable == "precipitation":
                precip_interp = self.config.simulation.precipitation_interp
                if precip_interp == "Nearest":
                    # Nearest neighbor doesn't use weights matrix
                    logger.debug(f"{variable}: using nearest neighbor (no weights needed)")
                    weights_cache[variable] = None
                else:  # "IDW"
                    # IDW with power=6 for precipitation
                    logger.debug(f"{variable}: pre-computing IDW weights (power=6)")
                    weights_cache[variable] = compute_idw_weights(
                        station_x,
                        station_y,
                        self.dtm,
                        self.xllcorner,
                        self.yllcorner,
                        resolution,
                        power=6.0,
                    )
            elif variable in ["temperature_min", "temperature_max"]:
                # Temperature: power=2, elevation correction applied to values (not weights)
                logger.debug(f"{variable}: pre-computing IDW weights (power=2)")
                weights_cache[variable] = compute_idw_weights(
                    station_x,
                    station_y,
                    self.dtm,
                    self.xllcorner,
                    self.yllcorner,
                    resolution,
                    power=2.0,
                )
            elif variable == "wind_speed":
                # Wind: power=0.5
                logger.debug(f"{variable}: pre-computing IDW weights (power=0.5)")
                weights_cache[variable] = compute_idw_weights(
                    station_x,
                    station_y,
                    self.dtm,
                    self.xllcorner,
                    self.yllcorner,
                    resolution,
                    power=0.5,
                )
            else:
                # Other variables (humidity, radiation): power=2, no elevation correction
                logger.debug(f"{variable}: pre-computing IDW weights (power=2)")
                weights_cache[variable] = compute_idw_weights(
                    station_x,
                    station_y,
                    self.dtm,
                    self.xllcorner,
                    self.yllcorner,
                    resolution,
                    power=2.0,
                )

        logger.success(f"Interpolation weights cached for {len(weights_cache)} variables")
        return weights_cache

    def _precompute_time_indices(
        self, simulation_times: pd.DatetimeIndex, variables: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Pre-compute time indices for all stations and all simulation timesteps.

        This method pre-computes the mapping from simulation timesteps to station data
        indices, avoiding repeated time lookups during the main simulation loop.
        This provides significant performance improvement for long simulations.

        Args:
            simulation_times: DatetimeIndex of all simulation timesteps
            variables: List of variable names to compute time indices for.
                If None, computes indices for all variables in forcing data.
                Examples: ["precipitation"], ["precipitation", "temperature_min"]

        Returns:
            Dictionary mapping variable names to (n_stations × n_timesteps) index arrays.
            For each variable and timestep, the array contains the index into the
            station's data array that is nearest to that timestep.
        """
        # Determine which variables to compute indices for
        if variables is None:
            # Default: compute indices for all variables
            variables_to_process = list(self.forcing.stations.keys())
            logger.info(f"Pre-computing time indices for {len(simulation_times)} timesteps (all variables)")
        else:
            # Validate that requested variables exist in forcing data
            available_vars = set(self.forcing.stations.keys())
            invalid_vars = [v for v in variables if v not in available_vars]
            if invalid_vars:
                raise ValueError(
                    f"Variables {invalid_vars} not found in forcing data. Available variables: {sorted(available_vars)}"
                )
            variables_to_process = variables
            logger.info(
                f"Pre-computing time indices for {len(simulation_times)} timesteps "
                f"({len(variables_to_process)} variables: {variables_to_process})"
            )

        time_indices = {}
        n_times = len(simulation_times)

        for variable in variables_to_process:
            var_stations = self.forcing.stations[variable]
            n_stations = len(var_stations)

            if n_stations == 0:
                logger.debug(f"No stations for {variable}, skipping time index computation")
                time_indices[variable] = None
                continue

            # Pre-allocate index array (n_stations × n_timesteps)
            indices = np.zeros((n_stations, n_times), dtype=np.int32)

            for i, station in enumerate(var_stations):
                if len(station["time"]) == 0:
                    # Station has no data - indices will be 0 but won't be used
                    continue

                # Convert station times to numpy array for searchsorted
                station_times = station["time"].values.astype("datetime64[ns]")
                sim_times = simulation_times.values.astype("datetime64[ns]")

                # Find insertion points (left side)
                left_idx = np.searchsorted(station_times, sim_times, side="left")

                # For each simulation time, find the nearest station time
                # We need to check both left_idx and left_idx-1 (if valid)
                max_idx = len(station_times) - 1

                for t in range(n_times):
                    left = left_idx[t]

                    # Clamp to valid range
                    if left > max_idx:
                        # Past the end - use last index
                        indices[i, t] = max_idx
                    elif left == 0:
                        # Before the start - use first index
                        indices[i, t] = 0
                    else:
                        # Check both neighbors and pick nearest
                        right = left
                        left_candidate = left - 1

                        left_dist = abs((sim_times[t] - station_times[left_candidate]).astype("timedelta64[ns]"))
                        right_dist = abs((sim_times[t] - station_times[right]).astype("timedelta64[ns]"))

                        if left_dist <= right_dist:
                            indices[i, t] = left_candidate
                        else:
                            indices[i, t] = right

            time_indices[variable] = indices
            logger.debug(f"Pre-computed {n_stations} x {n_times} time indices for {variable}")

        logger.debug(f"Time indices cached for {len(time_indices)} variables")
        return time_indices

    def _preprocess_network_topology(self) -> dict:
        """Preprocess network topology for fast routing.

        This method pre-extracts network topology to numpy arrays and caches them
        for use in every time step. This avoids repeated pandas operations during
        the main simulation loop.

        Returns:
            Dictionary containing pre-processed network topology:
                - 'upstream_1_idx': numpy array of first upstream indices
                - 'upstream_2_idx': numpy array of second upstream indices
                - 'n_upstream': numpy array of upstream counts
                - 'sorted_reach_idx': numpy array of reach indices sorted by calc_order
                - 'K': numpy array of storage coefficients (lag_time_s)
                - 'n_reaches': number of reaches
        """
        logger.debug("Preprocessing network topology for fast routing")

        network = self.network
        n_reaches = len(network)

        # Create mapping from mobidic_id to DataFrame index
        mobidic_id_to_idx = {int(network.at[idx, "mobidic_id"]): idx for idx in network.index}

        # Pre-extract topology to numpy arrays
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

        # Get sorted reach indices by calc_order
        sorted_reach_idx = network.sort_values("calc_order").index.values.astype(np.int32)

        # Extract K (lag time as storage coefficient)
        K = network["lag_time_s"].values

        topology = {
            "upstream_1_idx": upstream_1_idx,
            "upstream_2_idx": upstream_2_idx,
            "n_upstream": n_upstream,
            "sorted_reach_idx": sorted_reach_idx,
            "K": K,
            "n_reaches": n_reaches,
        }

        logger.success(f"Network topology preprocessed: {n_reaches} reaches cached")

        return topology

    def _accumulate_lateral_inflow(self, lateral_flow: np.ndarray) -> np.ndarray:
        """Accumulate lateral flow contributions to each reach.

        Following MATLAB's approach (mobidic_sid.m line 224):
        - Only accumulate from cells where ch > 0 (valid reach IDs)
        - Skip NaN cells (outside domain)
        - Skip -9999 cells (inside domain but cannot reach river network)

        Vectorized implementation using np.bincount for performance.

        Args:
            lateral_flow: 2D grid of lateral flow [m³/s]

        Returns:
            1D array of lateral inflow to each reach [m³/s]
        """
        n_reaches = len(self.network)

        # Flatten lateral flow and hillslope-reach mapping
        lateral_flow_flat = lateral_flow.ravel("F")
        hillslope_map_flat = self.hillslope_reach_map.ravel("F")

        # Create mask for valid cells: not NaN, >= 0, and finite lateral flow
        # MATLAB: ko = find(isfinite(zz) & (ch>0)); % contributing pixels
        valid_mask = np.isfinite(hillslope_map_flat) & (hillslope_map_flat >= 0) & np.isfinite(lateral_flow_flat)

        # Extract valid reach IDs and flow values
        valid_reach_ids = hillslope_map_flat[valid_mask].astype(np.int32)
        valid_flows = lateral_flow_flat[valid_mask]

        # Accumulate using bincount (vectorized, very fast)
        lateral_inflow = np.bincount(
            valid_reach_ids,
            weights=valid_flows,
            minlength=n_reaches,
        )

        return lateral_inflow

    def _should_save_state(self, step: int, current_time: datetime) -> bool:
        """Determine if state should be saved at current timestep.

        Args:
            step: Current timestep index (0-indexed)
            current_time: Current simulation datetime

        Returns:
            True if state should be saved, False otherwise
        """
        state_settings = self.config.output_states_settings

        if state_settings.output_states == "final":
            # Only save final state (handled after loop)
            return False
        elif state_settings.output_states == "all":
            # Save at intervals specified by output_interval
            if state_settings.output_interval is None:
                # No interval specified, save every timestep
                return True
            else:
                # Check if enough time has elapsed since start
                # We save at multiples of output_interval
                interval_steps = int(state_settings.output_interval / self.dt)
                return (step + 1) % interval_steps == 0
        elif state_settings.output_states == "list":
            # Save at specific datetimes specified in output_list
            # Convert datetime strings to datetime objects and check if current_time matches
            for date_str in state_settings.output_list:
                target_time = datetime.fromisoformat(date_str.replace(" ", "T"))
                if current_time == target_time:
                    return True
            return False
        else:
            return False

    def run(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        save_states_interval: int | None = None,
        save_report_interval: int | None = None,
    ) -> SimulationResults:
        """Run simulation.

        Args:
            start_date: Simulation start date (YYYY-MM-DD or datetime)
            end_date: Simulation end date (YYYY-MM-DD or datetime)
            save_states_interval: Interval for saving states [s]. If None, use config value.
            save_report_interval: Interval for saving reports [s]. If None, use config value.

        Returns:
            SimulationResults object containing time series and final state
        """

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"MOBIDICpy v{__version__} - SIMULATION")
        logger.info("=" * 80)
        logger.info(f"Basin: {self.config.basin.id}")
        logger.info(f"Parameter set: {self.config.basin.paramset_id}")
        logger.info("")

        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        # Initialize state (skip if already set via set_initial_state())
        if self.state is None:
            self.state = self._initial_state()
        else:
            logger.info("Using pre-set initial state (skipping default initialization)")

        # Initialize results container
        results = SimulationResults(self.config, simulation=self)
        discharge_ts = []
        lateral_inflow_ts = []
        time_ts = []

        # Calculate number of time steps (inclusive of end_date)
        n_steps = int((end_date - start_date).total_seconds() / self.dt) + 1
        logger.info(f"Number of time steps: {n_steps}")

        # Pre-compute time indices for all simulation timesteps (station mode only).
        # Include energy variables when the energy balance is active.
        simulation_times = pd.date_range(start=start_date, periods=n_steps, freq=f"{self.dt}s")
        if self.forcing_mode == "station":
            forcing_vars = ["precipitation"]
            if self.config.simulation.energy_balance != "None":
                forcing_vars.extend(_ENERGY_VARIABLES)
            self._time_indices_cache = self._precompute_time_indices(simulation_times, variables=forcing_vars)
        else:
            self._time_indices_cache = None

        # Validate output_list if using list-based state output
        state_settings = self.config.output_states_settings
        if state_settings.output_states == "list" and state_settings.output_list:
            logger.debug("Validating output_list against simulation time steps")
            missing_states = []
            for date_str in state_settings.output_list:
                try:
                    target_time = datetime.fromisoformat(date_str.replace(" ", "T"))
                    # Check if target_time exists in simulation_times
                    if target_time not in simulation_times:
                        missing_states.append(date_str)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid datetime format in output_list: '{date_str}' - {e}")
                    missing_states.append(date_str)

            if missing_states:
                logger.warning(
                    f"The following {len(missing_states)} state(s) from output_list "
                    f"will NOT be saved (not found among simulation time steps):"
                )
                for missing_state in missing_states:
                    logger.warning(f"  - {missing_state}")
                logger.warning(
                    f"Simulation runs from {start_date.isoformat()} to {end_date.isoformat()} with timestep={self.dt}s"
                )

        # Get total soil capacity
        wtot0 = self.wc0 + self.wg0

        # Cell area for unit conversion [m²]
        cell_area = self.resolution[0] * self.resolution[1]

        # Create ko mask: contributing pixels only (matching MATLAB mobidic_sid.m:224)
        # ko = find(isfinite(zz) & (ch>0));  % contributing pixels
        ko = np.isfinite(self.dtm) & (self.hillslope_reach_map >= 0)
        ko = np.where(ko.ravel("F"))[0]
        logger.debug(f"Contributing pixels (ko): {len(ko)} of {self.nrows * self.ncols} total cells")

        # Initialize flow variables for hillslope routing feedback (matching MATLAB mobidic_sid.m:1621-1625)
        # flr_prev: surface runoff from previous timestep [m/s] - will be routed to create pir
        # fld_prev: lateral flow from previous timestep [m/s] - will be routed to create pid
        flr_prev = np.zeros((self.nrows, self.ncols))
        fld_prev = np.zeros((self.nrows, self.ncols))

        # Energy balance preparation.
        # Skip energy balance when ET or PET is provided by the raster forcing.
        # simulation.energy_balance has no effect when _raster_et_source is set.
        energy_active = self.config.simulation.energy_balance == "1L" and self._raster_et_source is None
        # When energy balance is active with Kc, the output PET field is ETc → use "et" variable name.
        et_meteo_var = "et" if (energy_active and self._kc_is_active) else "pet"
        solar_cache: dict[int, tuple[float, float]] = {}
        td_rise_full = None  # Td evaluated at sunrise (per cell), updated by pre-pass each step
        if energy_active:
            lat = self.config.basin.baricenter.lat
            lon = self.config.basin.baricenter.lon
            kaps = self.config.parameters.energy.kaps
            nis = self.config.parameters.energy.nis
            tcost = self.config.parameters.energy.Tconst
            pair = const.P_AIR
            ch_flat_full = self.param_grids["CH"].ravel("F")[ko]
            alb_flat_full = self.param_grids["Alb"].ravel("F")[ko]
            td_rise_full = self.state.td.copy() if self.state.td is not None else None
            logger.info("Energy balance scheme: 1L")

        # Calculate progress logging interval: either 20 steps total or every 30 seconds
        # Use whichever results in fewer log messages
        steps_per_intervals = max(1, n_steps // 20)
        progress_step_interval = steps_per_intervals
        progress_time_interval = 30.0  # seconds

        # Prepare states directory and initialize state writer (if enabled)
        state_settings = self.config.output_states_settings
        state_writer = None

        # Only instantiate StateWriter if state output is enabled
        if state_settings.output_states not in [None, "None"]:
            states_dir = Path(self.config.paths.states)
            states_dir.mkdir(parents=True, exist_ok=True)

            reservoir_size = len(self.reservoirs) if self.reservoirs is not None else 0
            state_writer = StateWriter(
                output_path=states_dir / "states.nc",
                grid_metadata=self.gisdata.metadata,
                network_size=len(self.network),
                output_states=self.config.output_states,
                flushing=state_settings.flushing,
                max_file_size=state_settings.max_file_size,
                add_metadata={
                    "basin_id": self.config.basin.id,
                    "paramset_id": self.config.basin.paramset_id,
                },
                reservoir_size=reservoir_size,
            )
            logger.info(f"State output enabled: {state_settings.output_states}")
        else:
            logger.info("State output disabled (output_states=None)")

        # Initialize meteo writer (if enabled)
        meteo_writer = None
        if self.config.output_forcing_data.meteo_data:
            output_dir = Path(self.config.paths.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            from mobidic.io import MeteoWriter

            # Define which variables to save (precipitation + energy variables when active).
            # Skip energy variables when the energy balance is bypassed (e.g. PET from raster).
            meteo_variables = ["precipitation"]
            if energy_active:
                meteo_variables.extend(_ENERGY_VARIABLES)
                meteo_variables.append(et_meteo_var)

            meteo_writer = MeteoWriter(
                output_path=output_dir / "meteo_forcing.nc",
                grid_metadata=self.gisdata.metadata,
                variables=meteo_variables,
                add_metadata={
                    "basin_id": self.config.basin.id,
                    "paramset_id": self.config.basin.paramset_id,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )
            logger.info(f"Meteo data output enabled: {meteo_variables}")
        else:
            logger.info("Interpolated meteo data output disabled")

        # Main time loop
        logger.info("Starting simulation main loop")
        current_time = start_date
        simulation_start_time = time.time()
        last_log_time = simulation_start_time
        for step in range(n_steps):
            logger.debug(f"Time step {step + 1}/{n_steps}: {current_time}")

            # 1. Get meteorological forcing (interpolation for stations, direct sampling for rasters)
            try:
                precip = self._get_forcing_fn(
                    current_time,
                    "precipitation",
                    weights_cache=self._interpolation_weights,
                    time_step_index=step,
                )
            except (KeyError, ValueError):
                logger.warning(f"Precipitation data not found for {current_time}, using zero")
                precip = np.zeros((self.nrows, self.ncols))

            # Log precipitation statistics (precip is in m/s, convert to mm/h for readability)
            precip_mm_h = precip * 1000.0 * 3600.0  # m/s -> mm/h
            precip_valid = precip_mm_h[np.isfinite(precip_mm_h)]
            if len(precip_valid) > 0:
                logger.debug(
                    f"Precipitation: mean={np.mean(precip_valid):.4f} mm/h, max={np.max(precip_valid):.4f} mm/h"
                )

            # 2. Calculate PET (and run energy balance pre-pass when active)
            energy_pre_state: dict | None = None

            # Build the effective turbulent-exchange coefficient CH*Kc used by
            # the energy-balance. Translated from MATLAB  CH(ko).*Kc_FAO_map(ko)
            # in mobidic_sid.m. Kc is applied inside the energy balance rather than as a
            # post-hoc scaling of PET, since CH affects both the sensible and
            # latent heat fluxes in the nonlinear Ts solution.
            kc_now = self._get_kc(current_time)
            ch_eff_ko = None
            if energy_active:
                if isinstance(kc_now, np.ndarray):
                    kc_ko = kc_now.ravel("F")[ko]
                    ch_eff_ko = ch_flat_full * kc_ko
                elif kc_now != 1.0:
                    ch_eff_ko = ch_flat_full * float(kc_now)
                else:
                    ch_eff_ko = ch_flat_full

            if energy_active:
                # Fetch energy forcing variables (units from interpolation/raster:
                # temperatures in degC, humidity in %, radiation in W/m^2, wind in m/s).
                tmax_grid = (
                    self._get_forcing_fn(
                        current_time,
                        "temperature_max",
                        weights_cache=self._interpolation_weights,
                        time_step_index=step,
                    )
                    + 273.15
                )
                tmin_grid = (
                    self._get_forcing_fn(
                        current_time,
                        "temperature_min",
                        weights_cache=self._interpolation_weights,
                        time_step_index=step,
                    )
                    + 273.15
                )
                rh_grid = (
                    self._get_forcing_fn(
                        current_time,
                        "humidity",
                        weights_cache=self._interpolation_weights,
                        time_step_index=step,
                    )
                    / 100.0
                )
                wind_grid = self._get_forcing_fn(
                    current_time,
                    "wind_speed",
                    weights_cache=self._interpolation_weights,
                    time_step_index=step,
                )
                rs_grid = self._get_forcing_fn(
                    current_time,
                    "radiation",
                    weights_cache=self._interpolation_weights,
                    time_step_index=step,
                )

                # First-step override: replace Ts/Td with Tair_lin if state was not loaded
                # (matching MATLAB mobidic_sid.m behaviour).
                if step == 0 and not self._state_was_loaded:
                    tair_lin_grid = (tmax_grid + tmin_grid) / 2.0
                    self.state.ts = np.where(np.isfinite(self.dtm), tair_lin_grid, np.nan)
                    self.state.td = np.where(np.isfinite(self.dtm), tair_lin_grid, np.nan)
                    td_rise_full = self.state.td.copy()

                # Solar hours (cached per Julian day)
                jday = current_time.timetuple().tm_yday
                if jday not in solar_cache:
                    solar_cache[jday] = solar_hours(lat, lon, jday)
                hrise_h, hset_h = solar_cache[jday]
                hrise_s = hrise_h * 3600.0
                hset_s = hset_h * 3600.0
                ctim_s = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
                ftim_s = ctim_s + self.dt

                # Extract ko-flat inputs
                ts_init_ko = self.state.ts.ravel("F")[ko].copy()
                td_init_ko = self.state.td.ravel("F")[ko].copy()
                td_rise_ko = td_rise_full.ravel("F")[ko].copy()
                tmax_ko = tmax_grid.ravel("F")[ko]
                tmin_ko = tmin_grid.ravel("F")[ko]
                rh_ko = rh_grid.ravel("F")[ko]
                wind_ko = wind_grid.ravel("F")[ko]
                rs_ko = rs_grid.ravel("F")[ko]

                # Pre-pass with etrsuetp=1 (saturated soil assumption)
                ts_pre_ko, td_pre_ko, etp_ko, td_rise_new_ko = compute_energy_balance_1l(
                    ts=ts_init_ko,
                    td=td_init_ko,
                    td_rise=td_rise_ko,
                    rs=rs_ko,
                    u=wind_ko,
                    tair_max=tmax_ko,
                    tair_min=tmin_ko,
                    qair=rh_ko,
                    ch=ch_eff_ko,
                    alb=alb_flat_full,
                    kaps=kaps,
                    nis=nis,
                    tcost=tcost,
                    pair=pair,
                    ctim_s=ctim_s,
                    ftim_s=ftim_s,
                    hrise_s=hrise_s,
                    hset_s=hset_s,
                    etrsuetp=1.0,
                    dt=self.dt,
                    reentry=False,
                )

                # Build PET grid in m/s (etp_ko is in [m] over the timestep)
                pet_full = np.zeros(self.nrows * self.ncols)
                pet_full[ko] = etp_ko / self.dt
                pet = pet_full.reshape((self.nrows, self.ncols), order="F")

                # Cache pre-pass quantities for re-entry after the soil balance
                energy_pre_state = {
                    "ts_init": ts_init_ko,
                    "td_init": td_init_ko,
                    "td_rise_new": td_rise_new_ko,
                    "ts_pre": ts_pre_ko,
                    "td_pre": td_pre_ko,
                    "etp": etp_ko,
                    "rs": rs_ko,
                    "u": wind_ko,
                    "tmax": tmax_ko,
                    "tmin": tmin_ko,
                    "rh": rh_ko,
                    "ctim_s": ctim_s,
                    "ftim_s": ftim_s,
                    "hrise_s": hrise_s,
                    "hset_s": hset_s,
                }
            elif self._raster_et_source == "et":
                # Actual ET from raster: use directly, Kc already embedded
                pet = self._get_forcing_fn(
                    current_time,
                    "et",
                    weights_cache=self._interpolation_weights,
                    time_step_index=step,
                )
            elif self._raster_et_source == "pet":
                # PET from raster: apply Kc to obtain actual PETc
                pet = self._get_forcing_fn(
                    current_time,
                    "pet",
                    weights_cache=self._interpolation_weights,
                    time_step_index=step,
                )
                if isinstance(kc_now, np.ndarray):
                    pet = pet * kc_now
                elif kc_now != 1.0:
                    pet = pet * float(kc_now)
            else:
                pet = self._calculate_pet(current_time)
                # Apply Kc as post-hoc scaling when energy balance is not active
                if isinstance(kc_now, np.ndarray):
                    pet = pet * kc_now
                elif kc_now != 1.0:
                    pet = pet * float(kc_now)

            # 3. Save interpolated meteorological data (if enabled)
            if meteo_writer is not None:
                meteo_kwargs = {"precipitation": precip}
                if energy_active and energy_pre_state is not None:
                    pet_grid_full = np.full(self.nrows * self.ncols, np.nan)
                    pet_grid_full[ko] = energy_pre_state["etp"] / self.dt
                    pet_grid_full = pet_grid_full.reshape((self.nrows, self.ncols), order="F")
                    meteo_kwargs.update(
                        {
                            "temperature_max": tmax_grid - 273.15,
                            "temperature_min": tmin_grid - 273.15,
                            "humidity": rh_grid * 100.0,
                            "wind_speed": wind_grid,
                            "radiation": rs_grid,
                            et_meteo_var: pet_grid_full,
                        }
                    )
                meteo_writer.append(current_time, **meteo_kwargs)

            # 4. Hillslope routing of previous timestep's flows (matching MATLAB mobidic_sid.m:1621-1625)
            # This must happen BEFORE soil mass balance to provide upstream contributions
            logger.debug("Routing previous timestep's flows through hillslope")

            # Convert to discharge for routing
            flr_prev_discharge = flr_prev * cell_area
            fld_prev_discharge = fld_prev * cell_area

            # Route through hillslope
            pir_discharge = hillslope_routing(flr_prev_discharge, self.flow_dir)
            pid_discharge = hillslope_routing(fld_prev_discharge, self.flow_dir)
            logger.debug("Hillslope routing completed")

            # Convert back to rates
            pir = pir_discharge / cell_area
            pid = pid_discharge / cell_area

            # 5. Soil water balance (cell-by-cell)
            logger.debug("Computing soil water balance")

            # Convert precipitation to depth over time step [m]
            precip_depth = precip * self.dt

            # Flatten 2D arrays to 1D (MATLAB: uses linear indexing with ko)
            # Extract only contributing pixels (ko) for processing (matching MATLAB mobidic_sid.m:733-736)
            wc_flat_full = self.state.wc.ravel("F")
            wg_flat_full = self.state.wg.ravel("F")
            wp_flat_full = self.state.wp.ravel("F") if self.state.wp is not None else None
            ws_flat_full = self.state.ws.ravel("F")

            # Extract ko cells only (matching MATLAB: Wc_ko = Wc(ko))
            wc_flat = wc_flat_full[ko]
            wg_flat = wg_flat_full[ko]
            wp_flat = wp_flat_full[ko] if wp_flat_full is not None else None
            ws_flat = ws_flat_full[ko]
            wc0_flat = self.wc0.ravel("F")[ko]
            wg0_flat = self.wg0.ravel("F")[ko]
            wtot0_flat = wtot0.ravel("F")[ko]
            precip_flat = precip_depth.ravel("F")[ko]
            pet_flat = pet.ravel("F")[ko] * self.dt
            # Multiply rate parameters by dt (matching MATLAB mobidic_sid.m:1681-1682)
            ks_flat = self.param_grids["ks"].ravel("F")[ko] * self.dt
            gamma_flat = self.param_grids["gamma"].ravel("F")[ko] * self.dt
            kappa_flat = self.param_grids["kappa"].ravel("F")[ko] * self.dt
            beta_flat = self.param_grids["beta"].ravel("F")[ko] * self.dt
            cha_flat = self.param_grids["cha"].ravel("F")[ko]
            f0_flat = self.param_grids["f0"].ravel("F")[ko]
            alpsur_flat = self.param_grids["alpsur"].ravel("F")[ko] * self.dt

            # Prepare routed flows from previous timestep (matching MATLAB mobidic_sid.m:1680)
            # Convert from [m/s] to [m] by multiplying by dt
            pir_flat = pir.ravel("F")[ko] * self.dt
            pid_flat = pid.ravel("F")[ko] * self.dt

            # Call soil_mass_balance with flattened arrays
            # Parameters have been pre-multiplied by dt above (lines 609-612, 616)
            (
                wc_out_flat,
                wg_out_flat,
                wp_out_flat,
                ws_out_flat,
                surface_runoff_flat,
                lateral_flow_depth_flat,
                et_flat,
                percolation_flat,
                capillary_flux_flat,
                wg_before_flat,
            ) = soil_mass_balance(
                wc=wc_flat,
                wc0=wc0_flat,
                wg=wg_flat,
                wg0=wg0_flat,
                wp=wp_flat,
                wp0=None,  # Plant reservoir disabled for now
                ws=ws_flat,
                ws0=None,
                wtot0=wtot0_flat,
                precipitation=precip_flat,
                surface_runoff_in=pir_flat,  # Routed surface runoff from previous timestep
                lateral_flow_in=pid_flat,  # Routed lateral flow from previous timestep
                potential_et=pet_flat,
                hydraulic_conductivity=ks_flat,
                hydraulic_conductivity_min=None,
                hydraulic_conductivity_max=None,
                channelized_fraction=cha_flat,
                surface_flow_exp=np.zeros_like(wc_flat),  # Not used
                lateral_flow_coeff=beta_flat,
                percolation_coeff=gamma_flat,
                absorption_coeff=kappa_flat,
                rainfall_fraction=f0_flat,
                et_shape=0.0,
                capillary_rise_enabled=False,
                test_mode=False,
                alpsur=alpsur_flat,
            )

            # Write ko results back to full grids (matching MATLAB: Wc(ko) = Wc_ko)
            wc_flat_full[ko] = wc_out_flat
            wg_flat_full[ko] = wg_out_flat
            if wp_out_flat is not None:
                wp_flat_full[ko] = wp_out_flat
            ws_flat_full[ko] = ws_out_flat

            # Initialize full output arrays for fluxes (preserve NaN for cells outside domain)
            surface_runoff_full = np.full(self.nrows * self.ncols, np.nan)
            lateral_flow_depth_full = np.full(self.nrows * self.ncols, np.nan)
            # Set non-ko cells within basin to zero
            surface_runoff_full[np.isfinite(self.dtm.ravel("F"))] = 0.0
            lateral_flow_depth_full[np.isfinite(self.dtm.ravel("F"))] = 0.0
            # Update ko cells with computed values
            surface_runoff_full[ko] = surface_runoff_flat
            lateral_flow_depth_full[ko] = lateral_flow_depth_flat

            # Reshape outputs back to 2D (Fortran order to match MATLAB)
            self.state.wc = wc_flat_full.reshape((self.nrows, self.ncols), order="F")
            self.state.wg = wg_flat_full.reshape((self.nrows, self.ncols), order="F")
            self.state.wp = (
                wp_flat_full.reshape((self.nrows, self.ncols), order="F") if wp_flat_full is not None else None
            )
            self.state.ws = ws_flat_full.reshape((self.nrows, self.ncols), order="F")
            surface_runoff = surface_runoff_full.reshape((self.nrows, self.ncols), order="F")
            lateral_flow_depth = lateral_flow_depth_full.reshape((self.nrows, self.ncols), order="F")

            # Build actual ET rate grid [m/s] for state output
            if self.config.output_states.evapotranspiration:
                et_full = np.full(self.nrows * self.ncols, np.nan)
                et_full[ko] = et_flat / self.dt
                self.state.et = et_full.reshape((self.nrows, self.ncols), order="F")

            # 5. Convert outputs from depth [m] to rate [m/s] (matching MATLAB mobidic_sid.m:1684)
            # flr: surface runoff rate [m/s] - analogous to MATLAB flr after division by dt
            # fld: lateral flow rate [m/s] - analogous to MATLAB fld after division by dt
            flr = surface_runoff / self.dt
            fld = lateral_flow_depth / self.dt

            # 5a. Energy balance re-entry: refine Ts/Td using actual ET/PET ratio
            if energy_active and energy_pre_state is not None:
                etp_ko = energy_pre_state["etp"]
                safe_etp = np.where(etp_ko > 0.0, etp_ko, 1.0)
                etrsuetp_ko = np.where(etp_ko > 0.0, np.minimum(et_flat / safe_etp, 1.0), 0.0)

                # Re-run with corrected etrsuetp; restart from initial Ts/Td and updated td_rise
                ts_re_ko, td_re_ko, _, _ = compute_energy_balance_1l(
                    ts=energy_pre_state["ts_init"],
                    td=energy_pre_state["td_init"],
                    td_rise=energy_pre_state["td_rise_new"],
                    rs=energy_pre_state["rs"],
                    u=energy_pre_state["u"],
                    tair_max=energy_pre_state["tmax"],
                    tair_min=energy_pre_state["tmin"],
                    qair=energy_pre_state["rh"],
                    ch=ch_eff_ko,
                    alb=alb_flat_full,
                    kaps=kaps,
                    nis=nis,
                    tcost=tcost,
                    pair=pair,
                    ctim_s=energy_pre_state["ctim_s"],
                    ftim_s=energy_pre_state["ftim_s"],
                    hrise_s=energy_pre_state["hrise_s"],
                    hset_s=energy_pre_state["hset_s"],
                    etrsuetp=etrsuetp_ko,
                    dt=self.dt,
                    reentry=True,
                )

                # Use re-entry result where the soil was unsaturated; pre-pass otherwise
                needs_re = etrsuetp_ko < 1.0
                ts_final_ko = np.where(needs_re, ts_re_ko, energy_pre_state["ts_pre"])
                td_final_ko = np.where(needs_re, td_re_ko, energy_pre_state["td_pre"])

                # Write back to state
                ts_flat_full = self.state.ts.ravel("F")
                td_flat_full = self.state.td.ravel("F")
                ts_flat_full[ko] = ts_final_ko
                td_flat_full[ko] = td_final_ko
                self.state.ts = ts_flat_full.reshape((self.nrows, self.ncols), order="F")
                self.state.td = td_flat_full.reshape((self.nrows, self.ncols), order="F")

                # Persist td_rise for next timestep
                td_rise_full_flat = td_rise_full.ravel("F")
                td_rise_full_flat[ko] = energy_pre_state["td_rise_new"]
                td_rise_full = td_rise_full_flat.reshape((self.nrows, self.ncols), order="F")

            # 5b. Groundwater dynamics
            # Linear reservoir model: baseflow is added to surface runoff
            # before lateral inflow accumulation
            if self.config.parameters.groundwater.model == "Linear":
                logger.debug("Computing linear groundwater dynamics")

                # Recharge rate [m/s]: percolation - capillary_rise - global_loss_per_cell
                # percolation_flat and capillary_flux_flat are in [m] over dt (from soil balance)
                percolation_rate = percolation_flat / self.dt
                capillary_rate = capillary_flux_flat / self.dt

                # Global loss distributed uniformly to contributing cells [m/s]
                global_loss = self.config.parameters.groundwater.global_loss  # [m³/s]
                global_loss_per_cell = global_loss / len(ko) / cell_area if len(ko) > 0 else 0.0

                recharge = percolation_rate - capillary_rate - global_loss_per_cell

                h_flat_full = self.state.h.ravel("F")
                h_flat = h_flat_full[ko]
                kf_flat = self.param_grids["kf"].ravel("F")[ko]

                h_out_flat, baseflow_flat = groundwater_linear(h_flat, kf_flat, recharge, self.dt)

                h_flat_full[ko] = h_out_flat
                self.state.h = h_flat_full.reshape((self.nrows, self.ncols), order="F")

                # Multi-aquifer averaging
                # When Mf defines >1 positive classes, average h within each class.
                if self.aquifer_ids is not None:
                    h2d = self.state.h
                    for aquifer_id in self.aquifer_ids:
                        mask = self.mf == aquifer_id
                        if not np.any(mask):
                            continue
                        mean_h = np.nanmean(h2d[mask])
                        if np.isfinite(mean_h):
                            h2d[mask] = mean_h

                # Add baseflow to surface runoff before accumulation
                flr_flat_full = flr.ravel("F")
                flr_flat_full[ko] = flr_flat_full[ko] + baseflow_flat
                flr = flr_flat_full.reshape((self.nrows, self.ncols), order="F")

            # 6. Accumulate lateral inflow to reaches from surface runoff (matching MATLAB glob_route_day.m)
            # Convert to discharge for accumulation
            flr_discharge = flr * cell_area
            lateral_inflow = self._accumulate_lateral_inflow(flr_discharge)
            logger.debug("Lateral inflow accumulation completed")

            # Store lateral inflow in state
            self.state.lateral_inflow = lateral_inflow

            # Zero out flr for ALL cells that contributed to reaches (matching MATLAB glob_route_day.m line 33)
            # This prevents double-counting - flows from all contributing cells are consumed after accumulation
            flr[self.hillslope_reach_map >= 0] = 0.0
            # Note: fld is NOT zeroed - lateral flow continues to route between hillslope cells

            # Store flr and fld for next timestep's routing (at step 3)
            flr_prev = flr.copy()
            fld_prev = fld.copy()

            # 6. Reservoir routing (if reservoirs exist)
            # Prepare network topology for routing (may be modified if reservoirs exist)
            routing_network = self._network_topology

            if self.reservoirs is not None and self.state.reservoir_states is not None:
                logger.debug("Computing reservoir routing")

                # Call reservoir routing
                (
                    self.state.reservoir_states,
                    self.state.discharge,
                    flr,
                    fld,
                    self.state.wg,
                ) = reservoir_routing(
                    reservoirs_data=self.reservoirs.reservoirs,
                    reservoir_states=self.state.reservoir_states,
                    reach_discharge=self.state.discharge,
                    surface_runoff=flr,
                    lateral_flow=fld,
                    soil_wg=self.state.wg,
                    soil_wg0=self.wg0,
                    current_time=current_time,
                    dt=self.dt,
                    cell_area=cell_area,
                )

                # Add reservoir outflows to lateral inflow of outlet reaches (matching MATLAB glob_route_day.m:46-53)
                for i, reservoir in enumerate(self.reservoirs.reservoirs):
                    outlet_reach = reservoir.outlet_reach
                    if outlet_reach is not None:
                        lateral_inflow[outlet_reach] += self.state.reservoir_states[i].outflow

                # Clear upstream connections for outlet reaches (matching MATLAB glob_route_day.m:48-50)
                # This prevents double-counting: the reservoir has already collected upstream volumes
                # Create modified topology with cleared upstream connections for outlet reaches
                routing_network = self._network_topology.copy()
                routing_network["upstream_1_idx"] = self._network_topology["upstream_1_idx"].copy()
                routing_network["upstream_2_idx"] = self._network_topology["upstream_2_idx"].copy()
                routing_network["n_upstream"] = self._network_topology["n_upstream"].copy()

                for reservoir in self.reservoirs.reservoirs:
                    outlet_reach = reservoir.outlet_reach
                    if outlet_reach is not None:
                        # Clear upstream connections (MATLAB: ret(j).ramimonte=[nan nan])
                        routing_network["upstream_1_idx"][outlet_reach] = -1
                        routing_network["upstream_2_idx"][outlet_reach] = -1
                        # Set upstream count to 0 (MATLAB: ram_fin(j) = 0)
                        routing_network["n_upstream"][outlet_reach] = 0

            # 7. Channel routing
            logger.debug("Computing channel routing")
            self.state.discharge, routing_state = linear_channel_routing(
                network=routing_network,
                discharge_initial=self.state.discharge,
                lateral_inflow=lateral_inflow,
                dt=self.dt,
            )

            # 8. Store results
            discharge_ts.append(self.state.discharge.copy())
            lateral_inflow_ts.append(lateral_inflow.copy())
            time_ts.append(current_time)

            # 9. Save intermediate states if configured
            if state_writer is not None and self._should_save_state(step, current_time):
                logger.debug(f"Appending state to buffer at step {step + 1}/{n_steps}")
                state_writer.append_state(self.state, current_time)

            # Log progress based on step interval or time interval (whichever comes first)
            current_wall_time = time.time()
            time_since_last_log = current_wall_time - last_log_time
            is_step_interval = (step + 1) % progress_step_interval == 0
            is_time_interval = time_since_last_log >= progress_time_interval
            is_final_step = step == n_steps - 1

            if is_step_interval or is_time_interval or is_final_step:
                progress_bar = _create_progress_bar(step + 1, n_steps, bar_length=20)
                q_mean = np.mean(self.state.discharge)
                q_max = np.max(self.state.discharge)
                date_str = current_time.strftime("%Y-%m-%d %H:%M")

                logger.info(
                    f"{progress_bar} {step + 1}/{n_steps} | Simulation time: {date_str} | "
                    f"Q_mean={q_mean:.3f} m³/s | Q_max={q_max:.3f} m³/s"
                )
                last_log_time = current_wall_time

            # Advance time
            current_time += timedelta(seconds=self.dt)

        # Store results
        results.time_series["discharge"] = np.array(discharge_ts)
        results.time_series["lateral_inflow"] = np.array(lateral_inflow_ts)
        results.time_series["time"] = time_ts
        results.final_state = self.state

        # Calculate elapsed time
        simulation_end_time = time.time()
        elapsed_seconds = simulation_end_time - simulation_start_time
        elapsed_minutes = elapsed_seconds / 60
        elapsed_hours = elapsed_minutes / 60

        # Format elapsed time message
        if elapsed_hours >= 1:
            elapsed_str = f"{elapsed_hours:.2f} hours"
        elif elapsed_minutes >= 1:
            elapsed_str = f"{elapsed_minutes:.2f} minutes"
        else:
            elapsed_str = f"{elapsed_seconds:.2f} seconds"

        logger.success(f"Simulation completed: {n_steps} time steps in {elapsed_str}")

        # Auto-save reports based on configuration
        output_dir = Path(self.config.paths.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine reach selection
        settings = self.config.output_report_settings
        reach_selection = settings.reach_selection
        selected_reaches = None
        reach_file = None

        if settings.reach_selection == "file":
            reach_file = settings.sel_file
        elif settings.reach_selection == "list":
            selected_reaches = settings.sel_list

        # Get output format from configuration
        output_format = self.config.output_report_settings.output_format
        file_extension = "csv" if output_format.lower() == "csv" else "parquet"

        # Save discharge report if enabled
        if self.config.output_report.discharge:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            discharge_path = output_dir / f"discharge_{start_str}_{end_str}.{file_extension}"
            logger.info("Exporting discharges")
            results.save_report(
                output_path=discharge_path,
                reach_selection=reach_selection,
                selected_reaches=selected_reaches,
                reach_file=reach_file,
                output_format=output_format,
            )

        # Save lateral inflow report if enabled
        if self.config.output_report.lateral_inflow:
            lateral_inflow_path = output_dir / f"lateral_inflow_{start_str}_{end_str}.{file_extension}"
            logger.info("Exporting lateral inflows")
            results.save_lateral_inflow_report(
                output_path=lateral_inflow_path,
                reach_selection=reach_selection,
                selected_reaches=selected_reaches,
                reach_file=reach_file,
                output_format=output_format,
            )

        # Save final state if enabled (for "final" mode, only save the last state)
        if state_writer is not None:
            if state_settings.output_states == "final":
                final_time = results.time_series["time"][-1]
                logger.info("Saving final state")
                state_writer.append_state(self.state, final_time)

            # Close the state writer (flushes any remaining buffered states)
            state_writer.close()

        # Close the meteo writer (writes all buffered data to NetCDF)
        if meteo_writer is not None:
            meteo_writer.close()

        return results
