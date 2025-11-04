"""Main simulation engine for MOBIDIC.

This module implements the main time-stepping loop for the MOBIDIC hydrological model.
It orchestrates the water balance calculations, routing, and I/O operations.

Currently, this implements a simplified version without:
- Energy balance (uses simple PET instead)
- Groundwater models (percolation goes to baseflow)
- Reservoir routing

Translated from MATLAB: mobidic_sid.m (main simulation loop)
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Any
import time
import numpy as np
import pandas as pd
from loguru import logger

from mobidic.config import MOBIDICConfig
from mobidic.core import constants as const
from mobidic.preprocessing.meteo_preprocessing import MeteoData
from mobidic.core.soil_water_balance import soil_mass_balance
from mobidic.core.routing import hillslope_routing, linear_channel_routing
from mobidic.core.interpolation import precipitation_interpolation, station_interpolation
from mobidic.core.pet import calculate_pet


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
    ):
        """Initialize simulation state.

        Args:
            wc: Capillary water content [m]
            wg: Gravitational water content [m]
            wp: Plant/canopy water content [m] (None to disable)
            ws: Surface water content [m]
            discharge: River discharge for each reach [m³/s]
        """
        self.wc = wc
        self.wg = wg
        self.wp = wp
        self.ws = ws
        self.discharge = discharge


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

    def save_states(self, output_path: str | Path, time: datetime | None = None) -> None:
        """Save simulation states to NetCDF file.

        Args:
            output_path: Path to output NetCDF file
            time: Time of state (if None, uses last time from time series)
        """
        if self.final_state is None:
            raise ValueError("No state to save. Run simulation first.")

        if self.simulation is None:
            raise ValueError("Cannot save state without simulation object")

        if time is None:
            time = self.time_series["time"][-1]

        from mobidic.io import save_state

        save_state(
            state=self.final_state,
            output_path=output_path,
            time=time,
            grid_metadata=self.simulation.gisdata.metadata,
            network_size=len(self.simulation.network),
            add_metadata={
                "basin_id": self.config.basin.id,
                "paramset_id": self.config.basin.paramset_id,
            },
        )

    def save_report(
        self,
        output_path: str | Path,
        reach_selection: str = "all",
        selected_reaches: list[int] | None = None,
        add_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save discharge time series to Parquet file.

        Args:
            output_path: Path to output Parquet file
            reach_selection: "all", "outlets", or "list"
            selected_reaches: List of reach IDs (if reach_selection="list")
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
            add_metadata=add_metadata,
        )

    def save_lateral_inflow_report(
        self,
        output_path: str | Path,
        reach_selection: str = "all",
        selected_reaches: list[int] | None = None,
    ) -> None:
        """Save lateral inflow time series to Parquet file.

        Args:
            output_path: Path to output Parquet file
            reach_selection: "all", "outlets", or "list"
            selected_reaches: List of reach IDs (if reach_selection="list")
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
        )

    def save_final_state(
        self,
        output_path: str | Path,
        add_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save final simulation state to NetCDF file.

        Args:
            output_path: Path to output NetCDF file
            add_metadata: Additional metadata to include in file
        """
        if self.final_state is None:
            raise ValueError("No final state to save. Run simulation first.")

        if self.simulation is None:
            raise ValueError("Cannot save final state without simulation object")

        if "time" not in self.time_series or not self.time_series["time"]:
            raise ValueError("No time information available")

        from mobidic.io import save_state

        # Get final time
        final_time = self.time_series["time"][-1]

        save_state(
            state=self.final_state,
            output_path=output_path,
            time=final_time,
            grid_metadata=self.simulation.gisdata.metadata,
            network_size=len(self.simulation.network),
            add_metadata=add_metadata,
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
        forcing: MeteoData,
        config: MOBIDICConfig,
    ):
        """Initialize simulation.

        Args:
            gisdata: Preprocessed GIS data (from load_gisdata or run_preprocessing)
            forcing: Meteorological forcing data as MeteoData container
            config: MOBIDIC configuration
        """
        self.gisdata = gisdata
        self.forcing = forcing
        self.config = config

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

        # River network
        self.network = gisdata.network

        # Time step
        self.dt = config.simulation.timestep

        # Prepare parameter grids
        self.param_grids = self._prepare_grids()

        # Preprocess and cache network topology for fast routing
        self._network_topology = self._preprocess_network_topology()

        # Pre-compute interpolation weights for meteorological forcing
        # Currently, only precipitation is used in the main loop
        self._interpolation_weights = self._precompute_interpolation_weights(["precipitation"])

        # Time indices cache (will be populated in run() when simulation period is known)
        self._time_indices_cache = None

        # Initialize state
        self.state = None

        logger.info(
            f"Simulation initialized: grid={self.nrows}x{self.ncols}, "
            f"dt={self.dt}s, network={len(self.network)} reaches"
        )

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

        # Initialize river discharge
        discharge = np.zeros(len(self.network))

        logger.success(
            f"State initialized. Initial conditions (average): "
            f"Wc={np.nanmean(wc) * 1000:.1f} mm, "
            f"Wg={np.nanmean(wg) * 1000:.1f} mm, "
            f"Ws={np.nanmean(ws) * 1000:.1f} mm, "
            f"Wp={np.nanmean(wp) * 1000:.1f} mm"
        )

        return SimulationState(wc, wg, wp, ws, discharge)

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

                if precip_interp == "nearest":
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
                else:  # "idw"
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

                # Convert from mm (over dt) to m/s, matching MATLAB line 1240 (mobidic_sid.m):
                # pp=pp/1000/dt; % average intensity during dt [m/s]
                grid_values = grid_values / 1000.0 / self.dt
        else:
            # Use IDW with elevation correction for temperature and other variables
            # MATLAB calc_forcing_day.m uses different settings per variable:
            # - Temperature (min/max): switchregz=1, expon=2 (elevation correction, power=2)
            # - Humidity: switchregz=0, expon=2 (no elevation correction, power=2)
            # - Wind: switchregz=0, expon=0.5 (no elevation correction, power=0.5)
            # - Radiation: switchregz=0, expon=2 (no elevation correction, power=2)
            # Currently only temperature is implemented with elevation correction
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
                apply_elevation_correction=True,  # True for temperature, False for others
                power=2.0,  # 2.0 for most, 0.5 for wind
            )

        return grid_values

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
        f0_value = 0.85 * (1 - np.exp(-self.dt / (24 * 3600) * np.log(0.85 / 0.10)))
        param_grids["f0"] = np.full((self.nrows, self.ncols), f0_value)
        param_grids["f0"][np.isnan(self.dtm)] = np.nan

        # Hydraulic conductivity [m/s]
        ks_factor = self.config.parameters.multipliers.ks_factor
        if self.ks is not None:
            param_grids["ks"] = self.ks * ks_factor
        else:
            param_grids["ks"] = np.full((self.nrows, self.ncols), params.soil.ks) * ks_factor

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
        # From buildgis_mysql_include.m line 655-656
        param_grids["cha"] = self.flow_acc / np.nanmax(self.flow_acc)
        param_grids["cha"] = np.where(param_grids["cha"] > 0, param_grids["cha"], 0.0)

        # Surface alpha parameter alpsur
        param_grids["alpsur"] = self.alpsur * param_grids["alpha"]

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
                if precip_interp == "nearest":
                    # Nearest neighbor doesn't use weights matrix
                    logger.debug(f"{variable}: using nearest neighbor (no weights needed)")
                    weights_cache[variable] = None
                else:  # "idw"
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

        logger.info("=" * 80)
        logger.info("MOBIDIC SIMULATION")
        logger.info("=" * 80)
        logger.info(f"Basin: {self.config.basin.id}")
        logger.info(f"Parameter set: {self.config.basin.paramset_id}")
        logger.info("")

        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        logger.info(f"Starting simulation: {start_date} to {end_date}, dt={self.dt}s")

        # Initialize state
        self.state = self._initial_state()

        # Initialize results container
        results = SimulationResults(self.config, simulation=self)
        discharge_ts = []
        lateral_inflow_ts = []
        time_ts = []

        # Calculate number of time steps (inclusive of end_date)
        n_steps = int((end_date - start_date).total_seconds() / self.dt) + 1
        logger.info(f"Number of time steps: {n_steps}")

        # Pre-compute time indices for all simulation timesteps
        # Currently, only precipitation is used in the main loop
        simulation_times = pd.date_range(start=start_date, periods=n_steps, freq=f"{self.dt}s")
        self._time_indices_cache = self._precompute_time_indices(simulation_times, variables=["precipitation"])

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

        # Calculate progress logging interval: either 20 steps total or every 30 seconds
        # Use whichever results in fewer log messages
        steps_per_intervals = max(1, n_steps // 20)
        progress_step_interval = steps_per_intervals
        progress_time_interval = 30.0  # seconds

        # Main time loop
        logger.info("Starting simulation main loop")
        current_time = start_date
        simulation_start_time = time.time()
        last_log_time = simulation_start_time
        for step in range(n_steps):
            logger.debug(f"Time step {step + 1}/{n_steps}: {current_time}")

            # 1. Interpolate meteorological forcing (using cached weights and time indices for performance)
            try:
                precip = self._interpolate_forcing(
                    current_time,
                    "precipitation",
                    weights_cache=self._interpolation_weights,
                    time_step_index=step,
                )
            except (KeyError, ValueError):
                logger.warning(f"Precipitation data not found for {current_time}, using zero")
                precip = np.zeros((self.nrows, self.ncols))

            # 2. Calculate PET
            pet = self._calculate_pet(current_time)

            # 3. Hillslope routing of previous timestep's flows (matching MATLAB mobidic_sid.m:1621-1625)
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

            # 4. Soil water balance (cell-by-cell)
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
                et_shape=3.0,
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

            # 5. Convert outputs from depth [m] to rate [m/s] (matching MATLAB mobidic_sid.m:1684)
            # flr: surface runoff rate [m/s] - analogous to MATLAB flr after division by dt
            # fld: lateral flow rate [m/s] - analogous to MATLAB fld after division by dt
            flr = surface_runoff / self.dt
            fld = lateral_flow_depth / self.dt

            # 6. Accumulate lateral inflow to reaches from surface runoff (matching MATLAB glob_route_day.m)
            # Convert to discharge for accumulation
            flr_discharge = flr * cell_area
            lateral_inflow = self._accumulate_lateral_inflow(flr_discharge)
            logger.debug("Lateral inflow accumulation completed")

            # Zero out flr for ALL cells that contributed to reaches (matching MATLAB glob_route_day.m line 33)
            # This prevents double-counting - flows from all contributing cells are consumed after accumulation
            flr[self.hillslope_reach_map >= 0] = 0.0
            # Note: fld is NOT zeroed - lateral flow continues to route between hillslope cells

            # Store flr and fld for next timestep's routing (at step 3)
            flr_prev = flr.copy()
            fld_prev = fld.copy()

            # 7. Channel routing
            logger.debug("Computing channel routing")

            self.state.discharge, routing_state = linear_channel_routing(
                network=self._network_topology,
                discharge_initial=self.state.discharge,
                lateral_inflow=lateral_inflow,
                dt=self.dt,
            )

            # 8. Store results
            discharge_ts.append(self.state.discharge.copy())
            lateral_inflow_ts.append(lateral_inflow.copy())
            time_ts.append(current_time)

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
                    f"Q_mean={q_mean:.2f} m³/s | Q_max={q_max:.2f} m³/s"
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
        if settings.reach_selection == "file":
            # Load reach IDs from file
            import json

            with open(settings.sel_file) as f:
                selected_reaches = json.load(f)
            reach_selection = "list"
        elif settings.reach_selection == "list":
            selected_reaches = settings.sel_list
            reach_selection = "list"
        else:
            selected_reaches = None
            reach_selection = settings.reach_selection

        # Save discharge report if enabled
        if self.config.output_report.discharge:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            discharge_path = output_dir / f"discharge_{start_str}_{end_str}.parquet"
            logger.info("Exporting discharges")
            results.save_report(
                output_path=discharge_path,
                reach_selection=reach_selection,
                selected_reaches=selected_reaches,
            )

        # Save lateral inflow report if enabled
        if self.config.output_report.lateral_inflow:
            lateral_inflow_path = output_dir / f"lateral_inflow_{start_str}_{end_str}.parquet"
            logger.info("Exporting lateral inflows")
            results.save_lateral_inflow_report(
                output_path=lateral_inflow_path,
                reach_selection=reach_selection,
                selected_reaches=selected_reaches,
            )

        # Save final state if enabled
        states_dir = Path(self.config.paths.states)
        states_dir.mkdir(parents=True, exist_ok=True)
        state_settings = self.config.output_states_settings

        if state_settings.output_states in ["final", "all"]:
            final_time = results.time_series["time"][-1]
            state_filename = f"state_{final_time.strftime('%Y%m%d_%H%M%S')}.nc"
            state_path = states_dir / state_filename
            logger.info("Auto-saving final state")
            results.save_final_state(
                output_path=state_path,
                add_metadata={
                    "basin_id": self.config.basin.id,
                    "paramset_id": self.config.basin.paramset_id,
                },
            )

        return results
