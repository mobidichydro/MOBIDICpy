"""Main simulation engine for MOBIDIC.

This module implements the main time-stepping loop for the MOBIDIC hydrological model.
It orchestrates the water balance calculations, routing, and I/O operations.

For Phase 1, this implements a simplified version without:
- Energy balance (uses simple PET instead)
- Groundwater models (percolation goes to baseflow)
- Reservoir routing

Translated from MATLAB: mobidic_sid.m (main simulation loop)
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Any
import numpy as np
import pandas as pd
from loguru import logger

from mobidic.config import MOBIDICConfig
from mobidic.preprocessing.meteo_preprocessing import MeteoData
from mobidic.core.soil_water_balance import soil_mass_balance
from mobidic.core.routing import hillslope_routing, linear_channel_routing
from mobidic.utils.interpolation import precipitation_interpolation, temperature_interpolation
from mobidic.utils.pet import calculate_pet


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
            add_metadata={
                "basin_id": self.config.basin.id,
                "paramset_id": self.config.basin.paramset_id,
            },
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

        # Extract grids
        self.dtm = gisdata.grids["dtm"]
        self.flow_dir = gisdata.grids["flow_dir"]
        self.flow_acc = gisdata.grids["flow_acc"]
        self.wc0 = gisdata.grids["Wc0"]
        self.wg0 = gisdata.grids["Wg0"]
        self.ks = gisdata.grids["ks"]
        self.hillslope_reach_map = gisdata.hillslope_reach_map

        # Optional grids
        self.kf = gisdata.grids.get("kf")
        self.gamma = gisdata.grids.get("gamma")
        self.kappa = gisdata.grids.get("kappa")
        self.beta = gisdata.grids.get("beta")
        self.alpha = gisdata.grids.get("alpha")

        # River network
        self.network = gisdata.network

        # Time step
        self.dt = config.simulation.timestep

        # Initialize state
        self.state = None

        logger.info(
            f"Simulation initialized: grid={self.nrows}x{self.ncols}, "
            f"dt={self.dt}s, network={len(self.network)} reaches"
        )

    def _initialize_state(self) -> SimulationState:
        """Initialize simulation state variables.

        Returns:
            Initial simulation state
        """
        logger.info("Initializing simulation state")

        # Initialize soil water content from configuration
        wcsat = self.config.initial_conditions.Wcsat
        wgsat = self.config.initial_conditions.Wgsat
        ws_init = self.config.initial_conditions.Ws

        wc = self.wc0 * wcsat
        wg = self.wg0 * wgsat
        wp = np.zeros((self.nrows, self.ncols))  # Plant reservoir (if used)
        ws = np.full((self.nrows, self.ncols), ws_init)

        # Initialize river discharge
        discharge = np.zeros(len(self.network))

        logger.success(
            f"State initialized: "
            f"Wc={np.nanmean(wc) * 1000:.1f} mm, "
            f"Wg={np.nanmean(wg) * 1000:.1f} mm, "
            f"Ws={np.nanmean(ws) * 1000:.1f} mm"
        )

        return SimulationState(wc, wg, wp, ws, discharge)

    def _interpolate_forcing(self, time: datetime, variable: str) -> np.ndarray:
        """Interpolate meteorological forcing data to grid.

        Uses MATLAB-equivalent methods:
        - Precipitation: nearest neighbor (pluviomap.m)
        - Temperature: IDW with elevation correction (tempermap.m)

        Args:
            time: Current simulation time
            variable: Variable name ('precipitation', 'temperature_min', 'temperature_max', etc.)

        Returns:
            2D grid of interpolated values
        """
        # Get station data for this variable
        if variable not in self.forcing.stations:
            raise ValueError(f"Variable '{variable}' not found in forcing data")

        var_stations = self.forcing.stations[variable]

        if len(var_stations) == 0:
            raise ValueError(f"No stations found for variable '{variable}'")

        # Extract station coordinates and values for the given time
        station_x = []
        station_y = []
        station_elevation = []
        station_values = []

        target_time = pd.Timestamp(time)

        for station in var_stations:
            # Find the value at the nearest time
            if len(station["time"]) == 0:
                continue

            time_index = station["time"].get_indexer([target_time], method="nearest")[0]

            if time_index >= 0 and time_index < len(station["data"]):
                value = station["data"][time_index]

                # Only include station if data is not NaN
                if not np.isnan(value):
                    station_x.append(station["x"])
                    station_y.append(station["y"])
                    station_elevation.append(station["elevation"])
                    station_values.append(value)

        if len(station_values) == 0:
            raise ValueError(f"No valid data found for {variable} at time {time}")

        # Convert to numpy arrays
        station_x = np.array(station_x)
        station_y = np.array(station_y)
        station_elevation = np.array(station_elevation)
        station_values = np.array(station_values)

        # Use single resolution value (assume square cells)
        resolution = self.resolution[0]

        # Choose interpolation method based on variable
        if variable == "precipitation":
            # Use nearest neighbor for precipitation (MATLAB: pluviomap.m)
            grid_values = precipitation_interpolation(
                station_x,
                station_y,
                station_values,
                self.dtm,
                self.xllcorner,
                self.yllcorner,
                resolution,
            )
        else:
            # Use IDW with elevation correction for temperature and other variables (MATLAB: tempermap.m)
            grid_values = temperature_interpolation(
                station_x,
                station_y,
                station_elevation,
                station_values,
                self.dtm,
                self.xllcorner,
                self.yllcorner,
                resolution,
                weights_matrix=None,
                apply_elevation_correction=True,
                power=2.0,
            )

        return grid_values

    def _calculate_pet(self, time: datetime) -> np.ndarray:
        """Calculate potential evapotranspiration.

        For Phase 1 (no energy balance), uses MATLAB approach: constant 1 mm/day.
        MATLAB: etp = Mones/(1000*3600*24) [mobidic_sid.m line 332]

        Args:
            time: Current simulation time

        Returns:
            PET grid [m] over time step
        """
        # Use MATLAB approach: constant 1 mm/day when no energy balance
        pet = calculate_pet((self.nrows, self.ncols), self.dt, pet_rate_mm_day=1.0)

        return pet

    def _soil_parameters_to_grids(self) -> dict[str, np.ndarray]:
        """Convert soil parameters from config to grids.

        Returns:
            Dictionary of parameter grids
        """
        # Get parameters from config
        params = self.config.parameters

        # Create grids from scalar parameters or raster files
        # If raster exists, use it; otherwise use scalar value
        param_grids = {}

        # Hydraulic conductivity [m/s] -> [m] for time step
        if self.ks is not None:
            param_grids["ks"] = self.ks * self.dt
        else:
            param_grids["ks"] = np.full((self.nrows, self.ncols), params.soil.ks) * self.dt

        # Flow coefficients
        if self.gamma is not None:
            param_grids["gamma"] = self.gamma * self.dt
        else:
            param_grids["gamma"] = np.full((self.nrows, self.ncols), params.soil.gamma) * self.dt

        if self.kappa is not None:
            param_grids["kappa"] = self.kappa * self.dt
        else:
            param_grids["kappa"] = np.full((self.nrows, self.ncols), params.soil.kappa) * self.dt

        if self.beta is not None:
            param_grids["beta"] = self.beta * self.dt
        else:
            param_grids["beta"] = np.full((self.nrows, self.ncols), params.soil.beta) * self.dt

        if self.alpha is not None:
            param_grids["alpha"] = self.alpha * self.dt
        else:
            param_grids["alpha"] = np.full((self.nrows, self.ncols), params.soil.alpha) * self.dt

        # Other parameters
        param_grids["f0"] = np.full((self.nrows, self.ncols), 0.5)  # Rainfall fraction
        param_grids["cha"] = np.full((self.nrows, self.ncols), self.config.parameters.multipliers.chan_factor)

        # Calculate surface flow parameter (from alpha)
        # alpsur = surface flow velocity, exp_alp = exp(-alpha * alpsur * dt)
        alpsur = 0.01  # m/s, typical hillslope velocity
        param_grids["surface_flow_param"] = alpsur * param_grids["alpha"]

        return param_grids

    def _accumulate_lateral_inflow(self, lateral_flow: np.ndarray) -> np.ndarray:
        """Accumulate lateral flow contributions to each reach.

        Following MATLAB's approach (mobidic_sid.m line 224):
        - Only accumulate from cells where ch > 0 (valid reach IDs)
        - Skip NaN cells (outside domain)
        - Skip -9999 cells (inside domain but cannot reach river network)

        Args:
            lateral_flow: 2D grid of lateral flow [m³/s]

        Returns:
            1D array of lateral inflow to each reach [m³/s]
        """
        n_reaches = len(self.network)
        lateral_inflow = np.zeros(n_reaches)

        # Flatten lateral flow and hillslope-reach mapping
        lateral_flow_flat = lateral_flow.ravel()
        hillslope_map_flat = self.hillslope_reach_map.ravel()

        # Sum lateral flow for each reach
        # MATLAB: ko = find(isfinite(zz) & (ch>0)); % contributing pixels
        for cell_idx, reach_id in enumerate(hillslope_map_flat):
            # Skip NaN (outside domain) and negative values like -9999 (unassigned)
            if not np.isnan(reach_id) and reach_id >= 0:
                reach_id = int(reach_id)
                # Additional safety check for valid reach index
                if reach_id < n_reaches and not np.isnan(lateral_flow_flat[cell_idx]):
                    lateral_inflow[reach_id] += lateral_flow_flat[cell_idx]

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
        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        logger.info(f"Starting simulation: {start_date} to {end_date}, dt={self.dt}s")

        # Initialize state
        self.state = self._initialize_state()

        # Get soil parameters
        soil_params = self._soil_parameters_to_grids()

        # Initialize results container
        results = SimulationResults(self.config, simulation=self)
        discharge_ts = []
        time_ts = []

        # Calculate number of time steps
        n_steps = int((end_date - start_date).total_seconds() / self.dt)
        logger.info(f"Number of time steps: {n_steps}")

        # Get total soil capacity
        wtot0 = self.wc0 + self.wg0

        # Cell area for unit conversion [m²]
        cell_area = self.resolution[0] * self.resolution[1]

        # Main time loop
        current_time = start_date
        for step in range(n_steps):
            logger.debug(f"Time step {step + 1}/{n_steps}: {current_time}")

            # 1. Interpolate meteorological forcing
            try:
                precip = self._interpolate_forcing(current_time, "precipitation")
            except (KeyError, ValueError):
                logger.warning(f"Precipitation data not found for {current_time}, using zero")
                precip = np.zeros((self.nrows, self.ncols))

            # 2. Calculate PET
            pet = self._calculate_pet(current_time)

            # 3. Soil water balance (cell-by-cell)
            logger.debug("Computing soil water balance")

            # Convert precipitation to depth over time step [m]
            precip_depth = precip * self.dt

            # Flatten 2D arrays to 1D (MATLAB: uses linear indexing with ko)
            # soil_mass_balance expects 1D arrays as per MATLAB implementation
            wc_flat = self.state.wc.ravel()
            wg_flat = self.state.wg.ravel()
            wp_flat = self.state.wp.ravel() if self.state.wp is not None else None
            ws_flat = self.state.ws.ravel()
            wc0_flat = self.wc0.ravel()
            wg0_flat = self.wg0.ravel()
            wtot0_flat = wtot0.ravel()
            precip_flat = precip_depth.ravel()
            pet_flat = pet.ravel()
            ks_flat = soil_params["ks"].ravel()
            gamma_flat = soil_params["gamma"].ravel()
            kappa_flat = soil_params["kappa"].ravel()
            beta_flat = soil_params["beta"].ravel()
            alpha_flat = soil_params["alpha"].ravel()
            cha_flat = soil_params["cha"].ravel()
            f0_flat = soil_params["f0"].ravel()
            surface_flow_param_flat = soil_params["surface_flow_param"].ravel()

            # Call soil_mass_balance with flattened arrays
            # Pre-multiply parameters by dt (already done in _soil_parameters_to_grids)
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
                surface_runoff_in=np.zeros_like(wc_flat),  # No upstream for now
                lateral_flow_in=np.zeros_like(wc_flat),  # No upstream for now
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
                surface_flow_param=surface_flow_param_flat,
            )

            # Reshape outputs back to 2D
            self.state.wc = wc_out_flat.reshape((self.nrows, self.ncols))
            self.state.wg = wg_out_flat.reshape((self.nrows, self.ncols))
            self.state.wp = wp_out_flat.reshape((self.nrows, self.ncols)) if wp_out_flat is not None else None
            self.state.ws = ws_out_flat.reshape((self.nrows, self.ncols))
            surface_runoff = surface_runoff_flat.reshape((self.nrows, self.ncols))
            lateral_flow_depth = lateral_flow_depth_flat.reshape((self.nrows, self.ncols))

            # 4. Hillslope routing
            logger.debug("Computing hillslope routing")

            # Convert lateral flow from depth [m] to discharge [m³/s]
            lateral_flow_discharge = lateral_flow_depth / self.dt * cell_area

            # Route hillslope flow
            accumulated_flow = hillslope_routing(
                lateral_flow_discharge,
                self.flow_dir,
                flow_dir_type="Grass",  # MOBIDIC notation
            )

            # 5. Accumulate lateral inflow to reaches
            lateral_inflow = self._accumulate_lateral_inflow(accumulated_flow)

            # 6. Channel routing
            logger.debug("Computing channel routing")

            self.state.discharge, routing_state = linear_channel_routing(
                network=self.network,
                discharge_initial=self.state.discharge,
                lateral_inflow=lateral_inflow,
                dt=self.dt,
                storage_coeff="storage_coeff",
            )

            # 7. Store results
            discharge_ts.append(self.state.discharge.copy())
            time_ts.append(current_time)

            # Log progress every 100 steps
            if (step + 1) % 100 == 0 or step == n_steps - 1:
                logger.info(
                    f"Step {step + 1}/{n_steps} completed. "
                    f"Q_mean={np.mean(self.state.discharge):.2f} m³/s, "
                    f"Q_max={np.max(self.state.discharge):.2f} m³/s"
                )

            # Advance time
            current_time += timedelta(seconds=self.dt)

        # Store results
        results.time_series["discharge"] = np.array(discharge_ts)
        results.time_series["time"] = time_ts
        results.final_state = self.state

        logger.success(f"Simulation completed: {n_steps} time steps")

        return results
