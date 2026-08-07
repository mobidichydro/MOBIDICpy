"""PestSetup orchestrator: creates PEST++ working directory and runs calibration.

This module builds the complete PEST++ setup from a CalibrationConfig:
- Generates template (.tpl) and instruction (.ins) files
- Builds the PEST control file (.pst) via pyemu
- Generates the forward model runner script
- Optionally runs the initial forward run
- Executes PEST++ (local or cluster mode)
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from mobidic.calibration.config import CalibrationConfig, load_calibration_config

if TYPE_CHECKING:
    from mobidic.calibration.results import CalibrationResults
from mobidic.calibration.forward_model import generate_forward_run_script
from mobidic.calibration.instruction import generate_instruction_file
from mobidic.calibration.observation import (
    align_observations_to_simulation,
    load_observations,
    observation_weights,
)
from mobidic.calibration.template import generate_model_input_csv, generate_template_file

# Suppress pyemu warning about missing flopy (not needed for PEST++ calibration)
warnings.filterwarnings("ignore", message="Failed to import legacy module", module="pyemu")

# PEST++ tool name mapping to executable names
PEST_TOOL_MAP = {
    "glm": "pestpp-glm",
    "ies": "pestpp-ies",
    "sen": "pestpp-sen",
    "swp": "pestpp-swp",
    "da": "pestpp-da",
    "opt": "pestpp-opt",
    "mou": "pestpp-mou",
    "sqp": "pestpp-sqp",
}


def _all_states_enabled(mobidic_config):
    """Return an OutputStates with every flag on, whatever the user configured.

    ``StateWriter`` writes a state variable only when its flag is set, and
    ``load_state`` silently zero-fills anything missing. A warm-up state saved
    with the user's own (usually partial) flags would therefore drop ``flr`` and
    ``fld`` and reintroduce a one-timestep discontinuity at every cycle boundary.
    """
    output_states = mobidic_config.output_states.model_copy(deep=True)
    for field in type(output_states).model_fields:
        setattr(output_states, field, True)
    return output_states


class PestSetup:
    """Orchestrator for PEST++ calibration of MOBIDICpy.

    Creates a complete PEST++ working directory with all necessary files
    and provides methods to execute calibration.

    Args:
        calib_config: Calibration configuration (CalibrationConfig object or path to YAML).
        base_path: Base directory for resolving relative paths in the config.
    """

    def __init__(self, calib_config: CalibrationConfig | str | Path, base_path: Path | None = None):
        if isinstance(calib_config, (str, Path)):
            config_path = Path(calib_config)
            self.calib_config = load_calibration_config(config_path)
            self.base_path = base_path or config_path.parent
        else:
            self.calib_config = calib_config
            self.base_path = base_path or Path.cwd()

        self._pest_exe = PEST_TOOL_MAP[self.calib_config.pest_tool]
        self._working_dir = None
        self._obs_data = None  # Cached observation alignment data
        self._n_obs_per_group = None
        self._sweep_csv_name = None  # Basename of the copied sweep CSV (swp tool)
        self._pst = None  # Last built pyemu.Pst (used by validate_interface)
        self._cycle_schedule = None  # CycleSchedule for sequential da
        self._cycle_table_names = None  # PESTPP-DA option -> cycle table basename
        self._state_spec = None  # StateSpec for joint state-parameter estimation
        self._network = None  # Network the state spec was built against (for the localizer)

    @property
    def working_dir(self) -> Path:
        """Path to the PEST++ working directory."""
        if self._working_dir is None:
            wd = Path(self.calib_config.working_dir)
            if not wd.is_absolute():
                wd = self.base_path / wd
            self._working_dir = wd
        return self._working_dir

    def setup(self) -> Path:
        """Create complete PEST++ working directory with all files.

        Returns:
            Path to the working directory.
        """

        cc = self.calib_config
        wd = self.working_dir

        logger.info(f"Setting up PEST++ working directory: {wd}")

        # Create working directory
        wd.mkdir(parents=True, exist_ok=True)

        # Sequential data assimilation needs cycle machinery the batch path has
        # no concept of. Batch DA (da.mode='batch') deliberately takes the
        # existing path: it is the IES setup run by a different executable.
        if cc.is_sequential_da:
            return self._setup_da(wd)

        # Resolve MOBIDIC config path
        mobidic_config_path = Path(cc.mobidic_config)
        if not mobidic_config_path.is_absolute():
            mobidic_config_path = self.base_path / mobidic_config_path

        # Load MOBIDIC config to get paths
        from mobidic.config import load_config

        mobidic_config = load_config(mobidic_config_path)
        gisdata_path = Path(mobidic_config.paths.gisdata)
        network_path = Path(mobidic_config.paths.network)

        # Determine forcing path
        if mobidic_config.paths.meteoraster:
            forcing_path = Path(mobidic_config.paths.meteoraster)
        elif cc.use_raster_forcing:
            # Will be generated by initial forward run
            forcing_path = wd / "forcing_raster.nc"
        else:
            forcing_path = Path(mobidic_config.paths.meteodata)

        # Determine calibration period (controls which observations are used)
        if cc.calibration_period:
            calib_start = cc.calibration_period.start_date
            calib_end = cc.calibration_period.end_date
        else:
            # Use first/last observation time
            all_obs = []
            for obs_group in cc.observations:
                obs_df = load_observations(obs_group, self.base_path)
                all_obs.append(obs_df)
            all_times = pd.concat(all_obs)["time"]
            calib_start = str(all_times.min())
            calib_end = str(all_times.max())

        # Determine simulation period (the full window the forward model runs,
        # including warm-up). Defaults to calibration period if not specified.
        if cc.simulation_period:
            sim_start = cc.simulation_period.start_date
            sim_end = cc.simulation_period.end_date
        else:
            sim_start = calib_start
            sim_end = calib_end

        # Validate simulation period against forcing time range
        self._validate_calibration_period(forcing_path, mobidic_config, sim_start, sim_end)

        # Run initial forward run if raster forcing needs generation
        if cc.use_raster_forcing and not forcing_path.exists():
            logger.info("Running initial forward run to generate raster forcing...")
            self._run_initial_forward(mobidic_config_path, mobidic_config, forcing_path, sim_start, sim_end)

        # Load observations and align to simulation timesteps
        # Observations are filtered to calibration period, but aligned to the
        # full simulation time grid (so sim_indices are correct)
        self._load_and_align_observations(sim_start, sim_end, calib_start, calib_end, mobidic_config)

        # Check if routing parameters are being calibrated
        routing_keys = {"parameters.routing.wcel", "parameters.routing.Br0", "parameters.routing.NBr"}
        routing_calibrated = any(p.parameter_key in routing_keys for p in cc.parameters)

        # For the sweep tool, copy the parameter sweep CSV into the working dir and
        # validate that its columns cover all calibration parameters.
        if cc.pest_tool == "swp":
            self._prepare_sweep_file(wd)

        # 1. Generate template file
        generate_template_file(cc, wd / "model_input.csv.tpl")

        # 2. Generate initial model_input.csv
        generate_model_input_csv(cc, wd / "model_input.csv")

        # 3. Generate instruction file
        ins_path, obs_names = generate_instruction_file(cc, self._n_obs_per_group, wd / "model_output.csv.ins")

        # 4. Generate forward model runner script
        obs_data_json = json.dumps(self._obs_data)
        observation_reaches = [og.reach_id for og in cc.observations]

        generate_forward_run_script(
            base_config_path=mobidic_config_path,
            gisdata_path=gisdata_path,
            network_path=network_path,
            forcing_path=forcing_path,
            start_date=sim_start,
            end_date=sim_end,
            observation_reaches=observation_reaches,
            obs_data_json=obs_data_json,
            routing_params_calibrated=routing_calibrated,
            output_script_path=wd / "forward_run.py",
        )

        # 5. Build PEST control file via pyemu
        pst = self._build_pst(obs_names, wd)
        self._pst = pst

        # Write .pst file
        pst_path = wd / f"{self.calib_config.case_name}.pst"
        pst_version = self.calib_config.pest_options.get("pst_version", 2)
        pst.write(str(pst_path), version=pst_version)
        logger.info(f"PEST control file written: {pst_path} (version={pst_version})")

        logger.success(f"PEST++ setup complete in {wd}")
        return wd

    def run(self, num_workers: int | None = None, start_manager: bool = True) -> "CalibrationResults":
        """Execute PEST++ with parallel workers.

        Args:
            num_workers: Override number of workers (default: from config or os.cpu_count()).
            start_manager: If True (default), start the manager on this machine (local mode).
                If False, start agents only for cluster mode.

        Returns:
            CalibrationResults with parsed output.
        """
        import pyemu

        cfg = self.calib_config.parallel
        n_workers = num_workers or cfg.num_workers or os.cpu_count()
        is_cluster = cfg.manager_ip is not None and not start_manager

        wd = self.working_dir

        # Create a clean template directory containing only PEST files (no subdirectories).
        # This prevents recursive directory nesting: if worker_dir == worker_root, pyemu
        # copies the growing directory into each new worker, causing exponential growth.
        template_dir = wd / "_template"
        if template_dir.exists():
            shutil.rmtree(template_dir)
        template_dir.mkdir()
        for f in wd.iterdir():
            if f.is_file():
                shutil.copy2(f, template_dir / f.name)

        pst_rel_path = f"{self.calib_config.case_name}.pst"

        if is_cluster:
            logger.info(f"Starting {n_workers} PEST++ agents connecting to {cfg.manager_ip}:{cfg.port}")
            pyemu.os_utils.start_workers(
                worker_dir=str(template_dir),
                exe_rel_path=self._pest_exe,
                pst_rel_path=pst_rel_path,
                num_workers=n_workers,
                worker_root=str(wd),
                port=cfg.port,
                master_dir=None,
                local=cfg.manager_ip,
            )
        else:
            logger.info(f"Starting PEST++ ({self._pest_exe}) with {n_workers} local workers")
            pyemu.os_utils.start_workers(
                worker_dir=str(template_dir),
                exe_rel_path=self._pest_exe,
                pst_rel_path=pst_rel_path,
                num_workers=n_workers,
                worker_root=str(wd),
                port=cfg.port,
                master_dir=str(wd / "master"),
                local=True,
            )

        # Clean up template directory
        shutil.rmtree(template_dir, ignore_errors=True)

        return self.load_results()

    def load_results(self) -> "CalibrationResults":
        """Load results from a completed PEST++ run.

        Returns:
            CalibrationResults parsed from PEST++ output files.
        """
        from mobidic.calibration.results import CalibrationResults

        master_dir = self.working_dir / "master"
        if not master_dir.exists():
            master_dir = self.working_dir

        return CalibrationResults.from_pest_output(
            master_dir=master_dir,
            calib_config=self.calib_config,
        )

    def _load_and_align_observations(
        self,
        sim_start: str,
        sim_end: str,
        calib_start: str,
        calib_end: str,
        mobidic_config,
    ) -> None:
        """Load all observations and align to simulation time grid.

        Observations are filtered to the calibration period but aligned to the
        full simulation time grid so that sim_indices correctly reference
        positions in the forward model output (which runs the full simulation).

        Populates self._obs_data and self._n_obs_per_group.
        """
        dt = mobidic_config.simulation.timestep
        sim_times = pd.date_range(start=sim_start, end=sim_end, freq=f"{int(dt)}s")

        self._obs_data = []
        self._n_obs_per_group = {}

        for obs_group in self.calib_config.observations:
            # Filter observations to calibration period (not simulation period)
            obs_df = load_observations(obs_group, self.base_path, calib_start, calib_end)
            # Align to full simulation time grid
            sim_indices, obs_values, obs_times = align_observations_to_simulation(obs_df, sim_times)

            n_obs = len(sim_indices)
            self._n_obs_per_group[obs_group.name] = n_obs

            obs_info = {
                "name": obs_group.name,
                "reach_id": obs_group.reach_id,
                "n_obs": n_obs,
                "sim_indices": sim_indices.tolist(),
                "obs_values": obs_values.tolist(),
            }

            if obs_group.metrics:
                obs_info["metrics"] = [
                    {"metric": m.metric, "target": m.target, "weight": m.weight} for m in obs_group.metrics
                ]

            self._obs_data.append(obs_info)

            logger.info(f"Observation group '{obs_group.name}': {n_obs} matched observations")

    # ------------------------------------------------------------------
    # Sequential data assimilation (pest_tool='da', da.mode='sequential')
    # ------------------------------------------------------------------

    def _setup_da(self, wd: Path) -> Path:
        """Create a PESTPP-DA working directory for cycle-based assimilation.

        Args:
            wd: Working directory (already created).

        Returns:
            Path to the working directory.
        """
        from mobidic.calibration.da_cycles import (
            CYCLE_INPUT_KEY,
            CYCLE_METADATA_FILE,
            CYCLE_PARAM_NAME,
            assert_cycle_values_differ,
            build_cycle_schedule,
            build_observation_cycle_tables,
            build_parameter_cycle_table,
            write_cycle_metadata,
            write_cycle_tables,
        )
        from mobidic.calibration.da_states import (
            STATE_ID_INPUT_KEY,
            STATE_ID_PAR_NAME,
            STATE_SPEC_FILE,
            STATE_VALUE_WIDTH,
        )
        from mobidic.calibration.forward_model import generate_da_forward_run_script
        from mobidic.calibration.template import ExtraParameter
        from mobidic.config import load_config

        cc = self.calib_config
        da = cc.da

        # 1. Resolve the MOBIDIC config, gisdata, network and forcing
        mobidic_config_path = Path(cc.mobidic_config)
        if not mobidic_config_path.is_absolute():
            mobidic_config_path = self.base_path / mobidic_config_path
        mobidic_config = load_config(mobidic_config_path)

        gisdata_path = Path(mobidic_config.paths.gisdata)
        network_path = Path(mobidic_config.paths.network)
        dt_seconds = int(mobidic_config.simulation.timestep)

        if mobidic_config.paths.meteoraster:
            forcing_path = Path(mobidic_config.paths.meteoraster)
        else:
            if not cc.use_raster_forcing:
                logger.warning(
                    "Sequential DA multiplies the number of forward runs by n_cycles x n_reals x "
                    "noptmax; per-run station interpolation would dominate the run time. "
                    "Forcing use_raster_forcing=true."
                )
                cc.use_raster_forcing = True
            forcing_path = wd / "forcing_raster.nc"

        sim_start = cc.simulation_period.start_date
        sim_end = cc.simulation_period.end_date

        # 2. Build the cycle schedule
        schedule = build_cycle_schedule(sim_start, sim_end, da.cycle_length, dt_seconds)
        self._cycle_schedule = schedule

        # The warm-up must end exactly one timestep before the first cycle, so
        # that no forcing timestep is applied twice or skipped (section 3.2).
        warmup_window = None
        if da.warmup_period is not None:
            warmup_start = pd.Timestamp(da.warmup_period.start_date)
            warmup_end = schedule.starts[0] - pd.Timedelta(seconds=dt_seconds)
            configured_end = pd.Timestamp(da.warmup_period.end_date)
            if configured_end != warmup_end:
                logger.warning(
                    f"Warm-up end adjusted from {configured_end} to {warmup_end} (one timestep before "
                    f"the first cycle) so that no forcing timestep is applied twice or skipped."
                )
            if warmup_start > warmup_end:
                raise ValueError(
                    f"da.warmup_period.start_date ({warmup_start}) leaves no room for a warm-up before "
                    f"the first cycle at {schedule.starts[0]}"
                )
            warmup_window = (warmup_start, warmup_end)

        validate_start = str(warmup_window[0]) if warmup_window else sim_start
        self._validate_calibration_period(forcing_path, mobidic_config, validate_start, str(schedule.ends[-1]))

        if cc.use_raster_forcing and not forcing_path.exists():
            logger.info("Running initial forward run to generate raster forcing...")
            self._run_initial_forward(
                mobidic_config_path, mobidic_config, forcing_path, validate_start, str(schedule.ends[-1])
            )

        # 3. Warm-up run: produces the cycle-0 initial state
        warmup_state_path = None
        warmup_state = None
        simulation = None
        if warmup_window is not None:
            warmup_state, simulation, warmup_state_path = self._run_warmup(
                mobidic_config_path, forcing_path, warmup_window, wd / "warmup_state.nc"
            )

        # 4. State machinery (EnKF only)
        state_dir = None
        if da.states.restart_from == "previous_cycle":
            state_dir = Path(da.states.state_file_dir)
            if not state_dir.is_absolute():
                state_dir = wd / state_dir
            state_dir.mkdir(parents=True, exist_ok=True)
            self._check_state_dir_writable(state_dir)

        # 4b. Assimilation space (formulation 2). Built from a deterministic
        # reference run, so the bounds cover the flows the model actually produces.
        state_spec_path = None
        if da.states.estimate:
            self._state_spec = self._build_state_spec(
                mobidic_config_path, forcing_path, schedule, warmup_state, simulation
            )
            state_spec_path = self._state_spec.to_json(wd / STATE_SPEC_FILE)
            logger.info(f"Wrote the state spec: {state_spec_path}")

        # 5. Observations, cycle tables and cycle metadata
        obs_frames = {group.name: load_observations(group, self.base_path) for group in cc.observations}
        obs_table, weight_table, self._n_obs_per_group = build_observation_cycle_tables(
            obs_groups=cc.observations,
            obs_frames=obs_frames,
            schedule=schedule,
            assimilate=da.assimilate,
            forecast_cycles=da.forecast_cycles,
        )
        par_table = build_parameter_cycle_table(schedule)
        assert_cycle_values_differ(par_table)

        self._cycle_table_names = write_cycle_tables(wd, obs_table, weight_table, par_table)
        cycles_csv = write_cycle_metadata(schedule, wd / CYCLE_METADATA_FILE)

        # The forward model only needs each group's target reach; observation
        # values and weights are supplied per cycle by the cycle tables.
        self._obs_data = [{"name": g.name, "reach_id": g.reach_id} for g in cc.observations]

        # 6. Template / model_input.csv with the reserved data-assimilation rows
        extra_parameters = [ExtraParameter(CYCLE_INPUT_KEY, CYCLE_PARAM_NAME, 0.0, width=12)]
        if da.states.restart_from == "previous_cycle":
            extra_parameters.append(ExtraParameter(STATE_ID_INPUT_KEY, STATE_ID_PAR_NAME, -1.0, width=20))
        if self._state_spec is not None:
            spec = self._state_spec
            extra_parameters.extend(
                ExtraParameter(key, name, float(value), width=STATE_VALUE_WIDTH)
                for key, name, value in zip(spec.input_keys, spec.par_names, spec.initial)
            )

        generate_template_file(cc, wd / "model_input.csv.tpl", extra_parameters=extra_parameters)
        generate_model_input_csv(cc, wd / "model_input.csv", extra_parameters=extra_parameters)

        # 7. Instruction file: within-cycle slots, then the state observations
        extra_obs_names = self._da_state_obs_names()
        ins_path, obs_names = generate_instruction_file(
            cc, self._n_obs_per_group, wd / "model_output.csv.ins", extra_obs_names=extra_obs_names
        )
        logger.debug(f"Instruction file: {ins_path}")

        # 8. Forward run script
        routing_keys = {"parameters.routing.wcel", "parameters.routing.Br0", "parameters.routing.NBr"}
        routing_calibrated = any(p.parameter_key in routing_keys for p in cc.parameters)

        generate_da_forward_run_script(
            base_config_path=mobidic_config_path,
            gisdata_path=gisdata_path,
            network_path=network_path,
            forcing_path=forcing_path,
            cycles_csv=cycles_csv,
            obs_data_json=json.dumps(self._obs_data),
            restart_from=da.states.restart_from,
            routing_params_calibrated=routing_calibrated,
            output_script_path=wd / "forward_run.py",
            warmup_state_path=warmup_state_path,
            state_file_dir=state_dir,
            keep_cycles=da.states.keep_cycles,
            state_spec_path=state_spec_path,
        )

        # 9. Control file
        pst = self._build_da_pst(obs_names, obs_table, weight_table)
        self._pst = pst

        # 10. Cycle-0 prior. Only needed with estimated states: PEST++ draws its
        # own prior from the parameter bounds, and a state bounded to cover a
        # flood peak would be drawn essentially uniformly over that whole range.
        if self._state_spec is not None:
            self._write_prior_parameter_ensemble(pst, wd)

        pst_path = wd / f"{cc.case_name}.pst"
        pst.write(str(pst_path), version=2)
        logger.info(f"PEST control file written: {pst_path} (version=2)")

        formulation = "2 (joint state-parameter estimation)" if self._state_spec is not None else "1"
        logger.success(
            f"PESTPP-DA setup complete in {wd}: {schedule.n_cycles} cycles, "
            f"restart_from='{da.states.restart_from}', formulation {formulation}"
        )
        return wd

    def _da_state_obs_names(self) -> list[str]:
        """Observation names of the state parameters, in control-file order.

        The order fixes the tail of both the .ins file and ``model_output.csv``:
        the state identifier first, then the estimated states in spec order.
        """
        from mobidic.calibration.da_states import STATE_ID_OBS_NAME

        if self.calib_config.da.states.restart_from != "previous_cycle":
            return []
        names = [STATE_ID_OBS_NAME]
        if self._state_spec is not None:
            names.extend(self._state_spec.obs_names)
        return names

    def _build_state_spec(self, mobidic_config_path: Path, forcing_path: Path, schedule, warmup_state, simulation):
        """Build the assimilation space for joint state-parameter estimation.

        One block per estimated state variable. Channel discharge is per reach
        and absolute, so its upper bound has to cover the largest value the model
        actually produces at that reach: PESTPP-DA enforces bounds on the state it
        transfers between cycles, so a bound sized from the (low) warm-up flow
        would silently truncate the transfer during a flood rather than merely
        limit the update. A deterministic reference run over the whole
        assimilation period supplies those maxima, and is skipped entirely when
        only soil moisture is estimated: a saturation is bounded by definition.

        Args:
            mobidic_config_path: Path to the base MOBIDIC config.
            forcing_path: Path to the raster forcing.
            schedule: Cycle schedule.
            warmup_state: Final state of the warm-up run.
            simulation: The Simulation the warm-up used (for its network, grids
                and multiplier-adjusted soil capacities).

        Returns:
            A StateSpec.
        """
        import numpy as np

        from mobidic.calibration.da_states import (
            KIND_DISCHARGE,
            KIND_SURFACE_WATER,
            SOIL_KIND_FIELD,
            StateSpec,
            build_discharge_state_spec,
            build_reach_zone_map,
            build_soil_state_spec,
            build_surface_state_spec,
            resolve_estimate_kinds,
            soil_capacities,
            upstream_reaches,
        )

        cc = self.calib_config
        da_states = cc.da.states
        if warmup_state is None or simulation is None:  # pragma: no cover - the config validator forbids it
            raise ValueError("Joint state-parameter estimation requires a warm-up run to build the state prior")

        network = simulation.network
        self._network = network
        kinds = resolve_estimate_kinds(da_states.estimate)

        observed = [g.reach_id for g in cc.observations]
        if da_states.estimate_reaches == "upstream":
            reach_ids = upstream_reaches(network, observed)
            logger.info(
                f"Estimating states on the {len(reach_ids)} reach(es) upstream of {observed}, "
                f"of {len(network)} in the network"
            )
        else:
            reach_ids = None

        from mobidic.calibration.da_states import ZONE_PARAM_GRID, build_zone_parameter_spec

        soil_kinds = [kind for kind in kinds if kind in SOIL_KIND_FIELD]
        param_kinds = [kind for kind in kinds if kind in ZONE_PARAM_GRID]
        zonal = soil_kinds or param_kinds or KIND_SURFACE_WATER in kinds
        zone_map = zone_ids = None
        if zonal:
            zone_map, zone_ids = build_reach_zone_map(
                simulation.gisdata,
                network=network,
                reach_ids=reach_ids,
                min_zone_cells=da_states.min_zone_cells,
            )

        # A reference run is needed only by the states whose bounds are not
        # physical: discharge [m3/s] and surface storage [m]. A saturation is
        # confined to [0, 1] by definition, so a soil-only setup skips it.
        reference_discharge = reference_surface = None
        if KIND_DISCHARGE in kinds or KIND_SURFACE_WATER in kinds:
            reference_discharge, reference_surface = self._run_reference(
                mobidic_config_path,
                forcing_path,
                schedule,
                warmup_state,
                zone_map=zone_map if KIND_SURFACE_WATER in kinds else None,
                zone_ids=zone_ids,
            )

        specs = []

        if KIND_DISCHARGE in kinds:
            specs.append(
                build_discharge_state_spec(
                    network=network,
                    initial_discharge=np.asarray(warmup_state.discharge, dtype=np.float64),
                    reference_discharge=reference_discharge,
                    reach_ids=reach_ids,
                    bound_factor=da_states.bound_factor,
                    state_floor=da_states.state_floor,
                )
            )

        if soil_kinds:
            specs.append(
                build_soil_state_spec(
                    kinds=soil_kinds,
                    state=warmup_state,
                    capacities=soil_capacities(simulation),
                    zone_map=zone_map,
                    zone_ids=zone_ids,
                    saturation_bounds=da_states.saturation_bounds,
                    state_floor=da_states.state_floor,
                )
            )

        if KIND_SURFACE_WATER in kinds:
            specs.append(
                build_surface_state_spec(
                    state=warmup_state,
                    reference_surface=reference_surface,
                    zone_map=zone_map,
                    zone_ids=zone_ids,
                    bound_factor=da_states.bound_factor,
                    state_floor=da_states.state_floor,
                )
            )

        if param_kinds:
            specs.append(
                build_zone_parameter_spec(
                    kinds=param_kinds,
                    zone_map=zone_map,
                    zone_ids=zone_ids,
                    initial={"runoff_fraction": self._default_f0(mobidic_config_path)},
                    bounds={
                        "runoff_fraction": da_states.f0_bounds,
                        "conductivity": da_states.conductivity_bounds,
                    },
                )
            )

        spec = StateSpec.combine(*specs)

        # The ensemble update can only move the state in as many independent
        # directions as it has realizations, however wide the interface is.
        n_reals = int(cc.pest_options.get("ies_num_reals", 50))
        if len(spec) > 20 * n_reals:
            logger.warning(
                f"The assimilation space holds {len(spec)} states ({', '.join(spec.kinds)}) against "
                f"{n_reals} realization(s). The update is strongly rank-deficient; consider raising "
                "ies_num_reals, raising da.states.min_zone_cells, or estimating fewer state variables."
            )
        return spec

    @staticmethod
    def _default_f0(mobidic_config_path: Path) -> float:
        """The f0 the model would use, so the zone parameters start from it.

        Either the explicit ``parameters.soil.f0`` or the timestep-dependent
        default of ``mobidic_sid.m``.
        """
        import numpy as np

        from mobidic.config import load_config
        from mobidic.core import constants as const

        config = load_config(mobidic_config_path)
        if config.parameters.soil.f0 is not None:
            return float(config.parameters.soil.f0)
        dt = float(config.simulation.timestep)
        return float(
            const.F0_CONSTANT * (1 - np.exp(-dt / (24 * 3600) * np.log(const.F0_CONSTANT / (const.F0_CONSTANT - 0.75))))
        )

    def _run_reference(self, mobidic_config_path, forcing_path, schedule, warmup_state, zone_map=None, zone_ids=None):
        """Run the deterministic reference simulation and collect the bounding maxima.

        Runs cycle by cycle rather than in one pass, because the surface store is
        a grid: its per-zone value has to be sampled as the run proceeds. Cycle
        boundaries are exactly the right sampling points — the bound has to cover
        the state *as PESTPP-DA transfers it*, which happens only there. Chained
        cycles reproduce a continuous run exactly (see the anchor test in
        ``test_da_chaining.py``), so this costs nothing in accuracy.

        Args:
            mobidic_config_path: Path to the base MOBIDIC config.
            forcing_path: Path to the raster forcing.
            schedule: Cycle schedule.
            warmup_state: Final state of the warm-up run. **Not modified**: it is
                deep-copied before use, because ``Simulation`` writes its updates
                into the state object it is handed (``results.final_state`` *is*
                that object), and the caller reads ``parval1`` from it afterwards.
            zone_map: Zone map, when surface maxima are wanted.
            zone_ids: Ascending zone ids.

        Returns:
            Tuple ``(discharge_max_per_reach, surface_max_per_zone_or_None)``.
        """
        import copy

        import numpy as np

        from mobidic.calibration.da_states import zone_mean

        logger.info(
            f"Running the deterministic reference simulation over "
            f"[{schedule.starts[0]}, {schedule.ends[-1]}] to bound the state parameters"
        )
        reference, _ = self._build_simulation(mobidic_config_path, forcing_path)
        reference.set_initial_state(state=copy.deepcopy(warmup_state))

        discharge_max = None
        surface_max = None
        for cycle in range(schedule.n_cycles):
            results = reference.run(schedule.starts[cycle].to_pydatetime(), schedule.ends[cycle].to_pydatetime())
            peak = np.asarray(results.time_series["discharge"], dtype=np.float64).max(axis=0)
            discharge_max = peak if discharge_max is None else np.maximum(discharge_max, peak)
            if zone_map is not None:
                depth = zone_mean(results.final_state.ws, zone_map, zone_ids)
                surface_max = depth if surface_max is None else np.maximum(surface_max, depth)

        return discharge_max, surface_max

    def _write_prior_parameter_ensemble(self, pst, wd: Path) -> Path:
        """Draw and write the cycle-0 prior parameter ensemble.

        PEST++ derives its own prior spread from the parameter bounds
        (``sigma = (ub - lb) / par_sigma_range``). For a state parameter that is
        wrong by construction: the bounds have to span a flood peak, so the drawn
        prior would place a reach's initial discharge anywhere in that range and
        cycle 0 would start from noise. Drawing here instead lets the state prior
        be a small perturbation of the warm-up value while the bounds stay wide
        enough to hold the transferred state.

        Only cycle 0 is affected; from cycle 1 onwards the state ensemble is
        produced by the model runs themselves.

        Args:
            pst: The built pyemu.Pst.
            wd: Working directory.

        Returns:
            Path to the written ensemble CSV.
        """
        import numpy as np
        import pyemu

        from mobidic.calibration.da_states import KIND_DISCHARGE, KIND_SURFACE_WATER

        cc = self.calib_config
        da_states = cc.da.states
        spec = self._state_spec
        n_reals = int(cc.pest_options.get("ies_num_reals", 50))
        sigma_range = float(cc.pest_options.get("par_sigma_range", 4.0))

        cov = pyemu.Cov.from_parameter_data(pst, sigma_range=sigma_range)
        index = {name: i for i, name in enumerate(cov.row_names)}

        for block in spec.blocks:
            if block.kind == KIND_DISCHARGE:
                # Relative to the warm-up flow, with a floor so a dry reach still
                # gets a non-degenerate prior.
                sigma = np.maximum(da_states.prior_std * block.initial, da_states.prior_std_floor)
                units = "m3/s"
            elif block.is_parameter:
                # Dimensionless and scale-free: a relative spread stays
                # meaningful for the whole event, which is the point of them.
                sigma = da_states.zone_parameter_prior_std * block.initial
                units = "relative"
            elif block.kind == KIND_SURFACE_WATER:
                # Also relative, but the floor is a depth: between storms the
                # surface store is nearly empty, so a floor sized for a discharge
                # would swamp the whole state.
                sigma = np.maximum(da_states.prior_std * block.initial, da_states.surface_prior_std_floor)
                units = "m"
            else:
                # A saturation is a bounded fraction, so an absolute spread means
                # the same thing for a wet zone and a dry one.
                sigma = np.full(len(block), da_states.saturation_prior_std, dtype=np.float64)
                units = "saturation"

            for name, value in zip(block.par_names, sigma):
                row = index.get(name)
                if row is None:  # pragma: no cover - every state parameter is adjustable
                    continue
                cov.x[row, 0] = float(value) ** 2

            logger.info(
                f"Cycle-0 prior on the '{block.kind}' states: sigma {sigma.min():.3g} to "
                f"{sigma.max():.3g} {units} over {len(block)} state(s)"
            )

        ensemble = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov, num_reals=n_reals)
        ensemble.enforce()

        path = wd / "prior_par_en.csv"
        ensemble.to_csv(str(path))
        pst.pestpp_options["ies_par_en"] = path.name

        logger.info(f"Wrote the cycle-0 prior parameter ensemble: {path} ({n_reals} realizations)")
        return path

    @staticmethod
    def _check_state_dir_writable(state_dir: Path) -> None:
        """Fail early if the shared state-file directory cannot be written.

        With ``restart_from: previous_cycle`` the run that reads a state file is
        usually not on the agent that wrote it, so the directory must be shared
        and writable by every agent.
        """
        probe = state_dir / ".write_probe"
        try:
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as exc:
            raise ValueError(
                f"State-file directory {state_dir} is not writable: {exc}. Every PEST++ agent must "
                "be able to read and write it (a shared mount is required for agents on other machines)."
            ) from exc

    def _run_warmup(self, mobidic_config_path: Path, forcing_path: Path, window, output_path: Path):
        """Run the deterministic warm-up and persist its final state.

        The state writer is constructed with **every** state variable enabled,
        ignoring the user's ``output_states`` settings: ``StateWriter`` only
        writes ``flr``/``fld``, discharge and reservoir volumes when their flags
        are on, and ``load_state`` silently zero-fills anything missing, which
        would reintroduce a one-timestep discontinuity at every cycle boundary.

        Args:
            mobidic_config_path: Path to the base MOBIDIC config.
            forcing_path: Path to the raster forcing.
            window: ``(start, end)`` Timestamps of the warm-up.
            output_path: Path of the NetCDF state file to write.

        Returns:
            Tuple of (final SimulationState, the Simulation used).
        """
        from mobidic.io import StateWriter

        start, end = window
        logger.info(f"Running deterministic warm-up over [{start}, {end}]")

        simulation, config = self._build_simulation(mobidic_config_path, forcing_path)
        results = simulation.run(start.to_pydatetime(), end.to_pydatetime())

        output_path = Path(output_path)
        if output_path.exists():
            output_path.unlink()

        n_reservoirs = len(simulation.reservoirs.reservoirs) if simulation.reservoirs is not None else 0
        with StateWriter(
            output_path=output_path,
            grid_metadata=simulation.gisdata.metadata,
            network_size=len(simulation.network),
            output_states=_all_states_enabled(config),
            flushing=-1,
            reservoir_size=n_reservoirs,
            add_metadata={"purpose": "PESTPP-DA warm-up state"},
        ) as writer:
            writer.append_state(results.final_state, end.to_pydatetime())

        # StateWriter always chunks, so the file on disk is warmup_state_001.nc.
        # The forward model must be given that path: Simulation.set_initial_state()
        # checks the path exists before delegating to load_state(), so it never
        # reaches load_state's chunk-file fallback.
        written = self._verify_warmup_state(output_path, n_reservoirs > 0)
        logger.success(f"Warm-up state written: {written}")
        return results.final_state, simulation, written

    @staticmethod
    def _verify_warmup_state(output_path: Path, has_reservoirs: bool) -> Path:
        """Check that the warm-up state file holds every variable a cycle needs.

        ``load_state`` zero-fills missing variables without failing, so an
        incomplete warm-up state would silently produce a discontinuity rather
        than an error.
        """
        import xarray as xr

        path = Path(output_path)
        if not path.exists():
            chunk = path.parent / f"{path.stem}_001{path.suffix}"
            if not chunk.exists():
                raise FileNotFoundError(f"Warm-up state file was not written: {path}")
            path = chunk

        required = ["Wc", "Wg", "Ws", "flr", "fld", "discharge", "lateral_inflow"]
        if has_reservoirs:
            required.append("reservoir_volume")

        with xr.open_dataset(path) as ds:
            missing = [name for name in required if name not in ds]
        if missing:
            raise ValueError(
                f"Warm-up state file {path} is missing required variable(s) {missing}. "
                "Every cycle would then start from a partially zeroed state."
            )
        return path

    def _build_simulation(self, mobidic_config_path: Path, forcing_path: Path):
        """Construct a Simulation from the base config and the raster forcing."""
        from mobidic.config import load_config
        from mobidic.core.simulation import Simulation
        from mobidic.preprocessing.io import load_gisdata
        from mobidic.preprocessing.meteo_raster import MeteoRaster

        config = load_config(mobidic_config_path)
        config.output_states_settings.output_states = "None"
        config.output_report.discharge = False
        config.output_report.lateral_inflow = False
        config.output_forcing_data.meteo_data = False

        gisdata = load_gisdata(config.paths.gisdata, config.paths.network)
        forcing = MeteoRaster.from_netcdf(forcing_path)
        return Simulation(gisdata, forcing, config), config

    def _build_da_pst(self, obs_names: list[str], obs_table, weight_table):
        """Build the version-2 PESTPP-DA control file.

        Adds the ``cycle`` column to every interface section and the
        ``state_par_link`` column linking each state observation to the
        parameter it feeds. Both are read by PESTPP-DA only from the external
        CSV sections, which is why version 2 is mandatory.

        Args:
            obs_names: Ordered observation names (matching the .ins file).
            obs_table: Per-cycle observation values (for the activation check).
            weight_table: Per-cycle observation weights.

        Returns:
            pyemu.Pst object.
        """
        from mobidic.calibration.da_cycles import CYCLE_PARAM_NAME
        from mobidic.calibration.da_states import STATE_ID_OBS_NAME, STATE_ID_PAR_NAME

        cc = self.calib_config
        da = cc.da

        # --- extra parameters: the cycle number, and the state parameters ---
        extra_par = [
            {
                "parnme": CYCLE_PARAM_NAME,
                "partrans": "fixed",
                "parval1": 0.0,
                "parlbnd": -1.0,
                "parubnd": 1.0e6,
                "pargp": "da_control",
            }
        ]
        # (obs_name, par_name, group) triples for the state_par_link column
        state_pairs: list[tuple[str, str, str]] = []

        if da.states.restart_from == "previous_cycle":
            # Never 'log', and bounds wide enough that da_enforce_bounds cannot
            # clip the identifier into a different (or non-existent) file.
            extra_par.append(
                {
                    "parnme": STATE_ID_PAR_NAME,
                    "partrans": "fixed",
                    "parval1": -1.0,
                    "parlbnd": -1.0e6,
                    "parubnd": 1.0e6,
                    "pargp": "da_state_id",
                }
            )
            state_pairs.append((STATE_ID_OBS_NAME, STATE_ID_PAR_NAME, "da_state_id"))

        if self._state_spec is not None:
            # Formulation 2: the state parameters are *adjustable*, which is the
            # only difference from formulation 1 as far as the control file goes.
            # 'none', never 'log': a discharge state and a soil saturation are
            # both legitimately zero. One group per state variable, so the phi
            # report separates the discharge states from the soil states.
            for block in self._state_spec.blocks:
                # A conductivity multiplier spans orders of magnitude, so it is
                # estimated in log space; everything else is legitimately zero
                # at some point and must not be.
                transform = "log" if block.multiplier else "none"
                for name, value, lower, upper in zip(block.par_names, block.initial, block.lower, block.upper):
                    extra_par.append(
                        {
                            "parnme": name,
                            "partrans": transform,
                            "parval1": float(value),
                            "parlbnd": float(lower),
                            "parubnd": float(upper),
                            "pargp": block.group,
                        }
                    )
                # Only a dynamic state gets an observation and a state_par_link.
                # A distributed parameter is carried between cycles by PESTPP-DA
                # itself, exactly as ks_factor or wcel are.
                if block.linked:
                    state_pairs.extend((obs, par, block.group) for obs, par in zip(block.obs_names, block.par_names))

        extra_obs = [
            {"obsnme": obs_name, "obsval": 0.0, "weight": 0.0, "obgnme": group} for obs_name, _, group in state_pairs
        ]

        pst = self._build_pst(obs_names, self.working_dir, extra_par=extra_par, extra_obs=extra_obs)

        # Every parameter (static, cycle_num and states alike) applies to every cycle.
        pst.parameter_data["cycle"] = -1

        # State observations carry the link back to the parameter they feed.
        # PESTPP-DA reads this column only from the external version-2 CSV.
        pst.observation_data["state_par_link"] = ""
        pst.observation_data["cycle"] = -1
        for obs_name, par_name, _ in state_pairs:
            pst.observation_data.loc[obs_name, "state_par_link"] = par_name

        # Every observation appearing in a cycle table must carry a non-zero
        # weight in the control file: there the weight is an activation flag,
        # while the real per-cycle weights come from the weight cycle table.
        table_names = [n for n in obs_table.index if n in pst.observation_data.index]
        pst.observation_data.loc[table_names, "weight"] = 1.0
        pst.observation_data.loc[table_names, "obsval"] = 0.0

        pst.model_input_data["cycle"] = -1
        pst.model_output_data["cycle"] = -1

        pst.pestpp_options.update(self._cycle_table_names or {})
        if da.hotstart_cycle is not None:
            pst.pestpp_options["da_hotstart_cycle"] = da.hotstart_cycle
        if da.stop_cycle is not None:
            pst.pestpp_options["da_stop_cycle"] = da.stop_cycle
        # da_use_simulated_states is deliberately left at its default. Setting it
        # to false selects formulation 3/4, where the filter estimates the final
        # states and PESTPP-DA expects final-to-initial linkages in the
        # *parameter* data; its sanity check rejects the option otherwise. With
        # restart_from='warmup' no state is declared at all, so there is nothing
        # for the option to govern.

        if self._state_spec is not None:
            self._setup_localizer(pst)

        logger.info(
            f"Built PESTPP-DA control file: {len(pst.parameter_data)} parameters, "
            f"{len(pst.observation_data)} observations, {weight_table.shape[1]} cycles"
        )
        return pst

    #: PEST++ option names that already carry a localization matrix.
    _LOCALIZER_OPTIONS = ("da_localizer", "ies_localizer", "ies_localizer_filename")

    def _setup_localizer(self, pst) -> Path | None:
        """Write the topology localizer and register it, unless one is configured.

        With more adjustable parameters than realizations the ensemble update is
        rank-deficient, and the filter finds correlations between a gauge and
        distant states that are artefacts of the sample size. Restricting each
        gauge to its own upstream reaches is the structural part of the remedy.

        Args:
            pst: The built pyemu.Pst.

        Returns:
            Path to the written matrix, or None when no localizer is used.
        """
        from mobidic.calibration.da_states import build_upstream_localizer

        cc = self.calib_config
        if any(key in pst.pestpp_options for key in self._LOCALIZER_OPTIONS):
            logger.info("A localizer is already set in pest_options; not writing one")
            return None

        n_reals = int(cc.pest_options.get("ies_num_reals", 50))
        if cc.da.states.localizer == "none":
            logger.warning(
                f"Joint state-parameter estimation with {len(pst.adj_par_names)} adjustable parameter(s) "
                f"({len(self._state_spec)} of them states), {n_reals} realization(s) and "
                "da.states.localizer='none'. The ensemble update will be rank-deficient, so expect "
                "spurious correlations between the gauge(s) and distant reaches."
            )
            return None

        # Every adjustable parameter must be a column: PESTPP-DA treats one that
        # is missing from the localizer as fixed and stops adjusting it.
        state_names = set(self._state_spec.par_names)
        global_par_names = [p for p in pst.adj_par_names if p not in state_names]

        matrix = build_upstream_localizer(
            network=self._network,
            obs_reaches={g.name: g.reach_id for g in cc.observations},
            spec=self._state_spec,
            global_par_names=global_par_names,
        )

        # PEST ASCII matrix, as pyemu writes it. Mat::from_file() dispatches on
        # the extension and accepts only JCB/JCO (binary), MAT/COV (ASCII) and
        # CSV; anything else raises.
        import pyemu

        path = self.working_dir / "loc.mat"
        pyemu.Matrix.from_dataframe(matrix).to_ascii(str(path))
        pst.pestpp_options["da_localizer"] = path.name
        logger.info(f"Wrote the upstream localizer: {path} ({matrix.shape[0]} row(s) x {matrix.shape[1]} column(s))")
        return path

    def validate_interface(self) -> bool:
        """Ask PESTPP-DA to parse the cycle interface and exit.

        Runs the executable once with ``debug_parse_only = true``, which
        validates every cycle, template and instruction file without running a
        single forward model. This is by far the cheapest way to catch
        cycle/tpl/ins mismatches.

        Returns:
            True if PESTPP-DA parsed the interface successfully.

        Raises:
            RuntimeError: If ``setup()`` has not been run yet.
        """
        import subprocess

        if self._pst is None:
            raise RuntimeError("validate_interface() requires setup() to have been run first")

        wd = self.working_dir
        case = f"{self.calib_config.case_name}_parse_check"
        self._pst.pestpp_options["debug_parse_only"] = True
        try:
            self._pst.write(str(wd / f"{case}.pst"), version=2)
        finally:
            self._pst.pestpp_options.pop("debug_parse_only", None)

        logger.info(f"Validating the PEST interface: {self._pest_exe} {case}.pst")
        completed = subprocess.run(
            [self._pest_exe, f"{case}.pst"],
            cwd=str(wd),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            logger.error(completed.stdout[-4000:])
            logger.error(completed.stderr[-4000:])
            logger.error(f"{self._pest_exe} failed to parse the interface (exit code {completed.returncode})")
            return False

        logger.success("PESTPP-DA parsed the cycle interface successfully")
        return True

    def _prepare_sweep_file(self, wd: Path) -> None:
        """Copy the parameter sweep CSV into the working dir and validate its columns.

        The sweep CSV (an ``.par.csv`` produced by pestpp-ies) must contain one
        column per calibration parameter, since pestpp-swp runs the forward model
        once per row using those columns. PEST lowercases parameter names, so the
        column check is case-insensitive.

        Populates ``self._sweep_csv_name`` with the copied file's basename.

        Raises:
            FileNotFoundError: If the sweep CSV does not exist.
            ValueError: If any calibration parameter has no matching column.
        """
        cc = self.calib_config

        sweep_src = Path(cc.pest_options["sweep_parameter_csv_file"])
        if not sweep_src.is_absolute():
            sweep_src = self.base_path / sweep_src
        if not sweep_src.exists():
            raise FileNotFoundError(f"sweep_parameter_csv_file not found: {sweep_src}")

        # Validate that every parameter name appears as a column (case-insensitive)
        columns = pd.read_csv(sweep_src, nrows=0).columns
        column_set = {c.lower() for c in columns}
        missing = [p.name for p in cc.parameters if p.name.lower() not in column_set]
        if missing:
            raise ValueError(
                f"Sweep CSV '{sweep_src}' is missing columns for calibration "
                f"parameter(s): {missing}. Available columns: {list(columns)}"
            )

        shutil.copy2(sweep_src, wd / sweep_src.name)
        self._sweep_csv_name = sweep_src.name
        logger.info(f"Copied parameter sweep CSV to working dir: {sweep_src.name}")

    def _build_pst(
        self,
        obs_names: list[str],
        wd: Path,
        extra_par: list[dict] | None = None,
        extra_obs: list[dict] | None = None,
    ):
        """Build the PEST control file (.pst) using pyemu.

        Args:
            obs_names: List of all observation names.
            wd: Working directory.
            extra_par: Additional parameter rows appended after the calibration
                parameters (data assimilation: ``cycle_num``, state parameters).
            extra_obs: Additional observation rows appended after the
                time-series observations (data assimilation: state observations).

        Returns:
            pyemu.Pst object.
        """
        import pyemu

        cc = self.calib_config

        # Build parameter data
        par_data = []
        par_groups = set()
        for p in cc.parameters:
            par_groups.add(p.par_group)
            par_data.append(
                {
                    "parnme": p.name,
                    "partrans": p.transform if p.transform != "fixed" else "fixed",
                    "parchglim": "factor",
                    "parval1": p.initial_value,
                    "parlbnd": p.lower_bound,
                    "parubnd": p.upper_bound,
                    "pargp": p.par_group,
                    "scale": p.scale,
                    "offset": p.offset,
                    "dercom": 1,
                }
            )
        for extra in extra_par or []:
            par_groups.add(extra["pargp"])
            par_data.append(
                {
                    "parnme": extra["parnme"],
                    "partrans": extra["partrans"],
                    "parchglim": extra.get("parchglim", "relative"),
                    "parval1": extra["parval1"],
                    "parlbnd": extra["parlbnd"],
                    "parubnd": extra["parubnd"],
                    "pargp": extra["pargp"],
                    "scale": extra.get("scale", 1.0),
                    "offset": extra.get("offset", 0.0),
                    "dercom": 1,
                }
            )
        par_df = pd.DataFrame(par_data)
        par_df.index = par_df["parnme"]

        # Build observation data
        obs_data = []
        obs_name_set = set()

        for obs_group in cc.observations:
            n_obs = self._n_obs_per_group.get(obs_group.name, 0)
            obs_info = next((d for d in self._obs_data if d["name"] == obs_group.name), None)
            obs_values = obs_info.get("obs_values", []) if obs_info else []

            # Time-series observations. The weight encodes the assumed
            # observation error, which may depend on the observed value itself.
            group_weights = observation_weights(obs_values, obs_group) if len(obs_values) else []
            for i in range(n_obs):
                obs_name = f"{obs_group.name}_{i:04d}"
                obs_val = obs_values[i] if i < len(obs_values) else 0.0
                weight = float(group_weights[i]) if i < len(group_weights) else obs_group.weight
                obs_data.append(
                    {
                        "obsnme": obs_name,
                        "obsval": obs_val,
                        "weight": weight,
                        "obgnme": obs_group.name,
                    }
                )
                obs_name_set.add(obs_name)

            # Metric pseudo-observations
            if obs_group.metrics:
                for mc in obs_group.metrics:
                    obs_name = f"{obs_group.name}_{mc.metric}"
                    obs_data.append(
                        {
                            "obsnme": obs_name,
                            "obsval": mc.target,
                            "weight": mc.weight,
                            "obgnme": f"{obs_group.name}_metrics",
                        }
                    )
                    obs_name_set.add(obs_name)

        for extra in extra_obs or []:
            obs_data.append(
                {
                    "obsnme": extra["obsnme"],
                    "obsval": extra.get("obsval", 0.0),
                    "weight": extra.get("weight", 0.0),
                    "obgnme": extra["obgnme"],
                }
            )
            obs_name_set.add(extra["obsnme"])

        obs_df = pd.DataFrame(obs_data)
        obs_df.index = obs_df["obsnme"]

        # Create Pst object
        pst = pyemu.Pst.from_par_obs_names(
            par_names=par_df.index.tolist(),
            obs_names=obs_df.index.tolist(),
        )

        # Update parameter data
        pst.parameter_data.loc[par_df.index, "partrans"] = par_df["partrans"]
        pst.parameter_data.loc[par_df.index, "parchglim"] = par_df["parchglim"]
        pst.parameter_data.loc[par_df.index, "parval1"] = par_df["parval1"]
        pst.parameter_data.loc[par_df.index, "parlbnd"] = par_df["parlbnd"]
        pst.parameter_data.loc[par_df.index, "parubnd"] = par_df["parubnd"]
        pst.parameter_data.loc[par_df.index, "pargp"] = par_df["pargp"]
        pst.parameter_data.loc[par_df.index, "scale"] = par_df["scale"]
        pst.parameter_data.loc[par_df.index, "offset"] = par_df["offset"]
        pst.parameter_data.loc[par_df.index, "dercom"] = par_df["dercom"]

        # Update observation data
        pst.observation_data.loc[obs_df.index, "obsval"] = obs_df["obsval"]
        pst.observation_data.loc[obs_df.index, "weight"] = obs_df["weight"]
        pst.observation_data.loc[obs_df.index, "obgnme"] = obs_df["obgnme"]

        # Set model command
        pst.model_command = ["python forward_run.py"]

        # Set template and instruction file pairs
        pst.model_input_data = pd.DataFrame(
            {
                "pest_file": ["model_input.csv.tpl"],
                "model_file": ["model_input.csv"],
            }
        )
        pst.model_output_data = pd.DataFrame(
            {
                "pest_file": ["model_output.csv.ins"],
                "model_file": ["model_output.csv"],
            }
        )

        # Set control data from pest_options (with defaults)
        control_data_keys = {"noptmax", "relparmax", "facparmax"}
        pst.control_data.noptmax = cc.pest_options.get("noptmax", 20)
        pst.control_data.relparmax = cc.pest_options.get("relparmax", 10.0)
        pst.control_data.facparmax = cc.pest_options.get("facparmax", 10.0)
        # Count the adjustable rows rather than cc.parameters: with joint
        # state-parameter estimation most of them are state parameters, and a
        # maxsing of 4 would truncate the update to the first four singular values.
        pst.svd_data.maxsing = max(len(pst.adj_par_names), 1)

        # Apply additional PEST++ options (skip keys already handled as control_data)
        for key, value in cc.pest_options.items():
            if key in control_data_keys or key == "pst_version":
                continue
            pst.pestpp_options[key] = value

        # For the sweep tool, point PEST++ at the (copied) parameter sweep CSV
        if cc.pest_tool == "swp":
            pst.pestpp_options["sweep_parameter_csv_file"] = self._sweep_csv_name

        logger.info(
            f"Built PEST control file: {len(cc.parameters)} parameters, "
            f"{len(obs_data)} observations, tool={self._pest_exe}"
        )
        return pst

    def _validate_calibration_period(self, forcing_path: Path, mobidic_config, start_date: str, end_date: str) -> None:
        """Validate that the simulation period is contained within the forcing time range.

        The simulation period (which includes any warm-up before the calibration
        window) must fit within the available forcing data so that PEST++ forward
        runs can actually be executed.

        Args:
            forcing_path: Path to forcing NetCDF file.
            mobidic_config: Loaded MOBIDICConfig object.
            start_date: Simulation start date string.
            end_date: Simulation end date string.

        Raises:
            ValueError: If the simulation period extends outside the forcing data range.
        """
        sim_start = pd.Timestamp(start_date)
        sim_end = pd.Timestamp(end_date)

        # Determine the forcing time range
        forcing_start = None
        forcing_end = None

        if forcing_path.exists():
            # Load just enough to get the time range (MeteoRaster or MeteoData)
            try:
                from mobidic.preprocessing.meteo_raster import MeteoRaster

                forcing = MeteoRaster.from_netcdf(forcing_path, preload=False)
                forcing_start = forcing.start_date
                forcing_end = forcing.end_date
            except Exception:
                try:
                    from mobidic.preprocessing.meteo_preprocessing import MeteoData

                    forcing = MeteoData.from_netcdf(forcing_path)
                    forcing_start = forcing.start_date
                    forcing_end = forcing.end_date
                except Exception:
                    logger.warning(
                        f"Could not determine forcing time range from {forcing_path}. "
                        "Skipping calibration period validation."
                    )
                    return
        elif not self.calib_config.use_raster_forcing:
            # Forcing file should exist if we're not generating it
            logger.warning(f"Forcing file not found: {forcing_path}. Skipping calibration period validation.")
            return
        else:
            # Raster forcing will be generated — skip validation (it will use the simulation period)
            return

        if forcing_start is None or forcing_end is None:
            return

        # Check: simulation start_date >= forcing start_date
        if sim_start < forcing_start:
            raise ValueError(
                f"Simulation start_date ({start_date}) is before the forcing data start "
                f"({forcing_start}). The simulation period must be contained within the "
                f"available forcing data period."
            )

        # Check: simulation end_date <= forcing end_date
        if sim_end > forcing_end:
            raise ValueError(
                f"Simulation end_date ({end_date}) is after the forcing data end "
                f"({forcing_end}). The simulation period must be contained within the "
                f"available forcing data period."
            )

        logger.info(
            f"Simulation period validated: [{start_date}, {end_date}] is within "
            f"forcing period [{forcing_start}, {forcing_end}]"
        )

    def _run_initial_forward(self, config_path, config, forcing_path, start_date, end_date):
        """Run initial forward simulation to generate raster forcing."""
        from mobidic.core.simulation import Simulation
        from mobidic.preprocessing.io import load_gisdata
        from mobidic.preprocessing.meteo_preprocessing import MeteoData

        gisdata = load_gisdata(config.paths.gisdata, config.paths.network)

        # Enable meteo output so interpolated forcing is saved as raster
        config.output_forcing_data.meteo_data = True

        forcing = MeteoData.from_netcdf(config.paths.meteodata)
        sim = Simulation(gisdata, forcing, config)
        sim.run(start_date, end_date)

        # Copy the generated meteo_forcing.nc to the expected forcing_path
        output_dir = Path(config.paths.output)
        meteo_output = output_dir / "meteo_forcing.nc"
        if meteo_output.exists():
            shutil.copy2(meteo_output, forcing_path)
            logger.info(f"Rasterized forcing saved to {forcing_path}")
        else:
            raise FileNotFoundError(f"Expected rasterized forcing at {meteo_output} but not found")
