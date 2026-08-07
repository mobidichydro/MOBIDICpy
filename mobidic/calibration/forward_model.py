"""PEST++ forward model wrapper for MOBIDICpy.

This script is invoked by PEST++ for each model run. It:
1. Reads parameter values from model_input.csv (written by PEST++ via .tpl)
2. Applies parameter updates to the base MOBIDIC YAML config
3. Runs the simulation
4. Writes simulated observations to model_output.csv (read by PEST++ via .ins)

Can be called as: python -m mobidic.calibration.forward_model --args ...
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mobidic.calibration.parameter_mapping import apply_optimal_parameters, read_model_input_csv


def _recalculate_routing_parameters(network, config):
    """Recalculate routing parameters that depend on calibrated values.

    When wcel, Br0, or NBr are calibrated, the derived routing parameters
    (lag_time_s, width_m) in the network must be updated.

    Args:
        network: GeoDataFrame with river network.
        config: MOBIDICConfig with updated parameter values.

    Returns:
        Updated network GeoDataFrame.
    """
    routing = config.parameters.routing
    wcel = routing.wcel
    Br0 = routing.Br0
    NBr = routing.NBr
    n_Man = routing.n_Man

    network = network.copy()
    network["width_m"] = Br0 * (network["strahler_order"] ** NBr)
    network["lag_time_s"] = network["length_m"] / wcel
    network["n_manning"] = n_Man

    return network


#: Grids baked into ``gisdata.nc`` that are superseded by a calibrated scalar.
_PARAM_KEY_TO_GRID = {
    "parameters.soil.alpha": "alpha",
    "parameters.soil.beta": "beta",
    "parameters.soil.gamma": "gamma",
    "parameters.soil.kappa": "kappa",
    "parameters.soil.ks": "ks",
    "parameters.soil.kf": "kf",
}


def prepare_simulation(
    base_config_path: Path,
    param_updates: dict[str, float],
    gisdata_path: Path | None = None,
    network_path: Path | None = None,
    gisdata=None,
    routing_params_calibrated: bool = False,
    save_config_to: Path | None = None,
):
    """Build the config and pre-processed data for one parameter realization.

    Applies a set of calibrated values to the base MOBIDIC config, drops the
    gisdata grids those values supersede, recalculates the routing parameters if
    needed, and disables the outputs a calibration run does not want.

    Used by the forward models, and by post-processing that needs to re-run the
    model for a specific realization (for example to issue a forecast from an
    analysed state).

    Args:
        base_config_path: Path to the base MOBIDIC YAML config.
        param_updates: Mapping of dot-notation config path to value.
        gisdata_path: Path to gisdata.nc. Ignored when ``gisdata`` is given.
        network_path: Path to network.parquet. Ignored when ``gisdata`` is given.
        gisdata: Pre-loaded GISData to reuse instead of reading from disk. It is
            left untouched: the returned object is a shallow copy, so a caller
            can reuse the same GISData for many realizations.
        routing_params_calibrated: If True, recalculate routing params from config.
        save_config_to: Optional path to write the effective config to, for
            inspection.

    Returns:
        Tuple of (config, gisdata).
    """
    from mobidic.config import load_config, save_config
    from mobidic.preprocessing.io import load_gisdata

    config = load_config(base_config_path)
    apply_optimal_parameters(config, param_updates)

    if save_config_to is not None:
        save_config(config, save_config_to)

    if gisdata is None:
        if gisdata_path is None or network_path is None:
            raise ValueError("Either 'gisdata' or both 'gisdata_path' and 'network_path' must be provided")
        gisdata = load_gisdata(gisdata_path, network_path)
    else:
        # Work on a private view: the caller's object must survive being reused
        # for the next realization. The grids dict loses entries below and
        # 'network' is rebound; the arrays themselves are never written to.
        gisdata = copy.copy(gisdata)
        gisdata.grids = dict(gisdata.grids)

    # Remove gisdata grids for calibrated parameters so the simulation
    # uses the scalar config values (which PEST++ perturbs) instead of
    # the fixed raster grids baked into gisdata.nc.
    for param_key in param_updates:
        grid_name = _PARAM_KEY_TO_GRID.get(param_key)
        if grid_name and grid_name in gisdata.grids:
            del gisdata.grids[grid_name]
            logger.debug(f"Removed gisdata grid '{grid_name}' - using calibrated scalar value")

    if routing_params_calibrated:
        gisdata.network = _recalculate_routing_parameters(gisdata.network, config)
        logger.debug("Recalculated routing parameters from calibrated config")

    # Disable unnecessary outputs for calibration runs
    config.output_states_settings.output_states = "None"
    config.output_report.discharge = False
    config.output_report.lateral_inflow = False
    config.output_forcing_data.meteo_data = False

    return config, gisdata


def _prepare_simulation_inputs(
    base_config_path: Path,
    input_path: Path,
    gisdata_path: Path,
    network_path: Path,
    routing_params_calibrated: bool,
    param_updates: dict[str, float],
):
    """Load the config and pre-processed data with the calibrated values applied.

    Shared by the batch and data-assimilation forward models.

    Returns:
        Tuple of (config, gisdata).
    """
    return prepare_simulation(
        base_config_path=base_config_path,
        param_updates=param_updates,
        gisdata_path=gisdata_path,
        network_path=network_path,
        routing_params_calibrated=routing_params_calibrated,
        # Persist the effective config for inspection/debugging (paths absolute).
        save_config_to=input_path.parent / "_modified_config.yaml",
    )


def run_forward_model(
    base_config_path: Path,
    input_path: Path,
    output_path: Path,
    gisdata_path: Path,
    network_path: Path,
    forcing_path: Path,
    start_date: str,
    end_date: str,
    observation_reaches: list[int],
    obs_data_json: str | None = None,
    routing_params_calibrated: bool = False,
) -> None:
    """Execute a single PEST++ forward model run.

    Args:
        base_config_path: Path to the base MOBIDIC YAML config.
        input_path: Path to model_input.csv with PEST++-substituted parameters.
        output_path: Path to write model_output.csv for PEST++ to read.
        gisdata_path: Path to pre-processed gisdata.nc.
        network_path: Path to pre-processed network.parquet.
        forcing_path: Path to forcing data (NetCDF, either station or raster).
        start_date: Simulation start date.
        end_date: Simulation end date.
        observation_reaches: List of reach IDs to extract discharge.
        obs_data_json: JSON string with observation metadata for metric computation.
        routing_params_calibrated: If True, recalculate routing params from config.
    """
    from mobidic.core.simulation import Simulation
    from mobidic.preprocessing.meteo_raster import MeteoRaster

    # Step 1: Read PEST++-substituted parameter values
    param_updates = read_model_input_csv(input_path)
    logger.info(f"Read {len(param_updates)} parameters from {input_path}")

    # Steps 2-4: Apply parameters to the config and the pre-processed data
    config, gisdata = _prepare_simulation_inputs(
        base_config_path=base_config_path,
        input_path=input_path,
        gisdata_path=gisdata_path,
        network_path=network_path,
        routing_params_calibrated=routing_params_calibrated,
        param_updates=param_updates,
    )

    # Step 5: Load forcing and run simulation
    forcing = MeteoRaster.from_netcdf(forcing_path)

    sim = Simulation(gisdata, forcing, config)
    results = sim.run(start_date, end_date)

    # Step 6: Extract simulated discharge at observation reaches
    discharge_ts = results.time_series["discharge"]  # shape: (n_times, n_reaches)
    sim_times = results.time_series["time"]

    _write_model_output(
        discharge_ts=discharge_ts,
        sim_times=sim_times,
        observation_reaches=observation_reaches,
        network=gisdata.network,
        output_path=output_path,
        obs_data_json=obs_data_json,
    )


def run_da_forward_model(
    base_config_path: Path,
    input_path: Path,
    output_path: Path,
    gisdata_path: Path,
    network_path: Path,
    forcing_path: Path,
    cycles_csv: Path,
    obs_data_json: str,
    restart_from: str,
    routing_params_calibrated: bool = False,
    warmup_state_path: Path | None = None,
    state_file_dir: Path | None = None,
    keep_cycles: int = 2,
    state_spec_path: Path | None = None,
) -> None:
    """Execute a single PESTPP-DA forward run: one assimilation cycle.

    Args:
        base_config_path: Path to the base MOBIDIC YAML config.
        input_path: Path to model_input.csv with PEST++-substituted parameters.
        output_path: Path to write model_output.csv for PEST++ to read.
        gisdata_path: Path to pre-processed gisdata.nc.
        network_path: Path to pre-processed network.parquet.
        forcing_path: Path to raster forcing NetCDF.
        cycles_csv: Path to da_cycles.csv (cycle, start_date, end_date, n_steps).
        obs_data_json: JSON string with observation metadata (group name, reach id).
        restart_from: ``"warmup"`` or ``"previous_cycle"``.
        routing_params_calibrated: If True, recalculate routing params from config.
        warmup_state_path: Path to warmup_state.nc; required for ``"warmup"``.
        state_file_dir: Absolute path to the shared state-file directory
            (``"previous_cycle"`` only).
        keep_cycles: Cycles of state files to retain.
        state_spec_path: Path to da_state_spec.json; present only with joint
            state-parameter estimation.
    """
    from mobidic.calibration.da_cycles import CYCLE_INPUT_KEY, read_cycle_metadata
    from mobidic.calibration.da_states import (
        STATE_ID_INPUT_KEY,
        StateSpec,
        apply_zone_parameters,
        build_state_mask,
        extract_state_vector,
        insert_state_vector,
        new_state_id,
        read_state_file,
        remove_old_state_files,
        soil_capacities,
        state_file_path,
        write_state_file,
    )
    from mobidic.core.simulation import Simulation
    from mobidic.preprocessing.meteo_raster import MeteoRaster

    # Step 1-2: Read the interface and split off the reserved data-assimilation rows
    raw_values = read_model_input_csv(input_path)
    param_updates = {k: v for k, v in raw_values.items() if not _is_reserved_da_key(k)}

    if CYCLE_INPUT_KEY not in raw_values:
        raise ValueError(f"'{CYCLE_INPUT_KEY}' is missing from {input_path}; the cycle number cannot be determined")

    # Step 3: Resolve this cycle's absolute time window
    cycle = int(float(raw_values[CYCLE_INPUT_KEY]))
    cycles = read_cycle_metadata(cycles_csv)
    if cycle not in cycles.index:
        raise ValueError(f"Cycle {cycle} is not present in {cycles_csv} (cycles: {list(cycles.index)})")
    cycle_row = cycles.loc[cycle]
    cycle_start = cycle_row["start_date"]
    cycle_end = cycle_row["end_date"]
    n_steps_per_cycle = int(cycle_row["n_steps"])
    logger.info(f"DA forward run: cycle {cycle} covering [{cycle_start}, {cycle_end}] ({n_steps_per_cycle} steps)")

    # Step 4: Apply the static parameters
    config, gisdata = _prepare_simulation_inputs(
        base_config_path=base_config_path,
        input_path=input_path,
        gisdata_path=gisdata_path,
        network_path=network_path,
        routing_params_calibrated=routing_params_calibrated,
        param_updates=param_updates,
    )

    # Step 5: Build the simulation
    forcing = MeteoRaster.from_netcdf(forcing_path)
    sim = Simulation(gisdata, forcing, config)

    # The assimilation space, when the filter adjusts states as well as parameters.
    # A soil state is a saturation, so it has to be normalised by the capacities
    # *this* realization runs with: Wc_factor/Wg_factor are calibration
    # parameters, so wc0/wg0 differ from realization to realization.
    state_spec = None
    capacities = None
    if state_spec_path is not None:
        state_spec = StateSpec.from_json(state_spec_path)
        state_spec.check_against(gisdata.network)
        state_spec.check_grid((sim.nrows, sim.ncols))
        capacities = soil_capacities(sim)
        analysed_states = _read_state_vector(raw_values, state_spec, input_path)
        # Distributed parameters (f0, ks per zone) are properties, not states:
        # they are written into the simulation's parameter grids, which the main
        # loop reads every timestep. Done before the state insertion so that a
        # setup estimating both behaves the same either way.
        apply_zone_parameters(sim, state_spec, analysed_states)

    # Step 6: Initial state
    state_id = None
    if restart_from == "warmup":
        if warmup_state_path is None:
            raise ValueError("warmup_state_path is required when restart_from='warmup'")
        # The EnKS re-simulates the whole span from the first cycle onwards.
        run_start = pd.Timestamp(cycles.loc[cycles.index.min(), "start_date"])
        sim.set_initial_state(state_file=str(warmup_state_path), time_index=-1)
    elif restart_from == "previous_cycle":
        if state_file_dir is None:
            raise ValueError("state_file_dir is required when restart_from='previous_cycle'")
        run_start = cycle_start
        state_file_dir = Path(state_file_dir)
        mask = build_state_mask(gisdata)

        raw_state_id = raw_values.get(STATE_ID_INPUT_KEY)
        if raw_state_id is None:
            raise ValueError(f"'{STATE_ID_INPUT_KEY}' is missing from {input_path}")
        state_id = int(float(raw_state_id))

        if state_id < 0:
            # Cycle 0: start from the warm-up state, or from the config's initial
            # conditions when no warm-up was configured.
            if warmup_state_path is not None:
                sim.set_initial_state(state_file=str(warmup_state_path), time_index=-1)
            else:
                logger.info("No warm-up state configured; cycle 0 starts from the config initial conditions")
        else:
            previous = state_file_path(state_file_dir, cycle - 1, state_id)
            sim.set_initial_state(state=read_state_file(previous, mask))
    else:
        raise ValueError(f"Unknown restart_from '{restart_from}' (expected 'warmup' or 'previous_cycle')")

    # Step 6b: Insert the analysed state vector into the background state read
    # from the state file. Everything outside the assimilation space keeps its
    # background value, so this is an analysis increment, not a replacement.
    if state_spec is not None and state_spec.linked_blocks:
        if getattr(sim, "state", None) is None:
            raise ValueError(
                "Joint state-parameter estimation has no initial state to correct: no state file "
                "and no warm-up state were loaded. da.states.estimate requires da.warmup_period."
            )
        insert_state_vector(analysed_states, state_spec, sim.state, capacities=capacities)
        logger.info(
            f"Inserted analysed state(s) ({', '.join(b.kind for b in state_spec.linked_blocks)}) into the initial state"
        )

    # Step 7: Run the cycle
    results = sim.run(run_start.to_pydatetime(), cycle_end.to_pydatetime())

    discharge_ts = results.time_series["discharge"]
    # The reported slots are always this cycle's last n_steps_per_cycle steps:
    # identical to the whole run for 'previous_cycle', the tail for 'warmup'.
    slot_offset = len(discharge_ts) - n_steps_per_cycle
    if slot_offset < 0:
        raise ValueError(
            f"Cycle {cycle} produced {len(discharge_ts)} timestep(s), fewer than the "
            f"{n_steps_per_cycle} observation slots it must report"
        )

    # Step 8: Report the observation slots and, when states are carried, the new identifier
    extra_outputs: list[tuple[str, float]] = []
    if restart_from == "previous_cycle":
        from mobidic.calibration.da_states import STATE_ID_OBS_NAME

        new_id = new_state_id(state_file_dir, cycle)
        write_state_file(state_file_path(state_file_dir, cycle, new_id), results.final_state, mask)
        extra_outputs.append((STATE_ID_OBS_NAME, float(new_id)))

        # The forecast state vector, which PESTPP-DA transfers into the next
        # cycle's state parameters through the state_par_link column. The order
        # must match the one the .ins file was generated with.
        if state_spec is not None:
            forecast = extract_state_vector(results.final_state, state_spec, capacities=capacities)
            extra_outputs.extend((name, float(v)) for name, v in zip(state_spec.obs_names, forecast))

        remove_old_state_files(state_file_dir, cycle, keep_cycles)

    _write_model_output(
        discharge_ts=discharge_ts,
        sim_times=results.time_series["time"],
        observation_reaches=[],
        network=gisdata.network,
        output_path=output_path,
        obs_data_json=obs_data_json,
        slot_indices=list(range(slot_offset, slot_offset + n_steps_per_cycle)),
        extra_outputs=extra_outputs,
    )


def _read_state_vector(raw_values: dict[str, float], state_spec, input_path: Path) -> np.ndarray:
    """Collect the analysed state vector from ``model_input.csv``, in interface order.

    Args:
        raw_values: Everything read from model_input.csv.
        state_spec: Assimilation space.
        input_path: Source path, for the error message.

    Returns:
        1D array of length ``len(state_spec)``.

    Raises:
        ValueError: If any expected state row is absent. A missing row would
            otherwise silently leave that reach at its background value while
            PESTPP-DA believed the analysis had been applied.
    """
    keys = state_spec.input_keys
    missing = [k for k in keys if k not in raw_values]
    if missing:
        raise ValueError(
            f"{len(missing)} estimated state row(s) are missing from {input_path} "
            f"(first: {missing[0]}); the template file and the state spec disagree"
        )
    return np.array([raw_values[k] for k in keys], dtype=np.float64)


def _is_reserved_da_key(key: str) -> bool:
    """True for the reserved ``model_input.csv`` rows that are not MOBIDIC config paths."""
    from mobidic.calibration.da_cycles import CYCLE_INPUT_KEY
    from mobidic.calibration.da_states import STATE_ID_INPUT_KEY, is_state_input_key

    return key in (CYCLE_INPUT_KEY, STATE_ID_INPUT_KEY) or is_state_input_key(key)


#: Smallest normal double. Anything below it is subnormal.
_SMALLEST_NORMAL = float(np.finfo(np.float64).tiny)


def _pest_value(value: float) -> str:
    """Format a simulated value for ``model_output.csv``.

    PEST++ refuses to parse a **denormal** (subnormal) double and fails the whole
    run with an ``InstructionFile error``. The linear routing recursion
    ``Q(t+dt) = C3*Q(t) + C4*qL`` decays exponentially towards zero, so a reach
    with no inflow passes through the subnormal range and stays there. Such a
    discharge is numerically zero for every purpose, so flush it.
    """
    value = float(value)
    if abs(value) < _SMALLEST_NORMAL:
        value = 0.0
    return f"{value:.10e}"


def _write_model_output(
    discharge_ts: np.ndarray,
    sim_times: list,
    observation_reaches: list[int],
    network,
    output_path: Path,
    obs_data_json: str | None = None,
    slot_indices: list[int] | None = None,
    extra_outputs: list[tuple[str, float]] | None = None,
) -> None:
    """Write simulated observations to model_output.csv for PEST++.

    Output format: obs_name,value (one observation per line).

    Args:
        discharge_ts: Discharge time series array (n_times, n_reaches).
        sim_times: List of simulation time stamps.
        observation_reaches: List of reach IDs (mobidic_id) to extract.
        network: Network GeoDataFrame for mapping reach IDs to indices.
        output_path: Path to write model_output.csv.
        obs_data_json: JSON with observation data for metric computation.
        slot_indices: Indices into ``discharge_ts`` to report, overriding each
            group's ``sim_indices``. Used by sequential data assimilation, where
            observation names are within-cycle slots rather than absolute times.
        extra_outputs: ``(obs_name, value)`` pairs appended after the time
            series (state observations, in sequential data assimilation).
    """
    lines = ["obs_name,value"]

    # Build reach_id -> column index mapping
    reach_ids = network["mobidic_id"].values
    reach_to_idx = {int(rid): i for i, rid in enumerate(reach_ids)}

    # Parse observation metadata if provided
    obs_data = json.loads(obs_data_json) if obs_data_json else None

    for obs_info in obs_data or []:
        group_name = obs_info["name"]
        reach_id = obs_info["reach_id"]

        if reach_id not in reach_to_idx:
            raise ValueError(f"Reach ID {reach_id} not found in network")

        col_idx = reach_to_idx[reach_id]

        # Get simulated discharge for this reach
        sim_q = discharge_ts[:, col_idx]

        # Write time-series observations
        if slot_indices is not None:
            sim_indices = slot_indices
        else:
            n_obs = obs_info.get("n_obs", len(sim_q))
            sim_indices = obs_info.get("sim_indices", list(range(n_obs)))

        for i, si in enumerate(sim_indices):
            obs_name = f"{group_name}_{i:04d}"
            lines.append(f"{obs_name},{_pest_value(sim_q[si])}")

        # Write metric pseudo-observations if configured
        if "metrics" in obs_info and obs_info["metrics"]:
            from mobidic.calibration.metrics import METRIC_REGISTRY

            # Load observed data for metric computation
            obs_values = np.array(obs_info["obs_values"])
            sim_values = sim_q[np.array(sim_indices)]

            for metric_info in obs_info["metrics"]:
                metric_name = metric_info["metric"]
                func, _ = METRIC_REGISTRY[metric_name]
                metric_value = func(sim_values, obs_values)
                obs_name = f"{group_name}_{metric_name}"
                lines.append(f"{obs_name},{_pest_value(metric_value)}")

    # Fallback: if no obs_data provided, write all discharge for all observation reaches
    if obs_data is None:
        for reach_id in observation_reaches:
            if reach_id not in reach_to_idx:
                raise ValueError(f"Reach ID {reach_id} not found in network")
            col_idx = reach_to_idx[reach_id]
            sim_q = discharge_ts[:, col_idx]
            for i in range(len(sim_q)):
                obs_name = f"reach_{reach_id}_{i:04d}"
                lines.append(f"{obs_name},{_pest_value(sim_q[i])}")

    # State observations (sequential data assimilation). The identifier is
    # written without an exponent so it survives the round trip as an integer.
    for obs_name, value in extra_outputs or []:
        lines.append(f"{obs_name},{_pest_value(value)}")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {len(lines) - 1} observations to {output_path}")


def generate_forward_run_script(
    base_config_path: Path,
    gisdata_path: Path,
    network_path: Path,
    forcing_path: Path,
    start_date: str,
    end_date: str,
    observation_reaches: list[int],
    obs_data_json: str,
    routing_params_calibrated: bool,
    output_script_path: Path,
) -> Path:
    """Generate the forward_run.py script that PEST++ will execute.

    Args:
        base_config_path: Path to the base MOBIDIC YAML config.
        gisdata_path: Path to gisdata.nc.
        network_path: Path to network.parquet.
        forcing_path: Path to forcing NetCDF.
        start_date: Calibration period start date.
        end_date: Calibration period end date.
        observation_reaches: List of reach IDs to extract.
        obs_data_json: JSON string with observation metadata.
        routing_params_calibrated: Whether routing params are being calibrated.
        output_script_path: Path to write the forward_run.py script.

    Returns:
        Path to the generated script.
    """

    # Escape backslashes in paths for Windows compatibility
    def _path_str(p):
        return str(p).replace("\\", "/")

    script = f'''"""PEST++ forward model runner (auto-generated)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path({repr(_path_str(base_config_path.parent.parent))}).resolve()))

from mobidic.calibration.forward_model import run_forward_model

run_forward_model(
    base_config_path=Path({repr(_path_str(base_config_path))}),
    input_path=Path("model_input.csv"),
    output_path=Path("model_output.csv"),
    gisdata_path=Path({repr(_path_str(gisdata_path))}),
    network_path=Path({repr(_path_str(network_path))}),
    forcing_path=Path({repr(_path_str(forcing_path))}),
    start_date={repr(start_date)},
    end_date={repr(end_date)},
    observation_reaches={observation_reaches!r},
    obs_data_json={repr(obs_data_json)},
    routing_params_calibrated={routing_params_calibrated!r},
)
'''
    output_script_path = Path(output_script_path)
    output_script_path.write_text(script, encoding="utf-8")
    logger.info(f"Generated forward_run.py: {output_script_path}")
    return output_script_path


def generate_da_forward_run_script(
    base_config_path: Path,
    gisdata_path: Path,
    network_path: Path,
    forcing_path: Path,
    cycles_csv: Path,
    obs_data_json: str,
    restart_from: str,
    routing_params_calibrated: bool,
    output_script_path: Path,
    warmup_state_path: Path | None = None,
    state_file_dir: Path | None = None,
    keep_cycles: int = 2,
    state_spec_path: Path | None = None,
) -> Path:
    """Generate the forward_run.py script PESTPP-DA executes for each cycle.

    Every path is embedded as an absolute path: the script runs inside each
    agent's own run directory, so a relative path would resolve elsewhere.

    Args:
        base_config_path: Path to the base MOBIDIC YAML config.
        gisdata_path: Path to gisdata.nc.
        network_path: Path to network.parquet.
        forcing_path: Path to the raster forcing NetCDF.
        cycles_csv: Path to da_cycles.csv.
        obs_data_json: JSON string with observation metadata.
        restart_from: ``"warmup"`` or ``"previous_cycle"``.
        routing_params_calibrated: Whether routing params are being calibrated.
        output_script_path: Path to write the forward_run.py script.
        warmup_state_path: Path to warmup_state.nc, if a warm-up was run.
        state_file_dir: Shared state-file directory (``"previous_cycle"`` only).
        keep_cycles: Cycles of state files to retain.
        state_spec_path: Path to da_state_spec.json, with joint state-parameter
            estimation.

    Returns:
        Path to the generated script.
    """

    # Every path is resolved: the script executes inside each agent's own run
    # directory, so a relative path taken from the caller's working directory
    # would point somewhere else (or nowhere) by the time an agent runs it.
    def _path_str(p):
        return str(Path(p).resolve()).replace("\\", "/")

    def _opt_path(p):
        return "None" if p is None else f"Path({repr(_path_str(p))})"

    script = f'''"""PESTPP-DA forward model runner (auto-generated)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path({repr(_path_str(base_config_path.parent.parent))}).resolve()))

from mobidic.calibration.forward_model import run_da_forward_model

run_da_forward_model(
    base_config_path=Path({repr(_path_str(base_config_path))}),
    input_path=Path("model_input.csv"),
    output_path=Path("model_output.csv"),
    gisdata_path=Path({repr(_path_str(gisdata_path))}),
    network_path=Path({repr(_path_str(network_path))}),
    forcing_path=Path({repr(_path_str(forcing_path))}),
    cycles_csv={_opt_path(cycles_csv)},
    obs_data_json={repr(obs_data_json)},
    restart_from={repr(restart_from)},
    routing_params_calibrated={routing_params_calibrated!r},
    warmup_state_path={_opt_path(warmup_state_path)},
    state_file_dir={_opt_path(state_file_dir)},
    keep_cycles={keep_cycles!r},
    state_spec_path={_opt_path(state_spec_path)},
)
'''
    output_script_path = Path(output_script_path)
    output_script_path.write_text(script, encoding="utf-8")
    logger.info(f"Generated DA forward_run.py: {output_script_path}")
    return output_script_path


def main():
    """CLI entry point for forward model."""
    parser = argparse.ArgumentParser(description="PEST++ forward model for MOBIDICpy")
    parser.add_argument("--config", required=True, help="Path to base MOBIDIC YAML config")
    parser.add_argument("--input", required=True, help="Path to model_input.csv")
    parser.add_argument("--output", required=True, help="Path to write model_output.csv")
    parser.add_argument("--gisdata", required=True, help="Path to gisdata.nc")
    parser.add_argument("--network", required=True, help="Path to network.parquet")
    parser.add_argument("--forcing", required=True, help="Path to forcing NetCDF")
    parser.add_argument("--start-date", required=True, help="Simulation start date")
    parser.add_argument("--end-date", required=True, help="Simulation end date")
    parser.add_argument("--reaches", required=True, help="Comma-separated list of reach IDs")
    parser.add_argument("--obs-data", default=None, help="JSON string with observation metadata")
    parser.add_argument("--routing-calibrated", action="store_true", help="Recalculate routing params")

    args = parser.parse_args()

    observation_reaches = [int(r) for r in args.reaches.split(",")]

    run_forward_model(
        base_config_path=Path(args.config),
        input_path=Path(args.input),
        output_path=Path(args.output),
        gisdata_path=Path(args.gisdata),
        network_path=Path(args.network),
        forcing_path=Path(args.forcing),
        start_date=args.start_date,
        end_date=args.end_date,
        observation_reaches=observation_reaches,
        obs_data_json=args.obs_data,
        routing_params_calibrated=args.routing_calibrated,
    )


if __name__ == "__main__":
    main()
