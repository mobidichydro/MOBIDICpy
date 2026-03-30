"""PEST++ forward model wrapper for MOBIDICpy.

This script is invoked by PEST++ for each model run. It:
1. Reads parameter values from model_input.csv (written by PEST++ via .tpl)
2. Applies parameter updates to the base MOBIDIC YAML config
3. Runs the simulation
4. Writes simulated observations to model_output.csv (read by PEST++ via .ins)

Can be called as: python -m mobidic.calibration.forward_model --args ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from mobidic.calibration.parameter_mapping import apply_parameters_to_yaml, read_model_input_csv


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
    from mobidic.config import load_config
    from mobidic.core.simulation import Simulation
    from mobidic.preprocessing.io import load_gisdata
    from mobidic.preprocessing.meteo_raster import MeteoRaster

    # Step 1: Read PEST++-substituted parameter values
    param_updates = read_model_input_csv(input_path)
    logger.info(f"Read {len(param_updates)} parameters from {input_path}")

    # Step 2: Apply parameter updates to YAML and create modified config
    modified_yaml = input_path.parent / "_modified_config.yaml"
    apply_parameters_to_yaml(base_config_path, param_updates, modified_yaml)
    config = load_config(modified_yaml)

    # Step 3: Load pre-processed data
    gisdata = load_gisdata(gisdata_path, network_path)

    # Remove gisdata grids for calibrated parameters so the simulation
    # uses the scalar config values (which PEST++ perturbs) instead of
    # the fixed raster grids baked into gisdata.nc.
    _PARAM_KEY_TO_GRID = {
        "parameters.soil.alpha": "alpha",
        "parameters.soil.beta": "beta",
        "parameters.soil.gamma": "gamma",
        "parameters.soil.kappa": "kappa",
        "parameters.soil.ks": "ks",
        "parameters.soil.kf": "kf",
    }
    for param_key in param_updates:
        grid_name = _PARAM_KEY_TO_GRID.get(param_key)
        if grid_name and grid_name in gisdata.grids:
            del gisdata.grids[grid_name]
            logger.debug(f"Removed gisdata grid '{grid_name}' - using calibrated scalar value")

    # Step 4: Recalculate routing parameters if needed
    if routing_params_calibrated:
        gisdata.network = _recalculate_routing_parameters(gisdata.network, config)
        logger.info("Recalculated routing parameters from calibrated config")

    # Step 5: Load forcing and run simulation
    forcing = MeteoRaster.from_netcdf(forcing_path)

    # Disable unnecessary outputs for calibration runs
    config.output_states_settings.output_states = "None"
    config.output_report.discharge = False
    config.output_report.lateral_inflow = False
    config.output_forcing_data.meteo_data = False

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


def _write_model_output(
    discharge_ts: np.ndarray,
    sim_times: list,
    observation_reaches: list[int],
    network,
    output_path: Path,
    obs_data_json: str | None = None,
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
        n_obs = obs_info.get("n_obs", len(sim_q))
        sim_indices = obs_info.get("sim_indices", list(range(n_obs)))

        for i, si in enumerate(sim_indices):
            obs_name = f"{group_name}_{i:04d}"
            lines.append(f"{obs_name},{sim_q[si]:.10e}")

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
                lines.append(f"{obs_name},{metric_value:.10e}")

    # Fallback: if no obs_data provided, write all discharge for all observation reaches
    if obs_data is None:
        for reach_id in observation_reaches:
            if reach_id not in reach_to_idx:
                raise ValueError(f"Reach ID {reach_id} not found in network")
            col_idx = reach_to_idx[reach_id]
            sim_q = discharge_ts[:, col_idx]
            for i in range(len(sim_q)):
                obs_name = f"reach_{reach_id}_{i:04d}"
                lines.append(f"{obs_name},{sim_q[i]:.10e}")

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
