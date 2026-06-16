"""``mobidic simulation`` - run a hydrological simulation from a config file."""

from argparse import _SubParsersAction
from datetime import datetime
from pathlib import Path

from loguru import logger

from mobidic.cli._common import CLIError
from mobidic.cli._common import add_log_level_arg
from mobidic.cli._common import configure_cli_logging
from mobidic.cli._common import parse_date

# Default reference start time for hyetograph forcing when --start is not given.
DEFAULT_HYETOGRAPH_START = datetime(2000, 1, 1)


def register(subparsers: _SubParsersAction) -> None:
    """Register the ``simulation`` subcommand."""
    parser = subparsers.add_parser(
        "simulation",
        help="Run a hydrological simulation from a config file.",
        description="Load preprocessed GIS data and meteorological forcing, then run the MOBIDIC simulation.",
    )
    parser.add_argument("config", help="Path to the MOBIDIC YAML configuration file.")
    parser.add_argument(
        "--start",
        default=None,
        help="Override simulation start (ISO date, e.g., 2024-01-01). Default: first date available in forcing. "
        f"For hyetograph forcing, seeds the event start time (default {DEFAULT_HYETOGRAPH_START.date()}).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Override simulation end (ISO date, e.g., 2024-12-31). Default: last date available in forcing.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run GIS preprocessing first (and save gisdata/network) before simulating.",
    )
    add_log_level_arg(parser)
    parser.set_defaults(handler=run)


def _load_gisdata(config, do_preprocess: bool):
    """Return GISData, running preprocessing if requested or required."""
    from mobidic import load_gisdata
    from mobidic import run_preprocessing
    from mobidic import save_gisdata
    from mobidic import save_network

    gisdata_path = Path(config.paths.gisdata)
    network_path = Path(config.paths.network)

    if do_preprocess:
        gisdata = run_preprocessing(config)
        save_gisdata(gisdata, gisdata_path)
        save_network(gisdata.network, network_path, format="parquet")
        return gisdata

    if not gisdata_path.exists() or not network_path.exists():
        raise CLIError(
            "Preprocessed GIS data not found. Run 'mobidic preprocess <config>' first, "
            "or pass --preprocess to run it now."
        )
    return load_gisdata(gisdata_path, network_path)


def _build_forcing(config, config_file: Path, start: str | None):
    """Build the forcing object from config (station / raster / hyetograph)."""
    from mobidic import HyetographGenerator
    from mobidic import MeteoData
    from mobidic import MeteoRaster

    if config.paths.meteodata is not None:
        return MeteoData.from_netcdf(config.paths.meteodata)
    if config.paths.meteoraster is not None:
        return MeteoRaster.from_netcdf(config.paths.meteoraster)
    if config.paths.hyetograph is not None:
        start_time = parse_date(start) if start else DEFAULT_HYETOGRAPH_START
        return HyetographGenerator.from_config(
            config=config,
            base_path=config_file.parent,
            start_time=start_time,
        )
    # Should be unreachable: config validation requires exactly one forcing source.
    raise CLIError("No meteorological forcing configured (meteodata, meteoraster, or hyetograph).")


def run(args) -> int:
    """Execute the simulation command."""
    from mobidic import Simulation
    from mobidic import load_config

    config_file = Path(args.config).resolve()
    config = load_config(config_file)
    configure_cli_logging(config, args.log_level)

    gisdata = _load_gisdata(config, args.preprocess)
    forcing = _build_forcing(config, config_file, args.start)

    start_date = args.start if args.start else forcing.start_date
    end_date = args.end if args.end else forcing.end_date

    logger.info(f"Simulation period: {start_date} to {end_date}")

    sim = Simulation(gisdata, forcing, config)
    sim.run(start_date=start_date, end_date=end_date)

    logger.success("Simulation completed.")
    logger.info(f"Output reports directory: {config.paths.output}")
    logger.info(f"States directory: {config.paths.states}")
    return 0
