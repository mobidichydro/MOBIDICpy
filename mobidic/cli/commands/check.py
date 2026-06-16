"""``mobidic check`` - validate a config file and summarize it."""

from argparse import _SubParsersAction
from pathlib import Path

from loguru import logger

from mobidic.cli._common import add_log_level_arg
from mobidic.cli._common import configure_cli_logging


def register(subparsers: _SubParsersAction) -> None:
    """Register the ``check`` subcommand."""
    parser = subparsers.add_parser(
        "check",
        help="Validate a configuration file and print a summary.",
        description="Load and validate a MOBIDIC YAML config, then print a summary "
        "(basin, timestep, forcing mode, resolved input paths and whether they exist).",
    )
    parser.add_argument("config", help="Path to the MOBIDIC YAML configuration file.")
    add_log_level_arg(parser)
    parser.set_defaults(handler=run)


def _forcing_mode(config) -> tuple[str, Path | None]:
    """Return the active forcing mode and its configured path."""
    if config.paths.meteodata is not None:
        return "station (meteodata)", Path(config.paths.meteodata)
    if config.paths.meteoraster is not None:
        return "raster (meteoraster)", Path(config.paths.meteoraster)
    if config.paths.hyetograph is not None:
        return "hyetograph", Path(config.paths.hyetograph)
    return "unknown", None


def _exists_mark(path: Path | None) -> str:
    """Return a human-readable existence marker for a path."""
    if path is None:
        return "-"
    return "[found]" if path.exists() else "[missing]"


def run(args) -> int:
    """Execute the check command."""
    from mobidic import load_config

    config = load_config(args.config)
    configure_cli_logging(config, args.log_level)

    mode, forcing_path = _forcing_mode(config)

    logger.info("Configuration is valid.")
    logger.info(f"  Basin name:      {config.basin.id or '(unnamed)'}")
    logger.info(f"  Parameter set:   {config.basin.paramset_id or '(none)'}")
    logger.info(f"  Time step:       {config.simulation.timestep} s")
    logger.info(f"  Soil scheme:     {config.simulation.soil_scheme}")
    logger.info(f"  Energy balance:  {config.simulation.energy_balance}")
    logger.info(f"  Routing method:  {config.parameters.routing.method}")
    logger.info(f"  Forcing mode:    {mode}")

    logger.info("Inputs:")
    logger.info(f"  forcing:  {_exists_mark(forcing_path)} {forcing_path}")
    logger.info(f"  gisdata:  {_exists_mark(Path(config.paths.gisdata))} {config.paths.gisdata}")
    logger.info(f"  network:  {_exists_mark(Path(config.paths.network))} {config.paths.network}")
    logger.info(f"  dtm:      {_exists_mark(Path(config.raster_files.dtm))} {config.raster_files.dtm}")

    logger.info("Outputs:")
    logger.info(f"  reports:  {config.paths.output}")
    logger.info(f"  states:   {config.paths.states}")
    return 0
