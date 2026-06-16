"""``mobidic hyetograph`` - generate a design-storm NetCDF from IDF parameters."""

from argparse import _SubParsersAction
from datetime import datetime
from pathlib import Path

from loguru import logger

from mobidic.cli._common import add_log_level_arg
from mobidic.cli._common import configure_cli_logging
from mobidic.cli._common import parse_date

# Default reference start time when --start is not given.
DEFAULT_HYETOGRAPH_START = datetime(2000, 1, 1)


def register(subparsers: _SubParsersAction) -> None:
    """Register the ``hyetograph`` subcommand."""
    parser = subparsers.add_parser(
        "hyetograph",
        help="Generate a design-storm hyetograph NetCDF from IDF parameters.",
        description="Generate a synthetic design-storm hyetograph from the IDF rasters and "
        "settings in the config's 'hyetograph' section, writing it to paths.hyetograph.",
    )
    parser.add_argument("config", help="Path to the MOBIDIC YAML configuration file.")
    parser.add_argument(
        "--start",
        default=None,
        help=f"Reference event start time (ISO date). Default: {DEFAULT_HYETOGRAPH_START.date()}.",
    )
    add_log_level_arg(parser)
    parser.set_defaults(handler=run)


def run(args) -> int:
    """Execute the hyetograph command."""
    from mobidic import HyetographGenerator
    from mobidic import load_config

    config_file = Path(args.config).resolve()
    config = load_config(config_file)
    configure_cli_logging(config, args.log_level)

    start_time = parse_date(args.start) if args.start else DEFAULT_HYETOGRAPH_START

    # from_config generates the hyetograph and writes it to config.paths.hyetograph.
    forcing = HyetographGenerator.from_config(
        config=config,
        base_path=config_file.parent,
        start_time=start_time,
    )

    logger.success(f"Hyetograph written to: {config.paths.hyetograph}")
    logger.info(f"Event period: {forcing.start_date} to {forcing.end_date}")
    return 0
