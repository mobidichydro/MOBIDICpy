"""``mobidic convert-meteo`` - convert MATLAB .mat meteo data to NetCDF."""

from argparse import _SubParsersAction

from loguru import logger

from mobidic.cli._common import add_log_level_arg
from mobidic.cli._common import configure_cli_logging


def register(subparsers: _SubParsersAction) -> None:
    """Register the ``convert-meteo`` subcommand."""
    parser = subparsers.add_parser(
        "convert-meteo",
        help="Convert MATLAB .mat meteorological data to CF-compliant NetCDF.",
        description="Load station-based meteorological data from a MATLAB .mat file and "
        "write it to a CF-1.12 compliant NetCDF file usable as simulation forcing.",
    )
    parser.add_argument("input", help="Path to the input MATLAB .mat file.")
    parser.add_argument("output", help="Path to the output NetCDF (.nc) file.")
    parser.add_argument("--basin", default=None, help="Optional basin name to store as metadata.")
    add_log_level_arg(parser)
    parser.set_defaults(handler=run)


def run(args) -> int:
    """Execute the convert-meteo command."""
    from mobidic import convert_mat_to_netcdf

    configure_cli_logging(log_level=args.log_level)

    add_metadata = {"basin": args.basin} if args.basin else None
    convert_mat_to_netcdf(args.input, args.output, add_metadata=add_metadata)

    logger.success(f"Meteorological data written to: {args.output}")
    return 0
