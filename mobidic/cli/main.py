"""Top-level argument parser for the ``mobidic`` CLI."""

import argparse
import sys

from loguru import logger

from mobidic import __version__
from mobidic.cli._common import CLIError
from mobidic.cli.commands import calibration
from mobidic.cli.commands import check
from mobidic.cli.commands import hyetograph
from mobidic.cli.commands import meteo
from mobidic.cli.commands import preprocess
from mobidic.cli.commands import simulation

# Subcommand modules
COMMANDS = [preprocess, simulation, calibration, hyetograph, meteo, check]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands registered."""
    parser = argparse.ArgumentParser(
        prog="mobidic",
        description=f"MOBIDICpy v{__version__} - distributed and continuous hydrological model.",
    )
    parser.add_argument("--version", action="version", version=f"mobidic {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for command in COMMANDS:
        command.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``). Used by tests.

    Returns:
        Process exit code (0 on success, 1 on a handled error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "handler", None):
        parser.print_help()
        return 1

    try:
        return args.handler(args) or 0
    except (CLIError, FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1
    except ImportError as exc:
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
