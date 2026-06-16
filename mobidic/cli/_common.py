"""Shared helpers for MOBIDIC CLI subcommands."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

LOG_LEVELS = ["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"]


class CLIError(Exception):
    """Raised for expected, user-facing errors that should not show a traceback."""


def add_log_level_arg(parser: ArgumentParser) -> None:
    """Add the common ``--log-level`` option to a subcommand parser."""
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=None,
        help="Override the logging level (default: from config, or INFO).",
    )


def configure_cli_logging(config=None, log_level: str | None = None) -> None:
    """Configure logging for a CLI command.

    Explicit setting of ``--log-level`` takes precedence; otherwise the config's
    ``advanced`` section is used. With neither, the package default (INFO) is used.

    Args:
        config: Optional :class:`~mobidic.MOBIDICConfig` to read log settings from.
        log_level: Optional explicit level that overrides the config.
    """
    # Imported lazily so building the parser / printing --help stays fast.
    from mobidic.utils import configure_logger, configure_logger_from_config

    if log_level is not None:
        configure_logger(level=log_level)
    elif config is not None:
        configure_logger_from_config(config)


def parse_date(value: str) -> datetime:
    """Parse an ISO-8601 date/datetime string into a :class:`datetime`.

    Args:
        value: ISO date or datetime string (e.g. ``2023-11-01`` or
            ``2023-11-01T06:00``).

    Returns:
        Parsed datetime.

    Raises:
        CLIError: If the string is not valid ISO-8601.
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise CLIError(f"Invalid date '{value}': expected ISO format (e.g. 2023-11-01).") from exc


def resolved_config_dir(config_file: str | Path) -> Path:
    """Return the directory containing the config file (for resolving relative paths)."""
    return Path(config_file).resolve().parent
