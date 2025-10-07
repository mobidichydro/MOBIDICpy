"""Logging configuration for MOBIDIC package."""

import sys
from pathlib import Path

from loguru import logger


def configure_logger(
    level: str = "INFO",
    format_string: str | None = None,
    colorize: bool = True,
    log_file: str | Path | None = None,
) -> None:
    """Configure the logger for MOBIDIC package.

    This function configures the loguru logger with a consistent format
    for use across the MOBIDIC package and example scripts.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO.
        format_string: Custom format string for log messages. If None, uses default format.
        log_file: Optional path to log file. If provided, logs will be written to this file
                  in addition to stdout.
        colorize: Whether to use colored output (default: True).

    Examples:
        >>> from mobidic.utils import configure_logger
        >>> configure_logger(level="DEBUG", log_file="mobidic.log")
    """
    # Remove default handler
    logger.remove()

    # Default format string
    if format_string is None:
        if colorize:
            if level == "DEBUG":
                format_string = (
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                    "{message}"
                )
            else:
                format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
        else:
            format_string = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"

    # Add stdout handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=colorize,
    )

    # Add file handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        if level == "DEBUG":
            format_logfile = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        else:
            format_logfile = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        logger.add(
            log_file,
            format=format_logfile,
            level=level,
            rotation="10 MB",  # Rotate when file reaches 10 MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
        )
        logger.info(f"Logging to file: {log_file}")
