"""YAML configuration parser for MOBIDIC."""

from pathlib import Path
from typing import Union

import yaml
from loguru import logger
from pydantic import ValidationError

from .schema import MOBIDICConfig


def load_config(config_path: Union[str, Path]) -> MOBIDICConfig:
    """
    Load and validate MOBIDIC configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        MOBIDICConfig: Validated configuration object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
        ValueError: If the configuration does not match the schema.

    Examples:
        >>> config = load_config("examples/sample_config.yaml")
        >>> print(config.basin.id)
        'Basin'
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    # Load YAML file
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config_dict = yaml.safe_load(f)
            logger.success(f"Successfully parsed YAML file: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}") from e

    # Validate and parse with Pydantic
    try:
        config = MOBIDICConfig(**config_dict)
        logger.info("Configuration validated successfully")
    except ValidationError as e:
        # Add context to the error message and re-raise
        logger.error(f"Configuration validation failed for {config_path}")
        raise ValueError(f"Configuration validation failed for {config_path}:\n{e}") from e

    # Resolve all paths to absolute paths relative to the YAML file location
    config_dir = config_path.parent.resolve()
    logger.debug(f"Resolving paths relative to: {config_dir}")

    def resolve_path(path_str: str) -> Path:
        """Convert path to absolute, resolving relative paths from config directory."""
        if not path_str:
            return Path(path_str)
        path = Path(path_str)
        if path.is_absolute():
            return path
        else:
            return (config_dir / path).resolve()

    # Resolve paths in config.paths.*
    config.paths.meteodata = resolve_path(config.paths.meteodata)
    config.paths.gisdata = resolve_path(config.paths.gisdata)
    config.paths.network = resolve_path(config.paths.network)
    config.paths.states = resolve_path(config.paths.states)
    config.paths.output = resolve_path(config.paths.output)

    # Resolve vector file paths
    config.vector_files.river_network.shp = resolve_path(config.vector_files.river_network.shp)

    # Resolve raster file paths
    config.raster_files.dtm = resolve_path(config.raster_files.dtm)
    config.raster_files.flow_dir = resolve_path(config.raster_files.flow_dir)
    config.raster_files.flow_acc = resolve_path(config.raster_files.flow_acc)
    config.raster_files.Wc0 = resolve_path(config.raster_files.Wc0)
    config.raster_files.Wg0 = resolve_path(config.raster_files.Wg0)
    config.raster_files.ks = resolve_path(config.raster_files.ks)

    # Resolve optional raster file paths
    if config.raster_files.kf is not None:
        config.raster_files.kf = resolve_path(config.raster_files.kf)
    if config.raster_files.CH is not None:
        config.raster_files.CH = resolve_path(config.raster_files.CH)
    if config.raster_files.Alb is not None:
        config.raster_files.Alb = resolve_path(config.raster_files.Alb)
    if config.raster_files.Ma is not None:
        config.raster_files.Ma = resolve_path(config.raster_files.Ma)
    if config.raster_files.Mf is not None:
        config.raster_files.Mf = resolve_path(config.raster_files.Mf)
    if config.raster_files.gamma is not None:
        config.raster_files.gamma = resolve_path(config.raster_files.gamma)
    if config.raster_files.kappa is not None:
        config.raster_files.kappa = resolve_path(config.raster_files.kappa)
    if config.raster_files.beta is not None:
        config.raster_files.beta = resolve_path(config.raster_files.beta)
    if config.raster_files.alpha is not None:
        config.raster_files.alpha = resolve_path(config.raster_files.alpha)

    # Resolve output report settings sel_file
    if config.output_report_settings and config.output_report_settings.sel_file is not None:
        config.output_report_settings.sel_file = resolve_path(config.output_report_settings.sel_file)

    # Resolve advanced log_file
    if config.advanced and config.advanced.log_file is not None:
        config.advanced.log_file = resolve_path(config.advanced.log_file)

    return config


def save_config(config: MOBIDICConfig, output_path: Union[str, Path]) -> None:
    """
    Save MOBIDIC configuration to YAML file.

    Args:
        config: Configuration object to save.
        output_path: Path where the YAML file will be saved.

    Examples:
        >>> save_config(config, "config/config_modified.yaml")
    """
    output_path = Path(output_path)

    logger.info(f"Saving configuration to: {output_path}")

    # Convert Pydantic model to dictionary
    config_dict = config.model_dump(exclude_none=True, mode="python")

    # Save to YAML file
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

    logger.info(f"Configuration saved successfully to: {output_path}")
