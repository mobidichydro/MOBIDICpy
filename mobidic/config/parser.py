"""YAML configuration parser for MOBIDIC."""

from pathlib import Path
from typing import Union

import yaml
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
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML file
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}") from e

    # Validate and parse with Pydantic
    try:
        config = MOBIDICConfig(**config_dict)
    except ValidationError as e:
        # Add context to the error message and re-raise
        raise ValueError(f"Configuration validation failed for {config_path}:\n{e}") from e

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
