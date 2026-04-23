"""YAML configuration parser for MOBIDIC."""

from pathlib import Path
from typing import Union, get_args, get_origin

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
            logger.success(f"Successfully loaded YAML file: {config_path}")
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

    def resolve_path(path_str: Union[str, Path]) -> Path:
        """Convert path to absolute, resolving relative paths from config directory."""
        if not path_str:
            return Path(path_str)
        path = Path(path_str)
        if path.is_absolute():
            return path
        else:
            return (config_dir / path).resolve()

    def is_path_field(field_info) -> bool:
        """Check if a field is a PathField type.

        Pydantic v2 unwraps ``Annotated[...]`` metadata on non-Optional fields, so
        ``field_info.annotation`` arrives here as either ``Union[str, Path]`` (for
        non-Optional PathFields) or ``Optional[Annotated[Union[str, Path], ...]]``
        (which ``get_origin`` reports as ``Union`` with ``NoneType`` stripped).
        """
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is not Union:
            return False

        non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]

        # Non-Optional PathField: annotation is ``Union[str, Path]``.
        if len(non_none_args) == 2 and str in non_none_args and Path in non_none_args:
            return True

        # Optional PathField: the single inner annotation is ``Annotated[Union[str, Path], ...]``.
        if len(non_none_args) == 1:
            inner_args = get_args(non_none_args[0])
            if inner_args:
                base_args = get_args(inner_args[0])
                if len(base_args) == 2 and str in base_args and Path in base_args:
                    return True

        return False

    def resolve_path_fields(obj):
        """Recursively resolve all PathField attributes in a Pydantic model."""
        for field_name, field_info in type(obj).model_fields.items():
            field_value = getattr(obj, field_name)

            if field_value is None:
                continue

            if is_path_field(field_info):
                resolved = resolve_path(field_value)
                setattr(obj, field_name, resolved)
                logger.debug(f"Resolved {field_name}: {field_value} -> {resolved}")
            elif hasattr(type(field_value), "model_fields"):
                resolve_path_fields(field_value)

    # Recursively resolve all path fields in the config
    resolve_path_fields(config)

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

    # Convert Pydantic model to dictionary with JSON-compatible types (converts Path to str)
    config_dict = config.model_dump(exclude_none=True, mode="json")

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
