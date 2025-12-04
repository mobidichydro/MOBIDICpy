"""YAML configuration parser for MOBIDIC."""

from pathlib import Path
from typing import Annotated, Union, get_args, get_origin

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
        """Check if a field is a PathField type."""
        annotation = field_info.annotation

        # Handle Optional[PathField] - strip Optional first
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            # Remove NoneType if present (for Optional fields)
            non_none_args = [arg for arg in args if arg is not type(None)]

            # If we have a single non-None arg, check if it's Annotated
            if len(non_none_args) == 1:
                inner_annotation = non_none_args[0]
                inner_origin = get_origin(inner_annotation)

                # Check if it's Annotated[Union[str, Path], ...]
                if inner_origin is Annotated:
                    inner_args = get_args(inner_annotation)
                    if len(inner_args) > 0:
                        base_type = inner_args[0]
                        base_origin = get_origin(base_type)
                        if base_origin is Union:
                            base_args = get_args(base_type)
                            if len(base_args) == 2 and str in base_args and Path in base_args:
                                return True

            # Check if it's exactly Union[str, Path] (non-Optional case)
            if len(non_none_args) == 2 and str in non_none_args and Path in non_none_args:
                return True

        # Handle non-Optional Annotated[Union[str, Path], ...]
        if origin is Annotated:
            args = get_args(annotation)
            if len(args) > 0:
                base_type = args[0]
                base_origin = get_origin(base_type)
                if base_origin is Union:
                    base_args = get_args(base_type)
                    if len(base_args) == 2 and str in base_args and Path in base_args:
                        return True

        return False

    def resolve_path_fields(obj):
        """Recursively resolve all PathField attributes in a Pydantic model."""
        if obj is None:
            return

        # Get model fields metadata from the class, not the instance
        obj_class = type(obj)
        if hasattr(obj_class, "model_fields"):
            for field_name, field_info in obj_class.model_fields.items():
                field_value = getattr(obj, field_name)

                # Skip None values
                if field_value is None:
                    continue

                # Check if this field is a PathField
                if is_path_field(field_info):
                    # Resolve the path
                    resolved = resolve_path(field_value)
                    setattr(obj, field_name, resolved)
                    logger.debug(f"Resolved {field_name}: {field_value} -> {resolved}")
                # If it's a nested model, recurse
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
