"""Dot-notation YAML path traversal and parameter value manipulation.

This module handles resolving dot-notation config paths (e.g., 'parameters.multipliers.ks_factor')
to actual values in a MOBIDICConfig object or YAML dictionary, and applying parameter updates.
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def resolve_dot_path(obj: Any, dot_path: str) -> Any:
    """Resolve a dot-notation path on a Pydantic model or dict.

    Args:
        obj: A Pydantic model or dictionary to traverse.
        dot_path: Dot-separated path (e.g., 'parameters.multipliers.ks_factor').

    Returns:
        The value at the specified path.

    Raises:
        KeyError: If the path does not exist.
    """
    parts = dot_path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Key '{part}' not found in dict at path '{dot_path}'")
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            raise KeyError(f"Attribute '{part}' not found on {type(current).__name__} at path '{dot_path}'")
    return current


def set_dot_path(obj: dict, dot_path: str, value: Any) -> None:
    """Set a value at a dot-notation path in a nested dictionary.

    Args:
        obj: A nested dictionary to modify (in-place).
        dot_path: Dot-separated path (e.g., 'parameters.multipliers.ks_factor').
        value: The value to set.

    Raises:
        KeyError: If any intermediate key does not exist.
    """
    parts = dot_path.split(".")
    current = obj
    for part in parts[:-1]:
        if part not in current:
            raise KeyError(f"Key '{part}' not found at path '{dot_path}'")
        current = current[part]
    current[parts[-1]] = value


def apply_parameters_to_yaml(
    base_yaml_path: Path,
    parameter_updates: dict[str, float],
    output_yaml_path: Path,
) -> None:
    """Apply parameter updates to a YAML config and write the result.

    Reads the base YAML, updates the specified dot-notation paths with new values,
    and writes the modified config to output_yaml_path.

    Args:
        base_yaml_path: Path to the base MOBIDIC YAML config file.
        parameter_updates: Dict mapping dot-notation paths to new values.
        output_yaml_path: Path to write the modified YAML config.
    """
    with open(base_yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    for dot_path, value in parameter_updates.items():
        logger.debug(f"Setting {dot_path} = {value}")
        set_dot_path(config_dict, dot_path, float(value))

    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)


def read_model_input_csv(input_path: Path) -> dict[str, float]:
    """Read parameter values from a model_input.csv file written by PEST++ via .tpl.

    The CSV file has two columns: parameter_key, value.

    Args:
        input_path: Path to model_input.csv.

    Returns:
        Dict mapping parameter_key (dot-notation) to float value.
    """
    import csv

    params = {}
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        if len(header) < 2:
            raise ValueError(f"Expected at least 2 columns in {input_path}, got {len(header)}")
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                value = float(row[1].strip())
                params[key] = value
    return params
