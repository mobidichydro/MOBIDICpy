"""PEST++ template (.tpl) file generation.

Generates the template file that PEST++ uses to write parameter values into
model_input.csv. The .tpl file mirrors the CSV structure but with PEST++
parameter markers replacing the numeric values.
"""

from pathlib import Path
from typing import NamedTuple

from loguru import logger

from mobidic.calibration.config import CalibrationConfig

#: Minimum template field width. PEST++ fits a value to the field width and
#: silently truncates when the field is too narrow, so a state identifier in a
#: field that is too small would name the wrong state file.
MIN_FIELD_WIDTH = 12


class ExtraParameter(NamedTuple):
    """A non-calibration row of ``model_input.csv`` (cycle number, state values).

    Attributes:
        key: ``parameter_key`` written in the first CSV column. Data-assimilation
            rows use reserved names (``__cycle__``, ``__state_id__``) that the
            forward model handles separately.
        name: PEST++ parameter name substituted into the field.
        value: Initial value written to the non-template ``model_input.csv``.
        width: Template field width; at least :data:`MIN_FIELD_WIDTH`.
    """

    key: str
    name: str
    value: float
    width: int = MIN_FIELD_WIDTH


def generate_template_file(
    calib_config: CalibrationConfig,
    output_path: Path,
    delimiter: str = "~",
    extra_parameters: list[ExtraParameter] | None = None,
) -> Path:
    """Generate a PEST++ template (.tpl) file for model_input.csv.

    The template file generates a CSV with two columns: parameter_key, value.
    PEST++ fills in the value column using parameter markers.

    Args:
        calib_config: Calibration configuration with parameter definitions.
        output_path: Path to write the .tpl file.
        delimiter: PEST++ template delimiter character (default: ~).
        extra_parameters: Additional rows appended after the calibration
            parameters (used by sequential data assimilation).

    Returns:
        Path to the generated .tpl file.

    Raises:
        ValueError: If an extra parameter requests a field narrower than
            :data:`MIN_FIELD_WIDTH`.
    """
    lines = []

    # PEST++ template file header
    lines.append(f"ptf {delimiter}")

    # CSV header
    lines.append("parameter_key,value")

    # One line per calibration parameter
    for param in calib_config.parameters:
        # PEST++ marker: ~  param_name  ~
        # Parameter names are right-padded to at least 12 chars for readability
        marker = f"{delimiter} {param.name:<12s}{delimiter}"
        lines.append(f"{param.parameter_key},{marker}")

    for extra in extra_parameters or []:
        if extra.width < MIN_FIELD_WIDTH:
            raise ValueError(
                f"Template field for '{extra.name}' is {extra.width} characters wide; "
                f"at least {MIN_FIELD_WIDTH} are required to avoid silent truncation"
            )
        marker = f"{delimiter} {extra.name:<{extra.width}s}{delimiter}"
        lines.append(f"{extra.key},{marker}")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_extra = len(extra_parameters or [])
    logger.info(
        f"Generated template file with {len(calib_config.parameters)} parameters "
        f"(+{n_extra} data-assimilation rows): {output_path}"
    )
    return output_path


def generate_model_input_csv(
    calib_config: CalibrationConfig,
    output_path: Path,
    extra_parameters: list[ExtraParameter] | None = None,
) -> Path:
    """Generate the initial model_input.csv with initial parameter values.

    This file is the non-template version used for the initial forward run.

    Args:
        calib_config: Calibration configuration with parameter definitions.
        output_path: Path to write the CSV file.
        extra_parameters: Additional rows appended after the calibration
            parameters (used by sequential data assimilation).

    Returns:
        Path to the generated CSV file.
    """
    lines = ["parameter_key,value"]
    for param in calib_config.parameters:
        lines.append(f"{param.parameter_key},{param.initial_value}")
    for extra in extra_parameters or []:
        lines.append(f"{extra.key},{extra.value}")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_extra = len(extra_parameters or [])
    logger.info(
        f"Generated model_input.csv with {len(calib_config.parameters)} parameters "
        f"(+{n_extra} data-assimilation rows): {output_path}"
    )
    return output_path
