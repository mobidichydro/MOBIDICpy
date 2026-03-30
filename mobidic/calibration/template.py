"""PEST++ template (.tpl) file generation.

Generates the template file that PEST++ uses to write parameter values into
model_input.csv. The .tpl file mirrors the CSV structure but with PEST++
parameter markers replacing the numeric values.
"""

from pathlib import Path

from loguru import logger

from mobidic.calibration.config import CalibrationConfig


def generate_template_file(
    calib_config: CalibrationConfig,
    output_path: Path,
    delimiter: str = "~",
) -> Path:
    """Generate a PEST++ template (.tpl) file for model_input.csv.

    The template file generates a CSV with two columns: parameter_key, value.
    PEST++ fills in the value column using parameter markers.

    Args:
        calib_config: Calibration configuration with parameter definitions.
        output_path: Path to write the .tpl file.
        delimiter: PEST++ template delimiter character (default: ~).

    Returns:
        Path to the generated .tpl file.
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

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(f"Generated template file with {len(calib_config.parameters)} parameters: {output_path}")
    return output_path


def generate_model_input_csv(
    calib_config: CalibrationConfig,
    output_path: Path,
) -> Path:
    """Generate the initial model_input.csv with initial parameter values.

    This file is the non-template version used for the initial forward run.

    Args:
        calib_config: Calibration configuration with parameter definitions.
        output_path: Path to write the CSV file.

    Returns:
        Path to the generated CSV file.
    """
    lines = ["parameter_key,value"]
    for param in calib_config.parameters:
        lines.append(f"{param.parameter_key},{param.initial_value}")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(f"Generated model_input.csv with {len(calib_config.parameters)} parameters: {output_path}")
    return output_path
