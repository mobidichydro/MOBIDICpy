"""PEST++ instruction (.ins) file generation.

Generates the instruction file that PEST++ uses to read simulated observations
from model_output.csv written by the forward model.
"""

from pathlib import Path

from loguru import logger

from mobidic.calibration.config import CalibrationConfig


def _make_obs_names(
    calib_config: CalibrationConfig,
    n_obs_per_group: dict[str, int],
) -> list[str]:
    """Generate PEST++ observation names for all observation groups.

    Names follow the pattern: {group_name}_{index:04d}
    Metric pseudo-observations follow: {group_name}_{metric_name}

    Args:
        calib_config: Calibration configuration.
        n_obs_per_group: Dict mapping group name to number of time-series observations.

    Returns:
        Ordered list of all observation names.
    """
    obs_names = []
    for obs_group in calib_config.observations:
        n_obs = n_obs_per_group.get(obs_group.name, 0)
        # Time-series observations
        for i in range(n_obs):
            obs_names.append(f"{obs_group.name}_{i:04d}")
        # Metric pseudo-observations
        if obs_group.metrics:
            for mc in obs_group.metrics:
                obs_names.append(f"{obs_group.name}_{mc.metric}")
    return obs_names


def generate_instruction_file(
    calib_config: CalibrationConfig,
    n_obs_per_group: dict[str, int],
    output_path: Path,
    delimiter: str = "~",
) -> tuple[Path, list[str]]:
    """Generate a PEST++ instruction (.ins) file for model_output.csv.

    The model_output.csv is expected to have one observation per line,
    with format: obs_name,value

    The instruction file reads each line and extracts the value.

    Args:
        calib_config: Calibration configuration.
        n_obs_per_group: Dict mapping group name to number of time-series observations.
        output_path: Path to write the .ins file.
        delimiter: PEST++ instruction delimiter character (default: ~).

    Returns:
        Tuple of (path to .ins file, list of all observation names).
    """
    obs_names = _make_obs_names(calib_config, n_obs_per_group)

    lines = []

    # PEST++ instruction file header
    lines.append(f"pif {delimiter}")

    # Skip the CSV header line
    lines.append("l1")

    # One instruction per observation: read one line, extract the second comma-separated field
    for obs_name in obs_names:
        # l1 = advance one line, then read observation
        # ~,~ = search for comma delimiter, then read the whitespace-delimited token
        lines.append(f"l1 {delimiter},{delimiter} !{obs_name}!")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(f"Generated instruction file with {len(obs_names)} observations: {output_path}")
    return output_path, obs_names
