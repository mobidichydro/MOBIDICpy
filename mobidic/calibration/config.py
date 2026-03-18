"""Pydantic models for PEST++ calibration configuration."""

from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class CalibrationParameter(BaseModel):
    """A single parameter to be calibrated by PEST++."""

    name: str = Field(..., description="Parameter name (used in PEST++ control file)")
    parameter_key: str = Field(
        ..., description="Dot-notation path into MOBIDIC YAML config (e.g., parameters.multipliers.ks_factor)"
    )
    initial_value: float = Field(..., description="Starting value for optimization")
    lower_bound: float = Field(..., description="Lower bound for parameter")
    upper_bound: float = Field(..., description="Upper bound for parameter")
    transform: Literal["none", "log", "fixed"] = Field("none", description="Parameter transformation")
    scale: float = Field(1.0, description="Multiplication factor applied by PEST++")
    offset: float = Field(0.0, description="Additive offset applied by PEST++")
    par_group: str = Field("default", description="Parameter group name")

    @field_validator("name")
    @classmethod
    def check_name_no_spaces(cls, v: str) -> str:
        """PEST++ parameter names cannot contain spaces."""
        if " " in v:
            raise ValueError("Parameter name cannot contain spaces")
        return v

    @model_validator(mode="after")
    def check_bounds(self) -> "CalibrationParameter":
        """Validate that lower_bound < upper_bound and initial_value is within bounds."""
        if self.lower_bound >= self.upper_bound:
            raise ValueError(f"lower_bound ({self.lower_bound}) must be less than upper_bound ({self.upper_bound})")
        if not self.lower_bound <= self.initial_value <= self.upper_bound:
            raise ValueError(
                f"initial_value ({self.initial_value}) must be between "
                f"lower_bound ({self.lower_bound}) and upper_bound ({self.upper_bound})"
            )
        if self.transform == "log" and self.lower_bound <= 0:
            raise ValueError("lower_bound must be positive when transform='log'")
        return self


class MetricConfig(BaseModel):
    """Configuration for a derived metric used as pseudo-observation."""

    metric: str = Field(..., description="Metric name (nse, nse_log, pbias, peak_error, rmse, kge)")
    target: float = Field(..., description="Target value PEST++ tries to match (e.g., 1.0 for NSE)")
    weight: float = Field(1.0, description="Observation weight for this metric")

    @field_validator("metric")
    @classmethod
    def check_metric_name(cls, v: str) -> str:
        """Validate metric name is supported."""
        supported = {"nse", "nse_log", "pbias", "peak_error", "rmse", "kge"}
        if v not in supported:
            raise ValueError(f"Unsupported metric '{v}'. Supported: {sorted(supported)}")
        return v

    @field_validator("weight")
    @classmethod
    def check_weight_non_negative(cls, v: float) -> float:
        """Weights must be non-negative."""
        if v < 0:
            raise ValueError("Weight must be non-negative")
        return v


class ObservationGroup(BaseModel):
    """An observation group (e.g., discharge at a specific gauging station)."""

    name: str = Field(..., description="PEST++ observation group identifier (prefix for obs names)")
    obs_file: str = Field(..., description="Path to observed data CSV file (relative to calibration config)")
    reach_id: int = Field(..., description="Reach ID (mobidic_id) where observations are located")
    weight: float = Field(1.0, description="Default weight for all observations in this group")
    time_column: str = Field("time", description="Column name for timestamps in obs_file")
    value_column: str = Field(..., description="Column name for observed values in obs_file")
    metrics: Optional[list[MetricConfig]] = Field(None, description="Optional derived metrics as pseudo-observations")

    @field_validator("name")
    @classmethod
    def check_name_no_spaces(cls, v: str) -> str:
        """PEST++ observation names cannot contain spaces."""
        if " " in v:
            raise ValueError("Observation group name cannot contain spaces")
        return v

    @field_validator("weight")
    @classmethod
    def check_weight_non_negative(cls, v: float) -> float:
        """Weights must be non-negative."""
        if v < 0:
            raise ValueError("Weight must be non-negative")
        return v


class CalibrationPeriod(BaseModel):
    """A date range with start and end dates.

    Used for both calibration_period and simulation_period.
    """

    start_date: str = Field(..., description="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")

    @field_validator("start_date", "end_date")
    @classmethod
    def check_date_format(cls, v: str) -> str:
        """Validate that date strings are parseable."""
        try:
            pd.Timestamp(v)
        except ValueError as e:
            raise ValueError(f"Invalid date format: '{v}'. Expected 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.") from e
        return v

    @model_validator(mode="after")
    def check_start_before_end(self) -> "CalibrationPeriod":
        """Validate that start_date < end_date."""
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        if start >= end:
            raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")
        return self


class ParallelConfig(BaseModel):
    """Configuration for parallel PEST++ execution."""

    num_workers: Optional[int] = Field(None, description="Workers per node (default: all available CPUs)")
    port: int = Field(4004, description="TCP port for manager-agent communication")
    manager_ip: Optional[str] = Field(None, description="Manager IP for cluster mode (None = local mode)")

    @field_validator("port")
    @classmethod
    def check_port_range(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1024 <= v <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v


class CalibrationConfig(BaseModel):
    """Complete calibration configuration for PEST++ integration."""

    mobidic_config: str = Field(..., description="Path to MOBIDIC simulation YAML config file")

    simulation_period: Optional[CalibrationPeriod] = Field(
        None,
        description="Simulation period (start_date, end_date) for the forward model. "
        "Can be longer than calibration_period to include warm-up. "
        "Must be contained within the forcing data time range. "
        "If None, defaults to calibration_period.",
    )

    calibration_period: Optional[CalibrationPeriod] = Field(
        None,
        description="Calibration period (start_date, end_date): only observations within "
        "this window are used by PEST++. Must be contained within simulation_period. "
        "If None, defaults to full observation period.",
    )

    use_raster_forcing: bool = Field(
        False,
        description="If True, make a first forward run to rasterize station forcing for faster subsequent runs",
    )

    parameters: list[CalibrationParameter] = Field(..., min_length=1, description="List of parameters to calibrate")

    observations: list[ObservationGroup] = Field(..., min_length=1, description="List of observation groups")

    pest_tool: Literal["glm", "ies", "sen", "da", "opt", "mou", "sqp"] = Field("glm", description="PEST++ tool to use")

    pest_options: Optional[dict] = Field(
        default_factory=dict,
        description="PEST++ and tool-specific options (e.g., noptmax, "
        "relparmax, facparmax, pst_version, ies_num_reals)",
    )

    working_dir: str = Field("pest_run", description="Working directory for PEST++ files")

    parallel: Optional[ParallelConfig] = Field(default_factory=ParallelConfig)

    @model_validator(mode="after")
    def check_parameter_names_unique(self) -> "CalibrationConfig":
        """Ensure all parameter names are unique."""
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate parameter names: {set(duplicates)}")
        return self

    @model_validator(mode="after")
    def check_observation_names_unique(self) -> "CalibrationConfig":
        """Ensure all observation group names are unique."""
        names = [o.name for o in self.observations]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate observation group names: {set(duplicates)}")
        return self

    @model_validator(mode="after")
    def check_calibration_within_simulation(self) -> "CalibrationConfig":
        """Ensure calibration_period is contained within simulation_period.

        If simulation_period is not set, it defaults to calibration_period at runtime.
        But if both are set, calibration must be within simulation.
        """
        if self.simulation_period is not None and self.calibration_period is not None:
            sim_start = pd.Timestamp(self.simulation_period.start_date)
            sim_end = pd.Timestamp(self.simulation_period.end_date)
            cal_start = pd.Timestamp(self.calibration_period.start_date)
            cal_end = pd.Timestamp(self.calibration_period.end_date)

            if cal_start < sim_start:
                raise ValueError(
                    f"calibration_period.start_date ({self.calibration_period.start_date}) "
                    f"must be >= simulation_period.start_date ({self.simulation_period.start_date})"
                )
            if cal_end > sim_end:
                raise ValueError(
                    f"calibration_period.end_date ({self.calibration_period.end_date}) "
                    f"must be <= simulation_period.end_date ({self.simulation_period.end_date})"
                )
        return self


def load_calibration_config(config_path: str | Path) -> CalibrationConfig:
    """Load and validate calibration configuration from YAML file.

    Args:
        config_path: Path to the calibration YAML file.

    Returns:
        Validated CalibrationConfig object.
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Calibration config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return CalibrationConfig(**config_dict)
