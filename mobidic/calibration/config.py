"""Pydantic models for PEST++ calibration configuration."""

import os
import types
from pathlib import Path
from typing import Literal, Optional, Union, get_args, get_origin

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator


def _unwrap_optional(annotation: object) -> object:
    """Strip a single layer of Optional/Union-with-None wrapping."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _validate_mobidic_parameter_path(dot_path: str) -> None:
    """Verify dot-notation path resolves to a numeric field in MOBIDICConfig.

    Walks the Pydantic schema field-by-field to ensure every segment exists,
    and that the leaf type is float or int (i.e. a calibratable scalar).
    """
    from mobidic.config.schema import MOBIDICConfig

    parts = dot_path.split(".")
    if not parts or not all(p for p in parts):
        raise ValueError(
            f"parameter_key '{dot_path}' is not a valid dot-notation path "
            "(expected non-empty dot-separated segments, e.g. 'parameters.multipliers.ks_factor')"
        )

    current: object = MOBIDICConfig
    for i, part in enumerate(parts):
        if not (isinstance(current, type) and issubclass(current, BaseModel)):
            joined = ".".join(parts[:i]) or "<root>"
            raise ValueError(
                f"parameter_key '{dot_path}': cannot descend into '{part}' because "
                f"'{joined}' is not a Pydantic model in MOBIDICConfig schema"
            )
        if part not in current.model_fields:
            joined = ".".join(parts[:i]) or "<root>"
            valid = sorted(current.model_fields.keys())
            raise ValueError(
                f"parameter_key '{dot_path}': '{part}' is not a field of "
                f"{current.__name__} (at '{joined}'). Valid fields: {valid}"
            )
        current = _unwrap_optional(current.model_fields[part].annotation)

    if isinstance(current, type) and issubclass(current, BaseModel):
        raise ValueError(
            f"parameter_key '{dot_path}' resolves to nested model '{current.__name__}', "
            "not a scalar field. Provide a dot-path to a numeric leaf parameter."
        )
    if current is not float and current is not int:
        raise ValueError(
            f"parameter_key '{dot_path}' resolves to type {current!r}, but calibration "
            "parameters must point to a numeric (float or int) field in MOBIDICConfig"
        )


class CalibrationParameter(BaseModel):
    """A single parameter to be calibrated by PEST++."""

    name: str = Field(..., description="Parameter name (used in PEST++ control file; normalized to lower case)")
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
    def normalize_name(cls, v: str) -> str:
        """PEST++ parameter names cannot contain spaces, and are lower-cased.

        PEST++ itself is case-insensitive (it upper-cases every name it reads,
        including those of an ASCII matrix), but pyemu is not consistent:
        ``Matrix`` lower-cases its row and column names while
        ``Pst.parameter_data`` keeps the spelling of the control file. Anything
        that pairs the two - the prior ensemble draw of ``da.states.estimate``,
        which builds a ``Cov`` from the parameter data - then fails on a
        mixed-case name with "cov names are not in mean_values". Lower case is
        the one spelling both agree on.

        ``parameter_key`` is a path into the MOBIDIC schema and keeps its case;
        only the PEST-facing name is normalized.
        """
        if " " in v:
            raise ValueError("Parameter name cannot contain spaces")
        return v.lower()

    @field_validator("parameter_key")
    @classmethod
    def check_parameter_key_resolves(cls, v: str) -> str:
        """Validate that parameter_key points to a numeric field in MOBIDICConfig."""
        _validate_mobidic_parameter_path(v)
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

    metric: str = Field(
        ...,
        description="Metric name. Custom: nse, nse_log, pbias, peak_error. "
        "Plus all HydroErr metrics (rmse, kge, kge_2009, kge_2012, mle, mae, mape, "
        "r_squared, pearson_r, ve, d, d1, ...). See METRIC_REGISTRY for the full list.",
    )
    target: float = Field(..., description="Target value PEST++ tries to match (e.g., 1.0 for NSE)")
    weight: float = Field(1.0, description="Observation weight for this metric")

    @field_validator("metric")
    @classmethod
    def check_metric_name(cls, v: str) -> str:
        """Validate metric name is supported."""
        from mobidic.calibration.metrics import METRIC_REGISTRY

        if v not in METRIC_REGISTRY:
            raise ValueError(f"Unsupported metric '{v}'. Supported: {sorted(METRIC_REGISTRY.keys())}")
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
    weight: float = Field(
        1.0,
        description="Constant weight applied to every observation in this group. In PEST a weight is "
        "the reciprocal of the observation standard deviation, so weight=1/sigma. Ignored when "
        "relative_error is set.",
    )
    relative_error: Optional[float] = Field(
        None,
        gt=0.0,
        description="Observation standard deviation as a fraction of the observed value "
        "(e.g. 0.1 = 10%), giving weight_i = 1 / max(relative_error * |obs_i|, min_error). "
        "Preferable to a constant weight for a flood hydrograph spanning orders of magnitude: "
        "it stops the low-flow tail from dominating the objective function. "
        "When None (default), the constant `weight` is used.",
    )
    min_error: float = Field(
        0.0,
        ge=0.0,
        description="Floor on the observation standard deviation, in the units of the observations. "
        "Only used together with relative_error; it keeps near-zero observations from receiving an "
        "unbounded weight.",
    )
    time_column: str = Field("time", description="Column name for timestamps in obs_file")
    value_column: str = Field(..., description="Column name for observed values in obs_file")
    metrics: Optional[list[MetricConfig]] = Field(None, description="Optional derived metrics as pseudo-observations")

    @model_validator(mode="after")
    def check_error_model(self) -> "ObservationGroup":
        """A relative error model needs a positive floor to stay finite at zero flow."""
        if self.relative_error is not None and self.min_error <= 0.0:
            raise ValueError(
                f"Observation group '{self.name}': min_error must be > 0 when relative_error is set, "
                "otherwise an observed value of zero would receive an infinite weight"
            )
        return self

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
        """Validate that date strings are parseable as YYYY-MM-DD or YYYY-MM-DD HH:MM:SS."""
        import re

        if not re.match(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?$", v):
            raise ValueError(f"Invalid date format: '{v}'. Expected 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.")
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


class DAStateConfig(BaseModel):
    """Dynamic-state configuration for sequential data assimilation."""

    restart_from: Literal["warmup", "previous_cycle"] = Field(
        "warmup",
        description="How each cycle obtains its initial state. "
        "'warmup': re-simulate from the end of the warm-up every cycle (ensemble Kalman "
        "smoother); no states pass through the PEST interface. "
        "'previous_cycle': carry the previous cycle's state in a state file named by a scalar "
        "state identifier (ensemble Kalman filter); required to launch a forecast from an "
        "analysed state.",
    )

    # --- state files; used only when restart_from='previous_cycle' ---
    state_file_dir: str = Field(
        "da_states",
        description="Directory holding the state files, relative to the working directory. "
        "Must be readable and writable by every PEST++ agent. Resolved to an absolute path "
        "at setup time, since the forward model executes inside each agent's run directory.",
    )
    keep_cycles: int = Field(
        2,
        ge=2,
        description="Past cycles of state files to keep. 2 is enough for a normal run; "
        "increase it if a restart with hotstart_cycle is planned.",
    )

    # --- joint state-parameter estimation (formulation 2) ---
    estimate: list[str] = Field(
        default_factory=list,
        description="State variables the filter adjusts alongside the parameters.",
    )
    estimate_reaches: Literal["upstream", "all"] = Field(
        "upstream",
        description="Which reaches' discharge states are estimated.",
    )
    zones: Literal["reach"] = Field(
        "reach",
        description="Aggregation of soil-moisture cells into zones for joint state-parameter estimation.",
    )
    min_zone_cells: int = Field(
        1,
        ge=1,
        description="Soil-moisture zones holding fewer than this many cells are merged into the "
        "nearest downstream zone that is large enough.",
    )
    saturation_bounds: tuple[float, float] = Field(
        (0.0, 1.0),
        description="Bounds on a zone-averaged soil saturation.",
    )
    f0_bounds: tuple[float, float] = Field(
        (1.0e-6, 0.95),
        description="Bounds on a per-zone runoff fraction f0.",
    )
    conductivity_bounds: tuple[float, float] = Field(
        (0.1, 10.0),
        description="Bounds on a per-zone multiplier of the soil hydraulic conductivity ks. ",
    )
    zone_parameter_prior_std: float = Field(
        0.3,
        gt=0.0,
        description="Cycle-0 prior standard deviation of a distributed zone parameter, relative "
        "to its initial value.",
    )
    saturation_prior_std: float = Field(
        0.05,
        gt=0.0,
        description="Standard deviation of the cycle-0 prior on a soil-moisture state, in "
        "saturation units.",
    )
    bound_factor: float = Field(
        10.0,
        gt=1.0,
        description="Upper bound on an estimated discharge or surface-water state.",
    )
    prior_std: float = Field(
        0.1,
        gt=0.0,
        description="Standard deviation of the cycle-0 prior on each state parameter, relative to "
        "its initial value.",
    )
    prior_std_floor: float = Field(
        1.0e-3,
        gt=0.0,
        description="Absolute floor on the cycle-0 prior standard deviation of a *discharge* "
        "state [m3/s], so reaches that are dry at the end of the warm-up still receive a "
        "non-degenerate prior.",
    )
    surface_prior_std_floor: float = Field(
        1.0e-4,
        gt=0.0,
        description="The same floor for a *surface-water* state in meters.",
    )
    state_floor: float = Field(
        1.0e-30,
        gt=0.0,
        description="Lower bound of a discharge [m3/s] or surface-water [m] state.",
    )
    localizer: Literal["upstream", "none"] = Field(
        "upstream",
        description="Localization of the ensemble update.",
    )

    @field_validator("estimate")
    @classmethod
    def check_estimate(cls, v: list[str]) -> list[str]:
        """Validate the estimated state variables and reject overlapping entries."""
        from mobidic.calibration.da_states import resolve_estimate_kinds

        if len(set(v)) != len(v):
            raise ValueError(f"da.states.estimate contains duplicates: {v}")
        # Also rejects an unknown name, and 'soil_moisture' listed alongside one
        # of the two stores it already stands for.
        resolve_estimate_kinds(v)
        return v

    @field_validator("saturation_bounds")
    @classmethod
    def check_saturation_bounds(cls, v: tuple[float, float]) -> tuple[float, float]:
        """A saturation is a fraction, and the interval must not be degenerate."""
        lower, upper = float(v[0]), float(v[1])
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(f"da.states.saturation_bounds {v} must satisfy 0 <= lower < upper <= 1")
        return (lower, upper)

    @field_validator("f0_bounds", "conductivity_bounds")
    @classmethod
    def check_zone_parameter_bounds(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Bounds must be positive and non-degenerate (the lower one is also the denormal guard)."""
        lower, upper = float(v[0]), float(v[1])
        if not 0.0 < lower < upper:
            raise ValueError(f"bounds {v} must satisfy 0 < lower < upper")
        return (lower, upper)

    @model_validator(mode="after")
    def check_estimate_needs_state_files(self) -> "DAStateConfig":
        """Formulation 2 has to carry the full state, so it needs the EnKF restart mode."""
        if self.estimate and self.restart_from != "previous_cycle":
            raise ValueError(
                f"da.states.estimate={self.estimate} requires restart_from='previous_cycle' "
                f"(got '{self.restart_from}'). An adjusted state only means something when it is "
                "carried into the next cycle, which is what the state files do."
            )
        return self


class DAConfig(BaseModel):
    """PESTPP-DA specific configuration."""

    mode: Literal["batch", "sequential"] = Field(
        "sequential",
        description="'batch': no cycle information, PESTPP-DA behaves exactly like PESTPP-IES. "
        "'sequential': cycle-based data assimilation.",
    )
    cycle_length: Optional[str] = Field(
        None, description="Cycle length as a pandas offset string (e.g. '6h'); required in sequential mode"
    )
    warmup_period: Optional[CalibrationPeriod] = Field(
        None,
        description="Deterministic warm-up run performed once at setup time to produce the "
        "cycle-0 initial state. Required when states.restart_from='warmup'.",
    )
    assimilate: Literal["end", "all"] = Field(
        "all", description="'end': only the last timestep of a cycle is weighted; 'all': every timestep"
    )
    forecast_cycles: int = Field(0, ge=0, description="Trailing cycles with all weights zeroed (pure forecast)")
    states: DAStateConfig = Field(default_factory=DAStateConfig)
    hotstart_cycle: Optional[int] = Field(None, ge=0, description="Cycle to (re)start from")
    stop_cycle: Optional[int] = Field(None, ge=0, description="Last cycle to process")

    @field_validator("cycle_length")
    @classmethod
    def check_cycle_length(cls, v: str | None) -> str | None:
        """Validate that cycle_length parses as a positive pandas offset."""
        if v is None:
            return v
        try:
            delta = pd.Timedelta(v)
        except ValueError as exc:
            raise ValueError(f"cycle_length '{v}' is not a valid pandas offset string (e.g. '6h', '24h')") from exc
        if delta <= pd.Timedelta(0):
            raise ValueError(f"cycle_length '{v}' must be positive")
        return v

    @model_validator(mode="after")
    def check_cycle_range(self) -> "DAConfig":
        """stop_cycle must not precede hotstart_cycle."""
        if self.hotstart_cycle is not None and self.stop_cycle is not None and self.stop_cycle < self.hotstart_cycle:
            raise ValueError(f"da.stop_cycle ({self.stop_cycle}) must be >= da.hotstart_cycle ({self.hotstart_cycle})")
        return self

    @model_validator(mode="after")
    def check_estimate_needs_warmup(self) -> "DAConfig":
        """Formulation 2 needs a warm-up: it is where each state's prior comes from."""
        if self.states.estimate and self.warmup_period is None:
            raise ValueError(
                f"da.states.estimate={self.states.estimate} requires da.warmup_period. The warm-up "
                "supplies both the initial value of every state parameter and the state that cycle 0 "
                "starts from; without it the filter would have nothing to centre the prior on."
            )
        return self


class ParallelConfig(BaseModel):
    """Configuration for parallel PEST++ execution."""

    num_workers: Optional[int] = Field(None, description="Workers per node (null = all available CPUs)")
    port: int = Field(4004, description="TCP port for manager-agent communication")
    manager_ip: Optional[str] = Field(None, description="Manager IP for cluster mode (None = local mode)")

    @field_validator("num_workers")
    @classmethod
    def check_num_workers(cls, v: int | None) -> int | None:
        """Validate num_workers is positive and does not exceed available CPUs."""
        if v is None:
            return v
        if v <= 0:
            raise ValueError("num_workers must be a positive integer or null (all CPUs)")
        available = os.cpu_count()
        if available is not None and v > available:
            logger.warning(f"num_workers={v} exceeds available CPUs ({available}). Consider reducing it.")
        return v

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

    pest_tool: Literal["glm", "ies", "sen", "swp", "da", "opt", "mou", "sqp"] = Field(
        "glm", description="PEST++ tool to use"
    )

    case_name: Optional[str] = Field(
        None,
        description="File name prefix for PEST++ output files (e.g. 'calibration', 'sensitivity', 'sweep'). "
        "Defaults to 'sensitivity' when pest_tool='sen', 'sweep' when pest_tool='swp', 'calibration' otherwise.",
    )

    pest_options: Optional[dict] = Field(
        default_factory=dict,
        description="PEST++ and tool-specific options (e.g., noptmax, "
        "relparmax, facparmax, pst_version, ies_num_reals, sweep_parameter_csv_file). "
        "sweep_parameter_csv_file is required when pest_tool='swp'.",
    )

    working_dir: str = Field("pest_run", description="Working directory for PEST++ files")

    parallel: Optional[ParallelConfig] = Field(default_factory=ParallelConfig)

    da: DAConfig = Field(
        default_factory=DAConfig,
        description="PESTPP-DA (data assimilation) settings; used only when pest_tool='da'",
    )

    @property
    def is_sequential_da(self) -> bool:
        """True when this configuration describes cycle-based (sequential) data assimilation."""
        return self.pest_tool == "da" and self.da.mode == "sequential"

    @model_validator(mode="after")
    def set_case_name_default(self) -> "CalibrationConfig":
        """Default case_name per tool: 'sensitivity' (sen), 'sweep' (swp), 'assimilation' (da)."""
        if self.case_name is None:
            if self.pest_tool == "sen":
                self.case_name = "sensitivity"
            elif self.pest_tool == "swp":
                self.case_name = "sweep"
            elif self.pest_tool == "da":
                self.case_name = "assimilation"
            else:
                self.case_name = "calibration"
        return self

    @model_validator(mode="after")
    def check_da_settings(self) -> "CalibrationConfig":
        """Validate the PESTPP-DA section and its interaction with the rest of the config."""
        if self.pest_tool != "da":
            if "da" in self.model_fields_set:
                logger.warning(f"A 'da' section is configured but pest_tool='{self.pest_tool}'; it will be ignored.")
            return self

        da = self.da
        if da.mode != "sequential":
            return self

        if da.cycle_length is None:
            raise ValueError("da.cycle_length is required when pest_tool='da' and da.mode='sequential'")
        if self.simulation_period is None:
            raise ValueError("simulation_period is required when pest_tool='da' and da.mode='sequential'")

        if da.states.restart_from == "warmup" and da.warmup_period is None:
            raise ValueError(
                "da.warmup_period is required when da.states.restart_from='warmup' "
                "(there is nothing else to restart each cycle from)"
            )

        if da.warmup_period is not None:
            warmup_end = pd.Timestamp(da.warmup_period.end_date)
            sim_start = pd.Timestamp(self.simulation_period.start_date)
            if warmup_end > sim_start:
                raise ValueError(
                    f"da.warmup_period.end_date ({da.warmup_period.end_date}) must be at or before "
                    f"simulation_period.start_date ({self.simulation_period.start_date})"
                )

        groups_with_metrics = [o.name for o in self.observations if o.metrics]
        if groups_with_metrics:
            raise ValueError(
                f"Derived metrics are not supported in sequential DA (observation group(s): "
                f"{groups_with_metrics}); a metric computed over a single cycle window is not "
                "meaningful. Remove the 'metrics' entries or use pest_tool='ies'."
            )

        pst_version = (self.pest_options or {}).get("pst_version")
        if pst_version is not None and int(pst_version) != 2:
            raise ValueError(
                f"pest_options.pst_version must be 2 for sequential DA (got {pst_version}): the "
                "'cycle' and 'state_par_link' columns only exist in version 2 external files"
            )

        return self

    @model_validator(mode="after")
    def check_sweep_file_present(self) -> "CalibrationConfig":
        """Require pest_options.sweep_parameter_csv_file when pest_tool='swp'."""
        if self.pest_tool == "swp" and not (self.pest_options or {}).get("sweep_parameter_csv_file"):
            raise ValueError("pest_options.sweep_parameter_csv_file is required when pest_tool='swp'")
        return self

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
