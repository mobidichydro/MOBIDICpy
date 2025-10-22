"""Pydantic models for MOBIDIC configuration validation."""

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator, model_validator

# Custom type for path fields that accepts str or Path, and serializes (back to YAML) as str
PathField = Annotated[Union[str, Path], PlainSerializer(lambda x: str(x) if x else x, return_type=str)]


class BasinBaricenter(BaseModel):
    """Basin baricenter coordinates."""

    lon: float = Field(..., description="Longitude of basin baricenter [deg. East]")
    lat: float = Field(..., description="Latitude of basin baricenter [deg. North]")


class Basin(BaseModel):
    """Basin identification and metadata."""

    id: str = Field(..., description="Descriptive name of basin")
    paramset_id: str = Field(..., description="Descriptive name of parameter set / scenario")
    baricenter: BasinBaricenter


class Paths(BaseModel):
    """File paths for input and output data."""

    meteodata: PathField = Field(..., description="File where the meteo data files are stored")
    gisdata: PathField = Field(..., description="Consolidated dataset to be created by GIS preprocessing")
    network: PathField = Field(..., description="Consolidated hydrographic network to be created by GIS preprocessing")
    states: PathField = Field(..., description="Directory where the model states will be stored")
    output: PathField = Field(..., description="Directory where output report files will be stored")


class RiverNetworkVector(BaseModel):
    """River network shapefile configuration."""

    shp: PathField = Field(..., description="Shape of river network")


class VectorFiles(BaseModel):
    """Vector file paths and settings."""

    river_network: RiverNetworkVector


class RasterFiles(BaseModel):
    """Raster file paths."""

    dtm: PathField = Field(..., description="Grid of basin elevation in meters above sea level")
    flow_dir: PathField = Field(..., description="Grid of flow directions")
    flow_acc: PathField = Field(..., description="Grid of flow accumulation, as number of upstream cells")
    Wc0: PathField = Field(
        ..., description="Grid of maximum water holding capacity in soil small pores, in millimiters"
    )
    Wg0: PathField = Field(
        ..., description="Grid of maximum water holding capacity in soil large pores, in millimiters"
    )
    ks: PathField = Field(..., description="Grids of soil hydraulic conductivity, in millimiters per hour")
    kf: Optional[PathField] = Field(
        None, description="Grid of (real or ideal) aquifer conductivity, in meters per second"
    )
    CH: Optional[PathField] = Field(None, description="Grid of turbulent exchange coeff. for heat, non dimensional")
    Alb: Optional[PathField] = Field(None, description="Grid of surface albedo, non dimensional")
    Ma: Optional[PathField] = Field(
        None, description="Grid of binary mask (0,1) defining the artesian aquifer extension"
    )
    Mf: Optional[PathField] = Field(
        None, description="Grid of binary mask (0,1) defining the freatic aquifer extension"
    )
    gamma: Optional[PathField] = Field(None, description="Grid of percolation coefficient, in one over seconds")
    kappa: Optional[PathField] = Field(None, description="Grid of adsorption coefficient, in one over seconds")
    beta: Optional[PathField] = Field(None, description="Grid of hypodermic flow coefficient, in one over seconds")
    alpha: Optional[PathField] = Field(None, description="Grid of hillslope flow coefficient, in one over seconds")


class RasterSettings(BaseModel):
    """Raster processing settings."""

    flow_dir_type: Literal["Grass", "Arc"] = Field(
        ...,
        description="Flow direction pointer type: 'Grass' for 1-8 notation or 'Arc' for 1-2-4-8-16-32-64-128 notation",
    )


class SoilParameters(BaseModel):
    """Soil-related parameters."""

    Wc0: float = Field(
        ..., description="Default value of maximum water holding capacity in soil small pores, in millimiters"
    )
    Wg0: float = Field(
        ..., description="Default value of maximum water holding capacity in soil large pores, in millimiters"
    )
    ks: float = Field(..., description="Default value of soil hydraulic conductivity, in mm/h")
    ks_min: Optional[float] = Field(None, description="Default value of minimum soil hydraulic conductivity, in mm/h")
    ks_max: Optional[float] = Field(None, description="Default value of maximum soil hydraulic conductivity, in mm/h")
    kf: float = Field(..., description="Default value of (real or ideal) aquifer conductivity, in m/s")
    gamma: float = Field(..., description="Percolation coefficient, in 1/s")
    kappa: float = Field(..., description="Adsorption coefficient, in 1/s")
    beta: float = Field(..., description="Hypodermic flow coefficient, in 1/s")
    alpha: float = Field(..., description="Hillslope flow coefficient, in 1/s")

    @field_validator("Wc0", "Wg0", "ks", "kf", "gamma", "kappa", "beta", "alpha")
    @classmethod
    def check_positive(cls, v: float) -> float:
        """Validate that required parameters are non-negative."""
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v


class EnergyParameters(BaseModel):
    """Energy balance parameters."""

    Tconst: float = Field(..., description="Deep ground temperature, in deg. Kelvin")
    kaps: float = Field(..., description="Soil thermal conductivity, in W/m/K")
    nis: float = Field(..., description="Soil thermal diffusivity, in m²/s")
    CH: float = Field(..., description="Default value of turbulent exchange coeff. for heat, non dimensional")
    Alb: float = Field(..., description="Default value of surface albedo, non dimensional")

    @field_validator("Tconst", "kaps", "nis", "CH")
    @classmethod
    def check_positive(cls, v: float) -> float:
        """Validate that parameters are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("Alb")
    @classmethod
    def check_albedo_range(cls, v: float) -> float:
        """Validate that albedo is in valid range."""
        if not 0 <= v <= 1:
            raise ValueError("Albedo must be between 0 and 1")
        return v


class RoutingParameters(BaseModel):
    """Channel routing parameters."""

    method: Literal["Musk", "MuskCun", "Lag", "Linear"] = Field(..., description="Type of channel routing scheme")
    wcel: float = Field(..., description="Flood wave celerity in channels, in m/s")
    Br0: float = Field(..., description="Width of channels with first Strahler order, in meters")
    NBr: float = Field(
        ..., description="Exponent of equation W = Br0*O^NBr, where W=Width of channels and O=Strahler order"
    )
    n_Man: float = Field(..., description="Manning roughness coefficient for channels, in s/m^(1/3)")

    @field_validator("wcel", "Br0", "n_Man")
    @classmethod
    def check_positive(cls, v: float) -> float:
        """Validate that parameters are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("NBr")
    @classmethod
    def check_nbr_range(cls, v: float) -> float:
        """Validate that NBr is greater than 1."""
        if v <= 1:
            raise ValueError("NBr must be greater than 1")
        return v


class GroundwaterParameters(BaseModel):
    """Groundwater model parameters."""

    model: Literal["None", "Linear", "Linear_mult", "Dupuit", "MODFLOW"] = Field(
        ..., description="Groundwater model type"
    )
    global_loss: Optional[float] = Field(0.0, description="Global water loss from aquifers, in m³/s")

    @field_validator("global_loss")
    @classmethod
    def check_non_negative(cls, v: Optional[float]) -> float:
        """Validate that global_loss is non-negative."""
        if v is not None and v < 0:
            raise ValueError("global_loss must be non-negative")
        return v if v is not None else 0.0


class Multipliers(BaseModel):
    """Parameter multipliers for calibration."""

    ks_factor: Optional[float] = Field(1.0, description="Multiplying factor of soil hydraulic conductivity")
    Wc_factor: Optional[float] = Field(
        1.0, description="Multiplying factor of maximum water holding capacity in soil small pores"
    )
    Wg_factor: Optional[float] = Field(
        1.0, description="Multiplying factor of maximum water holding capacity in soil large pores"
    )
    Wg_Wc_tr: Optional[float] = Field(1.0, description="Transition factor between gravitational and capillary storage")
    CH_factor: Optional[float] = Field(1.0, description="Multiplying factor of turbulent exchange coeff. for heat")
    cel_factor: Optional[float] = Field(1.0, description="Multiplying factor for flood wave celerity")
    chan_factor: Optional[float] = Field(0.0, description="Scale factor for fraction of channalized flow")

    @field_validator("ks_factor", "Wc_factor", "Wg_factor", "Wg_Wc_tr", "CH_factor", "cel_factor")
    @classmethod
    def check_positive(cls, v: Optional[float]) -> float:
        """Validate that multipliers are positive."""
        if v is not None and v <= 0:
            raise ValueError("Multiplier must be positive")
        return v if v is not None else 1.0


class Parameters(BaseModel):
    """Global land and model parameters."""

    soil: SoilParameters
    energy: EnergyParameters
    routing: RoutingParameters
    groundwater: GroundwaterParameters
    multipliers: Optional[Multipliers] = Field(default_factory=Multipliers)


class InitialConditions(BaseModel):
    """Initial state conditions."""

    Ws: Optional[float] = Field(0.0, description="Initial depth of hillslope runoff, in meters")
    Wcsat: Optional[float] = Field(0.3, description="Initial relative saturation of capillary soil, non dimensional")
    Wgsat: Optional[float] = Field(
        0.01, description="Initial relative saturation of gravitational soil, non dimensional"
    )

    @field_validator("Ws")
    @classmethod
    def check_ws_non_negative(cls, v: Optional[float]) -> float:
        """Validate that Ws is non-negative."""
        if v is not None and v < 0:
            raise ValueError("Ws must be non-negative")
        return v if v is not None else 0.0

    @field_validator("Wcsat", "Wgsat")
    @classmethod
    def check_saturation_range(cls, v: Optional[float]) -> float:
        """Validate that saturation is in valid range [0, 1]."""
        if v is not None and not 0 <= v <= 1:
            raise ValueError("Saturation must be between 0 and 1")
        return v


class Simulation(BaseModel):
    """Simulation control parameters."""

    realtime: Literal[0, 1, -1] = Field(..., description="Option to wait for new data at end of computation")
    timestep: float = Field(..., description="Data and model time step, in seconds")
    resample: int = Field(
        ..., description="Degradation factor from grid data space resolution to model space resolution"
    )
    soil_scheme: Literal["Bucket", "CN"] = Field(..., description="Type of soil hydrology scheme")
    energy_balance: Literal["None", "1L", "5L", "Snow"] = Field(
        ..., description="Type of surface energy balance scheme"
    )

    @field_validator("timestep")
    @classmethod
    def check_timestep_positive(cls, v: float) -> float:
        """Validate that timestep is positive."""
        if v <= 0:
            raise ValueError("timestep must be positive")
        return v

    @field_validator("resample")
    @classmethod
    def check_resample_positive(cls, v: int) -> int:
        """Validate that resample is a positive integer."""
        if v <= 0:
            raise ValueError("resample must be a positive integer")
        return v


class OutputStates(BaseModel):
    """Output state options."""

    discharge: bool = Field(..., description="Option to save states of river network for results analysis")
    reservoir_states: bool = Field(
        ..., description="Option to save states of reservoirs and lakes for results analysis"
    )
    soil_capillary: bool = Field(..., description="Option to save states of soil small pores for results analysis")
    soil_gravitational: bool = Field(..., description="Option to save states of soil large pores for results analysis")
    surface_temperature: bool = Field(
        ..., description="Option to save states of land surface temperature for results analysis"
    )
    ground_temperature: bool = Field(
        ..., description="Option to save states of ground temperature for results analysis"
    )
    aquifer_head: bool = Field(..., description="Option to save states of aquifers for results analysis")
    et_prec: bool = Field(
        ..., description="Option to save states of evapotranspiration and precipitation for results analysis"
    )


class OutputStatesSettings(BaseModel):
    """Output state file settings."""

    output_format: Optional[Literal["netCDF"]] = Field("netCDF", description="Format for state output files")
    output_interval: Optional[float] = Field(None, description="Time interval for state output, in seconds")

    @field_validator("output_interval")
    @classmethod
    def check_interval_positive(cls, v: Optional[float]) -> Optional[float]:
        """Validate that output_interval is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("output_interval must be positive")
        return v


class OutputReport(BaseModel):
    """Output report options."""

    discharge: Optional[bool] = Field(True, description="Option to save discharge hydrograph at selected reach IDs")
    lateral_inflow: Optional[bool] = Field(
        False, description="Option to save lateral inflow hydrograph at selected reach IDs"
    )


class OutputReportSettings(BaseModel):
    """Output report file settings."""

    output_format: Optional[Literal["csv", "Parquet"]] = Field("Parquet", description="Format for report output files")
    report_interval: Optional[float] = Field(None, description="Time interval for report output, in seconds")
    reach_selection: Optional[Literal["all", "file", "list"]] = Field(
        "all", description="Method for selecting reaches to output"
    )
    sel_file: Optional[PathField] = Field(None, description="Path to JSON file containing reach IDs to output")
    sel_list: Optional[list[int]] = Field(None, description="List of reach IDs to output")

    @field_validator("report_interval")
    @classmethod
    def check_interval_positive(cls, v: Optional[float]) -> Optional[float]:
        """Validate that report_interval is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("report_interval must be positive")
        return v

    @model_validator(mode="after")
    def check_selection_consistency(self) -> "OutputReportSettings":
        """Validate that selection method matches provided data."""
        if self.reach_selection == "file" and self.sel_file is None:
            raise ValueError("sel_file must be provided when reach_selection='file'")
        if self.reach_selection == "list" and not self.sel_list:
            raise ValueError("sel_list must be provided when reach_selection='list'")
        return self


class Advanced(BaseModel):
    """Advanced settings."""

    log_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR"]] = Field("INFO", description="Logging level")
    log_file: Optional[PathField] = Field(None, description="Path to log file")


class MOBIDICConfig(BaseModel):
    """Complete MOBIDIC configuration."""

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on assignment
    )

    basin: Basin
    paths: Paths
    vector_files: VectorFiles
    raster_files: RasterFiles
    raster_settings: RasterSettings
    parameters: Parameters
    initial_conditions: Optional[InitialConditions] = Field(default_factory=InitialConditions)
    simulation: Simulation
    output_states: OutputStates
    output_states_settings: Optional[OutputStatesSettings] = Field(default_factory=OutputStatesSettings)
    output_report: Optional[OutputReport] = Field(default_factory=OutputReport)
    output_report_settings: Optional[OutputReportSettings] = Field(default_factory=OutputReportSettings)
    advanced: Optional[Advanced] = Field(default_factory=Advanced)
