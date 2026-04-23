# Examples

This page provides practical examples demonstrating the main features of MOBIDICpy. All example scripts are available in the `examples/` directory of the repository.


## 1) Arno River basin (November 2023 flood event)

All examples use data from the Arno River basin for a flood event that occurred in November 2023. The dataset is located in `examples/datasets/Arno_event_Nov_2023/` and includes raster data (DEM, flow direction, soil parameters), the river network shapefile, meteorological forcing (.mat), observed discharge, and reservoir data.

Configuration files for each workflow are in `examples/01-event-Arno-basin/`.

---

## 1.1a — Basic simulation

The complete MOBIDIC workflow from raw data to final outputs.

**Script**: `examples/01-event-Arno-basin/01a_run_example_Arno.py`

```python
from pathlib import Path
from mobidic import load_config, run_preprocessing, save_gisdata, save_network, MeteoData, Simulation

config = load_config("examples/01-event-Arno-basin/Arno.yaml")

# Preprocessing
gisdata = run_preprocessing(config)
save_gisdata(gisdata, config.paths.gisdata)
save_network(gisdata.network, config.paths.network, format="parquet")

# Convert meteorological data from MATLAB to NetCDF
meteo_data = MeteoData.from_mat("examples/datasets/Arno_event_Nov_2023/meteodata/meteodata.mat")
meteo_data.to_netcdf(config.paths.meteodata)

# Load forcing and run simulation
forcing = MeteoData.from_netcdf(config.paths.meteodata)
sim = Simulation(gisdata, forcing, config)
results = sim.run(start_date=forcing.start_date, end_date=forcing.end_date)
```

**What it demonstrates:**

- Loading YAML configuration files
- GIS preprocessing (DEM, flow direction, soil parameters, river network)
- Meteorological data conversion from MATLAB `.mat` to CF-compliant NetCDF
- Running a complete MOBIDIC simulation
- Saving discharge time series and model state

---

## 1.1b — Validation: Python vs MATLAB

Validates the Python implementation against MATLAB reference outputs.

**Script**: `examples/01-event-Arno-basin/01b_run_example_Arno_plots.py`

Compares discharge time series produced by MOBIDICpy against the MATLAB reference output at `examples/datasets/Arno_event_Nov_2023/output/matlab/discharge.csv`.

**Requirements**: Run `01a_run_example_Arno.py` first to generate the Python output.

---

## 1.2 — Station vs raster forcing comparison

Compare station-based forcing (with spatial interpolation) against pre-interpolated raster forcing.

**Script**: `examples/01-event-Arno-basin/02_run_example_Arno_raster_forcing.py`

```python
from mobidic import MeteoData, MeteoRaster, Simulation

# Run 1: Station-based with forcing output enabled
config.output_forcing_data.meteo_data = True
forcing_stations = MeteoData.from_netcdf(config.paths.meteodata)
sim1 = Simulation(gisdata, forcing_stations, config)
results1 = sim1.run(start_date, end_date)
# Interpolated grids saved to: output/meteo_forcing.nc

# Run 2: Raster-based (using exported interpolated data)
forcing_raster = MeteoRaster.from_netcdf("output/meteo_forcing.nc")
sim2 = Simulation(gisdata, forcing_raster, config)
results2 = sim2.run(start_date, end_date)
# Results are numerically identical (differences < 1e-6 m³/s)
```

**What it demonstrates:**

- Exporting interpolated meteorological data as raster forcing
- Using raster forcing for subsequent runs (skips interpolation, faster)
- Verifying that both approaches produce identical results

**Use cases:** Calibration runs, scenario analysis, large domains.

---

## 1.3 — Simulation restart capability

Run a simulation in two stages and compare against a continuous run.

**Script**: `examples/01-event-Arno-basin/03_run_example_Arno_restart.py`

```python
from pathlib import Path
from mobidic import load_config, load_gisdata, MeteoData, Simulation

config = load_config("examples/01-event-Arno-basin/Arno.yaml")
gisdata = load_gisdata(config.paths.gisdata, config.paths.network)
forcing = MeteoData.from_netcdf(config.paths.meteodata)

# First run: simulate to midpoint
sim1 = Simulation(gisdata, forcing, config)
results_1 = sim1.run(start_date=start_date, end_date=restart_point)
# States automatically saved to config.paths.states

# Second run: restart from saved state
sim2 = Simulation(gisdata, forcing, config)
state_file = Path(config.paths.states) / "states_001.nc"
sim2.set_initial_state(state_file=state_file, time_index=-1)
results_2 = sim2.run(start_date=restart_point, end_date=end_date)
```

**What it demonstrates:**

- Saving intermediate simulation states to NetCDF files
- Loading states from file using `set_initial_state()`
- Continuing simulations from saved checkpoints
- Validating restart accuracy against a continuous run

**Use cases:** Long-term simulations, checkpoint recovery, multi-stage modeling, real-time forecast updates.

---

## 1.4a — Reservoir routing

Simulate reservoirs with time-varying regulation curves.

**Script**: `examples/01-event-Arno-basin/04a_run_example_Arno_reservoirs.py`

```python
from mobidic import load_config, run_preprocessing, MeteoData, Simulation

config = load_config("examples/01-event-Arno-basin/Arno.reservoirs.yaml")
gisdata = run_preprocessing(config)  # Automatically processes reservoir data

forcing = MeteoData.from_netcdf(config.paths.meteodata)
sim = Simulation(gisdata, forcing, config)
results = sim.run(start_date=forcing.start_date, end_date=forcing.end_date)
```

**Configuration** (`Arno.reservoirs.yaml`):

```yaml
parameters:
  reservoirs:
    res_shape: reservoirs/reservoirs.shp
    stage_storage: reservoirs/stage_storage.csv
    regulation_curves: reservoirs/regulation_curves.csv
    regulation_schedule: reservoirs/regulation_schedule.csv

output_states:
  reservoir_states: true  # Save volume, stage, discharge
```

**What it demonstrates:**

- Reservoir preprocessing (polygon rasterization, inlet/outlet reach detection)
- Time-varying regulation curves (seasonal winter/summer operations)
- Reservoir state output (volume, stage, regulated discharge) to NetCDF

---

## 1.4b — Reservoir validation plots

Validates reservoir results against MATLAB reference outputs.

**Script**: `examples/01-event-Arno-basin/04b_run_example_Arno_reservoirs_plots.py`

**Requirements**: Run `04a_run_example_Arno_reservoirs.py` first.

---

## 1.5 — GLM calibration (PEST++)

Gradient-based model calibration using `pestpp-glm` (Gauss-Levenberg-Marquardt).

**Script**: `examples/01-event-Arno-basin/05_calibrate_Arno_glm.py`

**Prerequisites:**

```bash
make install-calib
# or: pip install "mobidicpy[calibration]" && get-pestpp :pyemu
# Ensure pestpp-glm is on PATH
```

```python
from mobidic.calibration import PestSetup
from mobidic.calibration.config import load_calibration_config

# Read calibration configuration
calib_config = load_calibration_config("Arno.calibration.yaml")

# Set up PEST++ and generate all working files
pest = PestSetup(calib_config)
working_dir = pest.setup()

# Run pestpp-glm
results = pest.run()

# Extract results
optimal = results.get_optimal_parameters()    # Optimal parameter values
phi_history = results.get_objective_function_history()  # Convergence
sens = results.get_parameter_sensitivities()  # Jacobian-based sensitivities
```

**What it demonstrates:**

- Setting up and running `pestpp-glm` from a single configuration file
- Parallel model runs with configurable number of workers
- Extracting optimal parameters, objective function history, and sensitivities
- Running a validation simulation with optimal parameters
- Plotting observed vs simulated discharge with NSE/KGE metrics

**Output:**

- Optimal parameter values (alpha, beta, gamma, kappa)
- Objective function convergence by iteration
- Parameter sensitivity from the Jacobian
- Discharge comparison plot and convergence curve

---

## 1.6 — IES ensemble calibration (PEST++)

Ensemble-based calibration using `pestpp-ies` (Iterative Ensemble Smoother).

**Script**: `examples/01-event-Arno-basin/06_calibrate_Arno_ies.py`

**Settings at the top of the script:**

```python
noptmax = 3          # Maximum optimization iterations
ies_num_reals = 20   # Number of ensemble members (generated automatically)
```

```python
from mobidic.calibration.config import CalibrationConfig, load_calibration_config
from mobidic.calibration import PestSetup

cc = load_calibration_config("Arno.calibration.yaml")

# Override tool to IES and set ensemble options
cc_dict = cc.model_dump()
cc_dict["pest_tool"] = "ies"
cc_dict["pest_options"] = {"noptmax": noptmax, "ies_num_reals": ies_num_reals}
cc_ies = CalibrationConfig(**cc_dict)

# Set up and run
pest = PestSetup(cc_ies, base_path=calib_config_path.parent)
pest.setup()
results = pest.run()

# Ensemble statistics
phi_history = results.get_objective_function_history()
# Returns DataFrame with columns: iteration, mean, std
```

**What it demonstrates:**

- Ensemble-based parameter estimation with automatically generated ensembles
- Ensemble statistics for the objective function (mean ± std per iteration)
- Plotting ensemble spread to visualise uncertainty reduction

**Output:**

- Optimal parameters from the ensemble mean
- Objective function history with ensemble spread (mean ± std per iteration)
- Discharge comparison plot and ensemble convergence curve

---

## 1.7 — Morris global sensitivity analysis (PEST++)

Global sensitivity analysis using `pestpp-sen` with the Morris OAT method.

**Script**: `examples/01-event-Arno-basin/07_sensitivity_Arno_Morris.py`

**Settings at the top of the script:**

```python
gsa_morris_r = 10    # Number of Morris trajectories
```

```python
from mobidic.calibration.config import CalibrationConfig, load_calibration_config
from mobidic.calibration import PestSetup

cc = load_calibration_config("Arno.calibration.yaml")

# Override tool to sensitivity analysis
cc_dict = cc.model_dump()
cc_dict["pest_tool"] = "sen"
cc_dict["pest_options"] = {
    "gsa_method": "morris",
    "gsa_morris_r": gsa_morris_r,
    "gsa_morris_obs_sen": False
}
cc_dict["parallel"] = {"num_workers": None}  # Use all available CPUs
cc_sen = CalibrationConfig(**cc_dict)

pest = PestSetup(cc_sen, base_path=calib_config_path.parent)
pest.setup()
results = pest.run()

# Read Morris sensitivity indices
sens = results.get_parameter_sensitivities()
# DataFrame with columns: parameter, mean, mean_abs, std_dev
```

**What it demonstrates:**

- Global sensitivity analysis with Morris OAT method
- Parallel model evaluation across all available CPUs
- Interpreting Morris indices (μ*, μ, σ)

**Output:**

- Table of Morris sensitivity indices per parameter (μ*, μ, σ)
- Higher μ* indicates greater influence on model output

---

## PEST++ calibration configuration

Examples 05–07 all share a single calibration configuration file `Arno.calibration.yaml`. Key sections:

```yaml
# PEST++ tool: glm | ies | sen | da | opt | mou | sqp
pest_tool: glm

# Full simulation window (includes warm-up period)
simulation_period:
  start_date: "2023-10-31 00:15:00"
  end_date: "2023-11-04 23:45:00"

# Only observations within this window contribute to the objective function
calibration_period:
  start_date: "2023-11-01 00:00:00"
  end_date: "2023-11-04 23:45:00"

# Rasterize forcing once and reuse for all model evaluations (faster)
use_raster_forcing: true

# Parameters (dot-notation keys into Arno.yaml)
parameters:
  - name: alpha
    parameter_key: parameters.soil.alpha
    initial_value: 8.0e-06
    lower_bound: 1.0e-08
    upper_bound: 1.0e-04
    transform: log          # log | none | fixed
    par_group: soil

# Observations with optional metric pseudo-observations
observations:
  - name: Q_292
    obs_file: "../datasets/Arno_event_Nov_2023/data/Q_Nave_di_Rosano_2023.csv"
    reach_id: 292
    weight: 1.0
    time_column: time
    value_column: Q_292
    metrics:
      - metric: nse
        target: 1.0
        weight: 10.0
      - metric: peak_error
        target: 0.0
        weight: 8.0

# Parallel execution (null = all available CPUs)
parallel:
  num_workers: 8
  port: 4004
```

See `examples/01-event-Arno-basin/Arno.calibration.yaml` for the fully annotated configuration.

---

## Design storm simulation with hyetograph generation

Generate synthetic design storm hyetographs from IDF parameters and run a design flood simulation.

```python
from datetime import datetime
from pathlib import Path
from mobidic import load_config, load_gisdata, Simulation
from mobidic.preprocessing.hyetograph import HyetographGenerator

config_file = Path("Arno_hyetograph.yaml")
config = load_config(config_file)
gisdata = load_gisdata(config.paths.gisdata, config.paths.network)

# Generate hyetograph forcing — all parameters read from config
forcing = HyetographGenerator.from_config(
    config=config,
    base_path=config_file.parent,
    start_time=datetime(2000, 1, 1)
)

sim = Simulation(gisdata, forcing, config)
results = sim.run(forcing.start_date, forcing.end_date)
```

**Configuration** (`Arno_hyetograph.yaml`):

```yaml
paths:
  hyetograph: output/design_storm.nc

hyetograph:
  a_raster: idf/a.tif        # IDF scale parameter
  n_raster: idf/n.tif        # IDF exponent
  k_raster: idf/k30.tif      # Return period factor (30-year event)
  duration_hours: 48
  timestep_hours: 1
  hyetograph_type: chicago_decreasing
  ka: 0.8                    # Areal reduction factor
```

**IDF formula:**

$$DDF(t) = k_a \cdot k \cdot a \cdot t^n$$

**What it demonstrates:**

- Generating synthetic design storms from spatially distributed IDF parameters
- Automatic resampling of IDF rasters to match model grid
- Chicago decreasing hyetograph method

---

---

## 2) Arno River basin (daily water balance 2017–2018)

All examples in this group use data from the Arno River basin for a continuous hydrological simulations covering two years (2017–2018), at a daily timestep. Configuration files are in `examples/02-daily-balance-Arno-basin/`.

---

## 2.1a — Daily balance simulation

Full MOBIDICpy workflow for a daily continuous hydrological balance simulation, including energy balance, groundwater, and reservoirs.

**Script**: `examples/02-daily-balance-Arno-basin/01a_run_example_Arno_daily.py`

```python
from pathlib import Path
from mobidic import load_config, run_preprocessing, save_gisdata, save_network, load_gisdata, MeteoData, Simulation

config_file = Path("examples/02-daily-balance-Arno-basin/Arno.daily.yaml")
config = load_config(config_file)

# Preprocessing (skipped if output already exists)
if not config.paths.gisdata.exists() or not config.paths.network.exists():
    gisdata = run_preprocessing(config)
    save_gisdata(gisdata, config.paths.gisdata)
    save_network(gisdata.network, config.paths.network, format="parquet")
else:
    gisdata = load_gisdata(config.paths.gisdata, config.paths.network)

# Convert meteorological data from MATLAB to NetCDF
meteo_data = MeteoData.from_mat("examples/datasets/Arno/matlab/meteodata/Arno_daily_balance_2017_2018.mat")
meteo_data.to_netcdf(config.paths.meteodata)

# Load forcing and run simulation
forcing = MeteoData.from_netcdf(config.paths.meteodata)
sim = Simulation(gisdata, forcing, config)
results = sim.run(start_date=forcing.start_date, end_date=forcing.end_date)
```

**What it demonstrates:**

- Daily timestep simulation (86400 s) over a two-year period (2017–2018)
- Energy balance scheme (`1L`) for PET computation
- Linear groundwater model
- Reservoir routing with seasonal regulation

---

## 2.1b — Validation: Python vs MATLAB and observed data

Validates daily simulation outputs against MATLAB reference and observed discharge.

**Script**: `examples/02-daily-balance-Arno-basin/01b_run_example_Arno_daily_plots.py`

Compares discharge and lateral inflow produced by MOBIDICpy against MATLAB reference outputs in `examples/datasets/Arno/matlab/output/Arno_daily_balance_2017_2018/`. Also compares simulated discharge at reach 278 (Nave di Rosano) against observed data (`Q_TOS01004659_2017_2018.parquet`).

**Requirements**: Run `01a_run_example_Arno_daily.py` first.

**What it demonstrates:**

- Time series and scatter-plot comparison against MATLAB reference (discharge and lateral inflow)
- Observed vs simulated discharge comparison (uncalibrated)

---

## Additional resources

### How to run the examples

```bash
# Basic simulation
python examples/01-event-Arno-basin/01a_run_example_Arno.py

# Validation plots (requires 01a output)
python examples/01-event-Arno-basin/01b_run_example_Arno_plots.py

# Station vs raster forcing
python examples/01-event-Arno-basin/02_run_example_Arno_raster_forcing.py

# Restart capability
python examples/01-event-Arno-basin/03_run_example_Arno_restart.py

# Reservoir model
python examples/01-event-Arno-basin/04a_run_example_Arno_reservoirs.py

# GLM calibration (requires PEST++ binaries)
python examples/01-event-Arno-basin/05_calibrate_Arno_glm.py

# IES ensemble calibration (requires PEST++ binaries)
python examples/01-event-Arno-basin/06_calibrate_Arno_ies.py

# Morris sensitivity analysis (requires PEST++ binaries)
python examples/01-event-Arno-basin/07_sensitivity_Arno_Morris.py

# Daily balance simulation (2017-2018)
python examples/02-daily-balance-Arno-basin/01a_run_example_Arno_daily.py

# Daily balance validation plots (requires 02a output)
python examples/02-daily-balance-Arno-basin/01b_run_example_Arno_daily_plots.py
```

### Data sources
The Arno River basin datasets used in these examples were obtained from the following sources:
- DEM, soil parameters, Corine Land Cover, and river network shapefile: Tuscany Regional Geoportal ([GEOscopio](https://www.regione.toscana.it/-/geoscopio)).
- Flow direction and flow accumulation rasters were derived from the DEM by GIS processing.
- Discharge observations and meteorological forcing: Tuscany Regional Functional Centre ([Centro Funzionale Regionale, CFR](https://www.cfr.toscana.it/)), and Hydrological Service of Tuscany ([Servizio Idrologico della Toscana, SIR](https://www.sir.toscana.it/)).



### See also

- [API Reference](reference/index.md) for detailed function documentation
- [Introduction](introduction.md) for model background and theory
- [Development Guide](development.md) for contributing
