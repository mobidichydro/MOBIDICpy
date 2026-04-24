# MOBIDICpy examples

This directory contains example scripts and data for running MOBIDIC hydrological simulations.

The example folder is organised as follows:


- `00-numerical_validation/`: Scripts and data for validating MOBIDICpy numerical implementation against original MATLAB implementation.

- `01-event-Arno-basin/`: A complete workflow for a flood event in the Arno River basin, including calibration and sensitivity analysis examples

- `02-daily-balance-Arno-basin/`: Example of continuous hydrological balance simulation over the Arno River basin for 2017–2018, including energy balance (1L scheme) and linear groundwater model, and reservoirs.

- `datasets/`: Example input data, including GIS rasters, river network shapefile, meteorological data, observed discharge, and reservoir information.

- `demos/`: Python scripts demonstrating specific features of MOBIDICpy, such as parser usage, hillslope to reach mapping, river network processing, and a complete pre-processing workflow.


## 1) Arno River basin (November 2023 flood event)

### 1.1a — Basic simulation

The `01a_run_example_Arno.py` script demonstrates the complete MOBIDICpy workflow using data from the Arno River basin in Italy, for a flood event occurred in November 2023. It serves as a comprehensive example of how to set up and run a simulation from raw data to final outputs.

**Location:** `examples/01-event-Arno-basin/01a_run_example_Arno.py`

**What it does:**

1. Loads configuration from `Arno.yaml`
2. Preprocesses GIS data (DEM, flow direction, soil parameters, river network)
3. Converts meteorological data from MATLAB `.mat` to CF-compliant NetCDF
4. Runs simulation over the forcing period
5. Saves final model state and discharge time series
6. Plots an example discharge hydrograph

### 1.1b — Validation: Python vs MATLAB

The `01b_run_example_Arno_plots.py` script validates the Python implementation against MATLAB reference outputs by comparing discharge time series.

**Location:** `examples/01-event-Arno-basin/01b_run_example_Arno_plots.py`

**Requirements:**
- Python discharge output from `01a_run_example_Arno.py`
- MATLAB reference discharge output in `datasets/Arno_event_Nov_2023/output/matlab/discharge.csv`

### 1.2 — Raster forcing comparison

The `02_run_example_Arno_raster_forcing.py` script demonstrates the raster-based meteorological forcing workflow and validates that it produces identical results to station-based forcing.

**Location:** `examples/01-event-Arno-basin/02_run_example_Arno_raster_forcing.py`

### 1.3 — Restart capability

The `03_run_example_Arno_restart.py` script demonstrates the simulation restart capability by running a simulation in two stages and comparing against a continuous run.

**Location:** `examples/01-event-Arno-basin/03_run_example_Arno_restart.py`

**Purpose:**

- Demonstrates how to save and load intermediate simulation states
- Validates that restarted simulations produce identical results to continuous runs
- Illustrates multi-stage modeling workflows

### 1.4a — Reservoir model

The `04a_run_example_Arno_reservoirs.py` script demonstrates reservoir modeling capabilities.

**Location:** `examples/01-event-Arno-basin/04a_run_example_Arno_reservoirs.py`

### 1.4b — Reservoir validation plots

The `04b_run_example_Arno_reservoirs_plots.py` script validates reservoir results against MATLAB reference outputs.

**Location:** `examples/01-event-Arno-basin/04b_run_example_Arno_reservoirs_plots.py`

### 1.5 — GLM calibration (PEST++)

The `05_calibrate_Arno_glm.py` script demonstrates gradient-based model calibration using `pestpp-glm` (Gauss-Levenberg-Marquardt).

**Location:** `examples/01-event-Arno-basin/05_calibrate_Arno_glm.py`

**What it does:**

1. Preprocesses GIS data and converts meteorological forcing to NetCDF
2. Initializes `PestSetup` from `Arno.calibration.yaml` and generates all PEST++ files
3. Runs `pestpp-glm` in parallel with configurable workers
4. Extracts optimal parameters, objective function history, and parameter sensitivities
5. Runs a validation simulation with optimal parameters
6. Plots observed vs simulated discharge (with NSE/KGE metrics) and objective function convergence

**Output:**

- Optimal parameter values (alpha, beta, gamma, kappa)
- Objective function history by iteration
- Parameter sensitivity from the Jacobian
- Plot discharge comparison and convergence curve

### 1.6 — IES ensemble calibration (PEST++)

The `06_calibrate_Arno_ies.py` script demonstrates ensemble-based calibration using `pestpp-ies` (Iterative Ensemble Smoother).

**Location:** `examples/01-event-Arno-basin/06_calibrate_Arno_ies.py`

**Settings at the top of the script:**
```python
noptmax = 3          # Maximum optimization iterations
ies_num_reals = 20   # Number of ensemble members
```

**What it does:**

1. Preprocesses GIS data and converts meteorological forcing to NetCDF
2. Overrides `pest_tool` to `ies` and configures ensemble settings
3. Runs `pestpp-ies` (ensemble generated automatically by PEST++)
4. Extracts optimal parameters and ensemble statistics for the objective function
5. Runs a validation simulation with optimal parameters
6. Plots observed vs simulated discharge and ensemble objective function history (mean ± std)

**Output:**

- Optimal parameters from the ensemble mean
- Objective function history with ensemble spread (mean ± std per iteration)
- Two-panel plot: discharge comparison and ensemble convergence

### 1.7 — Morris global sensitivity analysis (PEST++)

The `07_sensitivity_Arno_Morris.py` script demonstrates global sensitivity analysis using `pestpp-sen` with the Morris one-at-a-time (OAT) method.

**Location:** `examples/01-event-Arno-basin/07_sensitivity_Arno_Morris.py`

**Settings at the top of the script:**
```python
gsa_morris_r = 10    # Number of Morris trajectories (samples)
```

**What it does:**

1. Preprocesses GIS data and converts meteorological forcing to NetCDF
2. Overrides `pest_tool` to `sen` and configures Morris options
3. Runs `pestpp-sen` in parallel across all available CPUs
4. Reads Morris sensitivity indices (mean, mean_abs, std_dev) from `.msn` output file

**Output:**

- Table of Morris sensitivity indices per parameter (μ*, μ, σ)
- Higher μ* indicates greater influence on model output

---

## 2) Arno River basin (daily water balance 2017–2018)

Example of two-year (2017–2018) continuous hydrological balance simulation over the Arno River basin, using a daily timestep and including energy balance (1L scheme), linear groundwater model, and reservoirs. Configuration files are in `examples/02-daily-balance-Arno-basin/`.

### 2.1a — Daily balance simulation

The `01a_run_example_Arno_daily.py` script runs a full daily MOBIDIC simulation.

**Location:** `examples/02-daily-balance-Arno-basin/01a_run_example_Arno_daily.py`

**What it does:**

1. Loads configuration from `Arno.daily.yaml`
2. Preprocesses GIS data or loads previously preprocessed data (controlled by `force_preprocessing` flag)
3. Converts meteorological data from MATLAB `.mat` to CF-compliant NetCDF
4. Runs a daily simulation over 2017–2018 with energy balance (1L) and linear groundwater model
5. Saves discharge and lateral inflow reports for selected reaches
6. Plots a discharge hydrograph at a specific reach and network-wide statistics

**Key configuration (`Arno.daily.yaml`):**

| Setting | Value |
|---------|-------|
| Timestep | 86400 s (daily) |
| Energy balance | 1L |
| Groundwater model | Linear |
| Reservoirs | Enabled (seasonal regulation) |

### 2.1b — Validation: Python vs MATLAB and observed data

The `01b_run_example_Arno_daily_plots.py` script validates daily simulation results against MATLAB reference outputs and observed discharge.

**Location:** `examples/02-daily-balance-Arno-basin/01b_run_example_Arno_daily_plots.py`

**Requirements:**
- Python output from `01a_run_example_Arno_daily.py`
- MATLAB reference outputs in `datasets/Arno/matlab/output/Arno_daily_balance_2017_2018/`
- Observed discharge in `datasets/Arno/data/Q_TOS01004659_2017_2018.parquet`

**What it does:**

1. Compares discharge and lateral inflow time series against MATLAB reference
2. Computes RMSE and bias per reach
3. Compares simulated vs observed discharge at reach 278 (Nave di Rosano, uncalibrated)
4. Produces time series and scatter plots for each matched reach

---

## Quick start

### Prerequisites

Install MOBIDICpy with all dependencies:
   ```bash
   # Base package
   pip install -e .

   # For calibration and sensitivity analysis examples (05–07)
   make install-calib
   # or manually:
   pip install "mobidicpy[calibration]" && get-pestpp :pyemu
   ```
   Make sure the `pestpp-glm`, `pestpp-ies`, and `pestpp-sen` executables are on the `PATH` of the system.

### Running the examples

```bash
# Basic simulation
python examples/01-event-Arno-basin/01a_run_example_Arno.py

# Validation plots (requires 01a output)
python examples/01-event-Arno-basin/01b_run_example_Arno_plots.py

# Raster forcing comparison
python examples/01-event-Arno-basin/02_run_example_Arno_raster_forcing.py

# Restart capability
python examples/01-event-Arno-basin/03_run_example_Arno_restart.py

# Reservoir model
python examples/01-event-Arno-basin/04a_run_example_Arno_reservoirs.py

# GLM calibration
python examples/01-event-Arno-basin/05_calibrate_Arno_glm.py

# IES ensemble calibration
python examples/01-event-Arno-basin/06_calibrate_Arno_ies.py

# Morris sensitivity analysis
python examples/01-event-Arno-basin/07_sensitivity_Arno_Morris.py

# Daily balance simulation (2017-2018)
python examples/02-daily-balance-Arno-basin/01a_run_example_Arno_daily.py

# Daily balance validation plots (requires 02a output)
python examples/02-daily-balance-Arno-basin/01b_run_example_Arno_daily_plots.py
```

## Additional resources

- **User guide**: See [Documentation](https://mobidichydro.github.io/MOBIDICpy/) for detailed documentation
- **API reference**: See [API Reference](https://mobidichydro.github.io/MOBIDICpy/latest/reference/) for function descriptions
- **PEST++ documentation**: [PEST++ Github repository](https://github.com/usgs/pestpp)
- **GitHub issues**: [Report bugs or request features](https://github.com/mobidichydro/MOBIDICpy/issues)


### Data sources
The Arno River basin datasets used in these examples were obtained from the following sources:
- DEM, soil parameters, Corine Land Cover, and river network shapefile: Tuscany Regional Geoportal ([GEOscopio](https://www.regione.toscana.it/-/geoscopio)).
- Flow direction and flow accumulation rasters were derived from the DEM using [GRASS](https://grass.osgeo.org/) function `r.watershed`.
- Discharge observations and meteorological forcing: Tuscany Regional Functional Centre ([Centro Funzionale Regionale, CFR](https://www.cfr.toscana.it/)), and Hydrological Service of Tuscany ([Servizio Idrologico della Toscana, SIR](https://www.sir.toscana.it/)).


## Citation

If you use MOBIDICpy in your research, please cite it using the metadata in [`CITATION.cff`](https://github.com/mobidichydro/MOBIDICpy/blob/main/CITATION.cff)
