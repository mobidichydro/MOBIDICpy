# MOBIDICpy examples

This directory contains example scripts and data for running MOBIDIC hydrological simulations.

The example folder is organised as follows:


- `00-numerical_validation/`: Scripts and data for validating MOBIDICpy numerical implementation against original MATLAB implementation.

- `01-event-Arno-basin/`: A complete workflow for a flood event in the Arno River basin, including calibration and sensitivity analysis examples

- `datasets/`: Example input data, including GIS rasters, river network shapefile, meteorological data, observed discharge, and reservoir information.

- `demos/`: Python scripts demonstrating specific features of MOBIDICpy, such as parser usage, hillslope to reach mapping, river network processing, and a complete pre-processing workflow.


## Available examples

### 01a — Basic simulation

The `01a_run_example_Arno.py` script demonstrates the complete MOBIDICpy workflow using data from the Arno River basin in Italy, for a flood event occurred in November 2023. It serves as a comprehensive example of how to set up and run a simulation from raw data to final outputs.

**Location:** `examples/01-event-Arno-basin/01a_run_example_Arno.py`

**What it does:**

1. Loads configuration from `Arno.yaml`
2. Preprocesses GIS data (DEM, flow direction, soil parameters, river network)
3. Converts meteorological data from MATLAB `.mat` to CF-compliant NetCDF
4. Runs simulation over the forcing period
5. Saves final model state and discharge time series
6. Plots an example discharge hydrograph

### 01b — Validation: Python vs MATLAB

The `01b_run_example_Arno_plots.py` script validates the Python implementation against MATLAB reference outputs by comparing discharge time series.

**Location:** `examples/01-event-Arno-basin/01b_run_example_Arno_plots.py`

**Requirements:**
- Python discharge output from `01a_run_example_Arno.py`
- MATLAB reference discharge output in `datasets/Arno_event_Nov_2023/output/matlab/discharge.csv`

### 02 — Raster forcing comparison

The `02_run_example_Arno_raster_forcing.py` script demonstrates the raster-based meteorological forcing workflow and validates that it produces identical results to station-based forcing.

**Location:** `examples/01-event-Arno-basin/02_run_example_Arno_raster_forcing.py`

### 03 — Restart capability

The `03_run_example_Arno_restart.py` script demonstrates the simulation restart capability by running a simulation in two stages and comparing against a continuous run.

**Location:** `examples/01-event-Arno-basin/03_run_example_Arno_restart.py`

**Purpose:**

- Demonstrates how to save and load intermediate simulation states
- Validates that restarted simulations produce identical results to continuous runs
- Illustrates multi-stage modeling workflows

### 04a — Reservoir model

The `04a_run_example_Arno_reservoirs.py` script demonstrates reservoir modeling capabilities.

**Location:** `examples/01-event-Arno-basin/04a_run_example_Arno_reservoirs.py`

### 04b — Reservoir validation plots

The `04b_run_example_Arno_reservoirs_plots.py` script validates reservoir results against MATLAB reference outputs.

**Location:** `examples/01-event-Arno-basin/04b_run_example_Arno_reservoirs_plots.py`

### 05 — GLM calibration (PEST++)

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

### 06 — IES ensemble calibration (PEST++)

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

### 07 — Morris global sensitivity analysis (PEST++)

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

## Quick start

### Prerequisites

1. Install MOBIDICpy with all dependencies:
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
```

## Additional resources

- **User guide**: See `docs/` for detailed documentation
- **API reference**: See `docs/api/` for function descriptions
- **PEST++ documentation**: [PEST++ Users Guide](https://github.com/usgs/pestpp)
- **GitHub issues**: Report bugs or request features

## Citation

If you use MOBIDICpy in your research, please cite:

[Citation information will be added]
