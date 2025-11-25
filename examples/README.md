# MOBIDICpy examples

This directory contains example scripts and data for running MOBIDIC hydrological simulations.

## Available examples

### Arno River basin example

The `run_example_Arno.py` script demonstrates the complete MOBIDICpy workflow using data from the Arno River basin in Tuscany, Italy.

**Location:** `examples/run_example_Arno.py`

**Data:** `examples/Arno/`

### Validation: Python vs MATLAB

The `run_example_Arno_plots.py` script validates the Python implementation against MATLAB reference outputs by comparing discharge time series.

**Location:** `examples/run_example_Arno_plots.py`

**Requirements:**
- Python discharge output from `run_example_Arno.py`
- MATLAB reference discharge output (CSV format) in `Arno/output/matlab/discharge.csv`

**Purpose:** Regression testing to ensure the Python implementation produces results equivalent to the original MATLAB code.

### Restart capability demonstration

The `run_example_Arno_restart.py` script demonstrates the simulation restart capability by running a simulation in two stages and comparing against a continuous run.

**Location:** `examples/run_example_Arno_restart.py`

**Purpose:**
- Demonstrates how to save and load intermediate simulation states
- Shows how to continue simulations after interruption
- Validates that restarted simulations produce identical results to continuous runs
- Illustrates multi-stage modeling workflows

## Quick Start

### Prerequisites

1. Install MOBIDICpy in development mode:
   ```bash
   pip install -e .
   ```

2. Ensure you have the required dependencies:
   ```bash
   pip install matplotlib
   ```

### Running the Arno example

```bash
cd examples
python run_example_Arno.py
```

Or from the project root:

```bash
python examples/run_example_Arno.py
```

### Plotting the validation results

After running the main example, validate against MATLAB reference outputs:

```bash
python examples/run_example_Arno_plots.py
```

See [Validation against MATLAB reference](#validation-against-matlab-reference) for details.

### Running the restart example

Test the simulation restart capability:

```bash
python examples/run_example_Arno_restart.py
```

See [Restart from previous simulation state](#restart-from-previous-simulation-state) for details.

## What the example does

The `run_example_Arno.py` script performs the following steps:

1. **Configure logging**: Sets up logger with specified debug level

2. **Load configuration**: Reads the YAML configuration file (`Arno/Arno.yaml`)

3. **GIS preprocessing**:
   - Processes raster data (DEM, flow direction, soil parameters)
   - Processes river network shapefile
   - Builds network topology and Strahler ordering
   - Maps hillslope cells to river reaches
   - Saves preprocessed data to NetCDF and GeoParquet formats

4. **Meteorological data conversion**:
   - Converts MATLAB .mat format to CF-compliant NetCDF
   - Organizes station data by variable type

5. **Load forcing data**:
   - Loads meteorological forcing from NetCDF
   - Displays available variables and time range

6. **Run simulation**:
   - Creates Simulation object with GIS data, forcing, and configuration
   - Runs simulation over the entire forcing period
   - Computes soil water balance and channel routing at each time step
   - Stores discharge time series

7. **Save results**:
   - Saves final model state to NetCDF
   - Saves discharge time series to Parquet format

8. **Visualize results**:
   - Plots discharge hydrograph at a specific reach
   - Shows network-wide discharge statistics
   - Displays interactive plots

## Customization

You can customize the example by modifying these variables at the top of `run_example_Arno.py`:

```python
force_preprocessing = False  # Set to True to rerun preprocessing
debug_level = "DEBUG"        # Logging level: DEBUG, INFO, WARNING, ERROR
```

The simulation runs for the entire period available in the meteorological forcing data. To run shorter simulations for testing, modify the `end_date` calculation in the script (commented example provided).

## Output files

The script creates the following outputs:

### Preprocessed data
- `Arno/gisdata/Arno_gisdata.nc` - Gridded GIS data (NetCDF)
- `Arno/gisdata/Arno_net.parquet` - River network (GeoParquet)
- `Arno/meteodata/Arno_meteodata.nc` - Meteorological data (NetCDF)

### Simulation results
- `Arno/states/final_state_YYYYMMDD_HHMMSS.nc` - Final model state (NetCDF)
- `Arno/output/discharge_YYYYMMDD_YYYYMMDD.parquet` - Discharge time series (Parquet)

## Example data

The `Arno/` directory contains:

```
Arno/
├── Arno.yaml                    # Configuration file
├── raster/                      # Raster GIS data
│   ├── dtm.tif                  # Digital Elevation Model
│   ├── flowdir.tif              # Flow direction
│   ├── flowacc.tif              # Flow accumulation
│   ├── cap.tif                  # Capillary water capacity
│   ├── grav.tif                 # Gravitational water capacity
│   └── ks.tif                   # Hydraulic conductivity
├── shp/                         # Vector GIS data
│   └── Arno_river_network.shp   # River network shapefile
├── meteodata/                   # Meteorological data
│   └── meteodata.mat            # MATLAB format (original)
├── gisdata/                     # Preprocessed data (created by script)
├── states/                      # Model states (created by script)
└── output/                      # Results (created by script)
    └── matlab/                  # MATLAB reference outputs (for validation)
        └── discharge.csv        # MATLAB discharge time series
```

## Configuration file

The `Arno.yaml` file contains all model parameters and settings:

- **Basin information**: ID, coordinates
- **File paths**: Input/output locations
- **GIS data**: Raster and vector files
- **Model parameters**: Soil, routing, groundwater
- **Initial conditions**: Initial soil moisture, discharge
- **Simulation settings**: Time step, duration
- **Output options**: What to save and in what format

See `Arno/Arno.yaml` for detailed parameter descriptions.

## Validation against MATLAB reference

After running `run_example_Arno.py`, you can validate the Python implementation against MATLAB reference outputs using:

```bash
python examples/run_example_Arno_plots.py
```

### Comparison steps

1. **Load data**:
   - Loads Python discharge output (Parquet format)
   - Loads MATLAB discharge output (CSV format)

2. **Match reaches**:
   - Accounts for +1 offset in MATLAB reach IDs (MATLAB reach_id = Python reach_id + 1)
   - Identifies matching reaches between implementations

3. **Align time series**:
   - Finds common time range
   - Ensures exact timestamp alignment

4. **Calculate metrics**:
   - Root Mean Square Error (RMSE)
   - Bias

5. **Visualize comparison**:
   - Time series plots (Python vs MATLAB)
   - Scatter plots with 1:1 line

### Expected output

The script displays:
- Performance metrics for each matched reach
- Overall average NSE and RMSE
- Interactive comparison plots showing time series and scatter plots

## Restart from previous simulation state

The restart example demonstrates MOBIDICpy's ability to save intermediate states and continue simulations from previous simulation states.

```bash
python examples/run_example_Arno_restart.py
```

### What the restart example does

The script performs the following workflow:

1. **First simulation run**:
   - Runs simulation from start to a midpoint (e.g., 50% of total period)
   - Saves intermediate states to NetCDF file

2. **Load saved state**:
   - Loads the last saved state from the first run
   - Uses `Simulation.set_initial_state()` to initialize from file

3. **Second simulation run**:
   - Restarts simulation from the saved state
   - Continues to the end of the forcing period

4. **Continuous simulation**:
   - Runs a complete simulation without restart for comparison

5. **Validation**:
   - Compares restarted vs continuous results
   - Calculates maximum and mean differences
   - Verifies numerical equivalence (differences < 1e-6)

6. **Visualization**:
   - Plots discharge hydrographs (continuous vs restarted)
   - Shows difference between runs
   - Displays network-wide statistics

### Customization

You can customize the restart point by modifying the variable at the top of the script:

```python
restart_fraction = 0.5  # Run first simulation to 50% of period
```

### Use cases

The restart capability enables:
- **Long-term simulations**: Break long simulations into manageable chunks
- **Checkpoint recovery**: Resume after system interruptions
- **Multi-stage modeling**: Apply different parameters or forcings in different periods
- **Ensemble forecasting**: Initialize multiple forecasts from a single spin-up run
- **Real-time operations**: Save states at current time and restart with new forecast data

### Expected output

The script displays:
- Simulation periods for first run, second run, and continuous run
- Maximum difference between restarted and continuous results
- Mean difference and relative error
- Confirmation that results match within numerical precision
- Three-panel visualization comparing discharge time series

## Additional resources

- **User guide**: See `docs/` for detailed documentation
- **API reference**: See `docs/api/` for function descriptions
- **GitHub issues**: Report bugs or request features

## Citation

If you use MOBIDICpy in your research, please cite:

[Citation information will be added]
