# MOBIDIC Examples

This directory contains example scripts and data for running MOBIDIC hydrological simulations.

## Available Examples

### Arno River Basin Example

The `run_example_Arno.py` script demonstrates the complete MOBIDIC workflow using data from the Arno River basin in Tuscany, Italy.

**Location:** `examples/run_example_Arno.py`

**Data:** `examples/Arno/`

### Validation: Python vs MATLAB

The `run_example_Arno_plots.py` script validates the Python implementation against MATLAB reference outputs by comparing discharge time series, calculating performance metrics (NSE, RMSE, R²), and creating comparison plots.

**Location:** `examples/run_example_Arno_plots.py`

**Requirements:**
- Python discharge output from `run_example_Arno.py`
- MATLAB reference discharge output (CSV format) in `Arno/output/matlab/discharge.csv`

**Purpose:** Regression testing to ensure the Python implementation produces results equivalent to the original MATLAB code.

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

### Running the Arno Example

```bash
cd examples
python run_example_Arno.py
```

Or from the project root:

```bash
python examples/run_example_Arno.py
```

### Running the Validation Script (Optional)

After running the main example, validate against MATLAB reference outputs:

```bash
python examples/run_example_Arno_plots.py
```

See [Validation Against MATLAB Reference](#validation-against-matlab-reference) for details.

## What the Example Does

The `run_example_Arno.py` script performs the following steps:

1. **Configure Logging**: Sets up logger with specified debug level

2. **Load Configuration**: Reads the YAML configuration file (`Arno/Arno.yaml`)

3. **GIS Preprocessing**:
   - Processes raster data (DEM, flow direction, soil parameters)
   - Processes river network shapefile
   - Builds network topology and Strahler ordering
   - Maps hillslope cells to river reaches
   - Saves preprocessed data to NetCDF and GeoParquet formats

4. **Meteorological Data Conversion**:
   - Converts MATLAB .mat format to CF-compliant NetCDF
   - Organizes station data by variable type

5. **Load Forcing Data**:
   - Loads meteorological forcing from NetCDF
   - Displays available variables and time range

6. **Run Simulation**:
   - Creates Simulation object with GIS data, forcing, and configuration
   - Runs simulation over the entire forcing period
   - Computes soil water balance and channel routing at each time step
   - Stores discharge time series

7. **Save Results**:
   - Saves final model state to NetCDF
   - Saves discharge time series to Parquet format

8. **Visualize Results**:
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

## Output Files

The script creates the following outputs:

### Preprocessed Data
- `Arno/gisdata/Arno_gisdata.nc` - Gridded GIS data (NetCDF)
- `Arno/gisdata/Arno_net.parquet` - River network (GeoParquet)
- `Arno/meteodata/Arno_meteodata.nc` - Meteorological data (NetCDF)

### Simulation Results
- `Arno/states/final_state_YYYYMMDD_HHMMSS.nc` - Final model state (NetCDF)
- `Arno/output/discharge_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS.parquet` - Discharge time series (Parquet)

## Example Data

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

## Configuration File

The `Arno.yaml` file contains all model parameters and settings:

- **Basin information**: ID, coordinates
- **File paths**: Input/output locations
- **GIS data**: Raster and vector files
- **Model parameters**: Soil, routing, groundwater
- **Initial conditions**: Initial soil moisture, discharge
- **Simulation settings**: Time step, duration
- **Output options**: What to save and in what format

See `Arno/Arno.yaml` for detailed parameter descriptions.

## Validation Against MATLAB Reference

After running `run_example_Arno.py`, you can validate the Python implementation against MATLAB reference outputs using:

```bash
python examples/run_example_Arno_plots.py
```

### What the Validation Script Does

1. **Load Data**:
   - Loads Python discharge output (Parquet format)
   - Loads MATLAB discharge output (CSV format)

2. **Match Reaches**:
   - Accounts for +1 offset in MATLAB reach IDs (MATLAB reach_id = Python reach_id + 1)
   - Identifies matching reaches between implementations

3. **Align Time Series**:
   - Finds common time range
   - Ensures exact timestamp alignment

4. **Calculate Metrics**:
   - Nash-Sutcliffe Efficiency (NSE)
   - Root Mean Square Error (RMSE)
   - Bias
   - R² coefficient

5. **Visualize Comparison**:
   - Time series plots (Python vs MATLAB)
   - Scatter plots with 1:1 line
   - Performance metrics displayed on plots

### Expected Output

The script displays:
- Performance metrics for each matched reach
- Overall average NSE and RMSE
- Interactive comparison plots showing time series and scatter plots

**Typical validation results:**
- NSE > 0.99 indicates excellent agreement
- RMSE < 1 m³/s indicates minimal differences

## Next Steps

After running the example, you can:

1. **Modify parameters** in `Arno.yaml` to see how they affect results
2. **Extend or modify simulation period** by changing `start_date` and `end_date` in the script
3. **Create your own basin** by following the Arno example structure
4. **Analyze results** using the saved Parquet files:
   ```python
   import pandas as pd
   import glob

   # Find the latest discharge file
   discharge_files = glob.glob('Arno/output/discharge_*.parquet')
   df = pd.read_parquet(discharge_files[-1])
   print(df.head())
   ```

## Additional Resources

- **User Guide**: See `docs/` for detailed documentation
- **API Reference**: See `docs/api/` for function descriptions
- **GitHub Issues**: Report bugs or request features

## Citation

If you use MOBIDIC in your research, please cite:

[Citation information will be added]
