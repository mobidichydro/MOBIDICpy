# MOBIDIC Examples

This directory contains example scripts and data for running MOBIDIC hydrological simulations.

## Available Examples

### Arno River Basin Example

The `run_example_Arno.py` script demonstrates the complete MOBIDIC workflow using data from the Arno River basin in Tuscany, Italy.

**Location:** `examples/run_example_Arno.py`

**Data:** `examples/Arno/`

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

## What the Example Does

The `run_example_Arno.py` script performs the following steps:

1. **Load Configuration**: Reads the YAML configuration file (`Arno/Arno.yaml`)

2. **GIS Preprocessing**:
   - Processes raster data (DEM, flow direction, soil parameters)
   - Processes river network shapefile
   - Builds network topology and Strahler ordering
   - Maps hillslope cells to river reaches
   - Saves preprocessed data to NetCDF and GeoParquet formats

3. **Meteorological Data Conversion**:
   - Converts MATLAB .mat format to CF-compliant NetCDF
   - Organizes station data by variable type

4. **Run Simulation**:
   - Initializes model state (soil moisture, discharge)
   - Runs main time-stepping loop:
     - Interpolates meteorological forcing to grid
     - Calculates potential evapotranspiration
     - Computes soil water balance
     - Routes hillslope and channel flow
   - Stores discharge time series

5. **Save Results**:
   - Saves final model state to NetCDF
   - Saves discharge time series to Parquet format

6. **Visualize Results**:
   - Plots discharge hydrograph at basin outlet
   - Shows network-wide discharge statistics
   - Saves plots to PNG files

## Customization

You can customize the example by modifying these variables at the top of `run_example_Arno.py`:

```python
FORCE_PREPROCESSING = False  # Set to True to rerun preprocessing
SIMULATION_DAYS = 5          # Number of days to simulate
```

## Output Files

The script creates the following outputs:

### Preprocessed Data
- `Arno/gisdata/Arno_gisdata.nc` - Gridded GIS data (NetCDF)
- `Arno/gisdata/Arno_net.parquet` - River network (GeoParquet)
- `Arno/meteodata/Arno_meteodata.nc` - Meteorological data (NetCDF)

### Simulation Results
- `Arno/states/final_state_*.nc` - Final model state (NetCDF)
- `Arno/outputs/discharge_*.parquet` - Discharge time series (Parquet)
- `Arno/outputs/discharge_hydrograph_*.png` - Visualization plots

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
└── outputs/                     # Results (created by script)
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

## Troubleshooting

### "File not found" errors

Make sure you're running the script from the correct directory. The script uses relative paths from the `examples/` directory.

### Preprocessing takes too long

The first run performs GIS preprocessing which can take several minutes. Subsequent runs will be faster as the preprocessed data is cached.

To force re-running preprocessing:
```python
FORCE_PREPROCESSING = True
```

### Out of memory errors

Try reducing the number of simulation days:
```python
SIMULATION_DAYS = 1  # Simulate only 1 day
```

Or increase the grid degradation factor in `Arno.yaml`:
```yaml
simulation:
  resample: 2  # Coarsen grid by factor of 2
```

## Next Steps

After running the example, you can:

1. **Modify parameters** in `Arno.yaml` to see how they affect results
2. **Extend simulation period** by increasing `SIMULATION_DAYS`
3. **Create your own basin** by following the Arno example structure
4. **Analyze results** using the saved Parquet files:
   ```python
   import pandas as pd
   df = pd.read_parquet('Arno/outputs/discharge_*.parquet')
   print(df.head())
   ```

## Additional Resources

- **User Guide**: See `docs/` for detailed documentation
- **API Reference**: See `docs/api/` for function descriptions
- **GitHub Issues**: Report bugs or request features

## Citation

If you use MOBIDIC in your research, please cite:

[Citation information will be added]
