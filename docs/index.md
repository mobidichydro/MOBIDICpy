# MOBIDICpy Documentation

<p align="left">
  <img src="assets/logo_mobidic_color.svg" alt="MOBIDICpy Logo" width="200" class="theme-logo">
</p>

MOBIDIC (MOdello di Bilancio Idrologico DIstribuito e Continuo) is a physically-based distributed hydrological model that simulates the water and energy balance of the hydrological cycle at the cell level and run-off propagation in the river network.

**MOBIDICpy** is a Python implementation of the MOBIDIC model, originally developed in MATLAB by Castelli et al. See [References](#references) for more details.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mobidichydro/mobidicpy.git
cd mobidicpy

# Install the base package
pip install .

# For calibration and sensitivity analysis (PEST++)
make install-calib
# or manually:
pip install .[calibration] && get-pestpp :pyemu

# For development with all dependencies
pip install --no-cache-dir --editable .[dev]
```

### Basic Usage

```python
import mobidic

# 1. Load configuration
config = mobidic.load_config("config.yaml")

# 2. GIS preprocessing
gisdata = mobidic.run_preprocessing(config)

# 3. Load meteorological data
forcing = mobidic.MeteoData.from_netcdf(config.paths.meteodata)

# 4. Run simulation
sim = mobidic.Simulation(gisdata, forcing, config)
results = sim.run(start_date=forcing.start_date, end_date=forcing.end_date)
```

See [examples/01-event-Arno-basin/01a_run_example_Arno.py](https://github.com/mobidichydro/mobidicpy/blob/main/examples/01-event-Arno-basin/01a_run_example_Arno.py) for a complete working example with visualization.

## Features

**Currently implemented (v0.1)**

- Simulation setup with YAML configuration file and parameter validation
- GIS data I/O (raster and vector formats)
- Grid operations (resolution degradation, flow direction conversion)
- River network processing (topology, Strahler ordering, calculation of routing parameters)
- Hillslope-reach mapping
- Meteorological data preprocessing from MATLAB format (.mat to NetCDF conversion)
- Meteorological data spatial interpolation (IDW and nearest neighbor)
- Design storm hyetograph generation from IDF parameters (Chicago method)
- Soil water balance module (4 reservoirs: capillary, gravitational, plants, surface)
- Linear routing (hillslope and channel)
- Reservoir module (preprocessing, routing, time-varying regulation)
- Basic I/O (NetCDF states, Parquet time series, export of interpolated meteorological data)

- Calibration, sensitivity, and uncertainty analysis ([PEST++](https://github.com/usgs/pestpp) coupling via [pyEMU](https://github.com/pypest/pyemu))

**To be implemented**

- Meteorological data gap filling and quality control
- Energy balance schemes
- Groundwater models
- Advanced routing methods
- CLI interface

## Documentation structure

- **[Introduction](introduction.md)** - Background, references, and model overview
- **[Development](development.md)** - Developer setup, testing, and contribution guidelines
- **[Examples](examples.md)** - Practical usage examples with working code
- **[API Reference](reference/index.md)** - Complete API documentation

## References

Campo, L., Caparrini, F., Castelli, F. (2006). Use of multi-platform, multi-temporal remote-sensing data for calibration of a 
distributed hydrological model: an application in the Arno basin, Italy. Hydrol. Process., 20: 2693-2712. 
DOI: [10.1002/hyp.6061](https://doi.org/10.1002/hyp.6061)

Castelli, F. (1996). A simplified stochastic model for infiltration into a heterogeneous soil forced by random precipitation. Advances in water resources, 19(3), 133-144. DOI: [10.1016/0309-1708(95)00041-0](https://doi.org/10.1016/0309-1708(95)00041-0)

Castelli, F., Menduni, G., and Mazzanti, B. (2009). A distributed package for sustainable water
management: A case study in the Arno basin. Role of Hydrology in Water Resources Management,
327, 52–61.

Castillo, A., Castelli, F., Entekhabi, D. (2015). Gravitational and capillary soil moisture dynamics for distributed
hydrologic models, Hydrol. Earth Syst. Sci., 19, 1857–1869, DOI: [10.5194/hess-19-1857-2015](https://doi.org/10.5194/hess-19-1857-2015).

Castelli, F., Ercolani, G. (2016). Improvement of operational flood forecasting through the assimilation of satellite observations and 
multiple river flow data, Proc. IAHS, 373, 167–173. DOI: [10.5194/piahs-373-167-2016](https://doi.org/10.5194/piahs-373-167-2016).

Ercolani, G., Castelli, F. (2017), Variational assimilation of streamflow data in distributed flood forecasting, Water Resour. Res., 53, 158–183. 
DOI: [10.1002/2016WR019208](https://doi.org/10.1002/2016WR019208).

Ercolani, G., Chiaradia, E. A., Gandolfi, C., Castelli, F., Masseroni, D. (2018). Evaluating performances of green roofs for stormwater runoff mitigation in a high flood risk urban catchment. Journal of Hydrology, 566, 830-845. DOI: [10.1016/j.jhydrol.2018.09.050](https://doi.org/10.1016/j.jhydrol.2018.09.050)

Masi, M., Masseroni, D., Castelli, F. (2025). Coupled hydrologic, hydraulic, and surface water quality models for pollution management in urban–rural areas. 
Journal of Hydrology, 657, 133172. DOI: [10.1016/j.jhydrol.2025.133172](https://doi.org/10.1016/j.jhydrol.2025.133172).

Yang, J., Castelli, F., Chen, Y. (2014). Multiobjective sensitivity analysis and optimization of
distributed hydrologic model MOBIDIC. Hydrology and Earth System Sciences, 18(10), 4101–4112.
DOI: [10.5194/HESS-18-4101-2014](https://doi.org/10.5194/HESS-18-4101-2014)

Yang, J., Entekhabi, D., Castelli, F., Chua, L. (2014). Hydrologic response of a tropical watershed to urbanization. 
Journal of Hydrology, 517, 538-546. DOI: [10.1016/j.jhydrol.2014.05.053](https://doi.org/10.1016/j.jhydrol.2014.05.053).

## License

Copyright (c) 2026 University of Florence (Italy), Department of Civil and Environmental Engineering ([DICEA](https://www.dicea.unifi.it/)).

Licensed under the [Apache License, Version 2.0](https://github.com/mobidichydro/mobidicpy/blob/main/LICENSE).

## Support

- **Issues**: [GitHub Issues](https://github.com/mobidichydro/mobidicpy/issues)
- **Source Code**: [GitHub Repository](https://github.com/mobidichydro/mobidicpy)
