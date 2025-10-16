# MOBIDICpy Documentation

MOBIDIC (MOdello di Bilancio Idrologico DIstribuito e Continuo) is a raster-based distributed hydrological model that simulates the water and energy balance of the hydrological cycle at the cell level.

**MOBIDICpy** is a Python implementation of the MOBIDIC model, originally developed in MATLAB by Castelli et al. (2009).

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mobidichydro/mobidicpy.git
cd mobidicpy

# Install the package
pip install .

# For development with all dependencies
pip install --no-cache-dir --editable .[dev]
```

### Basic Usage

```python
from mobidic import load_config, run_preprocessing

# Load configuration from YAML
config = load_config("path/to/config.yaml")

# Run preprocessing pipeline
gisdata = run_preprocessing(config)

# Access processed data
print(f"Basin: {config.basin.id}")
print(f"Network reaches: {len(gisdata.network)}")
```

## Features

**Currently Implemented (v0.0.1 - Pre-Alpha)**

- Schema-driven configuration system with YAML and Pydantic validation
- GIS data I/O (raster and vector formats)
- Grid operations (resolution degradation, flow direction conversion)
- River network processing (topology, Strahler ordering, routing parameters)
- Hillslope-reach mapping
- Meteorological data preprocessing (MAT to NetCDF conversion)
- Consolidated I/O for preprocessed data

**Coming Soon**

- Soil water balance module
- Linear routing (hillslope and channel)
- Energy balance schemes (1L, 5L, Snow)
- Groundwater models (Linear, Dupuit, MODFLOW)
- Advanced routing (Muskingum-Cunge)
- Real-time simulation capability

## Documentation Structure

- **[Introduction](introduction.md)** - Background, references, and model overview
- **[Development](development.md)** - Developer setup, testing, and contribution guidelines
- **[Examples](examples.md)** - Practical usage examples with working code
- **[API Reference](reference/index.md)** - Complete API documentation

## Citation

If you use MOBIDICpy in your research, please cite:

> Castelli, F., Menduni, G., & Mazzanti, B. (2009). A distributed package for sustainable water management: A case study in the Arno basin. *Role of Hydrology in Water Resources Management*, 327, 52–61.

## License

MOBIDICpy is open-source software. See the repository for license details.

## Support

- **Issues**: [GitHub Issues](https://github.com/mobidichydro/mobidicpy/issues)
- **Source Code**: [GitHub Repository](https://github.com/mobidichydro/mobidicpy)
