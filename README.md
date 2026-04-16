
[![GitHub](https://img.shields.io/badge/github-mobidicpy-blue?logo=github)](https://github.com/mobidichydro/mobidicpy)
[![License](https://img.shields.io/github/license/mobidichydro/mobidicpy)](https://github.com/mobidichydro/mobidicpy/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mobidicpy.svg?colorB=blue)](https://pypi.org/project/mobidicpy/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19597504.svg)](https://doi.org/10.5281/zenodo.19597504)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/12552/badge)](https://www.bestpractices.dev/projects/12552)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
[![cffconvert](https://github.com/mobidichydro/mobidicpy/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/mobidichydro/mobidicpy/actions/workflows/cffconvert.yml)
[![Linting](https://github.com/mobidichydro/mobidicpy/actions/workflows/lint.yml/badge.svg)](https://github.com/mobidichydro/mobidicpy/actions/workflows/lint.yml)
[![Python package](https://github.com/mobidichydro/mobidicpy/actions/workflows/build.yml/badge.svg)](https://github.com/mobidichydro/mobidicpy/actions/workflows/build.yml)
[![Coverage](https://raw.githubusercontent.com/mobidichydro/mobidicpy/main/badges/coverage.svg)](https://github.com/mobidichydro/mobidicpy/actions/workflows/coverage.yml)
[![Docs](https://github.com/mobidichydro/mobidicpy/actions/workflows/documentation-test.yml/badge.svg)](https://github.com/mobidichydro/mobidicpy/actions/workflows/documentation-test.yml)

# `MOBIDICpy`

<p align="left">
  <img src="https://raw.githubusercontent.com/mobidichydro/mobidicpy/main/docs/assets/logo_mobidic_color_white_bg.svg" alt="MOBIDICpy Logo" width="160">
</p>

MOBIDIC (MOdello di Bilancio Idrologico DIstribuito e Continuo – distributed and continuous hydrological balance model) is a physically-based distributed hydrological model that simulates water and energy balances of the hydrological cycle at the catchment scale, and compute runoff generation and propagation through the river network.

MOBIDICpy is a Python implementation of the MOBIDIC model, originally developed in MATLAB by Castelli et al. See [References](#references) for more details.

## Installation

The package can be installed locally via pip:

```bash
# Clone the repository
git clone https://github.com/mobidichydro/mobidicpy.git
cd mobidicpy

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install the base package
pip install .

# For calibration and sensitivity analysis (PEST++)
pip install .[calibration] && get-pestpp :pyemu

# For development with all dependencies
pip install --no-cache-dir --editable .[dev]
```

The documentation can be built locally using [MkDocs](https://www.mkdocs.org/):

```bash
# Install documentation dependencies
pip install .[doc]

# Serve the documentation locally (http://127.0.0.1:8000)
python -m mkdocs serve
```

## Examples

Examples are available in the [`examples`](examples/) directory. 

## Documentation

The project's full documentation is available [here](https://mobidichydro.github.io/MOBIDICpy/).

## Contributing

If you want to contribute to the development of MOBIDICpy,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created using the [NLeSC/python-template](https://github.com/NLeSC/python-template).

## License

Copyright (c) 2026 University of Florence (Italy), Department of Civil and Environmental Engineering ([DICEA](https://www.dicea.unifi.it/)).

Licensed under the [Apache License, Version 2.0](LICENSE).


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
