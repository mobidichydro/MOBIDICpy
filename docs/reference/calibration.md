# Calibration

The calibration module provides tools for model calibration, global sensitivity analysis, and uncertainty quantification using [PEST++](https://github.com/usgs/pestpp) via [pyemu](https://github.com/pypest/pyemu).

## Installation

Calibration dependencies are optional and must be installed separately:

```bash
make install-calib
# or manually:
pip install .[calibration] && get-pestpp :pyemu
```

Make sure the PEST++ executables (`pestpp-glm`, `pestpp-ies`, `pestpp-sen`) are on the system `PATH`.

## Overview

The calibration workflow is driven by a single YAML configuration file (alongside the main MOBIDIC configuration). The `PestSetup` class orchestrates all steps: generating PEST++ template and instruction files, running the forward model, and parsing results.

**Currently supported PEST++ tools:**

| Tool | Method | Use case |
|------|--------|----------|
| `glm` | Gauss-Levenberg-Marquardt | Gradient-based calibration |
| `ies` | Iterative Ensemble Smoother | Ensemble-based calibration and uncertainty |
| `sen` | Sensitivity analysis | Global sensitivity analysis |


## Quick import

```python
from mobidic.calibration import (
    # Setup and results
    PestSetup,
    CalibrationResults,
    # Configuration
    CalibrationConfig,
    CalibrationParameter,
    CalibrationPeriod,
    MetricConfig,
    ObservationGroup,
    load_calibration_config,
    # Observations
    load_observations,
    align_observations_to_simulation,
    # Metrics
    nse,
    nse_log,
    kge,
    pbias,
    rmse,
    peak_error,
    compute_metrics,
)
```

## Workflow example

```python
from pathlib import Path
from mobidic.calibration import PestSetup, load_calibration_config

# Load configuration
calib_config = load_calibration_config("Arno.calibration.yaml")

# Set up PEST++ files
pest = PestSetup(calib_config)
working_dir = pest.setup()

# Run calibration
results = pest.run()

# Extract results
optimal = results.get_optimal_parameters()
phi = results.get_objective_function_history()
sens = results.get_parameter_sensitivities()
```

See [Examples 1.5](../examples.md) for complete working scripts.


## Calibration setup

::: mobidic.calibration.pest_setup.PestSetup

## Calibration results

::: mobidic.calibration.results.CalibrationResults

## Configuration

::: mobidic.calibration.config.CalibrationConfig

::: mobidic.calibration.config.CalibrationParameter

::: mobidic.calibration.config.ObservationGroup

::: mobidic.calibration.config.MetricConfig

::: mobidic.calibration.config.CalibrationPeriod

::: mobidic.calibration.config.load_calibration_config

## Observations

::: mobidic.calibration.observation.load_observations

::: mobidic.calibration.observation.align_observations_to_simulation

## Performance metrics

::: mobidic.calibration.metrics.nse

::: mobidic.calibration.metrics.nse_log

::: mobidic.calibration.metrics.kge

::: mobidic.calibration.metrics.pbias

::: mobidic.calibration.metrics.rmse

::: mobidic.calibration.metrics.peak_error

::: mobidic.calibration.metrics.compute_metrics