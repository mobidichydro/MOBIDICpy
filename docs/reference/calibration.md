# Calibration

The calibration module provides tools for model calibration, global sensitivity analysis, and uncertainty quantification using the [PEST++](https://github.com/usgs/pestpp) (Parameter ESTimation) software suite by USGS via [pyEMU](https://github.com/pypest/pyemu) (a Python interface to PEST++).

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

## Performance metrics

Four metrics are implemented directly in MOBIDICpy:

- `nse`: Nash-Sutcliffe Efficiency
- `nse_log`: Logarithmic Nash-Sutcliffe Efficiency
- `pbias`: Percent Bias
- `peak_error`: Peak Error

Several other metrics are implemented as simple wrappers around the corresponding functions in
the [HydroErr](https://hydroerr.readthedocs.io/en/stable/list_of_metrics.html)
package (~74 functions). `"kge"` is an alias for `kge_2012`. See the
[HydroErr documentation](https://hydroerr.readthedocs.io/en/stable/list_of_metrics.html)
for the mathematical definition of each metric.

### Usage

```python
from mobidic.calibration import compute_metrics, METRIC_REGISTRY
import numpy as np

sim = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Preferred: compute several metrics in one call
scores = compute_metrics(sim, obs, ["nse", "kge", "rmse", "r_squared", "ve"])

# Or look up a single metric directly
func, target = METRIC_REGISTRY["kge_2012"]
kge_value = func(sim, obs)

# Discover available names
print(sorted(METRIC_REGISTRY))
```

In a calibration YAML, any registry key is valid as a `metric:` name in an
observation group:

```yaml
observations:
  - name: gauge
    obs_file: obs.csv
    reach_id: 42
    value_column: discharge
    metrics:
      - {metric: kge_2012, target: 1.0, weight: 1.0}
      - {metric: r_squared, target: 1.0, weight: 0.5}
      - {metric: nse_log,   target: 1.0, weight: 1.0}
```

::: mobidic.calibration.metrics.compute_metrics


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

## Balancing observation groups

In PEST++, the objective function $\Phi$ is the weighted sum of squared residuals
across all observations. MOBIDICpy creates **one observation group per entry (one .csv file)**
in the `observations` list. The **pseudo-observations (metrics) associated to one entry are treated
as a distinct group**. The total objective function is therefore a sum
of group contributions:

$$\Phi = \sum_{g} \Phi_g = \sum_{g} \sum_{i \in g} \left( w_i \, (sim_i - obs_i) \right)^2$$

When groups contain very different numbers of observations, or when their
residuals live on very different numerical scales (e.g. discharge in m³/s vs.
a dimensionless NSE), the group with the largest raw contribution will
dominate the parameter estimation, and the others will have little influence
on the result, even if they were intended to constrain the model just as
strongly. This is a well-known issue discussed in the
[PEST manual](https://pesthomepage.org/documentation) (see the sections on
observation weights): the *relative* weights
between groups, not the absolute values, are what determine each group's
influence.

**Practical guidance:**

- Choose group weights so that each group's contribution $\Phi_g$ at the
  initial parameter values is of the same order of magnitude as the others
  (unless you deliberately want one group to dominate).
- After the first iteration, inspect the per-group contributions and rebalance
  if needed:
    - **GLM** (`pestpp-glm`): `<case>.iobj` — one row per iteration with a
      column per observation group.
    - **IES** (`pestpp-ies`): `<case>.phi.group.csv` — one row per realization
      with a column per group.
- The `weight` field on an observation entry scales every row of that time
  series; the `weight` field on a metric scales that single pseudo-observation.
  Increasing one group's weight by a factor $k$ multiplies its contribution
  to $\Phi$ by $k^2$.
- A common starting point is to set time-series weights inversely proportional
  to the expected residual magnitude (so weighted residuals are O(1)), and
  then tune metric weights to bring the metric group contributions into the
  same range.

