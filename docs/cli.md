# Command-Line Interface (CLI)

Installing MOBIDICpy provides a `mobidic` command that let the user run the
complete workflow from the terminal: preprocessing, simulation, calibration, and forcing
preparation.

The CLI is built on top of the [API](reference/index.md):
every command maps to the same functions documented in the API reference, so
anything that can be done from the CLI can also be done programmatically.

## Getting started

After [installing the package](index.md), check that the command is
available:

```bash
mobidic --version
mobidic --help
```

Each operation is a subcommand of `mobidic` with its own syntax and options.
Each subcommand has its own help, e.g. for the `simulation` subcommand:

```bash
mobidic simulation --help
```

All commands accept `--log-level {DEBUG,INFO,SUCCESS,WARNING,ERROR}` to override
the logging verbosity set in the`advanced` section of the YAML config file. Otherwise, the default `INFO` log level is used.

Here's an example of running a simulation with debug logging:
```bash
mobidic simulation path/to/config.yaml --log-level DEBUG
```

!!! note
    Relative paths in the config are resolved **relative to the config file's location**,
    not the current working directory.


## Typical workflow

```bash
# 1. (Optional) Convert MATLAB station data to NetCDF forcing
mobidic convert-meteo meteodata.mat meteodata/meteodata.nc

# 2. Validate the configuration and inspect resolved paths
mobidic check config.yaml

# 3. Run GIS preprocessing (creates consolidated gisdata + network)
mobidic preprocess config.yaml

# 4. Run the simulation
mobidic simulation config.yaml
```



## Commands

### `mobidic preprocess`

Run the GIS preprocessing pipeline and write the consolidated gisdata (NetCDF)
and river network (GeoParquet) to the paths defined in the config.

```bash
mobidic preprocess <config.yaml> [--force] [--log-level LEVEL]
```

| Option | Description |
| --- | --- |
| `--force` | Re-run preprocessing even if the gisdata and network files already exist. By default, if both exist the command reports them and exits without re-running. |

Wraps [`run_preprocessing`](reference/preprocessing.md), `save_gisdata`, and
`save_network`.

### `mobidic simulation`

Load the preprocessed GIS data and meteorological forcing, then run the
hydrological simulation. Reports and states are written automatically according
to the output options in the config.

```bash
mobidic simulation <config.yaml> [--start DATE] [--end DATE] [--preprocess] [--log-level LEVEL]
```

| Option | Description |
| --- | --- |
| `--start` | Override the simulation start (ISO date, e.g. `2023-11-01`). Default: first date in the forcing. |
| `--end` | Override the simulation end (ISO date). Default: last date in the forcing. |
| `--preprocess` | Run GIS preprocessing first (and save gisdata/network) before simulating. Without this flag, missing preprocessed data is an error. |

The **forcing mode** is detected automatically from the config (exactly one of
`paths.meteodata`, `paths.meteoraster`, or `paths.hyetograph` must be set):

- `meteodata` → station data, interpolated during the run ([`MeteoData`](reference/meteo.md))
- `meteoraster` → pre-interpolated grids ([`MeteoRaster`](reference/meteo.md))
- `hyetograph` → a design storm is generated on the fly ([`HyetographGenerator`](reference/meteo.md))

!!! note "Hyetograph start time"
    In hyetograph mode, `--start` seeds the event start time. If it is omitted,
    the default reference start `2000-01-01` is used.

### `mobidic calibration`

Set up a PEST++ working directory from a calibration YAML config and
(optionally) run the configured PEST++ tool. See the
[Calibration reference](reference/calibration.md) for the configuration format.

```bash
mobidic calibration <calibration.yaml> [--preprocess] [--setup-only] [--workers N] [--log-level LEVEL]
```

| Option | Description |
| --- | --- |
| `--preprocess` | Run GIS preprocessing for the referenced MOBIDIC config before setup. |
| `--setup-only` | Only generate the PEST++ working directory; do not run PEST++. |
| `--workers N` | Number of parallel workers (default: from the config, or all available CPUs). |

!!! warning "Requires the calibration extra dependencies"
    This command needs the optional calibration dependencies and PEST++
    executables on your `PATH`:

    ```bash
    pip install mobidicpy[calibration]
    ```

    Running it without them prints a clear error instead of a traceback.

### `mobidic hyetograph`

Generate a synthetic design-storm hyetograph from the IDF rasters and settings in
the config's `hyetograph` section, writing it to `paths.hyetograph`. This lets
you build the forcing file independently of running a simulation.

```bash
mobidic hyetograph <config.yaml> [--start DATE] [--log-level LEVEL]
```

| Option | Description |
| --- | --- |
| `--start` | Reference event start time (ISO date). Default: `2000-01-01`. |

### `mobidic convert-meteo`

Convert station-based meteorological data from a MATLAB `.mat` file to a
CF-1.12 compliant NetCDF file usable as simulation forcing. The output directory
is created automatically if it does not exist.

```bash
mobidic convert-meteo <input.mat> <output.nc> [--basin NAME] [--log-level LEVEL]
```

| Option | Description |
| --- | --- |
| `--basin` | Optional basin name stored as metadata in the output file. |

### `mobidic check`

Load and validate a configuration file, then print a summary: basin, time step,
soil/energy/routing settings, the detected forcing mode, and the resolved input
and output paths (with an indication of whether each input exists). This command
is read-only and useful for debugging configuration issues.

```bash
mobidic check <config.yaml> [--log-level LEVEL]
```

## Exit codes

Commands return `0` on success and `1` on a handled error (e.g. a missing file,
an invalid configuration, or a missing optional dependency). Such errors are
reported as concise messages rather than Python tracebacks, which makes the CLI
convenient to use in shell scripts and pipelines.
