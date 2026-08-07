"""Example: per-station plots of a multi-gauge sequential data assimilation run.

Prerequisites:
    - Run 08_assimilate_Arno_da.py first

Usage:
    python examples/01-event-Arno-basin/08b_assimilate_Arno_plots.py

"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from mobidic import MeteoRaster, Simulation, load_config, load_gisdata
from mobidic.calibration import PestSetup, load_calibration_config, prepare_simulation, read_state_file
from mobidic.calibration.da_cycles import CYCLE_METADATA_FILE, read_cycle_metadata, schedule_from_metadata
from mobidic.calibration.da_states import build_state_mask, state_file_path

# ---- Settings -------------------------------------------------------------
forecast_windows = [1, 2, 3, 4]  # Predict from the end of each of these cycles
save_figures = True  # Write one PNG per station to output/da_plots/
show_figures = True  # Open the figures interactively at the end

calib_config_path = Path(__file__).parent / "Arno.da.yaml"
figure_dir = Path(__file__).parent / "output" / "da_plots"

calib_config = load_calibration_config(calib_config_path)

mobidic_config = Path(calib_config.mobidic_config)
config_file = mobidic_config if mobidic_config.is_absolute() else Path(__file__).parent / mobidic_config

# Calibrated parameters (name -> parameter_key)
name_to_key = {p.name.lower(): p.parameter_key for p in calib_config.parameters}


# =========================================================================
# Step 1: Load the model and the completed assimilation
# =========================================================================
config = load_config(config_file)
print("Loading GIS data...")
gisdata = load_gisdata(config.paths.gisdata, config.paths.network)

pest = PestSetup(calib_config, base_path=calib_config_path.parent)
results = pest.load_results()

cycles_path = pest.working_dir / CYCLE_METADATA_FILE
if not cycles_path.exists():
    raise FileNotFoundError(f"{cycles_path} not found - run 08_assimilate_Arno_da.py first")
schedule = schedule_from_metadata(read_cycle_metadata(cycles_path))
state_dir = pest.working_dir / calib_config.da.states.state_file_dir

# Every ensemble file is read from disk on each call
_ensemble_cache: dict = {}
_read_ensembles = results.get_da_results


def cached_da_results(cycle=None, iteration=None):
    """get_da_results() memoized on its arguments."""
    key = (cycle, iteration)
    if key not in _ensemble_cache:
        _ensemble_cache[key] = _read_ensembles(cycle=cycle, iteration=iteration)
    return _ensemble_cache[key]


results.get_da_results = cached_da_results

estimated = calib_config.da.states.estimate
print(f"\nPESTPP-DA working directory: {pest.working_dir}")
print(f"Cycles: {schedule.n_cycles} x {schedule.n_steps_per_cycle} timesteps ({calib_config.da.cycle_length})")
print(f"Stations: {len(calib_config.observations)}")
if estimated:
    print(f"Formulation 2: estimating {estimated} alongside the parameters")
else:
    print("Formulation 1: states are transferred between cycles but never adjusted")


# =========================================================================
# Step 2: Observations, one series per station
# =========================================================================
last_assimilated = calib_config.da.stop_cycle if calib_config.da.stop_cycle is not None else schedule.n_cycles - 1
forecast_end = pd.Timestamp(calib_config.simulation_period.end_date)

events = {}  # group name -> observed Series over the event
peaks = {}  # group name -> (peak time, peak value)
for group in calib_config.observations:
    df_obs = pd.read_csv(pest.base_path / group.obs_file)
    df_obs["time"] = pd.to_datetime(df_obs[group.time_column])
    series = df_obs.set_index("time")[group.value_column]
    event = series.loc[schedule.starts[0] : forecast_end]
    events[group.name] = event
    peaks[group.name] = (event.idxmax(), event.max())

print("\nObserved peaks:")
for group in calib_config.observations:
    peak_time, peak_value = peaks[group.name]
    print(f"  {group.name:>8} (reach {group.reach_id:4d}, {group.value_column}): {peak_value:8.1f} m3/s at {peak_time}")


# =========================================================================
# Step 3: Run the predictions (once per realization, all stations extracted)
# =========================================================================
forcing = MeteoRaster.from_netcdf(pest.working_dir / "forcing_raster.nc")

warmup_state = pest.working_dir / "warmup_state_001.nc"
if not warmup_state.exists():
    warmup_state = pest.working_dir / "warmup_state.nc"

# Active domain mask for the state variables
state_mask = build_state_mask(gisdata)

# Column position of each observed reach in the network
position = {int(r): i for i, r in enumerate(gisdata.network["mobidic_id"].values)}
reach_index = {group.name: position[group.reach_id] for group in calib_config.observations}


def discharge_frame(sim_results):
    """Extract every observed reach's discharge as a time-indexed DataFrame."""
    discharge = sim_results.time_series["discharge"]
    return pd.DataFrame(
        {name: discharge[:, index] for name, index in reach_index.items()},
        index=pd.DatetimeIndex(sim_results.time_series["time"]),
    )


def run_prediction(param_row, state_path, start):
    """Run one free (non-assimilated) prediction from an analysed state."""
    updates = {name_to_key[n.lower()]: float(v) for n, v in param_row.items() if n.lower() in name_to_key}
    cfg, gis = prepare_simulation(
        base_config_path=config_file,
        param_updates=updates,
        gisdata=gisdata,
        routing_params_calibrated=True,
    )
    sim = Simulation(gis, forcing, cfg)
    sim.set_initial_state(state=read_state_file(state_path, state_mask))
    return discharge_frame(sim.run(start.to_pydatetime(), forecast_end.to_pydatetime()))


print("\nLaunching prediction runs:")
# forecasts[cycle][group name] -> DataFrame (time x realization)
forecasts: dict[int, dict[str, pd.DataFrame]] = {}
launch_times = {}
for cycle in forecast_windows:
    state_ids = results.get_da_state_ids(cycle)
    ensembles = results.get_da_results(cycle=cycle)["parameters"]
    if state_ids is None or not ensembles:
        print(f"  window {cycle}: no output found, skipping")
        continue
    par = ensembles[(cycle, max(it for _, it in ensembles))]

    start = schedule.starts[cycle] + pd.Timedelta(seconds=schedule.dt_seconds * schedule.n_steps_per_cycle)
    launch_times[cycle] = start
    print(f"  window {cycle}: from {start}, {len(state_ids)} realizations")

    members = {}
    for realization, state_id in state_ids.items():
        if realization not in par.index:
            continue
        members[realization] = run_prediction(
            par.loc[realization], state_file_path(state_dir, cycle, int(state_id)), start
        )
    if not members:
        continue
    forecasts[cycle] = {
        name: pd.DataFrame({realization: frame[name] for realization, frame in members.items()})
        for name in reach_index
    }

if not forecasts:
    raise RuntimeError(f"No prediction could be launched from cycles {forecast_windows} - check pest_run_da/master")


# =========================================================================
# Step 4: The assimilated trajectories and the open loop reference
# =========================================================================
print("\nReassembling the assimilated (posterior) trajectories...")
analyses = {group.name: results.get_da_timeseries(group.name, posterior=True) for group in calib_config.observations}

print("Running the open loop reference...")
# Same warm-up state as cycle 0, prior parameters, no assimilation at all.
open_loop_sim = Simulation(gisdata, forcing, config)
open_loop_sim.set_initial_state(state_file=str(warmup_state), time_index=-1)
open_loop = discharge_frame(open_loop_sim.run(schedule.starts[0].to_pydatetime(), forecast_end.to_pydatetime()))

# Analysis increments on the discharge states, read once per cycle for all stations
state_prior: dict[int, pd.DataFrame] = {}
state_posterior: dict[int, pd.DataFrame] = {}
if estimated:
    for cycle in range(last_assimilated + 1):
        prior = results.get_da_states(cycle=cycle, iteration=0)
        posterior = results.get_da_states(cycle=cycle)
        if prior is not None and posterior is not None:
            state_prior[cycle] = prior
            state_posterior[cycle] = posterior


# =========================================================================
# Step 5: Prediction accuracy metrics
# =========================================================================
def peak_error(series, peak_time, peak_value):
    """Percentage peak error and timing error against the observed peak."""
    simulated_peak = series.max()
    timing = (series.idxmax() - peak_time).total_seconds() / 3600.0
    return 100.0 * (simulated_peak - peak_value) / peak_value, timing, simulated_peak


# =========================================================================
# Step 6: One figure per station
# =========================================================================
with xr.open_dataset(pest.working_dir / "forcing_raster.nc") as ds:
    rain = ds["precipitation"].mean(dim=["y", "x"], skipna=True).to_series()

colours = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

if save_figures:
    figure_dir.mkdir(parents=True, exist_ok=True)


def plot_station(group):
    """Build the assimilation/forecast figure of a single station."""
    name = group.name
    event = events[name]
    peak_time, peak_value = peaks[name]
    assimilated_obs = event.loc[schedule.starts[0] : schedule.ends[last_assimilated]]
    analysis = analyses[name]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Rainfall (secondary axis)
    ax_rain = ax.twinx()
    ax_rain.bar(rain.index, rain.values, width=pd.Timedelta(minutes=15), color="k", align="edge")
    ax_rain.set_ylim(4 * rain.max(), 0)
    ax_rain.set_ylabel("Rainfall (mm h$^{-1}$)")

    # Observations
    ax.plot(event.index, event.values, ".", color="0.55", markersize=3, label="Observed", zorder=2)
    ax.plot(
        assimilated_obs.index,
        assimilated_obs.values,
        "D",
        color="0.25",
        markersize=3.2,
        markerfacecolor="none",
        linewidth=0.6,
        label="Observed (assimilated)",
        zorder=3,
    )

    # Assimilated trajectory: what each forecast is launched from
    if analysis is not None:
        ax.fill_between(
            analysis.index,
            analysis.quantile(0.1, axis=1),
            analysis.quantile(0.9, axis=1),
            color="k",
            alpha=0.12,
            linewidth=0,
            zorder=3,
        )
        ax.plot(
            analysis.index,
            analysis.mean(axis=1),
            "-",
            color="k",
            linewidth=1.3,
            label="DA analysis (assimilated)",
            zorder=4,
        )

    # Open loop
    ax.plot(
        open_loop.index, open_loop[name].values, "-", color="#1f77b4", linewidth=1.8, label="Open loop", zorder=4
    )

    # One prediction per assimilation window
    for i, (cycle, station_ensembles) in enumerate(forecasts.items()):
        colour = colours[i % len(colours)]
        ensemble = station_ensembles[name]
        mean = ensemble.mean(axis=1)
        lead = (peak_time - schedule.ends[cycle]).total_seconds() / 3600.0
        ax.fill_between(
            ensemble.index,
            ensemble.quantile(0.1, axis=1),
            ensemble.quantile(0.9, axis=1),
            color=colour,
            alpha=0.15,
            linewidth=0,
            zorder=4,
        )
        ax.plot(
            mean.index,
            mean.values,
            "-",
            color=colour,
            linewidth=1.8,
            label=f"DA win {cycle} (lead {lead:+.1f} h)",
            zorder=5,
        )
        # Mark where each prediction is launched
        ax.axvline(mean.index[0], color=colour, linewidth=0.8, linestyle=":", alpha=0.8, zorder=1)

    # Assimilation-window
    y_bar = ax.get_ylim()[1] * 0.60
    for i, cycle in enumerate(forecasts):
        ax.plot(
            [schedule.starts[cycle], schedule.ends[cycle]],
            [y_bar, y_bar],
            "-",
            color=colours[i % len(colours)],
            linewidth=4,
            solid_capstyle="butt",
            zorder=6,
        )
    ax.annotate(
        "Assim. win.",
        xy=(schedule.starts[min(forecasts)], y_bar),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
    )

    # Observed peak
    ax.plot([peak_time], [peak_value], "k*", markersize=11, zorder=7)
    ax.annotate(
        f"observed peak\n{peak_value:.0f} m$^3$s$^{{-1}}$",
        xy=(peak_time, peak_value),
        xytext=(12, 6),
        textcoords="offset points",
        fontsize=8,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Discharge (m$^3$s$^{-1}$)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(schedule.starts[0], forecast_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"Arno at reach {group.reach_id} - {name} ({group.value_column})\n"
        f"PESTPP-DA, {calib_config.da.cycle_length} cycles, EnKF ({calib_config.da.states.restart_from}), "
        f"{len(calib_config.observations)} gauges assimilated",
        fontsize=11,
    )

    fig.tight_layout()
    return fig


for group in calib_config.observations:
    name = group.name
    peak_time, peak_value = peaks[name]

    print("\n" + "=" * 78)
    print(f"{name} - reach {group.reach_id} ({group.value_column})")
    print("=" * 78)

    # Prediction accuracy
    print(f"{'run':>12} {'lead (h)':>9} {'Qpeak':>9} {'peak err':>10} {'timing err':>11}")
    print("-" * 78)
    pe, te, qp = peak_error(open_loop[name], peak_time, peak_value)
    print(f"{'open loop':>12} {'-':>9} {qp:9.1f} {pe:9.1f}% {te:10.2f} h")
    for cycle, station_ensembles in forecasts.items():
        lead = (peak_time - schedule.ends[cycle]).total_seconds() / 3600.0
        pe, te, qp = peak_error(station_ensembles[name].mean(axis=1), peak_time, peak_value)
        print(f"{'DA win %d' % cycle:>12} {lead:9.1f} {qp:9.1f} {pe:9.1f}% {te:10.2f} h")
    print(f"{'observed':>12} {'':>9} {peak_value:9.1f}")

    # How much of the correction the filter applied went into the state rather
    # than the parameters. With formulation 1 this table does not exist at all.
    if state_prior and group.reach_id in state_prior[min(state_prior)].columns:
        print("\nAnalysis increment on the discharge state at this reach:")
        print(f"{'cycle':>6} {'background':>12} {'analysed':>10} {'increment':>11} {'ens. sd':>9}")
        for cycle in sorted(state_prior):
            b = state_prior[cycle][group.reach_id]
            a = state_posterior[cycle][group.reach_id]
            print(f"{cycle:>6} {b.mean():12.2f} {a.mean():10.2f} {a.mean() - b.mean():+11.2f} {a.std():9.2f}")
    elif estimated:
        print(f"\nReach {group.reach_id} carries no discharge state (check da.states.estimate_reaches)")

    fig = plot_station(group)
    if save_figures:
        path = figure_dir / f"da_{name}_reach{group.reach_id:04d}.png"
        fig.savefig(path, dpi=150)
        print(f"\nFigure saved: {path}")

print(f"\n{len(calib_config.observations)} figure(s) created.")
if show_figures:
    plt.show()
else:
    plt.close("all")
