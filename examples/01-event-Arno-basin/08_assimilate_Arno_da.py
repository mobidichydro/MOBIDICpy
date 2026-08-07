"""Example: flood forecasting with sequential data assimilation (pestpp-da).

Prerequisites:
    - Install calibration dependencies and PEST++ binaries:
        make install-calib
            or (manually)
        pip install mobidicpy[calibration] && get-pestpp :pyemu
    - Ensure the pestpp-da executable is on PATH

Usage:
    python examples/01-event-Arno-basin/08_assimilate_Arno_da.py

"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from mobidic import MeteoData, MeteoRaster, Simulation, load_config, run_preprocessing, save_gisdata, save_network
from mobidic.calibration import PestSetup, load_calibration_config, prepare_simulation, read_state_file
from mobidic.calibration.da_states import build_state_mask, state_file_path

# ---- Settings -------------------------------------------------------------
forecast_windows = [1, 2, 3, 4]  # Predict from the end of each of these cycles
rerun_assimilation = True # If False, load the results from the previous run instead of re-running PESTPP-DA

calib_config_path = Path(__file__).parent / "Arno.da.yaml"
meteodata_mat_path = (
    Path(__file__).parent.parent / "datasets" / "Arno" / "matlab" / "meteodata" / "Arno_event_Nov_2023.mat"
)

calib_config = load_calibration_config(calib_config_path)

mobidic_config = Path(calib_config.mobidic_config)
config_file = mobidic_config if mobidic_config.is_absolute() else Path(__file__).parent / mobidic_config
obs_group = calib_config.observations[0]

# Calibrated parameters (name -> parameter_key)
name_to_key = {p.name.lower(): p.parameter_key for p in calib_config.parameters}

ies_num_reals = calib_config.pest_options.get('ies_num_reals')  

# =========================================================================
# Step 1: Preprocessing and meteorological forcing
# =========================================================================
config = load_config(config_file)
print("Running GIS preprocessing...")
gisdata = run_preprocessing(config)
save_gisdata(gisdata, config.paths.gisdata)
save_network(gisdata.network, config.paths.network, format="parquet")

print("Converting meteorological forcing...")
meteo = MeteoData.from_mat(meteodata_mat_path)
config.paths.meteodata.parent.mkdir(parents=True, exist_ok=True)
meteo.to_netcdf(config.paths.meteodata, add_metadata={"basin": config.basin.id})


# =========================================================================
# Step 2: Set up and run PESTPP-DA
# =========================================================================
pest = PestSetup(calib_config, base_path=calib_config_path.parent)
working_dir = pest.setup()
schedule = pest._cycle_schedule

estimated = calib_config.da.states.estimate
print(f"\nPESTPP-DA working directory: {working_dir}")
print(f"Cycles: {schedule.n_cycles} x {schedule.n_steps_per_cycle} timesteps ({calib_config.da.cycle_length})")
print(f"Assimilation stops after cycle {calib_config.da.stop_cycle}")
if estimated:
    print(f"Formulation 2: estimating {estimated} on {len(pest._state_spec)} reach(es) alongside the parameters")
else:
    print("Formulation 1: states are transferred between cycles but never adjusted")

results = pest.run() if rerun_assimilation else pest.load_results()
state_dir = pest.working_dir / calib_config.da.states.state_file_dir

# =========================================================================
# Step 3: Observations, peak, and the assimilated subset
# =========================================================================
df_obs = pd.read_csv(pest.base_path / obs_group.obs_file)
df_obs["time"] = pd.to_datetime(df_obs[obs_group.time_column])
df_obs = df_obs.set_index("time")[obs_group.value_column]

# Extract the event
event = df_obs.loc[schedule.starts[0] : pd.Timestamp(calib_config.simulation_period.end_date)]
peak_time = event.idxmax()
peak_value = event.max()
print(f"\nObserved peak: {peak_value:.1f} m3/s at {peak_time}")

# Observations actually used inside the assimilated cycles
last_assimilated = calib_config.da.stop_cycle if calib_config.da.stop_cycle is not None else schedule.n_cycles - 1
assimilated_obs = event.loc[schedule.starts[0] : schedule.ends[last_assimilated]]


# =========================================================================
# Step 4: Run predictions
# =========================================================================
forcing = MeteoRaster.from_netcdf(pest.working_dir / "forcing_raster.nc")
forecast_end = pd.Timestamp(calib_config.simulation_period.end_date)

# Read the warm-up state
warmup_state = pest.working_dir / "warmup_state_001.nc"
if not warmup_state.exists():
    warmup_state = pest.working_dir / "warmup_state.nc"

# Active domain mask for the state variables
state_mask = build_state_mask(gisdata)

# The indexes of the selected reaches
reach_index = {int(r): i for i, r in enumerate(gisdata.network["mobidic_id"].values)}[obs_group.reach_id]


# Some helpers
def discharge_series(sim_results):
    """Extract the observed reach's discharge as a time-indexed Series."""
    return pd.Series(
        sim_results.time_series["discharge"][:, reach_index],
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
    return discharge_series(sim.run(start.to_pydatetime(), forecast_end.to_pydatetime()))


print("\nLaunching prediction runs:")
forecasts = {}
lead_times = {}
for cycle in forecast_windows:
    state_ids = results.get_da_state_ids(cycle)
    ensembles = results.get_da_results(cycle=cycle)["parameters"]
    if state_ids is None or not ensembles:
        print(f"  window {cycle}: no output found, skipping")
        continue
    par = ensembles[(cycle, max(it for _, it in ensembles))]

    start = schedule.starts[cycle] + pd.Timedelta(seconds=schedule.dt_seconds * schedule.n_steps_per_cycle)
    lead_times[cycle] = (peak_time - schedule.ends[cycle]).total_seconds() / 3600.0
    print(f"  window {cycle}: from {start}, lead {lead_times[cycle]:+.1f} h, {len(state_ids)} realizations")

    members = {}
    for realization, state_id in state_ids.items():
        if realization not in par.index:
            continue
        members[realization] = run_prediction(
            par.loc[realization], state_file_path(state_dir, cycle, int(state_id)), start
        )
    forecasts[cycle] = pd.DataFrame(members)


# =========================================================================
# Step 5: The assimilated trajectory
# =========================================================================
print("\nReassembling the assimilated (posterior) trajectory...")
analysis = results.get_da_timeseries(obs_group.name, posterior=True)

if estimated:
    print("\nAnalysis increment on the discharge state at the observed reach:")
    print(f"{'cycle':>6} {'background':>12} {'analysed':>10} {'increment':>11} {'ens. sd':>9}")
    for cycle in range(last_assimilated + 1):
        prior = results.get_da_states(cycle=cycle, iteration=0)
        posterior = results.get_da_states(cycle=cycle)
        if prior is None or posterior is None:
            continue
        b, a = prior[obs_group.reach_id], posterior[obs_group.reach_id]
        print(f"{cycle:>6} {b.mean():12.2f} {a.mean():10.2f} {a.mean() - b.mean():+11.2f} {a.std():9.2f}")


# =========================================================================
# Step 6: Open loop reference (no assimilation, prior parameter values)
# =========================================================================
print("\nRunning the open loop reference...")
# Same warm-up state as cycle 0, prior parameters, no assimilation at all.
open_loop_sim = Simulation(gisdata, forcing, config)
open_loop_sim.set_initial_state(state_file=str(warmup_state), time_index=-1)
open_loop = discharge_series(open_loop_sim.run(schedule.starts[0].to_pydatetime(), forecast_end.to_pydatetime()))


# =========================================================================
# Step 7: Prediction accuracy metrics
# =========================================================================
def peak_error(series):
    """Percentage peak error and timing error against the observed peak."""
    simulated_peak = series.max()
    timing = (series.idxmax() - peak_time).total_seconds() / 3600.0
    return 100.0 * (simulated_peak - peak_value) / peak_value, timing, simulated_peak


print("\n" + "=" * 72)
print(f"{'run':>12} {'lead (h)':>9} {'Qpeak':>9} {'peak err':>10} {'timing err':>11}")
print("-" * 72)
pe, te, qp = peak_error(open_loop)
print(f"{'open loop':>12} {'-':>9} {qp:9.1f} {pe:9.1f}% {te:10.2f} h")
for cycle, ensemble in forecasts.items():
    pe, te, qp = peak_error(ensemble.mean(axis=1))
    print(f"{'DA win %d' % cycle:>12} {lead_times[cycle]:9.1f} {qp:9.1f} {pe:9.1f}% {te:10.2f} h")
print("=" * 72)
print(f"{'observed':>12} {'':>9} {peak_value:9.1f}")


# =========================================================================
# Step 8: Plot
# =========================================================================
# Catchment average rainfall
with xr.open_dataset(pest.working_dir / "forcing_raster.nc") as ds:
    rain = ds["precipitation"].mean(dim=["y", "x"], skipna=True).to_series()

colours = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

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
ax.plot(open_loop.index, open_loop.values, "-", color="#1f77b4", linewidth=1.8, label="Open loop", zorder=4)

# One prediction per assimilation window 
for i, (cycle, ensemble) in enumerate(forecasts.items()):
    colour = colours[i % len(colours)]
    mean = ensemble.mean(axis=1)
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
        label=f"DA win {cycle} (lead {lead_times[cycle]:+.1f} h)",
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
    f"Arno at reach {obs_group.reach_id}\n"
    f"PESTPP-DA, {calib_config.da.cycle_length} cycles, EnKF ({calib_config.da.states.restart_from}), "
    f"{ies_num_reals} realizations",
    fontsize=11,
)

fig.tight_layout()
plt.show()
