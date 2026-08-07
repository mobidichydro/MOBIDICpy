"""Anchor test: chained assimilation cycles must reproduce one continuous run.

This is the single most valuable test of the sequential data-assimilation
machinery. It proves three things at once:

1. the cycle windows are ``[t_c, t_{c+1} - dt]`` inclusive, so no forcing
   timestep is applied twice or skipped (section 3.2 of the plan);
2. the state file is *complete* — forgetting ``flr``/``fld`` produces a
   one-timestep discontinuity at every cycle boundary, which shows up here;
3. writing a state and reading it back is exact.
"""

from datetime import datetime
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import LineString

from mobidic.calibration.da_cycles import build_cycle_schedule
from mobidic.calibration.da_states import (
    KIND_SOIL_CAPILLARY,
    KIND_SOIL_GRAVITATIONAL,
    build_reach_zone_map,
    build_soil_state_spec,
    build_state_mask,
    build_surface_state_spec,
    extract_state_vector,
    insert_state_vector,
    read_state_file,
    soil_capacities,
    write_state_file,
    zone_saturation,
)
from mobidic.config.schema import MOBIDICConfig
from mobidic.core.simulation import Simulation
from mobidic.preprocessing.meteo_raster import MeteoRaster
from mobidic.preprocessing.preprocessor import GISData

NROWS, NCOLS = 12, 15
RESOLUTION = 500.0
XLL, YLL = 100_000.0, 200_000.0
DT = 900  # 15 minutes
N_TIMES = 24  # 6 hours of forcing
START = datetime(2023, 11, 1, 0, 0, 0)


@pytest.fixture
def forcing_path(tmp_path):
    """Deterministic raster forcing on the small test grid."""
    rng = np.random.default_rng(20231101)
    times = [START + i * np.timedelta64(DT, "s") for i in range(N_TIMES)]

    ds = xr.Dataset(
        {
            # Intense rainfall, so the scenario really produces runoff and
            # discharge: a chaining test on an all-zero series proves nothing.
            "precipitation": (["time", "y", "x"], 10.0 + rng.random((N_TIMES, NROWS, NCOLS)) * 20.0),
            "pet": (["time", "y", "x"], np.full((N_TIMES, NROWS, NCOLS), 0.05)),
        },
        coords={
            "time": times,
            "y": YLL + np.arange(NROWS) * RESOLUTION,
            "x": XLL + np.arange(NCOLS) * RESOLUTION,
        },
    )
    ds["crs"] = xr.DataArray(0, attrs={"spatial_ref": "EPSG:32632"})

    path = tmp_path / "forcing.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def config(tmp_path):
    return MOBIDICConfig(
        **{
            "basin": {"id": "chaining", "baricenter": {"lon": 11.0, "lat": 44.0}},
            "paths": {
                "meteoraster": "forcing.nc",
                "gisdata": "gis.nc",
                "network": "net.parquet",
                "states": str(tmp_path / "states"),
                "output": str(tmp_path / "output"),
            },
            "vector_files": {"river_network": {"shp": "network.shp"}},
            "raster_files": {
                "dtm": "dtm.tif",
                "flow_dir": "flowdir.tif",
                "flow_acc": "flowacc.tif",
                "Wc0": "wc0.tif",
                "Wg0": "wg0.tif",
                "ks": "ks.tif",
            },
            "raster_settings": {"flow_dir_type": "Grass"},
            "simulation": {"timestep": DT, "soil_scheme": "Bucket", "energy_balance": "None"},
            "initial_conditions": {"Wcsat": 0.85, "Wgsat": 0.85},
            "parameters": {
                "soil": {
                    "gamma": 2.689337e-07,
                    "kappa": 1.096651e-07,
                    "beta": 7.62e-06,
                    "alpha": 2.50e-05,
                },
                "energy": {},
                "routing": {"method": "Linear", "wcel": 5.18},
                "groundwater": {"model": "None"},
            },
            "output_states": {},
            "output_states_settings": {"output_states": "None"},
            "output_report": {"discharge": False, "lateral_inflow": False},
            "output_report_settings": {"format": "Parquet"},
        }
    )


@pytest.fixture
def gisdata(config):
    """Small synthetic basin with a masked-out corner, so the NaN mask is exercised."""
    rng = np.random.default_rng(7)

    dtm = rng.random((NROWS, NCOLS)) * 100 + 500
    flow_acc = np.arange(NROWS * NCOLS, dtype=float).reshape(NROWS, NCOLS)
    # Two cells outside the basin: the state file must round-trip their NaNs.
    dtm[0, 0] = np.nan
    flow_acc[0, 0] = np.nan
    dtm[NROWS - 1, NCOLS - 1] = np.nan
    flow_acc[NROWS - 1, NCOLS - 1] = np.nan

    grids = {
        "dtm": dtm,
        "flow_dir": np.full((NROWS, NCOLS), 3.0),  # all flow east (Grass)
        "flow_acc": flow_acc,
        "Wc0": np.full((NROWS, NCOLS), 0.02),
        "Wg0": np.full((NROWS, NCOLS), 0.02),
        "ks": np.full((NROWS, NCOLS), 5.0e-7),
        "alpsur": np.full((NROWS, NCOLS), 0.01),
        "cha": np.full((NROWS, NCOLS), 0.1),
        "slope": np.full((NROWS, NCOLS), 0.01),
    }

    metadata = {
        "shape": (NROWS, NCOLS),
        "resolution": (RESOLUTION, RESOLUTION),
        "transform": from_origin(XLL, YLL + NROWS * RESOLUTION, RESOLUTION, RESOLUTION),
        "crs": "EPSG:32632",
        "bounds": (XLL, YLL, XLL + NCOLS * RESOLUTION, YLL + NROWS * RESOLUTION),
        "xllcorner": XLL,
        "yllcorner": YLL,
    }

    network = gpd.GeoDataFrame(
        {
            "mobidic_id": [0, 1],
            "upstream_1": [-1, 0],
            "upstream_2": [-1, -1],
            "downstream": [1, -1],
            "strahler_order": [1, 1],
            "calc_order": [1, 2],
            "length_m": [3000.0, 4000.0],
            "width_m": [8.0, 10.0],
            "lag_time_s": [100.0, 120.0],
            "storage_coeff": [500.0, 700.0],
            "n_manning": [0.03, 0.03],
            "geometry": [
                LineString([(XLL + 2000, YLL + 3000), (XLL + 5000, YLL + 3000)]),
                LineString([(XLL + 5000, YLL + 3000), (XLL + 9000, YLL + 3000)]),
            ],
        },
        crs="EPSG:32632",
    )

    hillslope_reach_map = np.zeros((NROWS, NCOLS), dtype=int)
    hillslope_reach_map[:, NCOLS // 2 :] = 1

    return GISData(
        grids=grids,
        metadata=metadata,
        network=network,
        hillslope_reach_map=hillslope_reach_map,
        config=config,
    )


def _new_simulation(gisdata, config, forcing_path):
    return Simulation(gisdata, MeteoRaster.from_netcdf(forcing_path), config)


def _continuous_discharge(gisdata, config, forcing_path, start, end):
    sim = _new_simulation(gisdata, config, forcing_path)
    return sim.run(start, end).time_series["discharge"]


def _soil_spec(gisdata, config, forcing_path):
    """The zone-averaged soil assimilation space of this grid, and its capacities.

    The capacities come from a built Simulation, so they carry the Wc_factor /
    Wg_factor multipliers and the minimum-storage floor exactly as a forward run
    would see them.
    """
    sim = _new_simulation(gisdata, config, forcing_path)
    capacities = soil_capacities(sim)
    zone_map, zone_ids = build_reach_zone_map(gisdata)

    # parval1 is irrelevant here; only the zone geometry and bounds are used.
    template = SimpleNamespace(wc=0.5 * capacities[KIND_SOIL_CAPILLARY], wg=0.5 * capacities[KIND_SOIL_GRAVITATIONAL])
    spec = build_soil_state_spec(
        kinds=[KIND_SOIL_CAPILLARY, KIND_SOIL_GRAVITATIONAL],
        state=template,
        capacities=capacities,
        zone_map=zone_map,
        zone_ids=zone_ids,
    )
    return spec, capacities


def _surface_spec(gisdata, config, forcing_path):
    """The zone-averaged surface-storage assimilation space of this grid.

    Ws has no capacity, so unlike the soil blocks this one needs no capacities
    at all - only a reference maximum to bound the transferred state.
    """
    zone_map, zone_ids = build_reach_zone_map(gisdata)
    template = SimpleNamespace(ws=np.full((NROWS, NCOLS), 0.01))
    return build_surface_state_spec(
        state=template,
        reference_surface=np.full(len(zone_ids), 1.0),
        zone_map=zone_map,
        zone_ids=zone_ids,
    )


def _chained_discharge(gisdata, config, forcing_path, schedule, tmp_path, mutate=None):
    """Run the cycles one after another, carrying the state through a state file."""
    mask = build_state_mask(gisdata)
    blocks = []
    state_path = None

    for cycle in range(schedule.n_cycles):
        sim = _new_simulation(gisdata, config, forcing_path)
        if state_path is not None:
            state = read_state_file(state_path, mask)
            if mutate is not None:
                mutate(state)
            sim.set_initial_state(state=state)

        results = sim.run(schedule.starts[cycle].to_pydatetime(), schedule.ends[cycle].to_pydatetime())
        blocks.append(results.time_series["discharge"])

        state_path = tmp_path / f"state_c{cycle}.npz"
        write_state_file(state_path, results.final_state, mask)

    return np.concatenate(blocks, axis=0)


class TestCycleChaining:
    def test_chained_cycles_reproduce_a_continuous_run(self, gisdata, config, forcing_path, tmp_path):
        schedule = build_cycle_schedule(START, START + np.timedelta64((N_TIMES - 1) * DT, "s"), "1h", DT)
        assert schedule.n_cycles == 6
        assert schedule.n_steps_per_cycle == 4

        continuous = _continuous_discharge(
            gisdata, config, forcing_path, schedule.starts[0].to_pydatetime(), schedule.ends[-1].to_pydatetime()
        )
        # Guard against a vacuous comparison: the scenario must produce flow.
        assert continuous.max() > 1.0

        chained = _chained_discharge(gisdata, config, forcing_path, schedule, tmp_path)

        assert chained.shape == continuous.shape
        np.testing.assert_allclose(chained, continuous, rtol=1e-10, atol=0.0)

    def test_the_test_detects_a_state_variable_dropped_at_the_boundary(self, gisdata, config, forcing_path, tmp_path):
        """Guard against a vacuous pass: zeroing flr/fld must break the match.

        ``flr``/``fld`` are read before being written within a timestep, so they
        are genuine dynamic states. If the state file ever stopped carrying
        them, the comparison above has to fail rather than pass silently.
        """
        schedule = build_cycle_schedule(START, START + np.timedelta64((N_TIMES - 1) * DT, "s"), "1h", DT)

        continuous = _continuous_discharge(
            gisdata, config, forcing_path, schedule.starts[0].to_pydatetime(), schedule.ends[-1].to_pydatetime()
        )

        def drop_fluxes(state):
            state.flr = np.where(np.isnan(state.flr), np.nan, 0.0)
            state.fld = np.where(np.isnan(state.fld), np.nan, 0.0)

        degraded = _chained_discharge(gisdata, config, forcing_path, schedule, tmp_path, mutate=drop_fluxes)

        assert not np.allclose(degraded, continuous, rtol=1e-10, atol=0.0)

    def test_a_single_cycle_equals_the_continuous_run_over_the_same_window(
        self, gisdata, config, forcing_path, tmp_path
    ):
        """Sanity check with no state transfer involved."""
        schedule = build_cycle_schedule(START, START + np.timedelta64((N_TIMES - 1) * DT, "s"), "6h", DT)
        assert schedule.n_cycles == 1

        continuous = _continuous_discharge(
            gisdata, config, forcing_path, schedule.starts[0].to_pydatetime(), schedule.ends[0].to_pydatetime()
        )
        chained = _chained_discharge(gisdata, config, forcing_path, schedule, tmp_path)

        np.testing.assert_allclose(chained, continuous, rtol=1e-12, atol=0.0)

    def test_soil_state_round_trip_leaves_the_run_unchanged(self, gisdata, config, forcing_path, tmp_path):
        """Projecting Wc/Wg onto the zones and straight back must be a no-op.

        The filter changes nothing in this test: every cycle boundary encodes the
        soil stores as zone-averaged saturations and immediately decodes them.
        Anything the reduction loses on its own — a zone-indexing slip, a
        capacity normalised inconsistently between encode and decode, structure
        flattened inside a zone — shows up as a mismatch against the continuous
        run. Paired with the perturbation test below, which proves the decode is
        not silently doing nothing.
        """
        schedule = build_cycle_schedule(START, START + np.timedelta64((N_TIMES - 1) * DT, "s"), "1h", DT)
        spec, capacities = _soil_spec(gisdata, config, forcing_path)

        continuous = _continuous_discharge(
            gisdata, config, forcing_path, schedule.starts[0].to_pydatetime(), schedule.ends[-1].to_pydatetime()
        )

        def round_trip(state):
            vector = extract_state_vector(state, spec, capacities=capacities)
            insert_state_vector(vector, spec, state, capacities=capacities)

        chained = _chained_discharge(gisdata, config, forcing_path, schedule, tmp_path, mutate=round_trip)

        np.testing.assert_allclose(chained, continuous, rtol=1e-10, atol=0.0)

    def test_an_analysed_soil_state_actually_reaches_the_model(self, gisdata, config, forcing_path, tmp_path):
        """Guard against a vacuous pass: drying the zones must change the discharge."""
        schedule = build_cycle_schedule(START, START + np.timedelta64((N_TIMES - 1) * DT, "s"), "1h", DT)
        spec, capacities = _soil_spec(gisdata, config, forcing_path)

        continuous = _continuous_discharge(
            gisdata, config, forcing_path, schedule.starts[0].to_pydatetime(), schedule.ends[-1].to_pydatetime()
        )

        def dry_the_soil(state):
            vector = extract_state_vector(state, spec, capacities=capacities)
            insert_state_vector(0.5 * vector, spec, state, capacities=capacities)

        assimilated = _chained_discharge(gisdata, config, forcing_path, schedule, tmp_path, mutate=dry_the_soil)

        assert not np.allclose(assimilated, continuous, rtol=1e-6, atol=0.0)
        # Drier soil holds more of the rain back, so the discharge must fall.
        assert assimilated.sum() < continuous.sum()

    def test_decoding_preserves_the_pattern_inside_a_zone(self, gisdata, config, forcing_path, tmp_path):
        """The model spends the warm-up building that pattern; an update may only shift its level."""
        spec, capacities = _soil_spec(gisdata, config, forcing_path)
        sim = _new_simulation(gisdata, config, forcing_path)
        state = sim.run(START, START + np.timedelta64((N_TIMES - 1) * DT, "s")).final_state

        before = state.wc.copy()
        vector = extract_state_vector(state, spec, capacities=capacities)
        block = spec.block(KIND_SOIL_CAPILLARY)
        target = np.full(len(block), 0.5)
        insert_state_vector(np.concatenate([target, vector[len(block) :]]), spec, state, capacities=capacities)

        # Zone 0 is the western half of the grid (see the hillslope map fixture).
        zone = spec.zone_map == 0
        scale = state.wc[zone] / before[zone]
        assert np.ptp(scale) < 1e-9  # one common factor for the whole zone
        np.testing.assert_allclose(
            zone_saturation(state.wc, capacities[KIND_SOIL_CAPILLARY], spec.zone_map, block.ids),
            target,
            rtol=1e-9,
        )

    def test_surface_state_round_trip_leaves_the_run_unchanged(self, gisdata, config, forcing_path, tmp_path):
        """The same anchor property for Ws, which is a depth rather than a saturation."""
        schedule = build_cycle_schedule(START, START + np.timedelta64((N_TIMES - 1) * DT, "s"), "1h", DT)
        spec = _surface_spec(gisdata, config, forcing_path)

        continuous = _continuous_discharge(
            gisdata, config, forcing_path, schedule.starts[0].to_pydatetime(), schedule.ends[-1].to_pydatetime()
        )

        def round_trip(state):
            insert_state_vector(extract_state_vector(state, spec), spec, state)

        chained = _chained_discharge(gisdata, config, forcing_path, schedule, tmp_path, mutate=round_trip)
        np.testing.assert_allclose(chained, continuous, rtol=1e-10, atol=0.0)

    def test_an_analysed_surface_state_actually_reaches_the_model(self, gisdata, config, forcing_path, tmp_path):
        """Unlike soil moisture, Ws is on the runoff pathway, so it must move the discharge."""
        schedule = build_cycle_schedule(START, START + np.timedelta64((N_TIMES - 1) * DT, "s"), "1h", DT)
        spec = _surface_spec(gisdata, config, forcing_path)

        continuous = _continuous_discharge(
            gisdata, config, forcing_path, schedule.starts[0].to_pydatetime(), schedule.ends[-1].to_pydatetime()
        )

        def drain(state):
            insert_state_vector(0.5 * extract_state_vector(state, spec), spec, state)

        def flood(state):
            insert_state_vector(2.0 * extract_state_vector(state, spec), spec, state)

        drained = _chained_discharge(gisdata, config, forcing_path, schedule, tmp_path, mutate=drain)
        flooded = _chained_discharge(gisdata, config, forcing_path, schedule, tmp_path, mutate=flood)

        # Both directions must work: emptying the surface store lowers the
        # discharge and filling it raises it. A one-sided response would bias
        # the ensemble mean, which is what made soil moisture unusable.
        assert drained.sum() < continuous.sum() < flooded.sum()

    def test_simulation_mutates_the_state_object_it_is_given(self, gisdata, config, forcing_path):
        """Documents the hazard that forces PestSetup._run_reference to deep-copy.

        ``set_initial_state(state=X)`` stores X by reference and the main loop
        assigns into it (``results.final_state`` *is* X). Anything that runs a
        simulation and then reads the original state — the reference run that
        bounds the state parameters, for instance — must copy it first, or it
        silently reads the end of that run instead.
        """
        first = _new_simulation(gisdata, config, forcing_path)
        seed = first.run(START, START + np.timedelta64(3 * DT, "s")).final_state

        before = seed.wc.copy()
        second = _new_simulation(gisdata, config, forcing_path)
        second.set_initial_state(state=seed)
        results = second.run(START + np.timedelta64(4 * DT, "s"), START + np.timedelta64(8 * DT, "s"))

        assert results.final_state is seed
        assert not np.allclose(seed.wc, before, equal_nan=True)

    def test_wrong_window_convention_double_counts_a_timestep(self, gisdata, config, forcing_path):
        """[t_c, t_{c+1}] would apply the boundary forcing twice and add a step."""
        schedule = build_cycle_schedule(START, START + np.timedelta64((N_TIMES - 1) * DT, "s"), "1h", DT)

        correct = schedule.ends[0]
        wrong = schedule.starts[1]
        assert (wrong - correct).total_seconds() == DT

        sim = _new_simulation(gisdata, config, forcing_path)
        wrong_run = sim.run(schedule.starts[0].to_pydatetime(), wrong.to_pydatetime())
        assert len(wrong_run.time_series["time"]) == schedule.n_steps_per_cycle + 1
