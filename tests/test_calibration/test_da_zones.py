"""Zone-reduced soil-moisture assimilation (Wc and Wg).

A soil state is a *zone-averaged saturation*, not a field: the full-resolution
``Wc``/``Wg`` grids are carried in the state file, and only one number per zone
per store travels through the PEST interface. These tests pin the three
properties that make that reduction safe:

- the zones come from the hillslope-reach mapping the model already uses;
- encoding then decoding an unchanged saturation leaves the field untouched, so
  a cycle in which the filter changes nothing is a no-op;
- decoding rescales the realization's own field, so the pattern inside a zone
  survives an update and only its overall level moves.
"""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from mobidic.calibration.da_states import (
    KIND_DISCHARGE,
    KIND_SOIL_CAPILLARY,
    KIND_CONDUCTIVITY,
    KIND_RUNOFF_FRACTION,
    KIND_SOIL_GRAVITATIONAL,
    KIND_SURFACE_WATER,
    NO_ZONE,
    StateSpec,
    apply_zone_parameters,
    build_discharge_state_spec,
    build_reach_zone_map,
    build_soil_state_spec,
    build_zone_parameter_spec,
    build_surface_state_spec,
    build_upstream_localizer,
    extract_state_vector,
    insert_state_vector,
    rescale_to_zone_saturation,
    rescale_zone_field,
    resolve_estimate_kinds,
    state_par_name,
    zone_mean,
    zone_saturation,
)

NAN = float("nan")


def _network(n_reaches=4):
    """A chain 0 -> 1 -> 2 -> 3, so reach r has every lower id upstream of it."""
    return gpd.GeoDataFrame(
        {
            "mobidic_id": list(range(n_reaches)),
            "upstream_1": [NAN] + [float(i - 1) for i in range(1, n_reaches)],
            "upstream_2": [NAN] * n_reaches,
            "downstream": [float(i + 1) for i in range(n_reaches - 1)] + [NAN],
            "geometry": [LineString([(0, i), (1, i)]) for i in range(n_reaches)],
        }
    )


def _gisdata(hillslope, dtm=None, network=None):
    """Minimal stand-in for GISData: the zone builder only needs these three."""
    hillslope = np.asarray(hillslope, dtype=np.int64)
    if dtm is None:
        dtm = np.where(hillslope >= 0, 100.0, NAN)
    return SimpleNamespace(
        grids={"dtm": np.asarray(dtm, dtype=np.float64)},
        hillslope_reach_map=hillslope,
        network=_network() if network is None else network,
    )


def _state(**grids):
    return SimpleNamespace(discharge=np.zeros(4), **grids)


class TestResolveEstimateKinds:
    def test_soil_moisture_expands_to_both_stores(self):
        assert resolve_estimate_kinds(["soil_moisture"]) == (KIND_SOIL_CAPILLARY, KIND_SOIL_GRAVITATIONAL)

    def test_the_stores_can_be_selected_one_at_a_time(self):
        assert resolve_estimate_kinds([KIND_SOIL_CAPILLARY]) == (KIND_SOIL_CAPILLARY,)

    def test_order_is_stable_regardless_of_how_it_was_written(self):
        assert resolve_estimate_kinds(["soil_moisture", "discharge"]) == (
            KIND_DISCHARGE,
            KIND_SOIL_CAPILLARY,
            KIND_SOIL_GRAVITATIONAL,
        )

    def test_an_alias_overlapping_its_component_is_rejected(self):
        with pytest.raises(ValueError, match="more than once"):
            resolve_estimate_kinds(["soil_moisture", KIND_SOIL_CAPILLARY])

    def test_an_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown state variable"):
            resolve_estimate_kinds(["snow"])


class TestBuildReachZoneMap:
    def test_zones_follow_the_hillslope_reach_mapping(self):
        hillslope = np.array([[0, 0, 1, 1], [0, 2, 2, 1], [3, 3, 2, -9999], [3, 3, -9999, -9999]])
        zone_map, zone_ids = build_reach_zone_map(_gisdata(hillslope))

        assert zone_ids == (0, 1, 2, 3)
        np.testing.assert_array_equal(zone_map[zone_map != NO_ZONE], hillslope[hillslope >= 0])
        # The unassigned sentinel becomes 'no zone', never zone -9999.
        assert (zone_map[hillslope < 0] == NO_ZONE).all()

    def test_a_float_map_with_nan_padding_is_handled(self):
        """This is the form gisdata.nc actually round-trips: float64 with NaN, not int.

        A NaN comparison is False either way, but ``astype(int)`` on it would
        produce a large negative number, and an int cast of the whole grid would
        turn the padding into a spurious zone.
        """
        hillslope = np.array([[0.0, 1.0], [np.nan, 1.0]])
        gisdata = SimpleNamespace(
            grids={"dtm": np.full((2, 2), 100.0)},
            hillslope_reach_map=hillslope,
            network=_network(),
        )
        zone_map, zone_ids = build_reach_zone_map(gisdata)

        assert zone_ids == (0, 1)
        assert zone_map[1, 0] == NO_ZONE
        assert zone_map.dtype.kind == "i"

    def test_cells_outside_the_dtm_are_excluded(self):
        """The soil balance runs on isfinite(dtm) & (map >= 0); zones must match."""
        hillslope = np.array([[0, 0], [1, 1]])
        dtm = np.array([[100.0, NAN], [100.0, 100.0]])
        zone_map, zone_ids = build_reach_zone_map(_gisdata(hillslope, dtm=dtm))

        assert zone_map[0, 1] == NO_ZONE
        assert zone_ids == (0, 1)

    def test_restricting_to_the_upstream_closure_drops_the_rest(self):
        hillslope = np.array([[0, 1], [2, 3]])
        zone_map, zone_ids = build_reach_zone_map(_gisdata(hillslope), reach_ids=[0, 1])

        assert zone_ids == (0, 1)
        assert (zone_map[1, :] == NO_ZONE).all()

    def test_a_grid_mismatch_raises(self):
        gisdata = _gisdata(np.zeros((2, 2), dtype=np.int64))
        gisdata.grids["dtm"] = np.zeros((3, 3))
        with pytest.raises(ValueError, match="same preprocessing run"):
            build_reach_zone_map(gisdata)

    def test_no_zone_at_all_raises(self):
        with pytest.raises(ValueError, match="No soil-moisture zone"):
            build_reach_zone_map(_gisdata(np.full((2, 2), -9999, dtype=np.int64)))


class TestMergeSmallZones:
    """Small zones are merged downstream, following the drainage direction."""

    def _map(self):
        # zone 0: 1 cell, zone 1: 1 cell, zone 2: 4 cells, zone 3: 2 cells
        return np.array([[0, 1], [2, 2], [2, 2], [3, 3]])

    def test_small_zones_merge_into_the_first_large_downstream_zone(self):
        zone_map, zone_ids = build_reach_zone_map(_gisdata(self._map()), min_zone_cells=3)

        # 0 -> 1 -> 2: both undersized, both land in 2, the first big enough.
        # 3 is undersized too but is the outlet, so it has nowhere to go.
        assert zone_ids == (2, 3)
        assert (zone_map == 2).sum() == 6

    def test_a_large_enough_zone_is_left_alone(self):
        zone_map, zone_ids = build_reach_zone_map(_gisdata(self._map()), min_zone_cells=2)

        # 0 and 1 hold one cell each and merge; 3 holds two and stays.
        assert zone_ids == (2, 3)
        assert (zone_map == 3).sum() == 2

    def test_a_small_tail_with_nowhere_to_go_survives(self):
        """The outlet zone has no downstream neighbour to merge into."""
        zone_map, zone_ids = build_reach_zone_map(_gisdata(self._map()), min_zone_cells=100)

        assert zone_ids == (3,)
        assert (zone_map != NO_ZONE).sum() == 8

    def test_merging_never_leaves_the_estimated_set(self):
        """Merging past a gauge would move cells into a zone nobody estimates."""
        zone_map, zone_ids = build_reach_zone_map(_gisdata(self._map()), reach_ids=[0, 1, 2], min_zone_cells=3)

        assert zone_ids == (2,)
        assert (zone_map == NO_ZONE).sum() == 2  # the two cells of reach 3

    def test_no_merging_by_default(self):
        _, zone_ids = build_reach_zone_map(_gisdata(self._map()))
        assert zone_ids == (0, 1, 2, 3)


class TestZoneSaturation:
    def test_saturation_is_total_storage_over_total_capacity(self):
        """The ratio of sums, not the mean of ratios: the reduction stays mass-consistent."""
        zone_map = np.array([[0, 0], [1, 1]])
        values = np.array([[1.0, 3.0], [1.0, 1.0]])
        capacity = np.array([[2.0, 6.0], [4.0, 4.0]])

        np.testing.assert_allclose(zone_saturation(values, capacity, zone_map, (0, 1)), [0.5, 0.25])

    def test_a_capacity_weighted_zone_is_not_a_plain_cell_average(self):
        zone_map = np.array([[0, 0]])
        values = np.array([[1.0, 1.0]])
        capacity = np.array([[1.0, 9.0]])

        # sum(W)/sum(W0) = 2/10, whereas the mean of per-cell ratios would be 0.55.
        np.testing.assert_allclose(zone_saturation(values, capacity, zone_map, (0,)), [0.2])

    def test_cells_outside_every_zone_are_ignored(self):
        zone_map = np.array([[0, NO_ZONE]])
        values = np.array([[1.0, 1000.0]])
        capacity = np.array([[2.0, 2.0]])

        np.testing.assert_allclose(zone_saturation(values, capacity, zone_map, (0,)), [0.5])

    def test_nan_padding_does_not_poison_a_zone(self):
        zone_map = np.array([[0, 0]])
        values = np.array([[1.0, NAN]])
        capacity = np.array([[2.0, NAN]])

        np.testing.assert_allclose(zone_saturation(values, capacity, zone_map, (0,)), [0.5])

    def test_an_empty_zone_reports_zero_rather_than_nan(self):
        zone_map = np.array([[0, 0]])
        np.testing.assert_allclose(zone_saturation(np.ones((1, 2)), np.ones((1, 2)), zone_map, (0, 7)), [1.0, 0.0])


class TestRescaleToZoneSaturation:
    """Decoding must preserve the within-zone pattern, not flatten it."""

    def _fields(self):
        zone_map = np.array([[0, 0], [1, 1]])
        capacity = np.full((2, 2), 10.0)
        background = np.array([[1.0, 3.0], [5.0, 5.0]])
        return zone_map, capacity, background

    def test_an_unchanged_target_is_a_no_op(self):
        zone_map, capacity, background = self._fields()
        theta = zone_saturation(background, capacity, zone_map, (0, 1))

        updated = rescale_to_zone_saturation(background, capacity, zone_map, (0, 1), theta)
        np.testing.assert_allclose(updated, background)

    def test_the_zone_reaches_the_requested_saturation(self):
        zone_map, capacity, background = self._fields()
        target = np.array([0.4, 0.1])

        updated = rescale_to_zone_saturation(background, capacity, zone_map, (0, 1), target)
        np.testing.assert_allclose(zone_saturation(updated, capacity, zone_map, (0, 1)), target)

    def test_within_zone_structure_is_preserved_exactly(self):
        zone_map, capacity, background = self._fields()

        updated = rescale_to_zone_saturation(background, capacity, zone_map, (0, 1), np.array([0.4, 0.1]))
        # Zone 0 held 1 and 3; after any pure rescaling it still holds a 1:3 ratio.
        assert updated[0, 1] / updated[0, 0] == pytest.approx(3.0)

    def test_a_dry_zone_receives_the_target_uniformly(self):
        """There is no structure to preserve, so the only defensible rule is uniform."""
        zone_map = np.array([[0, 0]])
        capacity = np.array([[10.0, 30.0]])
        background = np.zeros((1, 2))

        updated = rescale_to_zone_saturation(background, capacity, zone_map, (0,), np.array([0.5]))
        np.testing.assert_allclose(updated, [[5.0, 15.0]])

    def test_the_field_never_exceeds_capacity(self):
        zone_map = np.array([[0, 0]])
        capacity = np.array([[10.0, 10.0]])
        background = np.array([[1.0, 9.0]])

        updated = rescale_to_zone_saturation(background, capacity, zone_map, (0,), np.array([1.0]))
        assert (updated <= capacity).all()

    def test_a_clipped_zone_is_allowed_to_fall_short(self):
        """Redistributing the shortfall would move water the analysis said nothing about."""
        zone_map = np.array([[0, 0]])
        capacity = np.array([[10.0, 10.0]])
        background = np.array([[1.0, 9.0]])

        updated = rescale_to_zone_saturation(background, capacity, zone_map, (0,), np.array([1.0]))
        assert zone_saturation(updated, capacity, zone_map, (0,))[0] < 1.0

    def test_cells_outside_every_zone_keep_their_background(self):
        zone_map = np.array([[0, NO_ZONE]])
        capacity = np.array([[10.0, 10.0]])
        background = np.array([[2.0, 7.0]])

        updated = rescale_to_zone_saturation(background, capacity, zone_map, (0,), np.array([0.5]))
        assert updated[0, 1] == 7.0

    def test_nan_padding_survives(self):
        zone_map = np.array([[0, NO_ZONE]])
        capacity = np.array([[10.0, NAN]])
        background = np.array([[2.0, NAN]])

        updated = rescale_to_zone_saturation(background, capacity, zone_map, (0,), np.array([0.5]))
        assert np.isnan(updated[0, 1])

    def test_the_background_array_is_not_written_through(self):
        zone_map, capacity, background = self._fields()
        original = background.copy()

        rescale_to_zone_saturation(background, capacity, zone_map, (0, 1), np.array([0.9, 0.9]))
        np.testing.assert_allclose(background, original)


class TestBuildSoilStateSpec:
    def _spec(self, **kwargs):
        zone_map = np.array([[0, 0], [1, 1]])
        capacity = np.full((2, 2), 10.0)
        state = _state(wc=np.array([[2.0, 4.0], [5.0, 5.0]]), wg=np.array([[1.0, 1.0], [0.0, 0.0]]))
        return build_soil_state_spec(
            kinds=[KIND_SOIL_CAPILLARY, KIND_SOIL_GRAVITATIONAL],
            state=state,
            capacities={KIND_SOIL_CAPILLARY: capacity, KIND_SOIL_GRAVITATIONAL: capacity},
            zone_map=zone_map,
            zone_ids=(0, 1),
            **kwargs,
        )

    def test_one_block_per_store_with_the_warm_up_saturation(self):
        spec = self._spec()

        assert spec.kinds == (KIND_SOIL_CAPILLARY, KIND_SOIL_GRAVITATIONAL)
        np.testing.assert_allclose(spec.block(KIND_SOIL_CAPILLARY).initial, [0.3, 0.5])
        assert len(spec) == 4

    def test_names_carry_the_store_and_the_zone(self):
        spec = self._spec()

        assert spec.par_names[:2] == ["sp_wc_0000", "sp_wc_0001"]
        assert spec.obs_names[2:] == ["st_wg_0000", "st_wg_0001"]
        assert spec.input_keys[0] == "__state__.wc.0000"

    def test_bounds_are_physical_and_the_lower_one_is_normal(self):
        """A saturation is confined to [0,1], so PEST++ can never truncate the transfer."""
        spec = self._spec()

        assert np.all(spec.upper <= 1.0)
        # Positive, never 0.0: a zero lower bound leaves subnormals feasible and
        # PEST++ refuses to queue a run whose parameter vector holds one.
        assert np.all(spec.lower > 0.0)
        assert np.all(spec.lower > np.finfo(np.float64).tiny)

    def test_a_dry_zone_starts_on_the_floor_not_below_it(self):
        """parval1 below parlbnd is a control-file error."""
        spec = self._spec()
        block = spec.block(KIND_SOIL_GRAVITATIONAL)

        assert block.initial[1] == pytest.approx(block.lower[1])

    def test_degenerate_bounds_are_rejected(self):
        with pytest.raises(ValueError, match="usable interval"):
            self._spec(saturation_bounds=(0.5, 0.5))

    def test_a_missing_capacity_is_reported(self):
        with pytest.raises(ValueError, match="No capacity grid"):
            build_soil_state_spec(
                kinds=[KIND_SOIL_CAPILLARY],
                state=_state(wc=np.ones((2, 2))),
                capacities={},
                zone_map=np.zeros((2, 2), dtype=np.int64),
                zone_ids=(0,),
            )


class TestCombinedSpec:
    """Discharge and both soil stores in one interface."""

    def _spec(self):
        zone_map = np.array([[0, 0], [1, 1]])
        capacity = np.full((2, 2), 10.0)
        state = _state(wc=np.array([[2.0, 4.0], [5.0, 5.0]]), wg=np.full((2, 2), 1.0))
        discharge = build_discharge_state_spec(
            network=_network(),
            initial_discharge=[1.0, 2.0, 3.0, 4.0],
            reference_discharge=[10.0] * 4,
            reach_ids=[0, 1],
        )
        soil = build_soil_state_spec(
            kinds=[KIND_SOIL_CAPILLARY, KIND_SOIL_GRAVITATIONAL],
            state=state,
            capacities={KIND_SOIL_CAPILLARY: capacity, KIND_SOIL_GRAVITATIONAL: capacity},
            zone_map=zone_map,
            zone_ids=(0, 1),
        )
        return StateSpec.combine(discharge, soil), capacity, state

    def test_the_interface_concatenates_the_blocks_in_order(self):
        spec, _, _ = self._spec()

        assert spec.kinds == (KIND_DISCHARGE, KIND_SOIL_CAPILLARY, KIND_SOIL_GRAVITATIONAL)
        assert len(spec) == 6
        assert spec.par_names[0] == "sp_q_0000"
        assert spec.par_names[-1] == "sp_wg_0001"
        assert len(spec.initial) == len(spec.lower) == len(spec.upper) == 6

    def test_extract_then_insert_is_the_identity(self):
        spec, capacity, state = self._spec()
        capacities = {KIND_SOIL_CAPILLARY: capacity, KIND_SOIL_GRAVITATIONAL: capacity}
        state.discharge = np.array([7.0, 8.0, 9.0, 10.0])

        vector = extract_state_vector(state, spec, capacities=capacities)
        insert_state_vector(vector, spec, state, capacities=capacities)

        np.testing.assert_allclose(extract_state_vector(state, spec, capacities=capacities), vector)

    def test_each_block_updates_only_its_own_grid(self):
        spec, capacity, state = self._spec()
        capacities = {KIND_SOIL_CAPILLARY: capacity, KIND_SOIL_GRAVITATIONAL: capacity}
        wg_before = state.wg.copy()

        vector = extract_state_vector(state, spec, capacities=capacities)
        vector[2:4] = 0.5  # the two capillary zones; the layout is q, wc, wg
        insert_state_vector(vector, spec, state, capacities=capacities)

        np.testing.assert_allclose(zone_saturation(state.wc, capacity, spec.zone_map, (0, 1)), [0.5, 0.5])
        np.testing.assert_allclose(state.wg, wg_before)

    def test_combining_the_same_kind_twice_is_rejected(self):
        spec, _, _ = self._spec()
        with pytest.raises(ValueError, match="more than one spec"):
            StateSpec.combine(spec, spec)

    def test_soil_states_need_their_capacities(self):
        spec, _, state = self._spec()
        with pytest.raises(ValueError, match="capacity grid"):
            extract_state_vector(state, spec)

    def test_json_roundtrip_carries_the_zone_map(self, tmp_path):
        spec, _, _ = self._spec()
        back = StateSpec.from_json(spec.to_json(tmp_path / "spec.json"))

        assert back.kinds == spec.kinds
        assert back.par_names == spec.par_names
        np.testing.assert_array_equal(back.zone_map, spec.zone_map)

    def test_a_missing_zone_map_file_raises(self, tmp_path):
        spec, _, _ = self._spec()
        path = spec.to_json(tmp_path / "spec.json")
        (tmp_path / "zone_map.npy").unlink()

        with pytest.raises(FileNotFoundError, match="zone map"):
            StateSpec.from_json(path)

    def test_a_grid_mismatch_is_caught(self):
        spec, _, _ = self._spec()
        spec.check_grid((2, 2))
        with pytest.raises(ValueError, match="different setup"):
            spec.check_grid((3, 3))

    def test_check_against_rejects_a_spec_that_lost_its_zone_map(self):
        spec, _, _ = self._spec()
        stripped = StateSpec(blocks=spec.blocks, zone_map=None)

        with pytest.raises(ValueError, match="no zone map"):
            stripped.check_against(_network())


class TestSurfaceWater:
    """Ws has no capacity, so its state is an absolute zone-mean depth [m]."""

    def _spec(self, reference=None, **kwargs):
        zone_map = np.array([[0, 0], [1, 1]])
        state = _state(ws=np.array([[1.0, 3.0], [2.0, 2.0]]))
        return (
            build_surface_state_spec(
                state=state,
                reference_surface=np.array([10.0, 10.0]) if reference is None else reference,
                zone_map=zone_map,
                zone_ids=(0, 1),
                **kwargs,
            ),
            zone_map,
            state,
        )

    def test_the_state_is_the_zone_mean_depth(self):
        spec, _, _ = self._spec()
        np.testing.assert_allclose(spec.block(KIND_SURFACE_WATER).initial, [2.0, 2.0])

    def test_names_use_the_ws_tag(self):
        spec, _, _ = self._spec()
        assert spec.par_names == ["sp_ws_0000", "sp_ws_0001"]
        assert spec.obs_names == ["st_ws_0000", "st_ws_0001"]
        assert spec.input_keys[0] == "__state__.ws.0000"

    def test_bounds_come_from_the_reference_run(self):
        """No capacity exists, so the bound cannot be physical (unlike a saturation)."""
        spec, _, _ = self._spec(reference=np.array([5.0, 20.0]), bound_factor=10.0)
        np.testing.assert_allclose(spec.upper, [50.0, 200.0])
        assert np.all(spec.lower > 0.0)

    def test_a_zone_that_never_holds_water_keeps_a_usable_interval(self):
        spec, _, _ = self._spec(reference=np.array([0.0, 0.0]))
        assert np.all(spec.upper > spec.lower)

    def test_a_mismatched_reference_is_rejected(self):
        with pytest.raises(ValueError, match="one per zone"):
            self._spec(reference=np.array([1.0]))

    def test_it_is_zonal_but_not_soil(self):
        """is_soil drives the capacity lookup; surface storage must not trigger it."""
        block = self._spec()[0].block(KIND_SURFACE_WATER)
        assert block.is_zonal and not block.is_soil

    def test_no_capacity_is_needed_to_project(self):
        spec, _, state = self._spec()
        np.testing.assert_allclose(extract_state_vector(state, spec), [2.0, 2.0])

    def test_decode_scales_every_pixel_by_one_factor(self):
        """This is exactly Ws_assim = Ws * theta_ws, with theta_ws = target/background."""
        spec, zone_map, state = self._spec()
        insert_state_vector(np.array([4.0, 1.0]), spec, state)

        # Zone 0 held 1 and 3 (mean 2); asking for mean 4 doubles both.
        np.testing.assert_allclose(state.ws[0], [2.0, 6.0])
        # Zone 1 held 2 and 2; asking for mean 1 halves both.
        np.testing.assert_allclose(state.ws[1], [1.0, 1.0])

    def test_the_target_is_reached_exactly_in_both_directions(self):
        """With no capacity there is no upper clip, so an increase can never be truncated."""
        spec, zone_map, state = self._spec()
        for target in ([0.1, 0.1], [50.0, 50.0], [2.0, 2.0]):
            s = _state(ws=np.array([[1.0, 3.0], [2.0, 2.0]]))
            insert_state_vector(np.array(target), spec, s)
            np.testing.assert_allclose(zone_mean(s.ws, zone_map, (0, 1)), target, rtol=1e-12)

    def test_an_empty_zone_receives_the_target_uniformly(self):
        spec, zone_map, _ = self._spec()
        s = _state(ws=np.zeros((2, 2)))
        insert_state_vector(np.array([3.0, 5.0]), spec, s)
        np.testing.assert_allclose(s.ws, [[3.0, 3.0], [5.0, 5.0]])

    def test_negative_targets_are_floored_at_zero(self):
        spec, zone_map, state = self._spec()
        insert_state_vector(np.array([-1.0, 1.0]), spec, state)
        assert (state.ws >= 0).all()

    def test_cells_outside_every_zone_and_nan_padding_survive(self):
        zone_map = np.array([[0, NO_ZONE]])
        background = np.array([[2.0, np.nan]])
        out = rescale_zone_field(background, zone_map, (0,), np.array([4.0]))
        assert out[0, 0] == 4.0 and np.isnan(out[0, 1])


class TestZoneMean:
    def test_it_is_a_plain_mean_over_the_zone_cells(self):
        zone_map = np.array([[0, 0], [1, NO_ZONE]])
        np.testing.assert_allclose(zone_mean(np.array([[1.0, 3.0], [7.0, 99.0]]), zone_map, (0, 1)), [2.0, 7.0])

    def test_nan_cells_are_ignored(self):
        zone_map = np.array([[0, 0]])
        np.testing.assert_allclose(zone_mean(np.array([[4.0, np.nan]]), zone_map, (0,)), [4.0])

    def test_an_empty_zone_is_zero_not_nan(self):
        np.testing.assert_allclose(zone_mean(np.ones((1, 2)), np.array([[0, 0]]), (0, 9)), [1.0, 0.0])


class TestZoneParameters:
    """Distributed parameters (f0, ks multiplier) rather than model states.

    These exist because a *state*'s ensemble spread collapses: the surface store
    grows ~25x during a storm while the absolute spread stays at its cycle-0
    value, so the relative spread falls to ~0.2% and the Kalman gain with it. A
    dimensionless parameter has no such scale.
    """

    def _spec(self, kinds=(KIND_RUNOFF_FRACTION,), **kwargs):
        return build_zone_parameter_spec(kinds, np.array([[0, 0], [1, 1]]), (0, 1), **kwargs)

    def test_they_are_parameters_not_states(self):
        spec = self._spec()
        block = spec.block(KIND_RUNOFF_FRACTION)
        assert block.is_parameter and block.is_zonal
        assert not block.linked

    def test_a_parameter_block_gets_no_observation_or_state_link(self):
        """PESTPP-DA carries an adjustable parameter's ensemble by itself."""
        spec = self._spec()
        assert spec.par_names == ["sp_f0_0000", "sp_f0_0001"]
        assert spec.obs_names == []
        assert spec.linked_blocks == ()
        assert len(spec.parameter_blocks) == 1

    def test_f0_starts_from_the_model_default_and_is_bounded(self):
        spec = self._spec()
        block = spec.block(KIND_RUNOFF_FRACTION)
        np.testing.assert_allclose(block.initial, 0.0187)
        assert np.all(block.lower > 0.0) and np.all(block.upper < 1.0)

    def test_the_conductivity_block_is_a_multiplier(self):
        spec = self._spec(kinds=(KIND_CONDUCTIVITY,))
        block = spec.block(KIND_CONDUCTIVITY)
        assert block.multiplier
        np.testing.assert_allclose(block.initial, 1.0)

    def test_degenerate_bounds_are_rejected(self):
        with pytest.raises(ValueError, match="0 < lower < upper"):
            self._spec(bounds={KIND_RUNOFF_FRACTION: (0.5, 0.5)})

    def test_a_non_parameter_kind_is_rejected(self):
        with pytest.raises(ValueError, match="not a zone parameter"):
            build_zone_parameter_spec([KIND_SURFACE_WATER], np.zeros((2, 2), dtype=np.int64), (0,))

    def test_extract_ignores_parameter_blocks(self):
        """There is nothing simulated to report back for a parameter."""
        spec = self._spec()
        assert extract_state_vector(_state(ws=np.zeros((2, 2))), spec).shape == (0,)

    def test_insert_leaves_the_state_alone(self):
        spec = self._spec()
        state = _state(ws=np.ones((2, 2)))
        insert_state_vector(np.array([0.5, 0.5]), spec, state)
        np.testing.assert_allclose(state.ws, np.ones((2, 2)))


class TestApplyZoneParameters:
    """The values must reach Simulation.param_grids, which the loop reads each step."""

    def _sim(self):
        return SimpleNamespace(
            param_grids={"f0": np.full((2, 2), 0.0187), "ks": np.full((2, 2), 2.0)},
        )

    def test_f0_replaces_the_grid_per_zone(self):
        spec = build_zone_parameter_spec([KIND_RUNOFF_FRACTION], np.array([[0, 0], [1, 1]]), (0, 1))
        sim = self._sim()
        assert apply_zone_parameters(sim, spec, np.array([0.30, 0.60])) == 1
        np.testing.assert_allclose(sim.param_grids["f0"], [[0.30, 0.30], [0.60, 0.60]])

    def test_ks_multiplies_the_model_grid(self):
        spec = build_zone_parameter_spec([KIND_CONDUCTIVITY], np.array([[0, 0], [1, 1]]), (0, 1))
        sim = self._sim()
        apply_zone_parameters(sim, spec, np.array([0.5, 4.0]))
        np.testing.assert_allclose(sim.param_grids["ks"], [[1.0, 1.0], [8.0, 8.0]])

    def test_cells_outside_every_zone_keep_the_model_value(self):
        """estimate_reaches: upstream leaves the rest of the basin untouched."""
        spec = build_zone_parameter_spec([KIND_RUNOFF_FRACTION], np.array([[0, NO_ZONE]]), (0,))
        sim = SimpleNamespace(param_grids={"f0": np.array([[0.0187, 0.0187]])})
        apply_zone_parameters(sim, spec, np.array([0.5]))
        np.testing.assert_allclose(sim.param_grids["f0"], [[0.5, 0.0187]])

    def test_values_are_clipped_to_the_bounds(self):
        spec = build_zone_parameter_spec([KIND_RUNOFF_FRACTION], np.array([[0, 0], [1, 1]]), (0, 1))
        sim = self._sim()
        apply_zone_parameters(sim, spec, np.array([-1.0, 99.0]))
        assert sim.param_grids["f0"].min() > 0.0
        assert sim.param_grids["f0"].max() < 1.0

    def test_only_the_parameter_part_of_a_mixed_vector_is_used(self):
        """A combined setup slices the interface vector by block."""
        soil = build_soil_state_spec(
            kinds=[KIND_SOIL_CAPILLARY],
            state=_state(wc=np.full((2, 2), 5.0)),
            capacities={KIND_SOIL_CAPILLARY: np.full((2, 2), 10.0)},
            zone_map=np.array([[0, 0], [1, 1]]),
            zone_ids=(0, 1),
        )
        params = build_zone_parameter_spec([KIND_RUNOFF_FRACTION], np.array([[0, 0], [1, 1]]), (0, 1))
        spec = StateSpec.combine(soil, params)
        sim = self._sim()

        apply_zone_parameters(sim, spec, np.array([0.9, 0.9, 0.25, 0.75]))
        np.testing.assert_allclose(sim.param_grids["f0"], [[0.25, 0.25], [0.75, 0.75]])


class TestLocalizerWithSoilStates:
    def test_a_gauge_may_update_only_the_zones_draining_into_it(self):
        """A zone is named after the reach its cells drain to, so one rule covers both."""
        zone_map = np.array([[0, 1], [2, 3]])
        capacity = np.full((2, 2), 10.0)
        soil = build_soil_state_spec(
            kinds=[KIND_SOIL_CAPILLARY],
            state=_state(wc=np.full((2, 2), 5.0)),
            capacities={KIND_SOIL_CAPILLARY: capacity},
            zone_map=zone_map,
            zone_ids=(0, 1, 2, 3),
        )
        network = _network()

        matrix = build_upstream_localizer(
            network=network,
            obs_reaches={"gauge": 1},  # chain 0 -> 1, so zones 0 and 1 drain to it
            spec=soil,
            global_par_names=["wcel"],
        )

        assert matrix.loc["gauge", state_par_name(KIND_SOIL_CAPILLARY, 0)] == 1.0
        assert matrix.loc["gauge", state_par_name(KIND_SOIL_CAPILLARY, 1)] == 1.0
        assert matrix.loc["gauge", state_par_name(KIND_SOIL_CAPILLARY, 2)] == 0.0
        # Every adjustable parameter must be a column, and a basin-wide one is
        # updatable by every group: a missing column is treated as fixed.
        assert matrix.loc["gauge", "wcel"] == 1.0
