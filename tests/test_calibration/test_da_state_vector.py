"""Joint state-parameter estimation: the assimilation space and its projections.

Covers :class:`StateSpec` and the two projections that move a state vector in
and out of a full model state (formulation 2 of the PESTPP-DA user guide).
"""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from mobidic.calibration.da_states import (
    STATE_INPUT_PREFIX,
    STATE_VALUE_WIDTH,
    StateSpec,
    build_discharge_state_spec,
    extract_state_vector,
    insert_state_vector,
    is_state_input_key,
    state_input_key,
    state_obs_name,
    state_par_name,
    upstream_reaches,
)
from mobidic.calibration.template import MIN_FIELD_WIDTH


def _network():
    """A four-reach network: 0 and 1 join into 2, which flows into 3.

    Reach 3 is downstream of the others, so it is *not* in the upstream closure
    of reach 2. 'No upstream reach' is NaN, matching the GeoParquet the
    preprocessing writes.
    """
    nan = float("nan")
    return gpd.GeoDataFrame(
        {
            "mobidic_id": [0, 1, 2, 3],
            "upstream_1": [nan, nan, 0.0, 2.0],
            "upstream_2": [nan, nan, 1.0, nan],
            "downstream": [2.0, 2.0, 3.0, nan],
            "geometry": [LineString([(0, i), (1, i)]) for i in range(4)],
        }
    )


def _state(discharge):
    return SimpleNamespace(discharge=np.asarray(discharge, dtype=np.float64))


class TestNames:
    def test_names_carry_the_reach_id(self):
        assert state_par_name("discharge", 278) == "sp_q_0278"
        assert state_obs_name("discharge", 278) == "st_q_0278"
        assert state_input_key("discharge", 278) == "__state__.q.0278"

    def test_input_keys_are_recognised_as_reserved(self):
        assert is_state_input_key(state_input_key("discharge", 7))
        assert not is_state_input_key("parameters.soil.alpha")
        assert not is_state_input_key("__cycle__")

    def test_the_value_field_is_wide_enough_for_the_template(self):
        """PEST++ fits a value to the field width and silently truncates."""
        assert STATE_VALUE_WIDTH >= MIN_FIELD_WIDTH


class TestUpstreamReaches:
    def test_closure_excludes_downstream_reaches(self):
        assert upstream_reaches(_network(), [2]) == [0, 1, 2]

    def test_closure_of_the_outlet_is_the_whole_network(self):
        assert upstream_reaches(_network(), [3]) == [0, 1, 2, 3]

    def test_headwater_closure_is_itself(self):
        assert upstream_reaches(_network(), [0]) == [0]

    def test_several_gauges_are_merged(self):
        assert upstream_reaches(_network(), [0, 1]) == [0, 1]

    def test_unknown_reach_raises(self):
        with pytest.raises(KeyError, match="not in the network"):
            upstream_reaches(_network(), [99])


class TestBuildDischargeStateSpec:
    def test_bounds_come_from_the_reference_run_not_the_initial_value(self):
        """Bounds are enforced on the transferred state, so they must cover the peak."""
        spec = build_discharge_state_spec(
            network=_network(),
            initial_discharge=[1.0, 2.0, 3.0, 4.0],
            reference_discharge=[10.0, 20.0, 30.0, 40.0],
            bound_factor=10.0,
        )
        np.testing.assert_allclose(spec.upper, [100.0, 200.0, 300.0, 400.0])
        np.testing.assert_allclose(spec.initial, [1.0, 2.0, 3.0, 4.0])

    def test_a_reach_that_never_flows_still_gets_a_usable_interval(self):
        """PEST requires parubnd > parlbnd, so the bound cannot collapse to zero."""
        spec = build_discharge_state_spec(
            network=_network(),
            initial_discharge=[0.0, 0.0, 0.0, 0.0],
            reference_discharge=[0.0, 0.0, 0.0, 0.0],
        )
        assert np.all(spec.upper > spec.lower)

    def test_a_reference_below_the_initial_value_does_not_shrink_the_bound(self):
        spec = build_discharge_state_spec(
            network=_network(),
            initial_discharge=[50.0, 0.0, 0.0, 0.0],
            reference_discharge=[1.0, 1.0, 1.0, 1.0],
            bound_factor=2.0,
        )
        assert spec.upper[0] == pytest.approx(100.0)

    def test_restricting_to_a_subset_keeps_ascending_order(self):
        spec = build_discharge_state_spec(
            network=_network(),
            initial_discharge=[1.0, 2.0, 3.0, 4.0],
            reference_discharge=[1.0, 2.0, 3.0, 4.0],
            reach_ids=[2, 0],
        )
        assert spec.reach_ids == (0, 2)
        assert spec.positions == (0, 2)
        assert spec.par_names == ["sp_q_0000", "sp_q_0002"]

    def test_wrong_array_length_raises(self):
        with pytest.raises(ValueError, match="one value per reach"):
            build_discharge_state_spec(
                network=_network(),
                initial_discharge=[1.0, 2.0],
                reference_discharge=[1.0, 2.0, 3.0, 4.0],
            )

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError, match="No reach selected"):
            build_discharge_state_spec(
                network=_network(),
                initial_discharge=[1.0, 2.0, 3.0, 4.0],
                reference_discharge=[1.0, 2.0, 3.0, 4.0],
                reach_ids=[],
            )


class TestStateFloor:
    """A zero lower bound lets the ensemble update produce a subnormal value.

    PEST++ then refuses to run it at all:
        ParameterEnsemble:: add_runs() error: denormal values for realization 5 : SP_Q_0897
    It accepts exactly 0.0 but nothing in the subnormal band, and the routing
    recursion parks a dry reach right there. A positive lower bound makes bounds
    enforcement clip such a value back to a normal number.
    """

    def _spec(self, **kwargs):
        return build_discharge_state_spec(
            network=_network(),
            initial_discharge=[1.0, 0.0, 3.0, 4.0],
            reference_discharge=[10.0, 0.0, 30.0, 40.0],
            **kwargs,
        )

    def test_the_lower_bound_is_positive_and_normal(self):
        lower = self._spec().lower
        assert np.all(lower > 0.0)
        # Comfortably above the smallest normal double, so a clipped value is normal.
        assert np.all(lower > np.finfo(np.float64).tiny)
        assert all(np.isfinite(v) and v.item() != 0.0 for v in lower)

    def test_a_dry_reach_starts_on_the_floor_not_below_it(self):
        """parval1 below parlbnd is a control-file error."""
        spec = self._spec()
        assert np.all(spec.initial >= spec.lower)
        assert spec.initial[1] == pytest.approx(spec.lower[1])

    def test_the_floor_is_configurable(self):
        spec = self._spec(state_floor=1.0e-12)
        np.testing.assert_allclose(spec.lower, 1.0e-12)

    def test_bounds_never_collapse_even_with_a_large_floor(self):
        spec = self._spec(state_floor=1.0e-3)
        assert np.all(spec.upper > spec.lower)

    def test_a_non_positive_floor_is_rejected(self):
        with pytest.raises(ValueError, match="state_floor must be positive"):
            self._spec(state_floor=0.0)


class TestStateSpecIO:
    def test_json_roundtrip_preserves_the_interface_order(self, tmp_path):
        spec = build_discharge_state_spec(
            network=_network(),
            initial_discharge=[1.0, 2.0, 3.0, 4.0],
            reference_discharge=[1.0, 2.0, 3.0, 4.0],
            reach_ids=[0, 2, 3],
        )
        back = StateSpec.from_json(spec.to_json(tmp_path / "spec.json"))

        assert back.kinds == spec.kinds
        assert back.reach_ids == spec.reach_ids
        assert back.positions == spec.positions
        assert back.par_names == spec.par_names
        np.testing.assert_allclose(back.upper, spec.upper)

    def test_input_keys_match_the_reserved_prefix(self):
        spec = build_discharge_state_spec(
            network=_network(),
            initial_discharge=[1.0] * 4,
            reference_discharge=[1.0] * 4,
        )
        assert all(k.startswith(STATE_INPUT_PREFIX) for k in spec.input_keys)
        assert len(set(spec.input_keys)) == len(spec)

    def test_check_against_accepts_the_network_it_was_built_from(self):
        network = _network()
        spec = build_discharge_state_spec(
            network=network,
            initial_discharge=[1.0] * 4,
            reference_discharge=[1.0] * 4,
        )
        spec.check_against(network)

    def test_check_against_detects_a_reordered_network(self):
        """A different row order would apply every state to the wrong reach."""
        network = _network()
        spec = build_discharge_state_spec(
            network=network,
            initial_discharge=[1.0] * 4,
            reference_discharge=[1.0] * 4,
        )
        reordered = network.iloc[::-1].reset_index(drop=True)

        with pytest.raises(ValueError, match="row order"):
            spec.check_against(reordered)

    def test_check_against_detects_a_shorter_network(self):
        network = _network()
        spec = build_discharge_state_spec(
            network=network,
            initial_discharge=[1.0] * 4,
            reference_discharge=[1.0] * 4,
        )
        with pytest.raises(ValueError, match="only 2 reaches"):
            spec.check_against(network.iloc[:2].reset_index(drop=True))


class TestStateVectorProjection:
    def _spec(self, reach_ids=(0, 2)):
        return build_discharge_state_spec(
            network=_network(),
            initial_discharge=[1.0, 2.0, 3.0, 4.0],
            reference_discharge=[10.0] * 4,
            reach_ids=list(reach_ids),
        )

    def test_extract_selects_the_estimated_reaches_in_order(self):
        spec = self._spec()
        vector = extract_state_vector(_state([10.0, 20.0, 30.0, 40.0]), spec)
        np.testing.assert_allclose(vector, [10.0, 30.0])

    def test_insert_leaves_the_background_untouched_outside_the_space(self):
        spec = self._spec()
        state = _state([10.0, 20.0, 30.0, 40.0])
        insert_state_vector(np.array([1.5, 3.5]), spec, state)
        np.testing.assert_allclose(state.discharge, [1.5, 20.0, 3.5, 40.0])

    def test_insert_then_extract_is_the_identity(self):
        spec = self._spec()
        state = _state([10.0, 20.0, 30.0, 40.0])
        analysed = np.array([7.0, 9.0])
        np.testing.assert_allclose(extract_state_vector(insert_state_vector(analysed, spec, state), spec), analysed)

    def test_insert_does_not_write_through_to_the_caller_array(self):
        """The background array may be shared with the state file reader."""
        spec = self._spec()
        background = np.array([10.0, 20.0, 30.0, 40.0])
        state = SimpleNamespace(discharge=background)
        insert_state_vector(np.array([1.0, 2.0]), spec, state)
        np.testing.assert_allclose(background, [10.0, 20.0, 30.0, 40.0])

    def test_negative_values_are_clipped(self):
        spec = self._spec()
        state = _state([10.0, 20.0, 30.0, 40.0])
        insert_state_vector(np.array([-5.0, 3.0]), spec, state)
        np.testing.assert_allclose(state.discharge, [0.0, 20.0, 3.0, 40.0])

    def test_wrong_length_raises(self):
        spec = self._spec()
        with pytest.raises(ValueError, match="Expected 2 state value"):
            insert_state_vector(np.array([1.0, 2.0, 3.0]), spec, _state([1.0] * 4))
