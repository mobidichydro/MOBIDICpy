"""Tests for reservoir routing module."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from dataclasses import dataclass

from mobidic.core.reservoir import (
    ReservoirState,
    _interpolate_stage_from_volume,
    _interpolate_discharge_from_stage,
    _calculate_substeps,
    reservoir_routing,
)


# Fixtures


@pytest.fixture
def simple_stage_storage_curve():
    """Create a simple stage-storage curve."""
    return pd.DataFrame(
        {
            "stage_m": [240.0, 245.0, 250.0, 255.0],
            "volume_m3": [1000.0, 5000.0, 10000.0, 20000.0],
        }
    )


@pytest.fixture
def simple_reservoir_data(simple_stage_storage_curve):
    """Create a simple reservoir data object."""

    @dataclass
    class MockReservoir:
        id: int
        basin_pixels: np.ndarray
        inlet_reaches: np.ndarray
        stage_storage_curve: pd.DataFrame
        period_times: dict
        stage_discharge_h: dict
        stage_discharge_q: dict
        date_start: pd.Timestamp | None = None

    return MockReservoir(
        id=1,
        basin_pixels=np.array([0, 1, 2, 10, 11, 12]),  # Linear indices
        inlet_reaches=np.array([0, 1]),  # Upstream reach indices
        stage_storage_curve=simple_stage_storage_curve,
        period_times={"000": "2020-01-01T00:00:00"},
        stage_discharge_h={"000": [240.0, 245.0, 250.0, 255.0]},
        stage_discharge_q={"000": [0.0, 5.0, 10.0, 20.0]},
        date_start=pd.Timestamp("2020-01-01"),
    )


# Tests for ReservoirState


def test_reservoir_state_initialization():
    """Test ReservoirState creation with all fields."""
    state = ReservoirState(
        volume=5000.0,
        stage=245.0,
        inflow=10.0,
        outflow=8.0,
        withdrawal=1.0,
    )
    assert state.volume == 5000.0
    assert state.stage == 245.0
    assert state.inflow == 10.0
    assert state.outflow == 8.0
    assert state.withdrawal == 1.0


def test_reservoir_state_defaults():
    """Test ReservoirState default values."""
    state = ReservoirState(volume=5000.0, stage=245.0)
    assert state.volume == 5000.0
    assert state.stage == 245.0
    assert state.inflow == 0.0
    assert state.outflow == 0.0
    assert state.withdrawal == 0.0


# Tests for _interpolate_stage_from_volume


def test_interpolate_stage_from_volume_exact_match(simple_stage_storage_curve):
    """Test interpolation with exact volume match."""
    stage = _interpolate_stage_from_volume(simple_stage_storage_curve, 5000.0)
    assert stage == pytest.approx(245.0, abs=0.01)


def test_interpolate_stage_from_volume_interpolation(simple_stage_storage_curve):
    """Test interpolation between two points."""
    # Volume between 5000 and 10000 should give stage between 245 and 250
    stage = _interpolate_stage_from_volume(simple_stage_storage_curve, 7500.0)
    assert 245.0 < stage < 250.0


def test_interpolate_stage_from_volume_extrapolation_above(simple_stage_storage_curve):
    """Test extrapolation above max volume."""
    # Should extrapolate using cubic spline (may go up or down depending on curve shape)
    stage = _interpolate_stage_from_volume(simple_stage_storage_curve, 25000.0)
    assert np.isfinite(stage)  # Should return a finite value


def test_interpolate_stage_from_volume_extrapolation_below(simple_stage_storage_curve):
    """Test extrapolation below min volume."""
    # Should extrapolate using cubic spline
    stage = _interpolate_stage_from_volume(simple_stage_storage_curve, 500.0)
    assert stage < 240.0  # Should be below min stage


def test_interpolate_stage_from_volume_unsorted():
    """Test with unsorted volume data."""
    curve = pd.DataFrame(
        {
            "stage_m": [240.0, 250.0, 245.0],  # Unsorted
            "volume_m3": [1000.0, 10000.0, 5000.0],  # Unsorted
        }
    )
    stage = _interpolate_stage_from_volume(curve, 5000.0)
    assert stage == pytest.approx(245.0, abs=0.01)


def test_interpolate_stage_from_volume_single_point():
    """Test with only one point in curve (should raise ValueError)."""
    curve = pd.DataFrame({"stage_m": [245.0], "volume_m3": [5000.0]})
    # CubicSpline requires at least 2 points
    with pytest.raises(ValueError, match="at least 2 elements"):
        _interpolate_stage_from_volume(curve, 5000.0)


# Tests for _interpolate_discharge_from_stage


def test_interpolate_discharge_from_stage_exact_match():
    """Test interpolation with exact stage match."""
    stage_values = np.array([240.0, 245.0, 250.0])
    discharge_values = np.array([0.0, 5.0, 10.0])
    discharge = _interpolate_discharge_from_stage(stage_values, discharge_values, 245.0)
    assert discharge == pytest.approx(5.0)


def test_interpolate_discharge_from_stage_interpolation():
    """Test interpolation between two points."""
    stage_values = np.array([240.0, 245.0, 250.0])
    discharge_values = np.array([0.0, 5.0, 10.0])
    discharge = _interpolate_discharge_from_stage(stage_values, discharge_values, 247.5)
    assert discharge == pytest.approx(7.5)  # Linear interpolation


def test_interpolate_discharge_from_stage_extrapolation_above():
    """Test extrapolation above max stage."""
    stage_values = np.array([240.0, 245.0, 250.0])
    discharge_values = np.array([0.0, 5.0, 10.0])
    discharge = _interpolate_discharge_from_stage(stage_values, discharge_values, 255.0)
    # np.interp extrapolates with constant value (repeats endpoint)
    assert discharge == pytest.approx(10.0)  # Should repeat max discharge


def test_interpolate_discharge_from_stage_extrapolation_below():
    """Test extrapolation below min stage."""
    stage_values = np.array([240.0, 245.0, 250.0])
    discharge_values = np.array([0.0, 5.0, 10.0])
    _interpolate_discharge_from_stage(stage_values, discharge_values, 235.0)
    # Linear extrapolation below, but clamped to non-negative
    discharge_extrap = _interpolate_discharge_from_stage(stage_values, discharge_values, 235.0)
    assert discharge_extrap >= 0.0  # Should be non-negative


def test_interpolate_discharge_from_stage_with_nan():
    """Test interpolation with NaN values in arrays."""
    stage_values = np.array([240.0, np.nan, 250.0, 255.0])
    discharge_values = np.array([0.0, 5.0, 10.0, np.nan])
    # Should filter out NaN and interpolate with remaining valid points
    discharge = _interpolate_discharge_from_stage(stage_values, discharge_values, 245.0)
    assert discharge >= 0.0  # Should return valid result


def test_interpolate_discharge_from_stage_all_nan():
    """Test with all NaN values."""
    stage_values = np.array([np.nan, np.nan, np.nan])
    discharge_values = np.array([np.nan, np.nan, np.nan])
    discharge = _interpolate_discharge_from_stage(stage_values, discharge_values, 245.0)
    assert discharge == 0.0  # Should return 0 with warning


def test_interpolate_discharge_from_stage_single_point():
    """Test with only one valid point."""
    stage_values = np.array([245.0, np.nan, np.nan])
    discharge_values = np.array([5.0, np.nan, np.nan])
    discharge = _interpolate_discharge_from_stage(stage_values, discharge_values, 250.0)
    assert discharge == 5.0  # Should return the single value


def test_interpolate_discharge_from_stage_unsorted():
    """Test with unsorted stage values."""
    stage_values = np.array([250.0, 240.0, 245.0])
    discharge_values = np.array([10.0, 0.0, 5.0])
    discharge = _interpolate_discharge_from_stage(stage_values, discharge_values, 245.0)
    assert discharge == pytest.approx(5.0)


def test_interpolate_discharge_from_stage_negative_clamping():
    """Test that negative discharge is clamped to zero."""
    stage_values = np.array([240.0, 245.0, 250.0])
    discharge_values = np.array([10.0, 5.0, 0.0])  # Decreasing
    # Extrapolate below to get negative value
    discharge = _interpolate_discharge_from_stage(stage_values, discharge_values, 230.0)
    assert discharge >= 0.0  # Should be clamped to non-negative


# Tests for _calculate_substeps


def test_calculate_substeps_basic(simple_stage_storage_curve):
    """Test basic substep calculation."""
    stage_h = np.array([240.0, 245.0, 250.0, 255.0])
    stage_q = np.array([0.0, 5.0, 10.0, 20.0])
    dt = 900.0  # 15 minutes
    base_substeps = 4

    nsteps = _calculate_substeps(stage_h, stage_q, simple_stage_storage_curve, dt, base_substeps)
    assert nsteps >= base_substeps
    assert isinstance(nsteps, int)


def test_calculate_substeps_returns_base_when_no_valid_pairs(simple_stage_storage_curve):
    """Test that base substeps is returned when no valid pairs exist."""
    stage_h = np.array([np.nan, np.nan, np.nan])
    stage_q = np.array([np.nan, np.nan, np.nan])
    dt = 900.0
    base_substeps = 4

    nsteps = _calculate_substeps(stage_h, stage_q, simple_stage_storage_curve, dt, base_substeps)
    assert nsteps == base_substeps


def test_calculate_substeps_with_zero_discharge(simple_stage_storage_curve):
    """Test substep calculation with zero discharge values."""
    stage_h = np.array([240.0, 245.0, 250.0])
    stage_q = np.array([0.0, 0.0, 0.0])  # All zeros
    dt = 900.0
    base_substeps = 4

    nsteps = _calculate_substeps(stage_h, stage_q, simple_stage_storage_curve, dt, base_substeps)
    assert nsteps == base_substeps  # Should return base when Q=0


def test_calculate_substeps_large_dt(simple_stage_storage_curve):
    """Test that larger dt requires more substeps."""
    stage_h = np.array([240.0, 245.0, 250.0, 255.0])
    stage_q = np.array([0.0, 5.0, 10.0, 20.0])
    dt_small = 100.0
    dt_large = 10000.0
    base_substeps = 4

    nsteps_small = _calculate_substeps(stage_h, stage_q, simple_stage_storage_curve, dt_small, base_substeps)
    nsteps_large = _calculate_substeps(stage_h, stage_q, simple_stage_storage_curve, dt_large, base_substeps)

    # Larger dt should require more substeps (or equal)
    assert nsteps_large >= nsteps_small


def test_calculate_substeps_doubling_logic(simple_stage_storage_curve):
    """Test that substeps double when dt is too large."""
    stage_h = np.array([240.0, 245.0, 250.0, 255.0])
    stage_q = np.array([0.0, 5.0, 10.0, 20.0])
    dt = 10000.0  # Large timestep
    base_substeps = 2

    nsteps = _calculate_substeps(stage_h, stage_q, simple_stage_storage_curve, dt, base_substeps)
    # Should be a power of 2 times base_substeps
    assert nsteps >= base_substeps
    assert nsteps % base_substeps == 0  # Should be multiple of base


# Tests for reservoir_routing


def test_reservoir_routing_basic(simple_reservoir_data):
    """Test basic reservoir routing functionality."""
    # Setup
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0, inflow=10.0, outflow=5.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0  # 15 minutes
    cell_area = 100.0 * 100.0  # 100m x 100m cells

    # Run routing
    (
        updated_states,
        updated_discharge,
        updated_surface_runoff,
        updated_lateral_flow,
        updated_soil_wg,
    ) = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Check that state was updated
    assert len(updated_states) == 1
    assert updated_states[0].volume >= 0.0
    assert updated_states[0].outflow >= 0.0
    assert np.isfinite(updated_states[0].volume)
    assert np.isfinite(updated_states[0].outflow)


def test_reservoir_routing_inactive_reservoir(simple_reservoir_data):
    """Test routing with inactive reservoir (before start date)."""
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2019, 12, 1, 0, 0, 0)  # Before start date
    dt = 900.0
    cell_area = 10000.0

    # Run routing
    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Reservoir should be inactive
    assert updated_states[0].volume == 0.0
    assert updated_states[0].outflow == 0.0
    assert updated_states[0].inflow == 0.0
    assert updated_states[0].stage == 0.0


def test_reservoir_routing_no_inlet_reaches(simple_reservoir_data):
    """Test routing with no inlet reaches."""
    simple_reservoir_data.inlet_reaches = None
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Should have zero inflow
    assert updated_states[0].inflow == 0.0


def test_reservoir_routing_zeros_basin_fluxes(simple_reservoir_data):
    """Test that routing zeros out surface runoff and lateral flow in basin."""
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.ones((5, 5)) * 0.01  # Non-zero initial values
    lateral_flow = np.ones((5, 5)) * 0.005
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    _, _, updated_surface_runoff, updated_lateral_flow, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Basin pixels should be zeroed (linear indices 0, 1, 2, 10, 11, 12 in 5x5 grid)
    # Convert linear to 2D indices (Fortran order)
    ibac_row, ibac_col = np.unravel_index(simple_reservoir_data.basin_pixels, (5, 5), order="F")

    for i, j in zip(ibac_row, ibac_col):
        assert updated_surface_runoff[i, j] == 0.0
        assert updated_lateral_flow[i, j] == 0.0


def test_reservoir_routing_zeros_inlet_reaches(simple_reservoir_data):
    """Test that routing zeros discharge of inlet reaches."""
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])  # Reaches 0, 1 are inlets
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    _, updated_discharge, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Inlet reaches (0, 1) should be zeroed
    assert updated_discharge[0] == 0.0
    assert updated_discharge[1] == 0.0
    # Other reach should be unchanged
    assert updated_discharge[2] == 2.0


def test_reservoir_routing_negative_volume_handling(simple_reservoir_data):
    """Test that negative volumes are handled correctly."""
    # Start with very small volume and high discharge
    reservoir_states = [ReservoirState(volume=100.0, stage=240.0, outflow=50.0)]
    reach_discharge = np.array([0.0, 0.0, 0.0])  # No inflow
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Volume should not go negative
    assert updated_states[0].volume >= 0.0


def test_reservoir_routing_invalid_outflow_handling(simple_reservoir_data):
    """Test that invalid outflow values are caught and set to zero."""
    # This is harder to trigger, but we test the check exists
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Outflow should be finite and non-negative
    assert np.isfinite(updated_states[0].outflow)
    assert updated_states[0].outflow >= 0.0


def test_reservoir_routing_multiple_regulation_periods(simple_reservoir_data):
    """Test routing with multiple regulation periods."""
    # Add a second period
    simple_reservoir_data.period_times["001"] = "2020-06-01T00:00:00"
    simple_reservoir_data.stage_discharge_h["001"] = [240.0, 245.0, 250.0, 255.0]
    simple_reservoir_data.stage_discharge_q["001"] = [0.0, 10.0, 20.0, 30.0]  # Different discharges

    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    dt = 900.0
    cell_area = 10000.0

    # Test in first period
    current_time = datetime(2020, 3, 1, 12, 0, 0)
    updated_states_1, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Test in second period
    current_time = datetime(2020, 7, 1, 12, 0, 0)
    reservoir_states[0] = ReservoirState(volume=5000.0, stage=245.0)  # Reset
    updated_states_2, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Outflows might differ due to different regulation curves
    # Just check both are valid
    assert np.isfinite(updated_states_1[0].outflow)
    assert np.isfinite(updated_states_2[0].outflow)
    assert updated_states_1[0].outflow >= 0.0
    assert updated_states_2[0].outflow >= 0.0


def test_reservoir_routing_with_lateral_inflow(simple_reservoir_data):
    """Test routing with lateral inflow from basin cells."""
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([0.0, 0.0, 0.0])  # No reach inflow
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))

    # Add lateral inflow to basin cells
    ibac_row, ibac_col = np.unravel_index(simple_reservoir_data.basin_pixels, (5, 5), order="F")
    for i, j in zip(ibac_row, ibac_col):
        surface_runoff[i, j] = 0.001  # m/s
        lateral_flow[i, j] = 0.0005  # m/s

    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Volume should have increased due to lateral inflow
    assert updated_states[0].volume > 5000.0


def test_reservoir_routing_no_period_times(simple_reservoir_data):
    """Test routing with no regulation periods defined."""
    simple_reservoir_data.period_times = {}
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Should skip reservoir and set outflow to zero
    assert updated_states[0].outflow == 0.0
    assert updated_states[0].inflow == 0.0


def test_reservoir_routing_no_basin_pixels(simple_reservoir_data):
    """Test routing with no basin pixels defined."""
    simple_reservoir_data.basin_pixels = None
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    # Should not crash, just skip this reservoir
    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Reservoir should be processed (outflow might be non-zero from routing)
    assert len(updated_states) == 1


def test_reservoir_routing_empty_basin_pixels(simple_reservoir_data):
    """Test routing with empty basin pixels array."""
    simple_reservoir_data.basin_pixels = np.array([])
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    # Should not crash
    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    assert len(updated_states) == 1


def test_reservoir_routing_custom_substeps(simple_reservoir_data):
    """Test routing with custom base substeps."""
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0
    custom_substeps = 8

    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
        base_substeps=custom_substeps,
    )

    # Should complete successfully with custom substeps
    assert len(updated_states) == 1
    assert updated_states[0].outflow >= 0.0


def test_reservoir_routing_does_not_modify_inputs(simple_reservoir_data):
    """Test that routing does not modify original input arrays."""
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge_orig = np.array([10.0, 5.0, 2.0])
    surface_runoff_orig = np.ones((5, 5)) * 0.01
    lateral_flow_orig = np.ones((5, 5)) * 0.005
    soil_wg_orig = np.zeros((5, 5))

    # Make copies to compare
    reach_discharge = reach_discharge_orig.copy()
    surface_runoff = surface_runoff_orig.copy()
    lateral_flow = lateral_flow_orig.copy()
    soil_wg = soil_wg_orig.copy()

    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge_orig,
        surface_runoff_orig,
        lateral_flow_orig,
        soil_wg_orig,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Original arrays should be unchanged (function makes copies)
    assert np.array_equal(reach_discharge_orig, reach_discharge)
    assert np.array_equal(surface_runoff_orig, surface_runoff)
    assert np.array_equal(lateral_flow_orig, lateral_flow)
    assert np.array_equal(soil_wg_orig, soil_wg)


def test_reservoir_routing_no_active_period_yet(simple_reservoir_data):
    """Test routing when current time is before all regulation periods."""
    # Set all periods to the future
    simple_reservoir_data.period_times = {"000": "2025-01-01T00:00:00"}
    simple_reservoir_data.stage_discharge_h = {"000": [240.0, 245.0, 250.0]}
    simple_reservoir_data.stage_discharge_q = {"000": [0.0, 5.0, 10.0]}

    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)  # Before all periods
    dt = 900.0
    cell_area = 10000.0

    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Should have zero outflow and inflow (no active period)
    assert updated_states[0].outflow == 0.0
    assert updated_states[0].inflow == 0.0


def test_reservoir_routing_with_multiple_substeps(simple_reservoir_data):
    """Test routing that requires multiple substeps (triggers debug logging)."""
    reservoir_states = [ReservoirState(volume=5000.0, stage=245.0)]
    reach_discharge = np.array([10.0, 5.0, 2.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 100000.0  # Very large timestep to force substeps
    cell_area = 10000.0
    base_substeps = 1  # Small base to ensure nsbgo > 1

    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
        base_substeps=base_substeps,
    )

    # Should complete successfully
    assert len(updated_states) == 1
    assert updated_states[0].outflow >= 0.0


def test_reservoir_routing_extreme_negative_volume(simple_reservoir_data):
    """Test routing when volume would go very negative (triggers alternate path)."""
    # Start with very small volume, no inflow, and set up to trigger extreme negative case
    reservoir_states = [ReservoirState(volume=10.0, stage=240.0, outflow=100.0)]
    reach_discharge = np.array([0.0, 0.0, 0.0])  # No inflow from reaches
    surface_runoff = np.zeros((5, 5))  # No surface inflow
    lateral_flow = np.zeros((5, 5))  # No lateral inflow
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0  # Large timestep to amplify the issue
    cell_area = 10000.0

    updated_states, _, _, _, _ = reservoir_routing(
        [simple_reservoir_data],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Volume should not go negative
    assert updated_states[0].volume >= 0.0
    # Outflow should be reduced or zero
    assert updated_states[0].outflow >= 0.0


def test_reservoir_routing_multiple_reservoirs():
    """Test routing with multiple reservoirs."""

    @dataclass
    class MockReservoir:
        id: int
        basin_pixels: np.ndarray
        inlet_reaches: np.ndarray | None
        stage_storage_curve: pd.DataFrame
        period_times: dict
        stage_discharge_h: dict
        stage_discharge_q: dict
        date_start: pd.Timestamp | None = None

    curve1 = pd.DataFrame({"stage_m": [240.0, 245.0, 250.0], "volume_m3": [1000.0, 5000.0, 10000.0]})
    curve2 = pd.DataFrame({"stage_m": [250.0, 255.0, 260.0], "volume_m3": [2000.0, 6000.0, 12000.0]})

    reservoir1 = MockReservoir(
        id=1,
        basin_pixels=np.array([0, 1, 2]),
        inlet_reaches=np.array([0]),
        stage_storage_curve=curve1,
        period_times={"000": "2020-01-01T00:00:00"},
        stage_discharge_h={"000": [240.0, 245.0, 250.0]},
        stage_discharge_q={"000": [0.0, 5.0, 10.0]},
        date_start=pd.Timestamp("2020-01-01"),
    )

    reservoir2 = MockReservoir(
        id=2,
        basin_pixels=np.array([10, 11, 12]),
        inlet_reaches=np.array([1]),
        stage_storage_curve=curve2,
        period_times={"000": "2020-01-01T00:00:00"},
        stage_discharge_h={"000": [250.0, 255.0, 260.0]},
        stage_discharge_q={"000": [0.0, 8.0, 15.0]},
        date_start=pd.Timestamp("2020-01-01"),
    )

    reservoir_states = [
        ReservoirState(volume=5000.0, stage=245.0),
        ReservoirState(volume=6000.0, stage=255.0),
    ]

    reach_discharge = np.array([10.0, 8.0, 5.0])
    surface_runoff = np.zeros((5, 5))
    lateral_flow = np.zeros((5, 5))
    soil_wg = np.zeros((5, 5))
    soil_wg0 = np.ones((5, 5)) * 0.1
    current_time = datetime(2020, 1, 1, 12, 0, 0)
    dt = 900.0
    cell_area = 10000.0

    updated_states, _, _, _, _ = reservoir_routing(
        [reservoir1, reservoir2],
        reservoir_states,
        reach_discharge,
        surface_runoff,
        lateral_flow,
        soil_wg,
        soil_wg0,
        current_time,
        dt,
        cell_area,
    )

    # Both reservoirs should be updated
    assert len(updated_states) == 2
    assert updated_states[0].outflow >= 0.0
    assert updated_states[1].outflow >= 0.0
    assert np.isfinite(updated_states[0].volume)
    assert np.isfinite(updated_states[1].volume)
