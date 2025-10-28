"""Tests for routing module."""

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import LineString
from mobidic.core.routing import hillslope_routing, linear_channel_routing


class TestHillslopeRouting:
    """Tests for hillslope_routing function."""

    def test_simple_accumulation_grass(self):
        """Test one-step routing with Grass notation."""
        # Create 3x3 grid where all cells drain to center
        lateral_flow = np.ones((3, 3)) * 0.1

        # All cells flow to center (1,1)
        # Grass notation: 1=bottom-left, 2=bottom, 3=bottom-right, 4=right,
        #                 5=top-right, 6=top, 7=top-left, 8=left
        # Note: Grass notation uses different numbering than Arc
        # Direction 2 = up (north), 6 = down (south), 4 = right (east), 8 = left (west)
        flow_direction = np.array(
            [
                [5, 6, 7],  # top row: (0,0)->down-right, (0,1)->down, (0,2)->down-left
                [4, 0, 8],  # middle: (1,0)->right, (1,1)=outlet, (1,2)->left
                [3, 2, 1],  # bottom: (2,0)->up-right, (2,1)->up, (2,2)->up-left
            ]
        )

        upstream_contribution = hillslope_routing(lateral_flow, flow_direction, "Grass")

        # Center cell receives flow from all 8 neighbors (one-step)
        # Does NOT include center's own flow
        assert np.isclose(upstream_contribution[1, 1], 0.8)  # 8 neighbors × 0.1

        # Edge cells receive no upstream flow (they are headwaters)
        assert np.isclose(upstream_contribution[0, 0], 0.0)
        assert np.isclose(upstream_contribution[2, 2], 0.0)

    def test_simple_accumulation_arc(self):
        """Test one-step routing with Arc notation."""
        lateral_flow = np.ones((3, 3)) * 0.1

        # All cells flow to center (1,1) using Arc notation
        # Arc notation: 1=down, 2=down-right, 4=right, 8=up-right,
        #               16=up, 32=up-left, 64=left, 128=down-left
        flow_direction = np.array(
            [
                [2, 1, 128],  # top row: (0,0)->down-right, (0,1)->down, (0,2)->down-left
                [4, 0, 64],  # middle: (1,0)->right, (1,1)=outlet, (1,2)->left
                [8, 16, 32],  # bottom: (2,0)->up-right, (2,1)->up, (2,2)->up-left
            ]
        )

        upstream_contribution = hillslope_routing(lateral_flow, flow_direction, "Arc")

        # Center cell receives flow from all 8 neighbors (one-step)
        # Does NOT include center's own flow
        assert np.isclose(upstream_contribution[1, 1], 0.8)  # 8 neighbors × 0.1

    def test_linear_cascade_grass(self):
        """Test one-step routing in linear cascade."""
        # 1x5 grid, flow from left to right
        lateral_flow = np.ones((1, 5)) * 1.0
        flow_direction = np.array([[4, 4, 4, 4, 0]])  # All flow right, last is outlet

        upstream_contribution = hillslope_routing(lateral_flow, flow_direction, "Grass")

        # One-step routing: each cell receives flow ONLY from immediate upstream neighbor
        assert np.isclose(upstream_contribution[0, 0], 0.0)  # First cell: no upstream
        assert np.isclose(upstream_contribution[0, 1], 1.0)  # Second: from first cell
        assert np.isclose(upstream_contribution[0, 2], 1.0)  # Third: from second cell
        assert np.isclose(upstream_contribution[0, 3], 1.0)  # Fourth: from third cell
        assert np.isclose(upstream_contribution[0, 4], 1.0)  # Outlet: from fourth cell

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        lateral_flow = np.array(
            [
                [1.0, np.nan, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        flow_direction = np.array(
            [
                [6, np.nan, 6],
                [6, 6, 6],
                [0, 2, 2],
            ]
        )

        upstream_contribution = hillslope_routing(lateral_flow, flow_direction, "Grass")

        # NaN cells are skipped in routing, result is 0.0 at that position
        assert np.isclose(upstream_contribution[0, 1], 0.0)

        # Other cells should receive upstream flow normally
        assert np.isfinite(upstream_contribution[2, 0])
        # Cell (1,0) receives flow from cell (2,0) which flows up (direction 2)
        assert upstream_contribution[1, 0] > 0

    def test_multiple_outlets(self):
        """Test grid with multiple outlet cells."""
        lateral_flow = np.ones((2, 3)) * 1.0

        # Create a simple 2x3 grid with three outlets
        # (1,0) flows to (0,0), (1,1) flows to (0,1), (1,2) flows to (0,2)
        # Top row cells are all outlets (no outflow)
        # Grass direction 2 = up (north), so bottom row flows to top row
        flow_direction = np.array(
            [
                [0, 0, 0],  # Three outlets (no outflow)
                [2, 2, 2],  # Bottom row flows up (direction 2 = north/up)
            ]
        )

        upstream_contribution = hillslope_routing(lateral_flow, flow_direction, "Grass")

        # Each top cell receives flow from cell below (one-step)
        # Does NOT include cell's own flow
        assert np.isclose(upstream_contribution[0, 0], 1.0)  # From (1,0)
        assert np.isclose(upstream_contribution[0, 1], 1.0)  # From (1,1)
        assert np.isclose(upstream_contribution[0, 2], 1.0)  # From (1,2)

        # Bottom cells have no upstream flow
        assert np.isclose(upstream_contribution[1, 0], 0.0)
        assert np.isclose(upstream_contribution[1, 1], 0.0)
        assert np.isclose(upstream_contribution[1, 2], 0.0)

    def test_invalid_flow_dir_type(self):
        """Test that invalid flow direction type raises error."""
        lateral_flow = np.ones((3, 3))
        flow_direction = np.ones((3, 3))

        with pytest.raises(ValueError, match="flow_dir_type must be"):
            hillslope_routing(lateral_flow, flow_direction, "Invalid")

    def test_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        lateral_flow = np.ones((3, 3))
        flow_direction = np.ones((2, 2))

        with pytest.raises(ValueError, match="must match"):
            hillslope_routing(lateral_flow, flow_direction, "Grass")


class TestLinearChannelRouting:
    """Tests for linear_channel_routing function."""

    def test_single_reach_no_upstream(self):
        """Test routing for single reach with no upstream contributions."""
        # Single reach with K=3600s (1 hour)
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0],
                "upstream_1": [np.nan],
                "upstream_2": [np.nan],
                "downstream": [np.nan],
                "calc_order": [0],
                "lag_time_s": [3600.0],
                "geometry": [LineString([(0, 0), (1, 1)])],
            }
        )

        Q_init = np.array([10.0])  # Initial discharge 10 m³/s
        qL = np.array([2.0])  # Lateral inflow 2 m³/s
        dt = 900.0  # 15 minutes

        Q_final, state = linear_channel_routing(network, Q_init, qL, dt)

        # Check coefficients
        expected_C3 = np.exp(-dt / 3600.0)
        expected_C4 = 1 - expected_C3
        assert np.isclose(state["C3"][0], expected_C3)
        assert np.isclose(state["C4"][0], expected_C4)

        # Check discharge: Q_final = C3 * Q_init + C4 * qL
        expected_Q = expected_C3 * 10.0 + expected_C4 * 2.0
        assert np.isclose(Q_final[0], expected_Q)

    def test_two_reach_cascade(self):
        """Test routing for two reaches in cascade."""
        # Reach 0 flows into reach 1
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0, 1],
                "upstream_1": [np.nan, 0],
                "upstream_2": [np.nan, np.nan],
                "downstream": [1, np.nan],
                "calc_order": [0, 1],
                "lag_time_s": [3600.0, 7200.0],  # K = 1h and 2h
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            }
        )

        Q_init = np.array([10.0, 5.0])
        qL = np.array([2.0, 1.0])
        dt = 900.0

        Q_final, state = linear_channel_routing(network, Q_init, qL, dt)

        # Reach 0 should be routed first
        C3_0 = state["C3"][0]
        C4_0 = state["C4"][0]
        expected_Q0 = C3_0 * Q_init[0] + C4_0 * qL[0]
        assert np.isclose(Q_final[0], expected_Q0)

        # Reach 1 should receive contribution from reach 0
        # The contribution is computed as mean integral of exponential decay
        assert state["qL_total"][1] > qL[1]  # Should be greater than lateral alone

        # Output discharge should be positive
        assert Q_final[0] > 0
        assert Q_final[1] > 0

    def test_junction_two_upstream(self):
        """Test routing at junction with two upstream reaches."""
        # Reaches 0 and 1 flow into reach 2
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0, 1, 2],
                "upstream_1": [np.nan, np.nan, 0],
                "upstream_2": [np.nan, np.nan, 1],
                "downstream": [2, 2, np.nan],
                "calc_order": [0, 0, 1],
                "lag_time_s": [3600.0, 3600.0, 7200.0],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(0, 2), (1, 1)]),
                    LineString([(1, 1), (2, 1)]),
                ],
            }
        )

        Q_init = np.array([10.0, 8.0, 5.0])
        qL = np.array([1.0, 1.0, 0.5])
        dt = 900.0

        Q_final, state = linear_channel_routing(network, Q_init, qL, dt)

        # Reach 2 should receive contributions from both reach 0 and reach 1
        # qL_total[2] should include lateral inflow + integrated upstream flows
        assert state["qL_total"][2] > qL[2]
        assert Q_final[2] > 0

    def test_zero_lateral_inflow(self):
        """Test routing with zero lateral inflow."""
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0],
                "upstream_1": [np.nan],
                "upstream_2": [np.nan],
                "downstream": [np.nan],
                "calc_order": [0],
                "lag_time_s": [3600.0],
                "geometry": [LineString([(0, 0), (1, 1)])],
            }
        )

        Q_init = np.array([10.0])
        qL = np.array([0.0])
        dt = 900.0

        Q_final, state = linear_channel_routing(network, Q_init, qL, dt)

        # With no lateral inflow, discharge should decay exponentially
        expected_Q = Q_init[0] * np.exp(-dt / 3600.0)
        assert np.isclose(Q_final[0], expected_Q)

    def test_very_large_lag_time(self):
        """Test routing with very large lag time (slow recession)."""
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0],
                "upstream_1": [np.nan],
                "upstream_2": [np.nan],
                "downstream": [np.nan],
                "calc_order": [0],
                "lag_time_s": [1e10],  # Very large K
                "geometry": [LineString([(0, 0), (1, 1)])],
            }
        )

        Q_init = np.array([10.0])
        qL = np.array([2.0])
        dt = 900.0

        Q_final, state = linear_channel_routing(network, Q_init, qL, dt)

        # With very large K, C3 ≈ 1 and C4 ≈ 0
        # So Q_final ≈ Q_init
        assert np.isclose(Q_final[0], Q_init[0], rtol=0.01)

    def test_very_small_lag_time(self):
        """Test routing with very small lag time (fast recession)."""
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0],
                "upstream_1": [np.nan],
                "upstream_2": [np.nan],
                "downstream": [np.nan],
                "calc_order": [0],
                "lag_time_s": [1.0],  # Very small K
                "geometry": [LineString([(0, 0), (1, 1)])],
            }
        )

        Q_init = np.array([10.0])
        qL = np.array([2.0])
        dt = 900.0

        Q_final, state = linear_channel_routing(network, Q_init, qL, dt)

        # With very small K, C3 ≈ 0 and C4 ≈ 1
        # So Q_final ≈ qL
        assert np.isclose(Q_final[0], qL[0], rtol=0.1)

    def test_missing_required_columns(self):
        """Test that missing required columns raises error."""
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0],
                "calc_order": [0],
                "geometry": [LineString([(0, 0), (1, 1)])],
            }
        )

        Q_init = np.array([10.0])
        qL = np.array([2.0])
        dt = 900.0

        with pytest.raises(ValueError, match="missing required columns"):
            linear_channel_routing(network, Q_init, qL, dt)

    def test_missing_lag_time_column(self):
        """Test that missing lag_time_s column raises error."""
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0],
                "upstream_1": [np.nan],
                "upstream_2": [np.nan],
                "downstream": [np.nan],
                "calc_order": [0],
                "geometry": [LineString([(0, 0), (1, 1)])],
            }
        )

        Q_init = np.array([10.0])
        qL = np.array([2.0])
        dt = 900.0

        with pytest.raises(ValueError, match="missing required columns"):
            linear_channel_routing(network, Q_init, qL, dt)

    def test_array_length_mismatch(self):
        """Test that array length mismatch raises error."""
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0, 1],
                "upstream_1": [np.nan, 0],
                "upstream_2": [np.nan, np.nan],
                "downstream": [1, np.nan],
                "calc_order": [0, 1],
                "lag_time_s": [3600.0, 7200.0],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            }
        )

        Q_init = np.array([10.0])  # Wrong length (should be 2)
        qL = np.array([2.0, 1.0])
        dt = 900.0

        with pytest.raises(ValueError, match="must match"):
            linear_channel_routing(network, Q_init, qL, dt)

    def test_negative_dt(self):
        """Test that negative time step raises error."""
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0],
                "upstream_1": [np.nan],
                "upstream_2": [np.nan],
                "downstream": [np.nan],
                "calc_order": [0],
                "lag_time_s": [3600.0],
                "geometry": [LineString([(0, 0), (1, 1)])],
            }
        )

        Q_init = np.array([10.0])
        qL = np.array([2.0])
        dt = -900.0

        with pytest.raises(ValueError, match="must be positive"):
            linear_channel_routing(network, Q_init, qL, dt)

    def test_mass_balance(self):
        """Test that mass is conserved in routing."""
        # Two reaches in cascade
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0, 1],
                "upstream_1": [np.nan, 0],
                "upstream_2": [np.nan, np.nan],
                "downstream": [1, np.nan],
                "calc_order": [0, 1],
                "lag_time_s": [3600.0, 3600.0],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            }
        )

        Q_init = np.array([0.0, 0.0])  # Start with zero discharge
        qL = np.array([10.0, 5.0])  # Only lateral inflows
        dt = 900.0

        Q_final, state = linear_channel_routing(network, Q_init, qL, dt)

        # Total outflow should not exceed total inflow
        # (some water is stored in reaches)
        total_inflow = np.sum(qL) * dt
        total_outflow = Q_final[1] * dt  # Only outlet discharge counts

        assert total_outflow <= total_inflow

    def test_routing_order_matters(self):
        """Test that routing follows calc_order correctly."""
        # Create network where reach 1 (upstream) has higher calc_order than reach 0
        # This should still work because we sort by calc_order internally
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0, 1],
                "upstream_1": [1, np.nan],  # 0 receives from 1
                "upstream_2": [np.nan, np.nan],
                "downstream": [np.nan, 0],
                "calc_order": [1, 0],  # 1 is processed before 0
                "lag_time_s": [3600.0, 3600.0],
                "geometry": [
                    LineString([(1, 1), (2, 2)]),
                    LineString([(0, 0), (1, 1)]),
                ],
            }
        )

        Q_init = np.array([5.0, 10.0])
        qL = np.array([1.0, 2.0])
        dt = 900.0

        Q_final, state = linear_channel_routing(network, Q_init, qL, dt)

        # Should process reach 1 first, then reach 0
        # Reach 0 should receive contribution from reach 1
        assert Q_final[0] > 0
        assert Q_final[1] > 0
