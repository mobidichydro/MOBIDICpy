"""Tests for spatial interpolation functions."""

import numpy as np
from mobidic.core.interpolation import (
    compute_idw_weights,
    precipitation_interpolation,
    station_interpolation,
)


class TestStationInterpolation:
    """Tests for station_interpolation function with vectorized weights."""

    def test_with_precomputed_weights_single_station(self):
        """Test interpolation with pre-computed weights and single station."""
        # Create simple 10x10 grid
        dtm = np.ones((10, 10)) * 100.0  # Flat elevation at 100m

        # Single station in center
        station_x = np.array([500.0])
        station_y = np.array([500.0])
        station_elevation = np.array([100.0])
        station_values = np.array([20.0])  # Temperature

        # Pre-compute weights
        weights = compute_idw_weights(
            station_x=station_x,
            station_y=station_y,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            power=2.0,
        )

        # Interpolate
        result = station_interpolation(
            station_x=station_x,
            station_y=station_y,
            station_elevation=station_elevation,
            station_values=station_values,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            weights_matrix=weights,
            apply_elevation_correction=False,
            power=2.0,
        )

        # All cells should have approximately the same value (no elevation correction)
        assert result.shape == dtm.shape
        assert np.nanmin(result) > 19.0
        assert np.nanmax(result) < 21.0

    def test_with_precomputed_weights_multiple_stations(self):
        """Test interpolation with pre-computed weights and multiple stations."""
        # Create 20x20 grid
        dtm = np.ones((20, 20)) * 200.0

        # Three stations in different locations
        station_x = np.array([500.0, 1500.0, 1000.0])
        station_y = np.array([500.0, 1500.0, 1000.0])
        station_elevation = np.array([200.0, 200.0, 200.0])
        station_values = np.array([10.0, 20.0, 15.0])

        # Pre-compute weights
        weights = compute_idw_weights(
            station_x=station_x,
            station_y=station_y,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            power=2.0,
        )

        # Interpolate with pre-computed weights (vectorized path)
        result = station_interpolation(
            station_x=station_x,
            station_y=station_y,
            station_elevation=station_elevation,
            station_values=station_values,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            weights_matrix=weights,
            apply_elevation_correction=False,
            power=2.0,
        )

        # Check result properties
        assert result.shape == dtm.shape
        assert np.all(np.isfinite(result))
        # Values should be within range of station values
        assert np.nanmin(result) >= 10.0 - 1.0  # Allow small tolerance
        assert np.nanmax(result) <= 20.0 + 1.0

    def test_vectorized_vs_onthefly_consistency(self):
        """Test that pre-computed weights (vectorized) matches on-the-fly computation."""
        # Create test grid
        dtm = np.random.rand(30, 30) * 500.0 + 100.0

        # Multiple stations
        np.random.seed(42)
        station_x = np.random.rand(10) * 3000.0
        station_y = np.random.rand(10) * 3000.0
        station_elevation = np.random.rand(10) * 500.0 + 100.0
        station_values = np.random.rand(10) * 20.0 + 10.0

        # Pre-compute weights
        weights = compute_idw_weights(
            station_x=station_x,
            station_y=station_y,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            power=2.0,
        )

        # Interpolate with pre-computed weights (vectorized path)
        result_precomputed = station_interpolation(
            station_x=station_x,
            station_y=station_y,
            station_elevation=station_elevation,
            station_values=station_values,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            weights_matrix=weights,
            apply_elevation_correction=False,
            power=2.0,
        )

        # Interpolate without pre-computed weights (on-the-fly path)
        result_onthefly = station_interpolation(
            station_x=station_x,
            station_y=station_y,
            station_elevation=station_elevation,
            station_values=station_values,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            weights_matrix=None,
            apply_elevation_correction=False,
            power=2.0,
        )

        # Results should match within numerical precision
        np.testing.assert_allclose(
            result_precomputed,
            result_onthefly,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Vectorized and on-the-fly paths should produce identical results",
        )

    def test_with_nan_stations(self):
        """Test interpolation with some NaN station values (tests k_ok filtering)."""
        # Create grid
        dtm = np.ones((15, 15)) * 300.0

        # Five stations, two with NaN values
        station_x = np.array([500.0, 1000.0, 1500.0, 500.0, 1000.0])
        station_y = np.array([500.0, 1000.0, 1500.0, 1500.0, 500.0])
        station_elevation = np.array([300.0, 300.0, 300.0, 300.0, 300.0])
        station_values = np.array([10.0, np.nan, 15.0, np.nan, 12.0])  # 2 NaN values

        # Pre-compute weights for all stations
        weights = compute_idw_weights(
            station_x=station_x,
            station_y=station_y,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            power=2.0,
        )

        # Interpolate (should only use the 3 valid stations)
        result = station_interpolation(
            station_x=station_x,
            station_y=station_y,
            station_elevation=station_elevation,
            station_values=station_values,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            weights_matrix=weights,
            apply_elevation_correction=False,
            power=2.0,
        )

        # Check that interpolation succeeded with valid stations
        assert result.shape == dtm.shape
        assert np.all(np.isfinite(result))
        # Values should be within range of valid station values (10, 15, 12)
        assert np.nanmin(result) >= 10.0 - 1.0
        assert np.nanmax(result) <= 15.0 + 1.0


class TestPrecipitationInterpolation:
    """Tests for precipitation_interpolation function."""

    def test_single_station(self):
        """Test with single station (constant value)."""
        dtm = np.ones((10, 10)) * 100.0
        station_x = np.array([500.0])
        station_y = np.array([500.0])
        station_values = np.array([10.0])

        result = precipitation_interpolation(
            station_x=station_x,
            station_y=station_y,
            station_values=station_values,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
        )

        assert result.shape == dtm.shape
        assert np.allclose(result, 10.0)

    def test_two_stations(self):
        """Test with two stations (distance comparison)."""
        dtm = np.ones((10, 10)) * 100.0
        station_x = np.array([300.0, 700.0])
        station_y = np.array([500.0, 500.0])
        station_values = np.array([5.0, 15.0])

        result = precipitation_interpolation(
            station_x=station_x,
            station_y=station_y,
            station_values=station_values,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
        )

        assert result.shape == dtm.shape
        # Left side closer to first station, right side closer to second
        assert result[5, 0] == 5.0  # Left edge
        assert result[5, 9] == 15.0  # Right edge


class TestComputeIDWWeights:
    """Tests for compute_idw_weights function."""

    def test_weights_shape(self):
        """Test that output weights have correct shape."""
        dtm = np.ones((20, 20)) * 100.0
        station_x = np.array([500.0, 1500.0])
        station_y = np.array([500.0, 1500.0])

        weights = compute_idw_weights(
            station_x=station_x,
            station_y=station_y,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            power=2.0,
        )

        assert weights.shape == (20, 20, 2)
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)  # All weights should be positive

    def test_weights_power_parameter(self):
        """Test that power parameter affects weights."""
        dtm = np.ones((10, 10)) * 100.0
        station_x = np.array([500.0])
        station_y = np.array([500.0])

        weights_p2 = compute_idw_weights(
            station_x=station_x,
            station_y=station_y,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            power=2.0,
        )

        weights_p3 = compute_idw_weights(
            station_x=station_x,
            station_y=station_y,
            dtm=dtm,
            xllcorner=0.0,
            yllcorner=0.0,
            resolution=100.0,
            power=3.0,
        )

        # Higher power should create steeper weight gradients
        assert not np.allclose(weights_p2, weights_p3)
