"""Tests for grid operations module."""

import numpy as np
import pytest
from mobidic.preprocessing.grid_operations import (
    decimate_flow_direction,
    decimate_raster,
)


class TestDecimateRaster:
    """Tests for decimate_raster function."""

    def test_basic_decimation(self):
        """Test basic raster decimation with factor 2.
        Tested vs MATLAB version (degrad_var.m)
        """
        # Create 4x4 array
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=float)

        # Decimate with factor 2
        decimated = decimate_raster(data, factor=2)

        # Expected: 2x2 array with means of 2x2 blocks
        expected = np.array([[3.5, 5.5], [11.5, 13.5]])

        np.testing.assert_allclose(decimated, expected)

    def test_decimation_with_nan(self):
        """Test decimation with NaN values.
        Tested vs MATLAB version (degrad_var.m)
        """
        # Create 4x4 array with some NaN
        data = np.array([[1, 2, 3, 4], [5, np.nan, 7, 8], [9, 10, np.nan, 12], [13, 14, 15, 16]], dtype=float)

        # Degrade with factor 2
        decimated = decimate_raster(data, factor=2)

        # Expected: means excluding NaN
        # Block [0:2, 0:2]: mean(1, 2, 5) = 2.666...
        # Block [0:2, 2:4]: mean(3, 4, 7, 8) = 5.5
        # Block [2:4, 0:2]: mean(9, 10, 13, 14) = 11.5
        # Block [2:4, 2:4]: mean(12, 15, 16) = 14.333...
        expected = np.array([[2.666666667, 5.5], [11.5, 14.333333333]])

        np.testing.assert_allclose(decimated, expected)

    def test_decimation_insufficient_valid_cells(self):
        """Test decimation when a block has too few valid cells."""
        # Create 4x4 array where one block has only 1 valid cell
        data = np.array(
            [[1, 2, 3, 4], [np.nan, np.nan, 7, 8], [np.nan, np.nan, 11, 12], [np.nan, np.nan, 15, 16]], dtype=float
        )

        # Degrade with factor 2, min_valid_fraction=0.5 (needs 2 cells out of 4)
        decimated = decimate_raster(data, factor=2, min_valid_fraction=0.5)

        # Block [0:2, 0:2] has 2 valid cells (1, 2) -> should have mean = 1.5
        # Block [0:2, 2:4] has 4 valid cells -> mean
        assert np.isfinite(decimated[0, 0])
        assert decimated[0, 0] == 1.5
        assert np.isfinite(decimated[0, 1])

    def test_decimation_factor_1(self):
        """Test decimation with factor 1 (no change)."""
        data = np.array([[1, 2], [3, 4]], dtype=float)
        decimated = decimate_raster(data, factor=1)

        np.testing.assert_array_equal(decimated, data)

    def test_decimation_invalid_factor(self):
        """Test decimation with invalid factor."""
        data = np.array([[1, 2], [3, 4]], dtype=float)

        with pytest.raises(ValueError, match="Decimation factor must be >= 1"):
            decimate_raster(data, factor=0)

        with pytest.raises(ValueError, match="Decimation factor must be >= 1"):
            decimate_raster(data, factor=-1)

    def test_decimation_non_divisible_size(self):
        """Test decimation when grid size is not divisible by factor."""
        # Create 5x5 array
        data = np.arange(25, dtype=float).reshape(5, 5)

        # Degrade with factor 2
        decimated = decimate_raster(data, factor=2)

        # Expected: 2x2 array (floor(5/2) = 2)
        assert decimated.shape == (2, 2)

    def test_decimation_large_factor(self):
        """Test decimation with large factor."""
        data = np.arange(100, dtype=float).reshape(10, 10)
        decimated = decimate_raster(data, factor=5)

        # Expected: 2x2 array
        assert decimated.shape == (2, 2)

    def test_decimation_preserves_mean(self):
        """Test that decimation approximately preserves the global mean."""
        data = np.random.rand(100, 100)
        decimated = decimate_raster(data, factor=2, min_valid_fraction=0.0)

        # Global means should be close
        np.testing.assert_allclose(np.mean(data), np.mean(decimated), rtol=0.01)


class TestDegradeFlowDirection:
    """Tests for decimate_flow_direction function.
    TODO: Double check against MATLAB version.
    """

    def test_basic_flow_decimation(self):
        """Test basic flow direction and accumulation decimation.
        Tested vs MATLAB version (degrad_flac.m).
        """
        # Create simple 4x4 flow direction (all cells flow to the south, direction 6)
        flow_dir = np.full((4, 4), 6.0)
        flow_acc = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=float)

        # Degrade with factor 2
        deg_dir, deg_acc = decimate_flow_direction(flow_dir, flow_acc, factor=2)

        # Expected output (Grass r.watershed: 1=NE, 2=N, 6=S; image orientation).
        # Bottom row of coarse cells cannot drain south (out of bounds), so the
        # invalid-direction handler picks the first valid neighbor in code order;
        # for (1,0) that's NE (code 1) at (0,1); for (1,1) NE is out of bounds so
        # it falls through to N (code 2) at (0,1).
        assert deg_dir.shape == (2, 2)
        assert deg_dir[0, 0] == 6  # Top-left flows south
        assert deg_dir[0, 1] == 6  # Top-right flows south
        assert deg_dir[1, 0] == 1  # Bottom-left -> NE (top-right coarse cell)
        assert deg_dir[1, 1] == 2  # Bottom-right -> N (top-right coarse cell)

        assert deg_acc.shape == (2, 2)
        np.testing.assert_allclose(deg_acc[0, 0], 0.5)
        np.testing.assert_allclose(deg_acc[0, 1], 0.5)
        np.testing.assert_allclose(deg_acc[1, 0], 1.0)
        np.testing.assert_allclose(deg_acc[1, 1], 1.0)

    def test_flow_decimation_with_nan(self):
        """Test flow decimation with NaN values."""
        flow_dir = np.array([[1, 2, 3, 4], [5, np.nan, 7, 8], [1, 2, np.nan, 4], [5, 6, 7, 8]], dtype=float)
        flow_acc = np.array([[1, 2, 3, 4], [5, np.nan, 7, 8], [9, 10, np.nan, 12], [13, 14, 15, 16]], dtype=float)

        deg_dir, deg_acc = decimate_flow_direction(flow_dir, flow_acc, factor=2)

        # Should handle NaN gracefully
        assert deg_dir.shape == (2, 2)
        assert deg_acc.shape == (2, 2)

    def test_flow_decimation_factor_1(self):
        """Test flow decimation with factor 1 (no change)."""
        flow_dir = np.array([[1, 2], [3, 4]], dtype=float)
        flow_acc = np.array([[1, 2], [3, 4]], dtype=float)

        deg_dir, deg_acc = decimate_flow_direction(flow_dir, flow_acc, factor=1)

        np.testing.assert_array_equal(deg_dir, flow_dir)
        np.testing.assert_array_equal(deg_acc, flow_acc)

    def test_flow_decimation_invalid_factor(self):
        """Test flow decimation with invalid factor."""
        flow_dir = np.array([[1, 2], [3, 4]], dtype=float)
        flow_acc = np.array([[1, 2], [3, 4]], dtype=float)

        with pytest.raises(ValueError, match="Decimation factor must be >= 1"):
            decimate_flow_direction(flow_dir, flow_acc, factor=0)

    def test_flow_decimation_shape_mismatch(self):
        """Test flow decimation with mismatched array shapes."""
        flow_dir = np.array([[1, 2], [3, 4]], dtype=float)
        flow_acc = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

        with pytest.raises(ValueError, match="must have the same shape"):
            decimate_flow_direction(flow_dir, flow_acc, factor=2)

    def test_flow_accumulation_normalization(self):
        """Test that flow accumulation is normalized by factor²."""
        # Create uniform flow accumulation
        flow_dir = np.full((4, 4), 6.0)  # All flow south
        flow_acc = np.full((4, 4), 100.0)

        deg_dir, deg_acc = decimate_flow_direction(flow_dir, flow_acc, factor=2)

        # Accumulation should be divided by 4 (factor **2)
        expected_acc = 100.0 / 4
        np.testing.assert_allclose(deg_acc[np.isfinite(deg_acc)], expected_acc)

    def test_flow_direction_max_accumulation_selection(self):
        """Test that the cell with max accumulation determines flow direction."""
        # Create simple flow direction grid
        flow_dir = np.array(
            [
                [6, 6, 6, 6],  # All cells flow south
                [6, 6, 6, 6],
                [6, 6, 6, 6],
                [6, 6, 6, 6],
            ],
            dtype=float,
        )
        flow_acc = np.array(
            [
                [100, 1, 50, 1],  # max at [0,0] for top-left, max at [0,2] for top-right
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=float,
        )

        deg_dir, deg_acc = decimate_flow_direction(flow_dir, flow_acc, factor=2)

        # All coarse cells should have valid directions (not NaN)
        assert np.all(np.isfinite(deg_dir))

        # Check that max accumulation cell is correctly identified
        # Top-left coarse cell should use cell with acc=100
        assert deg_acc[0, 0] == 100.0 / 4  # normalized by factor **2
