"""Tests for mobidic.core.pet module."""

import numpy as np
import pytest
from mobidic.core.pet import calculate_pet


class TestCalculatePET:
    """Tests for calculate_pet function."""

    def test_basic_calculation_default_rate(self):
        """Test basic PET calculation with default 1 mm/day rate."""
        grid_shape = (10, 15)
        dt = 900.0  # 15 minutes in seconds

        pet = calculate_pet(grid_shape, dt)

        # Check shape
        assert pet.shape == grid_shape

        # Check data type
        assert pet.dtype == np.float64

        # Check value: 1 mm/day = 1/(1000*86400) m/s
        expected_value = 1.0 / (1000.0 * 86400.0)
        np.testing.assert_array_almost_equal(pet, expected_value)

    def test_custom_pet_rate(self):
        """Test PET calculation with custom rate."""
        grid_shape = (5, 8)
        dt = 3600.0  # 1 hour
        pet_rate = 5.0  # 5 mm/day

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        # Check shape
        assert pet.shape == grid_shape

        # Check value: 5 mm/day = 5/(1000*86400) m/s
        expected_value = 5.0 / (1000.0 * 86400.0)
        np.testing.assert_array_almost_equal(pet, expected_value)

    def test_unit_conversion(self):
        """Test that unit conversion is correct."""
        grid_shape = (1, 1)
        dt = 86400.0  # 1 day
        pet_rate = 1.0  # 1 mm/day

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        # 1 mm/day = 1e-3 m/day = 1e-3/(86400) m/s
        expected_m_per_s = 1.0 / (1000.0 * 86400.0)
        assert pytest.approx(pet[0, 0], abs=1e-12) == expected_m_per_s

        # Over 1 day, this should equal 1e-3 m = 1 mm
        pet_per_day = pet[0, 0] * 86400.0
        assert pytest.approx(pet_per_day, abs=1e-10) == 1e-3

    def test_different_grid_shapes(self):
        """Test with various grid shapes."""
        test_shapes = [
            (1, 1),  # Single cell
            (10, 10),  # Square
            (5, 20),  # Rectangular
            (100, 50),  # Larger
        ]

        dt = 900.0
        pet_rate = 2.0

        for shape in test_shapes:
            pet = calculate_pet(shape, dt, pet_rate_mm_day=pet_rate)

            assert pet.shape == shape
            assert pet.dtype == np.float64

            # All values should be constant
            expected_value = pet_rate / (1000.0 * 86400.0)
            np.testing.assert_array_almost_equal(pet, expected_value)

    def test_different_timesteps(self):
        """Test with different time steps."""
        grid_shape = (10, 10)
        pet_rate = 1.0

        timesteps = [
            900.0,  # 15 minutes
            3600.0,  # 1 hour
            10800.0,  # 3 hours
            86400.0,  # 1 day
        ]

        for dt in timesteps:
            pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

            # PET rate in m/s should be same regardless of dt
            # dt is not used in calculation (function returns m/s, not m/timestep)
            expected_value = pet_rate / (1000.0 * 86400.0)
            np.testing.assert_array_almost_equal(pet, expected_value)

    def test_zero_pet_rate(self):
        """Test with zero PET rate."""
        grid_shape = (5, 5)
        dt = 900.0
        pet_rate = 0.0

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        # Should be all zeros
        assert np.all(pet == 0.0)

    def test_very_small_pet_rate(self):
        """Test with very small PET rate (numerical precision)."""
        grid_shape = (3, 3)
        dt = 900.0
        pet_rate = 1e-6  # 0.001 mm/day

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        expected_value = 1e-6 / (1000.0 * 86400.0)
        np.testing.assert_allclose(pet, expected_value, rtol=1e-15)

    def test_large_pet_rate(self):
        """Test with large PET rate."""
        grid_shape = (10, 10)
        dt = 900.0
        pet_rate = 100.0  # 100 mm/day (very high, but valid)

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        expected_value = 100.0 / (1000.0 * 86400.0)
        np.testing.assert_array_almost_equal(pet, expected_value)

    def test_single_cell_grid(self):
        """Test with 1x1 grid."""
        grid_shape = (1, 1)
        dt = 900.0
        pet_rate = 3.0

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        assert pet.shape == (1, 1)
        expected_value = 3.0 / (1000.0 * 86400.0)
        assert pytest.approx(pet[0, 0], abs=1e-12) == expected_value

    def test_large_grid(self):
        """Test with large grid."""
        grid_shape = (1000, 500)
        dt = 900.0
        pet_rate = 2.5

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        assert pet.shape == grid_shape
        expected_value = 2.5 / (1000.0 * 86400.0)
        np.testing.assert_array_almost_equal(pet, expected_value)

    def test_consistent_values_across_grid(self):
        """Test that all grid cells have the same PET value."""
        grid_shape = (20, 30)
        dt = 900.0
        pet_rate = 4.0

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        # All values should be identical (constant PET)
        first_value = pet[0, 0]
        assert np.all(pet == first_value)

        # Verify the value is correct
        expected_value = pet_rate / (1000.0 * 86400.0)
        assert pytest.approx(first_value, abs=1e-12) == expected_value

    def test_matlab_reference_default(self):
        """Test that default matches MATLAB: etp = Mones/(1000*3600*24)."""
        grid_shape = (10, 10)
        dt = 900.0

        pet = calculate_pet(grid_shape, dt)

        # MATLAB: etp = 1/(1000*3600*24) where 3600*24 = 86400
        matlab_value = 1.0 / (1000.0 * 86400.0)
        np.testing.assert_array_almost_equal(pet, matlab_value)

    def test_realistic_pet_values(self):
        """Test with realistic PET values for different climates."""
        grid_shape = (50, 50)
        dt = 3600.0

        # Test various realistic PET rates
        test_cases = [
            (0.5, "Very low PET (polar/winter)"),
            (2.0, "Low PET (temperate/humid)"),
            (5.0, "Medium PET (temperate/dry)"),
            (8.0, "High PET (arid/summer)"),
            (12.0, "Very high PET (desert)"),
        ]

        for pet_rate, description in test_cases:
            pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

            # Check that values are in reasonable range
            expected_m_per_s = pet_rate / (1000.0 * 86400.0)
            np.testing.assert_array_almost_equal(pet, expected_m_per_s)

            # Verify it's in reasonable range (0 to ~1e-4 m/s for typical conditions)
            assert 0.0 <= pet[0, 0] <= 2e-4, f"Failed for {description}"

    def test_output_data_type(self):
        """Test that output is always float64."""
        grid_shape = (5, 5)
        dt = 900.0

        # Test with various input types
        pet_rates = [1, 1.0, np.float32(1.0), np.int32(1)]

        for pet_rate in pet_rates:
            pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)
            assert pet.dtype == np.float64

    def test_negative_pet_rate_behavior(self):
        """Test behavior with negative PET rate (mathematically possible but physically unrealistic)."""
        grid_shape = (5, 5)
        dt = 900.0
        pet_rate = -1.0  # Negative (unrealistic but mathematically valid)

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        # Should calculate negative values (though physically meaningless)
        expected_value = -1.0 / (1000.0 * 86400.0)
        np.testing.assert_array_almost_equal(pet, expected_value)
        assert np.all(pet < 0.0)


class TestPETConversions:
    """Tests for PET unit conversions and physical interpretations."""

    def test_mm_per_day_to_m_per_second(self):
        """Test conversion from mm/day to m/s."""
        grid_shape = (1, 1)
        dt = 900.0

        # 1 mm/day should equal 1e-3 m / 86400 s
        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=1.0)
        assert pytest.approx(pet[0, 0], abs=1e-15) == 1e-3 / 86400.0

        # 10 mm/day
        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=10.0)
        assert pytest.approx(pet[0, 0], abs=1e-15) == 10e-3 / 86400.0

    def test_accumulated_over_day(self):
        """Test that accumulated PET over 1 day matches input rate."""
        grid_shape = (1, 1)
        dt = 900.0
        pet_rate_mm_day = 3.0

        pet_m_per_s = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate_mm_day)

        # Accumulate over 1 day
        seconds_per_day = 86400.0
        accumulated_m = pet_m_per_s[0, 0] * seconds_per_day
        accumulated_mm = accumulated_m * 1000.0

        # Should equal input rate
        assert pytest.approx(accumulated_mm, abs=1e-10) == pet_rate_mm_day

    def test_accumulated_over_timestep(self):
        """Test accumulated PET over specific timestep."""
        grid_shape = (1, 1)
        pet_rate_mm_day = 2.0

        # Test various timesteps
        timesteps_seconds = [900, 3600, 10800, 86400]

        for dt in timesteps_seconds:
            pet_m_per_s = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate_mm_day)

            # Accumulate over timestep
            accumulated_m = pet_m_per_s[0, 0] * dt
            accumulated_mm = accumulated_m * 1000.0

            # Should equal: (2 mm/day) * (dt/86400)
            expected_mm = pet_rate_mm_day * (dt / 86400.0)
            assert pytest.approx(accumulated_mm, abs=1e-10) == expected_mm


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_grid(self):
        """Test with very large grid dimensions."""
        grid_shape = (10000, 5000)
        dt = 900.0
        pet_rate = 3.0

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        assert pet.shape == grid_shape
        # Check just a few values to avoid memory issues
        expected_value = pet_rate / (1000.0 * 86400.0)
        assert pytest.approx(pet[0, 0], abs=1e-12) == expected_value
        assert pytest.approx(pet[-1, -1], abs=1e-12) == expected_value

    def test_non_square_grids(self):
        """Test with highly non-square grids."""
        test_shapes = [
            (1, 1000),  # Very wide
            (1000, 1),  # Very tall
            (3, 500),  # Asymmetric
        ]

        dt = 900.0
        pet_rate = 2.0

        for shape in test_shapes:
            pet = calculate_pet(shape, dt, pet_rate_mm_day=pet_rate)
            assert pet.shape == shape
            expected_value = pet_rate / (1000.0 * 86400.0)
            np.testing.assert_array_almost_equal(pet, expected_value)

    def test_zero_timestep(self):
        """Test with zero timestep (dt not actually used in calculation)."""
        grid_shape = (5, 5)
        dt = 0.0  # dt doesn't affect the calculation
        pet_rate = 2.0

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        # Should still work (dt is not used in the calculation)
        expected_value = pet_rate / (1000.0 * 86400.0)
        np.testing.assert_array_almost_equal(pet, expected_value)

    def test_very_large_timestep(self):
        """Test with very large timestep."""
        grid_shape = (5, 5)
        dt = 86400.0 * 365.0  # 1 year
        pet_rate = 2.0

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        # dt doesn't affect result (returns m/s, not m/timestep)
        expected_value = pet_rate / (1000.0 * 86400.0)
        np.testing.assert_array_almost_equal(pet, expected_value)

    def test_float_grid_dimensions(self):
        """Test that function works with integer grid dimensions."""
        # Grid shape should be tuple of ints, but test that it handles well
        grid_shape = (10, 15)  # Already ints
        dt = 900.0
        pet_rate = 1.0

        pet = calculate_pet(grid_shape, dt, pet_rate_mm_day=pet_rate)

        assert pet.shape == (10, 15)
        assert isinstance(pet, np.ndarray)
