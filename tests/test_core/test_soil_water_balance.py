"""
Tests for soil water balance module.

Tests both the capillary rise and soil mass balance functions.
"""

import numpy as np
from mobidic.core.soil_water_balance import capillary_rise, soil_mass_balance


class TestCapillaryRise:
    """Tests for capillary rise function."""

    def test_basic_calculation(self):
        """Test basic capillary rise calculation."""
        saturation = np.array([0.5, 0.7, 0.3])
        depth_to_water_table = np.array([1.0, 2.0, 0.5])
        hydraulic_conductivity = np.array([0.01, 0.01, 0.01])
        bubbling_pressure = np.array([0.2, 0.2, 0.2])
        param_a = np.array([-2.0, -2.0, -2.0])
        param_n = np.array([3.0, 3.0, 3.0])

        result = capillary_rise(
            saturation, depth_to_water_table, hydraulic_conductivity, bubbling_pressure, param_a, param_n
        )

        # Result should be finite (can be positive or negative representing upward/downward flux)
        assert np.all(np.isfinite(result))

    def test_zero_depth_returns_zero(self):
        """Test that zero or negative depth returns zero capillary rise."""
        saturation = np.array([0.5, 0.5, 0.5])
        depth_to_water_table = np.array([1.0, 0.0, -1.0])
        hydraulic_conductivity = np.array([0.01, 0.01, 0.01])
        bubbling_pressure = np.array([0.2, 0.2, 0.2])
        param_a = np.array([-2.0, -2.0, -2.0])
        param_n = np.array([3.0, 3.0, 3.0])

        result = capillary_rise(
            saturation, depth_to_water_table, hydraulic_conductivity, bubbling_pressure, param_a, param_n
        )

        assert np.isfinite(result[0])  # Positive depth should return finite value
        assert result[1] == 0  # Zero depth should have zero flux
        assert result[2] == 0  # Negative depth should have zero flux

    def test_nan_handling(self):
        """Test that NaN values are converted to zero."""
        saturation = np.array([0.5, np.nan, 0.3])
        depth_to_water_table = np.array([1.0, 1.0, 1.0])
        hydraulic_conductivity = np.array([0.01, 0.01, 0.01])
        bubbling_pressure = np.array([0.2, 0.2, 0.2])
        param_a = np.array([-2.0, -2.0, -2.0])
        param_n = np.array([3.0, 3.0, 3.0])

        result = capillary_rise(
            saturation, depth_to_water_table, hydraulic_conductivity, bubbling_pressure, param_a, param_n
        )

        # All results should be finite (NaN converted to 0)
        assert np.all(np.isfinite(result))


class TestSoilMassBalance:
    """Tests for soil mass balance function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a small test domain with 10 cells
        self.n_cells = 10

        # Initial states
        self.wc = np.full(self.n_cells, 0.1, dtype=np.float64)
        self.wc0 = np.full(self.n_cells, 0.3, dtype=np.float64)
        self.wg = np.full(self.n_cells, 0.05, dtype=np.float64)
        self.wg0 = np.full(self.n_cells, 0.2, dtype=np.float64)
        self.wp = np.full(self.n_cells, 0.001, dtype=np.float64)
        self.wp0 = np.full(self.n_cells, 0.005, dtype=np.float64)
        self.ws = np.full(self.n_cells, 0.0, dtype=np.float64)
        self.ws0 = np.full(self.n_cells, 0.01, dtype=np.float64)
        self.wtot0 = self.wc0 + self.wg0

        # Inputs (all pre-multiplied by dt)
        self.precipitation = np.full(self.n_cells, 0.01, dtype=np.float64)
        self.surface_runoff_in = np.zeros(self.n_cells, dtype=np.float64)
        self.lateral_flow_in = np.zeros(self.n_cells, dtype=np.float64)
        self.potential_et = np.full(self.n_cells, 0.005, dtype=np.float64)

        # Parameters (dimensionless, pre-multiplied by dt where applicable)
        self.ks = np.full(self.n_cells, 0.02, dtype=np.float64)
        self.channelized_fraction = np.full(self.n_cells, 0.1, dtype=np.float64)
        self.surface_flow_exp = np.full(self.n_cells, 0.9, dtype=np.float64)
        self.lateral_flow_coeff = np.full(self.n_cells, 0.2, dtype=np.float64)
        self.percolation_coeff = np.full(self.n_cells, 0.1, dtype=np.float64)
        self.absorption_coeff = np.full(self.n_cells, 0.3, dtype=np.float64)
        self.rainfall_fraction = np.full(self.n_cells, 0.5, dtype=np.float64)
        self.et_shape = 0.0
        self.surface_flow_param = np.full(self.n_cells, 0.5, dtype=np.float64)

    def test_basic_mass_balance(self):
        """Test basic mass balance without capillary rise."""
        (
            wc_out,
            wg_out,
            wp_out,
            ws_out,
            surface_runoff,
            lateral_flow,
            et,
            percolation,
            capillary_flux,
            wg_before_abs,
        ) = soil_mass_balance(
            self.wc,
            self.wc0,
            self.wg,
            self.wg0,
            self.wp,
            self.wp0,
            self.ws,
            self.ws0,
            self.wtot0,
            self.precipitation,
            self.surface_runoff_in,
            self.lateral_flow_in,
            self.potential_et,
            self.ks,
            None,
            None,
            self.channelized_fraction,
            self.surface_flow_exp,
            self.lateral_flow_coeff,
            self.percolation_coeff,
            self.absorption_coeff,
            self.rainfall_fraction,
            self.et_shape,
            capillary_rise_enabled=False,
            test_mode=False,
            surface_flow_param=self.surface_flow_param,
        )

        # Check mass balance
        water_in = (
            self.wc + self.wg + self.wp + self.ws + self.precipitation + self.surface_runoff_in + self.lateral_flow_in
        )
        water_out = (
            wc_out + wg_out + wp_out + ws_out + et + percolation + surface_runoff + lateral_flow - capillary_flux
        )

        np.testing.assert_allclose(water_in, water_out, rtol=1e-6, atol=1e-8)

    def test_mass_balance_with_upstream_flow(self):
        """Test mass balance with upstream surface and lateral flows."""
        self.surface_runoff_in = np.full(self.n_cells, 0.002, dtype=np.float64)
        self.lateral_flow_in = np.full(self.n_cells, 0.001, dtype=np.float64)

        (
            wc_out,
            wg_out,
            wp_out,
            ws_out,
            surface_runoff,
            lateral_flow,
            et,
            percolation,
            capillary_flux,
            wg_before_abs,
        ) = soil_mass_balance(
            self.wc,
            self.wc0,
            self.wg,
            self.wg0,
            self.wp,
            self.wp0,
            self.ws,
            self.ws0,
            self.wtot0,
            self.precipitation,
            self.surface_runoff_in,
            self.lateral_flow_in,
            self.potential_et,
            self.ks,
            None,
            None,
            self.channelized_fraction,
            self.surface_flow_exp,
            self.lateral_flow_coeff,
            self.percolation_coeff,
            self.absorption_coeff,
            self.rainfall_fraction,
            self.et_shape,
            capillary_rise_enabled=False,
            test_mode=False,
            surface_flow_param=self.surface_flow_param,
        )

        # Check mass balance
        water_in = (
            self.wc + self.wg + self.wp + self.ws + self.precipitation + self.surface_runoff_in + self.lateral_flow_in
        )
        water_out = (
            wc_out + wg_out + wp_out + ws_out + et + percolation + surface_runoff + lateral_flow - capillary_flux
        )

        np.testing.assert_allclose(water_in, water_out, rtol=1e-6, atol=1e-8)

    def test_no_plant_reservoir(self):
        """Test mass balance without plant reservoir (wp=None)."""
        (
            wc_out,
            wg_out,
            wp_out,
            ws_out,
            surface_runoff,
            lateral_flow,
            et,
            percolation,
            capillary_flux,
            wg_before_abs,
        ) = soil_mass_balance(
            self.wc,
            self.wc0,
            self.wg,
            self.wg0,
            None,  # No plant reservoir
            None,
            self.ws,
            self.ws0,
            self.wtot0,
            self.precipitation,
            self.surface_runoff_in,
            self.lateral_flow_in,
            self.potential_et,
            self.ks,
            None,
            None,
            self.channelized_fraction,
            self.surface_flow_exp,
            self.lateral_flow_coeff,
            self.percolation_coeff,
            self.absorption_coeff,
            self.rainfall_fraction,
            self.et_shape,
            capillary_rise_enabled=False,
            test_mode=False,
            surface_flow_param=self.surface_flow_param,
        )

        # wp_out should be None
        assert wp_out is None

        # Check mass balance (without wp)
        water_in = self.wc + self.wg + self.ws + self.precipitation + self.surface_runoff_in + self.lateral_flow_in
        water_out = wc_out + wg_out + ws_out + et + percolation + surface_runoff + lateral_flow - capillary_flux

        np.testing.assert_allclose(water_in, water_out, rtol=1e-6, atol=1e-8)

    def test_et_shape_parameter(self):
        """Test ET shape parameter effect."""
        # Use lower initial saturation to see ET shape effect more clearly
        # Reduce plant reservoir to zero so all ET comes from soil
        wc_dry = np.full(self.n_cells, 0.02, dtype=np.float64)
        wg_dry = np.full(self.n_cells, 0.01, dtype=np.float64)
        wp_zero = np.zeros(self.n_cells, dtype=np.float64)
        precip_zero = np.zeros(self.n_cells, dtype=np.float64)

        # Run with ET shape = 0 (default)
        result_default = soil_mass_balance(
            wc_dry,
            self.wc0,
            wg_dry,
            self.wg0,
            wp_zero,  # No plant water
            self.wp0,
            self.ws,
            self.ws0,
            self.wtot0,
            precip_zero,  # No precipitation
            self.surface_runoff_in,
            self.lateral_flow_in,
            self.potential_et,
            self.ks,
            None,
            None,
            self.channelized_fraction,
            self.surface_flow_exp,
            self.lateral_flow_coeff,
            self.percolation_coeff,
            self.absorption_coeff,
            self.rainfall_fraction,
            0.0,  # ET shape = 0
            capillary_rise_enabled=False,
            test_mode=False,
            surface_flow_param=self.surface_flow_param,
        )

        # Run with ET shape = 3 (recommended)
        result_shaped = soil_mass_balance(
            wc_dry,
            self.wc0,
            wg_dry,
            self.wg0,
            wp_zero,  # No plant water
            self.wp0,
            self.ws,
            self.ws0,
            self.wtot0,
            precip_zero,  # No precipitation
            self.surface_runoff_in,
            self.lateral_flow_in,
            self.potential_et,
            self.ks,
            None,
            None,
            self.channelized_fraction,
            self.surface_flow_exp,
            self.lateral_flow_coeff,
            self.percolation_coeff,
            self.absorption_coeff,
            self.rainfall_fraction,
            3.0,  # ET shape = 3
            capillary_rise_enabled=False,
            test_mode=False,
            surface_flow_param=self.surface_flow_param,
        )

        # ET should be different (shaped ET reduces soil ET for low saturation)
        # With low saturation, shaped ET should be lower than default
        assert not np.allclose(result_default[6], result_shaped[6])
        assert np.mean(result_shaped[6]) < np.mean(result_default[6])

    def test_ks_min_max_mode(self):
        """Test using ks_min and ks_max instead of single ks."""
        ks_min = np.full(self.n_cells, 0.01, dtype=np.float64)
        ks_max = np.full(self.n_cells, 0.03, dtype=np.float64)

        (
            wc_out,
            wg_out,
            wp_out,
            ws_out,
            surface_runoff,
            lateral_flow,
            et,
            percolation,
            capillary_flux,
            wg_before_abs,
        ) = soil_mass_balance(
            self.wc,
            self.wc0,
            self.wg,
            self.wg0,
            self.wp,
            self.wp0,
            self.ws,
            self.ws0,
            self.wtot0,
            self.precipitation,
            self.surface_runoff_in,
            self.lateral_flow_in,
            self.potential_et,
            None,  # No single ks
            ks_min,
            ks_max,
            self.channelized_fraction,
            self.surface_flow_exp,
            self.lateral_flow_coeff,
            self.percolation_coeff,
            self.absorption_coeff,
            self.rainfall_fraction,
            self.et_shape,
            capillary_rise_enabled=False,
            test_mode=False,
            surface_flow_param=self.surface_flow_param,
        )

        # Check mass balance
        water_in = (
            self.wc + self.wg + self.wp + self.ws + self.precipitation + self.surface_runoff_in + self.lateral_flow_in
        )
        water_out = (
            wc_out + wg_out + wp_out + ws_out + et + percolation + surface_runoff + lateral_flow - capillary_flux
        )

        np.testing.assert_allclose(water_in, water_out, rtol=1e-6, atol=1e-8)

    def test_capillary_rise_enabled(self):
        """Test mass balance with capillary rise enabled."""
        # Set up capillary rise parameters
        soil_depth = np.full(self.n_cells, 1.0, dtype=np.float64)
        depth_to_water_table = np.full(self.n_cells, 0.5, dtype=np.float64)
        capillary_conductivity = np.full(self.n_cells, 0.01, dtype=np.float64)  # Increased for more effect
        bubbling_pressure = np.full(self.n_cells, 0.3, dtype=np.float64)
        capillary_param_a = np.full(self.n_cells, -0.5, dtype=np.float64)
        capillary_param_n = np.full(self.n_cells, 2.0, dtype=np.float64)
        capillary_multiplier = np.ones(self.n_cells, dtype=np.float64)

        # Start with low Wc and zero Wg to activate capillary rise
        wc_init = np.full(self.n_cells, 0.05, dtype=np.float64)
        wg_init = np.zeros(self.n_cells, dtype=np.float64)

        (
            wc_out,
            wg_out,
            wp_out,
            ws_out,
            surface_runoff,
            lateral_flow,
            et,
            percolation,
            capillary_flux,
            wg_before_abs,
        ) = soil_mass_balance(
            wc_init,
            self.wc0,
            wg_init,
            self.wg0,
            self.wp,
            self.wp0,
            self.ws,
            self.ws0,
            self.wtot0,
            self.precipitation,
            self.surface_runoff_in,
            self.lateral_flow_in,
            self.potential_et,
            self.ks,
            None,
            None,
            self.channelized_fraction,
            self.surface_flow_exp,
            self.lateral_flow_coeff,
            self.percolation_coeff,
            self.absorption_coeff,
            self.rainfall_fraction,
            self.et_shape,
            capillary_rise_enabled=True,
            soil_depth=soil_depth,
            depth_to_water_table=depth_to_water_table,
            capillary_conductivity=capillary_conductivity,
            bubbling_pressure=bubbling_pressure,
            capillary_param_a=capillary_param_a,
            capillary_param_n=capillary_param_n,
            capillary_multiplier=capillary_multiplier,
            test_mode=False,
            surface_flow_param=self.surface_flow_param,
        )

        # Capillary flux should be finite (may be positive or negative)
        assert np.all(np.isfinite(capillary_flux))

        # Check mass balance (including capillary rise)
        water_in = (
            wc_init + wg_init + self.wp + self.ws + self.precipitation + self.surface_runoff_in + self.lateral_flow_in
        )
        water_out = (
            wc_out + wg_out + wp_out + ws_out + et + percolation + surface_runoff + lateral_flow - capillary_flux
        )

        np.testing.assert_allclose(water_in, water_out, rtol=1e-6, atol=1e-8)

    def test_zero_precipitation(self):
        """Test mass balance with zero precipitation."""
        self.precipitation = np.zeros(self.n_cells, dtype=np.float64)

        (
            wc_out,
            wg_out,
            wp_out,
            ws_out,
            surface_runoff,
            lateral_flow,
            et,
            percolation,
            capillary_flux,
            wg_before_abs,
        ) = soil_mass_balance(
            self.wc,
            self.wc0,
            self.wg,
            self.wg0,
            self.wp,
            self.wp0,
            self.ws,
            self.ws0,
            self.wtot0,
            self.precipitation,
            self.surface_runoff_in,
            self.lateral_flow_in,
            self.potential_et,
            self.ks,
            None,
            None,
            self.channelized_fraction,
            self.surface_flow_exp,
            self.lateral_flow_coeff,
            self.percolation_coeff,
            self.absorption_coeff,
            self.rainfall_fraction,
            self.et_shape,
            capillary_rise_enabled=False,
            test_mode=False,
            surface_flow_param=self.surface_flow_param,
        )

        # With no precipitation, total soil water should decrease or stay same
        # (wc can increase due to absorption from wg, but total should decrease due to ET)
        total_soil_in = self.wc + self.wg
        total_soil_out = wc_out + wg_out
        assert np.all(total_soil_out <= total_soil_in + 1e-8)

        # Check mass balance
        water_in = (
            self.wc + self.wg + self.wp + self.ws + self.precipitation + self.surface_runoff_in + self.lateral_flow_in
        )
        water_out = (
            wc_out + wg_out + wp_out + ws_out + et + percolation + surface_runoff + lateral_flow - capillary_flux
        )

        np.testing.assert_allclose(water_in, water_out, rtol=1e-6, atol=1e-8)

    def test_large_precipitation(self):
        """Test mass balance with large precipitation (should produce runoff)."""
        self.precipitation = np.full(self.n_cells, 0.1, dtype=np.float64)

        (
            wc_out,
            wg_out,
            wp_out,
            ws_out,
            surface_runoff,
            lateral_flow,
            et,
            percolation,
            capillary_flux,
            wg_before_abs,
        ) = soil_mass_balance(
            self.wc,
            self.wc0,
            self.wg,
            self.wg0,
            self.wp,
            self.wp0,
            self.ws,
            self.ws0,
            self.wtot0,
            self.precipitation,
            self.surface_runoff_in,
            self.lateral_flow_in,
            self.potential_et,
            self.ks,
            None,
            None,
            self.channelized_fraction,
            self.surface_flow_exp,
            self.lateral_flow_coeff,
            self.percolation_coeff,
            self.absorption_coeff,
            self.rainfall_fraction,
            self.et_shape,
            capillary_rise_enabled=False,
            test_mode=False,
            surface_flow_param=self.surface_flow_param,
        )

        # Large precipitation should produce surface runoff
        assert np.all(surface_runoff > 0)

        # Check mass balance
        water_in = (
            self.wc + self.wg + self.wp + self.ws + self.precipitation + self.surface_runoff_in + self.lateral_flow_in
        )
        water_out = (
            wc_out + wg_out + wp_out + ws_out + et + percolation + surface_runoff + lateral_flow - capillary_flux
        )

        np.testing.assert_allclose(water_in, water_out, rtol=1e-6, atol=1e-8)
