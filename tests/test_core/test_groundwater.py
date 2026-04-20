"""Tests for mobidic.core.groundwater module."""

import numpy as np
from mobidic.core.groundwater import groundwater_linear


class TestGroundwaterLinear:
    """Tests for groundwater_linear function."""

    def test_analytical_no_recharge(self):
        """Without recharge, head decays exponentially: h = h0 * exp(-kf*dt)."""
        h0 = np.array([1.0, 2.0, 0.5])
        kf = np.array([1e-6, 1e-6, 1e-6])
        recharge = np.zeros(3)
        dt = 3600.0

        h, q = groundwater_linear(h0, kf, recharge, dt)

        expected_h = h0 * np.exp(-kf * dt)
        np.testing.assert_allclose(h, expected_h, rtol=1e-12)

        # Baseflow is the average over dt: q = a*h0 * (1-exp(-a*dt))/(a*dt)
        expected_q = kf * h0 * (1.0 - np.exp(-kf * dt)) / (kf * dt)
        np.testing.assert_allclose(q, expected_q, rtol=1e-12)

    def test_steady_state(self):
        """With constant recharge and h0 = R/kf, the head is at steady state."""
        kf = np.array([1e-5])
        recharge = np.array([5e-7])
        h0 = recharge / kf  # Steady-state head
        dt = 3600.0

        h, q = groundwater_linear(h0, kf, recharge, dt)

        np.testing.assert_allclose(h, h0, rtol=1e-12)
        np.testing.assert_allclose(q, recharge, rtol=1e-12)

    def test_zero_kf_edge_case(self):
        """When kf <= 0 (no aquifer), head is advanced linearly and baseflow equals recharge."""
        h0 = np.array([1.0, 2.0])
        kf = np.array([0.0, 0.0])
        recharge = np.array([1e-7, -1e-7])
        dt = 1800.0

        h, q = groundwater_linear(h0, kf, recharge, dt)

        np.testing.assert_allclose(h, h0 + recharge * dt, rtol=1e-12)
        np.testing.assert_allclose(q, recharge, rtol=1e-12)

    def test_mixed_active_inactive(self):
        """Active and inactive cells are handled together."""
        h0 = np.array([1.0, 2.0, 0.5])
        kf = np.array([1e-6, 0.0, 2e-6])
        recharge = np.array([0.0, 1e-8, 1e-7])
        dt = 3600.0

        h, q = groundwater_linear(h0, kf, recharge, dt)

        # Middle cell: inactive, linear advance
        np.testing.assert_allclose(h[1], h0[1] + recharge[1] * dt, rtol=1e-12)
        np.testing.assert_allclose(q[1], recharge[1], rtol=1e-12)

        # Active cells: solve manually for comparison
        for i in (0, 2):
            exp_adt = np.exp(-kf[i] * dt)
            Rn = max(recharge[i], -0.5 * kf[i] * h0[i] * exp_adt / (1.0 - exp_adt))
            h_ref = Rn / kf[i] + (h0[i] - Rn / kf[i]) * exp_adt
            q_ref = (kf[i] * h0[i] - Rn) / (kf[i] * dt) * (1.0 - exp_adt) + Rn
            np.testing.assert_allclose(h[i], h_ref, rtol=1e-12)
            np.testing.assert_allclose(q[i], q_ref, rtol=1e-12)

    def test_negative_recharge_clamped(self):
        """Strong negative recharge is clamped so head stays non-negative (matching MATLAB)."""
        kf = np.array([1e-5])
        h0 = np.array([0.1])
        # Very large negative recharge that would drive h < 0 without clamping
        recharge = np.array([-1e-3])
        dt = 3600.0

        h, q = groundwater_linear(h0, kf, recharge, dt)

        # Head must remain non-negative due to MATLAB's Rn clamp
        assert h[0] >= 0.0 - 1e-15

    def test_matches_matlab_reference(self):
        """Full equation reproduction of MATLAB groundwater_linear.m."""
        rng = np.random.default_rng(seed=42)
        n = 50
        h0 = rng.uniform(0.05, 5.0, size=n)
        kf = rng.uniform(1e-7, 1e-4, size=n)
        recharge = rng.uniform(-1e-6, 1e-6, size=n)
        dt = 900.0

        h, q = groundwater_linear(h0, kf, recharge, dt)

        # Reference: direct translation of MATLAB lines 25-37
        exp_adt = np.exp(-kf * dt)
        Rn_ref = np.maximum(recharge, -0.5 * kf * h0 * exp_adt / (1.0 - exp_adt))
        q_ref = (kf * h0 - Rn_ref) / (kf * dt) * (1.0 - exp_adt) + Rn_ref
        R_over_a = Rn_ref / kf
        h_ref = R_over_a + (h0 - R_over_a) * exp_adt

        np.testing.assert_allclose(h, h_ref, rtol=1e-6)
        np.testing.assert_allclose(q, q_ref, rtol=1e-6)

    def test_shape_preserved(self):
        """Output arrays have the same shape as the inputs."""
        h0 = np.ones(10) * 0.5
        kf = np.ones(10) * 1e-5
        recharge = np.zeros(10)

        h, q = groundwater_linear(h0, kf, recharge, 900.0)

        assert h.shape == h0.shape
        assert q.shape == h0.shape
