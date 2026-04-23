"""Tests for mobidic.core.energy_balance module."""

import numpy as np
import pytest

from mobidic.core import constants as const
from mobidic.core.energy_balance import (
    _energy_balance_1l_numpy,
    compute_energy_balance_1l,
    diurnal_radiation_cycle,
    energy_balance_1l,
    saturation_specific_humidity,
    solar_hours,
    solar_position,
)


class TestSolarPosition:
    """Tests for solar_position function."""

    def test_equinox_noon_equator(self):
        """At the equator on the equinox at solar noon the sun is near zenith."""
        # March equinox is around day 80; at lon=0 solar noon is ~12.0
        _, el = solar_position(hour=12.0, day=80, lat=0.0, lon=0.0)
        assert el > 89.0  # near zenith

    def test_summer_solstice_tropic_of_cancer(self):
        """At the Tropic of Cancer on the summer solstice (day 172) sun is at zenith at noon."""
        _, el = solar_position(hour=12.0, day=172, lat=23.45, lon=0.0)
        assert el > 89.5  # essentially zenith

    def test_winter_night_polar(self):
        """High northern latitude in deep winter: sun stays below horizon all day."""
        # Day 355 (late December), lat 80N — well within polar night
        elevations = [solar_position(h, 355, 80.0, 0.0)[1] for h in np.linspace(0, 24, 25)]
        assert all(el < 0.0 for el in elevations)

    def test_continuity_around_noon(self):
        """Elevation is symmetric around solar noon at the equator on the equinox."""
        _, el_morning = solar_position(10.0, 80, 0.0, 0.0)
        _, el_afternoon = solar_position(14.0, 80, 0.0, 0.0)
        assert abs(el_morning - el_afternoon) < 0.5


class TestSolarHours:
    """Tests for solar_hours function."""

    def test_equinox_equator_12h(self):
        """On the equinox the day length is approximately 12 hours."""
        hrise, hset = solar_hours(lat=0.0, lon=0.0, jday=80)
        day_length = hset - hrise
        assert abs(day_length - 12.0) < 0.1

    def test_summer_longer_than_winter_north(self):
        """Northern hemisphere: summer day is longer than winter day."""
        hrise_s, hset_s = solar_hours(lat=45.0, lon=0.0, jday=172)  # ~June 21
        hrise_w, hset_w = solar_hours(lat=45.0, lon=0.0, jday=355)  # ~Dec 21
        assert (hset_s - hrise_s) > (hset_w - hrise_w)

    def test_sunrise_before_sunset(self):
        """Sunrise must precede sunset."""
        for jday in [1, 80, 172, 264, 355]:
            hrise, hset = solar_hours(lat=43.0, lon=11.0, jday=jday)
            assert hrise < hset
            assert 0 <= hrise <= 12
            assert 12 <= hset <= 24


class TestSaturationSpecificHumidity:
    """Tests for saturation_specific_humidity function."""

    def test_zero_celsius_reference(self):
        """qsat at 0°C, 1013 mb is approximately 3.78 g/kg."""
        T = np.array([273.15])
        q = saturation_specific_humidity(T, 1013.0)
        # 6.112 mb saturation vapor pressure at 0°C → q ≈ 0.622 * 6.112 / (1013 - 0.378*6.112)
        expected = 0.622 * 6.112 / (1013.0 - 0.378 * 6.112)
        np.testing.assert_allclose(q, expected, rtol=1e-6)

    def test_increases_with_temperature(self):
        """qsat is monotonically increasing with temperature."""
        T = np.linspace(263.15, 313.15, 11)
        q = saturation_specific_humidity(T, 1013.0)
        assert np.all(np.diff(q) > 0)

    def test_room_temperature(self):
        """qsat at 20°C, 1013 mb ≈ 14.7 g/kg."""
        T = np.array([293.15])
        q = saturation_specific_humidity(T, 1013.0)
        # Reference value from psychrometric tables ~ 0.0147
        np.testing.assert_allclose(q, 0.0147, atol=1e-3)


class TestDiurnalRadiationCycle:
    """Tests for diurnal_radiation_cycle function."""

    def test_average_mode_recovers_average(self):
        """The reconstructed cycle, integrated over the day, returns rs_avg * day_length."""
        rs_avg = 200.0
        t_sunrise = 6.0 * 3600.0
        t_sunset = 18.0 * 3600.0
        amp, c = diurnal_radiation_cycle(rs_avg, t_sunrise, t_sunset, mode="average")

        omega = 2.0 * np.pi / 86400.0
        phase = -np.pi / 2.0
        # Integrate amp*sin(omega*t+phase)+c from sunrise to sunset
        t = np.linspace(t_sunrise, t_sunset, 10001)
        rad = amp * np.sin(omega * t + phase) + c
        integral = np.trapezoid(rad, t)
        # Should equal rs_avg * (t_sunset - t_sunrise)
        np.testing.assert_allclose(integral, rs_avg * (t_sunset - t_sunrise), rtol=1e-3)

    def test_radiation_zero_at_sunset(self):
        """The reconstructed radiation (constant + sine) is zero at sunset."""
        rs_avg = 250.0
        t_sunrise = 6.0 * 3600.0
        t_sunset = 18.0 * 3600.0
        amp, c = diurnal_radiation_cycle(rs_avg, t_sunrise, t_sunset, mode="average")

        omega = 2.0 * np.pi / 86400.0
        phase = -np.pi / 2.0
        rad_set = amp * np.sin(omega * t_sunset + phase) + c
        np.testing.assert_allclose(rad_set, 0.0, atol=1e-6)

    def test_invalid_mode_raises(self):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            diurnal_radiation_cycle(100.0, 0.0, 86400.0, mode="bogus")


class TestEnergyBalance1L:
    """Tests for the analytical 1L solver."""

    def _make_inputs(self, n=4):
        """Build a realistic per-cell input set."""
        return {
            "ff": 2.0 * np.pi / 86400.0,
            "a_tem": np.full(n, 5.0),
            "a_rad": np.full(n, 400.0),
            "p_tem": -np.pi / 2.0 - np.pi / 6.0,
            "p_rad": -np.pi / 2.0,
            "c_tem": np.full(n, 290.0),
            "c_rad": np.full(n, 150.0),
            "td_ini": np.full(n, 290.0),
            "tm": np.full(n, 290.0),
            "u": np.full(n, 2.0),
            "pair": const.P_AIR,
            "hair": np.full(n, 0.6),
            "t_end": 3600.0,
            "step": 3600.0,
            "ch": np.full(n, 1e-3),
            "alb": np.full(n, 0.2),
            "kaps": 2.5,
            "nis": 0.8e-6,
            "tcost": 290.0,
            "etrsuetp": 1.0,
        }

    def test_returns_correct_shapes(self):
        kw = self._make_inputs(n=5)
        ts, td, evp = energy_balance_1l(**kw)
        assert ts.shape == (5,)
        assert td.shape == (5,)
        assert evp.shape == (5,)

    def test_evaporation_non_negative(self):
        """Evaporation is clipped to non-negative."""
        kw = self._make_inputs()
        _, _, evp = energy_balance_1l(**kw)
        assert np.all(evp >= 0.0)

    def test_etrsuetp_zero_no_evaporation(self):
        """When etrsuetp=0 (water-limited dry surface), evaporation vanishes."""
        kw = self._make_inputs()
        kw["etrsuetp"] = 0.0
        _, _, evp = energy_balance_1l(**kw)
        np.testing.assert_allclose(evp, 0.0, atol=1e-12)

    def test_finite_outputs(self):
        kw = self._make_inputs()
        ts, td, evp = energy_balance_1l(**kw)
        assert np.all(np.isfinite(ts))
        assert np.all(np.isfinite(td))
        assert np.all(np.isfinite(evp))


class TestEnergyBalance1LJitVsNumpy:
    """Regression: Numba-compiled kernel must match the NumPy reference."""

    def _heterogeneous_inputs(self, n: int, *, day: bool, rng: np.random.Generator):
        """Build per-cell inputs with realistic spatial variability."""
        tm = rng.uniform(278.0, 303.0, size=n)
        a_tem = rng.uniform(2.0, 8.0, size=n)
        c_tem = tm + rng.uniform(-1.0, 1.0, size=n)
        td_ini = tm + rng.uniform(-2.0, 2.0, size=n)
        hair = rng.uniform(0.3, 0.9, size=n)
        ch = rng.uniform(0.8e-3, 1.5e-3, size=n)
        alb = rng.uniform(0.1, 0.3, size=n)
        if day:
            a_rad = rng.uniform(300.0, 600.0, size=n)
            c_rad = rng.uniform(80.0, 250.0, size=n)
            u = rng.uniform(0.5, 5.0, size=n)
        else:
            a_rad = 0.0
            c_rad = 0.0
            u = 0.0
        return {
            "ff": 2.0 * np.pi / 86400.0,
            "a_tem": a_tem,
            "a_rad": a_rad,
            "p_tem": -np.pi / 2.0 - np.pi / 6.0,
            "p_rad": -np.pi / 2.0,
            "c_tem": c_tem,
            "c_rad": c_rad,
            "td_ini": td_ini,
            "tm": tm,
            "u": u,
            "pair": const.P_AIR,
            "hair": hair,
            "t_end": 10800.0,
            "step": 3600.0,
            "ch": ch,
            "alb": alb,
            "kaps": 2.5,
            "nis": 0.8e-6,
            "tcost": 290.0,
            "etrsuetp": rng.uniform(0.4, 1.0, size=n),
        }

    def test_day_matches_numpy(self):
        rng = np.random.default_rng(42)
        kw = self._heterogeneous_inputs(128, day=True, rng=rng)
        ts_ref, td_ref, evp_ref = _energy_balance_1l_numpy(**kw)
        ts_jit, td_jit, evp_jit = energy_balance_1l(**kw)
        np.testing.assert_allclose(ts_jit, ts_ref, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(td_jit, td_ref, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(evp_jit, evp_ref, rtol=1e-6, atol=1e-12)

    def test_night_matches_numpy(self):
        rng = np.random.default_rng(7)
        kw = self._heterogeneous_inputs(64, day=False, rng=rng)
        ts_ref, td_ref, evp_ref = _energy_balance_1l_numpy(**kw)
        ts_jit, td_jit, evp_jit = energy_balance_1l(**kw)
        np.testing.assert_allclose(ts_jit, ts_ref, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(td_jit, td_ref, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(evp_jit, evp_ref, rtol=1e-6, atol=1e-12)

    def test_scalar_etrsuetp_matches_numpy(self):
        """Scalar etrsuetp (pre-pass case) must produce the same result."""
        rng = np.random.default_rng(123)
        kw = self._heterogeneous_inputs(64, day=True, rng=rng)
        kw["etrsuetp"] = 1.0
        ts_ref, td_ref, evp_ref = _energy_balance_1l_numpy(**kw)
        ts_jit, td_jit, evp_jit = energy_balance_1l(**kw)
        np.testing.assert_allclose(ts_jit, ts_ref, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(td_jit, td_ref, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(evp_jit, evp_ref, rtol=1e-6, atol=1e-12)


class TestComputeEnergyBalance1L:
    """Tests for the orchestrator (sub-period dispatch)."""

    def _make_inputs(self, n=3):
        return {
            "ts": np.full(n, 290.0),
            "td": np.full(n, 290.0),
            "td_rise": np.full(n, 290.0),
            "rs": np.full(n, 200.0),
            "u": np.full(n, 2.0),
            "tair_max": np.full(n, 295.0),
            "tair_min": np.full(n, 285.0),
            "qair": np.full(n, 0.6),
            "ch": np.full(n, 1e-3),
            "alb": np.full(n, 0.2),
            "kaps": 2.5,
            "nis": 0.8e-6,
            "tcost": 290.0,
            "pair": const.P_AIR,
            "hrise_s": 6.0 * 3600.0,
            "hset_s": 18.0 * 3600.0,
            "etrsuetp": 1.0,
            "dt": 3600.0,
        }

    def test_night_only_no_etp(self):
        """A timestep entirely before sunrise produces zero PET."""
        kw = self._make_inputs()
        kw["ctim_s"] = 2.0 * 3600.0
        kw["ftim_s"] = 3.0 * 3600.0
        ts, td, etp, td_rise = compute_energy_balance_1l(**kw)
        np.testing.assert_allclose(etp, 0.0, atol=1e-12)
        assert ts.shape == td.shape == etp.shape == td_rise.shape

    def test_daytime_produces_positive_etp(self):
        """Daytime timestep with radiation and saturated surface produces positive PET."""
        kw = self._make_inputs()
        kw["ctim_s"] = 12.0 * 3600.0
        kw["ftim_s"] = 13.0 * 3600.0
        _, _, etp, _ = compute_energy_balance_1l(**kw)
        assert np.all(etp > 0.0)

    def test_night_after_sunset_skipped_on_reentry(self):
        """Re-entry on a fully-night step (post-sunset) returns inputs unchanged."""
        kw = self._make_inputs()
        kw["ctim_s"] = 20.0 * 3600.0
        kw["ftim_s"] = 21.0 * 3600.0
        kw["reentry"] = True
        ts, td, etp, td_rise = compute_energy_balance_1l(**kw)
        np.testing.assert_array_equal(ts, kw["ts"])
        np.testing.assert_array_equal(td, kw["td"])
        np.testing.assert_allclose(etp, 0.0, atol=1e-12)

    def test_step_crossing_sunrise(self):
        """A timestep that starts in night and ends in day still produces positive PET."""
        kw = self._make_inputs()
        kw["ctim_s"] = 5.5 * 3600.0
        kw["ftim_s"] = 7.0 * 3600.0
        _, _, etp, _ = compute_energy_balance_1l(**kw)
        assert np.all(etp > 0.0)

    def test_step_crossing_sunset(self):
        """Step starting in day and ending in night still produces positive PET."""
        kw = self._make_inputs()
        kw["ctim_s"] = 17.0 * 3600.0
        kw["ftim_s"] = 19.0 * 3600.0
        _, _, etp, _ = compute_energy_balance_1l(**kw)
        assert np.all(etp > 0.0)

    def test_etrsuetp_zero_yields_no_evp_in_reentry(self):
        """Re-entry with etrsuetp=0 gives zero evaporation contribution."""
        kw = self._make_inputs()
        kw["ctim_s"] = 12.0 * 3600.0
        kw["ftim_s"] = 13.0 * 3600.0
        kw["etrsuetp"] = 0.0
        kw["reentry"] = True
        _, _, etp, _ = compute_energy_balance_1l(**kw)
        np.testing.assert_allclose(etp, 0.0, atol=1e-12)
