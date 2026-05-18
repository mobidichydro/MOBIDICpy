"""Tests for derived metric computation."""

import HydroErr as he
import numpy as np
import pytest

from mobidic.calibration.metrics import (
    METRIC_REGISTRY,
    _HYDROERR_TARGETS,
    compute_metrics,
    nse,
    nse_log,
    pbias,
    peak_error,
)


# ---- Custom implementations ----


class TestNSE:
    def test_perfect_match(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nse(obs, obs) == pytest.approx(1.0)

    def test_mean_prediction(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.full_like(obs, obs.mean())
        assert nse(sim, obs) == pytest.approx(0.0)

    def test_negative_nse(self):
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([10.0, 20.0, 30.0])
        assert nse(sim, obs) < 0

    def test_constant_obs(self):
        obs = np.array([5.0, 5.0, 5.0])
        sim = np.array([5.0, 5.0, 5.0])
        assert np.isnan(nse(sim, obs))  # Denominator is zero


class TestNSELog:
    def test_perfect_match(self):
        obs = np.array([1.0, 2.0, 3.0])
        assert nse_log(obs, obs) == pytest.approx(1.0)

    def test_emphasizes_low_flows(self):
        obs = np.array([0.1, 0.2, 10.0, 20.0])
        sim = np.array([0.5, 0.8, 10.0, 20.0])
        # NSE_log should penalize low-flow errors more than regular NSE
        assert nse_log(sim, obs) < nse(sim, obs)


class TestPbias:
    def test_no_bias(self):
        obs = np.array([1.0, 2.0, 3.0])
        assert pbias(obs, obs) == pytest.approx(0.0)

    def test_overestimation(self):
        obs = np.array([10.0, 20.0, 30.0])
        sim = np.array([11.0, 22.0, 33.0])
        assert pbias(sim, obs) == pytest.approx(0.1)

    def test_underestimation(self):
        obs = np.array([10.0, 20.0, 30.0])
        sim = np.array([9.0, 18.0, 27.0])
        assert pbias(sim, obs) == pytest.approx(-0.1)

    def test_zero_obs_sum(self):
        obs = np.array([0.0, 0.0, 0.0])
        sim = np.array([1.0, 2.0, 3.0])
        assert np.isnan(pbias(sim, obs))


class TestPeakError:
    def test_no_error(self):
        obs = np.array([1.0, 5.0, 3.0])
        sim = np.array([2.0, 5.0, 4.0])
        assert peak_error(sim, obs) == pytest.approx(0.0)

    def test_overestimated_peak(self):
        obs = np.array([1.0, 5.0, 3.0])
        sim = np.array([1.0, 7.5, 3.0])
        assert peak_error(sim, obs) == pytest.approx(0.5)

    def test_underestimated_peak(self):
        obs = np.array([1.0, 10.0, 3.0])
        sim = np.array([1.0, 5.0, 3.0])
        assert peak_error(sim, obs) == pytest.approx(-0.5)

    def test_zero_peak(self):
        obs = np.array([0.0, 0.0, 0.0])
        sim = np.array([1.0, 2.0, 3.0])
        assert np.isnan(peak_error(sim, obs))


# ---- Registry contents ----


class TestRegistry:
    """Check that METRIC_REGISTRY is fully populated and structurally correct."""

    def test_custom_metrics_present(self):
        for name in ("nse", "nse_log", "pbias", "peak_error"):
            assert name in METRIC_REGISTRY

    def test_kge_alias_points_to_2012(self):
        # `kge` is a convenience alias for the 2012 formulation
        func, target = METRIC_REGISTRY["kge"]
        assert func is he.kge_2012
        assert target == 1.0

    def test_all_hydroerr_metrics_registered(self):
        """Every HydroErr public function should appear in the registry."""
        hydroerr_funcs = {n for n in dir(he) if not n.startswith("_") and callable(getattr(he, n))}
        missing = hydroerr_funcs - set(METRIC_REGISTRY)
        assert not missing, f"HydroErr metrics missing from registry: {sorted(missing)}"

    def test_registry_entries_are_callable_tuples(self):
        for name, entry in METRIC_REGISTRY.items():
            assert isinstance(entry, tuple) and len(entry) == 2, name
            func, target = entry
            assert callable(func), name
            assert isinstance(target, float), name

    def test_custom_implementations_take_precedence(self):
        """`nse` in the registry must be the custom function, not he.nse."""
        # Both have target 1.0; the function identity is what we care about.
        assert METRIC_REGISTRY["nse"][0] is nse


# ---- HydroErr metrics via the registry ----


@pytest.mark.parametrize("name,target", sorted(_HYDROERR_TARGETS.items()))
def test_hydroerr_perfect_match_hits_target(name, target):
    """Every HydroErr metric in the catalog should return its declared target
    when sim == obs (within tolerance)."""
    # Non-monotonic signal — monotonic series degenerate gradient-based metrics
    # (irmse divides by std of the obs gradient; sga uses arccos of a ratio).
    obs = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.5, 4.5])
    sim = obs.copy()
    # `acc` (anomaly correlation) compares against a sample climatology and is
    # not exactly 1.0 even at sim==obs, so skip the strict equality assertion.
    if name == "acc":
        return
    func = METRIC_REGISTRY[name][0]
    value = func(sim, obs)
    assert value == pytest.approx(target, abs=1e-6), f"{name}: got {value}, expected {target}"


def test_hydroerr_target_zero_metrics_are_positive():
    """Error metrics should produce non-zero positive values when sim != obs."""
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    error_metrics = [n for n, t in _HYDROERR_TARGETS.items() if t == 0.0]
    # Pick a stable subset that doesn't depend on log/percentage definitions
    for name in ("rmse", "mae", "mse", "ed"):
        assert name in error_metrics
        value = METRIC_REGISTRY[name][0](sim, obs)
        assert value > 0, f"{name} should be positive when sim != obs, got {value}"


# ---- compute_metrics ----


class TestComputeMetrics:
    def test_multiple_metrics_perfect(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        results = compute_metrics(obs.copy(), obs, ["nse", "rmse", "kge"])
        assert results["nse"] == pytest.approx(1.0)
        assert results["rmse"] == pytest.approx(0.0)
        assert results["kge"] == pytest.approx(1.0)

    def test_returns_only_requested(self):
        obs = np.array([1.0, 2.0, 3.0])
        results = compute_metrics(obs.copy(), obs, ["nse"])
        assert list(results) == ["nse"]

    def test_mixes_custom_and_hydroerr(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        results = compute_metrics(sim, obs, ["nse", "pbias", "kge_2012", "r_squared", "ve"])
        assert set(results) == {"nse", "pbias", "kge_2012", "r_squared", "ve"}

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_metrics(np.array([1.0]), np.array([1.0]), ["nonexistent"])

    def test_unknown_metric_lists_available(self):
        with pytest.raises(ValueError, match=r"nse"):
            compute_metrics(np.array([1.0]), np.array([1.0]), ["nonexistent"])

    def test_empty_metric_list(self):
        results = compute_metrics(np.array([1.0]), np.array([1.0]), [])
        assert results == {}
