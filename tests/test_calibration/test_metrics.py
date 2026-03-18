"""Tests for derived metric computation."""

import numpy as np
import pytest

from mobidic.calibration.metrics import (
    compute_metrics,
    kge,
    nse,
    nse_log,
    pbias,
    peak_error,
    rmse,
)


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
        # Sim overestimates low flows only
        sim = np.array([0.5, 0.8, 10.0, 20.0])
        # NSE_log should penalize the low-flow errors more than regular NSE
        regular = nse(sim, obs)
        log_based = nse_log(sim, obs)
        assert log_based < regular


class TestPbias:
    def test_no_bias(self):
        obs = np.array([1.0, 2.0, 3.0])
        assert pbias(obs, obs) == pytest.approx(0.0)

    def test_overestimation(self):
        obs = np.array([10.0, 20.0, 30.0])
        sim = np.array([11.0, 22.0, 33.0])
        result = pbias(sim, obs)
        assert result > 0  # Positive bias = overestimation

    def test_underestimation(self):
        obs = np.array([10.0, 20.0, 30.0])
        sim = np.array([9.0, 18.0, 27.0])
        result = pbias(sim, obs)
        assert result < 0


class TestPeakError:
    def test_no_error(self):
        obs = np.array([1.0, 5.0, 3.0])
        sim = np.array([2.0, 5.0, 4.0])
        assert peak_error(sim, obs) == pytest.approx(0.0)

    def test_overestimated_peak(self):
        obs = np.array([1.0, 5.0, 3.0])
        sim = np.array([1.0, 7.5, 3.0])  # Peak: 7.5 vs 5.0
        assert peak_error(sim, obs) == pytest.approx(0.5)

    def test_underestimated_peak(self):
        obs = np.array([1.0, 10.0, 3.0])
        sim = np.array([1.0, 5.0, 3.0])  # Peak: 5 vs 10
        assert peak_error(sim, obs) == pytest.approx(-0.5)


class TestRMSE:
    def test_perfect_match(self):
        obs = np.array([1.0, 2.0, 3.0])
        assert rmse(obs, obs) == pytest.approx(0.0)

    def test_known_error(self):
        obs = np.array([0.0, 0.0])
        sim = np.array([3.0, 4.0])
        expected = np.sqrt((9 + 16) / 2)
        assert rmse(sim, obs) == pytest.approx(expected)


class TestKGE:
    def test_perfect_match(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert kge(obs, obs) == pytest.approx(1.0)

    def test_imperfect_match(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = kge(sim, obs)
        assert result < 1.0
        assert result > 0.0  # Should still be reasonably good

    def test_constant_obs(self):
        obs = np.array([5.0, 5.0, 5.0])
        sim = np.array([4.0, 5.0, 6.0])
        assert np.isnan(kge(sim, obs))


class TestComputeMetrics:
    def test_multiple_metrics(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = obs.copy()
        results = compute_metrics(sim, obs, ["nse", "rmse", "kge"])
        assert results["nse"] == pytest.approx(1.0)
        assert results["rmse"] == pytest.approx(0.0)
        assert results["kge"] == pytest.approx(1.0)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_metrics(np.array([1.0]), np.array([1.0]), ["nonexistent"])
