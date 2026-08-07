"""Tests for observation loading and temporal alignment."""

import numpy as np
import pandas as pd
import pytest

from mobidic.calibration.config import ObservationGroup
from mobidic.calibration.observation import align_observations_to_simulation, load_observations


class TestLoadObservations:
    def _make_obs_csv(self, tmp_path, filename="obs.csv"):
        obs_path = tmp_path / filename
        obs_path.write_text(
            "time,Q_329\n"
            "2023-11-01 00:00:00,10.0\n"
            "2023-11-01 00:15:00,12.5\n"
            "2023-11-01 00:30:00,15.0\n"
            "2023-11-01 00:45:00,20.0\n"
            "2023-11-01 01:00:00,18.0\n"
        )
        return obs_path

    def test_load_basic(self, tmp_path):
        self._make_obs_csv(tmp_path)
        og = ObservationGroup(
            name="Q_329",
            obs_file="obs.csv",
            reach_id=329,
            value_column="Q_329",
        )
        df = load_observations(og, tmp_path)
        assert len(df) == 5
        assert "time" in df.columns
        assert "value" in df.columns
        assert df["value"].iloc[0] == 10.0

    def test_load_with_date_filter(self, tmp_path):
        self._make_obs_csv(tmp_path)
        og = ObservationGroup(
            name="Q_329",
            obs_file="obs.csv",
            reach_id=329,
            value_column="Q_329",
        )
        df = load_observations(og, tmp_path, start_date="2023-11-01 00:15:00", end_date="2023-11-01 00:45:00")
        assert len(df) == 3
        assert df["value"].iloc[0] == 12.5

    def test_load_missing_file(self, tmp_path):
        og = ObservationGroup(
            name="Q_329",
            obs_file="nonexistent.csv",
            reach_id=329,
            value_column="Q_329",
        )
        with pytest.raises(FileNotFoundError):
            load_observations(og, tmp_path)

    def test_load_missing_column(self, tmp_path):
        (tmp_path / "obs.csv").write_text("time,wrong_col\n2023-01-01,1.0\n")
        og = ObservationGroup(
            name="Q_329",
            obs_file="obs.csv",
            reach_id=329,
            value_column="Q_329",
        )
        with pytest.raises(ValueError, match="Value column"):
            load_observations(og, tmp_path)


class TestAlignObservations:
    def test_exact_match(self):
        obs_df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-11-01 00:00:00", "2023-11-01 00:15:00", "2023-11-01 00:30:00"]),
                "value": [10.0, 12.5, 15.0],
            }
        )
        sim_times = pd.date_range("2023-11-01", periods=3, freq="15min")

        sim_idx, obs_vals, obs_times = align_observations_to_simulation(obs_df, sim_times)
        assert len(sim_idx) == 3
        np.testing.assert_array_equal(sim_idx, [0, 1, 2])
        np.testing.assert_array_almost_equal(obs_vals, [10.0, 12.5, 15.0])

    def test_nearest_neighbor(self):
        obs_df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-11-01 00:07:00"]),  # 7 min offset
                "value": [10.0],
            }
        )
        sim_times = pd.date_range("2023-11-01", periods=4, freq="15min")

        sim_idx, obs_vals, _ = align_observations_to_simulation(obs_df, sim_times, tolerance_seconds=600)
        assert len(sim_idx) == 1
        assert sim_idx[0] == 0  # Nearest to 00:00

    def test_out_of_tolerance(self):
        obs_df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-11-01 01:00:00"]),  # 1 hour away
                "value": [10.0],
            }
        )
        sim_times = pd.date_range("2023-11-01", periods=2, freq="15min")  # 00:00, 00:15

        sim_idx, obs_vals, _ = align_observations_to_simulation(obs_df, sim_times, tolerance_seconds=120)
        assert len(sim_idx) == 0

    def test_empty_observations(self):
        obs_df = pd.DataFrame({"time": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float)})
        sim_times = pd.date_range("2023-11-01", periods=4, freq="15min")

        sim_idx, obs_vals, _ = align_observations_to_simulation(obs_df, sim_times)
        assert len(sim_idx) == 0


# ---- Observation error model ----


class TestObservationWeights:
    def _group(self, **overrides):
        from mobidic.calibration.config import ObservationGroup

        defaults = {"name": "Q", "obs_file": "obs.csv", "reach_id": 1, "value_column": "Q"}
        defaults.update(overrides)
        return ObservationGroup(**defaults)

    def test_constant_weight_by_default(self):
        from mobidic.calibration.observation import observation_weights

        w = observation_weights([1.0, 500.0], self._group(weight=2.5))
        np.testing.assert_allclose(w, [2.5, 2.5])

    def test_relative_error_gives_one_over_sigma(self):
        from mobidic.calibration.observation import observation_weights

        group = self._group(relative_error=0.1, min_error=5.0)
        w = observation_weights([470.9, 122.7], group)
        np.testing.assert_allclose(w, [1 / 47.09, 1 / 12.27], rtol=1e-12)

    def test_min_error_floors_the_weight_at_low_flow(self):
        from mobidic.calibration.observation import observation_weights

        group = self._group(relative_error=0.1, min_error=5.0)
        # 10% of 8.5 is 0.85, below the 5.0 floor
        np.testing.assert_allclose(observation_weights([8.5], group), [0.2])
        # A zero observation would otherwise give an infinite weight
        assert np.isfinite(observation_weights([0.0], group)).all()

    def test_high_flow_gets_less_weight_than_low_flow(self):
        from mobidic.calibration.observation import observation_weights

        group = self._group(relative_error=0.1, min_error=5.0)
        w = observation_weights([471.0, 8.5], group)
        assert w[0] < w[1]

    def test_negative_values_use_the_magnitude(self):
        from mobidic.calibration.observation import observation_weights

        group = self._group(relative_error=0.1, min_error=1.0)
        np.testing.assert_allclose(observation_weights([-100.0], group), observation_weights([100.0], group))

    def test_relative_error_requires_a_positive_floor(self):
        import pytest as _pytest
        from pydantic import ValidationError

        with _pytest.raises(ValidationError, match="min_error must be > 0"):
            self._group(relative_error=0.1)

    def test_relative_error_must_be_positive(self):
        import pytest as _pytest
        from pydantic import ValidationError

        with _pytest.raises(ValidationError):
            self._group(relative_error=0.0, min_error=1.0)
