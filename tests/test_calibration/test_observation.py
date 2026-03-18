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
