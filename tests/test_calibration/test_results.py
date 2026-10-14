"""Tests for CalibrationResults parsing of PEST++ output files (glm, ies, sen)."""

import pytest

from mobidic.calibration.config import CalibrationConfig
from mobidic.calibration.results import CalibrationResults


def _make_config(pest_tool: str) -> CalibrationConfig:
    return CalibrationConfig(
        mobidic_config="Arno.yaml",
        pest_tool=pest_tool,
        case_name="calib",
        parameters=[
            {
                "name": "ks_factor",
                "parameter_key": "parameters.multipliers.ks_factor",
                "initial_value": 1.0,
                "lower_bound": 0.01,
                "upper_bound": 100.0,
                "transform": "log",
            }
        ],
        observations=[{"name": "Q1", "obs_file": "obs.csv", "reach_id": 1, "value_column": "Q"}],
    )


def _results(tmp_path, pest_tool: str) -> CalibrationResults:
    return CalibrationResults.from_pest_output(tmp_path, _make_config(pest_tool))


class TestOptimalParameters:
    def test_reads_last_par_file(self, tmp_path):
        """The last .par file is used and PEST's lowercase names map back to parameter_key."""
        (tmp_path / "calib.1.par").write_text("single point\nks_factor 2.0 1.0 0.0\n", encoding="utf-8")
        (tmp_path / "calib.2.par").write_text("single point\nKS_FACTOR 3.5 1.0 0.0\n", encoding="utf-8")

        assert _results(tmp_path, "glm").get_optimal_parameters() == {"parameters.multipliers.ks_factor": 3.5}

    def test_unknown_parameter_raises(self, tmp_path):
        (tmp_path / "calib.par").write_text("single point\nother 2.0 1.0 0.0\n", encoding="utf-8")

        with pytest.raises(KeyError, match="'other' has no matching entry"):
            _results(tmp_path, "glm").get_optimal_parameters()

    def test_no_par_and_no_pst_returns_empty(self, tmp_path):
        assert _results(tmp_path, "glm").get_optimal_parameters() == {}


class TestObjectiveFunctionHistory:
    def test_glm_iobj(self, tmp_path):
        (tmp_path / "calib.iobj").write_text("iteration,model_runs_completed,total_phi\n0,1,100.0\n1,5,40.0\n")

        df = _results(tmp_path, "glm").get_objective_function_history()

        assert list(df.columns) == ["iteration", "phi"]
        assert df["phi"].tolist() == [100.0, 40.0]

    def test_ies_phi(self, tmp_path):
        (tmp_path / "calib.phi.actual.csv").write_text(
            "iteration,total_runs,mean,standard_deviation,min,max,r0,r1\n0,2,10.0,1.0,9.0,11.0,9.0,11.0\n"
        )

        df = _results(tmp_path, "ies").get_objective_function_history()

        assert df["phi"].tolist() == [10.0]
        assert df["std"].tolist() == [1.0]

    def test_rec_phi(self, tmp_path):
        (tmp_path / "calib.rec").write_text(
            "header\n   Starting phi for this iteration: 120.5\n   no number here\n   Total phi = 80.25\n"
        )

        df = _results(tmp_path, "sen").get_objective_function_history()

        assert df["iteration"].tolist() == [1, 2]
        assert df["phi"].tolist() == [120.5, 80.25]

    @pytest.mark.parametrize("tool", ["glm", "ies", "sen"])
    def test_missing_file_returns_none(self, tmp_path, tool):
        assert _results(tmp_path, tool).get_objective_function_history() is None


class TestResiduals:
    def test_reads_last_rei(self, tmp_path):
        (tmp_path / "calib.1.rei").write_text(
            "MODEL OUTPUTS\n Name Group Measured Modelled Residual Weight\n"
            " q_0001 flow 10.0 8.0 2.0 1.0\n q_0002 flow 12.0 13.0 -1.0 0.5\n short line\n"
        )

        df = _results(tmp_path, "glm").get_residuals()

        assert df["obs_name"].tolist() == ["q_0001", "q_0002"]
        assert df["residual"].tolist() == [2.0, -1.0]
        assert df["weight"].tolist() == [1.0, 0.5]

    def test_rei_without_data_returns_none(self, tmp_path):
        (tmp_path / "calib.rei").write_text("MODEL OUTPUTS\n Name Group Measured Modelled Residual Weight\n")
        assert _results(tmp_path, "glm").get_residuals() is None

    def test_missing_rei_returns_none(self, tmp_path):
        assert _results(tmp_path, "glm").get_residuals() is None


class TestSensitivities:
    def test_morris(self, tmp_path):
        (tmp_path / "calib.msn").write_text(
            "par_name,n_samples,sen_mean,sen_mean_abs,sen_std_dev\nks_factor,4,0.5,0.7,0.1\n"
        )

        df = _results(tmp_path, "sen").get_parameter_sensitivities()

        assert df["sen_mean_abs"].tolist() == [0.7]

    def test_missing_files_return_none(self, tmp_path):
        assert _results(tmp_path, "sen").get_parameter_sensitivities() is None
        assert _results(tmp_path, "glm").get_parameter_sensitivities() is None

    def test_unsupported_tool_returns_none(self, tmp_path):
        assert _results(tmp_path, "ies").get_parameter_sensitivities() is None


class TestEnsembleResults:
    def test_prior_and_posterior(self, tmp_path):
        for it, value in ((0, 1.0), (2, 3.0)):
            (tmp_path / f"calib.{it}.par.csv").write_text(f"real_name,ks_factor\nbase,{value}\n")
            (tmp_path / f"calib.{it}.obs.csv").write_text(f"real_name,q_0001\nbase,{10 * value}\n")

        res = _results(tmp_path, "ies").get_ensemble_results()

        assert res["prior_parameters"]["ks_factor"].tolist() == [1.0]
        assert res["posterior_parameters"]["ks_factor"].tolist() == [3.0]
        assert res["prior_observations"]["q_0001"].tolist() == [10.0]
        assert res["posterior_observations"]["q_0001"].tolist() == [30.0]

    def test_no_files_returns_none(self, tmp_path):
        assert _results(tmp_path, "ies").get_ensemble_results() is None
