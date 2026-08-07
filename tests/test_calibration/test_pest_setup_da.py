"""Tests for the PESTPP-DA path in PestSetup (no PEST++ execution)."""

import numpy as np
import pandas as pd
import pytest

from mobidic.calibration.config import CalibrationConfig
from mobidic.calibration.da_cycles import CYCLE_PARAM_NAME, build_cycle_schedule, build_parameter_cycle_table
from mobidic.calibration.da_states import STATE_ID_OBS_NAME, STATE_ID_PAR_NAME
from mobidic.calibration.pest_setup import PEST_TOOL_MAP, PestSetup, _all_states_enabled

DT = 900


def _make_da_config(**overrides) -> CalibrationConfig:
    defaults = {
        "mobidic_config": "Arno.yaml",
        "pest_tool": "da",
        "simulation_period": {"start_date": "2023-11-01 00:00:00", "end_date": "2023-11-01 01:45:00"},
        "da": {
            "cycle_length": "1h",
            "warmup_period": {"start_date": "2023-10-25 00:00:00", "end_date": "2023-11-01 00:00:00"},
        },
        "parameters": [
            {
                "name": "ks_factor",
                "parameter_key": "parameters.multipliers.ks_factor",
                "initial_value": 1.0,
                "lower_bound": 0.01,
                "upper_bound": 100.0,
                "transform": "log",
            }
        ],
        "observations": [
            {"name": "Q1", "obs_file": "obs.csv", "reach_id": 1, "value_column": "Q"},
        ],
    }
    defaults.update(overrides)
    return CalibrationConfig(**defaults)


def _prepared_setup(tmp_path, cfg):
    """A PestSetup with the bookkeeping _build_da_pst() needs, without running setup()."""
    setup = PestSetup(cfg, base_path=tmp_path)
    setup._working_dir = tmp_path / "wd"
    setup._working_dir.mkdir(exist_ok=True)

    schedule = build_cycle_schedule(
        cfg.simulation_period.start_date, cfg.simulation_period.end_date, cfg.da.cycle_length, DT
    )
    setup._cycle_schedule = schedule
    setup._n_obs_per_group = {"Q1": schedule.n_steps_per_cycle}
    setup._obs_data = [{"name": "Q1", "reach_id": 1}]
    setup._cycle_table_names = {
        "da_observation_cycle_table": "obs_cycle_table.csv",
        "da_weight_cycle_table": "weight_cycle_table.csv",
        "da_parameter_cycle_table": "par_cycle_table.csv",
    }

    names = [f"Q1_{i:04d}" for i in range(schedule.n_steps_per_cycle)]
    obs_table = pd.DataFrame(1.0, index=names, columns=list(range(schedule.n_cycles)))
    weight_table = obs_table.copy()
    return setup, schedule, obs_table, weight_table


def test_tool_map_has_da():
    assert PEST_TOOL_MAP["da"] == "pestpp-da"


def test_pest_exe_is_da(tmp_path):
    setup = PestSetup(_make_da_config(), base_path=tmp_path)
    assert setup._pest_exe == "pestpp-da"


def test_all_states_enabled_turns_every_flag_on():
    from mobidic.config.schema import MOBIDICConfig, OutputStates

    config = MOBIDICConfig.model_construct(output_states=OutputStates(discharge=False, soil_capillary=False))
    enabled = _all_states_enabled(config)

    for field in type(enabled).model_fields:
        assert getattr(enabled, field) is True, field
    # The original config is untouched
    assert config.output_states.discharge is False


def test_check_state_dir_writable_accepts_a_normal_directory(tmp_path):
    PestSetup._check_state_dir_writable(tmp_path)
    assert not (tmp_path / ".write_probe").exists()


def test_verify_warmup_state_reports_missing_variables(tmp_path):
    import xarray as xr

    path = tmp_path / "warmup_state.nc"
    xr.Dataset({"Wc": (("y", "x"), np.zeros((2, 2)))}).to_netcdf(path)

    with pytest.raises(ValueError, match="missing required variable"):
        PestSetup._verify_warmup_state(path, has_reservoirs=False)


def test_verify_warmup_state_accepts_a_complete_file(tmp_path):
    import xarray as xr

    grid = (("y", "x"), np.zeros((2, 2)))
    reach = (("reach",), np.zeros(3))
    path = tmp_path / "warmup_state.nc"
    xr.Dataset(
        {
            "Wc": grid,
            "Wg": grid,
            "Ws": grid,
            "flr": grid,
            "fld": grid,
            "discharge": reach,
            "lateral_inflow": reach,
        }
    ).to_netcdf(path)

    assert PestSetup._verify_warmup_state(path, has_reservoirs=False) == path


def test_verify_warmup_state_finds_the_chunked_file(tmp_path):
    import xarray as xr

    grid = (("y", "x"), np.zeros((2, 2)))
    reach = (("reach",), np.zeros(3))
    xr.Dataset(
        {
            "Wc": grid,
            "Wg": grid,
            "Ws": grid,
            "flr": grid,
            "fld": grid,
            "discharge": reach,
            "lateral_inflow": reach,
        }
    ).to_netcdf(tmp_path / "warmup_state_001.nc")

    found = PestSetup._verify_warmup_state(tmp_path / "warmup_state.nc", has_reservoirs=False)
    assert found.name == "warmup_state_001.nc"


class TestBuildDaPst:
    def test_enks_pst_has_cycle_columns_and_no_state_parameters(self, tmp_path):
        pytest.importorskip("pyemu")
        cfg = _make_da_config()
        setup, schedule, obs_table, weight_table = _prepared_setup(tmp_path, cfg)

        obs_names = [f"Q1_{i:04d}" for i in range(schedule.n_steps_per_cycle)]
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)

        assert (pst.parameter_data["cycle"] == -1).all()
        assert (pst.observation_data["cycle"] == -1).all()
        assert CYCLE_PARAM_NAME in pst.parameter_data.index
        assert pst.parameter_data.loc[CYCLE_PARAM_NAME, "partrans"] == "fixed"
        # No dynamic states are declared with restart_from='warmup'
        assert STATE_ID_PAR_NAME not in pst.parameter_data.index
        assert STATE_ID_OBS_NAME not in pst.observation_data.index
        assert (pst.observation_data["state_par_link"] == "").all()
        # da_use_simulated_states must stay at its default: setting it to false
        # selects formulation 3/4, and PESTPP-DA's sanity check then demands
        # final-to-initial linkages in the parameter data that we do not provide.
        assert "da_use_simulated_states" not in pst.pestpp_options

    def test_cycle_tables_are_registered_as_options(self, tmp_path):
        pytest.importorskip("pyemu")
        cfg = _make_da_config()
        setup, schedule, obs_table, weight_table = _prepared_setup(tmp_path, cfg)

        pst = setup._build_da_pst(list(obs_table.index), obs_table, weight_table)

        assert pst.pestpp_options["da_observation_cycle_table"] == "obs_cycle_table.csv"
        assert pst.pestpp_options["da_weight_cycle_table"] == "weight_cycle_table.csv"
        assert pst.pestpp_options["da_parameter_cycle_table"] == "par_cycle_table.csv"

    def test_hotstart_and_stop_cycle_are_forwarded(self, tmp_path):
        pytest.importorskip("pyemu")
        cfg = _make_da_config(
            da={
                "cycle_length": "1h",
                "warmup_period": {"start_date": "2023-10-25 00:00:00", "end_date": "2023-11-01 00:00:00"},
                "hotstart_cycle": 1,
                "stop_cycle": 3,
            }
        )
        setup, _, obs_table, weight_table = _prepared_setup(tmp_path, cfg)

        pst = setup._build_da_pst(list(obs_table.index), obs_table, weight_table)

        assert pst.pestpp_options["da_hotstart_cycle"] == 1
        assert pst.pestpp_options["da_stop_cycle"] == 3

    def test_enkf_pst_declares_the_state_identifier(self, tmp_path):
        pytest.importorskip("pyemu")
        cfg = _make_da_config(
            da={"cycle_length": "1h", "states": {"restart_from": "previous_cycle"}},
        )
        setup, _, obs_table, weight_table = _prepared_setup(tmp_path, cfg)

        obs_names = list(obs_table.index) + [STATE_ID_OBS_NAME]
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)

        assert pst.parameter_data.loc[STATE_ID_PAR_NAME, "partrans"] == "fixed"
        assert pst.parameter_data.loc[STATE_ID_PAR_NAME, "parval1"] == -1.0
        # Wide bounds so da_enforce_bounds cannot clip the identifier
        assert pst.parameter_data.loc[STATE_ID_PAR_NAME, "parlbnd"] <= -1.0
        assert pst.parameter_data.loc[STATE_ID_PAR_NAME, "parubnd"] >= 1.0e6
        assert pst.observation_data.loc[STATE_ID_OBS_NAME, "state_par_link"] == STATE_ID_PAR_NAME
        assert pst.observation_data.loc[STATE_ID_OBS_NAME, "weight"] == 0.0
        assert "da_use_simulated_states" not in pst.pestpp_options

    def test_cycle_table_observations_are_activated(self, tmp_path):
        pytest.importorskip("pyemu")
        cfg = _make_da_config()
        setup, _, obs_table, weight_table = _prepared_setup(tmp_path, cfg)

        pst = setup._build_da_pst(list(obs_table.index), obs_table, weight_table)

        # The control-file weight is an activation flag; the real weights live
        # in the weight cycle table.
        for name in obs_table.index:
            assert pst.observation_data.loc[name, "weight"] == 1.0
            assert pst.observation_data.loc[name, "cycle"] == -1

    def test_cycle_columns_survive_pst_write(self, tmp_path):
        """pyemu must emit the extra DataFrame columns into the external CSVs."""
        pytest.importorskip("pyemu")
        cfg = _make_da_config(
            da={"cycle_length": "1h", "states": {"restart_from": "previous_cycle"}},
        )
        setup, _, obs_table, weight_table = _prepared_setup(tmp_path, cfg)
        obs_names = list(obs_table.index) + [STATE_ID_OBS_NAME]
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)

        pst_path = setup.working_dir / "assimilation.pst"
        pst.write(str(pst_path), version=2)

        par_csv = next(setup.working_dir.glob("*.par_data.csv"))
        obs_csv = next(setup.working_dir.glob("*.obs_data.csv"))
        par_df = pd.read_csv(par_csv)
        obs_df = pd.read_csv(obs_csv)

        assert "cycle" in par_df.columns
        assert "cycle" in obs_df.columns
        assert "state_par_link" in obs_df.columns
        assert (par_df["cycle"] == -1).all()
        link = obs_df.set_index("obsnme").loc[STATE_ID_OBS_NAME, "state_par_link"]
        assert link == STATE_ID_PAR_NAME

    def test_instruction_names_match_the_control_file_order(self, tmp_path):
        pytest.importorskip("pyemu")
        from mobidic.calibration.instruction import generate_instruction_file

        cfg = _make_da_config(
            da={"cycle_length": "1h", "states": {"restart_from": "previous_cycle"}},
        )
        setup, _, obs_table, weight_table = _prepared_setup(tmp_path, cfg)

        _, obs_names = generate_instruction_file(
            cfg,
            setup._n_obs_per_group,
            setup.working_dir / "model_output.csv.ins",
            extra_obs_names=setup._da_state_obs_names(),
        )
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)

        assert list(pst.observation_data.index) == obs_names


class TestJointStateParameterPst:
    """Formulation 2: the state parameters become adjustable and carry a link."""

    def _setup_with_states(self, tmp_path, n_reaches=3, parameters=None, **da_overrides):
        import geopandas as gpd
        from shapely.geometry import LineString

        from mobidic.calibration.da_states import build_discharge_state_spec

        da = {
            "cycle_length": "1h",
            "warmup_period": {"start_date": "2023-10-25 00:00:00", "end_date": "2023-11-01 00:00:00"},
            "states": {"restart_from": "previous_cycle", "estimate": ["discharge"]},
        }
        da.update(da_overrides)
        cfg = _make_da_config(da=da, **({"parameters": parameters} if parameters else {}))
        setup, _, obs_table, weight_table = _prepared_setup(tmp_path, cfg)

        # A chain: 0 -> 1 -> 2 -> ... so the gauge at reach 1 has 0 and 1 upstream.
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": list(range(n_reaches)),
                "upstream_1": [float("nan")] + [float(i - 1) for i in range(1, n_reaches)],
                "upstream_2": [float("nan")] * n_reaches,
                "geometry": [LineString([(0, i), (1, i)]) for i in range(n_reaches)],
            }
        )
        setup._network = network
        setup._state_spec = build_discharge_state_spec(
            network=network,
            initial_discharge=np.arange(1.0, n_reaches + 1.0),
            reference_discharge=np.arange(10.0, 10.0 * (n_reaches + 1), 10.0),
        )
        return setup, obs_table, weight_table

    def test_state_parameters_are_adjustable_and_linked(self, tmp_path):
        pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_states(tmp_path)
        spec = setup._state_spec

        obs_names = list(obs_table.index) + setup._da_state_obs_names()
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)

        for par, obs, initial, upper in zip(spec.par_names, spec.obs_names, spec.initial, spec.upper):
            # 'none', never 'log': a discharge state is legitimately zero.
            assert pst.parameter_data.loc[par, "partrans"] == "none"
            assert pst.parameter_data.loc[par, "parval1"] == pytest.approx(initial)
            # Positive, never 0.0: PEST++ rejects a subnormal parameter value and
            # a zero lower bound lets the ensemble update produce one.
            assert pst.parameter_data.loc[par, "parlbnd"] > 0.0
            assert pst.parameter_data.loc[par, "parubnd"] == pytest.approx(upper)
            assert pst.parameter_data.loc[par, "cycle"] == -1
            assert pst.observation_data.loc[obs, "state_par_link"] == par
            assert pst.observation_data.loc[obs, "weight"] == 0.0

    def test_state_parameters_are_counted_in_maxsing(self, tmp_path):
        """A maxsing of 4 would truncate the update to four singular values."""
        pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_states(tmp_path, n_reaches=5)

        obs_names = list(obs_table.index) + setup._da_state_obs_names()
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)

        assert pst.svd_data.maxsing == len(pst.adj_par_names)
        assert pst.svd_data.maxsing >= 5

    def test_the_state_identifier_stays_fixed(self, tmp_path):
        """Only the physical states are estimated; the file identifier never is."""
        pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_states(tmp_path)

        obs_names = list(obs_table.index) + setup._da_state_obs_names()
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)

        assert pst.parameter_data.loc[STATE_ID_PAR_NAME, "partrans"] == "fixed"
        assert STATE_ID_PAR_NAME not in pst.adj_par_names

    def test_observation_order_is_identifier_then_states(self, tmp_path):
        """The .ins order fixes the tail of model_output.csv; both must agree."""
        setup, _, _ = self._setup_with_states(tmp_path)
        assert setup._da_state_obs_names() == [STATE_ID_OBS_NAME] + setup._state_spec.obs_names

    def test_prior_ensemble_keeps_the_state_spread_small(self, tmp_path):
        """The bounds must span a flood peak, so PEST++'s own draw would be noise."""
        pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_states(tmp_path)
        setup.calib_config.pest_options["ies_num_reals"] = 40
        spec = setup._state_spec

        obs_names = list(obs_table.index) + setup._da_state_obs_names()
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)
        path = setup._write_prior_parameter_ensemble(pst, setup.working_dir)

        ensemble = pd.read_csv(path, index_col=0)
        assert pst.pestpp_options["ies_par_en"] == path.name
        assert len(ensemble) == 40

        prior_std = setup.calib_config.da.states.prior_std
        for name, initial in zip(spec.par_names, spec.initial):
            column = ensemble[name]
            assert column.mean() == pytest.approx(initial, abs=0.5 * initial + 0.1)
            # Whereas a bounds-derived draw would give sigma = upper / 4.
            assert column.std() < 3.0 * max(prior_std * initial, setup.calib_config.da.states.prior_std_floor)

    def test_prior_ensemble_accepts_a_mixed_case_parameter_name(self, tmp_path):
        """pyemu's Cov lower-cases its names, so the control file must agree.

        A parameter spelled 'Wc_factor' in the YAML used to reach
        ``pst.parameter_data`` with that spelling while the Cov drawn from it
        held 'wc_factor', and the draw failed with "cov names are not in
        mean_values".
        """
        pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_states(
            tmp_path,
            parameters=[
                {
                    "name": "Wc_factor",
                    "parameter_key": "parameters.multipliers.Wc_factor",
                    "initial_value": 1.0,
                    "lower_bound": 0.1,
                    "upper_bound": 10.0,
                    "transform": "log",
                }
            ],
        )

        obs_names = list(obs_table.index) + setup._da_state_obs_names()
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)
        assert "wc_factor" in pst.adj_par_names

        ensemble = pd.read_csv(setup._write_prior_parameter_ensemble(pst, setup.working_dir), index_col=0)
        assert "wc_factor" in ensemble.columns

    def test_prior_ensemble_carries_the_fixed_interface_parameters(self, tmp_path):
        """PESTPP-DA reads cycle_num and sp_state_id from the ensemble too."""
        pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_states(tmp_path)

        obs_names = list(obs_table.index) + setup._da_state_obs_names()
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)
        ensemble = pd.read_csv(setup._write_prior_parameter_ensemble(pst, setup.working_dir), index_col=0)

        assert (ensemble[STATE_ID_PAR_NAME] == -1.0).all()
        assert CYCLE_PARAM_NAME in ensemble.columns

    def test_bounds_enforcement_turns_a_denormal_state_into_a_normal_one(self, tmp_path):
        """The mechanism the positive lower bound relies on.

        PEST++ refuses to queue a run whose parameter vector holds a subnormal
        double (`add_runs() error: denormal values for realization ...`), and the
        ensemble update produces them once a reach's discharge decays far enough.
        With parlbnd > 0, bounds enforcement clips them back into normal range
        before the run is queued.
        """
        pyemu = pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_states(tmp_path)

        obs_names = list(obs_table.index) + setup._da_state_obs_names()
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)

        denormal = 2.9954111345e-314  # the value PEST++ rejected in practice
        assert 0.0 < denormal < np.finfo(np.float64).tiny  # subnormal by construction

        name = setup._state_spec.par_names[0]
        df = pd.DataFrame([pst.parameter_data["parval1"].astype(float)], index=["0"])
        df.loc["0", name] = denormal
        ensemble = pyemu.ParameterEnsemble(pst=pst, df=df)

        ensemble.enforce()

        enforced = float(ensemble._df.loc["0", name])
        assert enforced == pytest.approx(setup.calib_config.da.states.state_floor)
        assert enforced >= np.finfo(np.float64).tiny  # normal, so add_runs accepts it

    def test_an_explicit_localizer_is_not_overwritten(self, tmp_path):
        pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_states(tmp_path)
        setup.calib_config.pest_options["da_localizer"] = "loc.mat"

        obs_names = list(obs_table.index) + setup._da_state_obs_names()
        pst = setup._build_da_pst(obs_names, obs_table, weight_table)

        assert pst.pestpp_options["da_localizer"] == "loc.mat"
        assert not (setup.working_dir / "localizer.csv").exists()


class TestUpstreamLocalizer:
    """The topology localizer.

    Same construction as private/pestpp-da/localizer/ for pestpp-ies: the
    upstream adjacency decides which per-reach parameters an observation may
    update, global parameters get 1.0 everywhere, and the result is written as a
    PEST ASCII matrix through pyemu and registered as the localizer option.
    """

    def _build(self, tmp_path, n_reaches=5, **states):
        pytest.importorskip("pyemu")
        overrides = {}
        if states:
            overrides["states"] = states
        setup, obs_table, weight_table = TestJointStateParameterPst()._setup_with_states(
            tmp_path, n_reaches=n_reaches, **overrides
        )
        obs_names = list(obs_table.index) + setup._da_state_obs_names()
        return setup, setup._build_da_pst(obs_names, obs_table, weight_table)

    @staticmethod
    def _matrix(setup):
        import pyemu

        return pyemu.Matrix.from_ascii(str(setup.working_dir / "loc.mat")).to_dataframe()

    def test_a_gauge_may_only_update_its_own_upstream_reaches(self, tmp_path):
        setup, pst = self._build(tmp_path)

        assert pst.pestpp_options["da_localizer"] == "loc.mat"
        assert (setup.working_dir / "loc.mat").exists()

        matrix = self._matrix(setup)
        matrix.columns = [c.lower() for c in matrix.columns]
        # The group is gauged at reach 1 of the chain 0 -> 1 -> ... -> 4, so only
        # reaches 0 and 1 drain into it.
        assert matrix.loc[:, ["sp_q_0000", "sp_q_0001"]].to_numpy().min() == 1.0
        assert matrix.loc[:, ["sp_q_0002", "sp_q_0003", "sp_q_0004"]].to_numpy().max() == 0.0

    def test_every_adjustable_parameter_is_a_column(self, tmp_path):
        """PESTPP-DA treats an adjustable parameter missing from the localizer as fixed."""
        setup, pst = self._build(tmp_path)
        columns = {c.lower() for c in self._matrix(setup).columns}
        assert {p.lower() for p in pst.adj_par_names} == columns

    def test_non_state_parameters_are_never_localized(self, tmp_path):
        """ks_factor is basin-wide; every gauge must be able to update it."""
        setup, _ = self._build(tmp_path)
        matrix = self._matrix(setup)
        matrix.columns = [c.lower() for c in matrix.columns]
        assert (matrix["ks_factor"] == 1.0).all()

    def test_rows_are_group_names_not_individual_observations(self, tmp_path):
        """Per-cycle weights change which observations are active; group names do not.

        A row naming an observation that the weight cycle table has zeroed for the
        current cycle is a hard error in Localizer::process_mat(). Group names are
        expanded to whatever is non-zero-weighted in that cycle instead. This is
        the one place the DA localizer has to differ from the IES one.
        """
        setup, _ = self._build(tmp_path)
        rows = [r.lower() for r in self._matrix(setup).index]
        assert rows == [g.name.lower() for g in setup.calib_config.observations]
        assert not any(r.startswith("q1_") for r in rows)

    def test_localizer_none_skips_the_matrix(self, tmp_path):
        setup, pst = self._build(tmp_path, restart_from="previous_cycle", estimate=["discharge"], localizer="none")
        assert "da_localizer" not in pst.pestpp_options
        assert not (setup.working_dir / "loc.mat").exists()

    def test_no_localizer_without_estimated_states(self, tmp_path):
        """Formulation 1 has nothing to localize."""
        pytest.importorskip("pyemu")
        cfg = _make_da_config(da={"cycle_length": "1h", "states": {"restart_from": "previous_cycle"}})
        setup, _, obs_table, weight_table = _prepared_setup(tmp_path, cfg)

        pst = setup._build_da_pst(list(obs_table.index) + [STATE_ID_OBS_NAME], obs_table, weight_table)

        assert "da_localizer" not in pst.pestpp_options

    def test_extension_is_one_pestpp_accepts(self, tmp_path):
        """Mat::from_file() dispatches on the extension and raises on anything else."""
        setup, _ = self._build(tmp_path)
        assert (setup.working_dir / "loc.mat").suffix.upper().lstrip(".") in {"JCB", "JCO", "MAT", "COV", "CSV"}


class TestSoilStatesInTheControlFile:
    """Formulation 2 with zone-averaged Wc/Wg alongside the discharge states."""

    def _setup_with_soil(self, tmp_path, n_reaches=3, **da_states):
        import geopandas as gpd
        from shapely.geometry import LineString
        from types import SimpleNamespace

        from mobidic.calibration.da_states import (
            KIND_SOIL_CAPILLARY,
            KIND_SOIL_GRAVITATIONAL,
            StateSpec,
            build_discharge_state_spec,
            build_reach_zone_map,
            build_soil_state_spec,
        )

        states = {"restart_from": "previous_cycle", "estimate": ["discharge", "soil_moisture"]}
        states.update(da_states)
        cfg = _make_da_config(
            da={
                "cycle_length": "1h",
                "warmup_period": {"start_date": "2023-10-25 00:00:00", "end_date": "2023-11-01 00:00:00"},
                "states": states,
            }
        )
        setup, _, obs_table, weight_table = _prepared_setup(tmp_path, cfg)

        network = gpd.GeoDataFrame(
            {
                "mobidic_id": list(range(n_reaches)),
                "upstream_1": [float("nan")] + [float(i - 1) for i in range(1, n_reaches)],
                "upstream_2": [float("nan")] * n_reaches,
                "downstream": [float(i + 1) for i in range(n_reaches - 1)] + [float("nan")],
                "geometry": [LineString([(0, i), (1, i)]) for i in range(n_reaches)],
            }
        )
        setup._network = network

        # One zone per reach: a 1 x n_reaches grid whose cells drain to 0, 1, ...
        hillslope = np.arange(n_reaches, dtype=np.int64).reshape(1, n_reaches)
        gisdata = SimpleNamespace(
            grids={"dtm": np.full((1, n_reaches), 100.0)},
            hillslope_reach_map=hillslope,
            network=network,
        )
        zone_map, zone_ids = build_reach_zone_map(gisdata)
        capacity = np.full((1, n_reaches), 0.02)

        setup._state_spec = StateSpec.combine(
            build_discharge_state_spec(
                network=network,
                initial_discharge=np.arange(1.0, n_reaches + 1.0),
                reference_discharge=np.full(n_reaches, 10.0),
            ),
            build_soil_state_spec(
                kinds=[KIND_SOIL_CAPILLARY, KIND_SOIL_GRAVITATIONAL],
                state=SimpleNamespace(wc=0.6 * capacity, wg=0.2 * capacity),
                capacities={KIND_SOIL_CAPILLARY: capacity, KIND_SOIL_GRAVITATIONAL: capacity},
                zone_map=zone_map,
                zone_ids=zone_ids,
            ),
        )
        return setup, obs_table, weight_table

    def _pst(self, setup, obs_table, weight_table):
        obs_names = list(obs_table.index) + setup._da_state_obs_names()
        return setup._build_da_pst(obs_names, obs_table, weight_table)

    def test_each_store_gets_its_own_group_and_saturation_bounds(self, tmp_path):
        pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_soil(tmp_path)
        pst = self._pst(setup, obs_table, weight_table)

        for block in setup._state_spec.soil_blocks:
            for par in block.par_names:
                row = pst.parameter_data.loc[par]
                assert row["partrans"] == "none"
                assert row["pargp"] == block.group
                # Physical bounds: a saturation cannot leave [0, 1], so PESTPP-DA
                # enforcing them can never truncate the transferred state.
                assert row["parubnd"] == pytest.approx(1.0)
                assert 0.0 < row["parlbnd"] < 1.0
                assert row["cycle"] == -1

        assert {b.group for b in setup._state_spec.soil_blocks} == {"state_wc", "state_wg"}

    def test_every_soil_state_is_linked_to_its_simulated_value(self, tmp_path):
        pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_soil(tmp_path)
        pst = self._pst(setup, obs_table, weight_table)

        links = pst.observation_data["state_par_link"]
        for block in setup._state_spec.soil_blocks:
            for par, obs in zip(block.par_names, block.obs_names):
                assert links.loc[obs] == par
        # Every link must resolve, or PESTPP-DA transfers a state to nothing.
        declared = [v for v in links if isinstance(v, str) and v]
        assert set(declared) <= set(pst.parameter_data.index)

    def test_the_interface_order_matches_the_instruction_file(self, tmp_path):
        """model_output.csv is written in this order; the .ins must read it in the same one."""
        from mobidic.calibration.instruction import generate_instruction_file

        setup, obs_table, weight_table = self._setup_with_soil(tmp_path)
        _, obs_names = generate_instruction_file(
            setup.calib_config,
            setup._n_obs_per_group,
            setup.working_dir / "model_output.csv.ins",
            extra_obs_names=setup._da_state_obs_names(),
        )

        spec = setup._state_spec
        assert obs_names[-len(spec) :] == spec.obs_names
        assert obs_names[-len(spec) - 1] == STATE_ID_OBS_NAME

    def test_the_prior_spread_is_absolute_for_a_saturation(self, tmp_path):
        """prior_std scales a discharge; a bounded fraction needs a fixed spread instead."""
        pytest.importorskip("pyemu")
        setup, obs_table, weight_table = self._setup_with_soil(tmp_path)
        setup.calib_config.pest_options["ies_num_reals"] = 60
        pst = self._pst(setup, obs_table, weight_table)

        path = setup._write_prior_parameter_ensemble(pst, setup.working_dir)
        ensemble = pd.read_csv(path, index_col=0)

        expected = setup.calib_config.da.states.saturation_prior_std
        for block in setup._state_spec.soil_blocks:
            for par, initial in zip(block.par_names, block.initial):
                column = ensemble[par]
                assert column.mean() == pytest.approx(initial, abs=0.5 * expected + 0.02)
                # Whereas a bounds-derived draw would give sigma = 1/4.
                assert column.std() < 3.0 * expected

    def test_min_zone_cells_shrinks_the_interface(self, tmp_path):
        from types import SimpleNamespace

        from mobidic.calibration.da_states import build_reach_zone_map

        network = self._setup_with_soil(tmp_path)[0]._network
        gisdata = SimpleNamespace(
            grids={"dtm": np.full((1, 3), 100.0)},
            hillslope_reach_map=np.array([[0, 1, 2]], dtype=np.int64),
            network=network,
        )

        _, all_zones = build_reach_zone_map(gisdata)
        _, merged = build_reach_zone_map(gisdata, min_zone_cells=2)

        assert len(all_zones) == 3
        assert len(merged) < 3

    def test_a_gauge_may_update_the_soil_zones_draining_into_it(self, tmp_path):
        pytest.importorskip("pyemu")
        import pyemu

        setup, obs_table, weight_table = self._setup_with_soil(tmp_path)
        self._pst(setup, obs_table, weight_table)

        matrix = pyemu.Matrix.from_ascii(str(setup.working_dir / "loc.mat")).to_dataframe()
        matrix.columns = [c.lower() for c in matrix.columns]
        # The gauge sits at reach 1 of the chain 0 -> 1 -> 2.
        assert matrix.loc[:, ["sp_wc_0000", "sp_wc_0001"]].to_numpy().min() == 1.0
        assert matrix.loc[:, ["sp_wc_0002", "sp_wg_0002"]].to_numpy().max() == 0.0


def test_parameter_cycle_table_never_repeats(tmp_path):
    schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 05:45:00", "1h", DT)
    values = build_parameter_cycle_table(schedule).loc[CYCLE_PARAM_NAME].to_numpy()
    assert np.all(np.diff(values) != 0)
