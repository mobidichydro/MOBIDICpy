"""Tests for configuration schema validation."""

import pytest
from pydantic import ValidationError

from mobidic.config.schema import (
    EnergyParameters,
    GroundwaterParameters,
    InitialConditions,
    Multipliers,
    OutputReportSettings,
    OutputStatesSettings,
    RoutingParameters,
    SoilParameters,
    Simulation,
)


class TestSoilParameters:
    """Tests for SoilParameters validation."""

    def test_negative_wc0(self):
        """Test that negative Wc0 is rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            SoilParameters(
                Wc0=-10.0,
                Wg0=50.0,
                ks=1.0,
                kf=1e-7,
                gamma=2.689e-7,
                kappa=1.096e-7,
                beta=7.62e-6,
                alpha=2.5e-5,
            )

    def test_negative_wg0(self):
        """Test that negative Wg0 is rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            SoilParameters(
                Wc0=100.0,
                Wg0=-50.0,
                ks=1.0,
                kf=1e-7,
                gamma=2.689e-7,
                kappa=1.096e-7,
                beta=7.62e-6,
                alpha=2.5e-5,
            )

    def test_negative_ks(self):
        """Test that negative ks is rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            SoilParameters(
                Wc0=100.0,
                Wg0=50.0,
                ks=-1.0,
                kf=1e-7,
                gamma=2.689e-7,
                kappa=1.096e-7,
                beta=7.62e-6,
                alpha=2.5e-5,
            )

    def test_negative_kf(self):
        """Test that negative kf is rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            SoilParameters(
                Wc0=100.0,
                Wg0=50.0,
                ks=1.0,
                kf=-1e-7,
                gamma=2.689e-7,
                kappa=1.096e-7,
                beta=7.62e-6,
                alpha=2.5e-5,
            )

    def test_negative_gamma(self):
        """Test that negative gamma is rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            SoilParameters(
                Wc0=100.0,
                Wg0=50.0,
                ks=1.0,
                kf=1e-7,
                gamma=-2.689e-7,
                kappa=1.096e-7,
                beta=7.62e-6,
                alpha=2.5e-5,
            )

    def test_negative_kappa(self):
        """Test that negative kappa is rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            SoilParameters(
                Wc0=100.0,
                Wg0=50.0,
                ks=1.0,
                kf=1e-7,
                gamma=2.689e-7,
                kappa=-1.096e-7,
                beta=7.62e-6,
                alpha=2.5e-5,
            )

    def test_negative_beta(self):
        """Test that negative beta is rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            SoilParameters(
                Wc0=100.0,
                Wg0=50.0,
                ks=1.0,
                kf=1e-7,
                gamma=2.689e-7,
                kappa=1.096e-7,
                beta=-7.62e-6,
                alpha=2.5e-5,
            )

    def test_negative_alpha(self):
        """Test that negative alpha is rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            SoilParameters(
                Wc0=100.0,
                Wg0=50.0,
                ks=1.0,
                kf=1e-7,
                gamma=2.689e-7,
                kappa=1.096e-7,
                beta=7.62e-6,
                alpha=-2.5e-5,
            )

    def test_optional_soil_parameters_defaults(self):
        """Test that Wc0, Wg0, ks, and kf have correct defaults."""
        params = SoilParameters(
            gamma=2.689e-7,
            kappa=1.096e-7,
            beta=7.62e-6,
            alpha=2.5e-5,
        )
        assert params.Wc0 == 0.0
        assert params.Wg0 == 0.0
        assert params.ks == 1.0
        assert params.kf == 1.0e-7


class TestEnergyParameters:
    """Tests for EnergyParameters validation."""

    def test_zero_tconst(self):
        """Test that zero or negative Tconst is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            EnergyParameters(Tconst=0.0, kaps=2.5, nis=0.8e-6, CH=1e-3, Alb=0.2)

    def test_negative_tconst(self):
        """Test that negative Tconst is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            EnergyParameters(Tconst=-290.0, kaps=2.5, nis=0.8e-6, CH=1e-3, Alb=0.2)

    def test_zero_kaps(self):
        """Test that zero kaps is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            EnergyParameters(Tconst=290.0, kaps=0.0, nis=0.8e-6, CH=1e-3, Alb=0.2)

    def test_zero_nis(self):
        """Test that zero nis is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            EnergyParameters(Tconst=290.0, kaps=2.5, nis=0.0, CH=1e-3, Alb=0.2)

    def test_zero_ch(self):
        """Test that zero CH is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            EnergyParameters(Tconst=290.0, kaps=2.5, nis=0.8e-6, CH=0.0, Alb=0.2)

    def test_albedo_below_zero(self):
        """Test that albedo < 0 is rejected."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            EnergyParameters(Tconst=290.0, kaps=2.5, nis=0.8e-6, CH=1e-3, Alb=-0.1)

    def test_albedo_above_one(self):
        """Test that albedo > 1 is rejected."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            EnergyParameters(Tconst=290.0, kaps=2.5, nis=0.8e-6, CH=1e-3, Alb=1.5)

    def test_optional_energy_parameters_defaults(self):
        """Test that all energy parameters have correct defaults."""
        params = EnergyParameters()
        assert params.Tconst == 290.0
        assert params.kaps == 2.5
        assert params.nis == 0.8e-6
        assert params.CH == 1e-3
        assert params.Alb == 0.2


class TestRoutingParameters:
    """Tests for RoutingParameters validation."""

    def test_zero_wcel(self):
        """Test that zero wcel is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            RoutingParameters(method="Linear", wcel=0.0, Br0=1.0, NBr=1.5, n_Man=0.03)

    def test_zero_br0(self):
        """Test that zero Br0 is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            RoutingParameters(method="Linear", wcel=5.18, Br0=0.0, NBr=1.5, n_Man=0.03)

    def test_zero_n_man(self):
        """Test that zero n_Man is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            RoutingParameters(method="Linear", wcel=5.18, Br0=1.0, NBr=1.5, n_Man=0.0)

    def test_nbr_equals_one(self):
        """Test that NBr = 1 is rejected."""
        with pytest.raises(ValidationError, match="greater than 1"):
            RoutingParameters(method="Linear", wcel=5.18, Br0=1.0, NBr=1.0, n_Man=0.03)

    def test_nbr_less_than_one(self):
        """Test that NBr < 1 is rejected."""
        with pytest.raises(ValidationError, match="greater than 1"):
            RoutingParameters(method="Linear", wcel=5.18, Br0=1.0, NBr=0.5, n_Man=0.03)

    def test_optional_routing_parameters_defaults(self):
        """Test that Br0, NBr, and n_Man have correct defaults."""
        params = RoutingParameters(method="Linear", wcel=5.18)
        assert params.Br0 == 1.0
        assert params.NBr == 1.5
        assert params.n_Man == 0.03


class TestGroundwaterParameters:
    """Tests for GroundwaterParameters validation."""

    def test_negative_global_loss(self):
        """Test that negative global_loss is rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            GroundwaterParameters(model="Linear", global_loss=-1.0)

    def test_none_global_loss_defaults_to_zero(self):
        """Test that None global_loss defaults to 0.0."""
        params = GroundwaterParameters(model="None", global_loss=None)
        assert params.global_loss == 0.0

    def test_valid_global_loss(self):
        """Test that valid global_loss is accepted."""
        params = GroundwaterParameters(model="Linear", global_loss=0.5)
        assert params.global_loss == 0.5


class TestMultipliers:
    """Tests for Multipliers validation."""

    def test_zero_ks_factor(self):
        """Test that zero ks_factor is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            Multipliers(ks_factor=0.0)

    def test_zero_wc_factor(self):
        """Test that zero Wc_factor is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            Multipliers(Wc_factor=0.0)

    def test_zero_wg_factor(self):
        """Test that zero Wg_factor is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            Multipliers(Wg_factor=0.0)

    def test_zero_wg_wc_tr(self):
        """Test that zero Wg_Wc_tr is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            Multipliers(Wg_Wc_tr=0.0)

    def test_zero_ch_factor(self):
        """Test that zero CH_factor is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            Multipliers(CH_factor=0.0)

    def test_zero_cel_factor(self):
        """Test that zero cel_factor is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            Multipliers(cel_factor=0.0)

    def test_none_values_default_to_one(self):
        """Test that None multiplier values default to 1.0."""
        multipliers = Multipliers(ks_factor=None, Wc_factor=None)
        assert multipliers.ks_factor == 1.0
        assert multipliers.Wc_factor == 1.0


class TestInitialConditions:
    """Tests for InitialConditions validation."""

    def test_negative_ws(self):
        """Test that negative Ws is rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            InitialConditions(Ws=-0.1, Wcsat=0.3, Wgsat=0.01)

    def test_wcsat_below_zero(self):
        """Test that Wcsat < 0 is rejected."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            InitialConditions(Ws=0.0, Wcsat=-0.1, Wgsat=0.01)

    def test_wcsat_above_one(self):
        """Test that Wcsat > 1 is rejected."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            InitialConditions(Ws=0.0, Wcsat=1.5, Wgsat=0.01)

    def test_wgsat_below_zero(self):
        """Test that Wgsat < 0 is rejected."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            InitialConditions(Ws=0.0, Wcsat=0.3, Wgsat=-0.1)

    def test_wgsat_above_one(self):
        """Test that Wgsat > 1 is rejected."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            InitialConditions(Ws=0.0, Wcsat=0.3, Wgsat=1.5)

    def test_omitted_values_use_defaults(self):
        """Test that omitted values use defaults."""
        ic = InitialConditions()
        assert ic.Ws == 0.0
        assert ic.Wcsat == 0.3
        assert ic.Wgsat == 0.01


class TestSimulation:
    """Tests for Simulation validation."""

    def test_zero_timestep(self):
        """Test that zero timestep is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            Simulation(
                timestep=0.0,
                decimation=1,
                soil_scheme="Bucket",
                energy_balance="None",
            )

    def test_negative_timestep(self):
        """Test that negative timestep is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            Simulation(
                timestep=-900.0,
                decimation=1,
                soil_scheme="Bucket",
                energy_balance="None",
            )

    def test_zero_decimation(self):
        """Test that zero decimation is rejected."""
        with pytest.raises(ValidationError, match="positive integer"):
            Simulation(
                timestep=900.0,
                decimation=0,
                soil_scheme="Bucket",
                energy_balance="None",
            )

    def test_negative_decimation(self):
        """Test that negative decimation is rejected."""
        with pytest.raises(ValidationError, match="positive integer"):
            Simulation(
                timestep=900.0,
                decimation=-1,
                soil_scheme="Bucket",
                energy_balance="None",
            )


class TestOutputStatesSettings:
    """Tests for OutputStatesSettings validation."""

    def test_zero_output_interval(self):
        """Test that zero output_interval is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            OutputStatesSettings(output_interval=0.0)

    def test_negative_output_interval(self):
        """Test that negative output_interval is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            OutputStatesSettings(output_interval=-3600.0)

    def test_invalid_datetime_in_output_list(self):
        """Test that invalid datetime strings are rejected."""
        with pytest.raises(ValidationError, match="not a valid datetime"):
            OutputStatesSettings(output_states="list", output_list=["not-a-date"])

    def test_malformed_datetime_in_output_list(self):
        """Test that malformed datetime strings are rejected."""
        with pytest.raises(ValidationError, match="not a valid datetime"):
            OutputStatesSettings(output_states="list", output_list=["2024-13-45 99:99:99"])

    def test_valid_datetime_in_output_list(self):
        """Test that valid datetime strings are accepted."""
        settings = OutputStatesSettings(
            output_states="list",
            output_list=["2024-01-01 00:00:00", "2024-12-31 23:59:59"],
        )
        assert len(settings.output_list) == 2

    def test_output_list_required_when_output_states_is_list(self):
        """Test that output_list is required when output_states='list'."""
        with pytest.raises(ValidationError, match="output_list must be provided"):
            OutputStatesSettings(output_states="list", output_list=None)

    def test_output_list_required_when_output_states_is_list_empty(self):
        """Test that non-empty output_list is required when output_states='list'."""
        with pytest.raises(ValidationError, match="output_list must be provided"):
            OutputStatesSettings(output_states="list", output_list=[])

    def test_zero_flushing(self):
        """Test that zero flushing is rejected."""
        with pytest.raises(ValidationError, match="must be either -1"):
            OutputStatesSettings(flushing=0)

    def test_negative_flushing_except_minus_one(self):
        """Test that negative flushing (except -1) is rejected."""
        with pytest.raises(ValidationError, match="must be either -1"):
            OutputStatesSettings(flushing=-2)

    def test_valid_flushing_minus_one(self):
        """Test that flushing=-1 is accepted."""
        settings = OutputStatesSettings(flushing=-1)
        assert settings.flushing == -1

    def test_valid_flushing_positive(self):
        """Test that positive flushing is accepted."""
        settings = OutputStatesSettings(flushing=10)
        assert settings.flushing == 10

    def test_none_flushing_defaults_to_minus_one(self):
        """Test that None flushing defaults to -1."""
        settings = OutputStatesSettings(flushing=None)
        assert settings.flushing == -1

    def test_zero_max_file_size(self):
        """Test that zero max_file_size is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            OutputStatesSettings(max_file_size=0.0)

    def test_negative_max_file_size(self):
        """Test that negative max_file_size is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            OutputStatesSettings(max_file_size=-100.0)

    def test_none_max_file_size_defaults_to_500(self):
        """Test that None max_file_size defaults to 500.0."""
        settings = OutputStatesSettings(max_file_size=None)
        assert settings.max_file_size == 500.0


class TestOutputReportSettings:
    """Tests for OutputReportSettings validation."""

    def test_zero_report_interval(self):
        """Test that zero report_interval is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            OutputReportSettings(report_interval=0.0)

    def test_negative_report_interval(self):
        """Test that negative report_interval is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            OutputReportSettings(report_interval=-3600.0)

    def test_reach_selection_file_without_sel_file(self):
        """Test that reach_selection='file' without sel_file is rejected."""
        with pytest.raises(ValidationError, match="sel_file must be provided"):
            OutputReportSettings(reach_selection="file", sel_file=None)

    def test_reach_selection_list_without_sel_list(self):
        """Test that reach_selection='list' without sel_list is rejected."""
        with pytest.raises(ValidationError, match="sel_list must be provided"):
            OutputReportSettings(reach_selection="list", sel_list=None)

    def test_reach_selection_list_with_empty_sel_list(self):
        """Test that reach_selection='list' with empty sel_list is rejected."""
        with pytest.raises(ValidationError, match="sel_list must be provided"):
            OutputReportSettings(reach_selection="list", sel_list=[])

    def test_valid_reach_selection_file(self):
        """Test that reach_selection='file' with sel_file is accepted."""
        settings = OutputReportSettings(reach_selection="file", sel_file="reaches.json")
        assert settings.sel_file == "reaches.json"

    def test_valid_reach_selection_list(self):
        """Test that reach_selection='list' with sel_list is accepted."""
        settings = OutputReportSettings(reach_selection="list", sel_list=[1, 2, 3])
        assert settings.sel_list == [1, 2, 3]
