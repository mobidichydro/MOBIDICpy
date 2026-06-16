"""Tests for the top-level CLI parser and dispatch."""

import pytest

from mobidic import __version__
from mobidic.cli._common import CLIError
from mobidic.cli._common import parse_date
from mobidic.cli.main import build_parser
from mobidic.cli.main import main

EXPECTED_COMMANDS = ["preprocess", "simulation", "calibration", "hyetograph", "convert-meteo", "check"]


class TestParser:
    """Tests for parser construction."""

    def test_all_commands_registered(self):
        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
        for command in EXPECTED_COMMANDS:
            assert command in subparsers_action.choices

    def test_each_command_sets_handler(self):
        parser = build_parser()
        for command in EXPECTED_COMMANDS:
            args = (
                parser.parse_args([command, "dummy_arg"])
                if command != "convert-meteo"
                else parser.parse_args([command, "in.mat", "out.nc"])
            )
            assert callable(args.handler)

    def test_version(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--version"])
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestDispatch:
    """Tests for main() dispatch and exit codes."""

    def test_no_command_returns_1(self):
        assert main([]) == 1

    def test_missing_config_returns_1(self):
        # FileNotFoundError from load_config is caught and converted to exit code 1.
        assert main(["check", "does_not_exist.yaml"]) == 1

    def test_check_valid_config_returns_0(self, config_file):
        assert main(["check", str(config_file)]) == 0

    def test_simulation_without_preprocessed_data_returns_1(self, config_file):
        # gisdata/network do not exist and --preprocess not given -> CLIError -> 1.
        assert main(["simulation", str(config_file)]) == 1


class TestCommonHelpers:
    """Tests for shared helpers in _common."""

    def test_parse_date_valid(self):
        dt = parse_date("2023-11-01")
        assert dt.year == 2023 and dt.month == 11 and dt.day == 1

    def test_parse_date_invalid_raises_clierror(self):
        with pytest.raises(CLIError):
            parse_date("not-a-date")
