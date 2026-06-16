"""Tests for individual CLI subcommands (argument parsing)."""

import pytest

from mobidic.cli.main import build_parser


class TestArgumentParsing:
    """Per-command argument parsing (no model execution)."""

    def test_preprocess_force_flag(self):
        args = build_parser().parse_args(["preprocess", "cfg.yaml", "--force"])
        assert args.config == "cfg.yaml"
        assert args.force is True

    def test_simulation_overrides(self):
        args = build_parser().parse_args(
            ["simulation", "cfg.yaml", "--start", "2023-11-01", "--end", "2023-11-02", "--preprocess"]
        )
        assert args.start == "2023-11-01"
        assert args.end == "2023-11-02"
        assert args.preprocess is True

    def test_convert_meteo_args(self):
        args = build_parser().parse_args(["convert-meteo", "in.mat", "out.nc", "--basin", "Arno"])
        assert args.input == "in.mat"
        assert args.output == "out.nc"
        assert args.basin == "Arno"

    def test_hyetograph_start(self):
        args = build_parser().parse_args(["hyetograph", "cfg.yaml", "--start", "2000-01-01"])
        assert args.start == "2000-01-01"

    def test_calibration_flags(self):
        args = build_parser().parse_args(["calibration", "calib.yaml", "--setup-only", "--workers", "4"])
        assert args.setup_only is True
        assert args.workers == 4

    def test_invalid_log_level_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["check", "cfg.yaml", "--log-level", "TRACE"])
