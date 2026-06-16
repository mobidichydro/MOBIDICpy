"""``mobidic calibration`` - run PEST++ calibration from a calibration config."""

from argparse import _SubParsersAction
from pathlib import Path

from loguru import logger

from mobidic.cli._common import CLIError
from mobidic.cli._common import add_log_level_arg
from mobidic.cli._common import configure_cli_logging


def register(subparsers: _SubParsersAction) -> None:
    """Register the ``calibration`` subcommand."""
    parser = subparsers.add_parser(
        "calibration",
        help="Run PEST++ calibration from a calibration config file.",
        description="Set up a PEST++ working directory from a calibration YAML config and "
        "(optionally) run the configured PEST++ tool. Requires the 'calibration' extras "
        "(pip install mobidicpy[calibration]) and PEST++ executables on PATH.",
    )
    parser.add_argument("config", help="Path to the calibration YAML configuration file.")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run GIS preprocessing for the specified MOBIDIC config file before setup.",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only generate the PEST++ working directory; do not run PEST++.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: from config or all CPUs).",
    )
    add_log_level_arg(parser)
    parser.set_defaults(handler=run)


def _import_calibration():
    """Import calibration entry points, raising an error if extras are missing."""
    try:
        from mobidic.calibration import PestSetup
        from mobidic.calibration import load_calibration_config

        return PestSetup, load_calibration_config
    except ImportError as exc:
        raise CLIError(
            "Calibration support is not installed. Install it with: pip install mobidicpy[calibration]"
        ) from exc


def _maybe_preprocess(calib_config, base_path: Path, log_level: str | None) -> None:
    """Run preprocessing for the MOBIDIC config referenced by the calibration config."""
    from mobidic import load_config
    from mobidic import run_preprocessing
    from mobidic import save_gisdata
    from mobidic import save_network

    mobidic_config_path = Path(calib_config.mobidic_config)
    if not mobidic_config_path.is_absolute():
        mobidic_config_path = base_path / mobidic_config_path

    config = load_config(mobidic_config_path)
    configure_cli_logging(config, log_level)

    gisdata = run_preprocessing(config)
    save_gisdata(gisdata, config.paths.gisdata)
    save_network(gisdata.network, config.paths.network, format="parquet")
    logger.success("Preprocessing complete for calibration run.")


def run(args) -> int:
    """Execute the calibration command."""
    pest_setup_cls, load_calibration_config = _import_calibration()

    configure_cli_logging(log_level=args.log_level)

    config_path = Path(args.config).resolve()
    calib_config = load_calibration_config(config_path)

    if args.preprocess:
        _maybe_preprocess(calib_config, config_path.parent, args.log_level)

    pest = pest_setup_cls(config_path)
    working_dir = pest.setup()
    logger.success(f"PEST++ working directory ready: {working_dir}")

    if args.setup_only:
        logger.info("--setup-only given; skipping PEST++ execution.")
        return 0

    results = pest.run(num_workers=args.workers)

    optimal = results.get_optimal_parameters()
    if optimal:
        logger.info("Optimal parameters:")
        for name, value in optimal.items():
            logger.info(f"  {name}: {value:.6g}")

    logger.success("Calibration complete.")
    return 0
