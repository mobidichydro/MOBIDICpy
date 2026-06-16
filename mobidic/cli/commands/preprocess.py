"""``mobidic preprocess`` - run GIS preprocessing and save consolidated data."""

from argparse import _SubParsersAction
from pathlib import Path

from loguru import logger

from mobidic.cli._common import add_log_level_arg
from mobidic.cli._common import configure_cli_logging


def register(subparsers: _SubParsersAction) -> None:
    """Register the ``preprocess`` subcommand."""
    parser = subparsers.add_parser(
        "preprocess",
        help="Run GIS preprocessing and save consolidated gisdata and network.",
        description="Run the GIS preprocessing workflow and write the consolidated "
        "gisdata (NetCDF) and river network (GeoParquet) to the paths in the config.",
    )
    parser.add_argument("config", help="Path to the MOBIDIC YAML configuration file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run preprocessing even if gisdata and network already exist.",
    )
    add_log_level_arg(parser)
    parser.set_defaults(handler=run)


def run(args) -> int:
    """Execute the preprocess command."""
    from mobidic import load_config
    from mobidic import load_gisdata
    from mobidic import run_preprocessing
    from mobidic import save_gisdata
    from mobidic import save_network

    config = load_config(args.config)
    configure_cli_logging(config, args.log_level)

    gisdata_path = Path(config.paths.gisdata)
    network_path = Path(config.paths.network)

    if not args.force and gisdata_path.exists() and network_path.exists():
        logger.info("Preprocessed data already exists; skipping (use --force to re-run):")
        logger.info(f"  gisdata: {gisdata_path}")
        logger.info(f"  network: {network_path}")
        gisdata = load_gisdata(gisdata_path, network_path)
        logger.info(f"Grid size: {gisdata.metadata['shape']}, reaches: {len(gisdata.network)}")
        return 0

    gisdata = run_preprocessing(config)

    save_gisdata(gisdata, gisdata_path)
    save_network(gisdata.network, network_path, format="parquet")

    logger.success(f"GIS data saved to: {gisdata_path}")
    logger.success(f"Network saved to: {network_path}")
    logger.info(f"Grid size: {gisdata.metadata['shape']}, reaches: {len(gisdata.network)}")
    return 0
