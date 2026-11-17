"""Command line entry point: bosch-xai download | prepare | train | explain."""

import argparse

from bosch_xai import data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bosch-xai", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="create folders and fetch the Kaggle data")
    prepare = commands.add_parser("prepare", help="build the feature table")
    prepare.add_argument(
        "--sample", action="store_true", help="all failures + 10%% of other parts (seed 42)"
    )
    args = parser.parse_args(argv)

    if args.command == "download":
        data.download()
    elif args.command == "prepare":
        data.prepare(sample=args.sample)
