"""Command line entry point: bosch-xai download | prepare | train | explain."""

import argparse

from bosch_xai import data, explain, model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bosch-xai", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="create folders and fetch the Kaggle data")
    prepare = commands.add_parser("prepare", help="build the feature table")
    prepare.add_argument(
        "--sample", action="store_true", help="all failures + 10%% of other parts (seed 42)"
    )
    commands.add_parser("train", help="cross-validate, tune the threshold, score the holdout")
    explain_cmd = commands.add_parser("explain", help="SHAP plots, local cases, failure slices")
    explain_cmd.add_argument(
        "--llm", action="store_true", help="add Anthropic-written slice summaries (extra 'llm')"
    )
    args = parser.parse_args(argv)

    if args.command == "download":
        data.download()
    elif args.command == "prepare":
        data.prepare(sample=args.sample)
    elif args.command == "train":
        model.train()
    elif args.command == "explain":
        explain.explain(use_llm=args.llm)
