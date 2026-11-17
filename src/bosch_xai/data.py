"""Paths, Kaggle download and feature preparation."""

import sys
import zipfile
from pathlib import Path

COMPETITION = "bosch-production-line-performance"
RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"
RAW = Path("data/raw")
PROCESSED = Path("data/processed")
MODELS = Path("models")
REPORTS = Path("reports")
FEATURES = PROCESSED / "features.parquet"
RAW_FILES = ("train_numeric.csv", "train_date.csv", "train_categorical.csv")
SEED = 42


def make_dirs() -> None:
    """Create the runtime folders; safe to call repeatedly."""
    for folder in (RAW, PROCESSED, MODELS, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)


def extract_zips(raw: Path, names: tuple[str, ...] = RAW_FILES) -> None:
    """Extract `<name>.zip` for every wanted CSV that is not extracted yet.

    The competition bundle ships each CSV as its own zip; test files are left packed.
    """
    for name in names:
        archive = raw / f"{name}.zip"
        if archive.exists() and not (raw / name).exists():
            print(f"Extracting {archive} ...")
            with zipfile.ZipFile(archive) as zf:
                zf.extract(name, raw)


def download() -> None:
    """Create the folders and fetch the competition data into data/raw."""
    make_dirs()
    extract_zips(RAW)  # also covers zips a user placed there by hand
    if all((RAW / name).exists() for name in RAW_FILES):
        print(f"All files already in {RAW}, nothing to download.")
        return

    import kagglehub
    from kagglehub.exceptions import KaggleApiHTTPError, UnauthenticatedError

    try:
        # With output_dir kagglehub writes into data/raw instead of its cache: no 14 GB duplicate.
        kagglehub.competition_download(COMPETITION, output_dir=str(RAW))
    except (UnauthenticatedError, KaggleApiHTTPError) as err:
        sys.exit(
            f"Download failed: {err}\n"
            f"1. Accept the competition rules: {RULES_URL}\n"
            "2. Create a token under Kaggle > Settings > API and either export it as\n"
            "   KAGGLE_API_TOKEN or save kaggle.json to ~/.kaggle/kaggle.json."
        )
    extract_zips(RAW)
    print(f"Data ready in {RAW}.")
