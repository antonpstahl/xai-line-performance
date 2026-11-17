"""Paths, Kaggle download and feature preparation."""

import re
import sys
import zipfile
from pathlib import Path

import polars as pl

COMPETITION = "bosch-production-line-performance"
RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"
RAW = Path("data/raw")
PROCESSED = Path("data/processed")
MODELS = Path("models")
REPORTS = Path("reports")
FEATURES = PROCESSED / "features.parquet"
RAW_FILES = ("train_numeric.csv", "train_date.csv", "train_categorical.csv")
SEED = 42
STATION = re.compile(r"^(L\d+_S\d+)_")


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


def station_of(feature: str) -> str:
    """Station of a column like L3_S32_F3850 or L3_S32_time; 'part-level' for the rest."""
    match = STATION.match(feature)
    return match.group(1) if match else "part-level"


def date_features(date_columns: list[str]) -> tuple[list[pl.Expr], list[pl.Expr]]:
    """Expressions for one timestamp per station, then part start/end/duration and visit flags.

    All date columns of a station hold the same value, so the minimum collapses them losslessly
    while ignoring missing entries.
    """
    by_station: dict[str, list[str]] = {}
    for col in date_columns:
        if col != "Id":
            by_station.setdefault(station_of(col), []).append(col)
    station_times = [pl.min_horizontal(cols).alias(f"{s}_time") for s, cols in by_station.items()]
    times = [f"{s}_time" for s in by_station]
    visited = [
        pl.col(f"{s}_time").is_not_null().cast(pl.Int8).alias(f"{s}_visited") for s in by_station
    ]
    start, end = pl.min_horizontal(times), pl.max_horizontal(times)
    derived = [start.alias("start_time"), end.alias("end_time"), (end - start).alias("duration")]
    return station_times, derived + visited


def scan(path: Path) -> pl.LazyFrame:
    """Lazy CSV scan with an explicit float32 schema (halves memory, skips type inference)."""
    columns = pl.read_csv(path, n_rows=0).columns
    schema = {c: pl.Float32 for c in columns} | {"Id": pl.Int64}
    if "Response" in schema:
        schema["Response"] = pl.Int8
    return pl.scan_csv(path, schema=schema)


def prepare(sample: bool, raw: Path = RAW, out: Path = FEATURES) -> Path:
    """Join numeric features with date-derived features and write them to parquet.

    With `sample`, keep all failures and 10% of the other parts for fast iteration.
    """
    numeric, dates = scan(raw / "train_numeric.csv"), scan(raw / "train_date.csv")
    ids = numeric.select("Id", "Response").collect().sort("Id")
    if sample:
        ok = ids.filter(pl.col("Response") == 0).sample(fraction=0.1, seed=SEED)
        ids = pl.concat([ids.filter(pl.col("Response") == 1), ok])
    keep = ids.lazy().select("Id")

    station_times, derived = date_features(dates.collect_schema().names())
    date_part = dates.join(keep, on="Id", how="semi").select("Id", *station_times)
    features = (
        numeric.join(keep, on="Id", how="semi")
        .join(date_part.with_columns(derived), on="Id")
        .sort("Id")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    features.sink_parquet(out, metadata={"sample": str(sample).lower()})
    print(f"Wrote {len(ids)} parts ({ids['Response'].sum()} failures) to {out}.")
    return out
