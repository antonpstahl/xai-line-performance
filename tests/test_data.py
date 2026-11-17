import zipfile

import kagglehub
import polars as pl
import pytest

from bosch_xai import data


def write_zip(path, name, text="Id,Response\n1,0\n"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, text)


def test_extract_zips_only_wanted_and_missing(tmp_path):
    write_zip(tmp_path / "train_numeric.csv.zip", "train_numeric.csv")
    write_zip(tmp_path / "test_numeric.csv.zip", "test_numeric.csv")
    (tmp_path / "train_date.csv").write_text("already here")
    write_zip(tmp_path / "train_date.csv.zip", "train_date.csv")

    data.extract_zips(tmp_path)

    assert (tmp_path / "train_numeric.csv").read_text().startswith("Id,Response")
    assert not (tmp_path / "test_numeric.csv").exists()
    assert (tmp_path / "train_date.csv").read_text() == "already here"


def test_download_places_files_and_skips_second_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_download(handle, output_dir):
        calls.append((handle, output_dir))
        for name in data.RAW_FILES:
            write_zip(tmp_path / output_dir / f"{name}.zip", name)
        return output_dir

    monkeypatch.setattr(kagglehub, "competition_download", fake_download)
    data.download()
    data.download()

    assert calls == [(data.COMPETITION, "data/raw")]
    for folder in (data.RAW, data.PROCESSED, data.MODELS, data.REPORTS):
        assert (tmp_path / folder).is_dir()
    for name in data.RAW_FILES:
        assert (tmp_path / "data/raw" / name).exists()


def test_download_explains_missing_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fail(handle, output_dir):
        raise kagglehub.exceptions.UnauthenticatedError()

    monkeypatch.setattr(kagglehub, "competition_download", fail)
    with pytest.raises(SystemExit) as exc:
        data.download()
    assert data.RULES_URL in str(exc.value) and "KAGGLE_API_TOKEN" in str(exc.value)


def test_prepare_collapses_stations(tmp_path):
    nan = ""
    (tmp_path / "train_numeric.csv").write_text(
        "Id,L0_S0_F0,L0_S1_F2,Response\n1,0.5,,0\n2,,1.5,1\n3,0.1,0.2,0\n"
    )
    (tmp_path / "train_date.csv").write_text(
        f"Id,L0_S0_D1,L0_S0_D3,L0_S1_D5\n1,10.0,10.0,{nan}\n2,{nan},{nan},20.5\n3,1.0,1.0,4.0\n"
    )
    out = data.prepare(sample=False, raw=tmp_path, out=tmp_path / "f.parquet")
    df = pl.read_parquet(out)

    assert df["Id"].to_list() == [1, 2, 3]
    assert df["L0_S0_time"].to_list() == [10.0, None, 1.0]
    assert df["L0_S1_time"].to_list() == [None, 20.5, 4.0]
    assert df["L0_S0_visited"].to_list() == [1, 0, 1]
    assert df["start_time"].to_list() == [10.0, 20.5, 1.0]
    assert df["duration"].to_list() == [0.0, 0.0, 3.0]
    assert pl.read_parquet_metadata(out)["sample"] == "false"


def test_prepare_sample_keeps_all_failures(tmp_path):
    rows = "\n".join(f"{i},{i},{int(i % 50 == 0)}" for i in range(1, 1001))
    (tmp_path / "train_numeric.csv").write_text(f"Id,L0_S0_F0,Response\n{rows}\n")
    dates = "\n".join(f"{i},{i}" for i in range(1, 1001))
    (tmp_path / "train_date.csv").write_text(f"Id,L0_S0_D1\n{dates}\n")

    df = pl.read_parquet(data.prepare(sample=True, raw=tmp_path, out=tmp_path / "f.parquet"))

    assert df["Response"].sum() == 20
    assert len(df) == 20 + 98  # exactly 10% of the 980 non-failures


def test_station_of():
    assert data.station_of("L3_S32_F3850") == "L3_S32"
    assert data.station_of("L0_S5_visited") == "L0_S5"
    assert data.station_of("duration") == "part-level"
