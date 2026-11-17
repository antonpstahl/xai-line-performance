import zipfile

import kagglehub
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
