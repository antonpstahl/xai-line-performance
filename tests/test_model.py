import numpy as np
import polars as pl
from sklearn.metrics import matthews_corrcoef

from bosch_xai.model import best_threshold, feature_names, time_split


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    p = np.round(np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1), 2)  # rounded to create ties

    threshold, mcc = best_threshold(y, p)

    brute = max(matthews_corrcoef(y, p >= t) for t in np.unique(p))
    assert np.isclose(mcc, brute)
    assert np.isclose(matthews_corrcoef(y, p >= threshold), mcc)


def test_best_threshold_perfect_separation():
    y = np.array([0, 0, 1, 1])
    assert best_threshold(y, np.array([0.1, 0.2, 0.8, 0.9])) == (0.8, 1.0)


def test_time_split_holds_out_latest_parts():
    df = pl.DataFrame({"Id": range(10), "start_time": [None, 5, 4, 3, 2, 1, 0, 9, 8, 7]})
    dev, holdout = time_split(df)
    assert holdout["start_time"].to_list() == [8, 9]
    assert dev["start_time"].null_count() == 1 and len(dev) == 8


def test_start_time_is_sort_key_not_feature():
    df = pl.DataFrame({"Id": [1], "Response": [0], "start_time": [1.0], "duration": [2.0]})
    assert feature_names(df) == ["duration"]
