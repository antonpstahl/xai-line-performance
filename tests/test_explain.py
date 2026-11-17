import numpy as np

from bosch_xai.explain import by_station, failure_slices, local_cases, summarize_slice

FEATURES = ["L0_S0_F0", "L0_S0_F2", "L1_S24_F5", "duration"]


def test_by_station_sums_features():
    stations, totals = by_station(np.array([1.0, 2.0, 4.0, 8.0]), FEATURES)
    assert dict(zip(stations, totals, strict=True)) == {"L0_S0": 3, "L1_S24": 4, "part-level": 8}


def test_summarize_slice():
    shap = np.array([[0.5, 0.1, -1.0, 0.2], [0.3, 0.1, -1.0, 0.0]])
    s = summarize_slice(shap, np.array([1, 0]), np.array([True, True]), FEATURES, 1)

    assert (s["parts"], s["failures"], s["failure_rate"]) == (2, 1, 0.5)
    assert (s["flagged"], s["precision"]) == (2, 0.5)
    assert s["top_features"][0] == {"feature": "L0_S0_F0", "mean_shap": 0.4}
    assert s["top_stations"][0] == {"station": "L0_S0", "mean_shap": 0.5}
    assert s["top_stations"][-1]["station"] == "L1_S24"


def test_summarize_slice_without_flagged_parts_has_no_precision():
    s = summarize_slice(np.zeros((1, 4)), np.array([0]), np.array([False]), FEATURES, 1)
    assert s["precision"] is None


def test_failure_slices_cover_high_risk_parts():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(int)
    p = np.clip(0.5 * y + 0.5 * rng.random(400), 0, 1)
    shap = rng.normal(size=(400, 4)) + y[:, None]

    slices = failure_slices(shap, y, p, 0.6, FEATURES)

    high = p >= 0.6
    assert sum(s["parts"] for s in slices) == high.sum()
    assert sum(s["flagged"] for s in slices) == high.sum()
    assert [s["slice"] for s in slices] == list(range(1, len(slices) + 1))
    assert all(a["parts"] >= b["parts"] for a, b in zip(slices, slices[1:], strict=False))


def test_local_cases_pick_one_of_each_kind():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.4])
    assert local_cases(y, p, 0.5) == {"True positive": 0, "False positive": 2, "False negative": 4}
