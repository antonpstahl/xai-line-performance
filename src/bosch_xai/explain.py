"""TreeSHAP on the holdout: global importance, three local cases and failure slices."""

import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # files only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans

from bosch_xai.data import FEATURES, REPORTS, SEED, station_of
from bosch_xai.model import METRICS, MODEL, matrix, time_split

N_SLICES = 5
HIGH_RISK_SHARE = 0.02  # high risk = flagged by the threshold or in the top 2% of scores
LOCAL_TOP = 8


def top(values: np.ndarray, names: list[str], n: int) -> dict[str, float]:
    """The n largest values by name, descending, rounded for reporting."""
    return {names[i]: round(float(values[i]), 3) for i in np.argsort(-values, kind="stable")[:n]}


def by_station(values: np.ndarray, features: list[str]) -> tuple[list[str], np.ndarray]:
    """Sum per-feature values per station."""
    stations = sorted({station_of(f) for f in features})
    index = {s: i for i, s in enumerate(stations)}
    totals = np.zeros(len(stations))
    np.add.at(totals, [index[station_of(f)] for f in features], values)
    return stations, totals


def barh(values: dict[str, float], title: str, xlabel: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.25 * len(values)), facecolor="#fcfcfb")
    ax.barh(list(values)[::-1], list(values.values())[::-1], color="#2a78d6", height=0.7)
    ax.set_facecolor("#fcfcfb")
    ax.set_title(title, loc="left", color="#0b0b0b")
    ax.set_xlabel(xlabel, color="#52514e")
    ax.tick_params(colors="#52514e", length=0)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(y=0.01)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize_slice(
    shap: np.ndarray, y: np.ndarray, flagged: np.ndarray, features: list[str], slice_id: int
) -> dict:
    """Size, outcomes and the features/stations that push the slice's risk up most."""
    mean = shap.mean(axis=0)
    stations, station_mean = by_station(mean, features)
    return {
        "slice": slice_id,
        "parts": len(y),
        "failures": int(y.sum()),
        "failure_rate": round(float(y.mean()), 4),
        "flagged": int(flagged.sum()),
        "precision": round(float(y[flagged].mean()), 4) if flagged.any() else None,
        "top_features": [{"feature": f, "mean_shap": v} for f, v in top(mean, features, 5).items()],
        "top_stations": [
            {"station": s, "mean_shap": v} for s, v in top(station_mean, stations, 3).items()
        ],
    }


def failure_slices(
    shap: np.ndarray, y: np.ndarray, p: np.ndarray, threshold: float, features: list[str]
) -> list[dict]:
    """k-means on the SHAP vectors of high-risk parts; slice 1 is the largest."""
    high = p >= min(threshold, np.quantile(p, 1 - HIGH_RISK_SHARE))
    shap, y, flagged = shap[high], y[high], p[high] >= threshold
    labels = KMeans(min(N_SLICES, len(y)), random_state=SEED, n_init=10).fit_predict(shap)
    by_size = np.argsort(-np.bincount(labels), kind="stable")
    base_rate = round(float(y.mean()), 4)
    return [
        summarize_slice(shap[labels == c], y[labels == c], flagged[labels == c], features, i)
        | {"high_risk_failure_rate": base_rate}
        for i, c in enumerate(by_size, 1)
    ]


def local_cases(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, int]:
    """Row index of the highest-scoring true positive, false positive and false negative."""
    fail, pred = y.astype(bool), p >= threshold
    masks = {
        "True positive": fail & pred,
        "False positive": ~fail & pred,
        "False negative": fail & ~pred,
    }
    return {name: int(np.flatnonzero(m)[np.argmax(p[m])]) for name, m in masks.items() if m.any()}


def render_local(cases, ids, X, y, p, shap, base, features) -> str:
    lines = ["# Local explanations (holdout)", "", "SHAP in log-odds; positive = more risk.", ""]
    for name, i in cases.items():
        lines += [
            f"## {name}: part {ids[i]}",
            "",
            f"Score {p[i]:.3f}, actual {'failure' if y[i] else 'no failure'}, "
            f"base value {base[i]:.3f}.",
            "",
            "| feature | value | SHAP |",
            "|---|---|---|",
        ]
        for j in np.argsort(-np.abs(shap[i]))[:LOCAL_TOP]:
            value = "missing" if np.isnan(X[i, j]) else f"{X[i, j]:.3f}"
            lines.append(f"| {features[j]} | {value} | {shap[i, j]:+.3f} |")
        lines.append("")
    return "\n".join(lines)


def listing(items: list[dict], key: str) -> str:
    return ", ".join(f"{t[key]} ({t['mean_shap']:+})" for t in items)


def render_slices(slices: list[dict], threshold: float) -> str:
    lines = [
        "# Failure slices (holdout)",
        "",
        f"High-risk parts (score >= {threshold:.3f} or top {HIGH_RISK_SHARE:.0%}) clustered by",
        "their SHAP vectors. `mean_shap` is the mean log-odds contribution within the slice.",
        "",
    ]
    for s in slices:
        lines += [
            f"## Slice {s['slice']}",
            "",
            f"- parts: {s['parts']}, failures: {s['failures']}, failure rate: "
            f"{s['failure_rate']} (all high-risk parts: {s['high_risk_failure_rate']})",
            f"- flagged by threshold: {s['flagged']}, precision: {s['precision']}",
            "- top features: " + listing(s["top_features"], "feature"),
            "- top stations: " + listing(s["top_stations"], "station"),
        ]
        if "llm_explanation" in s:
            issues = "; ".join(s["faithfulness_violations"])
            check = f"FAILED: {issues}" if issues else "passed"
            lines += ["", f"> {s['llm_explanation']}", "", f"Faithfulness check: {check}"]
        lines.append("")
    return "\n".join(lines)


def explain(use_llm: bool = False) -> None:
    """Write importance plots, local explanations and failure slices to reports/."""
    threshold = json.loads(METRICS.read_text())["threshold"]
    booster = lgb.Booster(model_file=MODEL)
    features = booster.feature_name()
    _, holdout = time_split(pl.read_parquet(FEATURES))
    X, y = matrix(holdout, features), holdout["Response"].to_numpy()
    p = booster.predict(X)
    contrib = booster.predict(X, pred_contrib=True)  # TreeSHAP; last column is the base value
    shap, base = contrib[:, :-1], contrib[:, -1]

    mean_abs = np.abs(shap).mean(axis=0)
    barh(
        top(mean_abs, features, 20),
        "Top 20 features (holdout)",
        "mean |SHAP| (log-odds)",
        REPORTS / "top_features.png",
    )
    stations, station_abs = by_station(mean_abs, features)
    barh(
        top(station_abs, stations, len(stations)),
        "Importance per station (holdout)",
        "sum of mean |SHAP| over the station's features (log-odds)",
        REPORTS / "station_importance.png",
    )

    cases = local_cases(y, p, threshold)
    ids = holdout["Id"].to_numpy()
    text = render_local(cases, ids, X, y, p, shap, base, features)
    (REPORTS / "local_explanations.md").write_text(text)

    slices = failure_slices(shap, y, p, threshold, features)
    if use_llm:
        from bosch_xai.llm import explain_slices

        explain_slices(slices)
    (REPORTS / "slices.json").write_text(json.dumps(slices, indent=2))
    (REPORTS / "slices.md").write_text(render_slices(slices, threshold))
    print(f"Wrote plots, local_explanations.md and {len(slices)} slices to {REPORTS}.")
