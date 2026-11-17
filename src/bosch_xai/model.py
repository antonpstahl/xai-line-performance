"""LightGBM training with stratified CV, an MCC-tuned threshold and a time-ordered holdout."""

import json
import platform
from importlib.metadata import version

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from bosch_xai.data import FEATURES, MODELS, REPORTS, SEED

MODEL = MODELS / "model.txt"
METRICS = REPORTS / "metrics.json"
N_FOLDS = 5
HOLDOUT_FRACTION = 0.2
MAX_ROUNDS = 2000
EARLY_STOPPING = 100
PARAMS = {
    "objective": "binary",
    "metric": "average_precision",  # PR-AUC: informative at 0.6% positives, unlike log loss
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,  # LightGBM docs: pair with deterministic for identical reruns
    "verbose": -1,
}


def time_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split into development parts and the latest HOLDOUT_FRACTION by start time.

    Parts without any timestamp sort first and therefore stay in development.
    """
    df = df.sort("start_time", "Id", nulls_last=False)
    n_dev = len(df) - int(len(df) * HOLDOUT_FRACTION)
    return df.head(n_dev), df.tail(len(df) - n_dev)


def feature_names(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("Id", "Response")]


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Return (threshold, MCC) maximising MCC for the rule `p >= threshold`.

    Evaluates every distinct score in one pass via cumulative counts instead of a grid.
    """
    order = np.argsort(-p, kind="stable")
    p_sorted, y_sorted = p[order], y[order].astype(float)
    tp, fp = np.cumsum(y_sorted), np.cumsum(1 - y_sorted)
    cut = np.r_[p_sorted[1:] != p_sorted[:-1], True]  # tied scores fall on the same side
    tp, fp, thresholds = tp[cut], fp[cut], p_sorted[cut]
    fn, tn = y_sorted.sum() - tp, (1 - y_sorted).sum() - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(mcc))
    return float(thresholds[best]), float(mcc[best])


def scores(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "mcc": round(matthews_corrcoef(y, p >= threshold), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
    }


def train() -> dict:
    """Run CV, pick the threshold on out-of-fold predictions, refit, score the holdout."""
    df = pl.read_parquet(FEATURES)
    sample = pl.read_parquet_metadata(FEATURES).get("sample") == "true"
    dev, holdout = time_split(df)
    features = feature_names(df)
    X, y = dev.select(features).to_numpy(), dev["Response"].to_numpy()
    full = lgb.Dataset(X, y, feature_name=features, free_raw_data=False)

    oof, rounds = np.zeros(len(y)), []
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)
    for i, (fit_idx, val_idx) in enumerate(folds, 1):
        booster = lgb.train(
            PARAMS,
            full.subset(fit_idx),
            MAX_ROUNDS,
            valid_sets=[full.subset(val_idx)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        oof[val_idx] = booster.predict(X[val_idx], num_iteration=booster.best_iteration)
        rounds.append(booster.best_iteration)
        print(f"fold {i}/{N_FOLDS}: best iteration {booster.best_iteration}")
    threshold, _ = best_threshold(y, oof)

    # Refit on all development parts with the mean early-stopping round of the folds.
    n_rounds = int(np.mean(rounds))
    final = lgb.train(PARAMS, full, n_rounds)
    MODELS.mkdir(parents=True, exist_ok=True)
    final.save_model(MODEL)
    y_hold = holdout["Response"].to_numpy()
    p_hold = final.predict(holdout.select(features).to_numpy())

    metrics = {
        "data": {
            "sample": sample,
            "parts": len(df),
            "failures": int(df["Response"].sum()),
            "features": len(features),
            "dev_parts": len(dev),
            "holdout_parts": len(holdout),
            "holdout_failures": int(y_hold.sum()),
        },
        "threshold": threshold,
        "oof": scores(y, oof, threshold),
        "holdout": scores(y_hold, p_hold, threshold) | {"flagged": int(sum(p_hold >= threshold))},
        "config": {
            "params": PARAMS,
            "n_folds": N_FOLDS,
            "holdout_fraction": HOLDOUT_FRACTION,
            "max_rounds": MAX_ROUNDS,
            "early_stopping": EARLY_STOPPING,
            "fold_rounds": rounds,
            "final_rounds": n_rounds,
        },
        "versions": {"python": platform.python_version()}
        | {pkg: version(pkg) for pkg in ("lightgbm", "polars", "scikit-learn", "numpy")},
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("threshold", "oof", "holdout")}, indent=2))
    return metrics
