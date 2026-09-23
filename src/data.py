"""Data loading and preparation for clustering.

Default is the **Iris** dataset (ships with scikit-learn — no download), which is
small, well separated, and ideal for debugging a from-scratch implementation.

Switch to your real security dataset (UNSW-NB15) by setting
`data.source: csv` and `data.csv_path` in config.yaml. See data/README.md.

IMPORTANT — clustering is UNSUPERVISED. `load_data` returns labels `y`, but they
are for **evaluation only**. Never pass `y` to the clustering algorithm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_data(cfg: dict):
    """Return (X, y, feature_names, class_names).

    X : float64 ndarray (n_samples, n_features) — standardized if configured
    y : int64 ndarray (n_samples,) — ground-truth labels, FOR EVALUATION ONLY
    """
    d = cfg["data"]
    source = d.get("source", "iris")

    if source == "iris":
        from sklearn.datasets import load_iris

        ds = load_iris()
        X = ds.data.astype("float64")
        y = ds.target.astype("int64")
        feature_names = list(ds.feature_names)
        class_names = list(ds.target_names)

    elif source == "csv":
        df = pd.read_csv(d["csv_path"])

        for col in d.get("drop_columns") or []:
            if col in df.columns:
                df = df.drop(columns=[col])

        target = d["target"]
        if target not in df.columns:
            raise KeyError(
                f"target column {target!r} not in {d['csv_path']}; "
                f"available: {list(df.columns)[:15]}..."
            )

        labels = df[target].astype("category")
        y = labels.cat.codes.to_numpy().astype("int64")
        class_names = list(labels.cat.categories)

        feats = df.drop(columns=[target]).select_dtypes("number")
        feature_names = list(feats.columns)
        X = feats.to_numpy().astype("float64")

    else:
        raise ValueError(f"unknown data.source {source!r} (expected 'iris' or 'csv')")

    # Optional subsample — a from-scratch K-means is fine on thousands of rows,
    # not millions. Stratified by label so every class stays represented.
    cap = d.get("subsample")
    if cap and len(X) > cap:
        rng = np.random.default_rng(cfg["seed"])
        idx = _stratified_sample(y, cap, rng)
        X, y = X[idx], y[idx]

    # Drop non-finite rows and zero-variance columns (they break standardization).
    finite = np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite]
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    feature_names = [f for f, k in zip(feature_names, keep) if k]

    if d.get("standardize", True):
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y, feature_names, class_names


def _stratified_sample(y: np.ndarray, cap: int, rng) -> np.ndarray:
    """Indices of a roughly class-balanced subsample of size <= cap."""
    classes = np.unique(y)
    per = max(1, cap // len(classes))
    picks = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        picks.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out
