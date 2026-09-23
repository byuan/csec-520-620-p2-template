"""Single entry point: `python -m src.cluster --config config.yaml`.

Reproducibility standard: this command must recreate your reported results.
It writes results/metrics.json and figures to the output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import yaml

from .data import load_data
from . import kmeans as km
from . import evaluate as ev


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    out = cfg["output"]["dir"]
    os.makedirs(out, exist_ok=True)

    # ---- data (labels are for EVALUATION ONLY) ---------------------------
    X, y, feature_names, class_names = load_data(cfg)
    impl = cfg["kmeans"].get("implementation", "sklearn")
    print(f"Loaded {X.shape[0]} points x {X.shape[1]} features "
          f"({cfg['data']['source']}); implementation={impl}")

    # ---- k selection sweep (elbow + silhouette) --------------------------
    sel = cfg.get("selection", {})
    k_min, k_max = sel.get("k_min", 2), sel.get("k_max", 8)

    def build_k(k):
        c = json.loads(json.dumps(cfg))   # cheap deep copy
        c["kmeans"]["k"] = k
        return km.build(c)

    ks, inertias, sils = ev.k_selection_sweep(X, build_k, k_min, k_max)
    ev.save_k_selection_plot(ks, inertias, sils,
                             os.path.join(out, "k_selection.png"),
                             chosen_k=cfg["kmeans"]["k"])
    best_k_by_silhouette = int(ks[int(np.nanargmax(sils))])
    print(f"k sweep {k_min}..{k_max}: best silhouette at k={best_k_by_silhouette}")

    # ---- fit at the configured k -----------------------------------------
    est = km.build(cfg).fit(X)
    labels = est.labels_

    # ---- evaluate ---------------------------------------------------------
    metrics = {
        "implementation": impl,
        "distance": cfg["kmeans"].get("distance", "euclidean"),
        "k": int(cfg["kmeans"]["k"]),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_iter": int(est.n_iter_) if est.n_iter_ is not None else None,
        "best_k_by_silhouette": best_k_by_silhouette,
        "k_sweep": {"k": ks, "inertia": inertias, "silhouette": sils},
    }
    metrics.update(ev.internal_metrics(X, labels, est.inertia_))
    metrics.update(ev.external_metrics(y, labels))

    mapping, y_pred = ev.map_clusters_to_classes(y, labels)
    metrics["cluster_to_class"] = mapping
    metrics.update(ev.classification_metrics(y, y_pred))
    metrics["confusion_matrix"] = ev.save_confusion_matrix(
        y, y_pred, os.path.join(out, "confusion_matrix.png"), class_names)

    ev.save_cluster_scatter(X, labels, est.centroids_,
                            os.path.join(out, "clusters.png"),
                            title=f"{impl} K-means (k={cfg['kmeans']['k']})")

    # ---- optional: agree with the reference implementation? --------------
    # The handout allows a library K-means ONLY as a correctness check.
    if impl == "scratch" and cfg.get("compare_to_reference", False):
        ref = km.KMeansReference(k=cfg["kmeans"]["k"], init=cfg["kmeans"].get("init", "kmeans++"),
                                 n_init=cfg["kmeans"].get("n_init", 10),
                                 max_iter=cfg["kmeans"].get("max_iter", 300),
                                 tol=float(cfg["kmeans"].get("tol", 1e-4)),
                                 seed=cfg["seed"]).fit(X)
        from sklearn.metrics import adjusted_rand_score
        metrics["reference_check"] = {
            "reference_inertia": float(ref.inertia_),
            "scratch_inertia": float(est.inertia_),
            "inertia_ratio": float(est.inertia_ / ref.inertia_) if ref.inertia_ else None,
            "ari_vs_reference": float(adjusted_rand_score(ref.labels_, labels)),
        }
        print("Reference check:", json.dumps(metrics["reference_check"], indent=2))

    ev.write_metrics(metrics, os.path.join(out, "metrics.json"))
    print("Results written to", out)
    summary = {k: metrics[k] for k in
               ("k", "inertia", "silhouette", "v_measure", "accuracy", "f1")
               if k in metrics}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
