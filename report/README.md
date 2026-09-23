# Report

Put your IEEE-format report here (`.tex` source and the built `.pdf`).

Required content for Project 2:

1. **Problem & dataset** — what you clustered and why.
2. **Implementation** — your K-means: initialization, assignment, update,
   convergence, restarts, empty-cluster handling.
3. **Choosing k** — elbow and/or silhouette evidence (`results/k_selection.png`).
4. **Evaluation** — cluster→class confusion matrix and metrics
   (`results/confusion_matrix.png`), plus internal measures. State explicitly
   that labels were used only to evaluate.
5. **Euclidean vs. Mahalanobis** — the diagonal C you chose, the comparison, and
   *why* it helps in terms of feature scale/variance.
   ⚠️ Inertia is **not** comparable across different distance metrics — compare
   silhouette or label-based measures instead.
6. **Conclusion & limitations.**

Every number in the report must match `results/metrics.json`.
