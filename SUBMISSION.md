# Submission Manifest — Project 2

Fill this in before submitting. The grading agent reads it first.

## Student
- Name (RIT ID):

## Course level
- [ ] 520 (undergraduate)
- [ ] 620 (graduate)

## Dataset
- Name / source / version / URL:
- How to obtain it (command or steps):
- Label column (used for EVALUATION ONLY):
- Subsample size and how you chose it:

## How to reproduce
```bash
make setup
make reproduce
```
Anything non-default the grader must know (data download, runtime):

## Your implementation
- Confirm `config.yaml` has `kmeans.implementation: scratch`: [ ]
- Initialization used (`random` / `kmeans++`):
- How you handle **empty clusters**:
- Convergence criterion and tolerance:
- Agreement with the reference (`reference_check` in metrics.json):
  - `inertia_ratio`:
  - `ari_vs_reference`:

## Choosing k
- k you report, and the evidence (elbow / silhouette):
- If the best silhouette k differs from the number of true classes, explain:

## Claimed results (must match `results/metrics.json`)
| Metric | Euclidean | Mahalanobis |
|---|---|---|
| Silhouette | | |
| V-measure | | |
| Accuracy | | |
| Macro F1 | | |

- The diagonal C you chose, and why:

## AI-use acknowledgment
Per the syllabus policy, briefly note any substantive use of AI assistants.
