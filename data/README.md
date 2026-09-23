# Data

**Do not commit datasets.** This directory is git-ignored (except this file).

## Default: Iris (no download needed)

`config.yaml` ships with `data.source: iris`, which loads the Iris dataset
bundled with scikit-learn. Use it to debug your from-scratch implementation:
150 points, 4 features, 3 well-understood classes.

## Project dataset: UNSW-NB15

The deliverable uses a realistic network-traffic dataset.

1. Download from the **UNSW Canberra** project page:
   <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
2. Use the **partitioned CSVs** — `UNSW_NB15_training-set.csv` and
   `UNSW_NB15_testing-set.csv` (~42 features; 1 normal + 9 attack categories).
3. Place them in this directory (they stay untracked).
4. Point `config.yaml` at the file:

```yaml
data:
  source: csv
  csv_path: data/UNSW_NB15_training-set.csv
  target: attack_cat          # labels: EVALUATION ONLY
  drop_columns: [id, label]   # id is an index; label duplicates attack_cat
  subsample: 4000
```

### Notes

- `attack_cat` is the 10-class label; `label` is the binary 0/1 version. Drop
  whichever you are not evaluating against — keeping both leaks the target.
- Categorical columns (`proto`, `service`, `state`) are dropped automatically by
  the numeric-only selector. Encoding them is a reasonable extension — say so in
  your report if you do.
- A from-scratch K-means is comfortable on a few thousand rows, not millions.
  Keep `subsample` modest; it is stratified by class.

### Citation

Moustafa, N., & Slay, J. (2015). *UNSW-NB15: A Comprehensive Data Set for Network
Intrusion Detection Systems.* Military Communications and Information Systems
Conference (MilCIS), IEEE.
