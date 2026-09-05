# Data

Both CSV files in this folder are generated locally with a fixed seed.
`synthetic_interactions.csv` contains fictional user-category weights sampled into independent
90-day training and 30-day later windows. `synthetic_ground_truth.csv` contains planted node labels
used only after detection for evaluation.

Identifiers beginning with `USR-` and `CAT-` are generated and do not refer to people, products,
accounts, or production categories. Run `make reproduce` to regenerate both files deterministically.
