# Data

Both CSV files in this folder are generated locally with a fixed seed. `synthetic_interactions.csv` contains fictional user-category interaction weights and an event-count train/holdout split. `synthetic_ground_truth.csv` contains planted node labels used only after detection for evaluation.

Identifiers beginning with `USR-` and `CAT-` are generated and do not refer to people, products, accounts, or production categories. Delete the CSV files and run `make reproduce` to regenerate them.
