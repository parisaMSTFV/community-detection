# Weighted edge-list input contract

The `analyze` command accepts a CSV representing a weighted bipartite graph. Each row is one
observed user-category edge.

| Column | Type | Rule |
|---|---|---|
| `user_id` | string | Required, non-null, non-blank, and from a namespace disjoint from category IDs |
| `category_id` | string | Required, non-null, non-blank, and from a namespace disjoint from user IDs |
| `weight` | number | Required, finite, and strictly positive |

Each `user_id` and `category_id` pair must be unique. Aggregate repeated events before running
the command. Extra columns are ignored. The edge list must contain at least two users and two
categories.

```bash
community-detection analyze \
  --edges path/to/weighted_edges.csv \
  --output-root artifacts/my_graph
```

The input file is read but never copied into the output directory. Node identifiers are written to
the assignment output, so the caller is responsible for using approved or public-safe identifiers
and protecting the output directory. The command writes:

- `reports/community_assignments.csv`: one detected community per user and category node;
- `reports/community_profiles.csv`: community sizes, internal interaction share, and leading
  category IDs;
- `reports/stability_pairs.csv`: pairwise agreement across the configured Louvain seeds;
- `reports/metrics.json`: graph diagnostics, stability, artifact fingerprint, and NetworkX version;
- `reports/run_summary.md`: a short result and evaluation-boundary summary;
- `reports/figures/community_sizes.png`: user and category counts by detected community.

This path does not accept labels, outcomes, or holdout interactions. Its modularity and seed
stability are unsupervised diagnostics, not evidence that the groups predict behavior or improve
campaign results. Review the assignments and category profiles before using them as hypotheses
for downstream analysis.
