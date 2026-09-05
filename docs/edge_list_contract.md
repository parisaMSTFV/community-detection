# Weighted edge-list input contract

The `analyze` command accepts an aggregate user-category edge snapshot. Each row represents one
unique edge in one observation window.

| Column | Parsed type | Rule |
|---|---|---|
| `user_id` | string | Required, non-null, non-blank, maximum 128 characters |
| `category_id` | string | Required, non-null, non-blank, maximum 128 characters |
| `weight` | number | Required, finite, and strictly positive |

Identifiers are parsed as strings before any type inference, preserving Unicode and leading zeroes.
User and category IDs have separate internal namespaces, so the same source value may occur in both
columns. Identifiers beginning with `=`, `+`, `-`, or `@` and values containing control characters
are rejected to reduce spreadsheet-formula and output-handling risk.

Each user-category pair must be unique. Aggregate repeated events before running the command. Extra
columns are ignored. The CSV must be UTF-8, no larger than 250 MB, and contain at least two users and
two categories.

## Default pseudonymized export

External node IDs are exported as namespace-aware HMAC-SHA256 tokens. The salt must contain at least
16 characters and is read from `COMMUNITY_DETECTION_ID_SALT`; it is not written to any artifact.

```bash
export COMMUNITY_DETECTION_ID_SALT="$(python -c 'import secrets; print(secrets.token_hex(32))')"
uv run community-detection analyze \
  --edges path/to/current_edges.csv \
  --output-root artifacts/current
```

Keep the salt in an approved secret manager if assignments must remain comparable across snapshots.
Changing it intentionally breaks linkage. Hashed identifiers are still pseudonymous data, not
anonymous data, and must follow the applicable access and retention policy.

Raw identifiers can be exported only with `--allow-raw-identifiers`. This override is intended for
approved public or fully synthetic inputs such as `examples/weighted_edges.csv`.

## Later snapshot

Supply an independently aggregated later window to measure recurring-edge agreement, unseen-edge
agreement, node coverage, and train-to-future partition ARI:

```bash
uv run community-detection analyze \
  --edges path/to/current_edges.csv \
  --future-edges path/to/later_edges.csv \
  --output-root artifacts/current
```

The later file follows the same three-column contract. It is never merged into training.

## Label continuity

`--reference-assignments` accepts an earlier `community_assignments.csv`. Current numeric labels are
mapped one-to-one to prior labels by maximum overlap on typed node IDs. New communities receive new
labels. Pseudonymized snapshots must use the same salt.

```bash
uv run community-detection analyze \
  --edges path/to/current_edges.csv \
  --reference-assignments path/to/previous/community_assignments.csv \
  --output-root artifacts/current
```

Reference assignments require unique `node_id` plus `node_type` pairs, a `user` or `category` node
type, and integer community labels.

## Outputs

- `community_assignments.csv`: typed node assignment, weighted degree, snapshot ID, and model version;
- `community_profiles.csv`: member counts, internal weight share, and leading category IDs;
- `community_quality.csv`: minimum-size eligibility by community;
- `component_diagnostics.csv`: connected-component size and weight diagnostics;
- `resolution_search.csv`: candidate scores and the evidence used for selection;
- `stability_pairs.csv`: pairwise agreement across configured Louvain seeds;
- `temporal_null_distribution.csv`: written only when a later snapshot is supplied;
- `label_alignment.csv`: written only when reference assignments are supplied;
- `model_manifest.json`: version, policy, dependency, config, source, and snapshot fingerprints;
- `metrics.json` and `run_summary.md`: decision gate, diagnostics, and evaluation boundaries;
- `figures/community_sizes.png`: user and category counts by detected community.

The input and later-window CSV files are read but never copied to the output directory. Use
`--fail-on-review` when an automated workflow should exit with status 2 if any configured guardrail
requires analyst review.

Modularity, seed stability, and a technical gate do not establish outcome validity or business
impact. A later snapshot improves the evidence boundary but still does not replace cohort review,
outcome validation, or a controlled experiment.
