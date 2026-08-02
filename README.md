# Customer–Category Community Detection

[![CI](https://github.com/parisaMSTFV/community-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/parisaMSTFV/community-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3C78A8)
![Data](https://img.shields.io/badge/data-fully%20synthetic-4A9D8F)

A reproducible network-science case study that discovers mixed user and category communities in a weighted bipartite graph, evaluates recovery and stability, and translates the partition into business-readable affinity profiles.

## Executive summary

Customer segmentation usually starts from a flat feature table. This project asks a different question: which users and product categories form densely connected behavioral neighborhoods? Synthetic interaction counts become edges in a bipartite graph. Weighted Louvain detection receives only the training graph; planted labels remain in a separate evaluation table.

The executed pipeline found 6 communities across 1,200 synthetic users and 30 categories. User-label recovery reached adjusted Rand index (ARI) `0.902`, category ARI reached `1.000`, and weighted modularity was `0.650`. Across five Louvain seeds, mean pairwise ARI was `0.978` and the minimum was `0.947`. Held-out interaction agreement was `0.826`, compared with `0.170` after category-label permutation, a `4.87×` lift.

These results describe the planted synthetic graph only. They show that the workflow can recover known structure under controlled ambiguity; they do not establish performance on real behavior or campaign impact.

## Business problem

Flat customer clusters can hide the topology linking customers to category interests. A mixed user-category community can support:

- audience discovery around category affinity rather than demographic assumptions;
- category-led campaign planning and cross-category exploration;
- identification of bridge categories that connect otherwise distinct interests;
- a network feature layer for downstream recommendation or experimentation.

The repository stops before activation. A detected community is a hypothesis about affinity, not proof that targeting it will create incremental value.

## Analytical questions

- Does weighted community detection recover the planted user and category structure without seeing evaluation labels?
- Is the partition stable when Louvain's random seed changes?
- Do held-out interaction events remain concentrated inside detected communities?
- Does that concentration exceed a topology-destroying shuffled-category baseline?
- Which category families and named categories characterize each detected group?

## Workflow

```mermaid
flowchart TD
    A["Synthetic interactions"] --> B["Event-weight train and holdout split"]
    B --> C["Weighted bipartite graph"]
    C --> D["Louvain detection"]
    D --> E["Recovery and modularity"]
    D --> F["Seed stability"]
    D --> G["Holdout vs shuffled null"]
    E --> H["Community profiles"]
    F --> H
    G --> H
```

## Synthetic data

The generator creates:

- 1,200 fictional users with identifiers such as `USR-00001`;
- 30 fictional categories with identifiers such as `CAT-001`;
- six planted affinity families: Digital, Home, Style, Wellness, Family, and Outdoor;
- 6,696 unique user-category edges and 52,588 total interaction events;
- a 20% event-count holdout on each observed edge, while retaining at least one training event;
- ambiguous users with a strong secondary affinity to make recovery non-trivial.

The planted label is never included in the interaction table passed to graph construction. It is joined only after detection. See [data provenance](DATA_PROVENANCE.md).

## Methodology

### Weighted bipartite graph

User and category nodes form separate node types. An undirected edge represents observed interaction, and its training event count is the edge weight. No user-user or category-category edge is supplied to Louvain.

### Louvain community detection

NetworkX Louvain detection runs at resolution `1.0`. Detected numeric labels are relabeled by descending user count for readable artifacts. This relabeling does not affect ARI or modularity.

### Recovery metrics

Adjusted Rand index compares detected and planted partitions while remaining invariant to arbitrary numeric community labels. It is calculated separately for users, categories, and all nodes. Weighted modularity measures how much interaction weight is concentrated within the detected partition, but it is not treated as sufficient evidence on its own.

### Stability

Detection is repeated with seeds `11`, `23`, `37`, `53`, and `71`. Every pair of partitions is compared with ARI. Reporting both the mean and minimum prevents a high average from hiding one unstable run.

### Holdout and baseline

The holdout contains event weight removed from observed user-category edges before detection. The score is the share of held-out weight whose user and category receive the same detected community. It tests consistency on future events for known edges, not unseen-edge prediction.

For the null baseline, detected category labels are shuffled 100 times while user labels and community-size frequencies are retained. This breaks user-category topology without changing the category-label distribution.

## Executed results

| Evaluation | Result |
|---|---:|
| Detected communities | 6 |
| User ARI | 0.902 |
| Category ARI | 1.000 |
| Overall ARI | 0.904 |
| Weighted modularity | 0.650 |
| Mean pairwise seed ARI | 0.978 |
| Minimum pairwise seed ARI | 0.947 |
| Held-out interaction agreement | 0.826 |
| Shuffled-category null mean | 0.170 |
| Agreement lift | 4.87× |
| Core artifact fingerprint | `ef7a9d8885609374` |

![Evaluation summary](reports/figures/evaluation_summary.png)

High category recovery and lower user recovery are expected in this fixture: categories have one planted family, while one quarter of users receive a deliberately strong secondary affinity.

## Visual results

![Community interaction profiles](reports/figures/community_profiles.png)

Each row is normalized by the training interaction weight of users assigned to that community. The dominant family supports interpretation, while the smaller off-diagonal shares show cross-community behavior.

![Category projection](reports/figures/category_projection.png)

This projection is a display artifact built from shared synthetic users. Detection itself runs on the original bipartite graph, not on this category-only projection.

![Community sizes](reports/figures/community_sizes.png)

`reports/community_profiles.csv` adds internal interaction share and the three leading category names for each community.

## Repository structure

```text
community-detection/
├── configs/analysis.json
├── data/
│   ├── README.md
│   ├── synthetic_interactions.csv
│   └── synthetic_ground_truth.csv
├── docs/interview_guide.md
├── reports/
│   ├── figures/
│   ├── metrics.json
│   ├── community_assignments.csv
│   ├── community_profiles.csv
│   ├── stability_pairs.csv
│   └── holdout_null_distribution.csv
├── scripts/check_sensitive.py
├── src/community_detection/
├── tests/
└── .github/workflows/ci.yml
```

## Reproduce the project

Python 3.11 or 3.12 is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make reproduce
make check
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

`community-detection smoke` runs the same pipeline in a temporary directory without changing checked-in artifacts.

## Tests and quality checks

The test suite checks deterministic generation, label isolation, event-weight reconciliation, input schema, bipartite edge types, node coverage, planted-label recovery, modularity, seed stability, holdout lift, normalized profiles, required outputs, and deterministic fingerprints.

GitHub Actions runs Ruff, format checks, Pytest on Python 3.11 and 3.12, the sensitive-content scan, and the full smoke pipeline without credentials or external data.

## Limitations

- The graph has planted family structure and is easier to interpret than real multi-intent behavior.
- Category ARI of 1.000 is specific to this synthetic design and should not be expected in production.
- The event-count holdout tests known edges and is not a link-prediction evaluation.
- Louvain can have a resolution limit and may merge small but meaningful groups.
- Modularity can reward partitions that are statistically convenient but not actionable.
- The analysis uses one behavior type and one static period; it does not measure temporal drift.
- No uplift test, campaign outcome, or business-value estimate is included.

## Potential next steps

A production-oriented extension would compare behavior layers, use time-based snapshots, evaluate unseen-edge prediction, monitor partition drift, and test whether community-based targeting adds incremental value over existing audience rules. Any such extension would require an approved public-safe dataset and separate validation.

## Portfolio distinction

This repository demonstrates graph modeling and network validation. A customer segmentation project based on engineered customer features answers a different question and should be evaluated with clustering quality, stability, and actionability metrics rather than network modularity.

## Author

Parisa Mostafavi · [LinkedIn](https://www.linkedin.com/in/parisa-mostafavi/)
