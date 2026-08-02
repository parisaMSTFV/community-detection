# Interview discussion guide

## What this project demonstrates

This project models customer-category affinity as network topology rather than a flat customer feature table. Louvain detects mixed user and category communities without access to planted labels. Evaluation then measures planted-label recovery, weighted modularity, algorithm-seed stability, and held-out interaction agreement against a category-label permutation baseline.

## Decisions to explain

- Why a bipartite graph preserves user-category structure that a category-only projection can lose.
- Why ground truth is isolated from the detection input.
- Why ARI is appropriate when numeric community labels are arbitrary.
- Why modularity alone is insufficient and can favor structurally convenient partitions.
- Why repeated seeds are required for a stochastic community algorithm.
- Why the holdout measures held-out event weight on observed edges rather than unseen-edge prediction.
- Why detected communities still require business profiling and controlled campaign tests before activation.
