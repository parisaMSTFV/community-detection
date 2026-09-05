# Interview discussion guide

## What this project demonstrates

This project models customer-category affinity as network topology rather than a flat customer
feature table. Louvain detects mixed user and category communities without access to planted labels.
The workflow evaluates planted-label recovery only after detection, searches resolution without
truth, and reports seed stability, component and population coverage, category-hub sensitivity, and
later-window behavior.

## Decisions to explain

- Why a bipartite graph preserves user-category structure that a category-only projection can lose.
- Why ground truth is isolated from the detection input.
- Why ARI is appropriate when numeric community labels are arbitrary.
- Why modularity alone is insufficient and can favor structurally convenient partitions.
- Why repeated seeds are required for a stochastic community algorithm.
- Why two independently simulated time windows are stronger evidence than randomly removing weight
  from already-known edges.
- Why recurring-edge and unseen-edge agreement must be reported separately.
- Why high modularity and perfect seed stability can still describe disconnected pairs with no
  usable population coverage.
- Why `log_tfidf` weighting is used to reduce domination by universally popular categories.
- Why the quality gate is a review floor rather than a claim of business validity.
- Why HMAC pseudonyms require an external salt and remain pseudonymous rather than anonymous.
- How snapshot IDs, model versions, and reference-label alignment support monitored reruns.
- Why detected communities still require business profiling and controlled campaign tests before activation.
