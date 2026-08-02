from community_detection.evaluation import (
    evaluate_seed_stability,
    holdout_agreement,
    modularity_score,
    recovery_metrics,
)
from community_detection.graph import build_bipartite_graph, detect_communities
from community_detection.synthetic import generate_interactions, split_interaction_weights


def _case():
    interactions, truth = generate_interactions(users=500)
    split = split_interaction_weights(interactions)
    graph = build_bipartite_graph(split)
    assignments, communities = detect_communities(graph, seed=11)
    return split, truth, graph, assignments, communities


def test_detected_partition_recovers_planted_structure() -> None:
    _, truth, graph, assignments, communities = _case()
    metrics = recovery_metrics(assignments, truth)
    assert metrics["user_ari"] >= 0.80
    assert metrics["category_ari"] >= 0.80
    assert modularity_score(graph, communities) > 0.30


def test_detected_partition_is_stable_across_seeds() -> None:
    _, _, graph, _, _ = _case()
    pairs, summary = evaluate_seed_stability(graph, seeds=(11, 23, 37), resolution=1.0)
    assert len(pairs) == 3
    assert summary["mean_pairwise_ari"] >= 0.80


def test_holdout_agreement_beats_shuffled_category_null() -> None:
    interactions, _, _, assignments, _ = _case()
    null_scores, summary = holdout_agreement(interactions, assignments, null_permutations=20)
    assert len(null_scores) == 20
    assert summary["observed_holdout_agreement"] > summary["null_mean_holdout_agreement"]
    assert summary["agreement_lift"] > 2
