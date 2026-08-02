from community_detection.graph import build_bipartite_graph, detect_communities
from community_detection.profiles import build_community_profiles
from community_detection.synthetic import generate_interactions, split_interaction_weights


def test_profiles_cover_detected_users_and_normalize_family_weight() -> None:
    interactions, _ = generate_interactions(users=300)
    split = split_interaction_weights(interactions)
    assignments, _ = detect_communities(build_bipartite_graph(split), seed=11)
    profiles, matrix = build_community_profiles(split, assignments)
    assert profiles["users"].sum() == 300
    assert (matrix.sum(axis=1).round(12) == 1).all()
    assert profiles["top_categories"].str.len().gt(0).all()
