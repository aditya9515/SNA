from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from sna_project.algorithms import (apply_weights, edge_similarity, from_groups, prepare_leiden,
                                    propagate, run_leiden, run_method, support_violations)
from sna_project.data import load_prepared, save_prepared
from sna_project.evaluation import evaluate, pairwise_stability
from sna_project.experiment import read_config


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("graph", [nx.path_graph(8), nx.cycle_graph(10), nx.complete_graph(8),
                                   nx.star_graph(8), nx.karate_club_graph()])
def test_original_matches_networkx(graph, seed):
    ours = run_method(graph, "flpa_original", seed)
    reference = from_groups(nx.community.fast_label_propagation_communities(graph, weight=None, seed=seed), graph)
    assert ours.status == "converged"
    assert ours.labels == reference
    assert not support_violations(graph, ours.labels)


def test_triangle_bridge_weights():
    graph = nx.Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
    similarity = edge_similarity(graph)
    assert similarity[(0, 1)] == pytest.approx(1 / 3)
    assert similarity[(0, 2)] == pytest.approx(1 / 4)
    assert similarity[(2, 3)] == 0
    weighted = apply_weights(graph, similarity, 1)
    assert weighted[0][1]["structural_weight"] == pytest.approx(4 / 3)
    assert not any("structural_weight" in attributes for *_, attributes in graph.edges(data=True))


@pytest.mark.parametrize("seed", range(5))
def test_lambda_zero_controls(seed):
    graph = nx.karate_club_graph()
    weighted = apply_weights(graph, edge_similarity(graph), 0)
    for a, b in [("c10_weighted_random", "c00_unit_random"), ("c11_weighted_retain", "c01_unit_retain")]:
        result_a, result_b = run_method(weighted, a, seed), run_method(graph, b, seed)
        assert result_a.labels == result_b.labels
        assert result_a.queue_pops == result_b.queue_pops


def test_limit_and_isolates():
    graph = nx.empty_graph(3)
    result = run_method(graph, "flpa_original", 0)
    assert len(set(result.labels.values())) == 3
    assert run_method(nx.cycle_graph(20), "flpa_original", 0, max_queue_pops=1).status == "max_queue_pops"
    assert run_method(nx.Graph(), "flpa_original", 0).labels == {}


@pytest.mark.parametrize("method", ["c00_unit_random", "c10_weighted_random", "c01_unit_retain", "c11_weighted_retain"])
def test_variant_support_and_reproducibility(method):
    graph = apply_weights(nx.karate_club_graph(), edge_similarity(nx.karate_club_graph()), .5)
    a, b = run_method(graph, method, 7), run_method(graph, method, 7)
    assert a.labels == b.labels
    assert a.status == b.status == "converged"
    assert not support_violations(graph, a.labels, "structural_weight" if "weighted" in method else None)


def test_preprocessing_alignment_and_reload(tmp_path):
    raw = nx.DiGraph()
    raw.add_nodes_from(["x", "y", "isolate"])
    raw.add_edges_from([("x", "y"), ("y", "x"), ("x", "x")])
    nx.set_node_attributes(raw, {"x": "a", "y": "a", "isolate": "b"}, "club")
    item = save_prepared(tmp_path, "fixture", raw, {"x": 0, "y": 0, "isolate": 1}, "fixture")
    loaded = load_prepared(tmp_path, "fixture")
    assert item.graph_hash == loaded.graph_hash
    assert len(loaded.graph) == 3 and loaded.graph.number_of_edges() == 1
    assert nx.number_of_isolates(loaded.graph) == 1
    assert all(not attributes for _, attributes in loaded.graph.nodes(data=True))
    edges = pd.read_csv(item.path / "edges.csv")
    edges.iloc[0, 0] = 99
    edges.to_csv(item.path / "edges.csv", index=False)
    with pytest.raises(ValueError, match="hash"):
        load_prepared(tmp_path, "fixture")


def test_metric_alignment_and_weight_policy():
    graph = nx.path_graph(4)
    labels = {2: 7, 0: 4, 3: 7, 1: 4}
    truth = {0: 1, 1: 1, 2: 2, 3: 2}
    metrics = evaluate(graph, labels, truth)
    assert metrics["ari"] == metrics["nmi"] == 1
    nx.set_edge_attributes(graph, 99, "structural_weight")
    assert evaluate(graph, labels, truth)["modularity"] == metrics["modularity"]
    with pytest.raises(ValueError):
        evaluate(graph, {0: 1}, truth)
    assert evaluate(nx.empty_graph(2), {0: 0, 1: 1}, None)["modularity"] is None


def test_stability_pairs():
    partitions = {str(seed): {0: 0, 1: 0, 2: 1} for seed in range(10)}
    pairs = pairwise_stability(partitions)
    assert len(pairs) == 45
    assert all(pair["stability_ari"] == 1 for pair in pairs)
    assert not pairwise_stability({"a": partitions["0"]})


def test_leiden_conversion_preserves_isolate():
    graph = nx.Graph([(10, 20)])
    graph.add_node(99)
    result = run_leiden(prepare_leiden(graph), 0)
    assert set(result.labels) == set(graph)
    assert result.labels[99] != result.labels[10]


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf")])
def test_invalid_weights(invalid):
    graph = nx.path_graph(3)
    with pytest.raises(ValueError):
        apply_weights(graph, edge_similarity(graph), invalid)


def test_bad_cache_and_config(tmp_path):
    with pytest.raises(ValueError):
        apply_weights(nx.path_graph(3), {}, .5)
    path = tmp_path / "bad.yaml"
    path.write_text("datasets: [karate]\nmethods: [flpa_original, flpa_original]\nseeds: [0]\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_config(path)
