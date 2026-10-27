from __future__ import annotations

from itertools import combinations
import math

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .contracts import groups_from_labels


def evaluate(graph: nx.Graph, labels: dict, truth: dict | None) -> dict:
    if set(labels) != set(graph):
        raise ValueError("Partition must assign every graph node exactly once")
    groups = groups_from_labels(labels)
    sizes = [len(group) for group in groups]
    metrics = {
        "ari": None, "nmi": None,
        "modularity": float(nx.community.modularity(graph, groups, weight=None, resolution=1))
        if graph.number_of_edges() else None,
        "communities": len(groups),
        "largest_fraction": max(sizes, default=0) / len(graph) if len(graph) else 0,
        "singleton_fraction": sum(size for size in sizes if size == 1) / len(graph) if len(graph) else 0,
        "label_coverage": 1.0 if truth is not None and set(truth) == set(graph) else 0.0,
    }
    if truth is not None:
        if set(truth) != set(graph):
            raise ValueError("Incomplete evaluation labels")
        nodes = sorted(graph)
        actual, predicted = [truth[u] for u in nodes], [labels[u] for u in nodes]
        metrics["ari"] = float(adjusted_rand_score(actual, predicted))
        metrics["nmi"] = float(normalized_mutual_info_score(actual, predicted, average_method="arithmetic"))
    return metrics


def pairwise_stability(partitions: dict[str, dict]) -> list[dict]:
    rows = []
    for (id_a, a), (id_b, b) in combinations(partitions.items(), 2):
        if set(a) != set(b):
            raise ValueError("Stability comparisons require the same node set")
        nodes = sorted(a)
        rows.append({"run_a": id_a, "run_b": id_b,
                     "stability_ari": float(adjusted_rand_score([a[u] for u in nodes], [b[u] for u in nodes]))})
    return rows
