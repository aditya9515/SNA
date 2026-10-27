"""Original FLPA and independently controlled weighting/retention variants.

The baseline follows Traag & Subelj (2023), Algorithm 3. Reference parity is
checked against NetworkX. No code claims that weighting or retention is new.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import copy
import math
import random
import time

import networkx as nx

from .contracts import AlgorithmResult, canonical_partition


METHODS = ["flpa_original", "c00_unit_random", "c10_weighted_random",
           "c01_unit_retain", "c11_weighted_retain", "asyn_lpa",
           "leiden_modularity", "nx_weighted_flpa"]
WEIGHTED_METHODS = {"c10_weighted_random", "c11_weighted_retain", "nx_weighted_flpa"}


def validate_graph(graph: nx.Graph) -> None:
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Expected a simple undirected graph without self-loops")


def edge_similarity(graph: nx.Graph) -> dict[tuple[int, int], float]:
    validate_graph(graph)
    neighbours = {u: set(graph[u]) for u in graph}
    similarity = {}
    for u, v in graph.edges:
        common = len(neighbours[u] & neighbours[v])
        union = len(neighbours[u]) + len(neighbours[v]) - common
        similarity[tuple(sorted((u, v)))] = common / union
    return similarity


def apply_weights(graph: nx.Graph, similarity: dict, lambda_value: float) -> nx.Graph:
    if not math.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda must be finite and non-negative")
    # NetworkX Graph.copy can rebuild adjacency order. Preserve it because queue
    # insertion order is part of a seeded asynchronous algorithm's trajectory.
    weighted = copy.deepcopy(graph)
    expected = {tuple(sorted(edge)) for edge in graph.edges}
    if set(similarity) != expected:
        raise ValueError("Similarity keys do not match graph topology")
    for (u, v), value in similarity.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid Jaccard similarity")
        weighted[u][v]["structural_weight"] = 1.0 + lambda_value * value
    return weighted


def label_support(graph: nx.Graph, labels: dict, node: int, weight: str | None) -> dict:
    if weight is None:
        return dict(Counter(labels[v] for v in graph[node]))
    contributions: dict[int, list[float]] = defaultdict(list)
    for neighbour in graph[node]:
        value = graph[node][neighbour].get(weight)
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError("Every weighted edge must have a finite positive weight")
        contributions[labels[neighbour]].append(value)
    return {label: math.fsum(values) for label, values in contributions.items()}


def maximal_candidates(support: dict, *, weighted: bool, atol: float, rtol: float) -> list:
    best = max(support.values())
    tolerance = atol + rtol * abs(best) if weighted else 0.0
    return [label for label, value in support.items() if best - value <= tolerance]


def support_violations(graph: nx.Graph, labels: dict, weight=None, atol=1e-12, rtol=1e-12) -> list:
    failures = []
    for node in graph:
        if graph.degree(node):
            scores = label_support(graph, labels, node, weight)
            candidates = maximal_candidates(scores, weighted=weight is not None, atol=atol, rtol=rtol)
            if labels[node] not in candidates:
                failures.append(node)
    return failures


def propagate(graph: nx.Graph, *, seed: int, weight=None, retain=False,
              queue_policy="original_filtered", reference_order=False,
              max_queue_pops=None, max_seconds=120.0, atol=1e-12, rtol=1e-12) -> AlgorithmResult:
    validate_graph(graph)
    if queue_policy not in {"original_filtered", "all_changed_neighbors"}:
        raise ValueError("Unknown queue policy")
    if atol < 0 or rtol < 0 or not math.isfinite(atol + rtol):
        raise ValueError("Invalid tie tolerance")
    rng = random.Random(seed)
    nodes = list(graph)
    labels = {node: i for i, node in enumerate(nodes)}
    rng.shuffle(nodes)
    queue, in_queue = deque(nodes), set(nodes)
    pops = changes = 0
    enqueues = len(nodes)
    peak = len(nodes)
    cap = max_queue_pops if max_queue_pops is not None else max(10000, 1000 * len(nodes))
    started = time.perf_counter()
    status = "converged"
    while queue:
        if pops >= cap:
            status = "max_queue_pops"
            break
        if time.perf_counter() - started > max_seconds:
            status = "max_time"
            break
        node = queue.popleft()
        in_queue.remove(node)
        pops += 1
        if not graph.degree(node):
            continue
        scores = label_support(graph, labels, node, weight)
        candidates = maximal_candidates(scores, weighted=weight is not None, atol=atol, rtol=rtol)
        if not reference_order:
            candidates.sort()
        chosen = labels[node] if retain and labels[node] in candidates else rng.choice(candidates)
        if chosen != labels[node]:
            labels[node] = chosen
            changes += 1
            for neighbour in graph[node]:
                affected = queue_policy == "all_changed_neighbors" or labels[neighbour] != chosen
                if affected and neighbour not in in_queue:
                    queue.append(neighbour)
                    in_queue.add(neighbour)
                    enqueues += 1
            peak = max(peak, len(queue))
    return AlgorithmResult(canonical_partition(labels), status, pops, changes, enqueues, peak,
                           {"remaining_queue": len(queue), "queue_policy": queue_policy,
                            "weight_attribute": weight, "retain_maximal": retain})


def from_groups(groups, graph) -> dict:
    labels = {}
    for i, group in enumerate(groups):
        for node in group:
            if node in labels:
                raise ValueError("Overlapping returned partition")
            labels[node] = i
    if set(labels) != set(graph):
        raise ValueError("Returned partition does not cover graph")
    return canonical_partition(labels)


def run_method(graph: nx.Graph, method: str, seed: int, *, max_queue_pops=None,
               max_seconds=120.0, atol=1e-12, rtol=1e-12) -> AlgorithmResult:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if method == "flpa_original":
        return propagate(graph, seed=seed, reference_order=True, max_queue_pops=max_queue_pops,
                         max_seconds=max_seconds, atol=atol, rtol=rtol)
    if method.startswith("c"):
        return propagate(graph, seed=seed,
                         weight="structural_weight" if method in WEIGHTED_METHODS else None,
                         retain=method in {"c01_unit_retain", "c11_weighted_retain"},
                         queue_policy="all_changed_neighbors", max_queue_pops=max_queue_pops,
                         max_seconds=max_seconds, atol=atol, rtol=rtol)
    if method == "asyn_lpa":
        groups = nx.community.asyn_lpa_communities(graph, weight=None, seed=seed)
    elif method == "nx_weighted_flpa":
        groups = nx.community.fast_label_propagation_communities(graph, weight="structural_weight", seed=seed)
    else:
        raise ValueError("Leiden requires the prepared adapter")
    return AlgorithmResult(from_groups(groups, graph), "completed")


def prepare_leiden(graph: nx.Graph):
    import igraph as ig
    nodes = list(graph)
    index = {u: i for i, u in enumerate(nodes)}
    converted = ig.Graph(n=len(nodes), edges=[(index[u], index[v]) for u, v in graph.edges], directed=False)
    return converted, nodes


def run_leiden(prepared, seed: int) -> AlgorithmResult:
    import leidenalg as la
    converted, nodes = prepared
    partition = la.find_partition(converted, la.ModularityVertexPartition,
                                  weights=None, n_iterations=-1, seed=seed)
    labels = {node: int(partition.membership[i]) for i, node in enumerate(nodes)}
    return AlgorithmResult(canonical_partition(labels), "completed",
                           details={"objective": "standard modularity", "resolution": 1,
                                    "n_iterations": -1})
