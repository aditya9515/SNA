from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import networkx as nx
import pandas as pd
import requests

from .contracts import PreparedGraph, fingerprint, write_json

DATA_SOURCES = {
    "karate": "https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html",
    "dolphins": "https://websites.umich.edu/~mejn/netdata/dolphins.zip",
    "football": "https://websites.umich.edu/~mejn/netdata/football.zip",
    "email_eu_core": "https://snap.stanford.edu/data/email-Eu-core.html",
}


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(url, timeout=90, headers={"User-Agent": "SNA-course-reproducibility/1.0"})
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def source_files(raw_dir: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(raw_dir.glob("*")) if p.is_file()}


def save_prepared(root: Path, dataset_id: str, raw_graph: nx.Graph,
                  truth: dict | None, source: str, extra: dict | None = None) -> PreparedGraph:
    original_nodes = sorted(raw_graph, key=lambda u: (type(u).__name__, str(u)))
    mapping = {u: i for i, u in enumerate(original_nodes)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(original_nodes)))
    pairs = sorted({tuple(sorted((mapping[u], mapping[v]))) for u, v in raw_graph.edges if u != v})
    graph.add_edges_from(pairs)
    labels = None
    if truth is not None:
        if set(truth) != set(raw_graph):
            raise ValueError(f"Incomplete reference labels: {dataset_id}")
        categories = {value: i for i, value in enumerate(sorted(set(truth.values()), key=str))}
        labels = {mapping[u]: categories[truth[u]] for u in original_nodes}
    graph_hash = fingerprint({"nodes": list(graph), "edges": pairs, "policy": "simple-undirected-all-nodes-v1"})
    directory = root / "data" / "processed" / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs, columns=["source", "target"]).to_csv(directory / "edges.csv", index=False)
    pd.DataFrame([{"internal_id": mapping[u], "original_id": str(u),
                   "original_name": str(raw_graph.nodes[u].get("label", u))}
                  for u in original_nodes]).to_csv(directory / "nodes.csv", index=False)
    if labels is not None:
        pd.DataFrame(sorted(labels.items()), columns=["internal_id", "reference_group"]).to_csv(directory / "labels.csv", index=False)
    elif (directory / "labels.csv").exists():
        (directory / "labels.csv").unlink()
    degrees = [degree for _, degree in graph.degree]
    metadata = {
        "dataset_id": dataset_id, "source_url": source,
        "graph_hash": graph_hash, "nodes": len(graph), "edges": graph.number_of_edges(),
        "source_nodes": len(raw_graph), "source_edges": raw_graph.number_of_edges(),
        "source_directed": raw_graph.is_directed(),
        "source_self_loops_removed": nx.number_of_selfloops(raw_graph),
        "connected_components": nx.number_connected_components(graph) if len(graph) else 0,
        "isolates": nx.number_of_isolates(graph), "density": nx.density(graph),
        "mean_degree": sum(degrees) / len(graph) if len(graph) else 0,
        "max_degree": max(degrees, default=0),
        "reference_groups": len(set(labels.values())) if labels is not None else None,
        "label_coverage": 1.0 if labels is not None else 0.0,
        "policy": "Undirected OR-symmetrisation; deduplicate; remove loops; retain every source node; discard source weights for common topology",
        "raw_sha256": source_files(root / "data" / "raw" / dataset_id),
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    write_json(directory / "metadata.json", metadata)
    return PreparedGraph(dataset_id, graph, graph_hash, metadata, labels, directory)


def prepare_real(root: Path, dataset_id: str) -> PreparedGraph:
    raw_dir = root / "data" / "raw" / dataset_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth = None
    extra = {}
    if dataset_id == "karate":
        raw = nx.karate_club_graph()
        truth = nx.get_node_attributes(raw, "club")
        write_json(raw_dir / "networkx_graph.json", nx.node_link_data(raw))
        extra["reference_label_source"] = "NetworkX club node attribute; source interaction weights excluded"
    elif dataset_id in {"dolphins", "football"}:
        archive = download(DATA_SOURCES[dataset_id], raw_dir / f"{dataset_id}.zip")
        with zipfile.ZipFile(archive) as zf:
            gml = next(name for name in zf.namelist() if name.lower().endswith(".gml"))
            raw_bytes = zf.read(gml)
            (raw_dir / Path(gml).name).write_bytes(raw_bytes)
            for name in zf.namelist():
                if name.lower().endswith(".txt"):
                    (raw_dir / Path(name).name).write_bytes(zf.read(name))
        raw = nx.parse_gml(raw_bytes.decode("ascii").splitlines(), label=None)
        if dataset_id == "football":
            truth = nx.get_node_attributes(raw, "value")
            extra["reference_label_source"] = "Original football.gml value attribute (conference)"
            extra["version_note"] = "Measured source version; 613 edges in this Newman archive, not the 616 reported in the FLPA paper"
        else:
            extra["reference_label_source"] = "Unavailable: source supplies names, not a validated two-group reference partition"
    elif dataset_id == "email_eu_core":
        edges_path = download("https://snap.stanford.edu/data/email-Eu-core.txt.gz", raw_dir / "email-Eu-core.txt.gz")
        labels_path = download("https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz", raw_dir / "department-labels.txt.gz")
        truth = {}
        with gzip.open(labels_path, "rt") as stream:
            for line in stream:
                node, group = map(int, line.split())
                truth[node] = group
        raw = nx.DiGraph()
        raw.add_nodes_from(sorted(truth))
        with gzip.open(edges_path, "rt") as stream:
            raw.add_edges_from(tuple(map(int, line.split())) for line in stream if line.strip() and not line.startswith("#"))
        extra["reference_label_source"] = "SNAP department-labels file; departments are proxy groups, not unique structural truth"
    else:
        raise ValueError(f"Unknown real dataset {dataset_id}")
    return save_prepared(root, dataset_id, raw, truth, DATA_SOURCES[dataset_id], extra)


def prepare_sbm(root: Path, dataset_id: str, *, seed: int, sizes: list[int], p_in: float, p_out: float,
                role: str) -> PreparedGraph:
    probabilities = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
    raw = nx.stochastic_block_model(sizes, probabilities, seed=seed)
    truth = nx.get_node_attributes(raw, "block")
    raw_dir = root / "data" / "raw" / dataset_id
    write_json(raw_dir / "generation.json", {"seed": seed, "sizes": sizes, "p_in": p_in,
                                            "p_out": p_out, "networkx_version": nx.__version__})
    return save_prepared(root, dataset_id, raw, truth, "Locally generated NetworkX stochastic_block_model",
                         {"role": role, "generation_seed": seed, "sizes": sizes,
                          "p_in": p_in, "p_out": p_out,
                          "reference_label_source": "Planted stochastic block membership"})


def load_prepared(root: Path, dataset_id: str) -> PreparedGraph:
    directory = root / "data" / "processed" / dataset_id
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    nodes = pd.read_csv(directory / "nodes.csv")
    edges = pd.read_csv(directory / "edges.csv")
    graph = nx.Graph()
    graph.add_nodes_from(int(n) for n in nodes.internal_id)
    graph.add_edges_from((int(u), int(v)) for u, v in edges.itertuples(index=False, name=None))
    actual_hash = fingerprint({"nodes": list(graph), "edges": sorted(tuple(sorted(e)) for e in graph.edges),
                               "policy": "simple-undirected-all-nodes-v1"})
    if actual_hash != metadata["graph_hash"]:
        raise ValueError(f"Graph hash mismatch: {dataset_id}")
    labels = None
    if (directory / "labels.csv").exists():
        frame = pd.read_csv(directory / "labels.csv")
        labels = {int(u): int(c) for u, c in frame.itertuples(index=False, name=None)}
        if set(labels) != set(graph):
            raise ValueError("Reference-label coverage mismatch")
    return PreparedGraph(dataset_id, graph, actual_hash, metadata, labels, directory)


def prepare_all(root: Path) -> list[PreparedGraph]:
    prepared = []
    for dataset in DATA_SOURCES:
        item = prepare_real(root, dataset)
        print(f"Prepared {dataset}: {len(item.graph)} nodes / {item.graph.number_of_edges()} edges", flush=True)
        prepared.append(item)
    for seed in range(100, 104):
        prepared.append(prepare_sbm(root, f"dev_sbm_{seed}", seed=seed, sizes=[50] * 4,
                                    p_in=.15, p_out=.01, role="development"))
    for level, p_out in [("low", .005), ("medium", .025), ("high", .06)]:
        for graph_seed in range(200, 203):
            prepared.append(prepare_sbm(root, f"test_sbm_{level}_{graph_seed}", seed=graph_seed,
                                        sizes=[100] * 4, p_in=.16, p_out=p_out, role="test"))
    summary = [{k: value for k, value in item.metadata.items() if not isinstance(value, (dict, list))} for item in prepared]
    (root / "results" / "summaries").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(root / "results" / "summaries" / "dataset_summary.csv", index=False)
    return prepared
