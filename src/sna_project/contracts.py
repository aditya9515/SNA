from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from typing import Any

import networkx as nx


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def canonical_partition(labels: dict[int, int]) -> dict[int, int]:
    groups: dict[int, list[int]] = {}
    for node in sorted(labels):
        groups.setdefault(labels[node], []).append(node)
    ordered = sorted(groups.values(), key=lambda group: group[0])
    return {node: i for i, group in enumerate(ordered) for node in group}


def groups_from_labels(labels: dict[int, int]) -> list[set[int]]:
    groups: dict[int, set[int]] = {}
    for node, label in labels.items():
        groups.setdefault(label, set()).add(node)
    return list(groups.values())


@dataclass
class PreparedGraph:
    dataset_id: str
    graph: nx.Graph
    graph_hash: str
    metadata: dict[str, Any]
    labels: dict[int, int] | None = None
    path: Path | None = None


@dataclass
class AlgorithmResult:
    labels: dict[int, int]
    status: str
    queue_pops: int | None = None
    label_changes: int | None = None
    enqueues: int | None = None
    max_queue_length: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
