from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml

from .algorithms import (METHODS, WEIGHTED_METHODS, apply_weights, edge_similarity,
                         prepare_leiden, run_leiden, run_method, support_violations)
from .contracts import fingerprint, write_json
from .data import load_prepared
from .evaluation import evaluate, pairwise_stability


def code_fingerprint() -> str:
    folder = Path(__file__).parent
    core = {"algorithms.py", "contracts.py", "data.py", "evaluation.py", "experiment.py"}
    return fingerprint({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(folder.glob("*.py")) if p.name in core})


def environment() -> dict:
    packages = ["networkx", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "igraph", "leidenalg"]
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "machine": platform.machine(), "packages": {p: importlib.metadata.version(p) for p in packages},
            "timing": "Sequential cold method cost; warm imports; preparation/conversion/execution measured separately",
            "utc": datetime.now(timezone.utc).isoformat(), "code_sha256": code_fingerprint()}


def select_parameter(root: Path) -> dict:
    candidate_rows = []
    for lambda_value in [0, .25, .5, 1]:
        qualities, stabilities, largest = [], [], []
        all_valid = True
        for graph_seed in range(100, 104):
            item = load_prepared(root, f"dev_sbm_{graph_seed}")
            weighted = apply_weights(item.graph, edge_similarity(item.graph), lambda_value)
            partitions = {}
            for seed in range(5):
                result = run_method(weighted, "c11_weighted_retain", seed)
                metrics = evaluate(item.graph, result.labels, item.labels)
                violations = support_violations(weighted, result.labels, "structural_weight")
                valid = result.status == "converged" and not violations
                all_valid &= valid
                run_id = f"lambda{lambda_value}_graph{graph_seed}_seed{seed}"
                write_json(root / "results" / "development" / run_id / "result.json",
                           {"graph_hash": item.graph_hash, "lambda": lambda_value,
                            "generation_seed": graph_seed, "algorithm_seed": seed,
                            "labels": result.labels, "status": result.status, "metrics": metrics,
                            "support_violations": violations})
                if valid:
                    partitions[run_id] = result.labels
                    qualities.append(metrics["ari"])
                    largest.append(metrics["largest_fraction"])
            scores = pairwise_stability(partitions)
            stabilities.append(float(np.mean([row["stability_ari"] for row in scores])) if scores else float("nan"))
        candidate_rows.append({"lambda_value": lambda_value, "mean_reference_ari": float(np.mean(qualities)),
                               "mean_stability": float(np.mean(stabilities)), "max_largest_fraction": max(largest),
                               "all_runs_valid": bool(all_valid)})
    reference = candidate_rows[0]["mean_reference_ari"]
    for row in candidate_rows:
        row["eligible"] = bool(row["all_runs_valid"] and row["mean_reference_ari"] >= reference - .02
                               and row["max_largest_fraction"] <= .95)
    eligible = [row for row in candidate_rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("No development candidate meets the predeclared quality gate")
    best_stability = max(row["mean_stability"] for row in eligible)
    selected = min(row["lambda_value"] for row in eligible if row["mean_stability"] >= best_stability - .01)
    record = {"selected_lambda": selected, "candidates": candidate_rows,
              "rule": "Mean development stability; ARI no more than .02 below lambda-zero; max group fraction <= .95; prefer smaller lambda within .01 stability",
              "development_graphs": [f"dev_sbm_{s}" for s in range(100, 104)],
              "algorithm_seeds": list(range(5)), "final_labels_used": False, "environment": environment()}
    write_json(root / "results" / "parameter_selection.json", record)
    pd.DataFrame(candidate_rows).to_csv(root / "results" / "summaries" / "parameter_selection.csv", index=False)
    print(json.dumps({"selected_lambda": selected, "candidates": candidate_rows}, indent=2), flush=True)
    return record


def read_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not config.get("datasets") or not config.get("seeds") or not config.get("methods"):
        raise ValueError("Configuration needs datasets, methods and seeds")
    if set(config["methods"]) - set(METHODS):
        raise ValueError("Unknown configured method")
    if len(config["methods"]) != len(set(config["methods"])) or len(config["datasets"]) != len(set(config["datasets"])):
        raise ValueError("Duplicate methods or datasets")
    if len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("Duplicate seeds")
    if any(not isinstance(seed, int) or seed < 0 for seed in config["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    return config


def run_experiment(root: Path, config_path: Path) -> Path:
    config = read_config(config_path)
    selection = json.loads((root / "results" / "parameter_selection.json").read_text())
    lambda_value = float(selection["selected_lambda"])
    effective = {**config, "selected_lambda": lambda_value, "timing_mode": "cold",
                 "tie_atol": 1e-12, "tie_rtol": 1e-12,
                 "queue_pop_limit": "max(10000, 1000*n)", "max_seconds_instrumented": 120,
                 "leiden_objective": "standard modularity", "leiden_iterations": -1}
    config_hash = fingerprint(effective)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / "results" / "runs" / f"{config['experiment_id']}_{attempt}"
    directory.mkdir(parents=True, exist_ok=False)
    env = environment()
    write_json(directory / "environment.json", env)
    write_json(directory / "config.json", effective)
    expected = [{"dataset_id": dataset, "method": method, "seed": seed}
                for dataset in config["datasets"] for method in config["methods"] for seed in config["seeds"]]
    write_json(directory / "manifest.json", {"config_hash": config_hash, "expected_count": len(expected), "runs": expected})
    # Warm external imports before any measured algorithm execution.
    import igraph, leidenalg  # noqa: F401
    for dataset_id in config["datasets"]:
        item = load_prepared(root, dataset_id)
        jobs = [(method, seed) for method in config["methods"] for seed in config["seeds"]]
        random.Random(config.get("schedule_seed", 2718)).shuffle(jobs)
        succeeded = 0
        for method, seed in jobs:
            run_id = f"{dataset_id}__{method}__seed{seed}"
            run_dir = directory / run_id
            run_dir.mkdir()
            row = {"run_id": run_id, "dataset_id": dataset_id, "method": method, "seed": seed,
                   "experiment_id": config["experiment_id"], "config_hash": config_hash,
                   "graph_hash": item.graph_hash, "lambda_value": lambda_value,
                   "timing_mode": "cold", "role": config.get("role", "real"),
                   "code_sha256": env["code_sha256"], "feature_prepare_s": 0.0,
                   "conversion_s": 0.0, "partition_valid": False}
            try:
                graph = item.graph
                if method in WEIGHTED_METHODS:
                    start = time.perf_counter()
                    similarity = edge_similarity(graph)
                    graph = apply_weights(graph, similarity, lambda_value)
                    row["feature_prepare_s"] = time.perf_counter() - start
                if method == "leiden_modularity":
                    start = time.perf_counter()
                    prepared = prepare_leiden(graph)
                    row["conversion_s"] = time.perf_counter() - start
                    start = time.perf_counter()
                    result = run_leiden(prepared, seed)
                else:
                    start = time.perf_counter()
                    result = run_method(graph, method, seed)
                row["algorithm_s"] = time.perf_counter() - start
                row["method_total_s"] = row["feature_prepare_s"] + row["conversion_s"] + row["algorithm_s"]
                row.update({"status": result.status, "queue_pops": result.queue_pops,
                            "label_changes": result.label_changes, "enqueues": result.enqueues,
                            "max_queue_length": result.max_queue_length})
                start = time.perf_counter()
                row.update(evaluate(item.graph, result.labels, item.labels))
                row["partition_valid"] = True
                violations = None
                if method == "flpa_original" or method.startswith("c") or method == "nx_weighted_flpa":
                    weight = "structural_weight" if method in WEIGHTED_METHODS else None
                    violations = support_violations(graph, result.labels, weight)
                    if violations and result.status in {"converged", "completed"}:
                        row["status"] = "failed_support_check"
                row["support_violations"] = len(violations) if violations is not None else None
                row["evaluation_s"] = time.perf_counter() - start
                pd.DataFrame(sorted(result.labels.items()), columns=["internal_id", "community_id"]).to_csv(run_dir / "partition.csv", index=False)
                write_json(run_dir / "diagnostics.json", {**result.details, "support_violation_nodes": violations})
                succeeded += row["status"] in {"converged", "completed"}
            except Exception as error:
                row["status"] = "failed"
                row["error"] = f"{type(error).__name__}: {error}"
                (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(run_dir / "record.json", row)
        print(f"{config['experiment_id']} / {dataset_id}: {succeeded}/{len(jobs)} successful", flush=True)
    return directory


def aggregate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    manifest_rows = []
    grouped_partitions = {}
    for directory in sorted((root / "results" / "runs").glob("*")):
        if not directory.is_dir():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        actual = list(directory.glob("*/record.json"))
        manifest_rows.append({"attempt": directory.name, "expected": manifest["expected_count"], "observed": len(actual)})
        if len(actual) != manifest["expected_count"]:
            raise RuntimeError(f"Incomplete attempt {directory}")
        for path in actual:
            row = json.loads(path.read_text())
            row["attempt"] = directory.name
            records.append(row)
            if row["status"] in {"converged", "completed"} and row["partition_valid"]:
                partition = pd.read_csv(path.parent / "partition.csv")
                labels = {int(u): int(c) for u, c in partition.itertuples(index=False, name=None)}
                key = (directory.name, row["dataset_id"], row["method"], row["lambda_value"])
                grouped_partitions.setdefault(key, {})[row["run_id"]] = labels
    frame = pd.DataFrame(records)
    output = root / "results" / "summaries"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_run_metrics.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output / "manifest_audit.csv", index=False)
    pairs, summary_rows = [], []
    keys = ["attempt", "experiment_id", "role", "dataset_id", "method", "lambda_value"]
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        identity = dict(zip(keys, key))
        valid = group[group.status.isin(["converged", "completed"]) & group.partition_valid]
        values = {**identity, "expected_runs": len(group), "successful_runs": len(valid), "failed_runs": len(group)-len(valid)}
        for metric in ["ari", "nmi", "modularity", "communities", "largest_fraction", "singleton_fraction",
                       "feature_prepare_s", "conversion_s", "algorithm_s", "method_total_s", "queue_pops", "label_changes"]:
            series = pd.to_numeric(valid[metric], errors="coerce").dropna()
            values[metric + "_mean"] = series.mean() if len(series) else np.nan
            values[metric + "_std"] = series.std(ddof=1) if len(series) > 1 else np.nan
            if metric.endswith("_s"):
                values[metric + "_median"] = series.median() if len(series) else np.nan
                values[metric + "_q25"] = series.quantile(.25) if len(series) else np.nan
                values[metric + "_q75"] = series.quantile(.75) if len(series) else np.nan
        partitions = grouped_partitions.get((key[0], key[3], key[4], key[5]), {})
        stability = pairwise_stability(partitions)
        scores = [row["stability_ari"] for row in stability]
        values["stability_mean"] = float(np.mean(scores)) if scores else np.nan
        values["stability_median"] = float(np.median(scores)) if scores else np.nan
        values["stability_pair_count"] = len(scores)
        pairs.extend({**identity, **row} for row in stability)
        summary_rows.append(values)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "method_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "stability_pairs.csv", index=False)
    comparisons = []
    for (attempt, dataset), group in frame.groupby(["attempt", "dataset_id"]):
        for treatment, control, interpretation in [
            ("c00_unit_random", "flpa_original", "scheduler"),
            ("c10_weighted_random", "c00_unit_random", "weights_without_retention"),
            ("c01_unit_retain", "c00_unit_random", "retention_without_weights"),
            ("c11_weighted_retain", "c01_unit_retain", "weights_with_retention"),
            ("c11_weighted_retain", "flpa_original", "complete_design")]:
            a = group[(group.method == treatment) & group.status.isin(["converged", "completed"])]
            b = group[(group.method == control) & group.status.isin(["converged", "completed"])]
            merged = a.merge(b, on="seed", suffixes=("_t", "_c"))
            if not len(merged):
                continue
            row = {"attempt": attempt, "dataset_id": dataset, "treatment": treatment,
                   "control": control, "interpretation": interpretation, "paired_runs": len(merged)}
            for metric in ["ari", "nmi", "modularity", "algorithm_s", "method_total_s"]:
                difference = merged[metric + "_t"] - merged[metric + "_c"]
                row[metric + "_difference_mean"] = difference.mean()
                row[metric + "_difference_std"] = difference.std(ddof=1)
            comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(output / "paired_differences.csv", index=False)
    print(f"Aggregated {len(frame)} runs into {len(summary)} groups", flush=True)
    return frame, summary
