from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd

from .algorithms import METHODS, apply_weights, edge_similarity
from .data import load_prepared

SHORT = {"flpa_original": "B0", "c00_unit_random": "C00", "c10_weighted_random": "C10",
         "c01_unit_retain": "C01", "c11_weighted_retain": "C11", "asyn_lpa": "LPA",
         "leiden_modularity": "Leiden", "nx_weighted_flpa": "NX-W"}
NAMES = {"karate": "Karate", "dolphins": "Dolphins", "football": "Football", "email_eu_core": "Email-Eu-core"}
COLORS = dict(zip(METHODS, ["#34495e", "#94a3b8", "#f59e0b", "#38bdf8", "#0f766e", "#c084fc", "#e85d75", "#8b6f47"]))


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture(path):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set(xlim=(-.3, 10), ylim=(0, 6))
    ax.axis("off")
    boxes = {"Raw data\n+ citations": (0, 4.6), "Clean graph\n+ node map": (2.5, 4.6),
             "Original FLPA\n+ comparisons": (5, 4.6), "Fixed structural\nweights": (2.5, 2.6),
             "Weighted votes\n+ retention": (5, 2.6), "Saved partitions\n+ run records": (7.5, 3.6),
             "Evaluation labels\n(separate)": (0, .5), "Quality + stability\n+ runtime": (5, .5),
             "Report + executed\nnotebook": (7.5, .5)}
    for text, (x, y) in boxes.items():
        color = "#d1fae5" if "Weighted" in text or "Quality" in text else "#e8eef7"
        ax.add_patch(FancyBboxPatch((x, y), 2, 1, boxstyle="round,pad=.08", facecolor=color, edgecolor="#1e3a5f"))
        ax.text(x + 1, y + .5, text, ha="center", va="center", fontsize=11)
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (2, 5), (4, 5), (5, 7), (6, 7), (7, 8)]
    positions = list(boxes.values())
    for a, b in edges:
        x1, y1 = positions[a]; x2, y2 = positions[b]
        if (a, b) == (5, 7):
            ax.plot([8.5, 8.5, 6], [3.5, 2, 2], color="#475569")
            ax.add_patch(FancyArrowPatch((6, 2), (6, 1.58), arrowstyle="-|>", mutation_scale=14, color="#475569"))
            continue
        if abs(y1-y2) < .1:
            start, end = (x1+2.08, y1+.5), (x2-.08, y2+.5)
        elif x1 == x2:
            start, end = (x1+1, y1-.08), (x2+1, y2+1.08)
        else:
            start, end = (x1+1.8, y1+.3), (x2+.1, y2+.7)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color="#475569", connectionstyle="arc3,rad=.08"))
    fig.suptitle("System architecture: topology drives detection; labels enter evaluation only", fontsize=13, fontweight="bold")
    save(fig, path)


def build_figures(root: Path):
    output = root / "reports" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "results" / "summaries" / "method_summary.csv")
    real = summary[summary.role == "real"]
    if real.duplicated(["dataset_id", "method"]).any():
        raise ValueError("Multiple final attempts: choose the intended attempt before plotting")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "font.family": "DejaVu Sans"})
    architecture(output / "architecture.png")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    datasets = list(NAMES)
    x = np.arange(len(datasets)); width = .095
    for ax, (metric, title) in zip(axes.flat, [("ari", "Reference ARI (Dolphins unavailable)"), ("nmi", "Reference NMI (Dolphins unavailable)"),
                                              ("modularity", "Modularity on common unweighted graph"), ("stability", "Mean cross-seed ARI (45 pairs per group)")]):
        for i, method in enumerate(METHODS):
            rows = real[real.method == method].set_index("dataset_id").reindex(datasets)
            values = rows[metric + "_mean"]
            ax.bar(x + (i - 3.5)*width, values, width, color=COLORS[method], label=SHORT[method])
        ax.set_xticks(x, [NAMES[d] for d in datasets], rotation=12)
        ax.set_title(title, fontsize=11); ax.set_ylim(bottom=-.05 if metric == "ari" else 0)
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=8, frameon=False)
    save(fig, output / "real_quality.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, dataset in zip(axes.flat, datasets):
        rows = real[real.dataset_id == dataset].set_index("method").reindex(METHODS)
        preparation = rows.feature_prepare_s_median.to_numpy() * 1000
        conversion = rows.conversion_s_median.to_numpy() * 1000
        algorithm = rows.algorithm_s_median.to_numpy() * 1000
        ax.bar(range(8), preparation, label="Weight preparation", color="#f59e0b")
        ax.bar(range(8), conversion, bottom=preparation, label="Backend conversion", color="#94a3b8")
        ax.bar(range(8), algorithm, bottom=preparation+conversion, label="Algorithm", color="#0f766e")
        ax.set_xticks(range(8), [SHORT[m] for m in METHODS], rotation=30)
        ax.set_ylabel("Milliseconds"); ax.set_title(NAMES[dataset]); ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save(fig, output / "runtime_breakdown.png")

    pairs = pd.read_csv(root / "results" / "summaries" / "stability_pairs.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, dataset in zip(axes.flat, datasets):
        groups = [pairs[(pairs.role == "real") & (pairs.dataset_id == dataset) & (pairs.method == method)].stability_ari.to_numpy() for method in METHODS]
        ax.boxplot(groups, tick_labels=[SHORT[m] for m in METHODS], showfliers=True)
        ax.set_title(NAMES[dataset]); ax.set_ylim(-.05, 1.05); ax.set_ylabel("Pairwise partition ARI")
    save(fig, output / "stability.png")

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), layout="constrained")
    for row, dataset in enumerate(["karate", "dolphins", "football"]):
        item = load_prepared(root, dataset)
        pos = nx.spring_layout(item.graph, seed=1947, iterations=100)
        for column, method in enumerate(["flpa_original", "c11_weighted_retain", "leiden_modularity"]):
            record = real[(real.dataset_id == dataset) & (real.method == method)].iloc[0]
            path = root / "results" / "runs" / record.attempt / f"{dataset}__{method}__seed0" / "partition.csv"
            partition = pd.read_csv(path).set_index("internal_id").community_id.to_dict()
            ax = axes[row, column]
            nx.draw_networkx(item.graph, pos=pos, ax=ax, with_labels=False, node_size=70 if dataset != "karate" else 130,
                             node_color=[partition[u] for u in item.graph], cmap=plt.cm.tab20,
                             edge_color="#94a3b8", width=.55)
            ax.set_title(f"{NAMES[dataset]} / {SHORT[method]} / seed 0"); ax.axis("off")
    save(fig, output / "communities.png")

    test = summary[summary.role == "test"].copy()
    if len(test):
        test["mixing"] = test.dataset_id.str.extract(r"test_sbm_(low|medium|high)_")[0]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        for ax, metric, title in zip(axes, ["ari_mean", "stability_mean", "method_total_s_median"],
                                     ["Reference ARI", "Cross-seed stability", "Median cold method time (s)"]):
            for method in METHODS:
                grouped = test[test.method == method].groupby("mixing")[metric]
                means = grouped.mean().reindex(["low", "medium", "high"])
                std = grouped.std().reindex(["low", "medium", "high"])
                ax.errorbar([0, 1, 2], means, yerr=std, label=SHORT[method], color=COLORS[method], marker="o", capsize=2, linewidth=1.4)
            ax.set_xticks([0, 1, 2], ["Low", "Medium", "High"])
            ax.set_xlabel("Between-block connection level"); ax.set_title(title, fontsize=11); ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=8, frameon=False)
        save(fig, output / "synthetic_stress.png")

    graph = nx.Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
    weighted = apply_weights(graph, edge_similarity(graph), 1)
    pos = {0: (-1, 1), 1: (-1, -1), 2: (0, 0), 3: (2, 0), 4: (3, 1), 5: (3, -1)}
    fig, ax = plt.subplots(figsize=(10, 3.5))
    nx.draw_networkx(graph, pos, ax=ax, labels=dict(enumerate("abcdef")), node_color=["#99f6e4"]*3+["#bfdbfe"]*3,
                     node_size=700, width=2)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax,
                                edge_labels={(u, v): f"{weighted[u][v]['structural_weight']:.3f}" for u, v in graph.edges})
    ax.set_title("Mechanism example at lambda = 1: bridge stays at weight 1"); ax.axis("off")
    save(fig, output / "weight_example.png")
    print(f"Saved figures to {output}", flush=True)
