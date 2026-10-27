"""Author the complete inline project and execute it in a folder containing only it.

Source extraction happens at authoring time. The delivered notebook contains
ordinary visible definitions and never imports/reads/executes the source package.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import textwrap

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NAME = "SNA_Project_with_Results.ipynb"
HASH_MARKER = "__INLINE_IMPLEMENTATION_HASH__"


def definitions(root, module, *names):
    """Copy full definitions, including decorators, into literal notebook cells."""
    source = (root / "src/sna_project" / module).read_text(encoding="utf-8")
    lines = source.splitlines()
    nodes = {n.name: n for n in ast.parse(source).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    result = []
    for name in names:
        node = nodes[name]
        first = min([node.lineno, *[d.lineno for d in node.decorator_list]])
        result.append("\n".join(lines[first-1:node.end_lineno]))
    return "\n\n\n".join(result)


def assignment(root, module, name):
    source = (root / "src/sna_project" / module).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.get_source_segment(source, node)
    raise KeyError((module, name))


def check_inline(notebook):
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith("sna_project") for n in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not node.level and not (node.module or "").startswith("sna_project")
            if isinstance(node, ast.Call):
                assert not (isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "__import__"})
        assert "sys.path.insert" not in cell.source
        assert "src/sna_project" not in cell.source
        assert "get_ipython().run_line_magic" not in cell.source


def build(root=ROOT, execute=True):
    root = root.resolve()
    cells = []

    def prose(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def step(title, why, inputs, outputs, source, details="", hidden_data=False):
        number = sum(c.cell_type == "code" for c in cells) + 1
        prose(f"## Step {number}. {title}\n\n{why}\n\n**Inputs:** {inputs}  \n**Produces:** {outputs}\n\n{details}")
        cell = nbf.v4.new_code_cell(textwrap.dedent(source).strip())
        cell.metadata["teaching"] = {"step": number, "title": title, "why": why,
                                     "inputs": inputs, "outputs": outputs, "details": details}
        if hidden_data:
            cell.metadata["jupyter"] = {"source_hidden": True}
        cells.append(cell)

    prose(r"""
    # Stability-Aware Structural Weighting for Fast Label Propagation
    ## Complete implementation, step by step, with executed results

    **DSE 4403 / Social Network Analysis**  
    Aditya Kosuru (230968316), Jaswanth P (230968014), Prithvi Ram Perivela (230968304)

    This notebook contains the full data preparation, graph algorithms, parameter selection,
    experiment runner, evaluation, aggregation, plotting and verification code **directly in cells**.
    Run all cells in order to perform **80 development runs and 1,040 final runs**.
    Each displayed experiment table and plot is calculated during that execution.

    The only code imports are standard Python and third-party libraries such as NetworkX,
    NumPy, pandas, scikit-learn, matplotlib, igraph and leidenalg. No project Python module,
    source directory, CLI command, dynamic source execution or previously computed result
    is required. Dataset snapshots and the bibliography are embedded in this file.
    After installing the libraries, this notebook can run by itself in an otherwise empty folder.

    **Run:** install the supplied pinned dependencies, select a Python 3.12 Jupyter kernel,
    and choose Run All. Existing outputs can be read immediately. A fresh temporary output
    directory is created for each run; its path and evidence ZIP are printed at the end.
    """)
    prose(r"""
    ### Research foundation and proposed contribution

    The base paper is Traag and Šubelj (2023), [*Large network community detection by fast label
    propagation*](https://doi.org/10.1038/s41598-023-29610-z), Scientific Reports 13, 2701.
    Its Algorithm 3 supplies the original asynchronous FLPA update and filtered work queue.
    Retaining an already maximal label is also an established strategy discussed in that paper.

    Our extension assigns an existing edge the fixed topology-derived weight
    $w(u,v)=1+\lambda |N(u)\cap N(v)|/|N(u)\cup N(v)|$, then studies weighted votes,
    retention and scheduling/order controls separately. The contribution is the specific
    implementation and controlled evaluation. Jaccard similarity, weighted propagation and
    retention are established ingredients; global originality of their combination is unproven.
    The results include tradeoffs and failures, rather than universal improvement.

    **There is no neural-model training.** Community IDs emerge from unsupervised neighbour
    updates. Lambda selection uses planted labels on separate development graphs; final labels
    are used for scoring only. Papers provide methods/context; the network datasets provide
    actual nodes and edges. Reviewing a neural paper does not mean its model is implemented.

    ```text
    embedded raw graphs -> cleaned graph + node map -> FLPA / weighted variants / comparisons
                                      |                              |
                             separate references                predictions
                                      +--------------+---------------+
                                                     v
                                          quality / stability / time
                                                     v
                                          tables / plots / evidence
    ```
    """)
    step("Import external libraries and create a fresh working directory",
         "All project functions are defined below. This cell imports only the libraries they need. A new output directory keeps repeated executions separate.",
         "Python environment and installed dependencies; no project files.",
         "Library aliases, team metadata and RUN_ROOT.", r"""
         from __future__ import annotations
         from collections import Counter, defaultdict, deque
         from dataclasses import dataclass, field
         from datetime import datetime, timezone
         from itertools import combinations
         from pathlib import Path
         from typing import Any
         import base64, copy, gzip, hashlib, importlib.metadata, io, json, math
         import platform, random, sys, tempfile, time, traceback, zipfile
         import numpy as np
         import pandas as pd
         import networkx as nx
         import requests
         import yaml
         import matplotlib
         matplotlib.use("Agg")
         import matplotlib.pyplot as plt
         from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
         from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
         from IPython.display import display, Image, JSON

         RUN_ROOT = Path(tempfile.mkdtemp(prefix="sna_full_notebook_"))
         pd.set_option("display.max_columns", 20)
         pd.set_option("display.max_rows", 120)
         TEAM = [
             {"name": "Aditya Kosuru", "registration_number": "230968316"},
             {"name": "Jaswanth P", "registration_number": "230968014"},
             {"name": "Prithvi Ram Perivela", "registration_number": "230968304"},
         ]
         print("Python:", sys.version.split()[0], "NetworkX:", nx.__version__)
         print("Fresh execution directory:", RUN_ROOT)
         display(pd.DataFrame(TEAM))
         """, "`nx` stores graphs, `pd` handles CSV tables, `np` handles numerical checks, and sklearn supplies evaluation metrics. igraph/leidenalg are imported in their comparison adapters. `tempfile` gives each execution a separate directory.")
    step("Define shared records, canonical labels and JSON/hash helpers",
         "Every stage uses the same explicit objects. Canonical numbering changes arbitrary group IDs without changing membership.",
         "Python dictionaries, graph objects and paths.", "PreparedGraph, AlgorithmResult, partition and artifact helpers.",
         definitions(root, "contracts.py", "fingerprint", "write_json", "canonical_partition", "groups_from_labels", "PreparedGraph", "AlgorithmResult"),
         "`PreparedGraph.labels` is the evaluation reference; `AlgorithmResult.labels` is the detector prediction. Atomic JSON writing avoids partial records. Graph hashes include isolates and edges.")

    raw_files = ["dolphins/dolphins.zip", "football/football.zip", "email_eu_core/email-Eu-core.txt.gz",
                 "email_eu_core/department-labels.txt.gz"]
    payload = {}
    for name in raw_files:
        raw = (root / "data/raw" / name).read_bytes()
        payload[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "base64": base64.b64encode(raw).decode("ascii")}
    payload_code = "# Encoded dataset bytes, not executable implementation. Expand to inspect.\nEMBEDDED_RAW_DATA = " + repr(payload)
    step("Include exact raw-data snapshots and source citations",
         "Embedding the four compressed source files makes this notebook runnable without a separate data folder or a network download.",
         "Newman Dolphins/Football archives and SNAP Email-Eu-core edge/department files.",
         "EMBEDDED_RAW_DATA: compressed source bytes and SHA-256 hashes.", payload_code,
         "This data-only cell may be collapsed by Jupyter to keep the long encoded strings out of the way; all implementation cells remain visible. Base64 encodes bytes, not Python code. Karate is provided by NetworkX. Sources: https://websites.umich.edu/~mejn/netdata/ and https://snap.stanford.edu/data/email-Eu-core.html .", hidden_data=True)
    step("Restore raw data and define the acquisition helpers",
         "Verify embedded bytes before writing them. The download helper also documents acquisition from source URLs, with bundled snapshots already available for this run.",
         "EMBEDDED_RAW_DATA and source URLs.", "RUN_ROOT/data/raw and download/source-hash helpers.",
         assignment(root, "data.py", "DATA_SOURCES") + "\n\n" + definitions(root, "data.py", "download", "source_files") + r'''

for relative_name, snapshot in EMBEDDED_RAW_DATA.items():
    restored = base64.b64decode(snapshot["base64"], validate=True)
    assert hashlib.sha256(restored).hexdigest() == snapshot["sha256"]
    destination = RUN_ROOT / "data/raw" / relative_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(restored)
print("Restored and verified", len(EMBEDDED_RAW_DATA), "raw source files.")
''', "`download()` requests a source only if its target is missing. Embedded snapshots cover every file requested here, so this execution reads the same source versions offline.")
    step("Implement graph cleaning and separate reference labels",
         "All detectors need the same simple, undirected topology. Keeping references outside the graph prevents label leakage.",
         "A raw graph, optional reference dictionary and source metadata.",
         "nodes.csv, edges.csv, optional labels.csv, metadata.json and PreparedGraph.",
         definitions(root, "data.py", "save_prepared"),
         "Retain all nodes; remove loops; OR-symmetrise directed links; deduplicate; discard original weights. Original IDs are sorted by type and string, so internal ID 2 need not be original ID 2. `nodes.csv` preserves the mapping. `labels.csv` is for scoring only.")
    step("Implement real-network loading and label extraction",
         "Each source has its own file format and reference semantics; make those choices explicit before detection.",
         "NetworkX Karate; embedded ZIP/GML and gzip text files.", "Four prepared real graphs with separate optional references.",
         definitions(root, "data.py", "prepare_real"),
         "Karate uses the original club attribute, Football the conference `value`, and email the department file. Dolphins has no validated two-group labels in this archive; its ARI/NMI remain unavailable. Football has 613 edges here, compared with 616 reported in the base paper.")
    step("Implement synthetic generation, processed-file loading and preparation",
         "Seeded SBM graphs provide planted reference groups and controlled difficulty. A hash check protects the graph reconstructed from CSVs.",
         "Graph-generation settings and raw/prepared files.", "prepare_sbm(), load_prepared(), prepare_all().",
         definitions(root, "data.py", "prepare_sbm", "load_prepared", "prepare_all"),
         "Development: four groups of 50, p_in=.15, p_out=.01, graph seeds 100–103. Final: four groups of 100, p_in=.16, p_out=.005/.025/.06, seeds 200–202 per level. These graph seeds differ from detector seeds 0–9. SBM implements the chosen stress study; optional LFR was not implemented.")
    step("Execute preparation and inspect actual nodes, edges and references",
         "Run the preceding functions now, then inspect their output before any detector sees it.",
         "Embedded raw snapshots and fixed SBM settings.", "17 prepared graphs and a dataset summary; data previews.", r'''
prepared = prepare_all(RUN_ROOT)
datasets = pd.read_csv(RUN_ROOT / "results/summaries/dataset_summary.csv")
real_ids = ["karate", "dolphins", "football", "email_eu_core"]
display(datasets[datasets.dataset_id.isin(real_ids)][[
    "dataset_id", "nodes", "edges", "connected_components", "isolates", "reference_groups", "label_coverage"]])
assert len(prepared) == 17
assert [(len(p.graph), p.graph.number_of_edges()) for p in prepared[:4]] == [(34,78),(62,159),(115,613),(1005,16064)]
assert all(not attrs for p in prepared for _, attrs in p.graph.nodes(data=True))
for filename in ["nodes.csv", "edges.csv", "labels.csv"]:
    print("Karate", filename)
    display(pd.read_csv(RUN_ROOT / "data/processed/karate" / filename).head())
print("Reference labels are separate; algorithm graph node attributes are empty.")
''', "Email starts with 25,571 directed edges and becomes 16,064 undirected edges, retaining 19 isolates. Adding nodes before edges is essential to retain them. The three previews show identity mapping, topology and evaluation answers as separate files.")
    step("Declare the eight methods and graph requirements",
         "Named configurations make weighting, ties and scheduling choices auditable instead of changing several rules silently.",
         "A simple undirected graph with no self-loops.", "Method names, weighted-method set and graph validation.",
         assignment(root, "algorithms.py", "METHODS") + "\n" + assignment(root, "algorithms.py", "WEIGHTED_METHODS") + "\n\n" + definitions(root, "algorithms.py", "validate_graph"),
         "B0 is original FLPA; C00/C10/C01/C11 are the controlled unit/weighted × random/retain variants. LPA, Leiden and NetworkX weighted FLPA are comparison methods. Only weighted methods read the structural edge attribute.")
    step("Implement Jaccard similarity and fixed structural weights",
         "Shared neighbours give a transparent local feature. Adding 1 keeps bridge edges active, and lambda 0 recovers unit weights.",
         "Graph topology and non-negative lambda.", "Edge similarities and a weighted graph copy.",
         definitions(root, "algorithms.py", "edge_similarity", "apply_weights"),
         r"For an existing edge, $J=|N(u)\cap N(v)|/|N(u)\cup N(v)|$ and $w=1+\lambda J$. Neighbour sets are open and independent of predicted labels. At lambda .25 weights lie within [1,1.25]. Deep copying preserves adjacency order, preventing feature preparation from changing the seeded trajectory.")
    step("Implement neighbour votes, ties and the stopping-condition check",
         "The same support calculation must drive updates and independently check the final local condition.",
         "Graph, current labels, optional weight attribute and numerical tolerance.", "Support dictionaries, maximal candidates and violation lists.",
         definitions(root, "algorithms.py", "label_support", "maximal_candidates", "support_violations"),
         "Unit votes use Counter. Weighted votes use math.fsum. Weighted maximality allows atol=rtol=1e-12; integer unit votes use exact comparisons. A local support check does not establish global optimality or useful communities.")
    step("Implement the complete queue-based propagation algorithm",
         "This is the actual FLPA implementation and all extension switches, directly in the notebook.",
         "Graph, seed, weight/retention/queue settings and limits.", "AlgorithmResult with assignments, termination and queue diagnostics.",
         definitions(root, "algorithms.py", "propagate"),
         "Start with a unique label per node. Shuffle using a local seeded RNG. Pop FIFO work, sum neighbour votes, retain a maximal current label when enabled or sample a maximal candidate, update immediately and requeue affected neighbours. `in_queue` prevents duplicate work. Isolates keep their initial label. Limits produce explicit failure statuses, not silently successful results.")
    step("Implement partition conversion and method dispatch",
         "Use one checked output format for our implementation and library comparisons, while making each switch visible.",
         "Graph, method ID and algorithm seed.", "Canonical labels and configured AlgorithmResult.",
         definitions(root, "algorithms.py", "from_groups", "run_method"),
         "B0 uses reference candidate order and the original filtered queue. C variants sort candidates and use all-changed-neighbours scheduling. Thus C00–B0 includes candidate enumeration as well as scheduling; C10–C00 and C11–C01 cleanly hold those choices fixed while changing weights. No reference-label argument is accepted.")
    step("Implement the Leiden comparison adapters",
         "An established modularity optimiser supplies a stronger comparison, including cases where propagation collapses.",
         "The same unweighted topology and a detector seed.", "igraph conversion and canonical Leiden assignments.",
         definitions(root, "algorithms.py", "prepare_leiden", "run_leiden"),
         "The mapping preserves all nodes, including isolates. Standard modularity is used, with unit edges, resolution 1 and n_iterations=-1. igraph/leidenalg are external libraries; the notebook contains the entire adapter rather than importing project code. This optimisation is not neural training.")
    step("Implement quality and group-size evaluation",
         "Accuracy against metadata, structural modularity and collapse diagnostics measure different properties and must be reported together.",
         "Original graph, predicted assignments and optional separate reference groups.", "ARI, NMI, modularity, number of groups and group-size fractions.",
         definitions(root, "evaluation.py", "evaluate"),
         "ARI/NMI are label-number invariant and missing for Dolphins. Modularity is always computed on the same original unweighted graph, even for weighted detectors. The node sets must match exactly. Near-one largest_fraction flags giant-group collapse; singleton_fraction measures the fraction of nodes in one-node groups.")
    step("Implement cross-seed stability",
         "Compare predictions with other predictions on the same graph to measure repeatability independently of reference quality.",
         "A dictionary of run IDs to predicted partitions.", "One ARI score per unordered output pair.",
         definitions(root, "evaluation.py", "pairwise_stability"),
         "Ten seeds give 10×9/2=45 pairs. They share runs and are dependent descriptive observations. One identical collapsed group can score stability 1, so stability is never sufficient evidence of useful detection.")
    step("Run baseline parity and mechanism checks",
         "Check the implementation before interpreting its full experiments.",
         "Five graph families, ten seeds, a toy bridge graph and control cases.", "50 exact baseline parity cases plus weighting/control checks.", r'''
parity_cases = 0
for parity_graph in [nx.path_graph(8), nx.cycle_graph(10), nx.complete_graph(8),
                     nx.star_graph(8), nx.karate_club_graph()]:
    for seed in range(10):
        own = run_method(parity_graph, "flpa_original", seed)
        reference = from_groups(nx.community.fast_label_propagation_communities(
            parity_graph, weight=None, seed=seed), parity_graph)
        assert own.status == "converged" and own.labels == reference
        assert not support_violations(parity_graph, own.labels)
        parity_cases += 1
example = nx.Graph([(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),(2,3)])
example_similarity = edge_similarity(example)
assert np.isclose(example_similarity[(0,1)], 1/3)
assert np.isclose(example_similarity[(0,2)], 1/4)
assert example_similarity[(2,3)] == 0
display(pd.DataFrame([{"edge": str(e), "Jaccard": j, "weight_lambda_1": 1+j}
                      for e,j in example_similarity.items()]))
control_graph = load_prepared(RUN_ROOT, "karate").graph
zero_graph = apply_weights(control_graph, edge_similarity(control_graph), 0)
for seed in range(5):
    for weighted_name, unit_name in [("c10_weighted_random","c00_unit_random"),
                                    ("c11_weighted_retain","c01_unit_retain")]:
        a,b = run_method(zero_graph,weighted_name,seed),run_method(control_graph,unit_name,seed)
        assert a.labels == b.labels and a.queue_pops == b.queue_pops
assert len(set(run_method(nx.empty_graph(3),"flpa_original",0).labels.values())) == 3
print("Passed", parity_cases, "exact NetworkX baseline comparisons, toy weights, lambda-zero controls and isolates.")
''', "The toy table uses lambda 1 only to show arithmetic clearly. It is not a source network or final performance result. Shared-neighbour edge (0,1) has Jaccard 1/3; bridge (2,3) has zero overlap and stays at weight 1.")
    step("Record the inline implementation and execution environment",
         "Run records need an identifier for the code actually executed and the external package versions.",
         "The visible code-cell sources and installed libraries.", "Implementation hash and environment recorder.",
         f'INLINE_IMPLEMENTATION_SHA256 = "{HASH_MARKER}"\n\ndef code_fingerprint() -> str:\n    return INLINE_IMPLEMENTATION_SHA256\n\n\n' + definitions(root, "experiment.py", "environment"),
         "The authoring tool computes this hash over the inline code cells with the hash literal normalised, and verifies source equivalence for the unchanged algorithms. It records notebook code provenance separately from the companion package's source-file fingerprint. Timings remain machine dependent.")
    step("Implement development-only lambda selection",
         "Use separate planted development graphs, not final labels, to choose the one scalar controlling the structural feature.",
         "Four development graphs, candidates [0,.25,.5,1] and algorithm seeds 0–4.", "80 development records and the frozen selection.",
         definitions(root, "experiment.py", "select_parameter"),
         "A candidate must have all runs valid, mean reference ARI within .02 of lambda zero and largest group fraction <=.95. Choose maximum mean stability, preferring the smallest lambda within .01 of the best. The detector never receives block labels; only development scoring uses them.")
    step("Execute all 80 development experiments",
         "Actually compute the candidate scores here rather than loading a saved selection table.",
         "Inline selection function and newly prepared development graphs.", "A newly measured candidate table and LAMBDA.", r'''
selection = select_parameter(RUN_ROOT)
LAMBDA = selection["selected_lambda"]
display(pd.DataFrame(selection["candidates"]))
development_count = len(list((RUN_ROOT / "results/development").glob("*/result.json")))
assert development_count == 4*4*5 == 80
assert selection["final_labels_used"] is False
eligible = [c for c in selection["candidates"] if c["eligible"]]
best = max(c["mean_stability"] for c in eligible)
assert LAMBDA == min(c["lambda_value"] for c in eligible if c["mean_stability"] >= best-.01)
print("Computed development runs:", development_count, "Frozen lambda:", LAMBDA)
''', "The recorded study selects .25. The code checks the declared selection rule instead of hard-coding .25 as the answer. This is hyperparameter selection, not neural-model training.")
    step("Declare final experiment configurations and validate them",
         "Fix dataset/method/seed choices before execution, and reject duplicate or invalid jobs.",
         "Four real IDs, nine final SBM IDs, eight method IDs and ten algorithm seeds.", "Two explicit YAML configurations and read_config().",
         definitions(root, "experiment.py", "read_config") + r'''

real_config = {"experiment_id":"notebook_real_v1", "role":"real", "datasets":real_ids,
               "methods":METHODS, "seeds":list(range(10)), "schedule_seed":2718}
stress_config = {"experiment_id":"notebook_synthetic_v1", "role":"test",
                 "datasets":[f"test_sbm_{level}_{seed}" for level in ["low","medium","high"] for seed in range(200,203)],
                 "methods":METHODS, "seeds":list(range(10)), "schedule_seed":2718}
(RUN_ROOT / "configs").mkdir()
for name,config in [("real",real_config),("stress",stress_config)]:
    config_path = RUN_ROOT / "configs" / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False),encoding="utf-8")
    assert read_config(config_path) == config
display(pd.DataFrame([
    {"role":"real","graphs":4,"methods":8,"seeds":10,"runs":4*8*10},
    {"role":"synthetic test","graphs":9,"methods":8,"seeds":10,"runs":9*8*10}]))
''', "The schedule seed shuffles invocation order; it does not create graph topology or determine a detector's tie choices. C10–C00 and C11–C01 isolate weights; C01–C00 isolates retention under unit votes. B0–C11 is a complete-design comparison.")
    step("Implement the complete seeded experiment runner",
         "Compute weights, backend conversion, detection, evaluation and artifacts in a transparent per-job loop.",
         "Newly prepared data, frozen lambda and YAML config.", "Expected manifests, partitions, metrics, timings, diagnostics and environment records.",
         definitions(root, "experiment.py", "run_experiment"),
         "Cold method cost = feature preparation + backend conversion + algorithm. Common loading, scoring, plotting and I/O are excluded. Each method/seed is measured once, sequentially, after warm library imports. Failures are recorded explicitly. The support check uses the method's weight rule, and scoring always uses the original unweighted topology.")
    step("Execute the complete real-network experiment",
         "Run every method/seed combination on every real graph now.",
         "4 graphs × 8 methods × 10 seeds and the frozen lambda.", "320 new real-network experiment records.", r'''
real_attempt = run_experiment(RUN_ROOT, RUN_ROOT / "configs/real.yaml")
real_records = list(real_attempt.glob("*/record.json"))
assert len(real_records) == 320
print("Computed real runs:",len(real_records),"in",real_attempt)
''', "These are freshly generated assignments. They are not read from the previously supplied result tables.")
    step("Execute the complete synthetic stress experiment",
         "Test recovery as between-block connectivity increases, using final graph instances excluded from parameter selection.",
         "9 graphs × 8 methods × 10 seeds and the same frozen lambda.", "720 new synthetic experiment records.", r'''
synthetic_attempt = run_experiment(RUN_ROOT, RUN_ROOT / "configs/stress.yaml")
synthetic_records = list(synthetic_attempt.glob("*/record.json"))
assert len(synthetic_records) == 720
print("Computed synthetic runs:",len(synthetic_records),"in",synthetic_attempt)
''', "Low/medium/high p_out values are .005/.025/.06 with p_in=.16. A high-mixing collapse is a measured failure case, even when outputs are perfectly stable across seeds.")
    step("Implement aggregation, pairwise stability and paired ablations",
         "Trace every reported mean back to complete saved run records and predicted assignments.",
         "New attempt manifests, per-run record.json and partition.csv files.", "Per-run/method summaries, stability pairs, manifest audit and paired differences.",
         definitions(root, "experiment.py", "aggregate"),
         "Only successful valid outputs enter performance summaries; expected/failed counts remain visible. Time summaries include median and quartiles. Stability compares every partition pair within a graph/method. Treatment/control differences are paired on detector seed; positive time difference means slower treatment. C00–B0 also changes candidate order.")
    step("Compute and check the fresh final summaries",
         "Aggregate this execution's 1,040 jobs and verify completeness before plotting.",
         "The two newly executed attempts.", "runs, summary, audit and a real-network result table.", r'''
runs, summary = aggregate(RUN_ROOT)
audit = pd.read_csv(RUN_ROOT / "results/summaries/manifest_audit.csv")
assert len(runs) == audit.expected.sum() == audit.observed.sum() == 1040
assert len(summary) == 104
assert runs.status.isin(["completed","converged"]).all() and runs.partition_valid.all()
assert not runs.duplicated(["dataset_id","method","seed"]).any()
real = summary[summary.role.eq("real")]
display(audit)
display(real[["dataset_id","method","successful_runs","ari_mean","ari_std","nmi_mean",
              "modularity_mean","stability_mean","communities_mean","largest_fraction_mean","method_total_s_median"]].round(5))
''', "There are 13 final graphs × 8 methods =104 summary rows, each based on ten seeds. This table contains fresh timings. Deterministic graph/seed results can be compared to the earlier report, while measured durations naturally differ between executions.")
    step("Independently recompute every new partition's scores",
         "Verify the recorded metrics from graph topology and assignments rather than merely printing stored values.",
         "All 1,040 new partition CSVs, reconstructed graphs and separate references.", "A count of verified partitions and a reproduced stability mean.", r'''
loaded = {dataset_id:load_prepared(RUN_ROOT,dataset_id) for dataset_id in runs.dataset_id.unique()}
checked = 0
for row in runs.itertuples(index=False):
    assignment = pd.read_csv(RUN_ROOT / "results/runs" / row.attempt / row.run_id / "partition.csv")
    labels = {int(u):int(c) for u,c in assignment.itertuples(index=False,name=None)}
    graph_item = loaded[row.dataset_id]
    metrics = evaluate(graph_item.graph, labels, graph_item.labels)
    for metric in ["ari","nmi","modularity","communities","largest_fraction","singleton_fraction"]:
        actual,stored = metrics[metric],getattr(row,metric)
        assert (actual is None and pd.isna(stored)) or np.isclose(actual,stored,atol=1e-12,rtol=1e-12)
    checked += 1
stability_partitions = {}
for row in runs[(runs.dataset_id=="karate") & (runs.method=="c11_weighted_retain")].itertuples(index=False):
    frame = pd.read_csv(RUN_ROOT / "results/runs" / row.attempt / row.run_id / "partition.csv")
    stability_partitions[row.run_id] = {int(u):int(c) for u,c in frame.itertuples(index=False,name=None)}
stability_check = pd.DataFrame(pairwise_stability(stability_partitions))
assert len(stability_check) == 45
stored_stability = real[(real.dataset_id=="karate") & (real.method=="c11_weighted_retain")].iloc[0].stability_mean
assert np.isclose(stability_check.stability_ari.mean(),stored_stability,atol=1e-12,rtol=1e-12)
print("Recomputed and verified",checked,"final partitions; Karate C11 stability recomputed from 45 pairs.")
''', "Dolphins missing reference metrics are compared as None/NaN rather than zero. This verification does not reproduce timing measurements; it checks partition-derived scores. The stability example compares predictions with each other, not with club labels.")

    step("Define plotting helpers and draw the architecture",
         "The remaining figures are constructed from the new experiment outputs, with their full drawing code visible.",
         "Current summaries and inline plotting code.", "Plot styling, output directory and a new architecture image.",
         "\n".join(assignment(root,"figures.py",n) for n in ["SHORT","NAMES","COLORS"]) + "\n\n" + definitions(root,"figures.py","save","architecture") + r'''

output = RUN_ROOT / "reports/figures"
output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                    "axes.titleweight":"bold","font.family":"DejaVu Sans"})
architecture(output / "architecture.png")
display(Image(filename=str(output / "architecture.png")))
''', "The save helper exports a white-background PNG and closes the matplotlib figure. Architecture arrows distinguish detection inputs from evaluation labels.")

    figure_source = definitions(root,"figures.py","build_figures")
    body = textwrap.dedent(figure_source.split("\n",1)[1])
    sections = [
        ("Compare real-network quality, modularity and stability", "fig, axes = plt.subplots(2, 2, figsize=(13, 8)", "real_quality.png",
         "ARI/NMI compare with available references; modularity uses the common original graph. Stability measures repeatability, so inspect it beside quality and largest-group size."),
        ("Plot the measured runtime components", "fig, axes = plt.subplots(2, 2, figsize=(12, 8)", "runtime_breakdown.png",
         "Preparation, backend conversion and detector costs are measured separately. Component medians need not sum to the median total. Python/compiled comparisons and tiny-graph overhead limit speed claims."),
        ("Plot the distribution of cross-seed stability", "pairs = pd.read_csv", "stability.png",
         "Each boxplot uses the 45 dependent ARI pair comparisons from ten seeds. Consistent collapse can look perfectly stable."),
        ("Draw actual predicted communities using a common layout", "fig, axes = plt.subplots(3, 3", "communities.png",
         "Each graph gets one fixed spring layout so visual positions do not change across methods. These are seed-0 examples, while the metric tables summarise ten seeds. Email remains in the quantitative tables."),
        ("Plot the synthetic stress test", 'test = summary[summary.role == "test"]', "synthetic_stress.png",
         "The plots average the three graph-instance summaries at each mixing level. Error bars describe variation across those graphs; they are not significance-test results."),
        ("Draw the structural-weight mechanism", "graph = nx.Graph([(0, 1)", "weight_example.png",
         "The illustrative graph has two triangles joined by a bridge. This plot uses lambda 1 for visible edge arithmetic, whereas final experiments use the selected lambda."),
    ]
    positions = [body.index(marker) for _,marker,_,_ in sections]
    for i,(title,marker,filename,details) in enumerate(sections):
        part = body[positions[i]:positions[i+1] if i+1<len(sections) else body.index('print(f"Saved figures')].rstrip()
        part = re.sub(r"\broot\b","RUN_ROOT",part)
        if i == 0:
            part = "datasets = list(NAMES)\nx = np.arange(len(datasets)); width = .095\n" + part
        step(title,"Create this figure now from this execution's data; its drawing code is included below.",
             "New summaries, saved predictions and/or the in-memory toy graph.", filename,
             part + f'\n\ndisplay(Image(filename=str(output / "{filename}")))', details)

    step("Display paired ablations and synthetic condition summaries",
         "Summarise the controlled comparisons and failure cases after inspecting the underlying graphs and measurements.",
         "Fresh paired_differences.csv and method_summary rows.", "Paired changes and a low/medium/high table.", r'''
differences = pd.read_csv(RUN_ROOT / "results/summaries/paired_differences.csv")
display(differences[differences.dataset_id.isin(real_ids)][[
    "dataset_id","interpretation","paired_runs","ari_difference_mean","modularity_difference_mean","method_total_s_difference_mean"]].round(6))
test = summary[summary.role.eq("test")].copy()
test["level"] = test.dataset_id.str.split("_").str[2]
display(test.groupby(["level","method"])[["ari_mean","stability_mean","largest_fraction_mean"]].mean().round(4))
''', "Positive ARI difference favours treatment on that metric; positive time difference means slower treatment. Reference agreement, modularity and stability can disagree. At high mixing, a one-group answer can be both wrong and stable.")
    bibliography = json.loads((root / "research/bibliography.json").read_text(encoding="utf-8"))
    step("Record the research-paper evidence",
         "Keep the methods and prior-art trail with the complete implementation, including the distinction between reviewed work and implemented code.",
         "The embedded ten-paper bibliography.", "A publication table with direct DOI links.",
         "bibliography = " + repr(bibliography) + r'''

display(pd.DataFrame(bibliography)[["id","title","year","journal","doi"]])
assert len(bibliography)==10 and all(p["year"]>=2022 for p in bibliography)
''', "R01 supplies FLPA; related weighting/embedding papers inform the comparison and novelty assessment. The literature is not the detector's training data. Full PDFs and reviews remain additional submission evidence; no PDF file is required to execute this notebook.")
    prose(r"""
    ### Interpretation

    The measurements are mixed: Karate exhibits a quality/stability tradeoff, Football can improve
    reference agreement while reducing modularity, and email remains dominated by a giant FLPA-family
    community. Under high synthetic mixing, propagation methods can collapse even with perfect
    repeatability, whereas Leiden retains useful recovery in this setting.

    The project therefore demonstrates a reproducible extension study and its limits. It does not
    establish universal superiority or reproduce the base paper's very large-network speed study.
    Ten detector seeds are not ten independent networks; the stability pairs share runs.
    This execution uses the same graph versions and algorithm settings as the written report.
    Fresh timing values are labelled as a rerun; the report preserves the earlier recorded timings.
    """)
    step("Verify completion and export the freshly computed evidence",
         "Finish with explicit counts, successful statuses, code provenance and a portable archive of this execution's artifacts.",
         "All newly generated graphs, records, partitions, summaries and figures.",
         "execution_validation.json and Notebook_Run_Evidence.zip in the printed fresh directory.", r'''
assert checked==1040 and development_count==80 and parity_cases==50
assert len(runs)==1040 and runs.status.isin(["completed","converged"]).all()
assert runs.partition_valid.all() and runs.code_sha256.eq(code_fingerprint()).all()
assert not any(name=="sna_project" or name.startswith("sna_project.") for name in sys.modules)
execution_report = {
    "mode":"complete_inline_implementation", "run_root":str(RUN_ROOT),
    "development_runs":development_count, "final_runs":len(runs),
    "real_runs":int(runs.role.eq("real").sum()), "synthetic_runs":int(runs.role.eq("test").sum()),
    "successful_final_runs":int(runs.status.isin(["completed","converged"]).sum()),
    "independently_verified_partitions":checked, "baseline_parity_cases":parity_cases,
    "selected_lambda":LAMBDA, "project_package_imported":False,
    "embedded_raw_sources":len(EMBEDDED_RAW_DATA), "inline_implementation_sha256":code_fingerprint(),
    "status":"passed"
}
write_json(RUN_ROOT / "execution_validation.json",execution_report)
evidence_zip = RUN_ROOT / "Notebook_Run_Evidence.zip"
evidence_files = sorted(p for p in RUN_ROOT.rglob("*") if p.is_file() and p!=evidence_zip)
with zipfile.ZipFile(evidence_zip,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in evidence_files:
        archive.write(path,arcname=path.relative_to(RUN_ROOT).as_posix())
with zipfile.ZipFile(evidence_zip) as archive:
    assert archive.testzip() is None
execution_report["evidence_zip_sha256"] = hashlib.sha256(evidence_zip.read_bytes()).hexdigest()
execution_report["evidence_zip"] = str(evidence_zip)
write_json(RUN_ROOT / "execution_validation.json",execution_report)
display(JSON(execution_report))
print("Full inline project completed. Fresh evidence archive:",evidence_zip)
''', "The last assertion also checks that the project package was never loaded into the kernel. Run All performs the actual study again, without overwriting the prior report evidence. The archive includes raw/prepared graphs, development/final assignments, configuration, metrics and plots for this execution.")

    document = nbf.v4.new_notebook(cells=cells)
    document.metadata.kernelspec = {"display_name":"SNA Project (Python 3.12)","language":"python","name":"sna-project"}
    document.metadata.language_info = {"name":"python","version":"3.12"}
    source_hash = hashlib.sha256("\n\n".join(c.source for c in cells if c.cell_type=="code").encode()).hexdigest()
    for cell in cells:
        if cell.cell_type=="code":
            cell.source = cell.source.replace(HASH_MARKER,source_hash)
    document.metadata["inline_implementation"] = {"code_sha256":source_hash,"hash_marker":HASH_MARKER,
        "project_imports":0,"embedded_raw_sources":len(payload),"full_workflow":True}
    check_inline(document)
    inline_definitions = {n.name:n for c in document.cells if c.cell_type=="code"
                          for n in ast.parse(c.source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    equivalent_definitions = 0
    for module in ["contracts.py","data.py","algorithms.py","evaluation.py","experiment.py"]:
        original_ast = ast.parse((root / "src/sna_project" / module).read_text(encoding="utf-8"))
        for node in original_ast.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name!="code_fingerprint":
                assert ast.dump(node,include_attributes=False)==ast.dump(inline_definitions[node.name],include_attributes=False),node.name
                equivalent_definitions += 1
    document.metadata.inline_implementation["unchanged_core_definitions"] = equivalent_definitions
    nbf.validate(document)
    target = root / NAME
    if not execute:
        nbf.write(document,target)
        print(json.dumps({"path":str(target),"code_cells":sum(c.cell_type=="code" for c in cells),"executed":False}))
        return
    isolated = Path(tempfile.mkdtemp(prefix="sna_notebook_only_"))
    nbf.write(document,isolated / NAME)
    assert {p.name for p in isolated.iterdir()} == {NAME}
    print("Executing notebook alone in:",isolated,flush=True)
    client = NotebookClient(document,timeout=1200,kernel_name="sna-project",resources={"metadata":{"path":str(isolated)}})
    client.execute()
    code_cells = [c for c in document.cells if c.cell_type=="code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not [o for c in code_cells for o in c.outputs if o.output_type=="error"]
    reports = [o["data"]["application/json"] for c in code_cells for o in c.outputs
               if "application/json" in o.get("data",{})]
    execution_report = reports[-1]
    assert execution_report["status"]=="passed" and execution_report["final_runs"]==1040
    run_root = Path(execution_report["run_root"])
    # Compare all deterministic assignments with the existing report's evidence.
    import pandas as pd
    fresh = pd.read_csv(run_root / "results/summaries/per_run_metrics.csv")
    official = pd.read_csv(root / "results/summaries/per_run_metrics.csv")
    official_by_id = {(r.dataset_id,r.method,int(r.seed)):r for r in official.itertuples(index=False)}
    same = 0
    for row in fresh.itertuples(index=False):
        old = official_by_id[(row.dataset_id,row.method,int(row.seed))]
        a = (run_root / "results/runs" / row.attempt / row.run_id / "partition.csv").read_bytes()
        b = (root / "results/runs" / old.attempt / old.run_id / "partition.csv").read_bytes()
        assert a==b, (row.dataset_id,row.method,row.seed)
        same += 1
    execution_report.update({"standalone_execution_directory":str(isolated),
        "initial_files_in_execution_directory":[NAME],"partitions_matching_report_evidence":same,
        "executed_code_cells":len(code_cells)})
    document.metadata["standalone_execution"] = execution_report
    check_inline(document)
    nbf.validate(document)
    nbf.write(document,target)
    evidence_dir = root / "results/notebook_execution"
    evidence_dir.mkdir(parents=True,exist_ok=True)
    shutil.copy2(run_root / "Notebook_Run_Evidence.zip",evidence_dir / "Notebook_Run_Evidence.zip")
    (evidence_dir / "validation.json").write_text(json.dumps(execution_report,indent=2),encoding="utf-8")
    for filename in ["method_summary.csv","per_run_metrics.csv","parameter_selection.csv","paired_differences.csv"]:
        shutil.copy2(run_root / "results/summaries" / filename,evidence_dir / filename)
    print(json.dumps({"path":str(target),"cells":len(document.cells),"code_cells":len(code_cells),
        "fresh_development_runs":80,"fresh_final_runs":1040,"matching_report_partitions":same,
        "standalone_execution":"passed","bytes":target.stat().st_size,"evidence":str(evidence_dir)},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--no-execute",action="store_true")
    args=parser.parse_args()
    build(args.root,not args.no_execute)
