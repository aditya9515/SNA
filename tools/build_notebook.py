"""Build and execute the deliverable notebook, retaining real outputs."""
import json
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SNA_Project_with_Results.ipynb"


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
markdown("""
# Stability-Aware Structural Weighting for Fast Label Propagation

**DSE 4403 - Social Network Analysis**  
Aditya Kosuru (230968316), Jaswanth P (230968014), Prithvi Ram Perivela (230968304)  
Base paper: Traag and Subelj, *Large network community detection by fast label propagation*, Scientific Reports 13, 2701 (2023), DOI: 10.1038/s41598-023-29610-z.

This executed notebook explains the implementation, runs live examples, independently checks the saved experiment metrics, and displays measured results. It is accompanied by the full `src/sna_project` source package. No plotted result is simulated. The extension is a controlled empirical design; weighted propagation and tie retention are established ideas, not claimed inventions.
"""),
code("""
from pathlib import Path
import sys, json, inspect
ROOT = Path.cwd().resolve()
if not (ROOT / "TEAM_DETAILS.json").exists():
    ROOT = next(p for p in [ROOT, *ROOT.parents] if (p / "TEAM_DETAILS.json").exists())
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import networkx as nx
from IPython.display import display, Image, Markdown
from sna_project.data import load_prepared
from sna_project.algorithms import (propagate, edge_similarity, apply_weights, run_method,
                                    prepare_leiden, run_leiden, from_groups, support_violations)
from sna_project.evaluation import evaluate, pairwise_stability
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 120)
selection = json.loads((ROOT / "results/parameter_selection.json").read_text())
LAMBDA = selection["selected_lambda"]
summary = pd.read_csv(ROOT / "results/summaries/method_summary.csv")
runs = pd.read_csv(ROOT / "results/summaries/per_run_metrics.csv")
print("Python:", sys.version.split()[0], "NetworkX:", nx.__version__)
print("Selected lambda:", LAMBDA)
print("Saved final runs:", len(runs), "Terminal statuses:", runs.status.value_counts().to_dict())
display(pd.DataFrame(json.loads((ROOT / "TEAM_DETAILS.json").read_text())["members"]))
"""),
markdown("""
## 1. System and data

Raw sources are preserved; source weights are disabled in the common baseline. Every algorithm receives a simple, undirected graph without evaluation labels. Original node IDs map to canonical internal IDs. Club, conference and department labels are stored separately for scoring. Dolphins has no validated two-group mapping in the downloaded source, so its ARI/NMI are unavailable. Football in the downloaded archive has 613 edges, whereas the base paper reports 616; this is an independent reproduction on a documented source version.
"""),
code("""
display(Image(filename=str(ROOT / "reports/figures/architecture.png")))
datasets = pd.read_csv(ROOT / "results/summaries/dataset_summary.csv")
real_ids = ["karate", "dolphins", "football", "email_eu_core"]
display(datasets[datasets.dataset_id.isin(real_ids)][["dataset_id", "nodes", "edges", "connected_components", "isolates", "reference_groups", "label_coverage"]])
for dataset_id in real_ids:
    item = load_prepared(ROOT, dataset_id)
    assert all(not attrs for _, attrs in item.graph.nodes(data=True))
print("Checked: algorithm graphs contain no reference-label attributes.")
"""),
markdown("""
## 2. Original FLPA implementation

Each node starts with its own label. Nodes are processed from a randomly initialised queue. A node samples a maximal neighbour label. A change queues neighbours whose current labels differ from the new label, with duplicate queue entries prevented. Isolates retain their singleton label. A final independent check verifies maximal support.

The next cell shows the actual package implementation used by the experiments, including the switches needed for controlled variants. `flpa_original` selects unit votes, random ties and the original filtered queue. The test suite includes 50 exact comparisons against NetworkX across five graphs and ten seeds.
"""),
code("""
print(inspect.getsource(propagate))
item = load_prepared(ROOT, "karate")
base = run_method(item.graph, "flpa_original", seed=0)
reference = from_groups(nx.community.fast_label_propagation_communities(item.graph, weight=None, seed=0), item.graph)
assert base.labels == reference
assert not support_violations(item.graph, base.labels)
print("Live baseline agrees exactly with NetworkX; status:", base.status)
display(pd.DataFrame([evaluate(item.graph, base.labels, item.labels)]))
"""),
markdown(r"""
## 3. Structural weights and retention

For each existing edge, use the full open-neighbour sets:

\[
J(u,v)=\frac{|N(u)\cap N(v)|}{|N(u)\cup N(v)|},\qquad w(u,v)=1+\lambda J(u,v).
\]

A candidate label receives the sum of its incident structural weights. The current label is retained if it is already maximal within the fixed numerical tolerance. Otherwise one maximal label is sampled. Weights depend only on topology and remain fixed during each run. Weight preparation is included in the cold method cost.
"""),
code("""
print(inspect.getsource(edge_similarity))
example = nx.Graph([(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),(2,3)])
similarity = edge_similarity(example)
display(pd.DataFrame([{"edge": str(edge), "Jaccard": value, "weight_lambda_1": 1+value}
                      for edge, value in similarity.items()]))
assert np.isclose(similarity[(0,1)], 1/3)
assert np.isclose(similarity[(0,2)], 1/4)
assert similarity[(2,3)] == 0
display(Image(filename=str(ROOT / "reports/figures/weight_example.png")))
weighted = apply_weights(item.graph, edge_similarity(item.graph), LAMBDA)
modified = run_method(weighted, "c11_weighted_retain", seed=0)
assert not support_violations(weighted, modified.labels, "structural_weight")
leiden = run_leiden(prepare_leiden(item.graph), seed=0)
live = pd.DataFrame([{"method": name, **evaluate(item.graph, result.labels, item.labels)}
                     for name, result in [("Original FLPA", base), ("Weighted + retention", modified), ("Leiden", leiden)]])
display(live.round(6))
"""),
markdown("""
## 4. Development-only parameter selection

Lambda candidates were 0, 0.25, 0.5 and 1. Four separate development SBM graphs (200 nodes; generation seeds 100-103) and five algorithm seeds were used, totalling 80 development runs. Candidates must retain reference ARI within 0.02 of lambda zero and avoid giant-group collapse on the balanced development graphs. Choose the highest mean stability, preferring a smaller lambda within 0.01. No final real/test labels were used to choose lambda.
"""),
code("""
display(pd.DataFrame(selection["candidates"]))
assert selection["final_labels_used"] is False
eligible = [c for c in selection["candidates"] if c["eligible"]]
best = max(c["mean_stability"] for c in eligible)
expected = min(c["lambda_value"] for c in eligible if c["mean_stability"] >= best-.01)
assert expected == LAMBDA
print("Selection rule independently checked. Frozen lambda =", LAMBDA)
"""),
markdown("""
## 5. Experiment design and controls

| ID | Weight | Tie rule | Queue |
|---|---|---|---|
| B0 | Unit | Random maximal | Original filtered |
| C00 | Unit | Random maximal | All changed neighbours |
| C10 | Structural | Random maximal | All changed neighbours |
| C01 | Unit | Retain maximal | All changed neighbours |
| C11 | Structural | Retain maximal | All changed neighbours |
| LPA | Unit | Library asynchronous LPA | Library |
| Leiden | Unit | Modularity optimisation | Library |
| NX-W | Structural | NetworkX weighted FLPA | Original filtered |

Eight configurations x ten seeds x four real graphs = 320 real runs. Nine separately generated final SBM graphs x eight configurations x ten seeds = 720 synthetic runs. Low/medium/high between-block probabilities are 0.005/0.025/0.06, with within-block probability 0.16 and four groups of 100 nodes. Generation seeds are 200-202. The synthetic experiment uses SBM rather than the optional LFR design, keeping planted structure and generation reproducible.

The scheduler control is necessary: comparing C11 with B0 alone changes weighting, ties and scheduling together. Weighting is isolated by C10-C00 and C11-C01.
"""),
code("""
audit = pd.read_csv(ROOT / "results/summaries/manifest_audit.csv")
display(audit)
assert audit.expected.sum() == audit.observed.sum() == 1040
assert len(runs) == 1040
assert runs.status.isin(["completed", "converged"]).all()
assert runs.partition_valid.all()
print("All expected final runs exist and have valid partitions.")
"""),
markdown("""
## 6. Independent audit of saved metrics

This cell reloads every final partition, recomputes its metrics from the processed graph and separate labels, and checks the saved values. It does not substitute new results or select favourable seeds.
"""),
code("""
loaded = {dataset_id: load_prepared(ROOT, dataset_id) for dataset_id in runs.dataset_id.unique()}
checked = 0
for row in runs.itertuples(index=False):
    path = ROOT / "results/runs" / row.attempt / row.run_id / "partition.csv"
    assignment = pd.read_csv(path)
    labels = dict(zip(assignment.internal_id.astype(int), assignment.community_id.astype(int)))
    graph_item = loaded[row.dataset_id]
    metrics = evaluate(graph_item.graph, labels, graph_item.labels)
    for metric in ["ari", "nmi", "modularity", "communities", "largest_fraction", "singleton_fraction"]:
        actual, stored = metrics[metric], getattr(row, metric)
        assert (actual is None and pd.isna(stored)) or np.isclose(actual, stored, atol=1e-12, rtol=1e-12)
    checked += 1
print("Independently recomputed and verified", checked, "saved final partitions.")
"""),
markdown("""
## 7. Measured results on real graphs

ARI/NMI are unavailable for Dolphins. Every group has ten algorithm seeds. Mean stability uses all 45 unordered partition pairs, which are dependent descriptive observations, not 45 independent replicates. Inspect group sizes along with stability: a constant one-group answer can be perfectly stable and uninformative.
"""),
code("""
real = summary[summary.role.eq("real")]
display(real[["dataset_id", "method", "successful_runs", "ari_mean", "ari_std", "nmi_mean", "modularity_mean", "stability_mean", "communities_mean", "largest_fraction_mean", "method_total_s_median"]].round(5))
display(Image(filename=str(ROOT / "reports/figures/real_quality.png")))
display(Image(filename=str(ROOT / "reports/figures/stability.png")))
display(Image(filename=str(ROOT / "reports/figures/communities.png")))
"""),
markdown("""
## 8. Ablations, runtime and synthetic stress

Cold method time includes weight construction and backend conversion where applicable, excluding common data loading, metrics and plotting. Run order was shuffled with a separate schedule seed and methods executed sequentially. Tiny graph timings are descriptive and subject to overhead; cross-library timings also reflect compiled versus Python backends. The stacked plot shows medians of the component times; their sum need not equal the median total.
"""),
code("""
differences = pd.read_csv(ROOT / "results/summaries/paired_differences.csv")
display(differences[differences.dataset_id.isin(real_ids)][["dataset_id", "interpretation", "paired_runs", "ari_difference_mean", "modularity_difference_mean", "method_total_s_difference_mean"]].round(6))
display(Image(filename=str(ROOT / "reports/figures/runtime_breakdown.png")))
display(Image(filename=str(ROOT / "reports/figures/synthetic_stress.png")))
test = summary[summary.role.eq("test")].copy()
test["level"] = test.dataset_id.str.split("_").str[2]
display(test.groupby(["level", "method"])[["ari_mean", "stability_mean", "largest_fraction_mean"]].mean().round(4))
"""),
markdown("""
## 9. Interpretation and limitations

The final report gives the exact numbers. The measured evidence is mixed: structural weighting often changes quality and repeatability more than retention does. Karate shows a quality/stability tradeoff, Football shows improved reference agreement but lower modularity, and the email graph retains an unhelpful giant community for FLPA-family methods. At high synthetic between-block connectivity, those methods collapse to one community even though Leiden still recovers useful structure. Thus high stability alone does not establish useful communities or superiority.

These experiments are an independent reproduction and controlled extension, not a replication of the base paper's very large-network runtime claims. Ten seeds and three synthetic graph instances per condition support descriptive analysis, not broad significance claims. Weighted label propagation, Jaccard similarity and maximal-label retention are established mechanisms; global originality of this combination has not been established.
"""),
code("""
bibliography = json.loads((ROOT / "research/bibliography.json").read_text(encoding="utf-8"))
display(pd.DataFrame(bibliography)[["id", "title", "year", "journal", "doi"]])
assert len(bibliography) == 10 and all(p["year"] >= 2022 for p in bibliography)
print("Ten recent journal papers indexed with primary-source evidence.")
"""),
markdown("""
## 10. Re-run instructions

Install the pinned dependencies and the package, then run:

```text
python -m sna_project prepare
python -m sna_project select
python -m sna_project run --config configs/real.yaml
python -m sna_project run --config configs/stress.yaml
python -m sna_project summarize
python -m sna_project plot
```

Use a fresh results directory for a complete rerun, or archive existing attempts as described in README.md. Default notebook execution audits bundled measured results and reruns live examples; it does not automatically overwrite the final experimental record. `pytest -q` runs the correctness suite. The submission contains the complete package, configurations, raw/processed data, run records and report-building scripts.
"""),
]

if __name__ == "__main__":
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "SNA Project (Python 3.12)", "language": "python", "name": "sna-project"}
    notebook.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
    nbf.write(notebook, OUTPUT)
    client = NotebookClient(notebook, timeout=600, kernel_name="sna-project", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(notebook, OUTPUT)
    errors = [out for cell in notebook.cells if cell.cell_type == "code" for out in cell.outputs if out.output_type == "error"]
    assert not errors
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells: {OUTPUT}")
