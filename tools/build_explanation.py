"""Add a detailed reading guide to the complete inline notebook and its outputs."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "SNA_Project_with_Results.ipynb"
TARGET = "explaination.ipynb"

FUNCTION_GUIDE = {
    "fingerprint": "Serialises a value with stable key ordering and hashes it. Graph/configuration identity should depend on content rather than arbitrary dictionary formatting.",
    "write_json": "Creates the parent directory, writes a temporary JSON file and replaces the destination. A reader should not encounter a half-written experiment record.",
    "canonical_partition": "Collects nodes by assigned label and renumbers groups by their smallest internal node ID. It changes identifiers, not membership, allowing exact comparison with library partitions.",
    "groups_from_labels": "Converts node-to-community assignments to sets of nodes, the representation expected by NetworkX modularity.",
    "PreparedGraph": "Stores topology, source metadata, hash and optional separate evaluation references. The detector is passed its graph, not the object containing reference answers.",
    "AlgorithmResult": "Carries predicted assignments, terminal status and queue counters. Separating predictions from reference labels prevents confusing the method output with supplied answers.",
    "download": "Reuses an existing raw snapshot or downloads the declared URL if missing. Embedded files are restored first, making this notebook's default execution offline after dependency installation.",
    "source_files": "Hashes each raw file for provenance. These file hashes are distinct from the cleaned topology hash, because different raw encodings can represent the same prepared graph.",
    "save_prepared": "Maps source IDs deterministically, retains all nodes, removes loops and repeated/reciprocal edges, builds an unweighted undirected graph, and writes topology separately from reference groups. Empty node attributes keep club/conference/department information out of voting.",
    "prepare_real": "Parses each real source format and extracts evaluation references. Karate uses club factions, Football conferences and email departments. Dolphins has no validated reference mapping in the archive. It sends raw data to the common preparation function.",
    "prepare_sbm": "Builds a block-probability matrix, generates a seeded NetworkX stochastic block model and stores generation settings. The generator's block membership becomes a separate scoring reference, not an input to propagation.",
    "load_prepared": "Adds every node before edges so isolates survive, verifies the graph hash, and returns labels separately when a label CSV exists. Stable internal IDs align predictions and references.",
    "prepare_all": "Prepares four real, four development and nine final synthetic graphs, with the graph seeds and probabilities shown in the code. It writes the dataset summary from actual prepared metadata.",
    "validate_graph": "Rejects directed, multi-edge or looped inputs because this experiment explicitly supports simple undirected topology. All methods therefore start from the same graph assumptions.",
    "edge_similarity": "Precomputes each node's open-neighbour set, intersects/unions the sets for existing edges and stores Jaccard overlap. Current community assignments never influence this feature.",
    "apply_weights": "Checks lambda and similarities, deep-copies the graph to preserve neighbour order, and adds structural_weight=1+lambda*Jaccard. Baseline edges remain active and lambda zero recovers unit support.",
    "label_support": "Counts neighbouring labels for unit voting or sums incident structural weights per candidate label. math.fsum reduces floating-point accumulation error; invalid/non-positive weights raise an error.",
    "maximal_candidates": "Finds labels whose support is maximal, with an explicit weighted tolerance. Unit integer votes use exact equality. Ties may contain the node's present label.",
    "support_violations": "Independently recomputes support for every non-isolated node and reports any final label that is not maximal. This validates a local stopping condition, not global community quality.",
    "propagate": "Initialises unique labels and a seeded shuffled FIFO queue; pops work; scores labels; retains an already maximal label when configured or samples a maximal candidate; updates immediately; and queues affected neighbours without duplicates. It records limits and counters instead of treating interrupted work as success.",
    "from_groups": "Converts library-produced node sets to a complete disjoint assignment, rejecting overlaps or omitted nodes and canonicalising community numbering.",
    "run_method": "Dispatches the named method and makes weight, retention, scheduling and candidate-order settings explicit. It invokes the visible propagation function or existing NetworkX comparators. It has no reference-label parameter.",
    "prepare_leiden": "Converts the same unweighted topology to igraph and retains an index-to-node mapping, including isolates. Backend conversion is timed separately from optimisation.",
    "run_leiden": "Calls the existing Leiden library for standard modularity with unit edges, a fixed seed and n_iterations=-1. It converts membership back to canonical original internal node assignments. This is a comparator, not a neural model we trained.",
    "evaluate": "Checks complete node coverage; computes common unweighted modularity and group diagnostics; and computes ARI/NMI only when a full reference exists. Missing Dolphins reference scores remain None/NaN.",
    "pairwise_stability": "Enumerates unordered pairs of predicted partitions and compares their memberships using ARI. Ten runs produce 45 dependent pairs. It does not consult reference labels.",
    "code_fingerprint": "Returns the authoring-time hash of the actual inline code-cell sources with the hash literal normalised. The notebook hash differs from the companion package's source-file hash, and each run records which implementation produced it.",
    "environment": "Records Python, OS and third-party package versions, the code identifier and timing boundary. Machine-dependent timings should not be expected to match exactly on a rerun.",
    "select_parameter": "Runs 4 lambdas × 4 development graphs × 5 seeds, scores predictions using development references, rejects invalid/collapsed or overly degraded candidates, and selects stability with a smaller-lambda preference. Final labels are excluded.",
    "read_config": "Parses the declared YAML and rejects missing settings, unknown methods, duplicates and invalid seeds. Configuration validation happens before running jobs.",
    "run_experiment": "Creates an expected-job manifest and shuffled sequential schedule. For every job it times weight preparation, conversion and detection; checks/scores predictions; saves partition/record/diagnostics; and preserves failures. All stages are visible in the notebook.",
    "aggregate": "Collects the newly generated records, rejects incomplete attempts, forms per-method summaries and stability pairs, and joins treatment/control seeds for paired differences. It records failed counts as well as successful performance values.",
    "save": "Writes the newly computed matplotlib figure as a white-background PNG and closes it. The following Image call embeds that new image, rather than reading an old project figure.",
    "architecture": "Draws the pipeline's nodes and arrows, including the separate reference-label path into evaluation. Its shapes and text are defined in the same notebook.",
}


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def build(root=ROOT):
    root = root.resolve()
    original = nbf.read(root / SOURCE,as_version=4)
    assert original.metadata.inline_implementation.full_workflow
    original_code = [c for c in original.cells if c.cell_type=="code"]
    assert all(c.execution_count is not None for c in original_code)
    assert not [o for c in original_code for o in c.outputs if o.output_type=="error"]
    cells = [md("""
# Explanation of the complete inline SNA project

This companion explains the **current complete-code version** of `SNA_Project_with_Results.ipynb`.
The main notebook now defines the full implementation directly in cells and runs all 80 development
and 1,040 final experiments. It imports only Python/third-party libraries and contains embedded
raw-data snapshots. No project package or precomputed result file is required to run it.

Read the overview first, then the original notebook's steps and the additional code-cell guides.
Every main-notebook cell is reproduced verbatim, including its successful saved execution output.
Outputs are copied from that exact execution; building this explanatory copy does not needlessly
rerun the entire study. Running all cells here will itself perform the same fresh full computation.

**Team:** Aditya Kosuru (230968316), Jaswanth P (230968014), Prithvi Ram Perivela (230968304).
"""),md(r"""
## Brief: implementation, paper foundation and proposed novelty

The project detects one community per node by propagating temporary labels through neighbours.
Its input is graph topology; its output is predicted community membership plus diagnostic results.
The base is Traag and Šubelj's [FLPA paper](https://doi.org/10.1038/s41598-023-29610-z), especially
Algorithm 3's asynchronous neighbour update and filtered queue. Retention is established prior art,
also discussed in the base paper.

The proposed extension gives each existing edge the fixed local weight
$1+\lambda J(N(u),N(v))$ and evaluates weighting, tie retention and queue/order settings using
explicit controls. Jaccard, weighting and retention are established ideas; the contribution is this
specified implementation and empirical evaluation. Global originality and universal improvement
are not established. No GNN, learned embedding, gradient optimiser or neural training is present.
Lambda selection is a development-only parameter procedure, not neural training.

The complete path is embedded raw data → cleaned graph/map → predicted communities → evaluation
with separate references → summaries/plots. Development uses four 200-node SBM graphs to select
lambda. Final evaluation uses four real graphs and nine 400-node SBM graphs. The notebook actually
executes these stages rather than importing our package or merely displaying its old outputs.
"""),md("""
## Where data enters and what the imports mean

| Item | Where it comes from | How it is used |
|---|---|---|
| Karate | `nx.karate_club_graph()` | Members/ties; club labels separated for scoring |
| Dolphins / Football | Embedded Newman ZIP bytes | GML topology; Football conference references; no validated Dolphins references |
| Email-Eu-core | Embedded SNAP gzip files | Email links and separate department proxy labels |
| Development/final SBM | Inline `nx.stochastic_block_model` calls | Reproducible planted graphs, with distinct development/final settings |
| Papers | Embedded publication metadata and DOI links | Methods, prior art and discussion; not graph inputs or training examples |
| `nodes.csv` | Inline preprocessing | Stable internal IDs and original-name mapping, preserving isolates |
| `edges.csv` | Inline preprocessing | Simple undirected detection topology |
| `labels.csv` | Extracted/generated references | Evaluation and development-only selection, never neighbour voting |
| `partition.csv` | Actual inline detector execution | Predicted node-to-community assignments |
| Results/figures | Actual inline runner, aggregation and plot cells | Fresh measurements, tables and charts |

Standard imports handle paths, JSON, hashing, queue state, random seeds, compressed data and files.
NetworkX stores graphs and supplies baseline comparators; NumPy checks numerical values; pandas
reads/aggregates CSVs; sklearn supplies ARI/NMI; matplotlib draws figures; igraph/leidenalg supply
the Leiden comparator. These are external libraries, not imports of project source files.
`RUN_ROOT` is a fresh temporary directory printed by the first code cell. All generated file paths
in the workflow are relative to it. Every run produces its own evidence ZIP.

The encoded raw-data cell can be collapsed because it contains long data strings. The actual
Python implementation cells remain visible. Source bytes are hash-checked before use.
""")]
    code_number = 0
    for position,original_cell in enumerate(original.cells,1):
        copied = copy.deepcopy(original_cell)
        copied.id = nbf.v4.new_markdown_cell("").id
        copied.metadata.update({"original_cell_index":position,
                                "original_source_sha256":hashlib.sha256(copied.source.encode()).hexdigest()})
        if copied.cell_type=="code":
            code_number += 1
            copied.metadata["original_code_cell_number"] = code_number
            info = copied.metadata.teaching
            guide = [f"### Code-cell guide {code_number}: {info.title}","",
                     f"**What and why:** {info.why}","",f"**Inputs:** {info.inputs}","",
                     f"**Outputs and downstream use:** {info.outputs}","",info.details]
            nodes = [n for n in ast.parse(copied.source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            if nodes:
                guide += ["","| Visible definition | How and why it is implemented |","|---|---|"]
                for node in nodes:
                    guide.append(f"| `{node.name}` | {FUNCTION_GUIDE[node.name]} |")
            else:
                guide += ["","**Execution:** the statements below run the already-defined inline functions or implement the visible plotting/data step. Their output belongs to this full execution, rather than a previously imported results notebook."]
            guide += ["","**Connection:** Python retains the defined functions and variables for the following cells. Run cells in order; restarting the kernel requires running the preceding definitions and data steps again."]
            cells.append(md("\n".join(guide)))
        cells.append(copied)
    cells.append(md("""
## Points to explain during presentation

- FLPA finds groups through local neighbour votes; the queue reduces redundant revisits.
- Jaccard weights are fixed topology features; positive baseline weight preserves every edge.
- Retention keeps an already maximal current label; it is established prior art.
- C10–C00 and C11–C01 isolate weighting under fixed controlled scheduling/order.
- C00–B0 also changes candidate ordering, so its `scheduler` label describes a combined control.
- Development references select lambda; final references score predictions and never guide votes.
- Stability compares outputs across seeds; reference ARI compares them with known/planted groups.
- Email and high-mixing failures show why high stability does not prove useful communities.
- All code, preparation, experiments, metric checks and plot construction are visible in cells.
- The execution was tested with only the notebook in its working folder; all project-module imports were absent.

The PDF retains the earlier report's measured timing values. The inline notebook records a fresh
execution using the same graphs/settings; all 1,040 predicted assignments were verified against
the report evidence by the authoring validation. Timings vary between executions.
"""))
    document = nbf.v4.new_notebook(cells=cells,metadata=copy.deepcopy(original.metadata))
    document.metadata["explanation_provenance"] = {
        "source_notebook":SOURCE,"source_notebook_sha256":hashlib.sha256((root / SOURCE).read_bytes()).hexdigest(),
        "original_cells_explained":len(original.cells),"original_code_cells_reproduced":code_number,
        "source_cell_mapping":"verbatim; all original cells retained",
        "execution_origin":"outputs copied from the exact executed source cells; no additional experiment rerun",
    }
    nbf.validate(document)
    mapped = [c for c in document.cells if "original_cell_index" in c.metadata]
    assert len(mapped)==len(original.cells)
    assert all(c.source==s.source and c.cell_type==s.cell_type for c,s in zip(mapped,original.cells))
    nbf.write(document,root / TARGET)
    print(json.dumps({"path":str(root / TARGET),"main_cells_explained":len(mapped),
                      "full_code_cells":code_number,"output_origin":"exact executed main cells","status":"passed"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args()
    build(args.root)
