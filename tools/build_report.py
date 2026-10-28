"""Generate the report from measured artifacts and verified review records."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
import textwrap

import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                               SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
SHORT = {"flpa_original": "B0", "c00_unit_random": "C00", "c10_weighted_random": "C10",
         "c01_unit_retain": "C01", "c11_weighted_retain": "C11", "asyn_lpa": "LPA",
         "leiden_modularity": "Leiden", "nx_weighted_flpa": "NX-W"}
ORDER = list(SHORT)
REAL = ["karate", "dolphins", "football", "email_eu_core"]
NAMES = {"karate": "Karate club", "dolphins": "Dolphins", "football": "Football", "email_eu_core": "Email-Eu-core"}


def fmt(value, digits=3):
    return "N/A" if pd.isna(value) else f"{value:.{digits}f}"


def mean_std(row, metric):
    return "N/A" if pd.isna(row[metric + "_mean"]) else f"{row[metric + '_mean']:.3f} +/- {row[metric + '_std']:.3f}"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(str(value) for value in row) + " |" for row in rows]])


def build_markdown():
    summary = pd.read_csv(ROOT / "results/summaries/method_summary.csv")
    per_run = pd.read_csv(ROOT / "results/summaries/per_run_metrics.csv")
    real = summary[summary.role == "real"].set_index(["dataset_id", "method"])
    datasets = pd.read_csv(ROOT / "results/summaries/dataset_summary.csv").set_index("dataset_id")
    selection = json.loads((ROOT / "results/parameter_selection.json").read_text())
    bibliography = json.loads((ROOT / "research/bibliography.json").read_text(encoding="utf-8"))
    reviews = json.loads((ROOT / "research/review_notes.json").read_text(encoding="utf-8"))
    review_by_id = {record["id"]: record for record in reviews}
    env_file = sorted((ROOT / "results/runs").glob("final_real*/environment.json"))[-1]
    env = json.loads(env_file.read_text())
    abstract = (
        "Community detection reveals groups of densely connected entities, but fast label propagation can produce inconsistent or trivial partitions. "
        "This project independently implements the original queue algorithm and evaluates fixed Jaccard structural weights with maximal-label retention. "
        "A separate synthetic development set selects the weighting parameter without consulting final benchmark labels. "
        "Eight configurations are evaluated over ten seeds on four real networks and nine synthetic graphs, producing 1040 final runs. "
        "Controlled ablations distinguish weighting, retention and scheduling effects. "
        "Weighting improves repeatability on several small networks, while reference agreement and modularity show dataset-dependent tradeoffs. "
        "On the email network, propagation variants still form an unhelpful giant community; under high synthetic mixing they collapse despite perfect repeatability. "
        "Leiden remains a stronger comparison in these difficult settings. "
        "The results support a reproducible study of quality, stability and computational cost rather than universal superiority or an unverified claim of algorithmic novelty."
    )
    # The recovered assignment calls for a 150-word abstract.
    abstract_words = abstract.split()
    if len(abstract_words) < 150:
        abstract += " Reproducible artifacts accompany every measured conclusion."
    assert len(abstract.split()) == 150, len(abstract.split())
    sections = ["# Abstract", abstract, "<!-- toc -->", "<!-- pagebreak -->", "# 1. Introduction and objectives", """
Social network analysis represents entities as nodes and relationships as edges. A friendship network, an association network among animals, an inter-team game schedule and an organisational email network all encode interactions through this representation. Community detection seeks a useful grouping of those nodes. A common intuition is that members of a group connect more strongly to each other than to the rest of the network. However, metadata groups, a modularity optimum and a locally stable label partition can disagree. This makes careful evaluation necessary: a visually attractive network drawing or a high value of one metric cannot establish that the result answers the intended question.

Label propagation is appealing because it requires neither a preset number of groups nor supervised training labels. Each node begins with an individual label and repeatedly adopts a label supported by its neighbours. The original asynchronous method is a foundational comparison [R11]. Traag and Subelj's Fast Label Propagation (FLPA) uses an explicit work queue so that label changes trigger consideration of affected neighbours [R01]. This reduces unnecessary scans while retaining a local maximal-support condition at normal termination. Our base paper is exactly this Scientific Reports article, not another overlapping-community algorithm that happens to use the acronym FLPA.

Two limitations motivate our investigation. First, treating all connections equally ignores local structural evidence. An edge embedded in a triangle may convey different information from a bridge between two regions. Second, random choices among tied labels and the sequence of updates can affect the result. The recovered project proposal combines a fixed structural weight derived from shared neighbours with retention of an already maximal current label. These are established ideas, so the implementation and report distinguish a tested engineering combination from a claim to have invented weighted propagation, Jaccard similarity or retention.

Our purpose is to investigate the complete quality-stability-cost tradeoff. Repeatability means that different random seeds produce similar partitions; it is useful for reproducibility, but a constant incorrect answer is also repeatable. Reference agreement asks whether discovered communities resemble club factions, conferences or departments. Modularity measures structure relative to a null model, but does not require agreement with those metadata. Runtime asks whether any practical benefit justifies additional preparation work. We therefore measure all of these quantities and keep community-size diagnostics alongside them.

The implementation separates graph preparation, detection, experimental control, evaluation and presentation. Every method receives the same processed topology, while reference labels are isolated from the algorithms. We first reproduce the unchanged baseline, test it against NetworkX, then implement independently controlled weighting and retention options. Two identifiable existing comparison works are asynchronous LPA [R11] and Leiden [R12]. An additional NetworkX weighted-FLPA reference makes clear that weight support is already available. We use separate synthetic development graphs for parameter selection, four real networks and nine final synthetic graphs for evaluation.

The research questions are: RQ1, does the specified design improve agreement across seeds? RQ2, does it preserve useful community structure? RQ3, what preparation and execution cost does it introduce? RQ4, which component explains the measured changes? RQ5, under which structural conditions does it fail? Successful completion means answering those questions with traceable evidence, including unfavourable findings. It does not require the extension to dominate every comparison method. This report consequently presents an independent reproduction and controlled empirical extension rather than numerical replication of the base paper's very large-network experiments.
"""
    ]
    sections += ["## 1.1 Base-paper and team contributions", """
The base paper contributes a queue-driven propagation rule, an analysis of local label maximality, and theoretical/empirical comparisons of propagation approaches, including large networks [R01]. Our reproduced scope is the core simple-undirected, unweighted update rule and its stopping behaviour. The source implementation and benchmark sizes used here differ from the paper's full experimental setup.

Our implemented contributions are (1) a configurable topology-derived weighting and maximal-label-retention extension with a separately controlled scheduler; and (2) a reproducible empirical pipeline with development-only selection, eight configurations, multi-seed analysis, failure accounting and independently auditable outputs. The literature does not establish global originality of the combination, and no such originality is claimed. All numerical results below were obtained by executing the submitted code.
""", "<!-- pagebreak -->", "# 2. Literature review: ten recent journal papers", """
Ten journal research articles published from 2022 onward were verified using publisher pages, DOI metadata and the downloaded full texts. Full PDFs, extracted text and bibliographic evidence are included under research/. The review is focused on propagation, structural weighting, alternative community principles and uncertainty across partitions. The articles below are reviewed literature, not ten reproduced algorithms. Their results use different datasets and protocols and are not merged into our measured result tables.
"""
    ]
    review_rows = []
    for paper in bibliography:
        review = review_by_id[paper["id"]]
        review_rows.append({**paper, **review})
        sections += [f"## 2.{int(paper['id'][1:])} {paper['title']} ({paper['year']}) [{paper['id']}]",
                     f"**Method and setting.** {review['method']} {review['datasets']}",
                     f"**Evidence and metrics.** {review['findings']} Metrics: {review['metrics']} Source locations: {review['locator']}",
                     f"**Assessment.** {review['advantage']} Limitation: {review['limitation']} Relevance: {review['relationship']}"]
    pd.DataFrame(review_rows).to_csv(ROOT / "research/literature_review.csv", index=False)
    sections += ["## 2.11 Synthesis and distinction from existing work", """
Recent work already uses importance, similarity and representation learning to guide community formation. The closest conceptual relatives are similarity-weighted propagation [R05], graph reweighting [R03] and stability/pruning variants [R10], while the base article itself discusses retention [R01]. Ensemble analysis [R07] also cautions against interpreting one partition as the only plausible answer. Our narrow distinction is the specified fixed local Jaccard weight, an explicit retention switch, and controlled measurement of its interaction with scheduling. Unlike embedding or attention models, it requires no trained representation. Unlike an overlapping detector, it assigns one group per node. The evidence must still demonstrate when this lightweight design helps, rather than assuming it will.
""", "<!-- pagebreak -->", "# 3. Methodology and algorithm design", "## 3.1 Graph assumptions and original FLPA", """
The supported graph is simple, undirected and loop-free, with all source nodes retained. Internal node IDs are deterministic integers; original IDs/names remain in a mapping file. Initial labels are unique per node. Unit support is the number of neighbours holding a candidate label. Original FLPA samples uniformly among labels with maximal unit support, updates immediately, and queues neighbours whose labels differ from the new label if they are not already queued. An isolate keeps its initial label. The queue starts with all nodes in a seeded random order. The original tie rule resamples even if the current label is maximal.
""", """```text
support(u, c) = sum_{v in N(u)} indicator(label(v) == c)
Q = all nodes in seeded random order; in_queue = set(Q)
while Q is not empty:
    pop u; remove u from in_queue
    if u is isolated: continue
    sample a maximal-support label c_new
    if c_new differs from label(u):
        assign c_new immediately
        enqueue each neighbour v with label(v) != c_new
        unless v is already queued
```""", "## 3.2 Static structural weights", """
For each existing edge, use full open-neighbour sets, excluding each node from its own set but not removing the opposite endpoint before the union. Similarity and weight are:
""", """```text
common(u,v) = |N(u) intersection N(v)|
J(u,v) = common(u,v) / (degree(u) + degree(v) - common(u,v))
w(u,v) = 1 + lambda * J(u,v),  lambda >= 0
weighted_support(u,c) = sum_{v in N(u)} w(u,v) * indicator(label(v)==c)
```""", f"The frozen lambda selected on development graphs is **{selection['selected_lambda']}**. Weights are fixed during a run, positive and symmetric. Topology remains unchanged, and source interaction weights are not used. Similarities are calculated only for existing edges using neighbour sets; no dense node-by-node matrix is created. At lambda zero, every structural weight equals one.",
                  "![Figure 1. Exact mechanism example at lambda one; this explanatory example is separate from the selected experimental lambda.](reports/figures/weight_example.png)",
                  "## 3.3 Retention, tolerances and queue controls", """
The retain-maximal policy keeps the current label when it lies among maximal weighted candidates; otherwise it samples among maximal candidates. A floating score is tied when best minus score is at most atol + rtol * abs(best), with atol = rtol = 1e-12. Unweighted counts use exact comparisons. Stable neighbour iteration and a local seeded generator make runs reproducible. Deep-copying a weighted graph preserves adjacency order, because rebuilding that order can change an asynchronous queue trajectory.

During initial modification we use the conservative policy of queuing all neighbours of a changed node. Extra visits can change random tie resampling and execution cost. We therefore isolate this scheduler from the effects of weights and retention. The four Cxx ablations share the same conservative scheduler; B0 retains the original filtered queue.
""",
                  md_table(["ID", "Votes", "Tie rule", "Queue"], [
                      ["B0", "Unit", "Random maximal", "Original filtered"],
                      ["C00", "Unit", "Random maximal", "All changed neighbours"],
                      ["C10", "Structural", "Random maximal", "All changed neighbours"],
                      ["C01", "Unit", "Retain maximal", "All changed neighbours"],
                      ["C11", "Structural", "Retain maximal", "All changed neighbours"],
                      ["LPA", "Unit", "Asynchronous LPA [R11]", "Library"],
                      ["Leiden", "Unit", "Modularity objective [R12]", "Library"],
                      ["NX-W", "Structural", "NetworkX weighted FLPA", "Original filtered"]]),
                  "C00-B0 isolates scheduling; C10-C00 isolates weighting without retention; C01-C00 isolates retention without weighting; C11-C01 isolates weighting with retention. C11-B0 compares the complete design and must not be attributed to weights alone. Lambda-zero C10/C00 and C11/C01 agree under aligned settings; lambda-zero C11 does not become B0 because ties and queues still differ.",
                  "## 3.4 Stopping, correctness and computational cost", """
Instrumented propagation has a safety cap of max(10000, 1000*n) queue pops and a 120-second wall-time budget checked inside its loop. A cap-limited result would be retained and reported separately, not counted as converged. External library comparisons use their documented stopping policies; these in-loop caps do not pretend to interrupt an external library. Normal FLPA completion is independently checked for complete node coverage and maximal final support. This establishes a local condition, not a globally optimal community partition.

Neighbour-set preparation uses memory proportional to the sparse graph representation. A direct common-neighbour intersection has an approximate work cost proportional to the sum over edges of min(degree(u), degree(v)); algorithm execution also depends on queue revisits. We make no universal linear-time claim for the entire modified system. Cold timing includes actual weight computation and application, backend conversion where required, and algorithm execution. It excludes common graph loading, evaluation, plotting and result-file writes.
""", "<!-- pagebreak -->", "# 4. System design and implementation", "![Figure 2. Complete implemented pipeline. Reference labels are kept out of community detection.](reports/figures/architecture.png)",
                  md_table(["Module", "Responsibility and connection"], [
                      ["data.py", "Retrieve sources, prepare topology, preserve node maps and separate labels; feed every algorithm the same graph."],
                      ["algorithms.py", "Original FLPA, structural features, controlled variants and comparison adapters; return assignments, counters and status."],
                      ["contracts.py", "Shared objects, canonical partitions, fingerprints and atomic JSON writes."],
                      ["experiment.py", "Select lambda on development graphs, schedule seeded runs, persist records and aggregate evidence."],
                      ["evaluation.py", "Check coverage, compute agreement/modularity/size metrics and pairwise partition stability."],
                      ["figures.py", "Generate plots and fixed-layout network drawings from saved results."],
                      ["cli.py", "Expose preparation, selection, execution, summary and plotting commands."],
                      ["tools/", "Acquire literature; build/execute notebook; generate the report and package."],
                      ["tests/", "Behavioural checks, exact reference parity, label alignment and data-integrity checks."]]),
                  "The notebook contains the complete implementation directly in cells: data preparation, algorithms, parameter selection, experiments, evaluation, aggregation and plotting. It embeds raw dataset snapshots and imports no project module. The equivalent package provides a CLI alternative. Run folders contain assignments, records and diagnostics; attempt folders store configuration, environment and manifests. Failures and isolates are preserved.",
                  "<!-- pagebreak -->", "# 5. Datasets and experimental setup", "## 5.1 Real datasets and preprocessing"]
    rows = []
    for dataset in REAL:
        row = datasets.loc[dataset]
        rows.append([NAMES[dataset], int(row.nodes), int(row.edges), int(row.connected_components), int(row.isolates),
                     "Unavailable" if pd.isna(row.reference_groups) else int(row.reference_groups)])
    sections += [md_table(["Network", "Nodes", "Edges", "Components", "Isolates", "Ref. groups"], rows), """
Karate is loaded from NetworkX, with its club attribute extracted for evaluation and its original edge weights disabled. Dolphins and Football come from Newman's original GML archives [D1]. The retrieved Football version has 613 edges, unlike the 616 reported by the FLPA paper; we use and identify the measured version rather than altering it to force agreement. Dolphins contains names but no verified two-group mapping in this source, so supervised agreement scores are unavailable. This limitation is disclosed throughout.

Email-Eu-core is retrieved from SNAP together with department labels [D2]. Its 25,571 source directed edges become 16,064 unique undirected non-self edges under OR-symmetrisation. Reciprocal pairs are deduplicated and source nodes retained. Departments are proxy reference groups rather than a unique definition of structural communities. The resulting graph includes isolated nodes, which are retained in all algorithms and scores. Every algorithm graph has no club, conference or department attribute; those mappings are used only by evaluation.

Source archives, raw checksums, original names/IDs, canonical edge lists and measured statistics are included. If an artifact is changed, loading verifies its graph hash and rejects inconsistent processed data.
""", "## 5.2 Development and synthetic test design", """
Four development stochastic block model (SBM) graphs have four groups of 50 nodes, within-group probability 0.15, between-group probability 0.01 and generation seeds 100-103. Each lambda candidate (0, 0.25, 0.5, 1) is tested with algorithm seeds 0-4, for 80 development runs. Candidates must keep mean reference ARI within 0.02 of lambda zero and avoid a largest group exceeding 95% on these balanced examples. Select the highest mean stability; prefer a smaller lambda within 0.01. This rule was fixed before final evaluation.

The final synthetic study uses nine separate SBM graphs: four groups of 100 nodes, within-group probability 0.16, between-group probability 0.005/0.025/0.06 and generation seeds 200-202 at each level. This replaces the optional LFR stress design with a simpler fully specified planted experiment. Both generation seeds and algorithm seeds are saved and have separate roles. These are actual generated/analysed graphs; the measured outcomes below are not simulated results invented for a report.
""", md_table(["Lambda", "Dev. ARI", "Dev. stability", "Largest fraction", "Eligible"], [
                      [row["lambda_value"], fmt(row["mean_reference_ari"]), fmt(row["mean_stability"]), fmt(row["max_largest_fraction"]), row["eligible"]]
                      for row in selection["candidates"]]),
                  "Lambda 0.25 is selected. The larger positive candidates produced the same development summary under this tested range, so the predeclared smaller-parameter tie rule applies. This does not establish universal parameter insensitivity. Final benchmark labels did not enter selection.",
                  "## 5.3 Software, configuration and metrics",
                  f"The measured environment uses Python {env['python'].split()[0]} on {env['platform']}. Processor: {env['processor'] or env['machine']}. Exact package versions and source fingerprints are stored with each attempt, and requirements-lock.txt captures the tested dependencies.",
                  md_table(["Package", "Version"], [[name, version] for name, version in env["packages"].items()]), """
Eight configurations, ten algorithm seeds (0-9), four real networks and nine final synthetic graphs yield 320 real and 720 synthetic runs. Jobs run sequentially with a separately seeded shuffled order (2718), with third-party imports warmed before measurement. Every expected attempt returned a complete partition and successful terminal status. Timing uses a single measured invocation for each method/seed: values on very small graphs are descriptive and can be dominated by overhead. Compiled Leiden and Python propagation costs include backend effects and should not be read as language-independent algorithmic speed claims.

ARI and arithmetic-normalised NMI compare partitions on an identical ordered node set; ARI can be negative and NMI is not chance adjusted. Modularity is evaluated for every method on the common original processed graph, unweighted, at resolution one. Community count, largest-group fraction and singleton fraction diagnose collapse or fragmentation. Cross-seed stability is the mean ARI of all 45 unordered pairs from ten outputs. Those pairs share runs and are dependent; no significance test treats them as independent samples. Quality tables give mean +/- sample standard deviation over ten algorithm seeds. Runtime tables report the median actual cold method total, with the full dispersion and component timings in the CSV outputs.

The LPA comparator explicitly disables source weights. Leiden uses leidenalg.ModularityVertexPartition, no source weights, standard modularity and n_iterations=-1, with a recorded seed and preserved node map. NX-W uses NetworkX FLPA with the same structural weights; it is a reference for an existing weighting capability, not a new comparison algorithm invented here.
""", "<!-- pagebreak -->", "# 6. Measured results on real networks"]
    for dataset in REAL:
        sections.append(f"## 6.{REAL.index(dataset)+1} {NAMES[dataset]}")
        rows = []
        for method in ORDER:
            row = real.loc[(dataset, method)]
            rows.append([SHORT[method], mean_std(row, "ari"), mean_std(row, "nmi"), mean_std(row, "modularity"),
                         fmt(row.stability_mean), fmt(row.communities_mean, 1), fmt(row.largest_fraction_mean)])
        sections.append(md_table(["Method", "ARI +/- SD", "NMI +/- SD", "Q +/- SD", "Stability", "Groups", "Largest"], rows))
        b, c, l = [real.loc[(dataset, method)] for method in ["flpa_original", "c11_weighted_retain", "leiden_modularity"]]
        if dataset == "karate":
            narrative = f"C11 raises stability from {b.stability_mean:.3f} to {c.stability_mean:.3f} and modularity from {b.modularity_mean:.3f} to {c.modularity_mean:.3f}, but reference ARI falls from {b.ari_mean:.3f} to {c.ari_mean:.3f}. The extension finds about {c.communities_mean:.1f} groups rather than the two metadata factions. This is a stability/structural-quality versus metadata-agreement tradeoff, not an accuracy improvement. C00 also differs substantially from B0, confirming that scheduler effects matter."
        elif dataset == "dolphins":
            narrative = f"C11 improves stability from {b.stability_mean:.3f} to {c.stability_mean:.3f} and modularity from {b.modularity_mean:.3f} to {c.modularity_mean:.3f}. Leiden attains modularity {l.modularity_mean:.3f}. Without a verified reference partition, these are structural and repeatability findings only; they cannot establish improved recovery of the animals' two-group split."
        elif dataset == "football":
            narrative = f"C11 increases reference ARI from {b.ari_mean:.3f} to {c.ari_mean:.3f} and stability from {b.stability_mean:.3f} to {c.stability_mean:.3f}. However, modularity decreases from {b.modularity_mean:.3f} to {c.modularity_mean:.3f}, and the method finds about {c.communities_mean:.1f} groups. Leiden's stability is {l.stability_mean:.3f}, with modularity {l.modularity_mean:.3f}. Different metrics therefore prefer different partitions."
        else:
            narrative = f"B0 has stability {b.stability_mean:.3f} despite reference ARI {b.ari_mean:.4f}; about {100*b.largest_fraction_mean:.1f}% of nodes fall in its largest group. C11's reference ARI is only {c.ari_mean:.4f} and its largest group still contains {100*c.largest_fraction_mean:.1f}% of nodes. Leiden reaches ARI {l.ari_mean:.3f} and modularity {l.modularity_mean:.3f}. The proposed local extension fails to solve the giant-community problem here, and the perfectly stable B0 answer is an example of stability without useful recovery."
        sections.append(narrative)
    sections += ["![Figure 3. Real-network quality and repeatability; Dolphins supervised scores are unavailable. Ten seeds per group.](reports/figures/real_quality.png)",
                 "![Figure 4. Distributions over 45 dependent cross-seed pairs per graph and method; descriptive stability evidence.](reports/figures/stability.png)",
                 "![Figure 5. Representative seed zero on three small real networks, reusing one layout per network. Colours are arbitrary group identifiers.](reports/figures/communities.png)",
                 "<!-- pagebreak -->", "# 7. Component ablation and computational cost", "## 7.1 Separating mechanisms"]
    rows = []
    for dataset in REAL:
        for a, b, label in [("c00_unit_random", "flpa_original", "Scheduler"),
                             ("c10_weighted_random", "c00_unit_random", "Weights, random ties"),
                             ("c01_unit_retain", "c00_unit_random", "Retention, unit votes"),
                             ("c11_weighted_retain", "c01_unit_retain", "Weights, retained ties")]:
            treatment, control = real.loc[(dataset, a)], real.loc[(dataset, b)]
            rows.append([NAMES[dataset], label, fmt(treatment.ari_mean-control.ari_mean),
                         fmt(treatment.modularity_mean-control.modularity_mean), fmt(treatment.stability_mean-control.stability_mean)])
    sections += [md_table(["Network", "Controlled change", "Delta ARI", "Delta Q", "Delta stability"], rows), """
The retention-only change does not consistently improve repeatability. Weighting is associated with substantial stability gains on the small graphs under controlled scheduling, while its quality effects depend on the dataset. On Football, C10 slightly exceeds C11 in reference ARI and stability, so the combined configuration is not automatically the strongest propagation variant. On the email graph neither weighting nor retention produces a useful metadata partition. Stability differences compare group-level summaries; per-seed quality/cost differences and valid pair counts are separately saved in paired_differences.csv. No significance claim is made from these seed samples.
""", "## 7.2 Preparation-inclusive runtime"]
    rows = []
    for dataset in REAL:
        for method in ["flpa_original", "c10_weighted_random", "c11_weighted_retain", "asyn_lpa", "leiden_modularity"]:
            row = real.loc[(dataset, method)]
            rows.append([NAMES[dataset], SHORT[method], fmt(row.feature_prepare_s_median*1000, 2),
                         fmt(row.conversion_s_median*1000, 2), fmt(row.algorithm_s_median*1000, 2),
                         fmt(row.method_total_s_median*1000, 2)])
    sections += [md_table(["Network", "Method", "Prep ms", "Convert ms", "Algorithm ms", "Total ms"], rows),
                 "Component values are medians of actual component measurements; their sum can differ from the median total. Structural feature preparation is a material part of C10/C11 cost, especially on the email graph. These measurements do not establish an end-to-end speed benefit. Weight caching could amortise preparation when repeatedly analysing the same topology, but no cached-cost experiment is presented as measured here. Peak-memory measurements were not collected, so no memory-efficiency claim is made.",
                 "![Figure 6. Cold cost breakdown. Stacks show component medians, not the median of each run's total.](reports/figures/runtime_breakdown.png)",
                 "<!-- pagebreak -->", "# 8. Synthetic stress experiment"]
    test = summary[summary.role == "test"].copy()
    test["level"] = test.dataset_id.str.split("_").str[2]
    means = test.groupby(["level", "method"])[["ari_mean", "stability_mean", "largest_fraction_mean"]].mean()
    rows = []
    for level in ["low", "medium", "high"]:
        for method in ORDER:
            row = means.loc[(level, method)]
            rows.append([level.title(), SHORT[method], fmt(row.ari_mean), fmt(row.stability_mean), fmt(row.largest_fraction_mean)])
    sections += [md_table(["Between-block level", "Method", "Reference ARI", "Stability", "Largest fraction"], rows),
                 "Values average graph-level summaries over three independent graph-generation seeds at each level; each graph has ten algorithm seeds. The plot's error bars are the standard deviation of the three graph-level summaries, not uncertainty estimated from dependent stability pairs.",
                 "![Figure 7. Increasing between-block connectivity exposes a failure mode; error bars describe variation across three generated graphs.](reports/figures/synthetic_stress.png)",
                 f"At medium between-block connectivity, B0 has mean ARI {means.loc[('medium','flpa_original'),'ari_mean']:.3f}, C11 reaches {means.loc[('medium','c11_weighted_retain'),'ari_mean']:.3f}, and Leiden reaches {means.loc[('medium','leiden_modularity'),'ari_mean']:.3f}. At high connectivity, propagation-family configurations collapse to one group, giving ARI zero and stability one. Leiden still achieves mean ARI {means.loc[('high','leiden_modularity'),'ari_mean']:.3f}. Low-connectivity graphs are easy for almost every configuration. This confirms that neither high stability nor success on easy graphs is sufficient evidence for broad performance.",
                 "<!-- pagebreak -->", "# 9. Verification, reproducibility and limitations", "## 9.1 Verification evidence", """
The correctness suite has 69 passing checks. Fifty exact parity cases compare B0 with NetworkX across path, cycle, complete, star and Karate graphs over ten seeds. Other checks cover lambda-zero controls, structural weights on a hand-calculated triangle/bridge example, repeatability, final support invariants, cap reporting, isolates, empty graphs, label/ID alignment, corrupted graph hashes, comparison-backend conversion, metric invariance and stability-pair counts. A graph-copy ordering defect was found during development and corrected before the final frozen experiment.

All 1,040 expected final runs are present: 320 real and 720 synthetic. The 80 development runs are saved separately. No final run failed or reached an instrumented limit. The complete inline notebook was also executed in a folder containing only that notebook. It freshly computed 80 development and 1,040 final runs, then independently recomputed all final partition metrics. All 1,040 predictions matched this report's evidence. The notebook records fresh timings; this report retains the original measured durations, which vary between executions.
""", "## 9.2 Reproduction workflow", """```text
python -m venv .venv
# activate the environment; install requirements-lock.txt and the local package
python -m pytest -q
python -m sna_project prepare
python -m sna_project select
python -m sna_project run --config configs/real.yaml
python -m sna_project run --config configs/stress.yaml
python -m sna_project summarize
python -m sna_project plot
```""", "The README gives setup and reproduction instructions. SNA_Project_with_Results.ipynb contains the full code step by step, embedded source datasets, executed experiments, plots and metric checks. Run All needs only that notebook and installed external libraries; it creates a fresh output directory. explaination.ipynb explains the same code and data flow. Equivalent package source, dependencies, literature and run evidence accompany them. Only Aditya Kosuru, first in the supplied team order, should upload the folder.",
                  "## 9.3 Practical and scientific limits", """
This study covers four real networks, with only three having fully verified reference labels. Dolphins scores cannot support supervised recovery claims. Football differs from the paper's reported edge count, and email departments are proxy metadata rather than definitive structural communities. Synthetic SBM graphs contain planted assumptions that do not capture every real social process. The test graphs are modest in size and cannot support the base paper's large-network scalability claims.

Ten algorithm seeds and three synthetic graph instances per condition provide descriptive evidence. Pairwise stability observations are dependent, and no population-level significance claims are made. Single-invocation small-graph timing is noisy; different language backends and conversion overhead matter. No peak-memory, cached-preparation, overlapping-community or neural-model experiment was performed. The literature review establishes relevant prior ideas but does not prove global novelty. Finally, local support maximality does not rule out useless giant communities, as both email and high-connectivity synthetic results demonstrate.
""", "<!-- pagebreak -->", "# 10. Conclusion and future work", """
We implemented the original FLPA rule, a fixed structural-weight extension, maximal-label retention and a controlled scheduler comparison within a complete reproducible pipeline. The experiments show that local weighting can improve repeatability and sometimes reference agreement, but the gains are dataset dependent and incur preparation overhead. Retention alone is not consistently beneficial, and the combined method does not uniformly beat weighting without retention. The system's central empirical finding is therefore a tradeoff rather than universal improvement.

RQ1 is answered conditionally: repeatability improves on several small graphs, but not on email, and trivial collapse can be perfectly repeatable. RQ2 is also conditional: metadata agreement and modularity can move in different directions. RQ3 shows material feature-preparation cost rather than an end-to-end speed gain. RQ4 shows why both component and scheduler controls are needed. RQ5 identifies giant-community collapse on email and high-connectivity planted graphs as a clear failure mode. Leiden is a stronger comparison in those difficult settings.

Future work should study more independent networks, alternative local structural signals, carefully justified scheduling optimisations, cached preparation under repeated analysis and explicit collapse-resistant objectives. Any new setting should be selected on separate development evidence and compared under a frozen protocol. A future novelty claim would require a closer systematic comparison with the most similar weighted and retention-based propagation work. The submitted artifacts make the present positive and negative findings inspectable and reproducible.
""", "<!-- pagebreak -->", "# References"]
    for paper in bibliography:
        sections.append(f"[{paper['id']}] {paper['authors']}. {paper['title']}. {paper['journal']} ({paper['year']}). {paper['doi_url']}")
    sections += [
        "[R11] U. N. Raghavan, R. Albert and S. Kumara. Near linear time algorithm to detect community structures in large-scale networks. Physical Review E 76, 036106 (2007). https://doi.org/10.1103/PhysRevE.76.036106",
        "[R12] V. A. Traag, L. Waltman and N. J. van Eck. From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports 9, 5233 (2019). https://doi.org/10.1038/s41598-019-41695-z",
        "[D1] M. E. J. Newman. Network data repository; original GML archives and source citations for Dolphins and Football. https://websites.umich.edu/~mejn/netdata/",
        "[D2] SNAP. Email-Eu-core network and department-label files. https://snap.stanford.edu/data/email-Eu-core.html",
        "[D3] NetworkX documentation. Karate club graph, FLPA implementation and modularity. https://networkx.org/documentation/stable/",
        "[D4] scikit-learn documentation. Adjusted Rand Index and Normalized Mutual Information. https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation",
        "[D5] leidenalg documentation. ModularityVertexPartition and seeded find_partition. https://leidenalg.readthedocs.io/en/latest/reference.html",

    ]
    body = "\n\n".join(section.strip() for section in sections) + "\n"
    body = body.replace("\n\n<!-- pagebreak -->\n\n# 2. Literature", "\n\n# 2. Literature")
    body = body.replace("\n\n<!-- pagebreak -->\n\n# 3. Methodology", "\n\n# 3. Methodology")
    body = body.replace("\n\n<!-- pagebreak -->\n\n# 6. Measured", "\n\n# 6. Measured")
    (ROOT / "Project_Report.md").write_text(body, encoding="utf-8")
    return body


def inline(text):
    value = html.escape(text)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", value)
    return value


def build_pdf(body):
    for name, filename, style in [("Body", "times.ttf", "normal"), ("BodyBold", "timesbd.ttf", "bold")]:
        windows_font = Path("C:/Windows/Fonts") / filename
        font = str(windows_font) if windows_font.exists() else fm.findfont(fm.FontProperties(family="Liberation Serif", weight=style))
        pdfmetrics.registerFont(TTFont(name, font))
    # Code uses the standard Courier face; all prose uses Times New Roman.
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportBody", fontName="Body", fontSize=12, leading=16, spaceAfter=8, alignment=TA_JUSTIFY, textColor=colors.black))
    styles.add(ParagraphStyle("H1", fontName="BodyBold", fontSize=16, leading=21, textColor=colors.black, spaceAfter=14, keepWithNext=True))
    styles.add(ParagraphStyle("H2", fontName="BodyBold", fontSize=13, leading=17, textColor=colors.black, spaceBefore=10, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle("Cell", fontName="Body", fontSize=9, leading=11.5, spaceAfter=0, textColor=colors.black))
    styles.add(ParagraphStyle("CellHeader", fontName="BodyBold", fontSize=9, leading=11.5, textColor=colors.black))
    styles.add(ParagraphStyle("Caption", fontName="Body", fontSize=10, leading=13, textColor=colors.black, spaceAfter=12))
    styles.add(ParagraphStyle("ReportCode", fontName="Courier", fontSize=8, leading=11, textColor=colors.black, spaceAfter=10))
    styles.add(ParagraphStyle("Institute", fontName="BodyBold", fontSize=20, leading=26, textColor=colors.black, alignment=TA_CENTER, spaceAfter=14))
    styles.add(ParagraphStyle("CoverTitle", fontName="BodyBold", fontSize=20, leading=27, textColor=colors.black, alignment=TA_CENTER, spaceAfter=22))
    styles.add(ParagraphStyle("CoverSmall", fontName="Body", fontSize=12, leading=17, spaceAfter=12, textColor=colors.black, alignment=TA_CENTER))
    styles.add(ParagraphStyle("CoverTableHeader", fontName="BodyBold", fontSize=12, leading=16, textColor=colors.black))
    width = A4[0] - 88
    story = [Spacer(1, 16), Paragraph("MIT", styles["Institute"]),
             Paragraph("DSE 4403 / SOCIAL NETWORK ANALYSIS", styles["CoverSmall"]),
             Paragraph("Stability-Aware Structural Weighting for Fast Label Propagation in Social Networks", styles["CoverTitle"]),
             Paragraph("Project report and reproducible experimental study", styles["CoverSmall"]), Spacer(1, 14)]
    team = json.loads((ROOT / "TEAM_DETAILS.json").read_text())
    table = Table([[Paragraph("Team member", styles["CoverTableHeader"]), Paragraph("Registration number", styles["CoverTableHeader"])],
                   *[[Paragraph(m["name"], styles["ReportBody"]), Paragraph(m["registration_number"], styles["ReportBody"])] for m in team["members"]]], colWidths=[width*.62, width*.38])
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), colors.white),
                               ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
                               ("GRID", (0,0), (-1,-1), .5, colors.black),
                               ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                               ("TOPPADDING", (0,0), (-1,-1), 8)]))
    story += [table, Spacer(1, 28), Paragraph("Base paper", styles["H2"]),
              Paragraph("Traag and Subelj (2023): Large network community detection by fast label propagation. Scientific Reports 13, 2701. DOI: 10.1038/s41598-023-29610-z.", styles["ReportBody"]),
              Spacer(1, 16), Paragraph("1,040 final runs / 80 development runs / 69 passing correctness checks", styles["CoverSmall"]),
              Paragraph("Prepared 26 October 2026. Designated uploader: Aditya Kosuru, first in the supplied team order.", styles["CoverSmall"]), PageBreak()]
    lines = body.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1; continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak()); index += 1; continue
        if line == "<!-- toc -->":
            story.append(Spacer(1, 18))
            story.append(Paragraph("Contents", styles["H2"]))
            toc = TableOfContents()
            toc.levelStyles = [ParagraphStyle("TOC", fontName="Body", fontSize=11, leading=15, spaceBefore=5, textColor=colors.black)]
            story.append(toc); index += 1; continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["H1"])); index += 1; continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["H2"])); index += 1; continue
        if line.startswith("```"):
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.extend(textwrap.wrap(lines[index], width=91, replace_whitespace=False, drop_whitespace=False) or [""])
                index += 1
            story.append(Preformatted("\n".join(code_lines), styles["ReportCode"])); index += 1; continue
        if line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                image = Image(str(ROOT / match.group(2)))
                ratio = width / image.imageWidth
                image.drawWidth, image.drawHeight = width, image.imageHeight * ratio
                story.append(KeepTogether([image, Spacer(1, 5), Paragraph(inline(match.group(1)), styles["Caption"])]))
            index += 1; continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                values = [value.strip() for value in lines[index].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-+:?$", value) for value in values):
                    rows.append(values)
                index += 1
            count = len(rows[0])
            fractions = [1 / count] * count
            if count == 2: fractions = [.29, .71]
            if count == 7: fractions = [.10, .18, .18, .18, .13, .11, .12]
            if count == 6: fractions = [.25, .12, .15, .16, .14, .18]
            if count == 5: fractions = [.23, .29, .16, .16, .16]
            if count == 4: fractions = [.12, .18, .30, .40]
            cells = [[Paragraph(inline(value), styles["CellHeader"] if ri == 0 else styles["Cell"]) for value in row] for ri, row in enumerate(rows)]
            table = Table(cells, colWidths=[width*f for f in fractions], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), colors.white),
                                       ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
                                       ("GRID", (0,0), (-1,-1), .5, colors.black),
                                       ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 6),
                                       ("BOTTOMPADDING", (0,0), (-1,-1), 6), ("LEFTPADDING", (0,0), (-1,-1), 5),
                                       ("RIGHTPADDING", (0,0), (-1,-1), 5)]))
            if len(rows) <= 6:
                story.append(KeepTogether([table, Spacer(1, 10)]))
            else:
                story.extend([table, Spacer(1, 10)])
            continue
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "```", "![", "<!--")):
            paragraph.append(lines[index].strip()); index += 1
        story.append(Paragraph(inline(" ".join(paragraph)), styles["ReportBody"]))

    class ReportDoc(SimpleDocTemplate):
        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph) and flowable.style.name == "H1":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))

    def decorate(canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(colors.black); canvas.line(44, A4[1]-31, A4[0]-44, A4[1]-31)
            canvas.setFont("Body", 9); canvas.setFillColor(colors.black)
            canvas.drawString(44, A4[1]-24, "SNA project | Structural weighting and fast label propagation")
        canvas.setFont("Body", 9); canvas.setFillColor(colors.black)
        canvas.drawString(44, 24, "230968316 / 230968014 / 230968304")
        canvas.drawRightString(A4[0]-44, 24, f"Page {doc.page}")
        canvas.restoreState()
    output = ROOT / "Project_Report.pdf"
    doc = ReportDoc(str(output), pagesize=A4, rightMargin=44, leftMargin=44, topMargin=47, bottomMargin=43,
                    title="Stability-Aware Structural Weighting for Fast Label Propagation in Social Networks",
                    author="Aditya Kosuru; Jaswanth P; Prithvi Ram Perivela")
    doc.multiBuild(story, onFirstPage=decorate, onLaterPages=decorate)
    print(output)


if __name__ == "__main__":
    build_pdf(build_markdown())
