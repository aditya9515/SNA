# SNA project - completed implementation and submission

[Project report](Project_Report.pdf) | [Executed notebook](SNA_Project_with_Results.ipynb) | [Cell-by-cell explanation](explaination.ipynb) | [Download complete project ZIP](https://github.com/aditya9515/SNA/archive/refs/heads/main.zip)

**Stability-Aware Structural Weighting for Fast Label Propagation in Social Networks**  
DSE 4403 - Social Network Analysis | 26 October 2026

| Team member, in supplied order | Registration number |
|---|---|
| Aditya Kosuru | 230968316 |
| Jaswanth P | 230968014 |
| Prithvi Ram Perivela | 230968304 |

**Only one member should upload the folder. Designated uploader: Aditya Kosuru**, first in the supplied team order.

## What to submit

- `Project_Report.pdf` - final report with the abstract, introduction, ten-paper review, design, experiments, measured results, limitations and references.
- `Project_Report.md` - editable report source.
- `SNA_Project_with_Results.ipynb` - complete implementation directly in 37 code cells: raw data, preprocessing, algorithms, 80 development runs, 1,040 final runs, evaluation and plot construction; executed successfully without the project package.
- `explaination.ipynb` - detailed guide to the current complete-code notebook, explaining all 77 main cells and reproducing the same 37 code cells and their executed outputs.
- `src/sna_project/`, `configs/`, `tests/`, `tools/` - full implementation and reproduction tools.
- `data/` - source archives, prepared graphs, node maps and separate labels.
- `results/` - development selection, final assignments/records, effective configurations, environments and summaries.
- `research/` - ten recent journal papers, source evidence and the completed literature matrix.
- `reports/figures/` - figures used in the notebook and report.
- `requirements-lock.txt`, `TEAM_DETAILS.json`, `VALIDATION.json` and packaged `CHECKSUMS.json` - environment, identity, validation and integrity evidence.

Open the PDF and notebook first. Their results are already saved; viewing them does not require rerunning the experiments.

## Completed work

- Original FLPA, four controlled weighting/retention variants, asynchronous LPA, Leiden and a NetworkX weighted-FLPA reference.
- 80 development runs on four separate synthetic graphs; selected lambda **0.25**.
- **1,040 final runs:** 320 real-network runs and 720 synthetic runs, with ten algorithm seeds per configuration.
- **69 passing correctness checks**, including 50 exact comparisons of original FLPA with NetworkX.
- Complete inline notebook executed with only that notebook in its working folder: 80 fresh development and 1,040 fresh final runs; every predicted partition matches the report evidence. It independently recomputes all final metrics.

The findings are mixed. Weighting often improves small-graph repeatability, but quality varies and preparation costs increase. FLPA variants still form an unhelpful giant group on email and collapse in the hardest synthetic setting. These outcomes are reported. Weighting and tie retention are established ideas; global originality and universal superiority are not claimed.

## Datasets

| Real network | Nodes | Edges | Reference labels |
|---|---:|---:|---|
| Karate | 34 | 78 | Club factions |
| Dolphins | 62 | 159 | Unavailable in source; ARI/NMI not fabricated |
| Football | 115 | 613 | Conferences |
| Email-Eu-core | 1005 | 16064 | Departments |

Football's retrieved source has 613 edges, whereas the base paper reports 616. Email is converted by OR-symmetrisation, reciprocal deduplication and loop removal, keeping every node. All methods receive the same topology; labels are kept out of detection.

The final synthetic study uses SBM rather than the optional LFR plan: four groups of 100 nodes; within-group probability 0.16; between-group probabilities 0.005/0.025/0.06; generation seeds 200-202 at each level. Development uses different graphs and seeds. Full settings are recorded.

## Windows setup

Use Python 3.12; the original environment used 3.12.15. The notebook imports only standard/third-party libraries, includes the four raw compressed dataset snapshots, and does not require `src/`, `configs/`, `data/` or old result files. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
```

With uv, use `uv venv --python 3.12 .venv`, then `uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt` and `uv pip install --python .venv/Scripts/python.exe -e .`.

Register the notebook kernel and open Jupyter:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name sna-project --display-name "SNA Project (Python 3.12)"
.\.venv\Scripts\python.exe -m pip install notebook
.\.venv\Scripts\python.exe -m notebook SNA_Project_with_Results.ipynb
```

Choose the SNA Project kernel and Run All. This performs the complete 80-run development selection and 1,040-run final study using functions written directly in the cells. It generates a fresh temporary output directory and an evidence ZIP, whose paths are printed. Saved outputs are already included.

The PDF retains the original study's timing values. The new notebook is a fresh execution with identical graph/seed assignments; all 1,040 predictions were verified against the report evidence, while timings vary between runs. Its fresh evidence is also packaged in `results/notebook_execution/Notebook_Run_Evidence.zip`.

To re-author the notebook from the equivalent package functions and execute it in an otherwise empty working folder:

```powershell
.\.venv\Scripts\python.exe tools/build_notebook.py
.\.venv\Scripts\python.exe tools/build_explanation.py
```

## Optional equivalent package/CLI workflow

The complete notebook is sufficient for the requested source-code submission. The equivalent package remains available for a command-line workflow. Install it with `.\.venv\Scripts\python.exe -m pip install -e .`, then keep new experiments separate so final tables do not mix attempts. From this folder:

```powershell
$reproductionRoot = Join-Path (Get-Location) "reproduction"
$realConfiguration = (Resolve-Path configs/real.yaml).Path
$stressConfiguration = (Resolve-Path configs/stress.yaml).Path
.\.venv\Scripts\python.exe -m sna_project prepare --root $reproductionRoot
.\.venv\Scripts\python.exe -m sna_project select --root $reproductionRoot
.\.venv\Scripts\python.exe -m sna_project run --root $reproductionRoot --config $realConfiguration
.\.venv\Scripts\python.exe -m sna_project run --root $reproductionRoot --config $stressConfiguration
.\.venv\Scripts\python.exe -m sna_project summarize --root $reproductionRoot
.\.venv\Scripts\python.exe -m sna_project plot --root $reproductionRoot
```

The inline notebook uses embedded raw snapshots and runs offline after installing dependencies. The optional CLI may need internet for missing source archives. Repeated CLI calls create new attempt folders; select/archive attempts before combining results. The notebook automatically uses a fresh output root.

Regenerate the report from the current project's official results:

```powershell
.\.venv\Scripts\python.exe tools/build_report.py
.\.venv\Scripts\python.exe tools/render_report.py
.\.venv\Scripts\python.exe tools/validate_artifacts.py
```

## Source map

| File | Purpose |
|---|---|
| `data.py` | Acquisition, graph preparation, stable IDs, labels and integrity checks |
| `algorithms.py` | FLPA, structural weights, controlled variants and comparison adapters |
| `evaluation.py` | Agreement, modularity, group diagnostics and stability |
| `experiment.py` | Parameter selection, seeded runs, timing, artifacts and aggregation |
| `figures.py` | Evidence plots and common-layout drawings |
| `contracts.py` | Shared objects, canonical labels, hashes and atomic JSON writes |
| `cli.py` | Preparation, selection, execution, summary and plotting commands |

These equivalent modules are in `src/sna_project/`. The complete notebook defines their computation directly in its own cells and imports none of them. `tools/build_full_notebook.py` assembles the literal code at authoring time; no source extraction, dynamic execution or project-file reading happens when the delivered notebook runs.

## Limits and provenance

This is an independent reproduction and empirical extension, not numerical replication of the base paper's full large-network experiments. Ten algorithm seeds are not ten independent social networks. Pairwise stability values share runs. Tiny-graph single-invocation timing and Python/compiled backend differences limit speed claims. Peak memory, cached-cost and neural-model comparisons were not measured.

Earlier numbered Markdown files retained in the development workspace describe the original planning stage. This README, the actual source and the final report describe the delivered system. Optional planned features are not presented as completed measurements.
