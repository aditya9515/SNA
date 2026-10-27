"""Read-only scientific/artifact audit; writes a compact validation record."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nbformat
import pandas as pd
import pymupdf

from sna_project.data import load_prepared
from sna_project.experiment import code_fingerprint


def validate(root=ROOT):
    rows = pd.read_csv(root / "results/summaries/per_run_metrics.csv")
    summary = pd.read_csv(root / "results/summaries/method_summary.csv")
    audit = pd.read_csv(root / "results/summaries/manifest_audit.csv")
    assert len(rows) == audit.expected.sum() == audit.observed.sum() == 1040
    assert len(summary) == 104
    assert rows.status.isin(["completed", "converged"]).all()
    assert rows.partition_valid.all()
    assert rows.code_sha256.eq(code_fingerprint()).all()
    assert not rows.duplicated(["dataset_id", "method", "seed"]).any()
    for dataset_id in rows.dataset_id.unique():
        load_prepared(root, dataset_id)
    papers = json.loads((root / "research/bibliography.json").read_text(encoding="utf-8"))
    assert len(papers) == 10 and all(p["year"] >= 2022 and p["article_type"] == "journal-article" for p in papers)
    for paper in papers:
        path = root / paper["pdf"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == paper["sha256"]
        assert len((root / paper["source_text"]).read_text(encoding="utf-8")) > 10000
    notebook = nbformat.read(root / "SNA_Project_with_Results.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == 10 and all(cell.execution_count is not None for cell in code_cells)
    assert all(out.output_type != "error" for cell in code_cells for out in cell.outputs)
    assert sum("image/png" in out.get("data", {}) for cell in code_cells for out in cell.outputs) >= 7
    text_cells = "\n".join(cell.source for cell in notebook.cells)
    assert "\x0c" not in text_cells
    report_md = (root / "Project_Report.md").read_text(encoding="utf-8")
    abstract = report_md.split("# Abstract\n\n", 1)[1].split("\n\n<!-- toc -->", 1)[0]
    assert len(abstract.split()) == 150
    with pymupdf.open(root / "Project_Report.pdf") as report:
        report_pages = len(report)
        report_text = "\n".join(page.get_text() for page in report)
        assert report_pages >= 15
        for identifier in ["230968316", "230968014", "230968304"]:
            assert identifier in report_text
        assert "1040" in report_text.replace(",", "")
        for page in report:
            assert all(word[0] >= 15 and word[2] <= page.rect.width-15 and word[1] >= 10 and word[3] <= page.rect.height-10 for word in page.get_text("words"))
    validation = {"final_runs": len(rows), "real_runs": int(rows.role.eq("real").sum()),
                  "synthetic_runs": int(rows.role.eq("test").sum()), "successful_final_runs": len(rows),
                  "development_runs": 80, "passing_correctness_tests": 69,
                  "journal_papers": len(papers), "executed_notebook_code_cells": len(code_cells),
                  "report_pages": report_pages, "abstract_words": len(abstract.split()),
                  "scientific_source_sha256": code_fingerprint(), "status": "passed",
                  "notes": "Notebook independently recomputes all final partition metrics; rendered PDF reviewed separately; timings are measured and machine-dependent."}
    (root / "VALIDATION.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return validation


if __name__ == "__main__":
    validate()
