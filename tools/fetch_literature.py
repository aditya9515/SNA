"""Acquire ten primary journal papers and preserve bibliographic evidence."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from bs4 import BeautifulSoup
from pypdf import PdfReader
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research"
PAPERS = [
    ("R01", "s41598-023-29610-z", "Large network community detection by fast label propagation"),
    ("R02", "s41598-023-46556-4", "A community partitioning algorithm for cyberspace"),
    ("R03", "s41598-024-68152-w", "Iterative embedding and reweighting of complex networks reveals community structure"),
    ("R04", "s41598-024-84962-4", "GATFELPA integrates graph attention networks and enhanced label propagation for robust community detection"),
    ("R05", "s41598-025-25905-5", "Graph embedding based label propagation for community detection in social networks"),
    ("R06", "s41598-022-20986-y", "Community detection in networks by dynamical optimal transport formulation"),
    ("R07", "s42005-022-00816-3", "Representative community divisions of networks"),
    ("R08", "s41467-024-52355-w", "Network community detection via neural embeddings"),
    ("R09", "s42005-024-01635-4", "Local dominance unveils clusters in networks"),
    ("R10", None, "Betweenness-driven overlapping label propagation community detection"),
]


def acquire(entry):
    ident, article, expected_title = entry
    doi = f"10.1038/{article}" if article else "10.33039/ami.2025.10.012"
    source = f"https://www.nature.com/articles/{article}" if article else "https://ami.uni-eszterhazy.hu/uploads/papers/finalpdf/AMI_61_from229to247.pdf"
    directory = OUT / "sources" / ident
    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "papers" / f"{ident}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    crossref_path = directory / "crossref.json"
    if not crossref_path.exists():
        response = requests.get("https://api.crossref.org/works/" + doi, timeout=90)
        response.raise_for_status()
        crossref_path.write_text(response.text, encoding="utf-8")
    metadata = json.loads(crossref_path.read_text(encoding="utf-8"))["message"]
    if article and not (directory / "publisher.html").exists():
        try:
            response = requests.get(source, timeout=20)
            if response.ok:
                (directory / "publisher.html").write_text(response.text, encoding="utf-8")
        except requests.RequestException as error:
            (directory / "html_access_note.txt").write_text(str(error), encoding="utf-8")
    if not pdf_path.exists():
        local = {"R01": "01_fast_label_propagation.pdf", "R02": "02_pr_lfm_community_partitioning.pdf"}.get(ident)
        if local:
            shutil.copy2(ROOT.parent / "papers" / local, pdf_path)
        else:
            try:
                response = requests.get(source + ".pdf" if article else source, timeout=35)
                response.raise_for_status()
                if not response.content.startswith(b"%PDF"):
                    raise ValueError(f"Not a PDF: {ident}")
                pdf_path.write_bytes(response.content)
            except (requests.RequestException, ValueError) as error:
                (directory / "pdf_access_note.txt").write_text(str(error), encoding="utf-8")
                if not (directory / "publisher.html").exists():
                    raise
    if pdf_path.exists():
        try:
            reader = PdfReader(pdf_path)
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception:
            import pymupdf
            with pymupdf.open(pdf_path) as document:
                pages = [page.get_text() for page in document]
        full_text = "\n\n".join(f"--- PDF PAGE {i+1} ---\n{text}" for i, text in enumerate(pages))
        page_count = len(pages)
    else:
        soup = BeautifulSoup((directory / "publisher.html").read_text(encoding="utf-8"), "html.parser")
        article_node = soup.find("article") or soup.find("main")
        if article_node is None or len(article_node.get_text()) < 10000:
            raise RuntimeError(f"No full article available for {ident}")
        full_text = article_node.get_text("\n", strip=True)
        page_count = None
    (directory / "full_text.txt").write_text(full_text, encoding="utf-8")
    dates = metadata.get("published", metadata.get("issued", {})).get("date-parts", [[None]])[0]
    record = {"id": ident, "title": metadata.get("title", [expected_title])[0], "doi": doi,
              "authors": "; ".join(" ".join([a.get("given", ""), a.get("family", "")]).strip() for a in metadata.get("author", [])),
              "journal": metadata.get("container-title", [""])[0], "year": dates[0],
              "article_type": metadata.get("type"), "source_url": source,
              "doi_url": "https://doi.org/" + doi, "pages": page_count,
              "pdf": f"research/papers/{ident}.pdf" if pdf_path.exists() else None,
              "source_text": f"research/sources/{ident}/full_text.txt",
              "sha256": hashlib.sha256(pdf_path.read_bytes() if pdf_path.exists() else full_text.encode()).hexdigest(),
              "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    print(f"{ident}: {record['year']} / {page_count or 'HTML'} pages / {record['title']}", flush=True)
    return record


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(acquire, PAPERS))
    (OUT / "bibliography.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    assert len(records) == 10 and all(r["year"] >= 2022 and r["article_type"] == "journal-article" for r in records)
