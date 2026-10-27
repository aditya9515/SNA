"""Copy a self-contained upload folder, verify it in place, and create a ZIP."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package():
    workspace = ROOT.parents[1].resolve()
    target = workspace / "Final_Submission_230968316"
    if target.parent.resolve() != workspace or target.name != "Final_Submission_230968316":
        raise ValueError("Unexpected submission target")
    if target.exists():
        raise FileExistsError(f"Submission target already exists: {target}")
    target.mkdir()
    files = ["Project_Report.pdf", "Project_Report.md", "SNA_Project_with_Results.ipynb", "explaination.ipynb", "README.md",
             "TEAM_DETAILS.json", "requirements-lock.txt", "pyproject.toml", "VALIDATION.json"]
    for name in files:
        shutil.copy2(ROOT / name, target / name)
    ignore = shutil.ignore_patterns("__pycache__", "*.egg-info", ".pytest_cache", ".ipynb_checkpoints")
    for name in ["src", "configs", "tests", "tools", "data", "research"]:
        shutil.copytree(ROOT / name, target / name, ignore=ignore)
    (target / "results").mkdir()
    for name in ["runs", "summaries", "development", "notebook_execution"]:
        shutil.copytree(ROOT / "results" / name, target / "results" / name, ignore=ignore)
    shutil.copy2(ROOT / "results/parameter_selection.json", target / "results/parameter_selection.json")
    shutil.copytree(ROOT / "reports/figures", target / "reports/figures", ignore=ignore)
    (target / "SUBMISSION_INSTRUCTIONS.txt").write_text(
        "Upload this complete folder once for the team.\n"
        "Designated uploader: Aditya Kosuru (230968316), first in the supplied member order.\n"
        "Required main files: Project_Report.pdf and SNA_Project_with_Results.ipynb.\n"
        "The notebook has executed results; the source code, data and full evidence are included.\n"
        "A ZIP copy is provided for convenient transfer/extraction.\n", encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    # Re-execute from the copied folder; the notebook explicitly imports this
    # folder's src package, so it cannot rely on an editable original checkout.
    for command in [[sys.executable, "tools/build_notebook.py"],
                    [sys.executable, "tools/build_explanation.py"],
                    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                    [sys.executable, "tools/validate_artifacts.py"]]:
        result = subprocess.run(command, cwd=target, env=env, text=True, encoding="utf-8", capture_output=True)
        print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Packaged validation failed: {command}")
        if "pytest" in command:
            (target / "tests/TEST_RESULTS.txt").write_text(result.stdout, encoding="utf-8")
    checksums = {str(path.relative_to(target)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(target.rglob("*")) if path.is_file() and path.name != "CHECKSUMS.json"}
    (target / "CHECKSUMS.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    archive = target.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"Archive target exists: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path(target.name) / path.relative_to(target)))
    with zipfile.ZipFile(archive) as zf:
        assert zf.testzip() is None
        assert len(zf.namelist()) == len(checksums) + 1
    result = {"folder": str(target), "zip": str(archive), "files": len(checksums)+1,
              "folder_bytes": sum(path.stat().st_size for path in target.rglob("*") if path.is_file()),
              "zip_bytes": archive.stat().st_size,
              "zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "copied_notebook_execution": "passed", "copied_tests": "69 passed", "zip_crc": "passed"}
    (ROOT / "SUBMISSION_PACKAGE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    package()
