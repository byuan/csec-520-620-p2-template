#!/usr/bin/env python3
"""Automated grading harness for CSEC 520/620 — Project 2 (K-means).

Runs the OBJECTIVE checks a grading agent needs as evidence, then writes
grading/auto_report.json. The agent reads that file, adds qualitative judgment
(discussion quality, correctness, code quality), and produces the final grade
per grading/AGENT_GRADING.md.

Requires PyYAML (included in requirements.txt). Run from the repo root:  python grading/grade.py
"""
from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["config.yaml", "src/cluster.py", "src/data.py", "src/kmeans.py",
            "src/distances.py", "src/evaluate.py", "README.md", "requirements.txt"]
# Clustering metrics we expect in results/metrics.json.
METRIC_KEYS = {"k", "inertia", "silhouette", "v_measure", "accuracy", "f1"}
UNIT_INTERVAL = {"silhouette", "homogeneity", "completeness", "v_measure",
                 "nmi", "accuracy", "precision", "recall", "f1"}

_venv = ROOT / ".venv" / "bin" / "python"
PYEXE = str(_venv) if _venv.exists() else sys.executable


def run(cmd, timeout=900):
    """Run a command from ROOT; return (returncode, tail_of_output)."""
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return 127, str(e)


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def check_structure():
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    return {"id": "structure", "status": "pass" if not missing else "fail",
            "evidence": "all present" if not missing else f"missing: {missing}"}


def check_tests():
    rc, out = run([PYEXE, "-m", "pytest", "-q"], timeout=300)
    status = "pass" if rc == 0 else ("dep_missing" if "No module named" in out else "fail")
    return {"id": "tests", "status": status, "returncode": rc, "evidence": out[-800:]}


def check_reproduce():
    rc, out = run([PYEXE, "-m", "src.cluster", "--config", "config.yaml"])
    if rc == 0:
        status = "pass"
    elif "NotImplementedError" in out:
        status = "fail"          # stubs never filled in
    elif "No module named" in out:
        status = "dep_missing"
    else:
        status = "fail"
    return {"id": "reproduce_runs", "status": status, "returncode": rc,
            "evidence": out[-1200:]}


def _metrics():
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        output_dir = cfg["output"]["dir"]
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("output.dir must be a non-empty path string")
        p = ROOT / output_dir / "metrics.json"
        m = json.loads(p.read_text())
        if not isinstance(m, dict):
            raise ValueError("metrics.json must contain a JSON object")
        return m, None
    except Exception as e:
        return None, f"cannot read configured metrics: {e}"


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_metrics():
    m, err = _metrics()
    if m is None:
        return {"id": "metrics_present", "status": "fail", "evidence": err}
    missing = METRIC_KEYS - set(m)
    bad = {}
    for k in (METRIC_KEYS | UNIT_INTERVAL | {"adjusted_rand"}) & m.keys():
        v = m[k]
        valid = _number(v)
        if valid:
            if k == "k":
                valid = type(v) is int and v >= 1
            elif k == "inertia":
                valid = v >= 0
            elif k in {"silhouette", "adjusted_rand"}:
                valid = -1 <= v <= 1
            else:
                valid = 0 <= v <= 1
        if not valid:
            bad[k] = repr(v)
    ok = not missing and not bad
    head = {k: m[k] for k in sorted(METRIC_KEYS & set(m))}
    return {"id": "metrics_present", "status": "pass" if ok else "fail",
            "evidence": {"metrics": head, "missing_keys": sorted(missing),
                         "invalid_values": bad}}


def _scratch_class():
    """Return the ast.ClassDef for KMeansScratch, or None."""
    try:
        tree = ast.parse(_read(ROOT / "src" / "kmeans.py"))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "KMeansScratch":
            return node
    return None


def check_scratch_implemented():
    """The four core pieces must no longer be NotImplementedError stubs."""
    cls = _scratch_class()
    if cls is None:
        return {"id": "scratch_implemented", "status": "fail",
                "evidence": "KMeansScratch class not found in src/kmeans.py"}
    wanted = {"_init_centroids", "_assign", "_update", "fit"}
    stubs, found = [], set()
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            found.add(node.name)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Raise):
                    exc = sub.exc
                    name = getattr(exc, "id", None) or getattr(
                        getattr(exc, "func", None), "id", None)
                    if name == "NotImplementedError":
                        stubs.append(node.name)
    missing = sorted(wanted - found)
    ok = not stubs and not missing
    return {"id": "scratch_implemented", "status": "pass" if ok else "fail",
            "evidence": {"still_stubs": sorted(set(stubs)), "methods_missing": missing}}


def check_no_library_kmeans_in_scratch():
    """KMeansScratch must not delegate to a library K-means."""
    cls = _scratch_class()
    if cls is None:
        return {"id": "no_library_kmeans_in_scratch", "status": "fail",
                "evidence": "KMeansScratch class not found"}
    hits = []
    for node in ast.walk(cls):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = ",".join(a.name for a in node.names)
            if "sklearn" in mod or "scipy.cluster" in mod or "KMeans" in names:
                hits.append(f"import {mod}.{names}")
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name in {"KMeans", "MiniBatchKMeans", "kmeans", "kmeans2", "vq"}:
                hits.append(f"call {name}()")
    return {"id": "no_library_kmeans_in_scratch",
            "status": "clean" if not hits else "violation",
            "evidence": hits or "no library clustering call inside KMeansScratch"}


def check_config_uses_scratch():
    txt = _read(ROOT / "config.yaml")
    m = re.search(r"^\s*implementation:\s*([A-Za-z_]+)", txt, re.M)
    impl = m.group(1) if m else None
    return {"id": "config_uses_scratch",
            "status": "pass" if impl == "scratch" else "fail",
            "evidence": f"kmeans.implementation = {impl!r} (must be 'scratch' for submission)"}


def check_reference_agreement():
    """If the run recorded a reference check, report how close scratch came."""
    m, err = _metrics()
    if m is None:
        return {"id": "reference_agreement", "status": "review", "evidence": err}
    rc = m.get("reference_check")
    if not isinstance(rc, dict) or not rc:
        return {"id": "reference_agreement", "status": "review",
                "evidence": "no reference_check in metrics.json "
                            "(set compare_to_reference: true to record one)"}
    ratio, ari = rc.get("inertia_ratio"), rc.get("ari_vs_reference")
    ok = (_number(ratio) and 0 <= ratio <= 1.10) or (_number(ari) and 0.80 <= ari <= 1)
    return {"id": "reference_agreement", "status": "pass" if ok else "review",
            "evidence": rc}


def check_mahalanobis_implemented():
    src = _read(ROOT / "src" / "distances.py")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"id": "mahalanobis_implemented", "status": "fail",
                "evidence": "src/distances.py does not parse"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "mahalanobis_sqdist":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Raise):
                    exc = sub.exc
                    name = getattr(exc, "id", None) or getattr(
                        getattr(exc, "func", None), "id", None)
                    if name == "NotImplementedError":
                        return {"id": "mahalanobis_implemented", "status": "fail",
                                "evidence": "mahalanobis_sqdist is still a stub"}
            return {"id": "mahalanobis_implemented", "status": "pass",
                    "evidence": "mahalanobis_sqdist has a real body"}
    return {"id": "mahalanobis_implemented", "status": "fail",
            "evidence": "mahalanobis_sqdist not found"}


def check_label_leakage():
    """Clustering must not receive labels. Heuristic — agent should confirm."""
    hits = []
    patterns = [r"\.fit\s*\([^)]*\by\b[^)]*\)", r"fit\s*\(\s*X\s*,\s*y", r"labels\s*=\s*y\b"]
    for pyf in sorted((ROOT / "src").glob("*.py")):
        text = _read(pyf)
        for pat in patterns:
            for m in re.finditer(pat, text):
                hits.append(f"{pyf.name}: {m.group(0).strip()}")
    return {"id": "label_leakage_scan", "status": "review" if hits else "clean",
            "evidence": hits or "no sign of labels passed into clustering (heuristic only)"}


def check_report():
    rdir = ROOT / "report"
    files = ([p.name for p in rdir.glob("**/*")
              if p.suffix.lower() in {".tex", ".pdf", ".md"}] if rdir.exists() else [])
    has_real = any(f.lower().endswith((".tex", ".pdf")) for f in files)
    return {"id": "report_present", "status": "pass" if has_real else "review",
            "evidence": f"report files: {files}" if files else "no report/ .tex or .pdf found"}


def check_git_hygiene():
    txt = _read(ROOT / ".gitignore")
    committed = ([p.name for p in (ROOT / "data").glob("*")
                  if p.name not in {"README.md", ".gitkeep"}]
                 if (ROOT / "data").exists() else [])
    ok = "data/" in txt and "results/" in txt
    return {"id": "git_hygiene", "status": "pass" if ok else "review",
            "evidence": {"gitignore_covers_data_results": ok,
                         "possible_committed_data": committed}}


def main():
    checks = [check_structure(), check_tests(), check_reproduce(), check_metrics(),
              check_config_uses_scratch(), check_scratch_implemented(),
              check_no_library_kmeans_in_scratch(), check_reference_agreement(),
              check_mahalanobis_implemented(), check_label_leakage(),
              check_report(), check_git_hygiene()]
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo": ROOT.name,
        "project": "Project 2 — K-means Clustering",
        "checks": checks,
        "note_for_agent": (
            "These are OBJECTIVE signals only. status 'dep_missing' means the grader "
            "environment lacked a dependency — set up the env and re-run before "
            "penalizing reproducibility. 'scratch_implemented' and "
            "'no_library_kmeans_in_scratch' are the key integrity checks for this "
            "project: the student must write Lloyd's algorithm themselves. Confirm "
            "heuristic flags by reading the code, then score every rubric criterion "
            "per AGENT_GRADING.md."
        ),
    }
    out = ROOT / "grading" / "auto_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
