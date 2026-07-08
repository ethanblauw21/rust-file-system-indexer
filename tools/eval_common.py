#!/usr/bin/env python3
"""Shared eval helpers for tools/eval_search.py.

Mirrors the sibling `indexer` repo's tools/eval_common.py (ADR-007/ADR-019): metric
math, the git-SHA stamp, and a parameterized append-baseline live here so a future
second harness (e.g. a dedicated latency/size profiler) grades and persists results
identically instead of reimplementing "what counts as a hit" per script.
"""
import json
import os
import subprocess

MUTATION_COLUMNS = ("base", "synonym", "typo", "reorder", "truncate", "recase", "filename_handle")

# Robustness is computed over exactly these 5 perturbations — NOT `base` (the
# unperturbed query; a robustness score should measure resilience to *change*, not
# baseline hit-ability) and NOT `filename_handle` (a qualitatively different probe —
# searching a file's own basename — reported separately, same as the original
# dogfood run's M9 finding). This matches docs/dogfooding/2026-06-22-dogfood-run.md's
# "Mean query-mutation robustness ... /5" definition used by both R1 and R2 (the
# "16×7 mutation suite" language in that doc refers to the full 7-column matrix
# reported per query; robustness itself is the /5 subset). Do not silently redefine
# this set — if an experiment needs a different robustness definition, name it
# differently (e.g. "robustness7") rather than changing what "robustness" means for
# existing baseline rows.
ROBUSTNESS_COLUMNS = ("synonym", "typo", "reorder", "truncate", "recase")


def read_jsonl(path):
    """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def git_sha(cwd=None):
    """Short HEAD SHA for stamping a baseline record; ``"unknown"`` if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_dirty(cwd=None):
    """True if the working tree has uncommitted changes — a sweep run against a dirty
    tree is still useful for iteration, but must never be silently mistaken for a
    SHA-reproducible result. Callers stamp this alongside git_sha() in every record."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True
        )
        return bool(out.strip())
    except Exception:
        return None


def path_matches(returned_file, target_rel_path):
    """True if a search result's `file` field (an absolute or index-root-relative URI,
    OS-native separators) corresponds to a fixture's `target` (a repo-relative path with
    forward slashes, e.g. "src/adapters/cpp_adapter.py"). Matches on a normalized path
    suffix so we don't need to know the exact corpus checkout root at match time."""
    a = returned_file.replace("\\", "/").rstrip("/")
    b = target_rel_path.replace("\\", "/").rstrip("/")
    return a == b or a.endswith("/" + b)


def append_baseline(record, path, key_fields=("label", "mode", "weights", "corpus")):
    """Append ``record`` to a JSONL baseline, replacing any row with the same key.

    Rewrites the file sorted by key so diffs stay stable and git-friendly. Returns
    ``path``. Mirrors the sibling repo's eval_common.append_baseline (ADR-019 §4).
    """
    existing = []
    if os.path.exists(path):
        existing = list(read_jsonl(path))

    def _key(r):
        return tuple(json.dumps(r.get(f), sort_keys=True) for f in key_fields)

    key = _key(record)
    existing = [r for r in existing if _key(r) != key]
    existing.append(record)
    existing.sort(key=lambda r: (r.get("timestamp", ""), _key(r)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in existing:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0
