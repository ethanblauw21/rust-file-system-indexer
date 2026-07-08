#!/usr/bin/env python3
"""tools/eval_search.py — the mutation/sweep eval harness for hybrid search ranking.

Runs the 16-query x 7-mutation-column suite (tools/fixtures/queries.jsonl) against a
prepared union index (see tools/corpus/prepare_corpus.py), in one or more search
modes, and reports:

  - per-variant top-3 / top-10 hit-rate (summed across all 16 queries), for each
    of the 7 columns: base, synonym, typo, reorder, truncate, recase, filename_handle
  - mean mutation robustness (0-5 per query, over the 5 perturbation columns only —
    see eval_common.ROBUSTNESS_COLUMNS) and the fully-robust (5/5) query count
  - index size on disk and per-query search latency (p50/mean), so a tokenization or
    fusion change is judged on cost as well as quality (per the agent charter)

Every run is stamped with the current git SHA (+ dirty flag) and appended to
tools/baselines/search_eval_baseline.jsonl, keyed on (mode, weights, corpus) so a
repeat run with the same knobs overwrites its own row instead of accumulating noise.

This is a MEASURING INSTRUMENT, not a ranking-change tool: it never edits src/**. A
proposed ranking/tokenization change is specced and delegated to rust-implementer;
this script runs the BEFORE sweep, then the AFTER sweep once the change lands, and
the two runs' baseline rows are compared by hand (or via `--compare`) for the verdict.

Usage:
    # Prove the metric/plumbing logic without a built binary or corpus (see docstring
    # of _selftest for exactly what it does and does not prove):
    python tools/eval_search.py selftest

    # Real sweep once a corpus is prepared (see tools/corpus/prepare_corpus.py) and
    # the binary is built:
    python tools/eval_search.py run --index-dir .fileSystem-index-eval \\
        --modes hybrid,dense --label "R3-before"

    # Same, sweeping a ranking knob (paired before/after for a gated experiment):
    python tools/eval_search.py run --index-dir .fileSystem-index-eval \\
        --modes hybrid --rrf-sparse-weight 0.6 --label "R3-w0.6"

    # See the exact commands without running anything:
    python tools/eval_search.py run --index-dir .fileSystem-index-eval --dry-run
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (  # noqa: E402
    MUTATION_COLUMNS, ROBUSTNESS_COLUMNS, append_baseline, git_dirty, git_sha,
    mean, path_matches, read_jsonl,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = _HERE
_REPO_ROOT = os.path.dirname(_TOOLS)
_DEFAULT_QUERIES = os.path.join(_TOOLS, "fixtures", "queries.jsonl")
_DEFAULT_BASELINE = os.path.join(_TOOLS, "baselines", "search_eval_baseline.jsonl")

_FALLBACK_WARNING = "falling back to sparse"


def _resolve_binary(explicit):
    if explicit:
        return explicit
    exe = "file_indexer.exe" if os.name == "nt" else "file_indexer"
    for profile in ("release", "debug"):
        cand = os.path.join(_REPO_ROOT, "target", profile, exe)
        if os.path.exists(cand):
            return cand
    return None


def _dir_size_bytes(path):
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class SearchRunner:
    """Wraps `<binary> search ... --output json` and enforces the no-silent-degradation
    rule: if the binary silently falls back from dense/hybrid to sparse (no embedder
    configured — see src/main.rs's auto-fallback), that is loudly surfaced, never
    swallowed into a mislabeled "dense" or "hybrid" result row."""

    def __init__(self, binary, index_dir, weights, top_k=10, allow_fallback=False, verbose=False):
        self.binary = binary
        self.index_dir = index_dir
        self.weights = weights  # {"RRF_SPARSE_WEIGHT": ..., "PATH_BOOST_WEIGHT": ...} or {}
        self.top_k = top_k
        self.allow_fallback = allow_fallback
        self.verbose = verbose
        self.warnings = []

    def _env(self):
        env = dict(os.environ)
        for k, v in self.weights.items():
            if v is not None:
                env[k] = str(v)
        return env

    def command(self, query, mode):
        return [
            self.binary, "search", query,
            "--mode", mode,
            "--top-k", str(self.top_k),
            "--output", "json",
            "--index-dir", self.index_dir,
        ]

    def run(self, query, mode):
        """Returns (files_ranked: list[str], latency_s: float)."""
        cmd = self.command(query, mode)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        latency = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(
                f"search failed (exit {proc.returncode}) for mode={mode!r} query={query!r}\n"
                f"stderr: {proc.stderr.strip()}"
            )
        if _FALLBACK_WARNING in proc.stderr and mode != "sparse":
            msg = (f"mode={mode!r} query={query!r} SILENTLY FELL BACK TO SPARSE "
                   f"(no embedder configured) — this result is NOT a {mode} result.")
            self.warnings.append(msg)
            if not self.allow_fallback:
                raise RuntimeError(
                    "REFUSING to record a mislabeled result: " + msg +
                    "\nSet NOMIC_ONNX_PATH (see README) or pass --allow-sparse-fallback "
                    "to intentionally run a degraded smoke test."
                )
        if self.verbose and proc.stderr.strip():
            print(f"    [stderr] {proc.stderr.strip()}", file=sys.stderr)
        try:
            results = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"non-JSON stdout for mode={mode!r} query={query!r}: {e}\n{proc.stdout[:500]}")
        files = [r["file"] for r in sorted(results, key=lambda r: r["rank"])]
        return files, latency


def _load_fixtures(path):
    rows = list(read_jsonl(path))
    if not rows:
        raise SystemExit(f"No fixtures in {path}")
    return rows


def run_sweep(runner, fixtures, mode, log=print):
    """Runs every (query, mutation-column) pair once. Returns a result dict with
    per-variant hit sums, per-query robustness scores, and latency samples. Never
    silently drops a query on error — a failed query aborts the sweep (see
    SearchRunner.run), because a partial sweep masquerading as a full one is exactly
    the "silent cap" the charter forbids."""
    per_variant_top3 = {c: 0 for c in MUTATION_COLUMNS}
    per_variant_top10 = {c: 0 for c in MUTATION_COLUMNS}
    per_query_robustness = []
    latencies = []
    n = len(fixtures)

    for row in fixtures:
        target = row["target"]
        robust_hits = 0
        for col in MUTATION_COLUMNS:
            query_text = row[col]
            files, latency = runner.run(query_text, mode)
            latencies.append(latency)
            top10_hit = any(path_matches(f, target) for f in files[:10])
            top3_hit = any(path_matches(f, target) for f in files[:3])
            if top10_hit:
                per_variant_top10[col] += 1
            if top3_hit:
                per_variant_top3[col] += 1
            if col in ROBUSTNESS_COLUMNS and top10_hit:
                robust_hits += 1
            log(f"  q{row['id']:>2} [{col:<15}] top3={'Y' if top3_hit else '.'} "
                f"top10={'Y' if top10_hit else '.'}  \"{query_text}\"")
        per_query_robustness.append(robust_hits)

    mean_robustness = mean(per_query_robustness)
    fully_robust = sum(1 for r in per_query_robustness if r == len(ROBUSTNESS_COLUMNS))

    return {
        "n_queries": n,
        "mode": mode,
        "per_variant_top3": per_variant_top3,
        "per_variant_top10": per_variant_top10,
        "per_query_robustness": per_query_robustness,
        "mean_robustness": round(mean_robustness, 4),
        "fully_robust": fully_robust,
        "fully_robust_of": n,
        "latency_p50_s": round(statistics.median(latencies), 4) if latencies else None,
        "latency_mean_s": round(mean(latencies), 4) if latencies else None,
        "n_searches": len(latencies),
    }


def print_report(result, weights, index_size_bytes):
    print(f"\n=== mode={result['mode']}  weights={weights or '(defaults)'} ===")
    print(f"{'variant':<16} {'top-3':>8} {'top-10':>8}")
    for c in MUTATION_COLUMNS:
        n = result["n_queries"]
        print(f"{c:<16} {result['per_variant_top3'][c]:>4}/{n:<3} {result['per_variant_top10'][c]:>4}/{n:<3}")
    print(f"\nmean mutation robustness (of {len(ROBUSTNESS_COLUMNS)}): {result['mean_robustness']}")
    print(f"fully robust ({len(ROBUSTNESS_COLUMNS)}/{len(ROBUSTNESS_COLUMNS)}): "
          f"{result['fully_robust']}/{result['fully_robust_of']}")
    if result["latency_p50_s"] is not None:
        print(f"latency: p50={result['latency_p50_s']}s  mean={result['latency_mean_s']}s "
              f"over {result['n_searches']} searches")
    if index_size_bytes is not None:
        print(f"index size: {index_size_bytes / 1e6:.1f} MB ({index_size_bytes} bytes)")


def cmd_run(args):
    binary = _resolve_binary(args.binary)
    fixtures = _load_fixtures(args.queries)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    weights = {}
    if args.rrf_sparse_weight is not None:
        weights["RRF_SPARSE_WEIGHT"] = args.rrf_sparse_weight
    if args.path_boost_weight is not None:
        weights["PATH_BOOST_WEIGHT"] = args.path_boost_weight

    if args.dry_run:
        runner = SearchRunner("<binary>", args.index_dir, weights, top_k=args.top_k)
        for mode in modes:
            for row in fixtures[:1]:
                for col in MUTATION_COLUMNS:
                    print(" ".join(runner.command(row[col], mode)))
        print(f"... ({len(fixtures)} queries x {len(MUTATION_COLUMNS)} columns x {len(modes)} modes total)")
        return

    if binary is None:
        raise SystemExit(
            "No built file_indexer binary found under target/{release,debug}, and none "
            "given via --binary. Build it first (`cargo build --release`) — this wave "
            "deliberately does not run a live sweep against in-flux source; see "
            "tools/README.md."
        )

    index_size = _dir_size_bytes(args.index_dir)
    if index_size == 0:
        print(f"warning: {args.index_dir} is empty or missing — did you run "
              f"prepare_corpus.py build-index? Proceeding anyway (searches will 404/empty).",
              file=sys.stderr)

    sha = git_sha(_REPO_ROOT)
    dirty = git_dirty(_REPO_ROOT)
    timestamp = datetime.now(timezone.utc).isoformat()
    all_results = {}

    for mode in modes:
        runner = SearchRunner(binary, args.index_dir, weights, top_k=args.top_k,
                               allow_fallback=args.allow_sparse_fallback, verbose=args.verbose)
        print(f"\nRunning sweep: mode={mode} ...")
        result = run_sweep(runner, fixtures, mode)
        if runner.warnings:
            for w in runner.warnings:
                print(f"WARNING: {w}", file=sys.stderr)
        print_report(result, weights, index_size)
        all_results[mode] = result

        if not args.no_record:
            record = {
                "timestamp": timestamp,
                "git_sha": sha,
                "git_dirty": dirty,
                "label": args.label,
                "mode": mode,
                "weights": weights,
                "corpus": args.corpus_label,
                "index_dir": os.path.abspath(args.index_dir),
                "index_size_bytes": index_size,
                **{k: v for k, v in result.items() if k != "mode"},
            }
            path = append_baseline(record, args.baseline)
            print(f"-> recorded to {path}")

    return all_results


def _selftest():
    """Proves the metric/plumbing logic (mutation-column iteration, top-3/top-10
    scoring, robustness aggregation, path matching, baseline append/dedupe) end to
    end using a stubbed SearchRunner — NO built binary or prepared corpus required.

    This is deliberately NOT a substitute for a real sweep: it fabricates search
    results rather than querying a real index, so it proves the harness computes the
    RIGHT NUMBER from a given set of results, not that the engine returns good
    results. Confirming the latter (reproducing 4.69/12-of-16 or deriving a new
    baseline) is the deferred follow-up once the binary under concurrent hardening
    stabilizes — see tools/README.md "Corpus provenance".
    """
    fixtures = _load_fixtures(_DEFAULT_QUERIES)
    assert len(fixtures) == 16, f"expected 16 fixture queries, found {len(fixtures)}"
    for row in fixtures:
        for col in MUTATION_COLUMNS:
            assert col in row, f"fixture {row['id']} missing column {col!r}"

    class StubRunner:
        """Every mutation of every query hits at rank 1 except: query id 3's `typo`
        column (drops out of top-10 -> exercises a miss) and query id 7's `reorder`
        column (rank 4 -> top-10 hit but NOT top-3 -> exercises the top-3/top-10 split)."""
        warnings = []

        def run(self, query_text, mode):  # noqa: ARG002 (mode unused in stub)
            for row in fixtures:
                for col in MUTATION_COLUMNS:
                    if row[col] == query_text:
                        if row["id"] == 3 and col == "typo":
                            return ["distractor/unrelated_1.py", "distractor/unrelated_2.py"], 0.01
                        if row["id"] == 7 and col == "reorder":
                            return ["d1", "d2", "d3", row["target"]], 0.01
                        return [row["target"]], 0.01
            raise AssertionError(f"unrecognized stub query: {query_text!r}")

    result = run_sweep(StubRunner(), fixtures, "hybrid", log=lambda *a, **k: None)

    assert result["n_queries"] == 16
    # base/synonym/reorder/truncate/recase/filename_handle all hit top-3 for every
    # query except id 7's `reorder` (rank 4, top-10 only) and id 3's `typo` (miss).
    assert result["per_variant_top3"]["base"] == 16
    assert result["per_variant_top3"]["reorder"] == 15, result["per_variant_top3"]
    assert result["per_variant_top10"]["reorder"] == 16
    assert result["per_variant_top10"]["typo"] == 15, result["per_variant_top10"]
    assert result["per_variant_top3"]["typo"] == 15
    # robustness: 16 queries at 5/5 except query 3 (typo misses -> 4/5).
    assert result["fully_robust"] == 15, result
    assert abs(result["mean_robustness"] - ((15 * 5 + 4) / 16)) < 1e-9, result["mean_robustness"]

    # append_baseline dedupe-by-key round-trip, isolated to a temp file.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "baseline.jsonl")
        r1 = {"timestamp": "t1", "mode": "hybrid", "weights": {}, "corpus": "c1", "mean_robustness": 1.0}
        r2 = {"timestamp": "t2", "mode": "hybrid", "weights": {}, "corpus": "c1", "mean_robustness": 2.0}
        append_baseline(r1, path)
        append_baseline(r2, path)  # same key -> replaces r1, not appended alongside it
        rows = list(read_jsonl(path))
        assert len(rows) == 1 and rows[0]["mean_robustness"] == 2.0, rows

        r3 = {"timestamp": "t3", "mode": "dense", "weights": {}, "corpus": "c1", "mean_robustness": 3.0}
        append_baseline(r3, path)
        rows = list(read_jsonl(path))
        assert len(rows) == 2, rows

    assert path_matches(r"C:\Users\x\corpus\codebase-indexer\src\ast_chunker.py", "src/ast_chunker.py")
    assert not path_matches(r"C:\Users\x\corpus\src\NOT_ast_chunker.py", "src/ast_chunker.py")

    print("selftest OK: 16 fixtures load, all have 7 columns, top-3/top-10 scoring is "
          "correct on a stubbed miss + a rank-4 partial hit, robustness math matches "
          "hand computation, append_baseline dedupes by key, path_matches normalizes "
          "OS separators and rejects a false-suffix match.")
    print("NOTE: this does NOT prove the engine's actual hit-rate — see docstring.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the mutation sweep against a prepared index")
    p_run.add_argument("--index-dir", required=True)
    p_run.add_argument("--binary", help="path to file_indexer executable (default: auto-detect under target/)")
    p_run.add_argument("--queries", default=_DEFAULT_QUERIES)
    p_run.add_argument("--baseline", default=_DEFAULT_BASELINE)
    p_run.add_argument("--modes", default="hybrid,dense",
                        help="comma-separated: any of hybrid,dense,sparse. Default runs BOTH "
                             "hybrid and dense so every change reports the paired lift R1/R2 used.")
    p_run.add_argument("--top-k", type=int, default=10)
    p_run.add_argument("--rrf-sparse-weight", type=float, default=None,
                        help="override RRF_SPARSE_WEIGHT env var for this run (sweep knob)")
    p_run.add_argument("--path-boost-weight", type=float, default=None,
                        help="override PATH_BOOST_WEIGHT env var for this run (sweep knob)")
    p_run.add_argument("--corpus-label", default="union-v1",
                        help="free-text tag identifying which corpus (per manifest.toml) this "
                             "run used — part of the baseline's dedupe key, so runs against "
                             "different corpora never silently overwrite each other")
    p_run.add_argument("--label", default=None, help="free-text note for this run (e.g. 'R3-before')")
    p_run.add_argument("--allow-sparse-fallback", action="store_true",
                        help="do not abort when dense/hybrid silently degrades to sparse "
                             "(no NOMIC_ONNX_PATH configured) — only for embedder-less smoke tests")
    p_run.add_argument("--no-record", action="store_true", help="print the report but don't append to baseline")
    p_run.add_argument("--dry-run", action="store_true", help="print the commands that would run and exit")
    p_run.add_argument("--verbose", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_self = sub.add_parser("selftest", help="verify harness metric logic without a binary/corpus")
    p_self.set_defaults(func=lambda _args: _selftest())

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
