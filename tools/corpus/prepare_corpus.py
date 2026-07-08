#!/usr/bin/env python3
"""tools/corpus/prepare_corpus.py — build the reproducible union eval corpus.

Reads tools/corpus/manifest.toml (pinned public repos: one "target" root that holds
the 16 fixture query targets, several "distractor" roots that supply unrelated
ranking noise) and:

  1. `fetch`       — clone each root at its pinned commit SHA into a git-ignored
                      working dir (tools/corpus/.cache/<name>), skipping roots
                      already checked out at that exact SHA.
  2. `build-index` — run the file_indexer binary's `index` subcommand once per
                      root, all into the SAME --index-dir, so the roots accumulate
                      into one union index (multi-root indexing works post-Defect-6;
                      see docs/dogfooding/2026-06-22-dogfood-run.md Defect 6). This
                      is the "union corpus" tools/eval_search.py measures against.

Mirrors the sibling `indexer` repo's tools/real_repo_prepare.py (ADR-019): pin by SHA,
git-ignore the cloned/derived artifacts, commit only the manifest + fixtures + results.

Usage:
    python tools/corpus/prepare_corpus.py list
    python tools/corpus/prepare_corpus.py fetch [--only NAME] [--force]
                                                 [--reuse-sibling-cache PATH]
    python tools/corpus/prepare_corpus.py build-index --index-dir DIR --binary PATH
                                                        [--only NAME] [--reindex]
"""
import argparse
import json
import os
import shutil
import stat
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_TOOLS)
_MANIFEST = os.path.join(_HERE, "manifest.toml")
_CACHE = os.path.join(_HERE, ".cache")
_PREPARED = os.path.join(_CACHE, "PREPARED.json")


def load_manifest(path=_MANIFEST):
    with open(path, "rb") as f:
        data = tomllib.load(f)
    roots = data.get("root", [])
    if not roots:
        raise SystemExit(f"No [[root]] entries in {path}")
    return roots


def _on_rm_error(func, path, exc_info):
    """rmtree onerror: clear the read-only bit git sets on pack files, then retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _rmtree(path):
    if os.path.isdir(path):
        shutil.rmtree(path, onerror=_on_rm_error)


def _git(args, cwd=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check,
                           capture_output=True, text=True)


def _load_prepared():
    if os.path.exists(_PREPARED):
        with open(_PREPARED, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_prepared(state):
    os.makedirs(_CACHE, exist_ok=True)
    with open(_PREPARED, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def cmd_list(_args):
    roots = load_manifest()
    prepared = _load_prepared()
    print(f"{'name':<20} {'kind':<12} {'sha':<12} {'prepared':<10} url")
    for r in roots:
        st = prepared.get(r["name"], {})
        got_sha = st.get("sha", "")
        status = "OK" if got_sha == r["sha"] else ("stale" if got_sha else "missing")
        print(f"{r['name']:<20} {r['kind']:<12} {r['sha'][:10]:<12} {status:<10} {r['url']}")


def _clone_at_sha(root, dest, force):
    if os.path.isdir(dest):
        if force:
            _rmtree(dest)
        else:
            cur = _git(["rev-parse", "HEAD"], cwd=dest, check=False).stdout.strip()
            if cur == root["sha"]:
                print(f"  {root['name']}: already at {cur[:10]}, skipping (use --force to reclone)")
                return cur
            print(f"  {root['name']}: present at {cur[:10] or '?'}, want {root['sha'][:10]} — re-checking out")
            _rmtree(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  {root['name']}: cloning {root['url']} ...")
    _git(["clone", "--quiet", root["url"], dest])
    _git(["checkout", "--quiet", root["sha"]], cwd=dest)
    got = _git(["rev-parse", "HEAD"], cwd=dest).stdout.strip()
    if got != root["sha"]:
        raise SystemExit(f"  {root['name']}: checked out {got}, expected {root['sha']} — manifest/remote mismatch")
    print(f"  {root['name']}: OK @ {got[:10]}")
    return got


def _reuse_from_sibling(root, dest, sibling_cache):
    """Copy an already-cloned, already-pinned repo out of the sibling indexer repo's
    ADR-019 cache (benchmarks/real_repo/corpus/<name>) instead of re-cloning, when the
    SHA there already matches the manifest. Saves bandwidth for the 5 distractor repos
    this manifest deliberately reuses from that ADR (see manifest.toml comment)."""
    src = os.path.join(sibling_cache, root["name"])
    if not os.path.isdir(src):
        return None
    cur = _git(["rev-parse", "HEAD"], cwd=src, check=False).stdout.strip()
    if cur != root["sha"]:
        print(f"  {root['name']}: sibling cache present but at {cur[:10] or '?'} (want {root['sha'][:10]}) — not reusing")
        return None
    _rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".git"))
    print(f"  {root['name']}: reused from sibling cache ({src}), verified @ {cur[:10]}")
    return cur


def cmd_fetch(args):
    roots = load_manifest()
    if args.only:
        wanted = set(args.only)
        roots = [r for r in roots if r["name"] in wanted]
        if not roots:
            raise SystemExit(f"--only matched nothing in manifest (have: {[r['name'] for r in load_manifest()]})")
    prepared = _load_prepared()
    for r in roots:
        dest = os.path.join(_CACHE, r["name"])
        sha = None
        if args.reuse_sibling_cache and not args.force:
            sha = _reuse_from_sibling(r, dest, args.reuse_sibling_cache)
        if sha is None:
            sha = _clone_at_sha(r, dest, args.force)
        prepared[r["name"]] = {"sha": sha, "kind": r["kind"], "path": dest}
    _save_prepared(prepared)
    print(f"\nPrepared {len(roots)} root(s) into {_CACHE}")


def _resolve_binary(explicit):
    if explicit:
        return explicit
    exe = "file_indexer.exe" if os.name == "nt" else "file_indexer"
    for profile in ("release", "debug"):
        cand = os.path.join(_REPO_ROOT, "target", profile, exe)
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        "Could not find a built file_indexer binary under target/{release,debug}. "
        "Build it first (`cargo build --release`) or pass --binary explicitly."
    )


def cmd_build_index(args):
    roots = load_manifest()
    if args.only:
        wanted = set(args.only)
        roots = [r for r in roots if r["name"] in wanted]
    prepared = _load_prepared()
    missing = [r["name"] for r in roots if r["name"] not in prepared]
    if missing:
        raise SystemExit(f"Roots not fetched yet: {missing} — run `fetch` first")
    binary = _resolve_binary(args.binary)
    os.makedirs(args.index_dir, exist_ok=True)
    for r in roots:
        path = prepared[r["name"]]["path"]
        cmd = [binary, "index", path, "--index-dir", args.index_dir]
        if args.reindex:
            cmd.append("--reindex")
        print(f"$ {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=_REPO_ROOT)
        if proc.returncode != 0:
            raise SystemExit(
                f"index of root '{r['name']}' ({path}) exited {proc.returncode} — "
                f"aborting union build (do NOT silently skip a root; a partial union "
                f"invalidates the sweep)."
            )
    print(f"\nUnion index built at {args.index_dir} from {len(roots)} root(s): "
          f"{[r['name'] for r in roots]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="show manifest + local prepared state")
    p_list.set_defaults(func=cmd_list)

    p_fetch = sub.add_parser("fetch", help="clone/checkout each root at its pinned SHA")
    p_fetch.add_argument("--only", nargs="*", help="only these root names")
    p_fetch.add_argument("--force", action="store_true", help="re-clone even if already at the pinned SHA")
    p_fetch.add_argument("--reuse-sibling-cache", metavar="PATH",
                          help="path to the sibling repo's benchmarks/real_repo/corpus dir; "
                               "copy already-cloned distractor repos from there instead of re-cloning "
                               "(only used when the SHA there already matches this manifest)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_idx = sub.add_parser("build-index", help="index every fetched root into one union --index-dir")
    p_idx.add_argument("--index-dir", required=True)
    p_idx.add_argument("--binary", help="path to the file_indexer executable (default: auto-detect under target/)")
    p_idx.add_argument("--only", nargs="*", help="only these root names")
    p_idx.add_argument("--reindex", action="store_true")
    p_idx.set_defaults(func=cmd_build_index)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
