# Search-eval harness (`tools/`)

The mutation/sweep instrument that gates every ranking, fusion, or tokenization
change to `src/search.rs` / `src/chunker.rs` / FTS5 tokenizer config. Owned by the
search-eval-scientist agent role. Promoted out of the informal `scratchpad/sweep_*.py`
scripts referenced (but never committed) in
`docs/dogfooding/2026-06-22-dogfood-run.md`.

**The rule this harness exists to enforce:** no ranking change ships without a paired
before/after run of this suite, in both dense-only and hybrid mode, on the same
corpus, with the git SHA stamped on both runs.

## Files

| Path | Role |
|---|---|
| `tools/corpus/manifest.toml` | Pinned public-repo roots (target + distractors) that make up the reproducible union eval corpus. |
| `tools/corpus/prepare_corpus.py` | Clones/checks out the manifest roots at their pinned SHAs, then indexes them all into one union `--index-dir` (multi-root accumulation, post-Defect-6). |
| `tools/fixtures/queries.jsonl` | The 16 hand-authored queries, each with 7 columns: `base`, `synonym`, `typo`, `reorder`, `truncate`, `recase`, `filename_handle`, plus the expected `target` file (relative to the `codebase-indexer` corpus root) and a `class` (`impl`/`doc`). |
| `tools/eval_common.py` | Shared metric/git-SHA/baseline-append helpers (mirrors the sibling `indexer` repo's `tools/eval_common.py`, ADR-007/ADR-019 pattern). |
| `tools/eval_search.py` | The sweep runner: drives `file_indexer search` over every (query, column, mode) triple, computes hit-rate + robustness + latency + index size, appends a stamped row to `tools/baselines/search_eval_baseline.jsonl`. |
| `tools/baselines/search_eval_baseline.jsonl` | Git-SHA-stamped, append-only (dedup-by-key) record of every sweep run. Starts empty in this commit — see "Corpus provenance" below for why no row is fabricated here yet. |

## Corpus provenance (read this before trusting any number)

The original **4.69 mean robustness / 12-of-16 fully robust** baseline
(`docs/dogfooding/2026-06-22-dogfood-run.md`, Remediation R1/R2) was measured against
a union of `codebase-indexer` (92 files) + `ethan`'s personal `homelab`/`OneDrive`/
`Documents` folders (1,526 files), 1,618 files total, on a machine and user profile
(`C:\Users\ethan\...`) this repo has no access to. **That exact corpus cannot be
rebuilt from anything committable** — it is not just unindexed, it is personal data
that should never be checked in.

What *is* reproducible: the 16 fixture queries in `docs/dogfooding/2026-06-22-dogfood-run.md`
are all grounded in `codebase-indexer`'s own file contents (`hybrid_retriever.py`,
`ast_chunker.py`, `adapters/cpp_adapter.py`, ADR-003/007/008/014/017, ...) — and that
project is a **public GitHub repo** (`ethanblauw21/codebase-indexer`), independently
reachable by URL + commit SHA, not tied to any one machine or username. `tools/corpus/manifest.toml`
pins it as the `target` root. The original distractor folders are swapped for five
public repos (click, zustand, p-queue, serilog, spdlog) already vetted and pinned by
the sibling `indexer` repo's ADR-019 real-repo eval — same reproducibility guarantee
(license-clear, SHA-pinned, clonable by anyone), comparable scale (~1,126 files vs.
the original ~1,618), same spirit (unrelated code+doc noise across languages).

**Consequence: running this harness will not reproduce "4.69" literally.** The
fixture *queries* are faithfully reconstructed from the dogfood doc's query/target
table (§"Mutation loop log"); the per-query *mutation text* (synonym/typo/reorder/
truncate/recase wordings) was **not** preserved anywhere in-repo and has been
freshly hand-authored here — a defensible reconstruction, not a byte-identical
replay. Treat the first real run of this harness as **establishing a new baseline**
on a corpus and fixture set that — unlike the original — is fully reproducible from
this commit forward, not as re-deriving the old number. Record that new number (and
this provenance note) in the ADR that reports it.

**This wave (2026-07-08) deliberately stops short of running that first real sweep.**
`src/search.rs`, `src/chunker.rs`, and related ranking code are being hardened
concurrently by another agent in this same work session — running the harness now
would sweep a moving target and produce a number that's stale before it's recorded.
The harness below is built, self-tested, and structurally ready; the first real
`run` is a deferred follow-up once that hardening lands and the binary is stable.

**Known ungated experiment to resolve through this harness, not around it:** branch
`origin/dogfooding/trigram-fuzzy-fts` (`4f3f28b`) already implements a trigram fuzzy
FTS5 channel for typo recovery, with no in-repo sweep evidence. Do not merge it
without a before/after pair through this harness.

## Setup

Requires:
- `cargo build --release` (per repo `CLAUDE.md` — needs the ONNX Runtime DLL +
  nomic-embed-text-v1.5 model; **dense/hybrid modes silently degrade to sparse-only**
  if `NOMIC_ONNX_PATH` is unset — see "Gotchas" below, this harness refuses to
  record a silently-degraded run by default).
- Python 3.11+ (uses stdlib `tomllib`; falls back to `tomli` on older Pythons).
- `git` on `PATH`.

## Building the corpus

```bash
# 1. See what the manifest defines and what's already fetched locally:
python tools/corpus/prepare_corpus.py list

# 2. Clone every root at its pinned SHA into tools/corpus/.cache/ (git-ignored):
python tools/corpus/prepare_corpus.py fetch

#    Optional: if you already have the sibling `indexer` repo's ADR-019 corpus cache
#    checked out (benchmarks/real_repo/corpus/{click,zustand,p-queue,serilog,spdlog}),
#    reuse it instead of re-cloning the 5 distractor repos:
python tools/corpus/prepare_corpus.py fetch \
    --reuse-sibling-cache "C:\Users\<you>\Documents\indexer\benchmarks\real_repo\corpus"

# 3. Build the release binary, then index every fetched root into one union dir
#    (repeated `index <root>` calls accumulate post-Defect-6 — do NOT `--reindex`
#    between roots, that would only affect change-detection within a root):
cargo build --release
python tools/corpus/prepare_corpus.py build-index --index-dir .fileSystem-index-eval
```

`build-index` aborts the whole build (does not silently skip) if any single root
fails to index — a partial union corpus would silently invalidate every subsequent
sweep number.

## Running a sweep

```bash
# Prove the harness's metric logic is correct without a binary or corpus:
python tools/eval_search.py selftest

# See the exact commands a real run would issue, without running anything:
python tools/eval_search.py run --index-dir .fileSystem-index-eval --dry-run

# Real sweep, BOTH dense-only and hybrid (the default `--modes`) — this is what
# every gated experiment must run, before and after the change:
python tools/eval_search.py run --index-dir .fileSystem-index-eval \
    --label "R3-before"

# Sweeping a ranking knob (e.g. tuning RRF_SPARSE_WEIGHT, mirroring R1):
python tools/eval_search.py run --index-dir .fileSystem-index-eval \
    --modes hybrid --rrf-sparse-weight 0.6 --label "R3-w0.6"
```

Each `run` prints, per mode: the 7-column top-3/top-10 hit table, mean mutation
robustness (over the 5 perturbation columns — see below), fully-robust count,
p50/mean search latency, and index size on disk — then appends one stamped JSONL row
per mode to `tools/baselines/search_eval_baseline.jsonl` (git SHA + dirty flag,
timestamp, weights, corpus label; re-running the same `mode`+`weights`+`corpus-label`
combination replaces its row rather than accumulating duplicates).

### What "mean robustness" counts

The dogfood doc's 7-column table (`base`, `synonym`, `typo`, `reorder`, `truncate`,
`recase`, `filename_handle`) is fully reported, but the headline **"mean robustness
X/5"** number is computed over only 5 of those columns —
`synonym, typo, reorder, truncate, recase` (`eval_common.ROBUSTNESS_COLUMNS`) — per
query, then averaged over all 16 queries. `base` is excluded because robustness
measures resilience to *perturbation*, not the ability to hit the unperturbed query;
`filename_handle` is excluded because it is a qualitatively different probe
(searching a file's own basename) reported separately, exactly as the original
dogfood run's M9 finding treated it. This is a documented convention, not an
accident — do not change `ROBUSTNESS_COLUMNS` without renaming the metric (e.g.
`robustness7`) so historical baseline rows stay comparable.

### Gotcha: silent dense->sparse fallback

`src/main.rs`'s `search` handler silently switches `dense`/`hybrid` mode to `sparse`
and only warns on stderr if `NOMIC_ONNX_PATH` isn't configured (searcher has no
embedder). `eval_search.py`'s `SearchRunner` watches stderr for that warning and
**aborts the sweep** rather than silently recording a sparse result under a `dense`
or `hybrid` label — pass `--allow-sparse-fallback` only for an intentional
embedder-less smoke test, never for a real gated sweep.

## Comparing a before/after pair

Both runs land as separate rows in `tools/baselines/search_eval_baseline.jsonl`
(different `label`, or different `weights`/`git_sha`). Diff the two rows'
`mean_robustness`, `fully_robust`, `per_variant_top3`/`per_variant_top10`, and
(for tokenization changes) `index_size_bytes`/`latency_*_s` by hand, or `jq`:

```bash
grep '"label": "R3-before"' tools/baselines/search_eval_baseline.jsonl | jq .
grep '"label": "R3-after"'  tools/baselines/search_eval_baseline.jsonl | jq .
```

A change ships only if it beats the paired before-row on the agreed metric without
regressing any query that currently hits rank 1 — see the search-eval-scientist
agent charter for the full ship/no-ship rule.
