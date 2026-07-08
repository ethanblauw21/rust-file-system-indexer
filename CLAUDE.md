# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
cargo build

# Run tests (all)
cargo test

# Run a single test
cargo test <test_name>

# Run tests for a specific module
cargo test storage::tests

# Run with logging (crate is `file_indexer`, binary is `file_indexer`)
RUST_LOG=file_indexer=debug cargo run -- index /path/to/dir

# Release build (opt-level=3, thin LTO, single codegen unit)
cargo build --release
```

Rust **Edition 2024**. Requires the `nomic-embed-text-v1.5` ONNX model + tokenizer and an ONNX Runtime shared lib (`ort` load-dynamic) — see `README.md`; these are not in the repo. Indexer/search commands will fail without them, but the bulk of the test suite does not.

### CLI subcommands (`src/main.rs`)

```bash
file_indexer index <root> [--reindex] [--exclude NAME]... [--no-embed] [--embed-batch-size N]
                                                              # incremental index; hard-errors if no
                                                              # embedder is configured unless --no-embed
file_indexer search <query> [--mode hybrid|dense|sparse] [--like PATH] [--tier N] [--type EXT] [--output table|json|md]
file_indexer tui                                            # interactive ratatui UI
file_indexer explain <path> [--all-chunks]                 # per-chunk breakdown for one file
file_indexer score [--rescore]                             # compute chunk quality scores
file_indexer scores [--flagged-only] [--tier N]            # display flagged chunks
file_indexer recheck [--dry-run]                           # reindex files whose chunker method drifted
file_indexer ingest [--batch N]                             # push pre-fetched NDJSON records via stdin
                                                              # (one JSON object per line: {uri, content,
                                                              # mime_type, modified_at?, meta?})
```

All commands take `--index-dir` (default `.fileSystem-index`). `main()` sets a nonzero process exit
code on any command failure (`src/main.rs`, e.g. lines 208, 222, 259 — every `Err` arm calls
`std::process::exit(1)`); previously the process always exited 0 regardless of outcome.

## Architecture

This is a Rust port of a Python file indexer (`C:\Users\edb\Documents\indexer\fileSystem\`). The system indexes local files into a hybrid search store combining SQLite FTS5 (BM25 sparse) and LanceDB (dense vector) search, with Reciprocal Rank Fusion for query time.

### Implementation Phases

The project was built in sequential steps 1–9, all complete, followed by an undesignated
**Phase 1+2 hardening pass** (config file, fail-loud embedder/dim checks, `--no-embed`,
nonzero exit codes — see Key Design Decisions below). The CLI (`clap` dispatcher) lives in
**`src/main.rs`**, not `search.rs`.

| Step | Module | Description |
| ------ | -------- | ------------- |
| 1 | `storage.rs` | `StorageClient` trait + `LocalStorageClient` |
| 2 | `db.rs` | SQLite schema, r2d2 pool, FTS5 virtual table |
| 3 | `chunker.rs` | Multi-modal 3-tier chunking pipeline |
| 4 | `indexer.rs` | ONNX embedding (nomic-embed-text-v1.5) + LanceDB |
| 5 | `search.rs` | Hybrid RRF search (`Searcher`) |
| 6 | `scorer.rs` | Chunk quality scoring (structural + coherence) |
| 7 | `chunker_map.rs` | Per-MIME chunker method registry + drift detection |
| 8 | `tui.rs` | Interactive ratatui terminal UI |
| 9 | `indexer.rs` | Self-healing / method-drift reindexing (`recheck`) |
| hardening | `config.rs` | `file_indexer.toml` loader, CLI > file > default resolution, fail-loud `embedding_dim` validation |
| hardening | `main.rs` | `ingest` subcommand (NDJSON push path), `--no-embed`, nonzero exit codes on failure |

### Module Responsibilities

- **`main.rs`** — CLI entry point. `clap` subcommand dispatcher + output formatters (table with terminal hyperlinks / JSON / Markdown). Declares the module tree (`mod chunker; mod config; mod db; ...`). `run_recheck` computes method drift by comparing each flagged chunk's stored `chunker_method` against the current `ChunkerMap` mapping. Every command's `Err` arm calls `std::process::exit(1)` (e.g. `main.rs:208,222,259,...`) — failures now produce a nonzero exit code instead of silently exiting 0. `run_index`/`run_search`/`run_score` read the recorded `embed_mode` (`index_meta` table, see `db.rs`) and print an explicit note when a corpus was built `--no-embed` instead of returning silently-empty dense results.
- **`error.rs`** — Single `IndexerError` enum used by every module. Variants cover Io, FileTooLarge, NotFound, UnsupportedMime, Parse, Database, Embedding, VectorStore, Config, Other.
- **`storage.rs`** — `StorageClient` trait (object-safe: returns `Box<dyn Iterator>`) with `LocalStorageClient` implementing a stack-based DFS walker. Enforces 50 MiB file cap *before* allocation, MIME detection via extension match, and static ignore lists for dirs/extensions/filenames.
- **`db.rs`** — r2d2 connection pool with WAL-mode pragmas. Schema: `files`, `chunks`, `edges`, `index_meta` (single `key`/`value` row store, e.g. `embed_mode`) tables + `chunks_fts` FTS5 external content table with three sync triggers. `chunks` carries `lance_id`, `chunker_method`, `structural_score`, `coherence_score` (added via `ALTER TABLE` migrations in `init`).
- **`chunker.rs`** — Dispatches by MIME type (via `ChunkerMap`) to per-format parsers. Outputs three tiers: Tier 1 (≤500 tokens, atomic), Tier 2 (≤1500 tokens, contextual), Tier 3 (unlimited, architectural summary). Also produces `Edge` records for cross-file links. Stamps each chunk with the `chunker_method` string it was produced by.
- **`chunker_map.rs`** — `ChunkerMap`: MIME-type → method-version string (e.g. `rust/v1`, `markdown/v1`). Persisted to `<index_dir>/chunker_map.toml`, seeded from `DEFAULT_METHODS` on first run. Bumping a method version here is what makes `recheck` detect drift and reindex affected files.
- **`config.rs`** — `RawConfig`: loads `file_indexer.toml`, trying `<index_dir>/file_indexer.toml` then `./file_indexer.toml`, else all-`None` defaults (absence is normal, no error, no file auto-created). See "Configuration file" in Key Design Decisions for the full precedence rules.
- **`indexer.rs`** — `IncrementalIndexer` drives the full pipeline: list → parallel I/O+chunk → serial DB write → batch embed → LanceDB insert. Change detection is 3-stage: mtime → MD5 → full re-index. Owns embedding constants: `EMBEDDING_DIM = 256` (`indexer.rs:24` — canonical, see Key Design Decisions), `HF_MODEL_ID`, `stable_id`, `Embedder` (ONNX Runtime + tokenizer, hard load timeout), and `LanceStore`. `index_root` hard-errors when no embedder resolved and `--no-embed` wasn't passed (`indexer.rs:770`); with `--no-embed` it records `embed_mode="no_embed"` in `index_meta` and skips embedding.
- **`search.rs`** — `Searcher`/`hybrid_search` run dense (LanceDB ANN) and sparse (FTS5 BM25) **sequentially, not in parallel**: the dense `Future` is constructed first but sparse runs to completion synchronously *before* dense is awaited (`search.rs:520-525`), so despite `async fn` there is currently no overlap between the two channels. Fuses with weighted RRF (`k=60`) plus a path/filename boost. Also backs `--like` (file-similarity) and `explain`.
- **`scorer.rs`** — Chunk quality scoring. `structural_score` is a weighted sum of local heuristics (token range, sentence completeness, bracket balance, whitespace density). `coherence_score` is async: cosine similarity of a chunk's vector against the **mean of its parent-tier (tier+1) vectors** fetched from LanceDB (`None` for tier 3 / unembedded chunks, and reported as such when the index was built `--no-embed`). Thresholds: structural `< 0.5`, coherence `< 0.6` → flagged.
- **`tui.rs`** — Interactive ratatui UI (search / stats / explain / chunk-preview / indexing / flagged screens), 150 ms debounce on live search, can trigger an index run with the `i` key.

### Key Design Decisions

- `StorageClient` is `Box<dyn ...>`-safe; `LocalStorageClient` is the only backend for now; a `GoogleDriveStorageClient` is planned for Phase 2.
- `Bytes` (ref-counted) is used instead of `Vec<u8>` so chunker parsers can hold zero-copy slices of the same buffer.
- `_stable_id` in Step 4 must produce byte-for-byte identical IDs to the Python version (SHA-256 of `"uri:tier:index"`, first 15 hex chars masked to 60 bits).
- The `lance_id` stored in `chunks.lance_id` (SQLite) must equal the `id` column in LanceDB for every chunk — this invariant underpins RRF fusion in Step 5.
- All parsers in `chunker.rs` accept `&[u8]`, not `Vec<u8>`, to keep the call site zero-copy.
- **`EMBEDDING_DIM` (canonical statement of this fact — other docs should reference this bullet, not restate the number):** `EMBEDDING_DIM = 256` (`src/indexer.rs:24`) uses the Matryoshka slice of nomic-embed-text-v1.5 (first 256 of 768 dims, re-normalized). ~1.5% retrieval accuracy drop, 66% vector storage reduction. Changing it requires rebuilding the binary and running `index --reindex`. **Dimension-mismatch handling is fail-loud, not auto-drop:** `LanceStore::open_or_create` (`indexer.rs:147-192`) compares the stored table's vector width against `EMBEDDING_DIM` and, on mismatch, HARD-ERRORS naming both values unless the caller passes `force_recreate: true` — and only `index --reindex` ever passes `true`; every other caller (`search`, `ingest`, `recheck`) always passes `false`, so a stale or misconfigured embedder can never silently wipe an existing vector table. (An older design intended silent auto-drop-and-rebuild; that never shipped and must not be documented as current behavior.) `file_indexer.toml`'s `[embedder].embedding_dim` is **validation-only** (`config.rs::validate_embedding_dim`, `indexer.rs:725`): setting it doesn't change the compiled dimension, it just hard-errors at startup if it disagrees with the compiled `EMBEDDING_DIM`, naming both values.
- **Configuration file (`file_indexer.toml`)** — `config.rs`. Resolution order for the file itself: `<index_dir>/file_indexer.toml` → `./file_indexer.toml` (cwd) → built-in defaults (absence is normal — no error, no file auto-created). Sections: `[embedder]` (`onnx_model_dir`, `ort_dylib_path`, `embedding_dim` — validation-only, see above — `required_onnxruntime_version` — parsed but not yet enforced, `load_timeout_secs`), `[fusion]` (`dense_weight`, `sparse_weight`, `path_weight`), `[indexer]` (`extra_ignore_dirs`, `embed_batch_size`). Per-setting precedence is **not uniform** across sections: embedder paths resolve as config-file value → env var fallback (`NOMIC_ONNX_PATH`, `ORT_DYLIB_PATH`) → unset (`indexer.rs:728-731`); `embed_batch_size` and CLI flags resolve CLI > file > default via `config::resolve` (`indexer.rs:726`); fusion weights resolve as **env var (if set) overrides file, which overrides the hardcoded `RRF_*_WEIGHT`/`PATH_BOOST_WEIGHT` constants** (`search.rs::FusionWeights::resolve`, `search.rs:38-47`) — this env-wins ordering is intentionally kept as a quick tuning-only override and documented as such, not a bug. `[indexer].extra_ignore_dirs` is unioned with `--exclude`, never overridden, since more ignoring is always safe.
- **`--no-embed`** — `index` hard-errors (nonzero exit) if no embedder resolves (no `onnx_model_dir`/`NOMIC_ONNX_PATH`, or load failure) unless `--no-embed` is passed (`indexer.rs:770`, `main.rs` `Index.no_embed`). `--no-embed` records `embed_mode="no_embed"` in the new `index_meta` SQLite table (`db.rs:202`, `set_meta`/`get_meta`); `search --mode dense|hybrid`, `score`, and `explain` read the recorded mode back and print an explicit note instead of returning silently-empty/NULL dense results (`main.rs:444-451, 794-799`).
- **ONNX Runtime version pin** — `Cargo.toml` pins `ort = { version = "2.0.0-rc.12", default-features = false, features = ["load-dynamic", "api-24"] }`, which requires an onnxruntime **1.24.x** shared library (`ORT_DYLIB_PATH`/`[embedder].ort_dylib_path`); the dogfood run pinned `1.24.2` specifically (`docs/dogfooding/2026-06-22-dogfood-run.md`). A version-mismatched DLL no longer hangs the process indefinitely: `Embedder::load` runs on a dedicated thread under a hard timeout (`indexer.rs:360-380`, default 30 s, `[embedder].load_timeout_secs`) and surfaces `IndexerError::Embedding` naming the likely cause instead of wedging `index`/`search`.
- IVF-PQ partition count (`nlist`) is computed dynamically as `clamp(4√N, 4, 1024)` at index-build time. For N=9000 vectors this gives ~380 partitions vs. the naive fixed 32, which was severely under-partitioned.
- `lance_id IS NULL` in `chunks` is the two-phase commit signal: SQLite writes happen per file, LanceDB writes are batched across files and lance_ids are set in SQLite only after each successful LanceDB batch. On crash, files with NULL lance_ids are detected via `has_unembedded` in `get_all_file_info` and re-indexed on the next run.
- Chunker versioning: each chunk records the `chunker_method` (e.g. `rust/v1`) that produced it. To re-chunk everything after improving a parser, bump the version string in `chunker_map.rs`'s `DEFAULT_METHODS` **and** delete `<index_dir>/chunker_map.toml` (or edit it) so the new mapping loads; `recheck` then detects the drift and reindexes only the affected files. `recheck` reindexes *drifted* files silently but only *reports* files that are flagged-but-not-drifted (they need manual review, not automatic re-chunking).

### Known Limitations / Recommended Follow-ups

- **Hybrid search is not actually parallel.** `hybrid_search` (`search.rs:506-536`) constructs the dense
  search `Future` (`dense_fut`, line 520), then calls `sparse_search(...)` *synchronously* and runs it
  to completion (line 524), and only then `.await`s `dense_fut` (line 525). Rust futures are lazy —
  nothing runs until polled — so despite the function being `async`, the sparse and dense channels
  execute strictly back-to-back on the calling task, not concurrently. Any doc that says dense and
  sparse "run in parallel" is describing an unshipped design, not the current code.
  **Recommended follow-up (not done this pass — a code change, out of scope for a docs-only edit):**
  wrap the blocking `sparse_search` call in `tokio::task::spawn_blocking` (it does synchronous SQLite
  I/O) and drive both channels concurrently with `tokio::join!(dense_fut, spawn_blocking(sparse_search))`.
  This would change *latency* (the two channels genuinely overlap) but not *result semantics* — fusion
  math is unaffected. Should go through `rust-implementer`, and any latency claim should be re-measured
  (e.g. by `search-eval-scientist`) rather than asserted.
- **`required_onnxruntime_version` is parsed but not enforced.** `config.rs`'s `RawEmbedderConfig`
  parses `[embedder].required_onnxruntime_version` so the field name is stable in `file_indexer.toml`,
  but nothing currently checks the loaded ORT dylib's actual version against it (`config.rs:33-38`,
  `#[allow(dead_code)]`). The version-mismatch protection that exists today is the load-timeout in
  `Embedder::load` (see the ONNX Runtime version pin bullet above), which detects a mismatch
  indirectly (by timing out) rather than by comparing version strings.

### Vector Store: LanceDB vs. FAISS

LanceDB is the correct choice for this use case: it stores vectors on disk (no RAM footprint at rest), integrates directly with Arrow/Lance columnar format, and requires zero infrastructure. FAISS is in-process and memory-mapped — it would require `IndexIVFPQ` with `OnDiskInvertedLists` to avoid loading the full index into RAM, and still needs manual serialization.

If migrating to FAISS for finer control: use `IndexIVFPQ` (Product Quantization cuts memory 8–16×), back it with `OnDiskInvertedLists` for memory-mapping, and set `nprobe = max(1, √nlist)` at query time. The `stable_id` scheme and SQLite metadata layer are FAISS-agnostic and would carry over unchanged.
