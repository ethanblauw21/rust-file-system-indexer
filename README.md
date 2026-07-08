# Rust File System Indexer

A fast, general-purpose file indexer with hybrid semantic search — **SQLite FTS5 (BM25 sparse)** + **LanceDB (dense vector)** fused with Reciprocal Rank Fusion — built in Rust.

## What it does

Indexes any directory of files (code, Markdown, CSV, Excel, plain text) into a persistent hybrid search store and provides fast semantic search via a CLI and interactive TUI.

**Key features:**

- **Hybrid search** — BM25 sparse + dense vector search fused with RRF for best-of-both recall
- **On-disk vectors** — LanceDB stores vectors on disk with no RAM footprint at rest
- **Incremental indexing** — 3-stage change detection (mtime → MD5 → full re-index) only re-embeds changed files
- **Multi-format** — Markdown, CSV, Excel (`.xlsx`), plain text and code files
- **MIME-aware chunking** — 3-tier chunker dispatches by content type
- **Crash recovery** — two-phase commit ensures no partially-indexed files survive a crash
- **Interactive TUI** — ratatui terminal interface for browsing and searching results

## Installation

### Prerequisites

- Rust (Edition 2024) — install via [rustup.rs](https://rustup.rs)
- The `nomic-embed-text-v1.5` ONNX model and tokenizer (downloaded separately — see below)
- ONNX Runtime shared library (loaded dynamically via `ort`)

### Build

```bash
cargo build --release
```

The binary lands at `target/release/file_indexer`.

### Model Setup

`file_indexer` needs **three** things to enable dense/hybrid search (sparse-only search works without
any of them — see `--no-embed` below):

| # | What | Exact requirement |
|---|------|--------------------|
| 1 | ONNX model | `nomic-embed-text-v1.5.onnx` from [nomic-embed-text-v1.5-ONNX](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-ONNX) on Hugging Face |
| 2 | Tokenizer | `tokenizer.json` from the same Hugging Face repo |
| 3 | ONNX Runtime shared library | **onnxruntime 1.24.x** (the project was built and dogfooded against `1.24.2`) — download the release for your OS/arch from the [onnxruntime GitHub releases](https://github.com/microsoft/onnxruntime/releases) |

Items 1 and 2 must sit **in the same directory** — the loader looks for
`<dir>/nomic-embed-text-v1.5.onnx` and `<dir>/tokenizer.json` (`src/indexer.rs:408-409`). Item 3 is a
platform shared library (`onnxruntime.dll` / `.so` / `.dylib`), loaded dynamically via `ort`'s
`load-dynamic` feature — nothing to compile, just point at the file.

> **Version matters.** `Cargo.toml` pins `ort` to `features = ["load-dynamic", "api-24"]`, which requires
> onnxruntime **1.24.x**. An older/newer DLL (e.g. 1.22.0) doesn't fail cleanly — it can hang the ONNX
> Runtime's native `GetApi()` call. `file_indexer` now aborts that hang after a load timeout (default 30s)
> and reports an `IndexerError::Embedding` naming the likely version mismatch, but picking the right
> version up front avoids the wait. See `docs/dogfooding/2026-06-22-dogfood-run.md` (Defect 1) for the
> original deadlock writeup.

#### Telling `file_indexer` where these live

Point at directory 1+2 and file 3 in **either** of two ways (CLI-flag override doesn't exist for these
paths; see `CLAUDE.md`'s "Configuration file" design-decision bullet for the exact resolution order and
the other config sections):

- **`file_indexer.toml`** (checked at `<index_dir>/file_indexer.toml`, then `./file_indexer.toml`):
  ```toml
  [embedder]
  onnx_model_dir = "C:/models/nomic"     # directory containing both .onnx and tokenizer.json
  ort_dylib_path = "C:/onnxruntime/lib/onnxruntime.dll"
  ```
- **Environment variables** (fallback used only when the config file doesn't set the value):
  `NOMIC_ONNX_PATH` (directory) and `ORT_DYLIB_PATH` (file path).

If neither is set, `index` **hard-errors** by default — pass `--no-embed` to build a sparse-only (BM25)
index instead of failing (see `CLAUDE.md` for what `--no-embed` records and how `search`/`score` report it).

#### Optional setup script

`docs/setup_model.ps1` is a convenience PowerShell script that creates a model directory and prints the
exact `file_indexer.toml` / env-var lines to set once you've downloaded the three files above into it —
it does not download the ONNX Runtime shared library for you, since the correct release asset depends on
your OS/arch and Microsoft's release page is the authoritative source for that URL.

> **Note:** The ONNX model files and indexed data are not included in this repository due to size.

## Usage

```bash
# Index a directory (incremental — only changed files are re-embedded)
file_indexer index /path/to/directory

# Search (hybrid mode by default)
file_indexer search "authentication middleware"

# Choose search mode explicitly
file_indexer search --mode hybrid  "query"
file_indexer search --mode dense   "query"
file_indexer search --mode sparse  "query"

# Find files similar to a reference file
file_indexer search --like path/to/reference.md

# Interactive TUI
file_indexer tui

# Show detailed chunk breakdown for a file
file_indexer explain path/to/file.md

# Chunk quality scoring
file_indexer score          # compute scores
file_indexer scores         # display flagged chunks

# Detect method drift on flagged files
file_indexer recheck

# Index without embeddings (sparse/BM25-only) — required if no ONNX model/tokenizer/
# ORT dylib is configured; `index` otherwise hard-errors rather than building a
# corpus with silently-broken dense search
file_indexer index /path/to/directory --no-embed

# Push pre-fetched records (NDJSON on stdin: one {uri, content, mime_type, ...} object
# per line) instead of walking a directory
cat records.ndjson | file_indexer ingest
```

### Output formats

```bash
file_indexer search "query"                    # table with hyperlinks (default)
file_indexer search "query" --output json      # JSON
file_indexer search "query" --output markdown  # Markdown
```

## Architecture

### Hybrid Search

Each query runs against both backends, then the results are fused:

| Backend | Technology | Algorithm |
|---------|-----------|-----------|
| Dense | LanceDB (IVF-PQ) | Approximate nearest-neighbor |
| Sparse | SQLite FTS5 | BM25 |
| Fusion | — | Weighted Reciprocal Rank Fusion (`k=60`) + filename/path-token boost |

**Not currently parallel.** `hybrid_search` builds the dense-search future first, then runs the sparse
(FTS5) query synchronously to completion, and only then awaits the dense future (`src/search.rs:520-525`).
Because Rust futures are lazy, nothing overlaps — the two channels run back-to-back on one task despite
the function being `async`. See `CLAUDE.md`'s "Known Limitations" section for the recommended
`tokio::join!` + `spawn_blocking` fix (not yet implemented; would change latency, not fusion results).

IVF partition count (`nlist`) is computed dynamically as `clamp(4√N, 4, 1024)` — prevents under-partitioning on large indexes. For N=9,000 vectors this yields ~380 partitions vs. a naive fixed 32.

### Three-Tier Chunking

Files are chunked at three granularities, dispatched by MIME type:

| Tier | Token budget | Purpose |
|------|-------------|---------|
| 1 | ≤500 | Atomic — per-symbol or per-paragraph |
| 2 | ≤1,500 | Contextual — section-level |
| 3 | Unlimited | Architectural — summary-level |

Supported formats: Markdown (`pulldown-cmark`), CSV (`csv` + `encoding_rs`), Excel (`calamine`), plain text/code. Files over 50 MiB are rejected before allocation.

### Embeddings

Uses a **Matryoshka slice** of `nomic-embed-text-v1.5` — a truncated, re-normalized prefix of the full
embedding. The exact dimension (`EMBEDDING_DIM`), the accuracy/storage tradeoff numbers, and the
dimension-mismatch handling are documented once, canonically, in `CLAUDE.md`'s "Key Design Decisions"
section (the `EMBEDDING_DIM` bullet) — see there rather than this file for the current value and the
full rationale, so the number can't drift between two docs.

Short version for orientation: changing the compiled dimension requires a rebuild and `index --reindex`;
on a stored-vs-expected mismatch, the vector store **hard-errors by default** (naming both dimensions)
rather than silently rebuilding — only `--reindex` grants permission to drop and recreate the table.
`file_indexer.toml`'s `[embedder].embedding_dim`, if set, is **validation-only**: it doesn't change the
compiled dimension, it just hard-errors at startup if it disagrees with it.

### Two-Phase Commit

- SQLite writes happen per file immediately after chunking
- LanceDB writes are batched across files; `lance_id` is set in SQLite only after each successful LanceDB batch
- On crash recovery, `lance_id IS NULL` in `chunks` signals incomplete commits — affected files are automatically re-indexed on next run

### Module Layout

| Module | Responsibility |
|--------|---------------|
| `main.rs` | CLI dispatcher (`clap`), output formatters (table / JSON / Markdown) |
| `error.rs` | Single `IndexerError` enum used across all modules |
| `storage.rs` | `StorageClient` trait + `LocalStorageClient` (stack-based DFS walker) |
| `db.rs` | SQLite schema, r2d2 pool, FTS5 virtual table with sync triggers |
| `chunker.rs` | 3-tier MIME-aware chunking pipeline |
| `chunker_map.rs` | Track which chunker was used per file |
| `config.rs` | `file_indexer.toml` loader (config-file > env-var-fallback / CLI > default resolution — see `CLAUDE.md`) |
| `indexer.rs` | `IncrementalIndexer` — list → chunk → embed → store pipeline; hard-errors on a missing embedder unless `--no-embed` |
| `search.rs` | Hybrid RRF searcher, query builder (dense + sparse run sequentially, not in parallel — see Hybrid Search above) |
| `scorer.rs` | Chunk quality scoring (structural heuristics + coherence metrics) |
| `tui.rs` | Interactive ratatui terminal UI |

## Development

```bash
# Build
cargo build

# Run all tests
cargo test

# Run a single test
cargo test <test_name>

# Run tests for a specific module
cargo test storage::tests

# Run with debug logging
RUST_LOG=file_indexer=debug cargo run -- index /path

# Release build (opt-level=3, thin LTO)
cargo build --release
```

## Key Design Decisions

- **LanceDB over FAISS** — LanceDB requires zero infrastructure and stores vectors on disk. FAISS would need `IndexIVFPQ` with `OnDiskInvertedLists` for comparable memory behavior, plus manual serialization.
- **Matryoshka-sliced embeddings** — see `CLAUDE.md`'s "Key Design Decisions" (`EMBEDDING_DIM` bullet) for the canonical dimension, rationale, and dimension-mismatch handling; not restated here to avoid the two docs drifting.
- **Stable IDs** — 60-bit deterministic IDs (SHA-256 of `"uri:tier:index"`, first 15 hex chars masked to 60 bits). ID scheme matches the Python predecessor for compatibility.
- **Zero-copy chunking** — `Bytes` (ref-counted) is used instead of `Vec<u8>` so chunker parsers hold zero-copy slices of the same buffer.
- **`StorageClient` trait** — Object-safe (`Box<dyn ...>`). `LocalStorageClient` is the current backend; a `GoogleDriveStorageClient` is planned for Phase 2.
- **`lance_id` invariant** — The `lance_id` stored in `chunks` (SQLite) must equal the `id` column in LanceDB for every chunk. This invariant underpins RRF fusion at query time.
- **Configuration file** — `file_indexer.toml`, loaded from `<index_dir>/file_indexer.toml` then `./file_indexer.toml`, else built-in defaults. Model paths configure here (or via `NOMIC_ONNX_PATH`/`ORT_DYLIB_PATH` as a fallback) — see "Model Setup" above and `CLAUDE.md`'s "Configuration file" bullet for the full precedence rules per section.
- **`--no-embed`** — `index` hard-errors if no embedder is configured, unless `--no-embed` is passed; the choice is recorded (`embed_mode` in the index) so `search`/`score` report a degraded corpus explicitly instead of silently returning empty dense results.

## Known Limitations

- **Hybrid search's dense and sparse channels run sequentially, not in parallel** — see "Hybrid Search" above and `CLAUDE.md` for the recommended (not yet implemented) `tokio::join!`/`spawn_blocking` fix.
- **`[embedder].required_onnxruntime_version` in `file_indexer.toml` is parsed but not enforced** — nothing currently checks the loaded ORT dylib's version against it; the only mismatch protection today is the load-timeout described in "Model Setup" above.

## Roadmap

- [ ] Google Drive storage backend (`GoogleDriveStorageClient`)
- [ ] PDF parsing support
- [ ] Make hybrid search's dense/sparse channels genuinely concurrent (`tokio::join!` + `spawn_blocking`)
- [ ] Enforce `[embedder].required_onnxruntime_version` against the loaded ORT dylib instead of only detecting a mismatch indirectly via load timeout
