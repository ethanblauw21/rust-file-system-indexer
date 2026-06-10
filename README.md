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

This project uses [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-ONNX) for embeddings. Download the ONNX model and tokenizer from Hugging Face and place them in the path configured in `src/indexer.rs`.

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
```

### Output formats

```bash
file_indexer search "query"                    # table with hyperlinks (default)
file_indexer search "query" --output json      # JSON
file_indexer search "query" --output markdown  # Markdown
```

## Architecture

### Hybrid Search

Queries are dispatched to both backends in parallel, then fused:

| Backend | Technology | Algorithm |
|---------|-----------|-----------|
| Dense | LanceDB (IVF-PQ) | Approximate nearest-neighbor |
| Sparse | SQLite FTS5 | BM25 |
| Fusion | — | Reciprocal Rank Fusion (`k=60`) |

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

Uses the **Matryoshka slice** of `nomic-embed-text-v1.5` — the first 256 of 768 dimensions. This gives ~1.5% retrieval accuracy loss with a **66% reduction** in vector storage. Changing `EMBEDDING_DIM` requires `--reindex`; the LanceDB table is auto-dropped on dimension mismatch at startup.

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
| `indexer.rs` | `IncrementalIndexer` — list → chunk → embed → store pipeline |
| `search.rs` | Hybrid RRF searcher, query builder |
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
- **256-dim Matryoshka** — First 256 of 768 dims from nomic-embed-text-v1.5. ~1.5% retrieval accuracy loss, 66% storage reduction.
- **Stable IDs** — 60-bit deterministic IDs (SHA-256 of `"uri:tier:index"`, first 15 hex chars masked to 60 bits). ID scheme matches the Python predecessor for compatibility.
- **Zero-copy chunking** — `Bytes` (ref-counted) is used instead of `Vec<u8>` so chunker parsers hold zero-copy slices of the same buffer.
- **`StorageClient` trait** — Object-safe (`Box<dyn ...>`). `LocalStorageClient` is the current backend; a `GoogleDriveStorageClient` is planned for Phase 2.
- **`lance_id` invariant** — The `lance_id` stored in `chunks` (SQLite) must equal the `id` column in LanceDB for every chunk. This invariant underpins RRF fusion at query time.

## Roadmap

- [ ] Google Drive storage backend (`GoogleDriveStorageClient`)
- [ ] PDF parsing support
- [ ] Configurable model path via config file
