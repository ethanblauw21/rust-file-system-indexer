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

# Run with logging
RUST_LOG=file_indexer=debug cargo run

# Release build
cargo build --release
```

## Architecture

This is a Rust port of a Python file indexer (`C:\Users\edb\Documents\indexer\fileSystem\`). The system indexes local files into a hybrid search store combining SQLite FTS5 (BM25 sparse) and LanceDB (dense vector) search, with Reciprocal Rank Fusion for query time.

### Implementation Phases

The project has been built in 5 sequential steps. All steps are complete:

| Step | Module | Status | Description |
| ------ | -------- | -------- | ------------- |
| 1 | `storage.rs` | ✅ Done | `StorageClient` trait + `LocalStorageClient` |
| 2 | `db.rs` | ✅ Done | SQLite schema, r2d2 pool, FTS5 virtual table |
| 3 | `chunker.rs` | ✅ Done | Multi-modal 3-tier chunking pipeline |
| 4 | `indexer.rs` | ✅ Done | ONNX embedding (nomic-embed-text-v1.5) + LanceDB |
| 5 | `search.rs` | ✅ Done | Hybrid RRF search + CLI (`clap`) |

Detailed specs for each pending step are in `docs/TODO Changes/step-N-*.md`.

### Module Responsibilities

- **`error.rs`** — Single `IndexerError` enum used by every module. Variants cover Io, FileTooLarge, NotFound, UnsupportedMime, Parse, Database, Embedding, VectorStore, Other.
- **`storage.rs`** — `StorageClient` trait (object-safe: returns `Box<dyn Iterator>`) with `LocalStorageClient` implementing a stack-based DFS walker. Enforces 50 MiB file cap *before* allocation, MIME detection via extension match, and static ignore lists for dirs/extensions/filenames.
- **`db.rs`** (Step 2) — r2d2 connection pool with WAL-mode pragmas. Schema: `files`, `chunks`, `edges` tables + `chunks_fts` FTS5 external content table with three sync triggers.
- **`chunker.rs`** (Step 3) — Dispatches by MIME type to per-format parsers. Outputs three tiers: Tier 1 (≤500 tokens, atomic), Tier 2 (≤1500 tokens, contextual), Tier 3 (unlimited, architectural summary). Also produces `Edge` records for cross-file links.
- **`indexer.rs`** (Step 4) — `IncrementalIndexer` drives the full pipeline: list → parallel I/O+chunk → serial DB write → batch embed → LanceDB insert. Change detection is 3-stage: mtime → MD5 → full re-index.
- **`search.rs`** (Step 5) — `Searcher` runs dense (LanceDB ANN) and sparse (FTS5 BM25) in parallel, fuses with RRF (`k=60`).

### Key Design Decisions

- `StorageClient` is `Box<dyn ...>`-safe; `LocalStorageClient` is the only backend for now; a `GoogleDriveStorageClient` is planned for Phase 2.
- `Bytes` (ref-counted) is used instead of `Vec<u8>` so chunker parsers can hold zero-copy slices of the same buffer.
- `_stable_id` in Step 4 must produce byte-for-byte identical IDs to the Python version (SHA-256 of `"uri:tier:index"`, first 15 hex chars masked to 60 bits).
- The `lance_id` stored in `chunks.lance_id` (SQLite) must equal the `id` column in LanceDB for every chunk — this invariant underpins RRF fusion in Step 5.
- All parsers in `chunker.rs` accept `&[u8]`, not `Vec<u8>`, to keep the call site zero-copy.
- `EMBEDDING_DIM = 256` uses the Matryoshka slice of nomic-embed-text-v1.5 (first 256 of 768 dims, re-normalized). ~1.5% retrieval accuracy drop, 66% vector storage reduction. Changing `EMBEDDING_DIM` requires `--reindex`; the LanceDB table is auto-dropped on dimension mismatch at startup.
- IVF-PQ partition count (`nlist`) is computed dynamically as `clamp(4√N, 4, 1024)` at index-build time. For N=9000 vectors this gives ~380 partitions vs. the naive fixed 32, which was severely under-partitioned.
- `lance_id IS NULL` in `chunks` is the two-phase commit signal: SQLite writes happen per file, LanceDB writes are batched across files and lance_ids are set in SQLite only after each successful LanceDB batch. On crash, files with NULL lance_ids are detected via `has_unembedded` in `get_all_file_info` and re-indexed on the next run.

### Vector Store: LanceDB vs. FAISS

LanceDB is the correct choice for this use case: it stores vectors on disk (no RAM footprint at rest), integrates directly with Arrow/Lance columnar format, and requires zero infrastructure. FAISS is in-process and memory-mapped — it would require `IndexIVFPQ` with `OnDiskInvertedLists` to avoid loading the full index into RAM, and still needs manual serialization.

If migrating to FAISS for finer control: use `IndexIVFPQ` (Product Quantization cuts memory 8–16×), back it with `OnDiskInvertedLists` for memory-mapping, and set `nprobe = max(1, √nlist)` at query time. The `stable_id` scheme and SQLite metadata layer are FAISS-agnostic and would carry over unchanged.
