# CLAUDE.md

Guidance for Claude Code in this repository.

## Commands

```bash
cargo build                       # release: cargo build --release (opt-level=3, thin LTO, one codegen unit)
cargo test                        # all tests; one: cargo test <name>; one module: cargo test storage::tests
RUST_LOG=file_indexer=debug cargo run -- index /path/to/dir   # crate and binary are `file_indexer`
```

Rust **Edition 2024**. Needs the `nomic-embed-text-v1.5` ONNX model + tokenizer and an ONNX Runtime shared lib (`ort` load-dynamic); see `README.md`. They are not in the repo; index/search commands fail without them, but most tests don't need them.

Subcommands (`src/main.rs`): `index`, `search`, `tui`, `explain`, `score`, `scores`, `recheck`, `ingest`; all take `--index-dir` (default `.fileSystem-index`). Flags and shapes: `docs/agents/cli-reference.md` or `--help`. Every command failure exits nonzero (`std::process::exit(1)` in each `Err` arm of `main.rs`).

## Architecture

A Rust port of a Python file indexer (the Python original at `indexer/fileSystem/` is no longer on disk). It indexes local files into a hybrid store: SQLite FTS5 (BM25 sparse) + LanceDB (dense vectors), fused with Reciprocal Rank Fusion at query time. The CLI (`clap` dispatcher) lives in `src/main.rs`, not `search.rs`. Modules: `storage` (StorageClient trait), `db` (SQLite + FTS5), `chunker` / `chunker_map` (3-tier chunking, per-MIME method versions), `indexer` (ONNX embedding + LanceDB, incremental pipeline, `recheck`), `search` (hybrid RRF), `scorer`, `tui`, `config`, `error`. Per-module detail: `docs/agents/modules-and-phases.md`.

## Key Design Decisions (invariants to preserve)

- `stable_id` must produce byte-for-byte identical IDs to the Python version (SHA-256 of `"uri:tier:index"`, first 15 hex chars masked to 60 bits).
- `chunks.lance_id` (SQLite) must equal the `id` column in LanceDB for every chunk: this underpins RRF fusion. `lance_id IS NULL` is the two-phase-commit signal; files with NULL lance_ids are re-indexed on the next run.
- `StorageClient` stays `Box<dyn ...>`-safe; chunker parsers take `&[u8]` (zero-copy), not `Vec<u8>`.
- **`EMBEDDING_DIM` (canonical statement of this fact; other docs, including `README.md`, reference this bullet and must not restate the number):** `EMBEDDING_DIM = 256` (`src/indexer.rs:26`) uses the Matryoshka slice of nomic-embed-text-v1.5 (first 256 of 768 dims, re-normalized). ~1.5% retrieval accuracy drop, 66% vector storage reduction. Changing it requires rebuilding the binary and running `index --reindex`. **Dimension-mismatch handling is fail-loud, not auto-drop:** `LanceStore::open_or_create` (`src/indexer.rs`) compares the stored table's vector width against `EMBEDDING_DIM` and hard-errors naming both values unless `force_recreate: true`; only `index --reindex` passes `true`, so `search`, `ingest` and `recheck` can never wipe an existing vector table. `file_indexer.toml`'s `[embedder].embedding_dim` is **validation-only** (`config.rs::validate_embedding_dim`): it hard-errors at startup if it disagrees with the compiled value.
- `index` hard-errors with no embedder unless `--no-embed`; `--no-embed` records `embed_mode="no_embed"` in `index_meta`, and `search`/`score`/`explain` print a note instead of silently returning empty dense results.
- Chunker versioning: to re-chunk everything after improving a parser, bump the version in `chunker_map.rs`'s `DEFAULT_METHODS` **and** delete `<index_dir>/chunker_map.toml`; `recheck` then reindexes only drifted files.
- Hybrid search runs sparse then dense **sequentially** (`hybrid_search` in `search.rs`), not in parallel; any doc saying otherwise describes an unshipped design. Not-yet-enforced: `[embedder].required_onnxruntime_version` is parsed only.
- **CUDA is opt-in** (`cargo build --features cuda`, ADR-006): a CUDA init failure is a named `IndexerError::Embedding`, never a silent CPU fallback, and the GPU path MUST use the fp16 model (`onnx_model_file = "model_fp16.onnx"`). Full text in `docs/agents/design-decisions.md`.
- `ort` is pinned in `Cargo.toml` and needs an onnxruntime 1.24.x lib; a mismatched DLL times out in `Embedder::load` (default 30 s) instead of hanging.

Config-file precedence (not uniform across sections), ORT pin, CUDA, nlist, known limitations and the LanceDB-vs-FAISS rationale: `docs/agents/design-decisions.md`, `docs/agents/known-limitations.md`, `docs/agents/vector-store-rationale.md`.
