# Implementation phases and module responsibilities

Moved from CLAUDE.md, verbatim except the line-number fixes noted in the PR. Read when working on this area. Line-number citations drift; re-check them against the code.

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
- **`indexer.rs`** — `IncrementalIndexer` drives the full pipeline: list → parallel I/O+chunk → serial DB write → batch embed → LanceDB insert. Change detection is 3-stage: mtime → MD5 → full re-index. Owns embedding constants: `EMBEDDING_DIM = 256` (`indexer.rs:26` — canonical, see Key Design Decisions), `HF_MODEL_ID`, `stable_id`, `Embedder` (ONNX Runtime + tokenizer, hard load timeout), and `LanceStore`. `index_root` hard-errors when no embedder resolved and `--no-embed` wasn't passed (`indexer.rs:772`); with `--no-embed` it records `embed_mode="no_embed"` in `index_meta` and skips embedding.
- **`search.rs`** — `Searcher`/`hybrid_search` run dense (LanceDB ANN) and sparse (FTS5 BM25) **sequentially, not in parallel**: the dense `Future` is constructed first but sparse runs to completion synchronously *before* dense is awaited (`search.rs:520-525`), so despite `async fn` there is currently no overlap between the two channels. Fuses with weighted RRF (`k=60`) plus a path/filename boost. Also backs `--like` (file-similarity) and `explain`.
- **`scorer.rs`** — Chunk quality scoring. `structural_score` is a weighted sum of local heuristics (token range, sentence completeness, bracket balance, whitespace density). `coherence_score` is async: cosine similarity of a chunk's vector against the **mean of its parent-tier (tier+1) vectors** fetched from LanceDB (`None` for tier 3 / unembedded chunks, and reported as such when the index was built `--no-embed`). Thresholds: structural `< 0.5`, coherence `< 0.6` → flagged.
- **`tui.rs`** — Interactive ratatui UI (search / stats / explain / chunk-preview / indexing / flagged screens), 150 ms debounce on live search, can trigger an index run with the `i` key.
