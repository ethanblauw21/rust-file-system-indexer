# COMPLETED at 5/13/26

# Step 2 — Database Schema & FTS5 (`db.rs`) ⏳

**Status:** Pending

## Goal
Implement the `rusqlite` connection pool and write the full schema creation script, replicating the exact FTS5 external content table and triggers from `db.py`.

## Python Reference
`C:\Users\edb\Documents\indexer\fileSystem\db.py` — `EnterpriseDB` class.

## What to Build

### Connection pool
- Use `r2d2` + `r2d2_sqlite` to create a `Pool<SqliteConnectionManager>`
- Apply all PRAGMAs on every new connection via a `CustomizeConnection` impl:
  - `journal_mode = WAL`
  - `synchronous = NORMAL`
  - `foreign_keys = ON`
  - `cache_size = -65536` (64 MiB)
  - `temp_store = MEMORY`
  - `mmap_size = 2147483648`
  - `threads = 4`

### Schema DDL (idempotent, `CREATE TABLE IF NOT EXISTS`)
Four tables exactly matching `db.py`:

```sql
files   (id, file_uri UNIQUE, mime_type, content_hash, size_bytes, indexed_at, modified_at)
chunks  (id, file_id FK→files, tier CHECK(1|2|3), chunk_index, content, token_count, meta, lance_id)
edges   (id, src_chunk FK→chunks, dst_uri, edge_type CHECK(LINKS_TO|REFERENCES|EMBEDS), meta)
```

**FTS5 virtual table** — external content table pointing at `chunks`:
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(content, content='chunks', content_rowid='id', tokenize='porter unicode61');
```

**Three sync triggers** — keep `chunks_fts` in lock-step with `chunks`:
```sql
chunks_ai  AFTER INSERT  → INSERT INTO chunks_fts
chunks_au  AFTER UPDATE  → delete old + insert new
chunks_ad  AFTER DELETE  → delete old
```

> Note: `faiss_id` column from Python is renamed `lance_id` (`INTEGER DEFAULT NULL`) to reflect the LanceDB backend used in Step 4.

### `EnterpriseDb` struct
Public API mirroring `EnterpriseDB` from Python:

| Method | Description |
|---|---|
| `new(path) -> Result<Self>` | Open/create DB, apply PRAGMAs, run DDL |
| `get_all_file_info() -> Result<HashMap<String, FileInfo>>` | Batch pre-fetch for change detection |
| `upsert_file(...) -> Result<i64>` | INSERT OR REPLACE, returns rowid |
| `delete_file(file_uri) -> Result<()>` | CASCADE deletes chunks + edges |
| `list_all_file_uris() -> Result<Vec<String>>` | For stale-file removal pass |
| `insert_chunks(file_id, chunks) -> Result<Vec<i64>>` | Batch insert, returns rowids |
| `delete_chunks_for_file(file_id) -> Result<()>` | Explicit purge (triggers handle FTS) |
| `set_chunk_lance_ids(map) -> Result<()>` | Bulk-update lance_id column |
| `get_chunks_by_lance_ids(ids) -> Result<HashMap<i64, ChunkRow>>` | O(top_k) join query |
| `insert_edges(edges) -> Result<()>` | Batch insert via `executemany` |
| `fts_search(query, limit) -> Result<Vec<FtsResult>>` | BM25 search with query sanitisation |
| `stats() -> Result<DbStats>` | Row counts for files/chunks/edges/fts |

### FTS5 query sanitisation
Replicate `_sanitize_fts_query` — strip `( ) " ^ : *` chars, collapse whitespace, return `""` for empty input.

### Transaction helper
All write paths wrapped in `BEGIN` / `COMMIT` / `ROLLBACK`. Use a closure-based helper:
```rust
fn with_tx<F, T>(conn: &Connection, f: F) -> Result<T, IndexerError>
where F: FnOnce(&Connection) -> Result<T, IndexerError>
```

## Key Rust Idioms to Apply
- `rusqlite::params![]` macro for all bound parameters (no format-string injection risk)
- `serde_json::to_string` / `from_str` for the `meta` JSON column (replaces `json.dumps` / `json.loads`)
- `r2d2::Pool::get()` returns a pooled connection — callers never hold a raw `Connection`
- `#[derive(Debug, Clone)]` on all row structs returned from queries
- Schema migration guard: `ALTER TABLE ... ADD COLUMN` wrapped in `if let Err(_) =` (column may already exist on upgrade)

## Tests to Write
- `schema_creates_all_tables` — verify `sqlite_master` contains all expected table/trigger names
- `upsert_and_get_file` — round-trip a `FileMetadata`
- `insert_chunks_and_fts_sync` — insert chunks, confirm `chunks_fts` rowcount matches
- `delete_file_cascades` — verify chunks and edges are removed on file delete
- `fts_search_returns_ranked_results` — insert known chunks, search for a keyword
- `fts_sanitize_rejects_special_chars` — ensure `"error: (null)"` doesn't panic
