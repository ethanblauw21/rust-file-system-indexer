# Step 7 — Stats Enhancements: MIME Breakdown + Embedding Coverage (`db.rs` + `main.rs`)

**Status:** Complete

## Goal

`search --stats` currently shows five raw counts. Two additions make it
actionable:

1. **MIME-type distribution** — shows what kinds of files are indexed
2. **Embedding coverage** — shows what fraction of chunks have been written to
   LanceDB, making partially-failed runs immediately visible

## Desired Output

```
Index Statistics
  Files
    1,204  text/plain
       89  text/markdown
       23  application/pdf
        8  text/x-python
        4  application/vnd.openxmlformats-officedocument.wordprocessingml.document

  Chunks
    14,208  total
    14,208  embedded  (100%)
         0  pending

  Edges        312
  FTS docs  14,119
  Vectors   14,208
```

When coverage < 100%, the pending count is shown in yellow/red if the terminal
supports ANSI; otherwise it remains plain text.

## Changes Required

### 1. `src/db.rs` — new `DbStats` fields and queries

Extend `DbStats`:

```rust
#[derive(Debug, Clone)]
pub struct DbStats {
    pub files:            i64,
    pub chunks:           i64,
    pub chunks_embedded:  i64,   // ← new: COUNT(lance_id IS NOT NULL)
    pub edges:            i64,
    pub chunks_fts_docs:  i64,
    pub mime_counts:      Vec<(String, i64)>,  // ← new: (mime_type, count) DESC
}
```

Update `stats()` to populate the new fields:

```rust
pub fn stats(&self) -> Result<DbStats, IndexerError> {
    let conn = self.conn()?;

    let files: i64 =
        conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
    let chunks: i64 =
        conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
    let chunks_embedded: i64 = conn.query_row(
        "SELECT COUNT(*) FROM chunks WHERE lance_id IS NOT NULL", [], |r| r.get(0),
    )?;
    let edges: i64 =
        conn.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
    let chunks_fts_docs: i64 =
        conn.query_row("SELECT COUNT(*) FROM chunks_fts", [], |r| r.get(0))?;

    let mut stmt = conn.prepare(
        "SELECT mime_type, COUNT(*) AS n FROM files GROUP BY mime_type ORDER BY n DESC",
    )?;
    let mime_counts: Vec<(String, i64)> = stmt
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(DbStats { files, chunks, chunks_embedded, edges, chunks_fts_docs, mime_counts })
}
```

### 2. `src/main.rs` — updated stats display

Replace the current five-line block in `run_search`:

```rust
if stats {
    match searcher.stats().await {
        Ok(s) => {
            let pending = s.db_stats.chunks - s.db_stats.chunks_embedded;
            let pct = if s.db_stats.chunks > 0 {
                100 * s.db_stats.chunks_embedded / s.db_stats.chunks
            } else {
                100
            };

            println!("Index Statistics");

            // MIME breakdown
            println!("\n  Files");
            if s.db_stats.mime_counts.is_empty() {
                println!("    (none)");
            } else {
                let max_count = s.db_stats.mime_counts[0].1;
                let width = fmt_num(max_count).len();
                for (mime, count) in &s.db_stats.mime_counts {
                    println!("    {:>width$}  {}", fmt_num(*count), mime, width = width);
                }
            }

            // Embedding coverage
            println!("\n  Chunks");
            println!("    {:>10}  total", fmt_num(s.db_stats.chunks));
            println!(
                "    {:>10}  embedded  ({}%)",
                fmt_num(s.db_stats.chunks_embedded), pct
            );
            if pending > 0 {
                println!("    {:>10}  pending", fmt_num(pending));
            }

            // Remaining counts
            println!();
            println!("  {:>10}  edges",    fmt_num(s.db_stats.edges));
            println!("  {:>10}  FTS docs", fmt_num(s.db_stats.chunks_fts_docs));
            println!("  {:>10}  vectors",  fmt_num(s.vec_total as i64));
        }
        Err(e) => eprintln!("Stats error: {}", e),
    }
    return;
}
```

### 3. `src/search.rs` — `IndexStats` update

`IndexStats` in `search.rs` wraps `DbStats`. Propagate the new fields:

```rust
pub struct IndexStats {
    pub db_stats:  DbStats,   // carry the full DbStats instead of flattened fields
    pub vec_total: usize,
}
```

Update `Searcher::stats()`:

```rust
pub async fn stats(&self) -> Result<IndexStats, IndexerError> {
    let db_stats  = self.db.stats()?;
    let vec_total = self.vectors.ntotal().await?;
    Ok(IndexStats { db_stats, vec_total })
}
```

And update the reference sites in `main.rs` to use `s.db_stats.files`, etc.

---

## Bonus: Search Result Paths Block (already implemented)

After the results table in `print_table`, a **Paths** section shows the full
absolute path of each unique result file in rank order. This lets the user
copy-paste or click the path directly to open the file or its folder.

```
   1    82%  T1   docs/auth.md              JWT middleware validates token…
   2    71%  T2   src/middleware.rs         pub fn validate_jwt(token: &str…

  Paths
    1  C:\Users\edb\Documents\Work\docs\auth.md
    2  C:\Users\edb\Documents\Work\src\middleware.rs
```

Implementation notes:
- Deduplicates by `file_uri` — if multiple chunks from the same file appear, the
  file is listed once at its first (best) rank.
- Uses `file_uri` directly from `ChunkRow`, which is the full Windows path stored
  by `LocalStorageClient`.
- On Windows, most terminals (Windows Terminal, VS Code) make absolute paths
  clickable. Running `explorer /select,"<path>"` from the terminal opens Explorer
  with the file highlighted.

---

## Tests

Add to `db.rs` tests:

```rust
#[test]
fn stats_mime_counts_and_coverage() {
    let (db, _dir) = open_test_db();
    let fid1 = db.upsert_file("file:///a.md",  "text/markdown", "h1", 10, 1.0, None).unwrap();
    let fid2 = db.upsert_file("file:///b.md",  "text/markdown", "h2", 10, 1.0, None).unwrap();
    let fid3 = db.upsert_file("file:///c.txt", "text/plain",    "h3", 10, 1.0, None).unwrap();

    // Insert one embedded and one pending chunk
    let ids = db.insert_chunks(fid1, &[ChunkInput { tier: 1, chunk_index: 0,
        content: "a".into(), token_count: None, meta: serde_json::json!({}) }]).unwrap();
    db.set_chunk_lance_ids(&std::collections::HashMap::from([(ids[0], 999i64)])).unwrap();
    db.insert_chunks(fid2, &[ChunkInput { tier: 1, chunk_index: 0,
        content: "b".into(), token_count: None, meta: serde_json::json!({}) }]).unwrap();

    let s = db.stats().unwrap();
    assert_eq!(s.files, 3);
    assert_eq!(s.chunks_embedded, 1);
    assert_eq!(s.chunks - s.chunks_embedded, 1);   // 1 pending

    // markdown should appear first (2 files vs 1 for text/plain)
    assert_eq!(s.mime_counts[0].0, "text/markdown");
    assert_eq!(s.mime_counts[0].1, 2);
    assert_eq!(s.mime_counts[1].0, "text/plain");
}
```
