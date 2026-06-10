# Step 9 — Self-Healing Indexer

**Status:** Complete

## Goal

After indexing, run a separate quality-scoring pass over all chunks. Flag
low-quality chunks using structural heuristics and (when an embedder is
available) embedding coherence. Store the chunker method version used per chunk
so a `recheck` subcommand can detect when the file-type→chunker mapping has
changed and automatically reindex only the affected flagged files.

Surface flagged chunk counts in the TUI on a new `Flagged` screen.

---

## Design decisions

| Decision | Choice |
|---|---|
| Metrics | Structural heuristics + embedding coherence (no LLM) |
| Scoring timing | Separate `score` pass — never inline with indexing |
| Score storage | `structural_score REAL`, `coherence_score REAL`, `is_flagged INTEGER` added to `chunks` |
| Chunker identity | Version-tagged strings per MIME group: `"xlsx/v1"`, `"markdown/v1"`, etc. |
| Mapping config | `<index_dir>/chunker_map.toml` — editable without recompile |
| Polling service | `recheck` CLI subcommand (Phase 1); long-running daemon is the intended future |
| TUI display | Summary table by MIME type (Phase 1); drill-down (explain-style) is the next iteration |

---

## Phase 1 — Foundation

### 1. DB schema migration (`src/db.rs`)

Add four columns to `chunks` at startup via `ALTER TABLE ... ADD COLUMN`:

```sql
ALTER TABLE chunks ADD COLUMN structural_score REAL;
ALTER TABLE chunks ADD COLUMN coherence_score  REAL;
ALTER TABLE chunks ADD COLUMN is_flagged       INTEGER NOT NULL DEFAULT 0;
ALTER TABLE chunks ADD COLUMN chunker_method   TEXT;
```

Use `IF NOT EXISTS` equivalents (check `PRAGMA table_info`) so existing indexes
migrate without errors.

Add query helpers:

- `fn set_chunk_scores(conn, chunk_id, structural, coherence, is_flagged)`
- `fn get_flagged_summary(conn) -> Vec<(mime, method, flagged_count, total_count)>`
- `fn get_unscored_chunks(conn) -> Vec<ChunkRow>` — returns chunks where
  `structural_score IS NULL`

### 2. Chunker method config (`src/chunker_map.rs` — new file)

`ChunkerMap` reads and writes `<index_dir>/chunker_map.toml`. On first run it
generates the file from the known MIME set:

```toml
[methods]
"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" = "xlsx/v1"
"text/markdown"   = "markdown/v1"
"text/plain"      = "plaintext/v1"
"application/pdf" = "pdf/v1"
"application/vnd.openxmlformats-officedocument.wordprocessingml.document" = "docx/v1"
"text/csv"        = "csv/v1"
# All other types fall back to:
"_default"        = "generic/v1"
```

Public API:

```rust
impl ChunkerMap {
    pub fn load_or_create(index_dir: &Path) -> Result<Self>;
    pub fn method_for(&self, mime: &str) -> &str;   // falls back to _default
}
```

To change a chunker: edit the version string in the TOML (e.g. `"xlsx/v1"` →
`"xlsx/v2"`). The `recheck` subcommand detects the drift automatically.

### 3. Annotate chunks with their method (`src/chunker.rs`, `src/indexer.rs`)

`FileChunker::chunk()` gains a `map` parameter and returns the method string
alongside the result:

```rust
pub fn chunk(
    &self,
    data: &[u8],
    meta: &FileMetadata,
    map: &ChunkerMap,
) -> Result<(ChunkResult, String), IndexerError>
```

`IncrementalIndexer` passes a `ChunkerMap` (loaded once at startup) and writes
`chunker_method` to SQLite for each chunk at DB write time.

---

## Phase 2 — Scorer (`src/scorer.rs` — new file)

### 4. Structural heuristics — `fn structural_score(chunk: &Chunk) -> f32`

Five checks, each contributing to a weighted 0.0–1.0 score:

| Check | What it catches | Weight |
|---|---|---|
| Token count range | < 15 tokens or ≥ 99% of tier token limit | 0.25 |
| Sentence completeness | Doesn't end with `.` `?` `!` `:` `"` `` ` `` | 0.25 |
| Bracket balance | Unmatched `(` `[` `{` `<` left open | 0.20 |
| Whitespace density | > 40% of content is whitespace / newlines | 0.15 |
| Repetition ratio | Unique token count / total token count < 0.3 | 0.15 |

Flagging threshold: `structural_score < 0.5` triggers `is_flagged = 1`.

### 5. Embedding coherence — `async fn coherence_score(...) -> Option<f32>`

```rust
async fn coherence_score(
    chunk_lance_id: &str,
    file_uri: &str,
    tier: u8,
    lance: &LanceStore,
) -> Option<f32>
```

- Fetch this chunk's embedding vector from LanceDB by `lance_id`
- Fetch sibling vectors of the next-higher tier (same `file_uri`, `tier + 1`)
- Return cosine similarity of this chunk vs mean of the parent-tier vectors
- Returns `None` when no embedder is available, `lance_id` is NULL, or no
  parent tier exists (Tier 3 chunks always return `None`)

Flagging threshold: `coherence_score < 0.6` triggers `is_flagged = 1`.

> Both thresholds are `const` values to start. Making them configurable via
> `chunker_map.toml` is a natural follow-up.

### 6. Score runner — `pub async fn score_all(db, lance, rescore) -> Result<ScoreStats>`

- Queries chunks where `structural_score IS NULL` (or all chunks if `rescore = true`)
- Runs structural score on every chunk (always)
- Runs coherence score on chunks that have a `lance_id` (requires embedder)
- Sets `is_flagged = 1` when either threshold is crossed
- Bulk-updates `structural_score`, `coherence_score`, `is_flagged` in SQLite
- Returns `ScoreStats { total, flagged, structural_only, coherence_only, both }`

---

## Phase 3 — CLI subcommands (`main.rs`)

### 7. `score` subcommand

```
cargo run -- score [--index-dir .fileSystem-index] [--rescore]
```

Calls `score_all()` and prints:

```
Scored 1 247 chunks — 38 flagged (31 structural, 5 coherence, 2 both)
```

`--rescore` forces rescoring of already-scored chunks (useful after changing
thresholds).

### 8. `recheck` subcommand

```
cargo run -- recheck [--index-dir .fileSystem-index] [--dry-run]
```

Logic:

1. Load current `chunker_map.toml`
2. Query `SELECT DISTINCT file_uri, mime_type, chunker_method FROM chunks WHERE is_flagged = 1`
3. For each `(file_uri, mime_type, stored_method)`: compare `stored_method` to
   `map.method_for(mime_type)`
4. Collect files where versions differ — these are candidates for reindex
5. Reindex each candidate file through the full `IncrementalIndexer` pipeline
   (forces re-chunk with the current method)
6. After reindex: call `score_all` on the newly indexed chunks
7. Print report:

```
Reindexed 4 files (method drift detected)
  xlsx/v1 → xlsx/v2 : 3 files
  markdown/v1 → markdown/v2 : 1 file
Skipped 34 files (flagged but method unchanged — manual review needed)
```

`--dry-run` prints the report without reindexing anything.

> **Future — daemon mode:** `recheck --watch [--interval 300]` runs the above
> on a loop. Intended as Phase 2 of this feature once the subcommand is stable.

---

## Phase 4 — TUI additions (`src/tui.rs`)

### 9. `/score` command (search bar)

Follows the same `/mode` pattern. When the user types `/score` and presses
Enter in the search bar:

- Launches `score_all()` in a `tokio::spawn` background task
- Status bar shows live progress: `Scoring… 312 / 1 247 chunks`
- On completion: `Scored 1 247 chunks — 38 flagged`

`/score --rescore` forces a full rescore.

### 10. `Screen::Flagged` — summary screen

Accessed via `f` key from results focus (`search_focused = false`). `Esc`
returns to search.

```
┌─ Flagged Chunks ────────────────────────────────────────────────────────────┐
│                                                                              │
│  MIME type                        Method         Flagged   Total      %     │
│  ────────────────────────────     ────────────   ───────   ─────   ────     │
│  text/markdown                    markdown/v1        12      87    14%      │
│  application/vnd...xlsx           xlsx/v1             4      32    12%      │
│  text/plain                       plaintext/v1        1      44     2%      │
│                                                                              │
│  Esc: close  ·  q: quit                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

MIME strings are truncated with middle ellipsis (same `truncate_mid` helper
already used in the stats overlay).

> **Future:** pressing Enter on a row drills into individual flagged chunks for
> that MIME type — explain-style workflow with per-chunk scores and file paths.

---

## Files changed

| File | Change |
|---|---|
| `src/db.rs` | Schema migration, `set_chunk_scores`, `get_flagged_summary`, `get_unscored_chunks` |
| `src/chunker.rs` | `chunk()` gains `map` param, returns `(ChunkResult, String)` |
| `src/indexer.rs` | Load `ChunkerMap` at startup, write `chunker_method` per chunk |
| `src/tui.rs` | `Screen::Flagged`, `/score` command, `f` key binding |
| `main.rs` | `score` and `recheck` subcommands wired to `clap` |
| `src/scorer.rs` | **New** — `structural_score`, `coherence_score`, `score_all` |
| `src/chunker_map.rs` | **New** — `ChunkerMap`, TOML read/write, default generation |

---

## Suggested implementation order

1. DB migration + query helpers (Phase 1, step 1)
2. `ChunkerMap` + TOML config (Phase 1, step 2)
3. Annotate chunks in indexer (Phase 1, step 3)
4. Structural scorer (Phase 2, step 4)
5. `score` CLI subcommand (Phase 3, step 7) — validate scorer end-to-end
6. Embedding coherence scorer (Phase 2, step 5–6)
7. `recheck` subcommand (Phase 3, step 8)
8. TUI `/score` command + `Screen::Flagged` (Phase 4)
