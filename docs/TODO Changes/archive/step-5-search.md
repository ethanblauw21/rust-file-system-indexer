# COMPLETED at 5/13/26

# Step 5 — Hybrid Search (`search.rs`) ⏳

**Status:** Pending

## Goal
Implement Reciprocal Rank Fusion (RRF) combining LanceDB semantic scores with SQLite FTS5 BM25 sparse scores. The SQLite `id` must map cleanly to the LanceDB `chunk_id`.

## Python Reference
`C:\Users\edb\Documents\indexer\fileSystem\search.py` — `hybrid_search`, `dense_search`, `sparse_search`, `_rrf_fuse`.

## Configuration
```rust
pub const RRF_K: f64 = 60.0;  // smoothing constant; higher = less rank-sensitive
```

## Output Types

```rust
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk_id:     i64,
    pub chunk:        ChunkRow,         // from Step 2 DB layer
    pub rrf_score:    Option<f64>,
    pub dense_score:  Option<f32>,
    pub sparse_score: Option<f64>,      // BM25 (negative; lower = better match)
    pub dense_rank:   Option<usize>,
    pub sparse_rank:  Option<usize>,
}
```

## Search Modes

### Dense search (`dense_search`)
1. Embed the query: `embed(&[query], "search_query: ")` → `Array1<f32>`
2. `vectors.search(query_vec, top_k * 2)` → `Vec<(lance_id, score)>`
3. Batch-fetch chunks from DB: `db.get_chunks_by_lance_ids(&lance_ids)`
4. Optional tier filter: applied post-fetch (or pre-filter via LanceDB `where` clause on `tier` column)
5. Return `Vec<SearchResult>` sorted by `dense_score` descending

### Sparse search (`sparse_search`)
1. `db.fts_search(query, limit)` → `Vec<FtsResult>`
2. Optional tier filter: `fts_results.retain(|r| r.tier == tier)`
3. Return `Vec<SearchResult>` sorted by `sparse_score` ascending (BM25 returns negatives; more negative = better)

### Hybrid search (`hybrid_search`) — RRF fusion
```rust
pub fn hybrid_search(
    query:    &str,
    top_k:    usize,
    tier:     Option<u8>,
    vectors:  &LanceStore,
    db:       &EnterpriseDb,
    embedder: &Embedder,
) -> Result<Vec<SearchResult>, IndexerError>
```

1. Run `dense_search` and `sparse_search` in parallel (`rayon::join` or two concurrent async tasks)
2. Call `rrf_fuse(dense, sparse, top_k)`
3. Return fused results sorted by `rrf_score` descending

### `rrf_fuse(dense, sparse, top_k) -> Vec<SearchResult>`
Exact port of Python `_rrf_fuse`:
```
for rank, result in enumerate(dense, start=1):
    scores[chunk_id] += 1.0 / (RRF_K + rank)

for rank, result in enumerate(sparse, start=1):
    scores[chunk_id] += 1.0 / (RRF_K + rank)

return top_k results sorted by score descending
```
Use `HashMap<i64, f64>` for score accumulation. When the same `chunk_id` appears in both channels, merge the `SearchResult` fields (keep both `dense_score` and `sparse_score`).

## `Searcher` struct
Owns all resources needed for a search session:

```rust
pub struct Searcher {
    pub db:       Arc<EnterpriseDb>,
    pub vectors:  Arc<LanceStore>,
    pub embedder: Arc<Embedder>,
}

impl Searcher {
    pub fn new(index_dir: &Path) -> Result<Self, IndexerError>
    pub fn search(&self, query: &str, opts: SearchOptions) -> Result<Vec<SearchResult>, IndexerError>
    pub fn stats(&self) -> Result<IndexStats, IndexerError>
}

pub struct SearchOptions {
    pub mode:  SearchMode,   // enum: Hybrid | Dense | Sparse
    pub top_k: usize,
    pub tier:  Option<u8>,
}
```

## CLI Binary (`main.rs` update)
Expose the same flags as `search.py`:

```
file_indexer search [OPTIONS] [QUERY]

Options:
  --mode    hybrid|dense|sparse  (default: hybrid)
  --top-k   N                    (default: 10)
  --tier    1|2|3
  --full                         print full chunk content
  --stats                        print index stats and exit
  --index-dir PATH               (default: .fileSystem-index)
```

Use `clap` (derive API) for argument parsing. Print results in the same tabular format as `search.py`:
```
  #  Score     T   Source                         Preview
  ─────────────────────────────────────────────────────────
  1  0.0312    T1  main.rs                        fn main() { ...
```

## ID Mapping Invariant
The `lance_id` stored in `chunks.lance_id` (SQLite) **must equal** the `id` column stored in the LanceDB table for the same chunk. This invariant is established in Step 4 (`_stable_id`) and consumed here. Both search paths must use the same `chunk_id` key (SQLite `chunks.id`) for RRF accumulation — not `lance_id` — so BM25 and ANN results for the same chunk are correctly merged.

## Key Rust Idioms to Apply
- `rayon::join(|| dense_search(...), || sparse_search(...))` for parallel dual-channel retrieval
- `HashMap::entry(k).and_modify(|v| *v += delta).or_insert(delta)` for score accumulation
- `clap` derive API for the CLI — no manual `ArgMatches` parsing
- `termcolor` or `crossterm` for the result table (or plain `println!` with fixed-width format strings)

## Tests to Write
- `rrf_fuse_combines_both_channels` — item appearing in both dense and sparse gets higher score than either alone
- `rrf_fuse_deduplicates` — same `chunk_id` in both channels appears once in output
- `rrf_k_smoothing` — rank-1 item scores `1/(60+1)` ≈ 0.0164
- `hybrid_search_returns_top_k` — result count ≤ `top_k`
- `dense_only_mode_no_sparse` — `SearchMode::Dense` populates `dense_score`, leaves `sparse_score` as `None`
- `sparse_only_mode_no_dense` — inverse of above
- `tier_filter_excludes_other_tiers` — `tier=Some(1)` returns no Tier 2 or 3 results
- `id_mapping_invariant` — for every result, `chunk.lance_id == lance_search_id`
