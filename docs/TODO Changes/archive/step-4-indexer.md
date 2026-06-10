# COMPLETED at 5/13/26

# Step 4 — Embedding & Vector DB (`indexer.rs`) ⏳

**Status:** Pending

## Goal
Implement the ONNX runtime for `nomic-embed-text-v1.5`, set up the LanceDB table to store vectors, and implement the Reservoir Sampling logic for training the IVF-PQ index.

## Python Reference
`C:\Users\edb\Documents\indexer\fileSystem\incremental_indexer.py` — `IncrementalIndexer` class.

## Configuration Constants
```rust
pub const HF_MODEL_ID:       &str = "nomic-ai/nomic-embed-text-v1.5";
pub const EMBEDDING_DIM:     usize = 768;
pub const EMBED_BATCH_SIZE:  usize = 256;
pub const NLIST:             usize = 32;   // IVF cluster count
pub const N_PROBE:           usize = 4;    // cells searched at query time
pub const TRAIN_MIN:         usize = NLIST;
pub const TRAIN_IDEAL:       usize = 39 * NLIST;
```

## ONNX Embedding (`ort` + `tokenizers`)

### Model loading
- Load `nomic-embed-text-v1.5.onnx` from a path configured at startup (env var `NOMIC_ONNX_PATH` or CLI flag)
- Use `ort::Session::builder()` with `with_optimization_level(GraphOptimizationLevel::Level3)`
- Cache the session in a `OnceLock<Session>` (single global instance, thread-safe)
- Use the `tokenizers` crate to load the matching `tokenizer.json` — same `OnceLock` pattern

### `embed(texts, prefix) -> Result<Array2<f32>, IndexerError>`
Replaces Python `_embed()`:
1. Prepend `prefix` to each text (`"search_document: "` for indexing, `"search_query: "` for queries)
2. Tokenize via `tokenizers::Tokenizer::encode_batch` — produces `input_ids`, `attention_mask`, `token_type_ids`
3. Convert to `ndarray::Array2<i64>` inputs (shape `[batch, seq_len]`)
4. Run ONNX session: `session.run(inputs)?`
5. Extract the `last_hidden_state` output tensor, apply mean-pooling over the sequence dimension
6. L2-normalise each row: `v /= max(norm(v), 1e-10)`
7. Return `Array2<f32>` of shape `(n, EMBEDDING_DIM)`, C-contiguous

### Batching
- Sort texts by length before batching (length-homogeneous batches avoid padding overhead — same optimisation as Python's `sorted(zip(texts, ids), key=lambda p: len(p[0]))`)
- Process `EMBED_BATCH_SIZE` texts per `session.run` call
- Use `rayon::iter` to parallelise tokenisation across CPU cores before the ONNX call

### Matryoshka truncation
If `dim < EMBEDDING_DIM`: slice `arr.slice(s![.., ..dim])` then re-normalise.

## LanceDB Vector Table

### Schema (Apache Arrow)
```
id:         Int64    (stable hash ID, same derivation as Python _stable_id)
chunk_id:   Int64    (SQLite rowid from Step 2)
tier:       Int8
vector:     FixedSizeList<Float32>[768]
```

### `_stable_id(file_uri, tier, chunk_index) -> i64`
60-bit deterministic ID — SHA-256 of `"{file_uri}:{tier}:{chunk_index}"`, take first 15 hex chars, mask with `0x0FFF_FFFF_FFFF_FFFF`. Must be identical to the Python implementation.

### `LanceStore` struct
```rust
pub struct LanceStore {
    db:    lancedb::Connection,
    table: lancedb::Table,
}
```

| Method | Description |
|---|---|
| `open_or_create(path) -> Result<Self>` | Open existing or create new LanceDB table |
| `add_vectors(ids, chunk_ids, tiers, vectors) -> Result<()>` | Batch insert via Arrow RecordBatch |
| `remove_ids(ids: &[i64]) -> Result<()>` | Delete by lance ID (replaces FAISS `IDSelectorBatch`) |
| `search(query_vec, top_k) -> Result<Vec<LanceResult>>` | ANN search, returns `(id, score)` pairs |
| `create_ivf_pq_index(nlist, nprobe) -> Result<()>` | Train IVF-PQ index on existing vectors |
| `ntotal() -> Result<usize>` | Vector count |

### IVF-PQ index creation
LanceDB handles training internally via `table.create_index(IvfPq::new().num_partitions(NLIST))`. No manual training loop needed — this replaces the entire `_maybe_upgrade_to_ivfpq` / `_flush_pending` / reservoir-sampling complexity from Python.

## Reservoir Sampling
Required only for the pre-training text buffer (to generate training embeddings for the IVF index if LanceDB's auto-training is insufficient). Implement Algorithm R:

```rust
fn update_reservoir(reservoir: &mut Vec<String>, new_texts: &[String], total_seen: usize) {
    for text in new_texts {
        let n = reservoir.len();
        if n < TRAIN_IDEAL {
            reservoir.push(text.clone());
        } else {
            let j = rand::thread_rng().gen_range(0..total_seen);
            if j < TRAIN_IDEAL {
                reservoir[j] = text.clone();
            }
        }
    }
}
```

## `IncrementalIndexer` struct

```rust
pub struct IncrementalIndexer {
    storage:  Arc<dyn StorageClient>,
    chunker:  FileChunker,
    db:       Arc<EnterpriseDb>,
    vectors:  LanceStore,
    index_dir: PathBuf,
}
```

### `index_root(root_uri) -> Result<Stats>`
1. `storage.list_files(root_uri)` → collect all URIs
2. `db.get_all_file_info()` → pre-fetch change-detection data (no DB access in threads)
3. **Parallel I/O phase** via `rayon::iter::par_bridge()`:
   - `get_metadata` → mtime pre-filter
   - `get_file_bytes` → MD5 hash check
   - `chunker.chunk(bytes, meta)` → `ChunkResult`
   - Returns `Option<FileWork>` (None = unchanged)
4. **Serial write phase** (main thread):
   - `db.upsert_file(...)` → `file_id`
   - `db.insert_chunks(file_id, chunks)` → `chunk_ids`
   - Buffer `(texts, lance_ids)` for batch embedding
   - `db.insert_edges(...)`
5. **Embedding phase** — flush buffer in `EMBED_BATCH_SIZE` batches:
   - `embed(batch_texts, "search_document: ")` → `Array2<f32>`
   - `vectors.add_vectors(lance_ids, chunk_ids, tiers, vecs)`
6. **IVF-PQ index creation** — call `vectors.create_ivf_pq_index(NLIST, N_PROBE)` if `ntotal >= TRAIN_MIN`
7. **Stale file removal** — delete from DB + LanceDB any URI no longer in `live_uris`

### `reindex_file(file_uri) -> Result<()>`
Force re-index of a single file: remove → read → chunk → embed → persist.

### Change detection (3-stage, same as Python)
1. mtime pre-filter (no I/O beyond stat)
2. MD5 hash confirmation (only if mtime changed)
3. Full re-index (only if hash differs or not in LanceDB)

## `_stable_id` must be byte-for-byte identical to Python
Write a test that computes a known URI's ID in both representations and asserts equality.

## Key Rust Idioms to Apply
- `Arc<dyn StorageClient>` for shared ownership across the parallel phase
- `rayon::iter::par_bridge()` to drive the I/O phase in parallel — chunk tasks are CPU-bound after the `get_file_bytes` await, so rayon is appropriate
- `ndarray::Array2::from_shape_vec` to construct embedding matrices without unsafe code
- `arrow_array::RecordBatch` for LanceDB inserts — build with `arrow_array::builder` types, not `serde_json`

## Tests to Write
- `stable_id_matches_python_reference` — test vector against known Python output
- `embed_returns_unit_vectors` — L2 norm of each output row ≈ 1.0 (tolerance 1e-5)
- `embed_batch_matches_single` — embedding 4 texts in one batch == 4 individual embeds
- `lancedb_add_and_search` — insert 10 vectors, search returns correct top-1
- `lancedb_remove_ids` — remove a vector, confirm it no longer appears in search
- `reservoir_uniform_distribution` — insert 10× `TRAIN_IDEAL` texts, verify reservoir size == `TRAIN_IDEAL`
- `index_root_skips_unchanged_files` — index once, re-index same root, second run has 0 new chunks
