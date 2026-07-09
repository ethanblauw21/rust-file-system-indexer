use crate::chunker::{ChunkResult, EdgeType, FileChunker};
use crate::chunker_map::ChunkerMap;
use crate::db::{ChunkInput, EdgeInput, EnterpriseDb, FileInfo};
use crate::error::IndexerError;
use crate::storage::StorageClient;
use arrow_array::{
    builder::{FixedSizeListBuilder, Float32Builder, Int64Builder, Int8Builder},
    ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch,
    RecordBatchIterator, RecordBatchReader,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use lancedb::query::{ExecutableQuery, QueryBase};
use ndarray::Array2;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Constants ─────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub const HF_MODEL_ID:      &str   = "nomic-ai/nomic-embed-text-v1.5";
/// Matryoshka slice dimension: first 256 of 768 dims (~1.5% accuracy drop, 66% storage reduction).
/// Changing this constant requires re-indexing: on a stored-vs-expected dimension mismatch
/// `LanceStore::open_or_create` hard-errors naming both dims unless `--reindex`
/// (`force_recreate`) is passed to authorize the destructive table rebuild.
pub const EMBEDDING_DIM:    usize  = 256;
/// Embed batch size tuned for CPU inference.
/// 256 was GPU-sized; at batch=256 × seq=512 the CPU takes 40–120 s per call.
/// 32 keeps each call under ~5 s while amortising tokenizer overhead.
pub const EMBED_BATCH_SIZE: usize  = 32;
pub const TRAIN_MIN:        usize  = 32;
#[allow(dead_code)]
pub const TRAIN_IDEAL:      usize  = 39 * TRAIN_MIN;
const MIN_NLIST:            usize  = 4;
/// Bound on how many ids are concatenated into a single `id IN (...)` filter
/// string passed to LanceDB (`delete` / `only_if`). LanceDB filters are plain
/// DataFusion SQL-expression strings, not bound parameters, so there is no
/// hard protocol limit — but building one unbounded string for tens of
/// thousands of ids risks pathological string-alloc cost and slow filter
/// parsing. 1000 mirrors SQLite's own default `SQLITE_MAX_VARIABLE_NUMBER`
/// (999) that already bounds `IN (...)` batches on the FTS5 side of this
/// codebase, keeping id-batch sizing consistent across both stores.
const MAX_IDS_PER_FILTER:   usize  = 1000;

/// IVF-PQ partition count heuristic: 4√N clamped to [MIN_NLIST, 1024].
pub fn compute_nlist(n: usize) -> usize {
    ((n as f64).sqrt() as usize * 4).clamp(MIN_NLIST, 1024)
}

// ── Stable ID (byte-for-byte identical to Python _stable_id) ─────────────────

/// 60-bit deterministic ID: SHA-256 of `"{file_uri}:{tier}:{chunk_index}"`,
/// first 15 hex chars interpreted as base-16, masked to 60 bits.
pub fn stable_id(file_uri: &str, tier: u8, chunk_index: usize) -> i64 {
    use sha2::{Digest, Sha256};
    let input = format!("{}:{}:{}", file_uri, tier, chunk_index);
    let hash  = Sha256::digest(input.as_bytes());
    let hex   = hex::encode(hash);
    let raw   = i64::from_str_radix(&hex[..15], 16).unwrap_or(0);
    raw & 0x0FFF_FFFF_FFFF_FFFFi64
}

// ── LanceDB schema ────────────────────────────────────────────────────────────

fn chunks_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id",       DataType::Int64, false),
        Field::new("chunk_id", DataType::Int64, false),
        Field::new("tier",     DataType::Int8,  false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIM as i32,
            ),
            false,
        ),
    ]))
}

fn build_record_batch(
    ids:       &[i64],
    chunk_ids: &[i64],
    tiers:     &[u8],
    vectors:   &[Vec<f32>],
) -> Result<RecordBatch, IndexerError> {
    let schema = chunks_schema();
    let n = ids.len();

    let id_col: ArrayRef = {
        let mut b = Int64Builder::with_capacity(n);
        for &v in ids { b.append_value(v); }
        Arc::new(b.finish())
    };
    let chunk_id_col: ArrayRef = {
        let mut b = Int64Builder::with_capacity(n);
        for &v in chunk_ids { b.append_value(v); }
        Arc::new(b.finish())
    };
    let tier_col: ArrayRef = {
        let mut b = Int8Builder::with_capacity(n);
        for &v in tiers { b.append_value(v as i8); }
        Arc::new(b.finish())
    };
    let vector_col: ArrayRef = {
        let mut b = FixedSizeListBuilder::new(Float32Builder::new(), EMBEDDING_DIM as i32);
        for vec in vectors {
            for &v in vec {
                b.values().append_value(v);
            }
            b.append(true);
        }
        Arc::new(b.finish())
    };

    RecordBatch::try_new(schema, vec![id_col, chunk_id_col, tier_col, vector_col])
        .map_err(|e| IndexerError::VectorStore(e.to_string()))
}

// ── LanceStore ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LanceResult {
    pub id:    i64,
    pub score: f32,
}

pub struct LanceStore {
    table: lancedb::Table,
}

async fn create_chunks_table(conn: &lancedb::Connection) -> Result<lancedb::Table, IndexerError> {
    let schema      = chunks_schema();
    let empty_batch = RecordBatch::new_empty(schema.clone());
    let reader: Box<dyn RecordBatchReader + Send> = Box::new(
        RecordBatchIterator::new(vec![Ok(empty_batch)].into_iter(), schema),
    );
    conn.create_table("chunks", reader)
        .execute()
        .await
        .map_err(|e| IndexerError::VectorStore(e.to_string()))
}

impl LanceStore {
    /// `force_recreate` permits the destructive drop-and-rebuild path on a vector
    /// dimension mismatch (see below) — only `index --reindex` should pass `true`;
    /// every other caller (search, ingest, recheck) must pass `false` so a stale
    /// or misconfigured embedder can never silently wipe an existing vector table.
    pub async fn open_or_create(path: &Path, force_recreate: bool) -> Result<Self, IndexerError> {
        std::fs::create_dir_all(path).map_err(|e| IndexerError::Io {
            path:   path.to_path_buf(),
            source: e,
        })?;
        let uri  = path.to_string_lossy();
        let conn = lancedb::connect(uri.as_ref())
            .execute()
            .await
            .map_err(|e| IndexerError::VectorStore(e.to_string()))?;

        let table = if let Ok(t) = conn.open_table("chunks").execute().await {
            // Verify the stored vector dimension matches EMBEDDING_DIM; recreate if not.
            let schema = t.schema().await.map_err(|e| IndexerError::VectorStore(e.to_string()))?;
            let stored_dim = schema.field_with_name("vector").ok().and_then(|f| {
                if let DataType::FixedSizeList(_, n) = f.data_type() { Some(*n) } else { None }
            });
            if stored_dim == Some(EMBEDDING_DIM as i32) {
                t
            } else if force_recreate {
                tracing::warn!(
                    "LanceDB vector dim mismatch (stored={:?}, expected={}) — dropping and recreating (--reindex)",
                    stored_dim, EMBEDDING_DIM
                );
                conn.drop_table("chunks", &[])
                    .await
                    .map_err(|e| IndexerError::VectorStore(e.to_string()))?;
                create_chunks_table(&conn).await?
            } else {
                return Err(IndexerError::VectorStore(format!(
                    "LanceDB vector table dimension mismatch: stored table has vectors of dim={:?}, but this build \
                     expects dim={} (EMBEDDING_DIM in src/indexer.rs). The embedding model or EMBEDDING_DIM changed \
                     since this index was built; querying or adding vectors against the mismatched table would corrupt \
                     or silently degrade dense search. This requires a one-time full reindex: rerun `index <root> \
                     --reindex` to rebuild the vector table (this also re-chunks and re-embeds every file — existing \
                     chunks/vectors are NOT preserved across --reindex), or point --index-dir at a fresh directory. \
                     Refusing to auto-drop the existing vector table.",
                    stored_dim, EMBEDDING_DIM
                )));
            }
        } else {
            create_chunks_table(&conn).await?
        };

        Ok(Self { table })
    }

    pub async fn add_vectors(
        &self,
        ids:       &[i64],
        chunk_ids: &[i64],
        tiers:     &[u8],
        vectors:   &[Vec<f32>],
    ) -> Result<(), IndexerError> {
        if ids.is_empty() {
            return Ok(());
        }
        let schema = chunks_schema();
        let batch  = build_record_batch(ids, chunk_ids, tiers, vectors)?;
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema)
        );
        self.table
            .add(reader)
            .execute()
            .await
            .map_err(|e| IndexerError::VectorStore(e.to_string()))?;
        Ok(())
    }

    pub async fn remove_ids(&self, ids: &[i64]) -> Result<(), IndexerError> {
        if ids.is_empty() {
            return Ok(());
        }
        for batch in ids.chunks(MAX_IDS_PER_FILTER) {
            let id_list: String = batch.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ");
            self.table
                .delete(&format!("id IN ({})", id_list))
                .await
                .map_err(|e| IndexerError::VectorStore(e.to_string()))?;
        }
        Ok(())
    }

    pub async fn search(
        &self,
        query_vec: Vec<f32>,
        top_k:     usize,
    ) -> Result<Vec<LanceResult>, IndexerError> {
        use futures::TryStreamExt;

        let mut stream = self.table
            .query()
            .nearest_to(query_vec)
            .map_err(|e| IndexerError::VectorStore(e.to_string()))?
            .limit(top_k)
            .execute()
            .await
            .map_err(|e| IndexerError::VectorStore(e.to_string()))?;

        let mut results: Vec<LanceResult> = Vec::new();
        while let Some(batch) = stream
            .try_next()
            .await
            .map_err(|e| IndexerError::VectorStore(e.to_string()))?
        {
            let id_col = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| IndexerError::VectorStore("missing 'id' column".into()))?;
            let dist_col = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>())
                .ok_or_else(|| IndexerError::VectorStore("missing '_distance' column".into()))?;

            for i in 0..batch.num_rows() {
                results.push(LanceResult {
                    id:    id_col.value(i),
                    score: dist_col.value(i),
                });
            }
        }
        Ok(results)
    }

    /// Fetch embedding vectors for the given LanceDB IDs.
    /// Returns a map from id → vector; IDs not found in the store are absent.
    pub async fn fetch_vectors_by_ids(
        &self,
        ids: &[i64],
    ) -> Result<HashMap<i64, Vec<f32>>, IndexerError> {
        use futures::TryStreamExt;

        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut result: HashMap<i64, Vec<f32>> = HashMap::new();
        for id_batch in ids.chunks(MAX_IDS_PER_FILTER) {
            let id_list: String = id_batch.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ");
            let mut stream = self.table
                .query()
                .only_if(format!("id IN ({})", id_list))
                .execute()
                .await
                .map_err(|e| IndexerError::VectorStore(e.to_string()))?;

            while let Some(batch) = stream
                .try_next()
                .await
                .map_err(|e| IndexerError::VectorStore(e.to_string()))?
            {
                let id_col = batch
                    .column_by_name("id")
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .ok_or_else(|| IndexerError::VectorStore("missing 'id' column".into()))?;
                let vec_col = batch
                    .column_by_name("vector")
                    .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                    .ok_or_else(|| IndexerError::VectorStore("missing 'vector' column".into()))?;

                for i in 0..batch.num_rows() {
                    let id   = id_col.value(i);
                    let vals = vec_col.value(i);
                    let floats = vals
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| IndexerError::VectorStore("vector values not f32".into()))?;
                    let vec: Vec<f32> = (0..floats.len()).map(|j| floats.value(j)).collect();
                    result.insert(id, vec);
                }
            }
        }
        Ok(result)
    }

    pub async fn delete_all(&self) -> Result<(), IndexerError> {
        self.table
            .delete("id >= 0")
            .await
            .map(|_| ())
            .map_err(|e| IndexerError::VectorStore(e.to_string()))
    }

    pub async fn create_ivf_pq_index(&self, nlist: usize) -> Result<(), IndexerError> {
        use lancedb::index::{Index, vector::IvfPqIndexBuilder};
        self.table
            .create_index(
                &["vector"],
                Index::IvfPq(IvfPqIndexBuilder::default().num_partitions(nlist as u32)),
            )
            .execute()
            .await
            .map_err(|e| IndexerError::VectorStore(e.to_string()))
    }

    pub async fn ntotal(&self) -> Result<usize, IndexerError> {
        self.table
            .count_rows(None)
            .await
            .map_err(|e| IndexerError::VectorStore(e.to_string()))
    }
}

// ── Embedder (ONNX Runtime + tokenizers) ─────────────────────────────────────

/// Runs `f` on a dedicated thread and waits up to `timeout` for it to finish.
/// A version-mismatched ONNX Runtime DLL can wedge indefinitely inside its own
/// native `GetApi()` call instead of returning an error (see the dogfood doc's
/// Defect 1) — we can't kill a stuck native call, but we CAN stop waiting on it
/// and surface a loud timeout error to the caller instead of hanging the whole
/// `index`/`search` command. The spawned thread is intentionally leaked on
/// timeout; there is no safe way to cancel a blocked FFI call from Rust.
fn run_with_timeout<T, F>(timeout: std::time::Duration, f: F) -> Result<T, IndexerError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, IndexerError> + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(_) => Err(IndexerError::Embedding(format!(
            "ONNX Runtime did not finish loading within {:?}. This almost always means \
             ORT_DYLIB_PATH points at an onnxruntime build whose version doesn't match what this \
             binary was compiled against (api-24, i.e. onnxruntime 1.24.x — see \
             docs/dogfooding/2026-06-22-dogfood-run.md Defect 1). Point ORT_DYLIB_PATH at the \
             documented onnxruntime 1.24.2 build and retry.",
            timeout
        ))),
    }
}

// ── Profiling (`--profile`) ───────────────────────────────────────────────────

/// Cumulative per-stage timing for `index --profile`.
///
/// `index_root` overlaps its producer (file I/O + chunking, on a dedicated
/// rayon pool) with its consumer (SQLite writes + batched embedding + LanceDB
/// writes) via a bounded channel — see the `phase1`/`rx.recv()` loop. Because
/// the two sides run CONCURRENTLY, these fields are SUMMED durations-in-stage,
/// not a partition of wall-clock: they need not (and in practice won't) add up
/// to the overall wall-clock time. `render` states this explicitly so the
/// output can't be misread as a wall-clock breakdown.
///
/// Entirely optional: `IncrementalIndexer::profile` is `None` by default, and
/// every call site branches on `if let Some(prof) = &self.profile`, so a
/// normal (non-profiled) run pays no timing overhead.
#[derive(Default)]
pub struct Profile {
    chunking_ns:     AtomicU64,
    tokenize_ns:     AtomicU64,
    inference_ns:    AtomicU64,
    sqlite_ns:       AtomicU64,
    lance_ns:        AtomicU64,
    ivf_ns:          AtomicU64,
    chunks_embedded: AtomicU64,
}

impl Profile {
    pub fn new() -> Self {
        Self::default()
    }

    fn add_chunking(&self, d: Duration)  { self.chunking_ns.fetch_add(d.as_nanos() as u64, Ordering::Relaxed); }
    fn add_tokenize(&self, d: Duration)  { self.tokenize_ns.fetch_add(d.as_nanos() as u64, Ordering::Relaxed); }
    fn add_inference(&self, d: Duration) { self.inference_ns.fetch_add(d.as_nanos() as u64, Ordering::Relaxed); }
    fn add_sqlite(&self, d: Duration)    { self.sqlite_ns.fetch_add(d.as_nanos() as u64, Ordering::Relaxed); }
    fn add_lance(&self, d: Duration)     { self.lance_ns.fetch_add(d.as_nanos() as u64, Ordering::Relaxed); }
    fn add_ivf(&self, d: Duration)       { self.ivf_ns.fetch_add(d.as_nanos() as u64, Ordering::Relaxed); }
    fn add_chunks_embedded(&self, n: usize) { self.chunks_embedded.fetch_add(n as u64, Ordering::Relaxed); }

    /// Total chunks that went through `Embedder::embed_timed` this run.
    pub fn chunks_embedded(&self) -> u64 {
        self.chunks_embedded.load(Ordering::Relaxed)
    }

    /// Render the accumulated stage totals as a human-readable table.
    /// `wall` is the overall wall-clock time for the `index_root` call;
    /// `total_units` is caller-supplied (chunks embedded, or indexed-chunk
    /// count under `--no-embed`) and only used for the `chunks/sec` line.
    pub fn render(&self, wall: Duration, total_units: usize) -> String {
        let stages: [(&str, Duration); 6] = [
            ("chunking",     Duration::from_nanos(self.chunking_ns.load(Ordering::Relaxed))),
            ("tokenize",     Duration::from_nanos(self.tokenize_ns.load(Ordering::Relaxed))),
            ("inference",    Duration::from_nanos(self.inference_ns.load(Ordering::Relaxed))),
            ("sqlite_write", Duration::from_nanos(self.sqlite_ns.load(Ordering::Relaxed))),
            ("lance_write",  Duration::from_nanos(self.lance_ns.load(Ordering::Relaxed))),
            ("ivf_build",    Duration::from_nanos(self.ivf_ns.load(Ordering::Relaxed))),
        ];
        let summed: f64 = stages.iter().map(|(_, d)| d.as_secs_f64()).sum();

        let mut out = String::new();
        out.push_str("=== index --profile ===\n");
        out.push_str(
            "note: chunking (producer) runs CONCURRENTLY with tokenize/inference/sqlite_write/\n\
             lance_write (consumer) — index_root overlaps them via a bounded channel. Stage\n\
             totals below are cumulative time-in-stage (summed across all files/batches), NOT a\n\
             partition of wall-clock, and need not sum to it.\n\n",
        );
        for (name, d) in &stages {
            let pct = if summed > 0.0 { d.as_secs_f64() / summed * 100.0 } else { 0.0 };
            out.push_str(&format!(
                "  {name:<12} {seconds:>10.3}s  {pct:>6.1}% of summed stage time\n",
                name = name, seconds = d.as_secs_f64(), pct = pct
            ));
        }
        out.push_str(&format!(
            "  {name:<12} {seconds:>10.3}s  (sum of all stages above)\n",
            name = "summed", seconds = summed
        ));
        out.push('\n');
        out.push_str(&format!("  wall-clock     {:>10.3}s\n", wall.as_secs_f64()));
        out.push_str(&format!("  chunks         {total_units:>10}\n"));
        let per_sec = if wall.as_secs_f64() > 0.0 { total_units as f64 / wall.as_secs_f64() } else { 0.0 };
        out.push_str(&format!("  chunks/sec     {per_sec:>10.1}\n"));
        out
    }
}

pub struct Embedder {
    session:   Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
}

impl Embedder {
    /// Loads the ONNX Runtime session + tokenizer with a hard timeout. A
    /// version-mismatched ORT dylib doesn't return an error from `ort::init_from` —
    /// it can hang indefinitely inside the native `GetApi()` call (see dogfood doc
    /// Defect 1), which would otherwise wedge the whole `index`/`search` command
    /// with no diagnostic. `ort_dylib_path` overrides `ORT_DYLIB_PATH` when set.
    /// `model_file` names the ONNX file within `onnx_dir` (e.g.
    /// `nomic-embed-text-v1.5.onnx` or, for the CUDA EP, `model_fp16.onnx` —
    /// see ADR-006). The tokenizer filename is always `tokenizer.json`.
    pub fn load(
        onnx_dir: &Path,
        model_file: &str,
        ort_dylib_path: Option<&str>,
        load_timeout: std::time::Duration,
    ) -> Result<Self, IndexerError> {
        let onnx_dir = onnx_dir.to_path_buf();
        let model_file = model_file.to_string();
        let ort_dylib_path = ort_dylib_path.map(str::to_string);
        run_with_timeout(load_timeout, move || {
            Self::load_inner(&onnx_dir, &model_file, ort_dylib_path.as_deref())
        })
    }

    /// ADR-006 Phase 1: when built with `--features cuda`, this registers the
    /// ONNX Runtime CUDA execution provider on the `SessionBuilder` before the
    /// CPU-only fallback path would otherwise run. Registration happens INSIDE
    /// `load_inner`, which `load` already wraps in `run_with_timeout`, so a
    /// CUDA init hang (missing driver, mismatched CUDA/cuDNN) is bounded by the
    /// same load timeout as the existing ORT-DLL-version defect instead of
    /// wedging indefinitely.
    ///
    /// `.error_on_failure()` is deliberate: `ort`'s default is to *silently*
    /// fall back to CPU if EP registration fails. A `cuda`-featured build that
    /// secretly ran on CPU would corrupt the throughput measurement this ADR
    /// is gated on (see docs/adr/ADR-006-cuda-embedding-execution-provider.md),
    /// so a CUDA registration failure must surface as a named
    /// `IndexerError::Embedding`, never a quiet no-op.
    fn load_inner(onnx_dir: &Path, model_file: &str, ort_dylib_path: Option<&str>) -> Result<Self, IndexerError> {
        use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

        let model_path = onnx_dir.join(model_file);
        let tok_path   = onnx_dir.join("tokenizer.json");

        let ort_dll = ort_dylib_path.map(str::to_string)
            .or_else(|| std::env::var("ORT_DYLIB_PATH").ok())
            .unwrap_or_default();
        ort::init_from(&ort_dll)
            .map_err(|e| IndexerError::Embedding(format!("ORT DLL load failed: {e}")))?
            .commit();

        #[allow(unused_mut)] // `mut` is only exercised by with_execution_providers under --features cuda
        let mut builder = SessionBuilder::new()
            .map_err(|e| IndexerError::Embedding(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| IndexerError::Embedding(e.to_string()))?;

        #[cfg(feature = "cuda")]
        {
            builder = builder
                .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                .map_err(|e| IndexerError::Embedding(format!(
                    "CUDA execution provider failed to register (built with --features cuda). \
                     Likely cause: missing/mismatched GPU onnxruntime build (need \
                     onnxruntime_providers_cuda.dll for onnxruntime 1.24.x / ort 2.0.0-rc.12), CUDA \
                     Toolkit 12.x, or cuDNN 9.x on PATH/ORT_DYLIB_PATH dir. Underlying error: {e}"
                )))?;
        }

        let session = builder
            .commit_from_file(&model_path)
            .map_err(|e| IndexerError::Embedding(e.to_string()))?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| IndexerError::Embedding(e.to_string()))?;

        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy:            tokenizers::PaddingStrategy::BatchLongest,
            direction:           tokenizers::PaddingDirection::Right,
            pad_to_multiple_of:  None,
            pad_id:              1,
            pad_type_id:         0,
            pad_token:           "<pad>".to_string(),
        }));
        tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: 512,
            strategy:   tokenizers::TruncationStrategy::LongestFirst,
            stride:     0,
            direction:  tokenizers::TruncationDirection::Right,
        })).map_err(|e| IndexerError::Embedding(e.to_string()))?;

        Ok(Self { session: Mutex::new(session), tokenizer })
    }

    /// Embed a batch of texts with the given task prefix.
    /// Returns an `Array2<f32>` of shape `(n, EMBEDDING_DIM)`, L2-normalised.
    pub fn embed(
        &self,
        texts:  &[&str],
        prefix: &str,
    ) -> Result<Array2<f32>, IndexerError> {
        self.embed_timed(texts, prefix).map(|(vecs, _)| vecs)
    }

    /// Same as `embed`, but also returns a `tokenize`/`inference` split for
    /// `index --profile`. `tokenize` covers `encode_batch`; `inference` covers
    /// everything after it (tensor build, `session.run`, mean-pool, normalise)
    /// — the part a CUDA execution provider would move to the GPU (ADR-006).
    pub fn embed_timed(
        &self,
        texts:  &[&str],
        prefix: &str,
    ) -> Result<(Array2<f32>, EmbedTimings), IndexerError> {
        if texts.is_empty() {
            let vecs = Array2::from_shape_vec((0, EMBEDDING_DIM), vec![])
                .map_err(|e| IndexerError::Embedding(e.to_string()))?;
            return Ok((vecs, EmbedTimings { tokenize: Duration::ZERO, inference: Duration::ZERO }));
        }

        // Sort by length to minimise padding (same optimisation as Python)
        let mut indexed: Vec<(usize, String)> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (i, format!("{}{}", prefix, t)))
            .collect();
        indexed.sort_by_key(|(_, t)| t.len());

        let sorted_texts: Vec<&str> = indexed.iter().map(|(_, t)| t.as_str()).collect();

        let t_tokenize = Instant::now();
        let encodings = self.tokenizer
            .encode_batch(sorted_texts, true)
            .map_err(|e| IndexerError::Embedding(e.to_string()))?;
        let tokenize = t_tokenize.elapsed();

        let t_inference = Instant::now();
        let batch_size = encodings.len();
        let seq_len    = encodings[0].get_ids().len();

        let mut ids_flat:   Vec<i64> = Vec::with_capacity(batch_size * seq_len);
        let mut mask_flat:  Vec<i64> = Vec::with_capacity(batch_size * seq_len);
        let mut types_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            for &v in enc.get_ids()           { ids_flat.push(v as i64); }
            for &v in enc.get_attention_mask() { mask_flat.push(v as i64); }
            for &v in enc.get_type_ids()       { types_flat.push(v as i64); }
        }

        // Build ort Tensors from flat vecs (avoids ndarray version mismatch with ort)
        let input_ids_t = ort::value::Tensor::<i64>::from_array(
            ([batch_size, seq_len], ids_flat)
        ).map_err(|e| IndexerError::Embedding(e.to_string()))?;
        let attn_mask_t = ort::value::Tensor::<i64>::from_array(
            ([batch_size, seq_len], mask_flat.clone())
        ).map_err(|e| IndexerError::Embedding(e.to_string()))?;
        let type_ids_t = ort::value::Tensor::<i64>::from_array(
            ([batch_size, seq_len], types_flat)
        ).map_err(|e| IndexerError::Embedding(e.to_string()))?;

        // Lock scope: SessionOutputs borrows from session and must be dropped
        // before the MutexGuard.  All data is extracted to plain Vecs here.
        let (pooled_flat, hidden_dim) = {
            let mut session = self.session.lock()
                .map_err(|_| IndexerError::Embedding("session mutex poisoned".into()))?;
            let outputs = session
                .run(ort::inputs![
                    "input_ids"      => input_ids_t,
                    "attention_mask" => attn_mask_t,
                    "token_type_ids" => type_ids_t,
                ])
                .map_err(|e| IndexerError::Embedding(e.to_string()))?;

            // last_hidden_state: shape [batch, seq_len, hidden]
            let (lhs_shape, lhs_data) = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .map_err(|e| IndexerError::Embedding(e.to_string()))?;

            let (b, s, h) = (
                lhs_shape[0] as usize,
                lhs_shape[1] as usize,
                lhs_shape[2] as usize,
            );

            // Mean-pool over sequence dimension weighted by attention mask
            let mut pooled = vec![0.0f32; b * h];
            for bi in 0..b {
                let mask_sum: f32 = mask_flat[bi * seq_len..(bi + 1) * seq_len]
                    .iter()
                    .map(|&m| m as f32)
                    .sum();
                let denom = mask_sum.max(1e-10);
                for si in 0..s {
                    let m = mask_flat[bi * seq_len + si] as f32;
                    for hi in 0..h {
                        pooled[bi * h + hi] += lhs_data[bi * s * h + si * h + hi] * m;
                    }
                }
                for hi in 0..h {
                    pooled[bi * h + hi] /= denom;
                }
            }
            (pooled, h)
        };

        let b = batch_size;
        let h = hidden_dim;

        // Slice to first EMBEDDING_DIM dims and L2-normalise (Matryoshka: re-normalise after slice).
        let slice_dim = EMBEDDING_DIM.min(h);
        let mut result_flat = vec![0.0f32; b * EMBEDDING_DIM];
        for (i, (orig_idx, _)) in indexed.iter().enumerate() {
            let start = i * h;
            let norm: f32 = pooled_flat[start..start + slice_dim]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .max(1e-10);
            let out_start = orig_idx * EMBEDDING_DIM;
            for j in 0..slice_dim {
                result_flat[out_start + j] = pooled_flat[start + j] / norm;
            }
        }

        let vecs = Array2::from_shape_vec((b, EMBEDDING_DIM), result_flat)
            .map_err(|e| IndexerError::Embedding(e.to_string()))?;
        let inference = t_inference.elapsed();

        Ok((vecs, EmbedTimings { tokenize, inference }))
    }
}

/// `tokenize`/`inference` split for a single `Embedder::embed_timed` call.
/// See `Profile` for how these accumulate across an `index --profile` run.
#[derive(Debug, Clone, Copy)]
pub struct EmbedTimings {
    pub tokenize:  Duration,
    pub inference: Duration,
}

// ── Reservoir sampling ────────────────────────────────────────────────────────

/// Algorithm R reservoir sampler.  Maintains a uniform random sample of
/// `TRAIN_IDEAL` texts from a potentially infinite stream.
#[allow(dead_code)]
pub fn update_reservoir(
    reservoir:  &mut Vec<String>,
    new_texts:  &[String],
    total_seen: &mut usize,
) {
    use rand::Rng;
    let mut rng = rand::rng();
    for text in new_texts {
        *total_seen += 1;
        if reservoir.len() < TRAIN_IDEAL {
            reservoir.push(text.clone());
        } else {
            let j = rng.random_range(0..*total_seen);
            if j < TRAIN_IDEAL {
                reservoir[j] = text.clone();
            }
        }
    }
}

// ── Parallel file processing ──────────────────────────────────────────────────

enum FileAction {
    Skip,
    Error,
    Process {
        meta:             crate::storage::FileMetadata,
        hash:             String,
        chunk_result:     ChunkResult,
        chunker_method:   String,
        existing_file_id: Option<i64>,
    },
}

/// Pure I/O + chunking step — no DB, no embedding.  Safe to call from rayon
/// worker threads.  All async work (DB writes, embedding) stays in the serial
/// phase of `index_root`.
fn process_file_sync(
    storage: &dyn StorageClient,
    chunker: &FileChunker,
    map:     &ChunkerMap,
    uri:     &str,
    info:    Option<&FileInfo>,
) -> FileAction {
    let meta = match storage.get_metadata(uri) {
        Ok(m)  => m,
        Err(e) => { tracing::warn!("Skipping {}: {}", uri, e); return FileAction::Error; }
    };

    if let Some(info) = info
        && info.modified_at == Some(meta.modified_at) && !info.has_unembedded {
            return FileAction::Skip;
        }

    let bytes = match storage.get_file_bytes(uri) {
        Ok(b)  => b,
        Err(e) => { tracing::warn!("Skipping {}: {}", uri, e); return FileAction::Error; }
    };

    let hash = md5_hex(&bytes);

    if let Some(info) = info
        && hash == info.content_hash && !info.has_unembedded {
            return FileAction::Skip;
        }

    tracing::info!("Indexing: {}", uri);
    let (chunk_result, chunker_method) = match chunker.chunk(&bytes, &meta, map) {
        Ok(r)  => r,
        Err(e) => { tracing::warn!("Chunker error for {}: {}", uri, e); return FileAction::Error; }
    };

    FileAction::Process {
        meta,
        hash,
        chunk_result,
        chunker_method,
        existing_file_id: info.map(|i| i.file_id),
    }
}

// ── Stats ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct Stats {
    pub indexed:   usize,
    pub skipped:   usize,
    pub removed:   usize,
    pub errors:    usize,
    pub vec_total: usize,
}

// ── Push ingress ────────────────────────────────────────────────────────────────

/// A single pre-fetched record handed to the push ingress (`index_records`).
///
/// Unlike the pull path, the bytes are supplied by the caller (never read from a
/// `StorageClient`), and `meta` is caller-owned record-level metadata merged onto
/// every chunk's `meta` under the reserved `"record"` key — so it returns on every
/// search hit via `ChunkRow.meta`/`FtsResult.meta`.
#[derive(Debug, Clone)]
pub struct IngestRecord {
    /// Caller-chosen identity, e.g. `"change://105/etc/nginx.conf@2026-06-22T18:03:01Z"`.
    /// Load-bearing: distinct URIs accumulate history; a reused URI overwrites.
    pub uri:         String,
    /// Bytes the caller already holds (a redacted diff, a rendered config, …).
    pub content:     Vec<u8>,
    /// Explicit MIME — there is no extension to dispatch on. Use a `text/*` type.
    pub mime_type:   String,
    /// Caller's logical timestamp (secs since epoch). Defaults to now if `None`.
    pub modified_at: Option<f64>,
    /// Record-level metadata, merged onto every chunk's `meta`.
    pub meta:        serde_json::Value,
}

// ── IncrementalIndexer ────────────────────────────────────────────────────────

pub struct IncrementalIndexer {
    pub storage:          Arc<dyn StorageClient>,
    #[allow(dead_code)]
    pub chunker:          FileChunker,
    pub db:               Arc<EnterpriseDb>,
    pub vectors:          LanceStore,
    pub chunker_map:      ChunkerMap,
    #[allow(dead_code)]
    pub index_dir:        PathBuf,
    /// Per-instance embed batch size, resolved from `file_indexer.toml`
    /// (`[indexer].embed_batch_size`) with `EMBED_BATCH_SIZE` as the default.
    /// Overridable by the caller after construction (e.g. `index --embed-batch-size`).
    pub embed_batch_size: usize,
    embedder:             Option<Arc<Embedder>>,
    /// `index --profile` accumulator. `None` on a normal run — every timing
    /// call site branches on `if let Some(prof) = &self.profile`, so a
    /// non-profiled run pays no timing overhead. Set via `with_profile`.
    profile:              Option<Arc<Profile>>,
}

impl IncrementalIndexer {
    /// `force_recreate` is threaded into `LanceStore::open_or_create` — it permits
    /// dropping and rebuilding the vector table on a dimension mismatch. Only
    /// `index --reindex` should pass `true`; every other caller (ingest, recheck)
    /// passes `false` so a stale/misconfigured embedder can never silently wipe an
    /// existing vector table.
    pub async fn new(
        storage:        Arc<dyn StorageClient>,
        index_dir:      &Path,
        force_recreate: bool,
    ) -> Result<Self, IndexerError> {
        std::fs::create_dir_all(index_dir).map_err(|e| IndexerError::Io {
            path:   index_dir.to_path_buf(),
            source: e,
        })?;
        let db_path     = index_dir.join("enterprise.db");
        let db          = Arc::new(EnterpriseDb::new(&db_path)?);
        let vectors     = LanceStore::open_or_create(&index_dir.join("lance"), force_recreate).await?;
        let chunker_map = ChunkerMap::load_or_create(index_dir)?;

        let raw_config = crate::config::RawConfig::load(index_dir)?;
        raw_config.validate_embedding_dim(EMBEDDING_DIM)?;
        let embed_batch_size = crate::config::resolve(None, raw_config.indexer.embed_batch_size, EMBED_BATCH_SIZE);

        let onnx_dir = raw_config.embedder.onnx_model_dir.clone()
            .or_else(|| std::env::var("NOMIC_ONNX_PATH").ok());
        let onnx_model_file = raw_config.embedder.onnx_model_file.clone()
            .or_else(|| std::env::var("NOMIC_ONNX_FILE").ok())
            .unwrap_or_else(|| "nomic-embed-text-v1.5.onnx".to_string());
        let ort_dylib = raw_config.embedder.ort_dylib_path.clone()
            .or_else(|| std::env::var("ORT_DYLIB_PATH").ok());
        let ort_load_timeout = std::time::Duration::from_secs(
            raw_config.embedder.load_timeout_secs.unwrap_or(30)
        );
        let embedder = onnx_dir
            .and_then(|p| Embedder::load(Path::new(&p), &onnx_model_file, ort_dylib.as_deref(), ort_load_timeout).ok())
            .map(Arc::new);

        Ok(Self {
            storage,
            chunker: FileChunker::new(),
            db,
            vectors,
            chunker_map,
            index_dir: index_dir.to_path_buf(),
            embed_batch_size,
            embedder,
            profile: None,
        })
    }

    /// Test-only override to deterministically force the "no embedder" branch,
    /// regardless of whether `NOMIC_ONNX_PATH` happens to be set in the ambient
    /// test environment — avoids mutating a global env var, which would race with
    /// other parallel tests.
    #[cfg(test)]
    pub(crate) fn with_embedder_for_test(mut self, embedder: Option<Arc<Embedder>>) -> Self {
        self.embedder = embedder;
        self
    }

    /// Enable per-stage timing for `index --profile`. See `Profile` for what
    /// is measured and why stage totals overlap rather than partition
    /// wall-clock.
    pub fn with_profile(mut self, profile: Arc<Profile>) -> Self {
        self.profile = Some(profile);
        self
    }

    #[allow(clippy::type_complexity)]
    pub async fn index_root(
        &self,
        root_uri:    &str,
        reindex:     bool,
        no_embed:    bool,
        on_start:    Option<&(dyn Fn(usize) + Sync)>,
        on_progress: Option<&(dyn Fn(usize, usize, &Stats) + Sync)>,
    ) -> Result<Stats, IndexerError> {
        if !no_embed && self.embedder.is_none() {
            return Err(IndexerError::NoEmbedder(
                "index requires an embedder for dense/hybrid search, but none is configured or it failed to \
                 load. Set NOMIC_ONNX_PATH to a directory containing nomic-embed-text-v1.5.onnx + \
                 tokenizer.json, and ensure ORT_DYLIB_PATH points at a matching onnxruntime build. If you \
                 intend to build a sparse-only (BM25) index on purpose, rerun with --no-embed — this is \
                 recorded in the index so search/score report the reduced mode plainly."
                    .to_string(),
            ));
        }
        self.db.set_meta("embed_mode", if no_embed { "no_embed" } else { "enabled" })?;

        let mut stats = Stats::default();

        if reindex {
            self.vectors.delete_all().await?;
            self.db.delete_all_chunks()?;
        }

        let file_uris: Vec<String> = self.storage
            .list_files(root_uri)
            .collect::<Result<Vec<_>, _>>()?;
        let live_uris: std::collections::HashSet<String> =
            file_uris.iter().cloned().collect();

        let stored_info = Arc::new(if reindex {
            std::collections::HashMap::new()
        } else {
            self.db.get_all_file_info()?
        });
        let stored_info_bg = stored_info.clone();

        let total = file_uris.len();
        if let Some(f) = on_start { f(total); }

        let (tx, mut rx) = tokio::sync::mpsc::channel::<FileAction>(256);
        let storage_bg    = self.storage.clone();
        let chunker_map_bg = self.chunker_map.clone();
        let profile_bg    = self.profile.clone();

        let phase1 = tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            // Run file I/O + chunking on a DEDICATED rayon pool, NOT the global one.
            // When the bounded channel fills, these workers park inside
            // `tx.blocking_send`. The consumer's embedding step tokenizes with
            // `Tokenizer::encode_batch`, which fans out over the GLOBAL rayon pool.
            // If the producer used that same global pool, every worker would be
            // parked on `blocking_send` and `encode_batch` would wait forever for a
            // free worker — a deadlock that strands the whole index (0 CPU) once the
            // file count exceeds the channel capacity. An isolated pool keeps the
            // global pool free for the tokenizer while preserving backpressure.
            let pool = rayon::ThreadPoolBuilder::new()
                .build()
                .expect("failed to build chunking thread pool");
            pool.install(|| {
                file_uris.par_iter().for_each(|uri| {
                    let info = stored_info_bg.get(uri.as_str());
                    let action = match &profile_bg {
                        Some(prof) => {
                            let t0 = Instant::now();
                            let action = process_file_sync(
                                storage_bg.as_ref(), &FileChunker::new(), &chunker_map_bg, uri, info,
                            );
                            prof.add_chunking(t0.elapsed());
                            action
                        }
                        None => process_file_sync(
                            storage_bg.as_ref(), &FileChunker::new(), &chunker_map_bg, uri, info,
                        ),
                    };
                    let _ = tx.blocking_send(action);
                });
            });
        });

        let mut text_buffer:  Vec<String> = Vec::new();
        let mut lance_id_buf: Vec<i64>   = Vec::new();
        let mut chunk_id_buf: Vec<i64>   = Vec::new();
        let mut tier_buf:     Vec<u8>    = Vec::new();

        let mut checked = 0usize;
        while let Some(action) = rx.recv().await {
            checked += 1;
            match action {
                FileAction::Skip  => stats.skipped += 1,
                FileAction::Error => stats.errors  += 1,
                FileAction::Process { meta, hash, chunk_result, chunker_method, existing_file_id } => {
                    if let Some(file_id) = existing_file_id {
                        let old_lance_ids = self.db.get_lance_ids_for_file(file_id)?;
                        self.vectors.remove_ids(&old_lance_ids).await?;
                    }
                    match &self.profile {
                        Some(prof) => {
                            let t0 = Instant::now();
                            self.write_file(
                                &meta, &hash, chunk_result, &chunker_method,
                                &serde_json::Value::Null,
                                &mut text_buffer, &mut lance_id_buf,
                                &mut chunk_id_buf, &mut tier_buf,
                            )?;
                            prof.add_sqlite(t0.elapsed());
                        }
                        None => {
                            self.write_file(
                                &meta, &hash, chunk_result, &chunker_method,
                                &serde_json::Value::Null,
                                &mut text_buffer, &mut lance_id_buf,
                                &mut chunk_id_buf, &mut tier_buf,
                            )?;
                        }
                    }
                    stats.indexed += 1;

                    if !no_embed && text_buffer.len() >= self.embed_batch_size {
                        self.flush_embeddings(
                            &text_buffer, &lance_id_buf, &chunk_id_buf, &tier_buf,
                        ).await?;
                        text_buffer.clear();
                        lance_id_buf.clear();
                        chunk_id_buf.clear();
                        tier_buf.clear();
                    }
                }
            }

            if total >= 20 && (checked.is_multiple_of(100) || checked == total) {
                tracing::info!(
                    "Checked {}/{} (indexed={} skipped={} errors={})",
                    checked, total, stats.indexed, stats.skipped, stats.errors
                );
                if let Some(f) = on_progress { f(checked, total, &stats); }
            }
        }

        phase1.await.map_err(|e| IndexerError::Other(e.to_string().into()))?;

        // Final flush for any chunks accumulated since the last EMBED_BATCH_SIZE boundary.
        if !no_embed && !text_buffer.is_empty() {
            self.flush_embeddings(&text_buffer, &lance_id_buf, &chunk_id_buf, &tier_buf).await?;
        }

        // Build IVF-PQ index with dynamically computed partition count.
        let ntotal = self.vectors.ntotal().await?;
        if ntotal >= TRAIN_MIN {
            let nlist = compute_nlist(ntotal);
            match &self.profile {
                Some(prof) => {
                    let t0 = Instant::now();
                    let _ = self.vectors.create_ivf_pq_index(nlist).await;
                    prof.add_ivf(t0.elapsed());
                }
                None => {
                    let _ = self.vectors.create_ivf_pq_index(nlist).await;
                }
            }
        }

        // Remove files no longer on disk, including their LanceDB vectors (issue: ghost vectors).
        //
        // Scope the prune to the root we just indexed. `list_all_file_uris` returns *every*
        // file in the index, which may span several roots (the index dir can be shared across
        // `index <root>` calls to build a union corpus). A file under a *different* root is still
        // on disk — it is simply not in this root's `live_uris` — so pruning on `live_uris` alone
        // would silently evict every other root. Only reconcile URIs under `root_uri`.
        let root_prefix = {
            let mut p = root_uri.to_string();
            if !p.ends_with(std::path::MAIN_SEPARATOR) {
                p.push(std::path::MAIN_SEPARATOR);
            }
            p
        };
        let all_uris = self.db.list_all_file_uris()?;
        for uri in all_uris {
            let under_this_root = uri == root_uri || uri.starts_with(&root_prefix);
            if under_this_root && !live_uris.contains(uri.as_str()) {
                if let Some(info) = stored_info.get(uri.as_str()) {
                    let old_lance_ids = self.db.get_lance_ids_for_file(info.file_id)?;
                    self.vectors.remove_ids(&old_lance_ids).await?;
                }
                self.db.delete_file(&uri)?;
                stats.removed += 1;
            }
        }

        stats.vec_total = self.vectors.ntotal().await?;
        Ok(stats)
    }

    #[allow(clippy::too_many_arguments)]
    fn write_file(
        &self,
        meta:           &crate::storage::FileMetadata,
        content_hash:   &str,
        chunk_result:   ChunkResult,
        chunker_method: &str,
        record_meta:    &serde_json::Value,
        text_buf:       &mut Vec<String>,
        lance_id_buf:   &mut Vec<i64>,
        chunk_id_buf:   &mut Vec<i64>,
        tier_buf:       &mut Vec<u8>,
    ) -> Result<(), IndexerError> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let file_id = self.db.upsert_file(
            &meta.file_uri,
            &meta.mime_type,
            content_hash,
            meta.size_bytes as i64,
            now,
            Some(meta.modified_at),
        )?;

        self.db.delete_chunks_for_file(file_id)?;

        let chunk_inputs: Vec<ChunkInput> = chunk_result.chunks.iter().map(|c| ChunkInput {
            tier:           c.tier,
            chunk_index:    c.chunk_index,
            content:        c.content.clone(),
            token_count:    Some(c.token_count),
            meta:           merge_record_meta(&c.meta, record_meta),
            chunker_method: Some(chunker_method.to_string()),
        }).collect();

        let chunk_ids = self.db.insert_chunks(file_id, &chunk_inputs)?;

        for (chunk, &chunk_id) in chunk_result.chunks.iter().zip(&chunk_ids) {
            let lance = stable_id(&meta.file_uri, chunk.tier, chunk.chunk_index);
            text_buf.push(chunk.content.clone());
            lance_id_buf.push(lance);
            chunk_id_buf.push(chunk_id);
            tier_buf.push(chunk.tier);
        }

        let edge_inputs: Vec<EdgeInput> = chunk_result.edges.iter().map(|e| EdgeInput {
            src_chunk: chunk_ids[e.src_chunk_index.min(chunk_ids.len().saturating_sub(1))],
            dst_uri:   e.dst_uri.clone(),
            edge_type: match e.edge_type {
                EdgeType::LinksTo    => crate::db::EdgeType::LinksTo,
                EdgeType::References => crate::db::EdgeType::References,
                EdgeType::Embeds     => crate::db::EdgeType::Embeds,
            },
            meta: e.meta.clone(),
        }).collect();
        self.db.insert_edges(&edge_inputs)?;

        Ok(())
    }

    /// Reindex a specific list of URIs, bypassing mtime/hash change detection.
    /// Used by `recheck` to force re-chunk files whose chunker method has changed.
    pub async fn index_uris(&self, uris: &[String]) -> Result<Stats, IndexerError> {
        if uris.is_empty() {
            return Ok(Stats::default());
        }

        // Gate on the index's RECORDED mode, not a caller-supplied flag: `index_uris`
        // has no `no_embed` parameter of its own, because recheck must not wrongly
        // hard-error on a corpus that was deliberately built with `--no-embed` (mode
        // recorded `no_embed` → skip embedding silently, as before). A corpus that WAS
        // supposed to have embeddings but whose embedder is now broken/missing DOES
        // hard-error — closing the same silent-degradation hole for recheck, not just
        // plain `index`.
        let recorded_mode = self.db.get_meta("embed_mode")?;
        let no_embed = recorded_mode.as_deref() == Some("no_embed");
        if !no_embed && self.embedder.is_none() {
            let mode_label = recorded_mode.as_deref().unwrap_or("unset (pre-dates embed_mode tracking)");
            return Err(IndexerError::NoEmbedder(format!(
                "recheck cannot re-embed drifted files: this index's recorded embed_mode is '{}' (embeddings \
                 expected), but no embedder is configured/available right now. Configure NOMIC_ONNX_PATH and \
                 retry, or rebuild the whole index with `index --reindex --no-embed` to intentionally switch \
                 this corpus to sparse-only.",
                mode_label
            )));
        }

        let mut stats = Stats::default();
        let all_info  = self.db.get_all_file_info()?;

        let mut text_buffer:  Vec<String> = Vec::new();
        let mut lance_id_buf: Vec<i64>    = Vec::new();
        let mut chunk_id_buf: Vec<i64>    = Vec::new();
        let mut tier_buf:     Vec<u8>     = Vec::new();

        for uri in uris {
            let existing_file_id = all_info.get(uri.as_str()).map(|i| i.file_id);
            // Pass info=None so process_file_sync never skips due to unchanged mtime/hash.
            match process_file_sync(self.storage.as_ref(), &self.chunker, &self.chunker_map, uri, None) {
                FileAction::Skip  => stats.skipped += 1,
                FileAction::Error => stats.errors  += 1,
                FileAction::Process { meta, hash, chunk_result, chunker_method, .. } => {
                    if let Some(file_id) = existing_file_id {
                        let old_ids = self.db.get_lance_ids_for_file(file_id)?;
                        self.vectors.remove_ids(&old_ids).await?;
                    }
                    self.write_file(
                        &meta, &hash, chunk_result, &chunker_method,
                        &serde_json::Value::Null,
                        &mut text_buffer, &mut lance_id_buf,
                        &mut chunk_id_buf, &mut tier_buf,
                    )?;
                    stats.indexed += 1;
                }
            }
        }

        if !no_embed && !text_buffer.is_empty() {
            self.flush_embeddings(&text_buffer, &lance_id_buf, &chunk_id_buf, &tier_buf).await?;
        }

        stats.vec_total = self.vectors.ntotal().await?;
        Ok(stats)
    }

    /// Push ingress: index caller-supplied records that are **not** walkable
    /// files. Reuses the pull pipeline from chunking onward (`FileChunker::chunk`
    /// → meta-aware `write_file` → `flush_embeddings`) but synthesizes
    /// `FileMetadata` from each record instead of reading a `StorageClient`, and
    /// threads the record's `meta` onto every chunk.
    ///
    /// A pushed record is authoritative — always chunked and embedded, with no
    /// mtime/MD5 skip (that is a *pull* optimization keyed on re-walking a path).
    /// Re-pushing the same `uri` overwrites that record's chunks and vectors;
    /// distinct URIs accumulate. Unlike `index_uris`, this rebuilds the IVF-PQ
    /// index at the end, because a push feed accumulates vectors over time.
    pub async fn index_records(&self, records: &[IngestRecord]) -> Result<Stats, IndexerError> {
        if records.is_empty() {
            return Ok(Stats::default());
        }

        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let mut stats = Stats::default();
        let all_info  = self.db.get_all_file_info()?;

        let mut text_buffer:  Vec<String> = Vec::new();
        let mut lance_id_buf: Vec<i64>    = Vec::new();
        let mut chunk_id_buf: Vec<i64>    = Vec::new();
        let mut tier_buf:     Vec<u8>     = Vec::new();

        for record in records {
            // Synthetic metadata — never touches StorageClient. `name` is the
            // last URI segment for display; falls back to the whole URI.
            let name = record.uri
                .rsplit(['/', '\\'])
                .find(|s| !s.is_empty())
                .unwrap_or(record.uri.as_str())
                .to_string();
            let meta = crate::storage::FileMetadata {
                file_uri:    record.uri.clone(),
                name,
                mime_type:   record.mime_type.clone(),
                size_bytes:  record.content.len() as u64,
                modified_at: record.modified_at.unwrap_or(now),
            };

            let hash = md5_hex(&record.content);
            let (chunk_result, chunker_method) =
                match self.chunker.chunk(&record.content, &meta, &self.chunker_map) {
                    Ok(r)  => r,
                    Err(e) => {
                        tracing::warn!("Chunker error for {}: {}", record.uri, e);
                        stats.errors += 1;
                        continue;
                    }
                };

            // Evict stale LanceDB vectors on re-push: `write_file` clears only the
            // SQLite chunks, so without this the prior record's vectors are
            // orphaned (the ghost-vector bug the pull path guards against).
            if let Some(info) = all_info.get(record.uri.as_str()) {
                let old_ids = self.db.get_lance_ids_for_file(info.file_id)?;
                self.vectors.remove_ids(&old_ids).await?;
            }

            self.write_file(
                &meta, &hash, chunk_result, &chunker_method,
                &record.meta,
                &mut text_buffer, &mut lance_id_buf,
                &mut chunk_id_buf, &mut tier_buf,
            )?;
            stats.indexed += 1;

            if text_buffer.len() >= self.embed_batch_size {
                self.flush_embeddings(&text_buffer, &lance_id_buf, &chunk_id_buf, &tier_buf).await?;
                text_buffer.clear();
                lance_id_buf.clear();
                chunk_id_buf.clear();
                tier_buf.clear();
            }
        }

        if !text_buffer.is_empty() {
            self.flush_embeddings(&text_buffer, &lance_id_buf, &chunk_id_buf, &tier_buf).await?;
        }

        // Rebuild IVF-PQ so a growing push feed keeps correct ANN recall.
        let ntotal = self.vectors.ntotal().await?;
        if ntotal >= TRAIN_MIN {
            let nlist = compute_nlist(ntotal);
            let _ = self.vectors.create_ivf_pq_index(nlist).await;
        }

        stats.vec_total = self.vectors.ntotal().await?;
        Ok(stats)
    }

    async fn flush_embeddings(
        &self,
        texts:      &[String],
        lance_ids:  &[i64],
        chunk_ids:  &[i64],
        tiers:      &[u8],
    ) -> Result<(), IndexerError> {
        let embedder = match &self.embedder {
            Some(e) => e.clone(),
            None    => return Ok(()),
        };
        for i in (0..texts.len()).step_by(self.embed_batch_size) {
            let end          = (i + self.embed_batch_size).min(texts.len());
            let batch_owned: Vec<String> = texts[i..end].to_vec();
            let batch_n      = batch_owned.len();
            tracing::info!("Embedding batch {}/{} ({} chunks)", i / self.embed_batch_size + 1, texts.len().div_ceil(self.embed_batch_size), batch_n);
            let emb          = embedder.clone();
            let (vecs, timings) = tokio::task::spawn_blocking(move || {
                let batch_refs: Vec<&str> = batch_owned.iter().map(String::as_str).collect();
                emb.embed_timed(&batch_refs, "search_document: ")
            })
            .await
            .map_err(|e| IndexerError::Embedding(e.to_string()))??;
            if let Some(prof) = &self.profile {
                prof.add_tokenize(timings.tokenize);
                prof.add_inference(timings.inference);
                prof.add_chunks_embedded(batch_n);
            }
            tracing::info!("Embedded {} chunks", batch_n);
            let flat_vecs: Vec<Vec<f32>> = (0..vecs.nrows())
                .map(|r| vecs.row(r).to_vec())
                .collect();
            match &self.profile {
                Some(prof) => {
                    let t0 = Instant::now();
                    self.vectors
                        .add_vectors(&lance_ids[i..end], &chunk_ids[i..end], &tiers[i..end], &flat_vecs)
                        .await?;
                    prof.add_lance(t0.elapsed());
                }
                None => {
                    self.vectors
                        .add_vectors(&lance_ids[i..end], &chunk_ids[i..end], &tiers[i..end], &flat_vecs)
                        .await?;
                }
            }
            // Set lance_ids in SQLite only after the LanceDB write succeeds.
            // This makes lance_id IS NULL the reliable crash-detection signal.
            let id_map: HashMap<i64, i64> = chunk_ids[i..end].iter().copied()
                .zip(lance_ids[i..end].iter().copied())
                .collect();
            self.db.set_chunk_lance_ids(&id_map)?;
        }
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn md5_hex(data: &[u8]) -> String {
    use md5::{Digest, Md5};
    let mut h = Md5::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

/// Shallow-merge caller-supplied record metadata into a chunk's chunker-derived
/// meta under the reserved `"record"` key. A `Value::Null` `record_meta` is a
/// no-op that returns the chunker meta unchanged — this is what the pull path
/// passes, keeping its `chunks.meta` byte-for-byte identical to before the push
/// path existed. The reserved key never collides with chunker-emitted keys.
fn merge_record_meta(
    chunk_meta:  &serde_json::Value,
    record_meta: &serde_json::Value,
) -> serde_json::Value {
    if record_meta.is_null() {
        return chunk_meta.clone();
    }
    let mut obj = match chunk_meta {
        serde_json::Value::Object(m) => m.clone(),
        _                            => serde_json::Map::new(),
    };
    obj.insert("record".to_string(), record_meta.clone());
    serde_json::Value::Object(obj)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn stable_id_matches_python_reference() {
        let id = stable_id("file:///test/a.txt", 1, 0);

        assert!(id >= 0, "ID should be non-negative");
        assert!(id <= 0x0FFF_FFFF_FFFF_FFFFi64, "ID should fit in 60 bits");

        assert_eq!(
            stable_id("file:///test/a.txt", 1, 0),
            stable_id("file:///test/a.txt", 1, 0),
            "stable_id must be deterministic"
        );

        assert_ne!(stable_id("file:///test/a.txt", 1, 0), stable_id("file:///test/a.txt", 1, 1));
        assert_ne!(stable_id("file:///test/a.txt", 1, 0), stable_id("file:///test/a.txt", 2, 0));
        assert_ne!(stable_id("file:///test/a.txt", 1, 0), stable_id("file:///test/b.txt", 1, 0));

        assert_eq!(id >> 60, 0, "top 4 bits must be zero (60-bit mask)");
    }

    fn load_embedder() -> Option<Embedder> {
        let path = std::env::var("NOMIC_ONNX_PATH").ok()?;
        let model_file = std::env::var("NOMIC_ONNX_FILE").unwrap_or_else(|_| "nomic-embed-text-v1.5.onnx".into());
        Embedder::load(Path::new(&path), &model_file, None, std::time::Duration::from_secs(30)).ok()
    }

    #[test]
    fn embed_returns_unit_vectors() {
        let embedder = match load_embedder() {
            Some(e) => e,
            None    => { println!("SKIP: NOMIC_ONNX_PATH not set"); return; }
        };
        let texts = vec!["hello world", "rust is great"];
        let vecs  = embedder.embed(&texts, "search_document: ").unwrap();
        assert_eq!(vecs.shape(), &[2, EMBEDDING_DIM]);
        for i in 0..vecs.nrows() {
            let norm: f32 = vecs.row(i).dot(&vecs.row(i)).sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "row {} norm = {}", i, norm);
        }
    }

    #[test]
    fn embed_batch_matches_single() {
        let embedder = match load_embedder() {
            Some(e) => e,
            None    => { println!("SKIP: NOMIC_ONNX_PATH not set"); return; }
        };
        let texts = vec!["alpha", "beta", "gamma", "delta"];
        let batch  = embedder.embed(&texts, "search_document: ").unwrap();
        for (i, t) in texts.iter().enumerate() {
            let single = embedder.embed(&[t], "search_document: ").unwrap();
            // Cosine similarity for unit vectors = dot product; int8 quantization
            // introduces small numerical differences, so use a lenient threshold.
            let cosine: f32 = batch.row(i).dot(&single.row(0));
            assert!(cosine > 0.95, "batch vs single cosine similarity too low for '{}': {}", t, cosine);
        }
    }

    /// ADR-006 Phase 1: compile-only on the default build (gated behind
    /// `--features cuda`); auto-skips at runtime when there's no GPU/CUDA
    /// toolkit present, mirroring `load_embedder`'s skip pattern above. This
    /// machine has no CUDA Toolkit/cuDNN installed, so `Embedder::load` is
    /// expected to error here — the skip path, not the assertion path, is
    /// what actually runs today.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_embedder_produces_unit_vectors() {
        let path = match std::env::var("NOMIC_ONNX_PATH") {
            Ok(p) => p,
            Err(_) => { println!("SKIP: NOMIC_ONNX_PATH not set"); return; }
        };
        let model_file = std::env::var("NOMIC_ONNX_FILE").unwrap_or_else(|_| "nomic-embed-text-v1.5.onnx".into());
        let embedder = match Embedder::load(Path::new(&path), &model_file, None, std::time::Duration::from_secs(30)) {
            Ok(e) => e,
            Err(e) => { println!("SKIP: CUDA embedder failed to load ({e}); GPU runtime likely absent"); return; }
        };
        let texts = vec!["hello world"];
        let vecs  = embedder.embed(&texts, "search_document: ").unwrap();
        assert_eq!(vecs.shape(), &[1, EMBEDDING_DIM]);
        let norm: f32 = vecs.row(0).dot(&vecs.row(0)).sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
    }

    #[test]
    fn run_with_timeout_returns_err_on_hang() {
        use std::time::{Duration, Instant};
        let start = Instant::now();
        let result = run_with_timeout(Duration::from_millis(50), || {
            std::thread::sleep(Duration::from_millis(500));
            Ok::<u32, IndexerError>(1)
        });
        assert!(result.is_err(), "hung closure should time out with an error");
        assert!(
            start.elapsed() < Duration::from_millis(400),
            "caller should not block for the full closure duration, elapsed={:?}",
            start.elapsed()
        );
    }

    #[test]
    fn run_with_timeout_returns_ok_when_fast() {
        use std::time::Duration;
        let result = run_with_timeout(Duration::from_secs(5), || Ok::<u32, IndexerError>(42));
        assert_eq!(result.unwrap(), 42);
    }

    fn make_unit_vecs(n: usize) -> Vec<Vec<f32>> {
        (0..n).map(|i| {
            let mut v = vec![0.0f32; EMBEDDING_DIM];
            v[i % EMBEDDING_DIM] = 1.0;
            v
        }).collect()
    }

    #[tokio::test]
    async fn lancedb_add_and_search() {
        let dir   = TempDir::new().unwrap();
        let store = LanceStore::open_or_create(dir.path(), false).await.unwrap();

        let ids       = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let chunk_ids = ids.clone();
        let tiers     = vec![1u8; 10];
        let vecs      = make_unit_vecs(10);

        store.add_vectors(&ids, &chunk_ids, &tiers, &vecs).await.unwrap();
        assert_eq!(store.ntotal().await.unwrap(), 10);

        let query = vecs[0].clone();
        let results = store.search(query, 3).await.unwrap();
        assert!(!results.is_empty(), "search should return results");
        assert_eq!(results[0].id, 1, "top-1 result should match exact query vector");
    }

    #[tokio::test]
    async fn lancedb_remove_ids() {
        let dir   = TempDir::new().unwrap();
        let store = LanceStore::open_or_create(dir.path(), false).await.unwrap();

        let ids   = vec![100i64, 200, 300];
        let vecs  = make_unit_vecs(3);
        store.add_vectors(&ids, &ids, &[1u8, 1, 1], &vecs).await.unwrap();
        assert_eq!(store.ntotal().await.unwrap(), 3);

        store.remove_ids(&[200i64]).await.unwrap();
        assert_eq!(store.ntotal().await.unwrap(), 2);

        let results = store.search(vecs[1].clone(), 3).await.unwrap();
        assert!(
            results.iter().all(|r| r.id != 200),
            "removed id=200 should not appear in search results"
        );
    }

    // Proves remove_ids batches its `id IN (...)` filter: with more ids than
    // MAX_IDS_PER_FILTER, a bug that only issued the first chunk's delete (or
    // dropped later chunks) would leave the tail ids behind instead of
    // ntotal() == 0.
    #[tokio::test]
    async fn lancedb_remove_ids_spans_multiple_filter_batches() {
        let dir   = TempDir::new().unwrap();
        let store = LanceStore::open_or_create(dir.path(), false).await.unwrap();

        let n: i64 = MAX_IDS_PER_FILTER as i64 + 500; // forces 2 batches (1000 + 500)
        let ids: Vec<i64> = (1..=n).collect();
        let vecs = make_unit_vecs(n as usize);
        store.add_vectors(&ids, &ids, &vec![1u8; n as usize], &vecs).await.unwrap();
        assert_eq!(store.ntotal().await.unwrap(), n as usize);

        store.remove_ids(&ids).await.unwrap();
        assert_eq!(
            store.ntotal().await.unwrap(), 0,
            "all ids should be removed, including the tail past MAX_IDS_PER_FILTER"
        );
    }

    // Proves fetch_vectors_by_ids batches its `id IN (...)` filter and merges
    // results across batches: a bug that only queried the first chunk would
    // silently under-return the map instead of erroring.
    #[tokio::test]
    async fn lancedb_fetch_vectors_by_ids_spans_multiple_filter_batches() {
        let dir   = TempDir::new().unwrap();
        let store = LanceStore::open_or_create(dir.path(), false).await.unwrap();

        let n: i64 = MAX_IDS_PER_FILTER as i64 + 500; // forces 2 batches (1000 + 500)
        let ids: Vec<i64> = (1..=n).collect();
        let vecs = make_unit_vecs(n as usize);
        store.add_vectors(&ids, &ids, &vec![1u8; n as usize], &vecs).await.unwrap();

        let fetched = store.fetch_vectors_by_ids(&ids).await.unwrap();
        assert_eq!(
            fetched.len(), n as usize,
            "every id across every batch should be fetched, including the tail past MAX_IDS_PER_FILTER"
        );
        assert!(fetched.contains_key(&1), "first-batch id should be present");
        assert!(fetched.contains_key(&n), "id from the batch past MAX_IDS_PER_FILTER should be present");
    }

    /// Builds a "chunks" table schema identical to `chunks_schema()` except the
    /// vector width is `dim` instead of `EMBEDDING_DIM` — used to simulate a stale
    /// table left behind by an older/different embedding model.
    fn mismatched_chunks_schema(dim: i32) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id",       DataType::Int64, false),
            Field::new("chunk_id", DataType::Int64, false),
            Field::new("tier",     DataType::Int8,  false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim,
                ),
                false,
            ),
        ]))
    }

    async fn create_mismatched_chunks_table(dir: &std::path::Path, dim: i32) {
        let uri  = dir.to_string_lossy();
        let conn = lancedb::connect(uri.as_ref()).execute().await.unwrap();
        let schema      = mismatched_chunks_schema(dim);
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(
            RecordBatchIterator::new(vec![Ok(empty_batch)].into_iter(), schema),
        );
        conn.create_table("chunks", reader).execute().await.unwrap();
    }

    #[tokio::test]
    async fn lance_store_dim_mismatch_without_force_recreate_errors() {
        let dir = TempDir::new().unwrap();
        create_mismatched_chunks_table(dir.path(), 128).await;

        let result = LanceStore::open_or_create(dir.path(), false).await;
        let msg = match result {
            Err(e) => e.to_string(),
            Ok(_)  => panic!("dim mismatch without force_recreate should error"),
        };
        assert!(msg.contains("128"), "error should mention the stored dim, got: {msg}");
        assert!(
            msg.contains(&EMBEDDING_DIM.to_string()),
            "error should mention the expected dim, got: {msg}"
        );
    }

    #[tokio::test]
    async fn lance_store_dim_mismatch_with_force_recreate_drops_and_rebuilds() {
        let dir = TempDir::new().unwrap();
        create_mismatched_chunks_table(dir.path(), 128).await;

        let store = LanceStore::open_or_create(dir.path(), true).await.unwrap();
        assert_eq!(store.ntotal().await.unwrap(), 0, "recreated table should start empty");

        // Proves the recreated table has the CORRECT (EMBEDDING_DIM) schema, not the
        // stale mismatched one — adding a real-width vector must succeed.
        let ids  = vec![1i64];
        let vecs = make_unit_vecs(1);
        store.add_vectors(&ids, &ids, &[1u8], &vecs).await.unwrap();
        assert_eq!(store.ntotal().await.unwrap(), 1);
    }

    #[test]
    fn reservoir_uniform_distribution() {
        let mut reservoir:  Vec<String> = Vec::new();
        let mut total_seen: usize       = 0;

        for i in 0..(10 * TRAIN_IDEAL) {
            update_reservoir(&mut reservoir, &[format!("text_{}", i)], &mut total_seen);
        }

        assert_eq!(
            reservoir.len(), TRAIN_IDEAL,
            "reservoir size should stabilise at TRAIN_IDEAL"
        );
        assert_eq!(total_seen, 10 * TRAIN_IDEAL);
    }

    #[tokio::test]
    async fn index_root_skips_unchanged_files() {
        if std::env::var("NOMIC_ONNX_PATH").is_err() {
            println!("SKIP: NOMIC_ONNX_PATH not set");
            return;
        }
        use crate::storage::LocalStorageClient;

        let root_dir = TempDir::new().unwrap();
        let idx_dir  = TempDir::new().unwrap();

        std::fs::write(root_dir.path().join("hello.txt"), b"hello world").unwrap();

        let storage  = Arc::new(LocalStorageClient::new());
        let indexer  = IncrementalIndexer::new(
            storage.clone(),
            idx_dir.path(),
            false,
        ).await.unwrap();

        let stats1   = indexer.index_root(&root_dir.path().to_string_lossy(), false, false, None, None).await.unwrap();
        assert_eq!(stats1.indexed, 1, "first run should index 1 file");

        let stats2   = indexer.index_root(&root_dir.path().to_string_lossy(), false, false, None, None).await.unwrap();
        assert_eq!(stats2.indexed, 0, "second run should skip unchanged file");
    }

    #[tokio::test]
    async fn index_root_hard_errors_without_embedder_and_without_no_embed() {
        use crate::storage::LocalStorageClient;

        let root_dir = TempDir::new().unwrap();
        let idx_dir  = TempDir::new().unwrap();
        std::fs::write(root_dir.path().join("hello.txt"), b"hello world").unwrap();

        let storage = Arc::new(LocalStorageClient::new());
        let indexer = IncrementalIndexer::new(storage, idx_dir.path(), false)
            .await.unwrap()
            .with_embedder_for_test(None);

        let result = indexer
            .index_root(&root_dir.path().to_string_lossy(), false, false, None, None)
            .await;
        assert!(
            matches!(result, Err(IndexerError::NoEmbedder(_))),
            "index without an embedder and without --no-embed should hard-error, got: {:?}",
            result.map(|_| ())
        );
    }

    #[tokio::test]
    async fn index_root_no_embed_flag_succeeds_and_records_mode() {
        use crate::storage::LocalStorageClient;

        let root_dir = TempDir::new().unwrap();
        let idx_dir  = TempDir::new().unwrap();
        let file_uri = root_dir.path().join("hello.txt").to_string_lossy().to_string();
        std::fs::write(&file_uri, b"hello world").unwrap();

        let storage = Arc::new(LocalStorageClient::new());
        let indexer = IncrementalIndexer::new(storage, idx_dir.path(), false)
            .await.unwrap()
            .with_embedder_for_test(None);

        let result = indexer
            .index_root(&root_dir.path().to_string_lossy(), false, true, None, None)
            .await;
        assert!(result.is_ok(), "index with --no-embed should succeed without an embedder: {:?}", result.err());

        assert_eq!(
            indexer.db.get_meta("embed_mode").unwrap(),
            Some("no_embed".to_string()),
            "embed_mode should be recorded as no_embed"
        );

        // Sparse-only indexing worked and is self-describing: chunks exist, but none
        // carry a lance_id (not a partial/broken embed — a deliberate no-embed run).
        let chunks = indexer.db.get_chunks_for_file(&file_uri).unwrap();
        assert!(!chunks.is_empty(), "file should still produce chunks under --no-embed");
        assert!(
            chunks.iter().all(|c| c.lance_id.is_none()),
            "no chunk should have a lance_id when --no-embed was used"
        );
    }

    #[tokio::test]
    async fn index_root_profile_records_chunking_and_sqlite_stages() {
        use crate::storage::LocalStorageClient;

        let root_dir = TempDir::new().unwrap();
        let idx_dir  = TempDir::new().unwrap();
        std::fs::write(root_dir.path().join("hello.txt"), b"hello world").unwrap();
        std::fs::write(root_dir.path().join("second.txt"), b"more content here").unwrap();

        let storage = Arc::new(LocalStorageClient::new());
        let profile = Arc::new(Profile::new());
        let indexer = IncrementalIndexer::new(storage, idx_dir.path(), false)
            .await.unwrap()
            .with_embedder_for_test(None)
            .with_profile(profile.clone());

        let t0 = std::time::Instant::now();
        let result = indexer
            .index_root(&root_dir.path().to_string_lossy(), false, true, None, None)
            .await;
        assert!(result.is_ok(), "no-embed indexing under --profile should still succeed: {:?}", result.err());
        let stats = result.unwrap();

        // Chunking runs in the producer (rayon pool) and sqlite writes in the
        // consumer for every processed file, so both accumulators must be
        // nonzero after indexing two real files.
        assert!(
            profile.chunking_ns.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "chunking_ns should be nonzero after indexing files"
        );
        assert!(
            profile.sqlite_ns.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "sqlite_ns should be nonzero after indexing files"
        );
        // --no-embed means no chunks went through the embedder.
        assert_eq!(profile.chunks_embedded(), 0, "no chunks should be embedded under --no-embed");

        let rendered = profile.render(t0.elapsed(), stats.indexed);
        for label in ["chunking", "tokenize", "inference", "sqlite_write", "lance_write", "ivf_build"] {
            assert!(rendered.contains(label), "render() output should mention stage '{}': {}", label, rendered);
        }
        assert!(rendered.contains("wall-clock"), "render() output should report wall-clock: {}", rendered);
        assert!(rendered.contains("chunks/sec"), "render() output should report chunks/sec: {}", rendered);
    }

    #[tokio::test]
    async fn index_root_records_enabled_mode_when_embedder_present() {
        if std::env::var("NOMIC_ONNX_PATH").is_err() {
            println!("SKIP: NOMIC_ONNX_PATH not set");
            return;
        }
        use crate::storage::LocalStorageClient;

        let root_dir = TempDir::new().unwrap();
        let idx_dir  = TempDir::new().unwrap();
        std::fs::write(root_dir.path().join("hello.txt"), b"hello world").unwrap();

        let storage = Arc::new(LocalStorageClient::new());
        let indexer = IncrementalIndexer::new(storage, idx_dir.path(), false).await.unwrap();

        let result = indexer
            .index_root(&root_dir.path().to_string_lossy(), false, false, None, None)
            .await;
        assert!(result.is_ok(), "index with a real embedder should succeed: {:?}", result.err());
        assert_eq!(indexer.db.get_meta("embed_mode").unwrap(), Some("enabled".to_string()));
    }

    #[tokio::test]
    async fn embed_batch_size_defaults_to_const_when_no_config() {
        use crate::storage::LocalStorageClient;

        let idx_dir = TempDir::new().unwrap();
        let storage = Arc::new(LocalStorageClient::new());
        let indexer = IncrementalIndexer::new(storage, idx_dir.path(), false).await.unwrap();

        assert_eq!(indexer.embed_batch_size, EMBED_BATCH_SIZE);
    }

    #[tokio::test]
    async fn embed_batch_size_reads_from_config_file() {
        use crate::storage::LocalStorageClient;

        let idx_dir = TempDir::new().unwrap();
        std::fs::write(
            idx_dir.path().join("file_indexer.toml"),
            "[indexer]\nembed_batch_size = 7\n",
        ).unwrap();

        let storage = Arc::new(LocalStorageClient::new());
        let indexer = IncrementalIndexer::new(storage, idx_dir.path(), false).await.unwrap();

        assert_eq!(indexer.embed_batch_size, 7);
    }

    #[tokio::test]
    async fn embedding_dim_mismatch_in_config_hard_errors() {
        use crate::storage::LocalStorageClient;

        let idx_dir = TempDir::new().unwrap();
        std::fs::write(
            idx_dir.path().join("file_indexer.toml"),
            "[embedder]\nembedding_dim = 768\n",
        ).unwrap();

        let storage = Arc::new(LocalStorageClient::new());
        let result = IncrementalIndexer::new(storage, idx_dir.path(), false).await;

        assert!(
            matches!(result, Err(IndexerError::Config(_))),
            "embedding_dim mismatch in file_indexer.toml should hard-error, got: {:?}",
            result.map(|_| ())
        );
    }

    #[tokio::test]
    async fn index_uris_skips_embedding_silently_when_recorded_no_embed() {
        use crate::storage::LocalStorageClient;

        let root_dir = TempDir::new().unwrap();
        let idx_dir  = TempDir::new().unwrap();
        let file_uri = root_dir.path().join("hello.txt").to_string_lossy().to_string();
        std::fs::write(&file_uri, b"hello world").unwrap();

        let storage = Arc::new(LocalStorageClient::new());
        let indexer = IncrementalIndexer::new(storage, idx_dir.path(), false)
            .await.unwrap()
            .with_embedder_for_test(None);

        indexer
            .index_root(&root_dir.path().to_string_lossy(), false, true, None, None)
            .await.unwrap();

        let result = indexer.index_uris(&[file_uri]).await;
        assert!(
            result.is_ok(),
            "recheck on an intentionally-no-embed corpus should not hard-error: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn index_uris_hard_errors_when_recorded_enabled_but_embedder_missing() {
        use crate::storage::LocalStorageClient;

        let idx_dir = TempDir::new().unwrap();
        let storage = Arc::new(LocalStorageClient::new());
        let indexer = IncrementalIndexer::new(storage, idx_dir.path(), false)
            .await.unwrap()
            .with_embedder_for_test(None);

        indexer.db.set_meta("embed_mode", "enabled").unwrap();

        let result = indexer.index_uris(&["file:///does/not/matter".to_string()]).await;
        assert!(
            matches!(result, Err(IndexerError::NoEmbedder(_))),
            "recheck on a recorded-enabled corpus with no embedder should hard-error, got: {:?}",
            result.map(|_| ())
        );
    }

    // Push ingress: runs WITHOUT a model. The embedder is optional, so a pushed
    // record still produces SQLite chunks + FTS rows (BM25-searchable), and the
    // record metadata round-trips onto every chunk — no NOMIC_ONNX_PATH needed.
    #[tokio::test]
    async fn index_records_pushes_searchable_chunks_with_meta() {
        use crate::storage::LocalStorageClient;

        let idx_dir = TempDir::new().unwrap();
        // The push path never touches storage, but the constructor requires one.
        let storage = Arc::new(LocalStorageClient::new());
        let indexer = IncrementalIndexer::new(storage, idx_dir.path(), false).await.unwrap();

        let uri = "change://105/etc/nginx.conf@2026-06-22T18:03:01Z";
        let record = IngestRecord {
            uri:         uri.to_string(),
            content:     b"server_name example.com;\nlisten 443 ssl;\nproxy_pass http://backend;\n".to_vec(),
            mime_type:   "text/plain".to_string(),
            modified_at: Some(1_700_000_000.0),
            meta:        serde_json::json!({ "tool": "write_file", "pre_hash": "abc", "vmid": 105 }),
        };

        let stats = indexer.index_records(&[record]).await.unwrap();
        assert_eq!(stats.indexed, 1, "one record should be indexed");

        // Chunks landed in SQLite and carry the record metadata under "record".
        let chunks = indexer.db.get_chunks_for_file(uri).unwrap();
        assert!(!chunks.is_empty(), "pushed record should produce chunks");
        let t12 = chunks.iter().find(|c| c.tier == 1 || c.tier == 2)
            .expect("expected at least one Tier-1/2 chunk");
        assert_eq!(t12.meta["record"]["tool"], "write_file");
        assert_eq!(t12.meta["record"]["vmid"], 105);

        // FTS5/BM25 finds the content with no embedding model loaded, and the
        // record meta round-trips on the FTS result too (the consumer's pivot).
        let hits = indexer.db.fts_search("backend", 10).unwrap();
        let hit  = hits.iter().find(|h| h.file_uri == uri)
            .expect("pushed record should be BM25-searchable without a model");
        assert_eq!(hit.meta["record"]["tool"], "write_file");
    }

    // Re-pushing the same URI is an overwrite, not an accumulation: the old
    // chunks are replaced (delete_chunks_for_file), so the count stays stable
    // and the content reflects the latest push.
    #[tokio::test]
    async fn index_records_repush_same_uri_overwrites() {
        use crate::storage::LocalStorageClient;

        let idx_dir = TempDir::new().unwrap();
        let storage = Arc::new(LocalStorageClient::new());
        let indexer = IncrementalIndexer::new(storage, idx_dir.path(), false).await.unwrap();

        let uri = "change://1/config";
        let mk  = |body: &str| IngestRecord {
            uri:         uri.to_string(),
            content:     body.as_bytes().to_vec(),
            mime_type:   "text/plain".to_string(),
            modified_at: None,
            meta:        serde_json::Value::Null,
        };

        indexer.index_records(&[mk("alpha token original")]).await.unwrap();
        indexer.index_records(&[mk("beta token replacement")]).await.unwrap();

        // Old content is gone; new content is searchable.
        assert!(indexer.db.fts_search("original", 10).unwrap().is_empty(),
            "original content should be overwritten");
        assert!(indexer.db.fts_search("replacement", 10).unwrap()
            .iter().any(|h| h.file_uri == uri),
            "replacement content should be searchable");
    }
}
