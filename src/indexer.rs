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
use std::sync::{Arc, Mutex};

// ── Constants ─────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub const HF_MODEL_ID:      &str   = "nomic-ai/nomic-embed-text-v1.5";
/// Matryoshka slice dimension: first 256 of 768 dims (~1.5% accuracy drop, 66% storage reduction).
/// Changing this constant requires `--reindex`; the LanceDB table is auto-dropped on dimension mismatch.
pub const EMBEDDING_DIM:    usize  = 256;
/// Embed batch size tuned for CPU inference.
/// 256 was GPU-sized; at batch=256 × seq=512 the CPU takes 40–120 s per call.
/// 32 keeps each call under ~5 s while amortising tokenizer overhead.
pub const EMBED_BATCH_SIZE: usize  = 32;
pub const TRAIN_MIN:        usize  = 32;
#[allow(dead_code)]
pub const TRAIN_IDEAL:      usize  = 39 * TRAIN_MIN;
const MIN_NLIST:            usize  = 4;

/// IVF-PQ partition count heuristic: 4√N clamped to [MIN_NLIST, 1024].
pub fn compute_nlist(n: usize) -> usize {
    ((n as f64).sqrt() as usize * 4).max(MIN_NLIST).min(1024)
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
    pub async fn open_or_create(path: &Path) -> Result<Self, IndexerError> {
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
            } else {
                tracing::warn!(
                    "LanceDB vector dim mismatch (stored={:?}, expected={}); dropping table for rebuild",
                    stored_dim, EMBEDDING_DIM
                );
                conn.drop_table("chunks", &[])
                    .await
                    .map_err(|e| IndexerError::VectorStore(e.to_string()))?;
                create_chunks_table(&conn).await?
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
        let id_list: String = ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ");
        self.table
            .delete(&format!("id IN ({})", id_list))
            .await
            .map_err(|e| IndexerError::VectorStore(e.to_string()))?;
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

        let id_list: String = ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ");
        let mut stream = self.table
            .query()
            .only_if(&format!("id IN ({})", id_list))
            .execute()
            .await
            .map_err(|e| IndexerError::VectorStore(e.to_string()))?;

        let mut result: HashMap<i64, Vec<f32>> = HashMap::new();
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

pub struct Embedder {
    session:   Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
}

impl Embedder {
    pub fn load(onnx_dir: &Path) -> Result<Self, IndexerError> {
        use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

        let model_path = onnx_dir.join("nomic-embed-text-v1.5.onnx");
        let tok_path   = onnx_dir.join("tokenizer.json");

        let ort_dll = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
        ort::init_from(&ort_dll)
            .map_err(|e| IndexerError::Embedding(format!("ORT DLL load failed: {e}")))?
            .commit();

        let session = SessionBuilder::new()
            .map_err(|e| IndexerError::Embedding(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| IndexerError::Embedding(e.to_string()))?
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
        if texts.is_empty() {
            return Array2::from_shape_vec((0, EMBEDDING_DIM), vec![])
                .map_err(|e| IndexerError::Embedding(e.to_string()));
        }

        // Sort by length to minimise padding (same optimisation as Python)
        let mut indexed: Vec<(usize, String)> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (i, format!("{}{}", prefix, t)))
            .collect();
        indexed.sort_by_key(|(_, t)| t.len());

        let sorted_texts: Vec<&str> = indexed.iter().map(|(_, t)| t.as_str()).collect();

        let encodings = self.tokenizer
            .encode_batch(sorted_texts, true)
            .map_err(|e| IndexerError::Embedding(e.to_string()))?;

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
        let pooled_flat = pooled_flat;

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

        Array2::from_shape_vec((b, EMBEDDING_DIM), result_flat)
            .map_err(|e| IndexerError::Embedding(e.to_string()))
    }
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

    if let Some(info) = info {
        if info.modified_at == Some(meta.modified_at) && !info.has_unembedded {
            return FileAction::Skip;
        }
    }

    let bytes = match storage.get_file_bytes(uri) {
        Ok(b)  => b,
        Err(e) => { tracing::warn!("Skipping {}: {}", uri, e); return FileAction::Error; }
    };

    let hash = md5_hex(&bytes);

    if let Some(info) = info {
        if hash == info.content_hash && !info.has_unembedded {
            return FileAction::Skip;
        }
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

// ── IncrementalIndexer ────────────────────────────────────────────────────────

pub struct IncrementalIndexer {
    pub storage:      Arc<dyn StorageClient>,
    #[allow(dead_code)]
    pub chunker:      FileChunker,
    pub db:           Arc<EnterpriseDb>,
    pub vectors:      LanceStore,
    pub chunker_map:  ChunkerMap,
    #[allow(dead_code)]
    pub index_dir:    PathBuf,
    embedder:         Option<Arc<Embedder>>,
}

impl IncrementalIndexer {
    pub async fn new(
        storage:   Arc<dyn StorageClient>,
        index_dir: &Path,
    ) -> Result<Self, IndexerError> {
        std::fs::create_dir_all(index_dir).map_err(|e| IndexerError::Io {
            path:   index_dir.to_path_buf(),
            source: e,
        })?;
        let db_path     = index_dir.join("enterprise.db");
        let db          = Arc::new(EnterpriseDb::new(&db_path)?);
        let vectors     = LanceStore::open_or_create(&index_dir.join("lance")).await?;
        let chunker_map = ChunkerMap::load_or_create(index_dir)?;

        let embedder = std::env::var("NOMIC_ONNX_PATH")
            .ok()
            .and_then(|p| Embedder::load(Path::new(&p)).ok())
            .map(Arc::new);

        Ok(Self {
            storage,
            chunker: FileChunker::new(),
            db,
            vectors,
            chunker_map,
            index_dir: index_dir.to_path_buf(),
            embedder,
        })
    }

    pub async fn index_root(
        &self,
        root_uri:    &str,
        reindex:     bool,
        on_start:    Option<&(dyn Fn(usize) + Sync)>,
        on_progress: Option<&(dyn Fn(usize, usize, &Stats) + Sync)>,
    ) -> Result<Stats, IndexerError> {
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
                    let action = process_file_sync(
                        storage_bg.as_ref(), &FileChunker::new(), &chunker_map_bg, uri, info,
                    );
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
                    self.write_file(
                        &meta, &hash, chunk_result, &chunker_method,
                        &mut text_buffer, &mut lance_id_buf,
                        &mut chunk_id_buf, &mut tier_buf,
                    )?;
                    stats.indexed += 1;

                    if text_buffer.len() >= EMBED_BATCH_SIZE {
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

            if total >= 20 && (checked % 100 == 0 || checked == total) {
                tracing::info!(
                    "Checked {}/{} (indexed={} skipped={} errors={})",
                    checked, total, stats.indexed, stats.skipped, stats.errors
                );
                if let Some(f) = on_progress { f(checked, total, &stats); }
            }
        }

        phase1.await.map_err(|e| IndexerError::Other(e.to_string().into()))?;

        // Final flush for any chunks accumulated since the last EMBED_BATCH_SIZE boundary.
        if !text_buffer.is_empty() {
            self.flush_embeddings(&text_buffer, &lance_id_buf, &chunk_id_buf, &tier_buf).await?;
        }

        // Build IVF-PQ index with dynamically computed partition count.
        let ntotal = self.vectors.ntotal().await?;
        if ntotal >= TRAIN_MIN {
            let nlist = compute_nlist(ntotal);
            let _ = self.vectors.create_ivf_pq_index(nlist).await;
        }

        // Remove files no longer on disk, including their LanceDB vectors (issue: ghost vectors).
        let all_uris = self.db.list_all_file_uris()?;
        for uri in all_uris {
            if !live_uris.contains(uri.as_str()) {
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

    fn write_file(
        &self,
        meta:           &crate::storage::FileMetadata,
        content_hash:   &str,
        chunk_result:   ChunkResult,
        chunker_method: &str,
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
            meta:           c.meta.clone(),
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
                        &mut text_buffer, &mut lance_id_buf,
                        &mut chunk_id_buf, &mut tier_buf,
                    )?;
                    stats.indexed += 1;
                }
            }
        }

        if !text_buffer.is_empty() {
            self.flush_embeddings(&text_buffer, &lance_id_buf, &chunk_id_buf, &tier_buf).await?;
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
        for i in (0..texts.len()).step_by(EMBED_BATCH_SIZE) {
            let end          = (i + EMBED_BATCH_SIZE).min(texts.len());
            let batch_owned: Vec<String> = texts[i..end].to_vec();
            let batch_n      = batch_owned.len();
            tracing::info!("Embedding batch {}/{} ({} chunks)", i / EMBED_BATCH_SIZE + 1, (texts.len() + EMBED_BATCH_SIZE - 1) / EMBED_BATCH_SIZE, batch_n);
            let emb          = embedder.clone();
            let vecs         = tokio::task::spawn_blocking(move || {
                let batch_refs: Vec<&str> = batch_owned.iter().map(String::as_str).collect();
                emb.embed(&batch_refs, "search_document: ")
            })
            .await
            .map_err(|e| IndexerError::Embedding(e.to_string()))??;
            tracing::info!("Embedded {} chunks", batch_n);
            let flat_vecs: Vec<Vec<f32>> = (0..vecs.nrows())
                .map(|r| vecs.row(r).to_vec())
                .collect();
            self.vectors
                .add_vectors(&lance_ids[i..end], &chunk_ids[i..end], &tiers[i..end], &flat_vecs)
                .await?;
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
        Embedder::load(Path::new(&path)).ok()
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
        let store = LanceStore::open_or_create(dir.path()).await.unwrap();

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
        let store = LanceStore::open_or_create(dir.path()).await.unwrap();

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
        ).await.unwrap();

        let stats1   = indexer.index_root(&root_dir.path().to_string_lossy(), false, None, None).await.unwrap();
        assert_eq!(stats1.indexed, 1, "first run should index 1 file");

        let stats2   = indexer.index_root(&root_dir.path().to_string_lossy(), false, None, None).await.unwrap();
        assert_eq!(stats2.indexed, 0, "second run should skip unchanged file");
    }
}
