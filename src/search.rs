use crate::db::{ChunkRow, DbStats, EdgeRow, EnterpriseDb, FileDetail, FtsResult, IncomingEdge};
use crate::error::IndexerError;
use crate::indexer::{Embedder, LanceStore};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const RRF_K: f64 = 60.0;

// ── Output types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk_id:         i64,
    pub chunk:            ChunkRow,
    pub rrf_score:        Option<f64>,
    pub dense_score:      Option<f32>,
    pub sparse_score:     Option<f64>,
    pub dense_rank:       Option<usize>,
    pub sparse_rank:      Option<usize>,
    /// Unified 0.0–1.0 confidence score regardless of search mode
    pub normalized_score: f64,
}

#[derive(Debug)]
pub struct ExplainData {
    pub detail:   FileDetail,
    pub summary:  Option<String>,
    pub chunks:   Vec<ChunkRow>,
    pub outgoing: Vec<EdgeRow>,
    pub incoming: Vec<IncomingEdge>,
}

#[derive(Debug, Clone)]
pub struct IndexStats {
    pub db_stats:  DbStats,
    pub vec_total: usize,
}

// ── Search mode ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SearchMode {
    Hybrid,
    Dense,
    Sparse,
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub mode:           SearchMode,
    pub top_k:          usize,
    pub tier:           Option<u8>,
    pub candidate_pool: usize,
    pub max_per_file:   usize,
    /// Lowercase file extension without the dot (e.g. "rs", "md"). Filters
    /// results to files with that extension before returning.
    pub ext_filter:     Option<String>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            mode:           SearchMode::Hybrid,
            top_k:          10,
            tier:           None,
            candidate_pool: 100,
            max_per_file:   3,
            ext_filter:     None,
        }
    }
}

// ── Searcher ──────────────────────────────────────────────────────────────────

pub struct Searcher {
    pub db:       Arc<EnterpriseDb>,
    pub vectors:  Arc<LanceStore>,
    pub embedder: Option<Arc<Embedder>>,
}

impl Searcher {
    pub async fn new(index_dir: &Path) -> Result<Self, IndexerError> {
        use crate::indexer::LanceStore;

        let db_path = index_dir.join("enterprise.db");
        let db      = Arc::new(EnterpriseDb::new(&db_path)?);
        let vectors = Arc::new(LanceStore::open_or_create(&index_dir.join("lance")).await?);

        let embedder = std::env::var("NOMIC_ONNX_PATH")
            .ok()
            .and_then(|p| Embedder::load(Path::new(&p)).ok())
            .map(Arc::new);

        Ok(Self { db, vectors, embedder })
    }

    pub async fn search(
        &self,
        query: &str,
        opts:  SearchOptions,
    ) -> Result<Vec<SearchResult>, IndexerError> {
        let ext = opts.ext_filter.as_deref();
        let embedder = self.embedder.as_deref()
            .ok_or_else(|| IndexerError::Embedding("NOMIC_ONNX_PATH not set — dense/hybrid search unavailable".into()));
        match opts.mode {
            SearchMode::Hybrid => {
                hybrid_search(
                    query, opts.top_k, opts.candidate_pool, opts.tier, opts.max_per_file,
                    ext, &self.vectors, &self.db, embedder?,
                ).await
            }
            SearchMode::Dense => {
                let mut res = dense_search(query, opts.top_k, opts.tier, ext, &self.vectors, &self.db, embedder?).await?;
                post_process(&mut res, opts.max_per_file);
                Ok(res)
            }
            SearchMode::Sparse => {
                let mut res = sparse_search(query, opts.top_k, opts.tier, ext, &self.db)?;
                post_process(&mut res, opts.max_per_file);
                Ok(res)
            }
        }
    }

    /// Find files semantically similar to the content of `path`.
    /// Reads up to 64 KiB of the file, embeds it, and returns the closest
    /// neighbours in the vector index (the source file itself is excluded).
    pub async fn search_like(
        &self,
        path: &Path,
        opts: SearchOptions,
    ) -> Result<Vec<SearchResult>, IndexerError> {
        let embedder = self.embedder.as_deref()
            .ok_or_else(|| IndexerError::Embedding("NOMIC_ONNX_PATH not set — --like requires the model".into()))?;

        // Read up to 64 KiB; the tokenizer truncates to 512 tokens anyway.
        let content = {
            use std::io::Read;
            let mut f = std::fs::File::open(path).map_err(|e| IndexerError::Io {
                path: path.to_path_buf(), source: e,
            })?;
            let mut buf = vec![0u8; 65536];
            let n = f.read(&mut buf).map_err(|e| IndexerError::Io {
                path: path.to_path_buf(), source: e,
            })?;
            buf.truncate(n);
            buf
        };
        let text = String::from_utf8_lossy(&content);

        let vecs = embedder.embed(&[text.as_ref()], "search_document: ")?;
        let query_vec: Vec<f32> = vecs.row(0).to_vec();

        let ext = opts.ext_filter.as_deref();
        let mut results = dense_search_by_vec(
            query_vec, opts.top_k + 1, opts.tier, ext, &self.vectors, &self.db,
        ).await?;

        // Remove the source file — it matches itself at ~1.0.
        let source_uri = path.to_string_lossy().to_string();
        results.retain(|r| r.chunk.file_uri != source_uri);
        results.truncate(opts.top_k);

        post_process(&mut results, opts.max_per_file);
        Ok(results)
    }

    /// Full explain: metadata, all chunks, T3 summary, outgoing + incoming edges.
    pub fn explain_full(&self, path: &Path) -> Result<Option<ExplainData>, IndexerError> {
        let file_uri = path.to_string_lossy().to_string();
        let detail = match self.db.get_file_detail(&file_uri)? {
            Some(d) => d,
            None    => return Ok(None),
        };
        let summary  = self.db.get_tier3_content(&file_uri)?;
        let chunks   = self.db.get_chunks_for_file(&file_uri)?;
        let outgoing = self.db.get_outgoing_edges(&file_uri)?;
        let incoming = self.db.get_incoming_edges(&file_uri)?;
        Ok(Some(ExplainData { detail, summary, chunks, outgoing, incoming }))
    }

    pub async fn stats(&self) -> Result<IndexStats, IndexerError> {
        let db_stats  = self.db.stats()?;
        let vec_total = self.vectors.ntotal().await?;
        Ok(IndexStats { db_stats, vec_total })
    }
}

// ── Extension filter helper ───────────────────────────────────────────────────

fn matches_ext(file_uri: &str, ext_filter: Option<&str>) -> bool {
    match ext_filter {
        None      => true,
        Some(ext) => std::path::Path::new(file_uri)
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .as_deref()
            == Some(ext),
    }
}

// ── Dense search ──────────────────────────────────────────────────────────────

pub async fn dense_search(
    query:      &str,
    top_k:      usize,
    tier:       Option<u8>,
    ext_filter: Option<&str>,
    vectors:    &LanceStore,
    db:         &EnterpriseDb,
    embedder:   &Embedder,
) -> Result<Vec<SearchResult>, IndexerError> {
    let vecs      = embedder.embed(&[query], "search_query: ")?;
    let query_vec: Vec<f32> = vecs.row(0).to_vec();
    dense_search_by_vec(query_vec, top_k, tier, ext_filter, vectors, db).await
}

async fn dense_search_by_vec(
    query_vec:  Vec<f32>,
    top_k:      usize,
    tier:       Option<u8>,
    ext_filter: Option<&str>,
    vectors:    &LanceStore,
    db:         &EnterpriseDb,
) -> Result<Vec<SearchResult>, IndexerError> {
    let lance_results = vectors.search(query_vec, top_k * 2).await?;
    if lance_results.is_empty() {
        return Ok(vec![]);
    }

    let lance_ids: Vec<i64> = lance_results.iter().map(|r| r.id).collect();
    let score_map: HashMap<i64, f32> = lance_results.iter().map(|r| (r.id, r.score)).collect();

    let chunk_map = db.get_chunks_by_lance_ids(&lance_ids)?;

    let mut results: Vec<SearchResult> = chunk_map
        .into_values()
        .filter(|c| tier.map_or(true, |t| c.tier == t))
        .filter(|c| matches_ext(&c.file_uri, ext_filter))
        .map(|chunk| {
            let dense_score = chunk.lance_id.and_then(|lid| score_map.get(&lid).copied());
            SearchResult {
                chunk_id:         chunk.id,
                chunk,
                rrf_score:        None,
                dense_score,
                sparse_score:     None,
                dense_rank:       None,
                sparse_rank:      None,
                normalized_score: 0.0,
            }
        })
        .collect();

    // Sort by dense_score ascending (smaller L2 distance = better)
    results.sort_by(|a, b| {
        a.dense_score.unwrap_or(f32::MAX)
            .partial_cmp(&b.dense_score.unwrap_or(f32::MAX))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);

    // Assign ranks and normalize scores (cosine similarity from L2 distance on unit vectors)
    for (i, r) in results.iter_mut().enumerate() {
        r.dense_rank = Some(i + 1);
        if let Some(dist) = r.dense_score {
            r.normalized_score = (1.0 - (dist * dist / 2.0) as f64).clamp(0.0, 1.0);
        }
    }

    Ok(results)
}

// ── Sparse search ─────────────────────────────────────────────────────────────

pub fn sparse_search(
    query:      &str,
    top_k:      usize,
    tier:       Option<u8>,
    ext_filter: Option<&str>,
    db:         &EnterpriseDb,
) -> Result<Vec<SearchResult>, IndexerError> {
    let mut fts: Vec<FtsResult> = db.fts_search(query, top_k * 2)?;

    if let Some(t) = tier {
        fts.retain(|r| r.tier == t);
    }
    if ext_filter.is_some() {
        fts.retain(|r| matches_ext(&r.file_uri, ext_filter));
    }
    fts.truncate(top_k);

    let mut results: Vec<SearchResult> = fts
        .into_iter()
        .enumerate()
        .map(|(i, r)| SearchResult {
            chunk_id:         r.id,
            chunk: ChunkRow {
                id:               r.id,
                file_id:          r.file_id,
                tier:             r.tier,
                chunk_index:      0,
                content:          r.content,
                token_count:      None,
                meta:             r.meta,
                lance_id:         None,
                file_uri:         r.file_uri,
                mime_type:        String::new(),
                is_flagged:       false,
                chunker_method:   None,
                structural_score: None,
                coherence_score:  None,
            },
            rrf_score:        None,
            dense_score:      None,
            sparse_score:     Some(r.bm25_score),
            dense_rank:       None,
            sparse_rank:      Some(i + 1),
            normalized_score: 0.0,
        })
        .collect();

    // Normalize: best BM25 result (most negative) maps to 1.0
    let max_abs = results
        .iter()
        .filter_map(|r| r.sparse_score)
        .map(|s| s.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    for r in &mut results {
        r.normalized_score = r.sparse_score.map(|s| s.abs() / max_abs).unwrap_or(0.0);
    }

    Ok(results)
}

// ── Hybrid search (RRF) ───────────────────────────────────────────────────────

pub async fn hybrid_search(
    query:          &str,
    top_k:          usize,
    candidate_pool: usize,
    tier:           Option<u8>,
    max_per_file:   usize,
    ext_filter:     Option<&str>,
    vectors:        &LanceStore,
    db:             &EnterpriseDb,
    embedder:       &Embedder,
) -> Result<Vec<SearchResult>, IndexerError> {
    // Use a deeper candidate pool for both channels so RRF has enough candidates to fuse
    let pool = candidate_pool.max(top_k);
    let dense_fut  = dense_search(query, pool, tier, ext_filter, vectors, db, embedder);
    let sparse_res = sparse_search(query, pool, tier, ext_filter, db)?;
    let dense_res  = dense_fut.await?;

    let mut results = rrf_fuse(dense_res, sparse_res, top_k);
    post_process(&mut results, max_per_file);
    Ok(results)
}

// ── Post-processing: tier dedup + per-file cap ────────────────────────────────

/// Drop T3 chunks from any file that also has T1/T2 chunks in the result set,
/// then cap the number of chunks returned per source file to `max_per_file`.
/// Results must already be sorted by descending relevance before calling this.
fn post_process(results: &mut Vec<SearchResult>, max_per_file: usize) {
    // &str slices into results; pre-compute keep flags before retain to release the borrow
    let has_specific: std::collections::HashSet<&str> = results
        .iter()
        .filter(|r| r.chunk.tier < 3)
        .map(|r| r.chunk.file_uri.as_str())
        .collect();
    let keep: Vec<bool> = results
        .iter()
        .map(|r| !(r.chunk.tier == 3 && has_specific.contains(r.chunk.file_uri.as_str())))
        .collect();
    drop(has_specific);

    let mut keep_iter = keep.into_iter();
    results.retain(|_| keep_iter.next().unwrap_or(true));

    // Cap per source file (results are already sorted by score, so we keep the best)
    let mut per_file: HashMap<String, usize> = HashMap::new();
    results.retain(|r| {
        let count = per_file.entry(r.chunk.file_uri.clone()).or_insert(0);
        *count += 1;
        *count <= max_per_file
    });
}

// ── RRF fusion ────────────────────────────────────────────────────────────────

pub fn rrf_fuse(
    dense:  Vec<SearchResult>,
    sparse: Vec<SearchResult>,
    top_k:  usize,
) -> Vec<SearchResult> {
    let mut scores: HashMap<i64, f64>         = HashMap::new();
    let mut merged: HashMap<i64, SearchResult> = HashMap::new();

    for (rank, mut r) in dense.into_iter().enumerate() {
        let rank1 = rank + 1;
        r.dense_rank = Some(rank1);
        let delta = 1.0 / (RRF_K + rank1 as f64);
        scores.entry(r.chunk_id).and_modify(|v| *v += delta).or_insert(delta);
        merged.entry(r.chunk_id).or_insert(r);
    }

    for (rank, r) in sparse.into_iter().enumerate() {
        let rank1 = rank + 1;
        let delta = 1.0 / (RRF_K + rank1 as f64);
        scores.entry(r.chunk_id).and_modify(|v| *v += delta).or_insert(delta);
        merged.entry(r.chunk_id)
            .and_modify(|existing| {
                // Merge sparse fields into the already-stored dense result
                existing.sparse_score = r.sparse_score;
                existing.sparse_rank  = Some(rank1);
            })
            .or_insert_with(|| {
                let mut nr = r;
                nr.sparse_rank = Some(rank1);
                nr
            });
    }

    // Apply accumulated RRF scores
    for (cid, score) in &scores {
        if let Some(r) = merged.get_mut(cid) {
            r.rrf_score = Some(*score);
        }
    }

    let mut results: Vec<SearchResult> = merged.into_values().collect();
    results.sort_by(|a, b| {
        b.rrf_score.unwrap_or(0.0)
            .partial_cmp(&a.rrf_score.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);

    // Normalize RRF scores: theoretical max is 2/(k+1) when rank-1 in both channels
    let rrf_max = 2.0 / (RRF_K + 1.0);
    for r in &mut results {
        r.normalized_score = r.rrf_score.map(|s| (s / rrf_max).clamp(0.0, 1.0)).unwrap_or(0.0);
    }

    results
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{ChunkInput, EnterpriseDb};
    use tempfile::TempDir;

    fn make_chunk_row(id: i64, tier: u8, content: &str, lance_id: Option<i64>) -> ChunkRow {
        ChunkRow {
            id,
            file_id:          1,
            tier,
            chunk_index:      0,
            content:          content.to_string(),
            token_count:      None,
            meta:             serde_json::Value::Object(Default::default()),
            lance_id,
            file_uri:         "file:///test.txt".to_string(),
            mime_type:        String::new(),
            is_flagged:       false,
            chunker_method:   None,
            structural_score: None,
            coherence_score:  None,
        }
    }

    fn make_result(chunk_id: i64, tier: u8, dense_rank: Option<usize>, sparse_rank: Option<usize>) -> SearchResult {
        SearchResult {
            chunk_id,
            chunk:            make_chunk_row(chunk_id, tier, "content", Some(chunk_id)),
            rrf_score:        None,
            dense_score:      if dense_rank.is_some() { Some(0.1) } else { None },
            sparse_score:     if sparse_rank.is_some() { Some(-1.0) } else { None },
            dense_rank,
            sparse_rank,
            normalized_score: 0.0,
        }
    }

    #[test]
    fn rrf_k_smoothing() {
        // Rank-1 item from a single channel should score 1/(60+1)
        let dense  = vec![make_result(1, 1, Some(1), None)];
        let sparse = vec![];
        let results = rrf_fuse(dense, sparse, 10);
        assert_eq!(results.len(), 1);
        let score = results[0].rrf_score.unwrap();
        let expected = 1.0 / (RRF_K + 1.0);
        assert!((score - expected).abs() < 1e-12, "score={} expected={}", score, expected);
    }

    #[test]
    fn rrf_fuse_deduplicates() {
        // Same chunk_id appearing in both channels should appear once
        let dense  = vec![make_result(42, 1, Some(1), None)];
        let sparse = vec![make_result(42, 1, None, Some(1))];
        let results = rrf_fuse(dense, sparse, 10);
        assert_eq!(results.len(), 1, "deduplication failed");
        assert_eq!(results[0].chunk_id, 42);
    }

    #[test]
    fn rrf_fuse_combines_both_channels() {
        // Item in both channels gets higher score than item in only one
        let both_id   = 1i64;
        let single_id = 2i64;

        // both_id appears rank-1 in dense; single_id appears rank-1 in sparse only
        let dense  = vec![make_result(both_id, 1, Some(1), None)];
        let sparse = vec![
            make_result(both_id, 1, None, Some(1)),
            make_result(single_id, 1, None, Some(2)),
        ];

        let mut results = rrf_fuse(dense, sparse, 10);
        results.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap());

        // both_id (rank-1 dense + rank-1 sparse) should outscore single_id (rank-2 sparse only)
        let both_score   = results.iter().find(|r| r.chunk_id == both_id).unwrap().rrf_score.unwrap();
        let single_score = results.iter().find(|r| r.chunk_id == single_id).unwrap().rrf_score.unwrap();
        assert!(both_score > single_score, "combined score {} should exceed single {}", both_score, single_score);
    }

    #[test]
    fn hybrid_search_returns_top_k() {
        // Build a list of N results and verify truncation to top_k
        let n     = 20usize;
        let top_k = 5usize;
        let dense: Vec<SearchResult> = (1..=n as i64)
            .map(|i| make_result(i, 1, Some(i as usize), None))
            .collect();
        let sparse: Vec<SearchResult> = vec![];
        let fused = rrf_fuse(dense, sparse, top_k);
        assert!(fused.len() <= top_k, "expected at most {} results, got {}", top_k, fused.len());
    }

    #[test]
    fn dense_only_mode_no_sparse() {
        // Results from rrf_fuse with empty sparse have no sparse_score
        let dense  = vec![make_result(1, 1, Some(1), None)];
        let sparse = vec![];
        let results = rrf_fuse(dense, sparse, 10);
        assert_eq!(results.len(), 1);
        assert!(results[0].sparse_score.is_none(), "dense-only result should have no sparse_score");
        assert!(results[0].dense_score.is_some(),  "dense-only result should have dense_score");
    }

    #[test]
    fn sparse_only_mode_no_dense() {
        let dense  = vec![];
        let sparse = vec![make_result(1, 1, None, Some(1))];
        let results = rrf_fuse(dense, sparse, 10);
        assert_eq!(results.len(), 1);
        assert!(results[0].dense_score.is_none(),  "sparse-only result should have no dense_score");
        assert!(results[0].sparse_score.is_some(), "sparse-only result should have sparse_score");
    }

    #[test]
    fn tier_filter_excludes_other_tiers() {
        let results: Vec<SearchResult> = vec![
            make_result(1, 1, Some(1), None),
            make_result(2, 2, Some(2), None),
            make_result(3, 3, Some(3), None),
        ];
        // Filter to tier 1 only
        let filtered: Vec<SearchResult> = results.into_iter()
            .filter(|r| r.chunk.tier == 1)
            .collect();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].chunk.tier, 1);
    }

    #[test]
    fn id_mapping_invariant() {
        // For every SearchResult, chunk.lance_id should equal the id used in dense search
        let chunk_id = 99i64;
        let lance_id = 12345i64;
        let r = SearchResult {
            chunk_id,
            chunk:            make_chunk_row(chunk_id, 1, "content", Some(lance_id)),
            rrf_score:        Some(0.1),
            dense_score:      Some(0.5),
            sparse_score:     None,
            dense_rank:       Some(1),
            sparse_rank:      None,
            normalized_score: 0.0,
        };
        assert_eq!(
            r.chunk.lance_id.unwrap(), lance_id,
            "lance_id stored in chunk must equal the LanceDB search id"
        );
    }

    #[tokio::test]
    async fn sparse_search_via_db() {
        let dir = TempDir::new().unwrap();
        let db  = EnterpriseDb::new(&dir.path().join("test.db")).unwrap();

        // Insert a file and a chunk
        let file_id = db.upsert_file("file:///test.txt", "text/plain", "abc", 100, 0.0, None).unwrap();
        db.insert_chunks(file_id, &[ChunkInput {
            tier:           1,
            chunk_index:    0,
            content:        "the quick brown fox jumps".to_string(),
            token_count:    Some(5),
            meta:           serde_json::json!({}),
            chunker_method: None,
        }]).unwrap();

        let results = sparse_search("quick fox", 10, None, None, &db).unwrap();
        assert!(!results.is_empty(), "FTS should find 'quick fox'");
        assert!(results[0].sparse_score.is_some());
        assert_eq!(results[0].sparse_rank, Some(1));
    }

    #[test]
    fn ext_filter_matches_extension() {
        assert!(matches_ext("C:\\Users\\edb\\file.rs", Some("rs")));
        assert!(matches_ext("/home/user/notes.md", Some("md")));
        assert!(!matches_ext("/home/user/notes.md", Some("rs")));
        assert!(matches_ext("/any/path/file.txt", None));
    }

    #[test]
    fn ext_filter_case_insensitive_path() {
        // Extension comparison uses to_lowercase so .RS and .rs both match "rs"
        assert!(matches_ext("C:\\Users\\edb\\FILE.RS", Some("rs")));
    }
}
