use crate::db::{ChunkRow, DbStats, EdgeRow, EnterpriseDb, FileDetail, FtsResult, IncomingEdge};
use crate::error::IndexerError;
use crate::indexer::{Embedder, LanceStore};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const RRF_K: f64 = 60.0;

/// Fusion weights. Dense (vector) is the stronger retriever on this corpus, so
/// the sparse (BM25) channel is down-weighted: an unweighted fusion let BM25
/// keyword noise from unrelated files dilute dense precision, so hybrid trailed
/// pure dense (dogfooding finding M7). Sparse still contributes — it rescues
/// exact-token and typo cases dense misses — just with less pull.
pub const RRF_DENSE_WEIGHT:  f64 = 1.0;
pub const RRF_SPARSE_WEIGHT: f64 = 0.4;
/// Weight for the filename/path-token match bonus. Content embedding alone buries
/// impl files under prose docs that describe the same concept (finding M8), and
/// misses literal basename lookups (M9). A file named after what the query asks
/// for is a strong relevance signal, so we add a bounded bonus proportional to how
/// much of the file's basename the query covers. Scaled to one rank-1 RRF channel.
pub const PATH_BOOST_WEIGHT: f64 = 1.5;

#[derive(Debug, Clone, Copy)]
pub struct FusionWeights {
    pub dense:  f64,
    pub sparse: f64,
    pub path:   f64,
}

impl FusionWeights {
    /// Resolve weights, allowing env overrides for tuning sweeps without a rebuild
    /// (`RRF_DENSE_WEIGHT` / `RRF_SPARSE_WEIGHT` / `PATH_BOOST_WEIGHT`).
    fn resolve() -> Self {
        let parse = |k: &str, default: f64| {
            std::env::var(k).ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(default)
        };
        Self {
            dense:  parse("RRF_DENSE_WEIGHT",  RRF_DENSE_WEIGHT),
            sparse: parse("RRF_SPARSE_WEIGHT", RRF_SPARSE_WEIGHT),
            path:   parse("PATH_BOOST_WEIGHT", PATH_BOOST_WEIGHT),
        }
    }
}

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

// ── Filename / path-token boost (M8 code-vs-docs, M9 filename handles) ─────────

/// Split a string into lowercase alphanumeric tokens of length ≥ 3. Short tokens
/// ("py", "md", "of", "a") are dropped as noise so file extensions and stop-words
/// don't contribute to filename matching. Symbol-bearing language names are
/// normalized first (see [`normalize_symbols`]) so `C++`/`C#` survive the split.
fn tokenize(s: &str) -> Vec<String> {
    normalize_symbols(s)
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

/// Map symbol-bearing language/tech names to stable alphanumeric tokens *before*
/// the generic alnum splitter runs. Without this, `C++` and `C#` collapse to the
/// 1-char token `c` (dropped by the ≥ 3 filter), so a query or filename about
/// those languages can never earn a path-coverage boost. Because both the query
/// and the file stem pass through here, the canonical forms always agree.
fn normalize_symbols(s: &str) -> String {
    s.to_ascii_lowercase()
        .replace("c++", " cpp ")
        .replace("c#", " csharp ")
        .replace("f#", " fsharp ")
        .replace(".net", " dotnet ")
}

/// Light stem match: equal, or a shared leading prefix of ≥ 5 chars (both tokens
/// ≥ 5 long). This folds `indexer`/`indexing` and `retriever`/`retrieval` together
/// without a full stemmer, while staying tight enough to avoid spurious matches.
fn token_matches(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if a.len() < 5 || b.len() < 5 {
        return false;
    }
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count() >= 5
}

/// Fraction of a file's basename tokens that the query covers, in [0.0, 1.0].
/// A file named after exactly what the query asks for (e.g. `incremental_indexer.py`
/// for "incremental indexing …") scores ~1.0; an unrelated name scores 0.0. Using
/// the *basename's* coverage (not the query's) keeps long queries from diluting a
/// short, on-point filename.
fn path_coverage(file_uri: &str, query_tokens: &[String]) -> f64 {
    let stem = Path::new(file_uri)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let file_tokens = tokenize(&stem);
    if file_tokens.is_empty() {
        return 0.0;
    }
    let matched = file_tokens
        .iter()
        .filter(|ft| query_tokens.iter().any(|qt| token_matches(ft, qt)))
        .count();
    matched as f64 / file_tokens.len() as f64
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
        .filter(|c| tier.is_none_or(|t| c.tier == t))
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

#[allow(clippy::too_many_arguments)]
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
    // Degrade gracefully: a sparse (FTS5) error or empty result must NOT discard the
    // dense channel. Fall back to dense-only fusion instead of aborting the whole query
    // (Defect 5 — `?` here previously zeroed out hybrid on any FTS5 syntax error).
    let sparse_res = sparse_search(query, pool, tier, ext_filter, db).unwrap_or_default();
    let dense_res  = dense_fut.await?;

    // Fuse the full candidate pool, then cap-per-file, *then* truncate to top_k.
    // Capping before truncation is essential: a single file whose chunks flood the
    // fused head (e.g. a path-boosted doc) would otherwise fill the top_k slots and
    // be cut down by the per-file cap afterwards, yielding far fewer than top_k
    // results and burying well-ranked candidates from other files.
    let mut results = rrf_fuse(dense_res, sparse_res, query, FusionWeights::resolve());
    post_process(&mut results, max_per_file);
    results.truncate(top_k);
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

/// Fuse the dense and sparse candidate lists with weighted RRF plus a filename
/// boost, returning **all** merged candidates sorted by descending score. The
/// caller is responsible for per-file capping and truncating to its `top_k` — see
/// `hybrid_search` for why capping must precede truncation.
pub fn rrf_fuse(
    dense:   Vec<SearchResult>,
    sparse:  Vec<SearchResult>,
    query:   &str,
    weights: FusionWeights,
) -> Vec<SearchResult> {
    let FusionWeights { dense: w_dense, sparse: w_sparse, path: w_path } = weights;
    let mut scores: HashMap<i64, f64>         = HashMap::new();
    let mut merged: HashMap<i64, SearchResult> = HashMap::new();

    for (rank, mut r) in dense.into_iter().enumerate() {
        let rank1 = rank + 1;
        r.dense_rank = Some(rank1);
        let delta = w_dense / (RRF_K + rank1 as f64);
        scores.entry(r.chunk_id).and_modify(|v| *v += delta).or_insert(delta);
        merged.entry(r.chunk_id).or_insert(r);
    }

    for (rank, r) in sparse.into_iter().enumerate() {
        let rank1 = rank + 1;
        let delta = w_sparse / (RRF_K + rank1 as f64);
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

    // Apply accumulated RRF scores, plus a filename/path-token bonus so a file
    // named after what the query asks for can climb out from under prose docs that
    // merely describe the same concept (M8) and so literal basename lookups surface
    // their file (M9). The bonus is scaled to one rank-1 RRF channel — a full
    // basename match is worth as much as topping a single retriever — and added
    // *before* truncation so a buried-but-well-named candidate can rise into top_k.
    let query_tokens = tokenize(query);
    let path_unit = w_path / (RRF_K + 1.0);
    for (cid, score) in &scores {
        if let Some(r) = merged.get_mut(cid) {
            let mut total = *score;
            if w_path > 0.0 && !query_tokens.is_empty() {
                total += path_unit * path_coverage(&r.chunk.file_uri, &query_tokens);
            }
            r.rrf_score = Some(total);
        }
    }

    let mut results: Vec<SearchResult> = merged.into_values().collect();
    results.sort_by(|a, b| {
        b.rrf_score.unwrap_or(0.0)
            .partial_cmp(&a.rrf_score.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Normalize: theoretical max is rank-1 in both weighted channels plus a full
    // filename match — (w_dense + w_sparse + w_path) / (k+1).
    let rrf_max = (w_dense + w_sparse + w_path) / (RRF_K + 1.0);
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
    fn path_coverage_matches_named_files() {
        // A query whose terms name the file scores ~1.0 (light stem folds
        // indexing→indexer, retrieval→retriever); an unrelated query scores 0.0.
        let q = tokenize("incremental indexing merkle tree drift detection");
        let cov = path_coverage("C:\\x\\src\\incremental_indexer.py", &q);
        assert!(cov >= 0.99, "named impl file should be fully covered, got {cov}");

        let q2 = tokenize("hybrid retrieval dense sparse fusion");
        assert!(path_coverage("C:\\x\\src\\hybrid_retriever.py", &q2) >= 0.99);

        // Literal basename lookup (M9) fully covers the file.
        let q3 = tokenize("incremental_indexer");
        assert!(path_coverage("C:\\x\\src\\incremental_indexer.py", &q3) >= 0.99);

        // Unrelated query gets no credit; extension alone never matches.
        let q4 = tokenize("quarterly budget spreadsheet");
        assert_eq!(path_coverage("C:\\x\\src\\incremental_indexer.py", &q4), 0.0);
        assert_eq!(path_coverage("C:\\x\\notes.py", &tokenize("python script")), 0.0);
    }

    #[test]
    fn symbol_languages_survive_tokenization() {
        // `C++`/`C#`/`F#`/`.NET` would otherwise vanish (1-char `c` dropped).
        assert_eq!(tokenize("C++"), vec!["cpp"]);
        assert_eq!(tokenize("C#"), vec!["csharp"]);
        assert_eq!(tokenize("F# parser"), vec!["fsharp", "parser"]);
        assert_eq!(tokenize("ASP.NET core"), vec!["asp", "dotnet", "core"]);

        // Query and filename normalize to the same canonical token, so a file
        // named after the language earns full path coverage.
        let q = tokenize("modern C++ memory model");
        assert!(path_coverage("C:\\x\\src\\cpp_notes.md", &q) >= 0.49);
    }

    #[test]
    fn token_matches_stems_lightly() {
        assert!(token_matches("indexer", "indexing"));   // shared prefix ≥5
        assert!(token_matches("retriever", "retrieval"));
        assert!(token_matches("hybrid", "hybrid"));       // exact
        assert!(!token_matches("index", "ledger"));       // unrelated
        assert!(!token_matches("cat", "car"));            // too short to fuzzy-match
    }

    #[test]
    fn path_boost_lifts_named_file_over_doc() {
        // A code file buried at dense-rank 5 whose name matches the query should
        // overtake a higher-ranked doc that only matches semantically, once the
        // path boost is applied.
        let mut doc  = make_result(1, 1, Some(1), None);       // rank-1 doc, no name match
        doc.chunk.file_uri = "C:\\x\\docs\\merkle-drift-notes.md".to_string();
        let mut code = make_result(2, 1, Some(5), None);       // rank-5 code, name matches
        code.chunk.file_uri = "C:\\x\\src\\incremental_indexer.py".to_string();

        let weights = FusionWeights { dense: 1.0, sparse: 0.4, path: 1.5 };
        let results = rrf_fuse(vec![doc, code], vec![], "incremental indexer drift", weights);
        assert_eq!(results[0].chunk_id, 2, "name-matching code file should rank first");
    }

    #[test]
    fn rrf_k_smoothing() {
        // Rank-1 item from a single channel should score 1/(60+1)
        let dense  = vec![make_result(1, 1, Some(1), None)];
        let sparse = vec![];
        let results = rrf_fuse(dense, sparse, "", FusionWeights { dense: 1.0, sparse: 1.0, path: 0.0 });
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
        let results = rrf_fuse(dense, sparse, "", FusionWeights { dense: 1.0, sparse: 1.0, path: 0.0 });
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

        let mut results = rrf_fuse(dense, sparse, "", FusionWeights { dense: 1.0, sparse: 1.0, path: 0.0 });
        results.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap());

        // both_id (rank-1 dense + rank-1 sparse) should outscore single_id (rank-2 sparse only)
        let both_score   = results.iter().find(|r| r.chunk_id == both_id).unwrap().rrf_score.unwrap();
        let single_score = results.iter().find(|r| r.chunk_id == single_id).unwrap().rrf_score.unwrap();
        assert!(both_score > single_score, "combined score {} should exceed single {}", both_score, single_score);
    }

    #[test]
    fn rrf_fuse_returns_sorted_candidates() {
        // rrf_fuse now returns ALL fused candidates, sorted by descending score;
        // truncation to top_k is the caller's job (post-cap). Verify both.
        let n     = 20usize;
        let top_k = 5usize;
        let dense: Vec<SearchResult> = (1..=n as i64)
            .map(|i| make_result(i, 1, Some(i as usize), None))
            .collect();
        let sparse: Vec<SearchResult> = vec![];
        let mut fused = rrf_fuse(dense, sparse, "", FusionWeights { dense: 1.0, sparse: 1.0, path: 0.0 });
        assert_eq!(fused.len(), n, "rrf_fuse should return all candidates, not truncate");
        let sorted = fused.windows(2).all(|w| w[0].rrf_score >= w[1].rrf_score);
        assert!(sorted, "results must be sorted by descending score");
        // Caller truncates to top_k.
        fused.truncate(top_k);
        assert_eq!(fused.len(), top_k);
        assert_eq!(fused[0].chunk_id, 1, "best (dense rank-1) candidate should lead");
    }

    #[test]
    fn dense_only_mode_no_sparse() {
        // Results from rrf_fuse with empty sparse have no sparse_score
        let dense  = vec![make_result(1, 1, Some(1), None)];
        let sparse = vec![];
        let results = rrf_fuse(dense, sparse, "", FusionWeights { dense: 1.0, sparse: 1.0, path: 0.0 });
        assert_eq!(results.len(), 1);
        assert!(results[0].sparse_score.is_none(), "dense-only result should have no sparse_score");
        assert!(results[0].dense_score.is_some(),  "dense-only result should have dense_score");
    }

    #[test]
    fn sparse_only_mode_no_dense() {
        let dense  = vec![];
        let sparse = vec![make_result(1, 1, None, Some(1))];
        let results = rrf_fuse(dense, sparse, "", FusionWeights { dense: 1.0, sparse: 1.0, path: 0.0 });
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
