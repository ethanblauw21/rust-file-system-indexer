mod chunker;
mod chunker_map;
mod db;
mod error;
mod indexer;
mod scorer;
mod search;
mod storage;
mod tui;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "file_indexer", about = "Hybrid file indexer + search")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Index files under a root directory
    Index {
        /// Root directory to index
        root: PathBuf,
        /// Index storage directory (default: .fileSystem-index)
        #[arg(long, default_value = ".fileSystem-index")]
        index_dir: PathBuf,
        /// Force re-index of all files regardless of mtime/hash
        #[arg(long)]
        reindex: bool,
        /// Folder names to exclude (may be repeated, e.g. --exclude AppData --exclude .rustup)
        #[arg(long)]
        exclude: Vec<String>,
    },
    /// Search the index
    Search {
        /// Query string (mutually exclusive with --like)
        query: Option<String>,
        /// Find files similar to this file instead of using a text query
        #[arg(long, value_name = "PATH")]
        like: Option<PathBuf>,
        /// Search mode: hybrid, dense, or sparse
        #[arg(long, default_value = "hybrid")]
        mode: String,
        /// Number of results
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// Candidate pool size fed into RRF (hybrid mode only)
        #[arg(long, default_value_t = 100)]
        candidate_pool: usize,
        /// Max chunks returned per source file
        #[arg(long, default_value_t = 3)]
        max_per_file: usize,
        /// Filter by tier (1, 2, or 3)
        #[arg(long)]
        tier: Option<u8>,
        /// Filter results to files with this extension (e.g. rs, py, md)
        #[arg(long = "type", value_name = "EXT")]
        file_type: Option<String>,
        /// Print full chunk content
        #[arg(long)]
        full: bool,
        /// Print index stats and exit
        #[arg(long)]
        stats: bool,
        /// Output format: table (default), json, or md
        #[arg(long, default_value = "table")]
        output: String,
        /// Index storage directory
        #[arg(long, default_value = ".fileSystem-index")]
        index_dir: PathBuf,
    },
    /// Launch the interactive TUI
    Tui {
        /// Index storage directory
        #[arg(long, default_value = ".fileSystem-index")]
        index_dir: PathBuf,
        /// Root directory for the index trigger (i key)
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// Print metadata, chunk structure, edges, and Tier-3 summary for an indexed file
    Explain {
        /// Path to the file (must be in the index)
        path: PathBuf,
        /// Index storage directory
        #[arg(long, default_value = ".fileSystem-index")]
        index_dir: PathBuf,
        /// Print full content of every chunk instead of one-line previews
        #[arg(long)]
        all_chunks: bool,
    },
    /// Score chunk quality using structural heuristics
    Score {
        /// Index storage directory
        #[arg(long, default_value = ".fileSystem-index")]
        index_dir: PathBuf,
        /// Re-score already-scored chunks (useful after changing thresholds)
        #[arg(long)]
        rescore: bool,
    },
    /// Display already-computed chunk quality scores (worst first)
    Scores {
        /// Index storage directory
        #[arg(long, default_value = ".fileSystem-index")]
        index_dir: PathBuf,
        /// Maximum chunks to show
        #[arg(long, default_value_t = 50)]
        limit: usize,
        /// Show only flagged chunks
        #[arg(long)]
        flagged_only: bool,
        /// Filter by tier (1, 2, or 3)
        #[arg(long)]
        tier: Option<u8>,
    },
    /// Reindex flagged files whose chunker method has changed
    Recheck {
        /// Index storage directory
        #[arg(long, default_value = ".fileSystem-index")]
        index_dir: PathBuf,
        /// Print what would be reindexed without doing it
        #[arg(long)]
        dry_run: bool,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("file_indexer=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Index { root, index_dir, reindex, exclude } => {
            run_index(root, index_dir, reindex, exclude).await;
        }
        Command::Search {
            query, like, mode, top_k, candidate_pool, max_per_file,
            tier, file_type, full, stats, output, index_dir,
        } => {
            run_search(
                query, like, mode, top_k, candidate_pool, max_per_file,
                tier, file_type, full, stats, output, index_dir,
            ).await;
        }
        Command::Explain { path, index_dir, all_chunks } => {
            run_explain(path, index_dir, all_chunks).await;
        }
        Command::Tui { index_dir, root } => {
            run_tui(index_dir, root).await;
        }
        Command::Score { index_dir, rescore } => {
            run_score(index_dir, rescore).await;
        }
        Command::Scores { index_dir, limit, flagged_only, tier } => {
            run_scores_show(index_dir, limit, flagged_only, tier).await;
        }
        Command::Recheck { index_dir, dry_run } => {
            run_recheck(index_dir, dry_run).await;
        }
    }
}

async fn run_index(root: PathBuf, index_dir: PathBuf, reindex: bool, exclude: Vec<String>) {
    use crate::indexer::{IncrementalIndexer, Stats};
    use crate::storage::LocalStorageClient;
    use std::sync::Arc;

    let root_str = root.to_string_lossy().to_string();
    tracing::info!("Indexing {} → {}", root_str, index_dir.display());
    if !exclude.is_empty() {
        tracing::info!("Excluding folders: {}", exclude.join(", "));
    }

    let storage = Arc::new(LocalStorageClient::with_extra_ignores(exclude));

    let t0 = std::time::Instant::now();

    let indexer = match IncrementalIndexer::new(storage, &index_dir).await {
        Ok(i) => i,
        Err(e) => { eprintln!("Error creating indexer: {}", e); return; }
    };

    let root_str_for_start = root_str.clone();
    let on_start = move |total: usize| {
        println!("Indexing {}  ({} files)", root_str_for_start, fmt_num(total as i64));
    };

    let on_progress = |checked: usize, total: usize, s: &Stats| {
        let width = fmt_num(total as i64).len();
        println!(
            "  [{:>width$} / {:>width$}]  indexed={:<8}  skipped={:<8}  errors={}",
            fmt_num(checked as i64),
            fmt_num(total as i64),
            fmt_num(s.indexed as i64),
            fmt_num(s.skipped as i64),
            s.errors,
            width = width,
        );
    };

    match indexer.index_root(&root_str, reindex, Some(&on_start), Some(&on_progress)).await {
        Ok(stats) => {
            println!();
            println!(
                "Done in {}  —  {} indexed, {} skipped, {} removed, {} errors",
                format_elapsed(t0.elapsed()),
                fmt_num(stats.indexed as i64),
                fmt_num(stats.skipped as i64),
                fmt_num(stats.removed as i64),
                fmt_num(stats.errors  as i64),
            );
            println!("  {} vectors in index", fmt_num(stats.vec_total as i64));
        }
        Err(e) => eprintln!("Indexing error: {}", e),
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_search(
    query:          Option<String>,
    like:           Option<PathBuf>,
    mode_str:       String,
    top_k:          usize,
    candidate_pool: usize,
    max_per_file:   usize,
    tier:           Option<u8>,
    file_type:      Option<String>,
    full:           bool,
    stats:          bool,
    output:         String,
    index_dir:      PathBuf,
) {
    use crate::search::{SearchMode, SearchOptions, Searcher};

    let searcher = match Searcher::new(&index_dir).await {
        Ok(s) => s,
        Err(e) => { eprintln!("Error opening index: {}", e); return; }
    };

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

                println!("\n  Chunks");
                println!("    {:>10}  total", fmt_num(s.db_stats.chunks));
                println!("    {:>10}  embedded  ({}%)", fmt_num(s.db_stats.chunks_embedded), pct);
                if pending > 0 {
                    println!("    {:>10}  pending", fmt_num(pending));
                }

                println!();
                println!("  {:>10}  edges",    fmt_num(s.db_stats.edges));
                println!("  {:>10}  FTS docs", fmt_num(s.db_stats.chunks_fts_docs));
                println!("  {:>10}  vectors",  fmt_num(s.vec_total as i64));
            }
            Err(e) => eprintln!("Stats error: {}", e),
        }
        return;
    }

    let mut mode = match mode_str.as_str() {
        "dense"  => SearchMode::Dense,
        "sparse" => SearchMode::Sparse,
        _        => SearchMode::Hybrid,
    };

    // Auto-fallback: if the embedder is not available, dense/hybrid silently degrade
    // into a hard error. Warn the user and switch to sparse instead.
    if searcher.embedder.is_none() && mode != SearchMode::Sparse && like.is_none() {
        eprintln!("warning: NOMIC_ONNX_PATH not set — falling back to sparse (FTS5) search");
        mode = SearchMode::Sparse;
    }

    let mode_label = if like.is_some() {
        "dense"
    } else {
        match mode {
            SearchMode::Hybrid => "hybrid",
            SearchMode::Dense  => "dense",
            SearchMode::Sparse => "sparse",
        }
    };

    let ext_filter = file_type.map(|t| t.trim_start_matches('.').to_lowercase());

    let opts = SearchOptions {
        mode,
        top_k,
        candidate_pool,
        max_per_file,
        tier,
        ext_filter,
    };

    // Capture query string before consuming it so we can label the results.
    let query_display = query.as_deref().map(str::to_string);

    // Resolve the result list — either --like or a text query.
    let results = if let Some(ref like_path) = like {
        if query.is_some() {
            eprintln!("Error: --like and a text query are mutually exclusive.");
            return;
        }
        match searcher.search_like(like_path, opts).await {
            Ok(r) => r,
            Err(e) => { eprintln!("Search error: {}", e); return; }
        }
    } else {
        let q = match query {
            Some(q) => q,
            None    => { eprintln!("Provide a query, --like <path>, or --stats"); return; }
        };
        match searcher.search(&q, opts).await {
            Ok(r) => r,
            Err(e) => { eprintln!("Search error: {}", e); return; }
        }
    };

    print_results(&results, full, &output, query_display.as_deref(), like.as_deref(), mode_label);
}

fn print_results(
    results:    &[crate::search::SearchResult],
    full:       bool,
    output:     &str,
    query:      Option<&str>,
    like_path:  Option<&std::path::Path>,
    mode_label: &str,
) {
    match output {
        "json" => print_json(results),
        "md"   => print_md(results, like_path),
        _      => print_table(results, full, query, like_path, mode_label),
    }
}

fn print_table(
    results:    &[crate::search::SearchResult],
    full:       bool,
    query:      Option<&str>,
    like_path:  Option<&std::path::Path>,
    mode_label: &str,
) {
    // Header line: what was searched and how many results
    let count = results.len();
    if let Some(q) = query {
        println!(
            "  {} result{} for \"{}\"  ·  {}\n",
            fmt_num(count as i64),
            if count == 1 { "" } else { "s" },
            q,
            mode_label,
        );
    } else if let Some(p) = like_path {
        let lossy = p.to_string_lossy();
        let name  = p.file_name().and_then(|n| n.to_str()).unwrap_or(&lossy);
        println!(
            "  {} file{} similar to \"{}\"  ·  {}\n",
            fmt_num(count as i64),
            if count == 1 { "" } else { "s" },
            name,
            mode_label,
        );
    }

    if results.is_empty() {
        println!("  No results found.");
        return;
    }

    println!(
        "  {:>3}  {:>6}  {:>4}  {:<40}  {}",
        "#", "Score", "Tier", "Source", "Preview"
    );
    println!("  {}", "─".repeat(96));

    for (i, r) in results.iter().enumerate() {
        let path     = std::path::Path::new(&r.chunk.file_uri);
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or(&r.chunk.file_uri);
        let parent   = path.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let source_raw = if parent.is_empty() {
            filename.to_string()
        } else {
            format!("{}/{}", parent, filename)
        };
        // Truncate to 40 display chars; add … so the cut is visible.
        let source: String = if source_raw.chars().count() > 40 {
            let cut = source_raw.char_indices().nth(39).map(|(i, _)| i).unwrap_or(source_raw.len());
            format!("{}…", &source_raw[..cut])
        } else {
            source_raw
        };

        let score_pct = (r.normalized_score * 100.0).round() as u32;

        let preview = if full {
            r.chunk.content.clone()
        } else {
            let first = r.chunk.content.lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("")
                .trim_start();
            let cutoff = first.char_indices().nth(60).map(|(i, _)| i).unwrap_or(first.len());
            if first.len() > cutoff {
                format!("{}…", first[..cutoff].trim_end())
            } else {
                first.to_string()
            }
        };

        println!(
            "  {:>3}  {:>5}%  T{:<1}   {:<40}  {}",
            i + 1, score_pct, r.chunk.tier, source, preview
        );
    }

    // Paths block: unique files in rank order so the user can open them directly.
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let unique: Vec<(usize, &str)> = results.iter().enumerate()
        .filter_map(|(i, r)| {
            if seen.insert(r.chunk.file_uri.as_str()) {
                Some((i + 1, r.chunk.file_uri.as_str()))
            } else {
                None
            }
        })
        .collect();

    println!("\n  Paths");
    let use_links = supports_hyperlinks();
    for (rank, uri) in &unique {
        let display = if use_links {
            hyperlink(&path_to_file_uri(uri), uri)
        } else {
            uri.to_string()
        };
        println!("    {}  {}", rank, display);
    }
}

fn print_json(results: &[crate::search::SearchResult]) {
    let json: Vec<serde_json::Value> = results.iter().enumerate().map(|(i, r)| {
        serde_json::json!({
            "rank":    i + 1,
            "score":   r.normalized_score,
            "tier":    r.chunk.tier,
            "file":    r.chunk.file_uri,
            "content": r.chunk.content,
        })
    }).collect();
    println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
}

fn print_md(results: &[crate::search::SearchResult], like_path: Option<&std::path::Path>) {
    if let Some(p) = like_path {
        println!("# Similar to: {}\n", p.display());
    } else {
        println!("# Search Results\n");
    }

    for (i, r) in results.iter().enumerate() {
        let filename = std::path::Path::new(&r.chunk.file_uri)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&r.chunk.file_uri);

        println!(
            "## {}. {} — Score: {:.4} | Tier {}\n",
            i + 1, filename, r.normalized_score, r.chunk.tier
        );
        println!("**File:** `{}`\n", r.chunk.file_uri);
        println!("```\n{}\n```\n", r.chunk.content);
        println!("---\n");
    }
}

async fn run_explain(path: PathBuf, index_dir: PathBuf, all_chunks: bool) {
    use crate::search::Searcher;

    let searcher = match Searcher::new(&index_dir).await {
        Ok(s) => s,
        Err(e) => { eprintln!("Error opening index: {}", e); return; }
    };

    let data = match searcher.explain_full(&path) {
        Ok(Some(d)) => d,
        Ok(None) => {
            eprintln!("No entry found for '{}'. The file may not be indexed.", path.display());
            return;
        }
        Err(e) => { eprintln!("Explain error: {}", e); return; }
    };

    let lossy    = path.to_string_lossy();
    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or(&lossy);
    let sep      = "─".repeat(72);

    // ── Header ────────────────────────────────────────────────────────────────
    println!("# {}", filename);
    println!("  {}\n", path.display());
    println!(
        "  {}  ·  {}  ·  Indexed: {}  ·  Modified: {}",
        format_bytes(data.detail.size_bytes),
        data.detail.mime_type,
        format_unix_ts(data.detail.indexed_at),
        data.detail.modified_at.map(format_unix_ts).unwrap_or_else(|| "—".to_string()),
    );
    // T1 tokens represent the actual file content without tier-overlap inflation.
    let t1_tokens: i64 = data.chunks.iter()
        .filter(|c| c.tier == 1)
        .filter_map(|c| c.token_count)
        .sum();
    println!(
        "  Chunks: {} T1 / {} T2 / {} T3  ·  ~{} tok",
        data.detail.t1_count, data.detail.t2_count, data.detail.t3_count,
        fmt_num(t1_tokens),
    );

    // ── Summary ───────────────────────────────────────────────────────────────
    if let Some(ref summary) = data.summary {
        println!("\n{}", sep);
        println!("Summary\n");
        println!("{}", summary);
    }

    // ── Chunk structure ───────────────────────────────────────────────────────
    if !data.chunks.is_empty() {
        println!("\n{}", sep);
        println!("Chunk Structure\n");
        for chunk in &data.chunks {
            let tok = chunk.token_count
                .map(|t| format!("{:>4} tok", t))
                .unwrap_or_else(|| "       —".to_string());
            // E = embedded in LanceDB, · = pending (lance_id IS NULL)
            let emb = if chunk.lance_id.is_some() { "E" } else { "·" };
            if all_chunks {
                println!("  T{}  #{:<3}  {}  {}  ", chunk.tier, chunk.chunk_index, tok, emb);
                println!("{}", chunk.content);
                println!();
            } else {
                let first  = chunk.content.lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("")
                    .trim_start();
                let cutoff = first.char_indices().nth(64).map(|(i, _)| i).unwrap_or(first.len());
                println!("  T{}  #{:<3}  {}  {}  {}", chunk.tier, chunk.chunk_index, tok, emb, &first[..cutoff]);
            }
        }
    }

    // ── Outgoing edges ────────────────────────────────────────────────────────
    println!("\n{}", sep);
    println!("References\n");
    if data.outgoing.is_empty() {
        println!("  (none)");
    } else {
        for e in &data.outgoing {
            let name = std::path::Path::new(&e.dst_uri)
                .file_name().and_then(|n| n.to_str()).unwrap_or(&e.dst_uri);
            println!("  →  {:<48}  {}", name, e.edge_type);
        }
    }

    // ── Incoming edges ────────────────────────────────────────────────────────
    println!("\n{}", sep);
    println!("Referenced By\n");
    if data.incoming.is_empty() {
        println!("  (none)");
    } else {
        for e in &data.incoming {
            let name = std::path::Path::new(&e.src_uri)
                .file_name().and_then(|n| n.to_str()).unwrap_or(&e.src_uri);
            println!("  ←  {:<48}  {}", name, e.edge_type);
        }
    }
}

async fn run_tui(index_dir: PathBuf, root: Option<PathBuf>) {
    use crate::search::Searcher;

    let searcher = match Searcher::new(&index_dir).await {
        Ok(s) => s,
        Err(e) => { eprintln!("Error opening index: {}", e); return; }
    };

    let mut app = tui::App::new(searcher, index_dir, root);
    if let Err(e) = app.run().await {
        eprintln!("TUI error: {}", e);
    }
}

async fn run_score(index_dir: PathBuf, rescore: bool) {
    use crate::scorer::score_all;
    use crate::search::Searcher;

    let searcher = match Searcher::new(&index_dir).await {
        Ok(s)  => s,
        Err(e) => { eprintln!("Error opening index: {}", e); return; }
    };

    match score_all(&searcher.db, &searcher.vectors, rescore, None).await {
        Ok(s) => println!(
            "Scored {} chunks — {} flagged ({} structural, {} coherence, {} both)",
            fmt_num(s.total           as i64),
            fmt_num(s.flagged         as i64),
            fmt_num(s.structural_only as i64),
            fmt_num(s.coherence_only  as i64),
            fmt_num(s.both            as i64),
        ),
        Err(e) => eprintln!("Score error: {}", e),
    }
}

async fn run_scores_show(
    index_dir:    PathBuf,
    limit:        usize,
    flagged_only: bool,
    tier:         Option<u8>,
) {
    use crate::search::Searcher;

    let searcher = match Searcher::new(&index_dir).await {
        Ok(s)  => s,
        Err(e) => { eprintln!("Error opening index: {}", e); return; }
    };

    let chunks = match searcher.db.get_scored_chunks(limit, flagged_only, tier) {
        Ok(c)  => c,
        Err(e) => { eprintln!("Error reading scores: {}", e); return; }
    };

    if chunks.is_empty() {
        println!("No scored chunks found. Run `score` first.");
        return;
    }

    let label = if flagged_only { " (flagged only)" } else { "" };
    println!(
        "  {} chunk{} — worst structural scores first{}\n",
        fmt_num(chunks.len() as i64),
        if chunks.len() == 1 { "" } else { "s" },
        label,
    );

    println!(
        "  {:>4}  {:>5}  {:>5}  {:>1}  {:<4}  {:<40}  {}",
        "#", "str", "coh", "F", "Tier", "Source", "Preview"
    );
    println!("  {}", "─".repeat(100));

    for (i, c) in chunks.iter().enumerate() {
        let path   = std::path::Path::new(&c.file_uri);
        let fname  = path.file_name().and_then(|n| n.to_str()).unwrap_or(&c.file_uri);
        let parent = path.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let src_raw = if parent.is_empty() {
            fname.to_string()
        } else {
            format!("{parent}/{fname}")
        };
        let source: String = if src_raw.chars().count() > 40 {
            let cut = src_raw.char_indices().nth(39).map(|(i, _)| i).unwrap_or(src_raw.len());
            format!("{}…", &src_raw[..cut])
        } else {
            src_raw
        };

        let str_s = c.structural_score
            .map(|s| format!("{:.2}", s))
            .unwrap_or_else(|| "  —  ".to_string());
        let coh_s = c.coherence_score
            .map(|s| format!("{:.2}", s))
            .unwrap_or_else(|| "  —  ".to_string());
        let flag  = if c.is_flagged { "⚠" } else { " " };

        let first = c.content.lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim_start();
        let cutoff = first.char_indices().nth(60).map(|(i, _)| i).unwrap_or(first.len());
        let preview = if first.len() > cutoff {
            format!("{}…", first[..cutoff].trim_end())
        } else {
            first.to_string()
        };

        println!(
            "  {:>4}  {:>5}  {:>5}  {}  T{:<3}  {:<40}  {}",
            i + 1, str_s, coh_s, flag, c.tier, source, preview
        );
    }
}

async fn run_recheck(index_dir: PathBuf, dry_run: bool) {
    use crate::chunker_map::ChunkerMap;
    use crate::indexer::IncrementalIndexer;
    use crate::scorer::score_all;
    use crate::storage::LocalStorageClient;
    use std::collections::HashMap;
    use std::sync::Arc;

    let map = match ChunkerMap::load_or_create(&index_dir) {
        Ok(m)  => m,
        Err(e) => { eprintln!("Error loading chunker map: {}", e); return; }
    };

    let storage = Arc::new(LocalStorageClient::new());
    let indexer = match IncrementalIndexer::new(storage, &index_dir).await {
        Ok(i)  => i,
        Err(e) => { eprintln!("Error opening index: {}", e); return; }
    };

    let flagged = match indexer.db.get_flagged_files_with_methods() {
        Ok(f)  => f,
        Err(e) => { eprintln!("Error querying flagged files: {}", e); return; }
    };

    if flagged.is_empty() {
        println!("No flagged chunks found.");
        return;
    }

    // Partition into drifted (method changed) vs unchanged.
    // drift_map: file_uri → (stored_method, current_method)
    let mut drift_map: HashMap<String, (String, String)> = HashMap::new();
    let mut skipped_uris: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (file_uri, mime_type, stored_method) in &flagged {
        let current = map.method_for(mime_type).to_string();
        if *stored_method != current {
            drift_map.entry(file_uri.clone())
                .or_insert_with(|| (stored_method.clone(), current));
        } else {
            skipped_uris.insert(file_uri.clone());
        }
    }
    // Don't count a file as "skipped" if it also has a drifted entry.
    let skipped: Vec<String> = skipped_uris.into_iter()
        .filter(|u| !drift_map.contains_key(u))
        .collect();

    // Tally drift counts by (old → new) for the report.
    let mut drift_counts: HashMap<(String, String), usize> = HashMap::new();
    for (_, (old, new)) in &drift_map {
        *drift_counts.entry((old.clone(), new.clone())).or_insert(0) += 1;
    }

    let drift_uris: Vec<String> = drift_map.keys().cloned().collect();

    if dry_run {
        if drift_uris.is_empty() {
            println!("Reindexed 0 files (no method drift detected)");
        } else {
            println!(
                "Would reindex {} file{} (method drift detected)",
                drift_uris.len(), if drift_uris.len() == 1 { "" } else { "s" }
            );
            let mut sorted: Vec<_> = drift_counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            for ((old, new), count) in &sorted {
                println!(
                    "  {} → {} : {} file{}",
                    old, new, count, if **count == 1 { "" } else { "s" }
                );
            }
        }
        println!(
            "Skipped {} file{} (flagged but method unchanged — manual review needed)",
            skipped.len(), if skipped.len() == 1 { "" } else { "s" }
        );
        return;
    }

    if drift_uris.is_empty() {
        println!("Reindexed 0 files (no method drift detected)");
        println!(
            "Skipped {} file{} (flagged but method unchanged — manual review needed)",
            skipped.len(), if skipped.len() == 1 { "" } else { "s" }
        );
        return;
    }

    match indexer.index_uris(&drift_uris).await {
        Ok(stats) => {
            if let Err(e) = score_all(&indexer.db, &indexer.vectors, false, None).await {
                eprintln!("Warning: scoring after recheck failed: {}", e);
            }

            println!(
                "Reindexed {} file{} (method drift detected)",
                stats.indexed, if stats.indexed == 1 { "" } else { "s" }
            );
            if stats.errors > 0 {
                println!("  {} error{}", stats.errors, if stats.errors == 1 { "" } else { "s" });
            }
            let mut sorted: Vec<_> = drift_counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            for ((old, new), count) in &sorted {
                println!(
                    "  {} → {} : {} file{}",
                    old, new, count, if **count == 1 { "" } else { "s" }
                );
            }
        }
        Err(e) => { eprintln!("Reindex error: {}", e); return; }
    }

    println!(
        "Skipped {} file{} (flagged but method unchanged — manual review needed)",
        skipped.len(), if skipped.len() == 1 { "" } else { "s" }
    );
}

fn fmt_num(n: i64) -> String {
    let s = n.abs().to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(','); }
        result.push(c);
    }
    let formatted: String = result.chars().rev().collect();
    if n < 0 { format!("-{}", formatted) } else { formatted }
}

fn format_bytes(bytes: i64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn format_elapsed(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{}.{}s", secs, d.subsec_millis() / 100)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Returns true when stdout is a real TTY (not piped/redirected).
/// On Windows, TTY implies Windows Terminal or VS Code — both support OSC 8.
fn supports_hyperlinks() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

/// Converts a raw Windows path to a file:// URI suitable for OSC 8.
fn path_to_file_uri(path: &str) -> String {
    // Replace backslashes with forward slashes, then percent-encode spaces.
    let forward = path.replace('\\', "/");
    let encoded = forward.replace(' ', "%20");
    format!("file:///{}", encoded)
}

/// Wraps `text` in an OSC 8 hyperlink pointing to `href`.
fn hyperlink(href: &str, text: &str) -> String {
    format!("\x1b]8;;{}\x1b\\{}\x1b]8;;\x1b\\", href, text)
}

fn format_unix_ts(ts: f64) -> String {
    // Days since Unix epoch → Gregorian date, no external crate needed.
    let secs      = ts as u64;
    let days      = secs / 86400;
    let time_secs = secs % 86400;
    let hh        = time_secs / 3600;
    let mm        = (time_secs % 3600) / 60;

    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    let z  = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y   = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp  = (5 * doy + 2) / 153;
    let d   = doy - (153 * mp + 2) / 5 + 1;
    let m   = if mp < 10 { mp + 3 } else { mp - 9 };
    let y   = if m <= 2 { y + 1 } else { y };

    format!("{:04}-{:02}-{:02} {:02}:{:02}", y, m, d, hh, mm)
}
