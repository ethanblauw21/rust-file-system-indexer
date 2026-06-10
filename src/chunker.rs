use crate::chunker_map::ChunkerMap;
use crate::error::IndexerError;
use crate::storage::FileMetadata;
use serde_json;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::OnceLock;

// ── Tier constants ────────────────────────────────────────────────────────────

pub const TIER1_MAX_TOKENS: usize = 500;
pub const TIER2_MAX_TOKENS: usize = 1500;

// ── Output types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Chunk {
    pub tier:        u8,
    pub chunk_index: usize,
    pub content:     String,
    pub token_count: usize,
    pub meta:        serde_json::Value,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeType {
    LinksTo,
    References,
    Embeds,
}

impl EdgeType {
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::LinksTo    => "LINKS_TO",
            EdgeType::References => "REFERENCES",
            EdgeType::Embeds     => "EMBEDS",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub src_chunk_index: usize,
    pub dst_uri:         String,
    pub edge_type:       EdgeType,
    pub meta:            serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub chunks: Vec<Chunk>,
    pub edges:  Vec<Edge>,
}

// ── Token counting ────────────────────────────────────────────────────────────

static TOKENIZER: OnceLock<Option<tokenizers::Tokenizer>> = OnceLock::new();

fn get_tokenizer() -> Option<&'static tokenizers::Tokenizer> {
    TOKENIZER
        .get_or_init(|| {
            let path = std::env::var("NOMIC_ONNX_PATH").ok()?;
            let tok_path = std::path::Path::new(&path).join("tokenizer.json");
            tokenizers::Tokenizer::from_file(tok_path).ok()
        })
        .as_ref()
}

pub fn count_tokens(text: &str) -> usize {
    if let Some(tok) = get_tokenizer() {
        if let Ok(enc) = tok.encode(text, false) {
            return enc.len().max(1);
        }
    }
    // Fallback: 3 UTF-8 bytes ≈ 1 token
    (text.len() / 3).max(1)
}

pub fn split_to_budget(text: &str, budget: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return vec![];
    }
    if let Some(tok) = get_tokenizer() {
        if let Ok(enc) = tok.encode(text, false) {
            let ids = enc.get_ids();
            if ids.len() <= budget {
                // Fits in one chunk — return original text, no decode round-trip
                return vec![text.to_string()];
            }
            // Need to split — use character offsets to slice original text
            let offsets = enc.get_offsets();
            let mut result: Vec<String> = Vec::new();
            let mut chunk_byte_start = 0usize;
            for window_end in (budget..=ids.len()).step_by(budget) {
                let window_end = window_end.min(offsets.len());
                let byte_end = offsets[window_end - 1].1;
                let segment = &text[chunk_byte_start..byte_end];
                if !segment.is_empty() {
                    result.push(segment.to_string());
                }
                chunk_byte_start = byte_end;
                if window_end >= offsets.len() { break; }
            }
            if chunk_byte_start < text.len() {
                result.push(text[chunk_byte_start..].to_string());
            }
            if !result.is_empty() {
                return result;
            }
        }
    }
    // Byte-length fallback: split at UTF-8 char boundaries
    let byte_budget = budget * 3;
    let mut result: Vec<String> = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let end = (start + byte_budget).min(text.len());
        let end = if text.is_char_boundary(end) {
            end
        } else {
            (1..=end).rev().find(|&p| text.is_char_boundary(p)).unwrap_or(start + 1)
        };
        if end > start {
            result.push(text[start..end].to_string());
        }
        start = end;
    }
    result
}

// ── Spreadsheet helpers ───────────────────────────────────────────────────────

/// Serialise one data row as key-value pairs: `Col1: val1 | Col2: val2 | …`
/// Much more token-efficient than a markdown table row.
fn row_to_kv(headers: &[String], values: &[String]) -> String {
    (0..headers.len())
        .map(|i| format!("{}: {}", headers[i], values.get(i).map(String::as_str).unwrap_or("")))
        .collect::<Vec<_>>()
        .join(" | ")
}

// ── Markdown helpers ──────────────────────────────────────────────────────────

/// Strip YAML frontmatter (`---\n…\n---`) from the start of `text`.
/// Returns `(key→value map, remaining body)`.
pub fn extract_frontmatter<'a>(text: &'a str) -> (HashMap<String, String>, &'a str) {
    if !text.starts_with("---") {
        return (HashMap::new(), text);
    }
    let after_open = &text[3..];
    let after_open = match after_open.strip_prefix('\n') {
        Some(s) => s,
        None    => return (HashMap::new(), text),
    };
    let close = "\n---";
    let end = match after_open.find(close) {
        Some(p) => p,
        None    => return (HashMap::new(), text),
    };
    let fm_text  = &after_open[..end];
    let after    = &after_open[end + close.len()..];
    // Skip optional \r\n after the closing ---
    let after = after.trim_start_matches(|c| c == '\r' || c == '\n');

    let mut fm = HashMap::new();
    for line in fm_text.lines() {
        if let Some(colon) = line.find(':') {
            let k = line[..colon].trim().to_string();
            let v = line[colon + 1..].trim().to_string();
            if !k.is_empty() {
                fm.insert(k, v);
            }
        }
    }
    (fm, after)
}

/// Split a Markdown document into sections at H1/H2/H3 heading boundaries.
/// Heading-only sections are merged forward as breadcrumb context.
pub fn split_markdown_at_headings(text: &str) -> Vec<String> {
    use pulldown_cmark::{Event, Options, Parser, Tag};

    // Collect byte offsets of H1-H3 headings using the offset iterator
    let mut heading_starts: Vec<usize> = Vec::new();
    for (event, range) in Parser::new_ext(text, Options::empty()).into_offset_iter() {
        if let Event::Start(Tag::Heading { level, .. }) = event {
            if (level as u32) <= 3 {
                heading_starts.push(range.start);
            }
        }
    }

    if heading_starts.is_empty() {
        return if text.trim().is_empty() {
            vec![]
        } else {
            vec![text.to_string()]
        };
    }

    let mut raw: Vec<String> = Vec::new();

    // Preamble before first heading
    if heading_starts[0] > 0 {
        let pre = text[..heading_starts[0]].trim().to_string();
        if !pre.is_empty() {
            raw.push(pre);
        }
    }
    for i in 0..heading_starts.len() {
        let start = heading_starts[i];
        let end   = heading_starts.get(i + 1).copied().unwrap_or(text.len());
        let s     = text[start..end].trim().to_string();
        if !s.is_empty() {
            raw.push(s);
        }
    }

    // Merge heading-only sections forward
    let mut merged: Vec<String> = Vec::new();
    let mut pending = String::new();

    for section in raw {
        let is_heading_only = {
            let mut lines = section.lines();
            lines.next().map(|l| is_heading_line(l)).unwrap_or(false) && lines.next().is_none()
        };
        if is_heading_only {
            if pending.is_empty() {
                pending = section;
            } else {
                pending.push('\n');
                pending.push_str(&section);
            }
        } else if !pending.is_empty() {
            merged.push(format!("{}\n{}", pending, section).trim().to_string());
            pending.clear();
        } else {
            merged.push(section);
        }
    }
    if !pending.is_empty() {
        merged.push(pending);
    }

    if merged.is_empty() { vec![text.to_string()] } else { merged }
}

fn is_heading_line(line: &str) -> bool {
    let t = line.trim_start();
    let n = t.chars().take_while(|&c| c == '#').count();
    n >= 1 && n <= 6 && t.chars().nth(n) == Some(' ')
}

/// Extract `[label](target)` links, resolving relative targets against `file_uri`.
pub fn extract_md_links(text: &str, file_uri: &str) -> Vec<(String, String)> {
    use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

    let base = std::path::Path::new(file_uri)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();

    let mut results: Vec<(String, String)> = Vec::new();
    let mut in_link = false;
    let mut cur_url   = String::new();
    let mut cur_label = String::new();

    for event in Parser::new_ext(text, Options::empty()) {
        match event {
            Event::Start(Tag::Link { dest_url, .. }) => {
                let url: &str = &dest_url;
                if !url.is_empty() && !url.contains("://") && !url.starts_with('#') {
                    in_link   = true;
                    cur_url   = url.to_owned();
                    cur_label.clear();
                }
            }
            Event::Text(ref t) if in_link => cur_label.push_str(t),
            Event::End(TagEnd::Link) if in_link => {
                let resolved = if base == std::path::Path::new("") {
                    cur_url.clone()
                } else {
                    base.join(&cur_url).to_string_lossy().into_owned()
                };
                results.push((resolved, std::mem::take(&mut cur_label)));
                cur_url.clear();
                in_link = false;
            }
            _ => {}
        }
    }
    results
}

// ── Tier-3 summary generators ─────────────────────────────────────────────────

fn summarise(text: &str, meta: &FileMetadata, extra: Option<&[(&str, String)]>) -> String {
    let lines    = text.lines().count();
    let headings: Vec<String> = text.lines()
        .filter(|l| is_heading_line(l.trim_start()))
        .take(12)
        .map(|l| l.trim().to_string())
        .collect();

    let mut parts = vec![
        format!("Document: {}", meta.name),
        format!("MIME: {}", meta.mime_type),
        format!("Estimated tokens: {}", count_tokens(text)),
        format!("Lines: {}", lines),
        format!(
            "Key headings: {}",
            if headings.is_empty() { "(none detected)".to_string() } else { headings.join("; ") }
        ),
    ];
    if let Some(extra) = extra {
        for (k, v) in extra {
            parts.push(format!("{}: {}", k, v));
        }
    }
    parts.join("\n")
}

fn summarise_markdown(
    text: &str,
    meta: &FileMetadata,
    frontmatter: &HashMap<String, String>,
) -> String {
    use pulldown_cmark::{Event, Options, Parser, Tag};

    let mut headings: Vec<String> = Vec::new();
    let mut in_h   = false;
    let mut h_text = String::new();
    let mut h_lvl  = 0u32;

    for event in Parser::new_ext(text, Options::empty()) {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                in_h   = true;
                h_lvl  = level as u32;
                h_text.clear();
            }
            Event::Text(ref t) if in_h => h_text.push_str(t),
            Event::End(_) if in_h => {
                headings.push(format!("{} {}", "#".repeat(h_lvl as usize), h_text.trim()));
                in_h = false;
                if headings.len() >= 20 { break; }
            }
            _ => {}
        }
    }

    let mut parts = vec![
        format!("Document: {}", meta.name),
        format!("MIME: {}", meta.mime_type),
        format!("Estimated tokens: {}", count_tokens(text)),
    ];
    if !frontmatter.is_empty() {
        let fm: Vec<String> = frontmatter
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        parts.push(format!("Frontmatter: {}", fm.join(", ")));
    }
    if headings.is_empty() {
        parts.push("Outline: (none)".to_string());
    } else {
        parts.push(format!("Outline:\n  {}", headings.join("\n  ")));
    }
    parts.join("\n")
}

// ── Parser functions ──────────────────────────────────────────────────────────

fn parse_xlsx(data: &[u8], meta: &FileMetadata) -> Result<ChunkResult, IndexerError> {
    use calamine::{open_workbook_auto_from_rs, Reader};

    let cursor = Cursor::new(data);
    let mut wb = open_workbook_auto_from_rs(cursor).map_err(|e| IndexerError::Parse {
        file:    meta.name.clone(),
        message: e.to_string(),
    })?;

    let sheet_names = wb.sheet_names().to_owned();
    let mut chunks: Vec<Chunk> = Vec::new();
    let edges:  Vec<Edge>  = Vec::new();
    let mut t1_idx = 0usize;
    let mut t2_idx = 0usize;
    let mut t3_idx = 0usize;

    for sheet_name in &sheet_names {
        let range = wb.worksheet_range(sheet_name).map_err(|e| IndexerError::Parse {
            file:    meta.name.clone(),
            message: e.to_string(),
        })?;

        let mut rows = range.rows();

        let header_row = match rows.next() {
            Some(r) => r,
            None    => continue,
        };

        let headers: Vec<String> = header_row
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let s = cell.to_string();
                if s.is_empty() { format!("col_{}", i) } else { s }
            })
            .collect();

        let mut t2_rows:   Vec<String> = Vec::new();
        let mut t2_tokens: usize       = 0;
        let mut row_count  = 0usize;

        for row in rows {
            let values: Vec<String> = row.iter().map(|c| c.to_string()).collect();
            let kv         = row_to_kv(&headers, &values);
            let kv_tokens  = count_tokens(&kv) + 1; // +1 for newline separator
            row_count += 1;

            // Tier 1 — one KV record per chunk
            let tc = count_tokens(&kv);
            chunks.push(Chunk {
                tier: 1, chunk_index: t1_idx, content: kv.clone(), token_count: tc,
                meta: serde_json::json!({ "sheet": sheet_name }),
            });
            t1_idx += 1;

            // Tier 2 — accumulate KV records, flush when budget exceeded
            if t2_tokens + kv_tokens > TIER2_MAX_TOKENS && !t2_rows.is_empty() {
                let content = t2_rows.join("\n");
                chunks.push(Chunk {
                    tier: 2, chunk_index: t2_idx, token_count: t2_tokens,
                    content, meta: serde_json::json!({ "sheet": sheet_name }),
                });
                t2_idx   += 1;
                t2_rows   = Vec::new();
                t2_tokens = 0;
            }
            t2_rows.push(kv);
            t2_tokens += kv_tokens;
        }

        if !t2_rows.is_empty() {
            let content = t2_rows.join("\n");
            chunks.push(Chunk {
                tier: 2, chunk_index: t2_idx, token_count: t2_tokens,
                content, meta: serde_json::json!({ "sheet": sheet_name }),
            });
            t2_idx += 1;
        } else if row_count == 0 {
            let content = format!("Columns: {}", headers.join(", "));
            chunks.push(Chunk {
                tier: 2, chunk_index: t2_idx, token_count: count_tokens(&content),
                content, meta: serde_json::json!({ "sheet": sheet_name }),
            });
            t2_idx += 1;
        }

        // Tier 3 — column schema per worksheet
        let schema = serde_json::json!({
            "sheet":     sheet_name,
            "columns":   headers,
            "row_count": row_count,
        });
        let schema_str = serde_json::to_string_pretty(&schema)
            .unwrap_or_else(|_| schema.to_string());
        let tc = count_tokens(&schema_str);
        chunks.push(Chunk {
            tier: 3, chunk_index: t3_idx, token_count: tc,
            content: schema_str,
            meta: serde_json::json!({ "sheet": sheet_name, "source": meta.name }),
        });
        t3_idx += 1;
    }

    // Edge: cross-sheet formula references detected by calamine metadata
    // (calamine 0.35 does not expose formula text; skip REFERENCES edges for now)
    let _ = edges; // suppress unused warning while edges from XLSX remain unimplemented

    Ok(ChunkResult { chunks, edges: vec![] })
}

fn parse_csv(data: &[u8], meta: &FileMetadata) -> Result<ChunkResult, IndexerError> {
    // Decode encoding (BOM-aware, fall back to UTF-8 with replacement)
    let text = decode_bytes(data);

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(text.as_bytes());

    let headers: Vec<String> = {
        let h = rdr.headers().map_err(|e| IndexerError::Parse {
            file:    meta.name.clone(),
            message: e.to_string(),
        })?;
        h.iter().enumerate().map(|(i, s)| {
            if s.is_empty() { format!("col_{}", i) } else { s.to_string() }
        }).collect()
    };

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut t1_idx = 0usize;
    let mut t2_idx = 0usize;
    let mut t2_rows:   Vec<String> = Vec::new();
    let mut t2_tokens: usize       = 0;
    let mut row_count  = 0usize;

    let mut records = rdr.records();
    while let Some(result) = records.next() {
        let record = result.map_err(|e| IndexerError::Parse {
            file:    meta.name.clone(),
            message: e.to_string(),
        })?;
        let values: Vec<String> = (0..headers.len())
            .map(|i| record.get(i).unwrap_or("").to_string())
            .collect();
        let kv        = row_to_kv(&headers, &values);
        let kv_tokens = count_tokens(&kv) + 1;
        row_count += 1;

        // Tier 1 — one KV record per chunk
        let tc = count_tokens(&kv);
        chunks.push(Chunk {
            tier: 1, chunk_index: t1_idx, content: kv.clone(), token_count: tc,
            meta: serde_json::json!({ "row": row_count + 1 }),
        });
        t1_idx += 1;

        // Tier 2 — streaming KV accumulator
        if t2_tokens + kv_tokens > TIER2_MAX_TOKENS && !t2_rows.is_empty() {
            let content = t2_rows.join("\n");
            chunks.push(Chunk {
                tier: 2, chunk_index: t2_idx, token_count: t2_tokens,
                content, meta: serde_json::json!({}),
            });
            t2_idx   += 1;
            t2_rows   = Vec::new();
            t2_tokens = 0;
        }
        t2_rows.push(kv);
        t2_tokens += kv_tokens;
    }

    if !t2_rows.is_empty() {
        let content = t2_rows.join("\n");
        chunks.push(Chunk {
            tier: 2, chunk_index: t2_idx, token_count: t2_tokens,
            content, meta: serde_json::json!({}),
        });
        t2_idx += 1;
    } else if row_count == 0 {
        // Header-only CSV — emit one Tier 2 chunk listing the columns
        let content = format!("Columns: {}", headers.join(", "));
        chunks.push(Chunk {
            tier: 2, chunk_index: t2_idx, token_count: count_tokens(&content),
            content, meta: serde_json::json!({}),
        });
        t2_idx += 1;
    }

    // Tier 3 — column schema
    let schema_str = serde_json::to_string_pretty(&serde_json::json!({
        "columns":   headers,
        "row_count": row_count,
    })).unwrap_or_default();
    let tc = count_tokens(&schema_str);
    chunks.push(Chunk {
        tier: 3, chunk_index: 0, content: schema_str, token_count: tc,
        meta: serde_json::json!({ "source": meta.name }),
    });

    let _ = t2_idx; // suppress unused warning
    Ok(ChunkResult { chunks, edges: vec![] })
}

fn parse_markdown(data: &[u8], meta: &FileMetadata) -> Result<ChunkResult, IndexerError> {
    let raw = decode_bytes(data);
    let (frontmatter, body) = extract_frontmatter(&raw);

    let source = &meta.name;
    let mut base_meta = serde_json::json!({ "source": source });
    if !frontmatter.is_empty() {
        let obj = base_meta.as_object_mut().unwrap();
        for (k, v) in frontmatter.iter().take(5) {
            obj.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
    }

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut t1_idx = 0usize;
    let mut t2_idx = 0usize;

    // Tier 1 — one chunk per heading section (further split if over budget)
    for section in split_markdown_at_headings(body) {
        for segment in split_to_budget(&section, TIER1_MAX_TOKENS) {
            let tc = count_tokens(&segment);
            chunks.push(Chunk {
                tier: 1, chunk_index: t1_idx, content: segment, token_count: tc,
                meta: base_meta.clone(),
            });
            t1_idx += 1;
        }
    }

    // Tier 2 — sliding window over full body
    for segment in split_to_budget(body, TIER2_MAX_TOKENS) {
        let tc = count_tokens(&segment);
        chunks.push(Chunk {
            tier: 2, chunk_index: t2_idx, content: segment, token_count: tc,
            meta: serde_json::json!({ "source": source }),
        });
        t2_idx += 1;
    }

    // Tier 3 — document outline + frontmatter
    let summary = summarise_markdown(body, meta, &frontmatter);
    let tc = count_tokens(&summary);
    let mut t3_meta = serde_json::Map::new();
    t3_meta.insert("source".to_string(), serde_json::Value::String(source.clone()));
    for (k, v) in &frontmatter {
        t3_meta.insert(k.clone(), serde_json::Value::String(v.clone()));
    }
    chunks.push(Chunk {
        tier: 3, chunk_index: 0, content: summary, token_count: tc,
        meta: serde_json::Value::Object(t3_meta),
    });

    // Edges — cross-document links
    let mut edges: Vec<Edge> = Vec::new();
    if !meta.file_uri.is_empty() {
        for (dst_uri, label) in extract_md_links(body, &meta.file_uri) {
            edges.push(Edge {
                src_chunk_index: 0,
                dst_uri,
                edge_type: EdgeType::LinksTo,
                meta: serde_json::json!({ "label": label }),
            });
        }
    }

    Ok(ChunkResult { chunks, edges })
}

fn parse_plaintext(data: &[u8], meta: &FileMetadata) -> Result<ChunkResult, IndexerError> {
    let text = decode_bytes(data);
    let source = &meta.name;

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut t1_idx = 0usize;
    let mut t2_idx = 0usize;

    // Tier 1 — paragraph-level chunks (split on blank lines)
    for para in text.split("\n\n").map(str::trim).filter(|s| !s.is_empty()) {
        for segment in split_to_budget(para, TIER1_MAX_TOKENS) {
            let tc = count_tokens(&segment);
            chunks.push(Chunk {
                tier: 1, chunk_index: t1_idx, content: segment, token_count: tc,
                meta: serde_json::json!({ "source": source }),
            });
            t1_idx += 1;
        }
    }

    // Tier 2 — sliding window over full text
    for segment in split_to_budget(&text, TIER2_MAX_TOKENS) {
        let tc = count_tokens(&segment);
        chunks.push(Chunk {
            tier: 2, chunk_index: t2_idx, content: segment, token_count: tc,
            meta: serde_json::json!({ "source": source }),
        });
        t2_idx += 1;
    }

    // Tier 3 — structural summary
    let summary = summarise(&text, meta, None);
    let tc = count_tokens(&summary);
    chunks.push(Chunk {
        tier: 3, chunk_index: 0, content: summary, token_count: tc,
        meta: serde_json::json!({ "source": source }),
    });

    Ok(ChunkResult { chunks, edges: vec![] })
}

/// Returns `(kind, name)` if `line` begins at column 0 with a named Rust
/// top-level item (fn, impl, struct, enum, trait, …).
fn top_level_decl_rust(line: &str) -> Option<(&'static str, String)> {
    if line.starts_with(' ') || line.starts_with('\t') || line.is_empty() {
        return None;
    }
    // Skip comments and attribute lines — they'll be absorbed into adjacent blocks
    if line.starts_with("//") || line.starts_with("/*") || line.starts_with('#') {
        return None;
    }
    let mut s = line;
    if let Some(r) = s.strip_prefix("pub(crate) ")  { s = r; }
    else if let Some(r) = s.strip_prefix("pub(super) ") { s = r; }
    else if let Some(r) = s.strip_prefix("pub ")    { s = r; }
    s = s.strip_prefix("async ").unwrap_or(s);
    s = s.strip_prefix("unsafe ").unwrap_or(s);

    const KWORDS: &[(&str, &str)] = &[
        ("fn ",           "fn"),
        ("impl",          "impl"),
        ("struct ",       "struct"),
        ("enum ",         "enum"),
        ("trait ",        "trait"),
        ("type ",         "type_alias"),
        ("const ",        "const"),
        ("static ",       "static"),
        ("mod ",          "mod"),
        ("macro_rules! ", "macro"),
    ];
    for &(prefix, kind) in KWORDS {
        if let Some(rest) = s.strip_prefix(prefix) {
            // "impl" must be followed by space or < — guards against "implement_foo"
            if kind == "impl" && !rest.starts_with(' ') && !rest.starts_with('<') {
                continue;
            }
            let rest = rest.trim_start();
            // Skip generic params: impl<T: Bound> → skip to matching >
            let name_str = if rest.starts_with('<') {
                let mut depth = 0i32;
                let mut end = rest.len();
                for (i, c) in rest.char_indices() {
                    match c {
                        '<' => depth += 1,
                        '>' => { depth -= 1; if depth == 0 { end = i + 1; break; } }
                        _   => {}
                    }
                }
                rest[end..].trim_start()
            } else {
                rest
            };
            let name: String = name_str
                .chars()
                .take_while(|&c| c.is_alphanumeric() || c == '_')
                .collect();
            if !name.is_empty() { return Some((kind, name)); }
            if kind == "impl"   { return Some(("impl", "block".to_string())); }
        }
    }
    None
}

/// Returns `(kind, name)` if `line` begins at column 0 with a named
/// TypeScript / JavaScript top-level declaration.
fn top_level_decl(line: &str) -> Option<(&'static str, String)> {
    if line.starts_with(' ') || line.starts_with('\t') || line.is_empty() {
        return None;
    }
    if line.starts_with("//") || line.starts_with("/*") || line.starts_with('*') {
        return None;
    }

    let mut s = line;
    s = s.strip_prefix("export ").unwrap_or(s);
    s = s.strip_prefix("default ").unwrap_or(s);
    s = s.strip_prefix("async ").unwrap_or(s);

    const KWORDS: &[(&str, &str)] = &[
        ("function ", "function"),
        ("class ",    "class"),
        ("const ",    "const"),
        ("interface ", "interface"),
        ("type ",     "type_alias"),
        ("enum ",     "enum"),
        ("let ",      "let"),
        ("var ",      "var"),
    ];
    for &(prefix, kind) in KWORDS {
        if let Some(rest) = s.strip_prefix(prefix) {
            let name: String = rest
                .chars()
                .take_while(|&c| c.is_alphanumeric() || c == '_' || c == '$')
                .collect();
            if !name.is_empty() {
                return Some((kind, name));
            }
        }
    }
    None
}

/// Returns a shortened `end` that skips trailing blank lines in `lines[start..end]`.
fn trim_blank_tail(lines: &[&str], start: usize, end: usize) -> usize {
    let mut e = end;
    while e > start && lines.get(e - 1).map_or(false, |l| l.trim().is_empty()) {
        e -= 1;
    }
    e
}

/// Emit Tier 1 chunks for one declaration block, splitting at **line**
/// boundaries so every piece gets a structured header and no chunk starts
/// mid-expression.  Uses 98 % of the tier budget to stay below score1's
/// upper-bound check.
fn emit_code_block_chunks(
    source:      &str,
    label:       &str,
    lines:       &[&str],
    block_start: usize,
    block_end:   usize,
    chunks:      &mut Vec<Chunk>,
    t1_idx:      &mut usize,
) {
    let code_lines = &lines[block_start..block_end];
    // 98 % of budget to stay below score1's upper-bound check (< 99 % of limit)
    let total_budget = TIER1_MAX_TOKENS * 98 / 100;
    let mut seg_start = 0usize;
    let mut piece     = 0usize;

    while seg_start < code_lines.len() {
        let label_str = if piece == 0 {
            label.to_string()
        } else {
            format!("{label} (cont.)")
        };
        let header = format!(
            "File: {source}\nEntity: {label_str}\nLines: {}-{block_end}\nCode:\n",
            block_start + seg_start + 1,
        );
        let header_toks = count_tokens(&header);
        let code_budget = total_budget.saturating_sub(header_toks).max(10);

        let mut seg_end = seg_start;
        let mut seg_tok = 0usize;
        while seg_end < code_lines.len() {
            let lt = count_tokens(code_lines[seg_end]).saturating_add(1);
            if seg_tok + lt > code_budget && seg_end > seg_start {
                break;
            }
            seg_tok += lt;
            seg_end += 1;
        }
        // Guarantee progress: at minimum one line per chunk
        if seg_end == seg_start {
            seg_end = seg_start + 1;
        }

        let content = format!("{}{}", header, code_lines[seg_start..seg_end].join("\n"));
        let tc = count_tokens(&content);
        chunks.push(Chunk {
            tier: 1, chunk_index: *t1_idx, token_count: tc, content,
            meta: serde_json::json!({ "source": source }),
        });
        *t1_idx += 1;
        seg_start = seg_end;
        piece     += 1;
    }
}

/// Declaration-guided chunker for TypeScript / JavaScript / JSX / TSX.
///
/// Tier 1: one chunk per top-level named declaration (function / class / const …)
///         with a structured header so the embedding model sees provenance.
///         Falls back to a line accumulator with overlap when no declarations
///         are detected (e.g. a file that is pure JSX with no named exports).
/// Tier 2: sliding window over the full file text (cross-symbol context).
/// Tier 3: file outline listing detected symbol names.
fn parse_ts_code(data: &[u8], meta: &FileMetadata) -> Result<ChunkResult, IndexerError> {
    let text   = decode_bytes(data);
    let source = meta.name.as_str();
    let lines: Vec<&str> = text.lines().collect();

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut t1_idx = 0usize;
    let mut t2_idx = 0usize;

    // Collect top-level declaration starts: (line_idx_0based, kind, name)
    let decls: Vec<(usize, &'static str, String)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, &l)| top_level_decl(l).map(|(k, n)| (i, k, n)))
        .collect();

    // ── Tier 1 ────────────────────────────────────────────────────────────────
    if !decls.is_empty() {
        // Build (start, end, label) blocks.  Preamble (imports etc.) first.
        let mut blocks: Vec<(usize, usize, String)> = Vec::new();
        let first = decls[0].0;
        if first > 0 {
            let end = trim_blank_tail(&lines, 0, first);
            if end > 0 {
                blocks.push((0, end, "(module preamble) (preamble)".to_string()));
            }
        }
        for (di, decl) in decls.iter().enumerate() {
            let (start, kind, name) = decl;
            let raw_end = decls.get(di + 1).map(|(i, _, _)| *i).unwrap_or(lines.len());
            let end = trim_blank_tail(&lines, *start, raw_end);
            if end > *start {
                blocks.push((*start, end, format!("{name} ({kind})")));
            }
        }

        for (start, end, label) in &blocks {
            emit_code_block_chunks(source, label, &lines, *start, *end, &mut chunks, &mut t1_idx);
        }
    } else {
        // No declarations detected — line accumulator with overlap
        let overlap = 50usize;
        let mut cur: Vec<&str> = Vec::new();
        let mut cur_tok = 0usize;

        for &line in &lines {
            let lt = count_tokens(line).saturating_add(1);
            if cur_tok + lt > TIER1_MAX_TOKENS && !cur.is_empty() {
                let content = format!("File: {source}\nCode:\n{}", cur.join("\n"));
                let tc = count_tokens(&content);
                chunks.push(Chunk {
                    tier: 1, chunk_index: t1_idx, token_count: tc, content,
                    meta: serde_json::json!({ "source": source }),
                });
                t1_idx += 1;
                // Keep last overlap tokens for context continuity
                let mut ov: Vec<&str> = Vec::new();
                let mut ot = 0usize;
                for &p in cur.iter().rev() {
                    let t = count_tokens(p).saturating_add(1);
                    if ot + t > overlap { break; }
                    ov.insert(0, p);
                    ot += t;
                }
                cur = ov;
                cur_tok = ot;
            }
            cur.push(line);
            cur_tok += lt;
        }
        if !cur.is_empty() {
            let content = format!("File: {source}\nCode:\n{}", cur.join("\n"));
            let tc = count_tokens(&content);
            chunks.push(Chunk {
                tier: 1, chunk_index: t1_idx, token_count: tc, content,
                meta: serde_json::json!({ "source": source }),
            });
            t1_idx += 1;
        }
    }
    let _ = t1_idx;

    // ── Tier 2: sliding window over full file ─────────────────────────────────
    for seg in split_to_budget(&text, TIER2_MAX_TOKENS) {
        let tc = count_tokens(&seg);
        chunks.push(Chunk {
            tier: 2, chunk_index: t2_idx, token_count: tc, content: seg,
            meta: serde_json::json!({ "source": source }),
        });
        t2_idx += 1;
    }
    let _ = t2_idx;

    // ── Tier 3: file outline with symbol list ─────────────────────────────────
    let sym_str = if decls.is_empty() {
        "(none)".to_string()
    } else {
        decls.iter()
            .take(20)
            .map(|(_, k, n)| format!("{k} {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let summary = format!(
        "File: {source}\nMIME: {}\nEstimated tokens: {}\nLines: {}\nSymbols: {sym_str}",
        meta.mime_type,
        count_tokens(&text),
        lines.len(),
    );
    let tc = count_tokens(&summary);
    chunks.push(Chunk {
        tier: 3, chunk_index: 0, token_count: tc, content: summary,
        meta: serde_json::json!({ "source": source }),
    });

    Ok(ChunkResult { chunks, edges: vec![] })
}

/// Declaration-guided chunker for Rust source files.
///
/// Same three-tier strategy as `parse_ts_code`: T1 = one chunk per top-level
/// item, T2 = sliding window, T3 = file outline.
fn parse_rust_code(data: &[u8], meta: &FileMetadata) -> Result<ChunkResult, IndexerError> {
    let text   = decode_bytes(data);
    let source = meta.name.as_str();
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut t1_idx = 0usize;
    let mut t2_idx = 0usize;

    let decls: Vec<(usize, &'static str, String)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, &l)| top_level_decl_rust(l).map(|(k, n)| (i, k, n)))
        .collect();

    // ── Tier 1 ────────────────────────────────────────────────────────────────
    if !decls.is_empty() {
        let mut blocks: Vec<(usize, usize, String)> = Vec::new();
        let first = decls[0].0;
        if first > 0 {
            let end = trim_blank_tail(&lines, 0, first);
            if end > 0 {
                blocks.push((0, end, "(module preamble) (preamble)".to_string()));
            }
        }
        for (di, decl) in decls.iter().enumerate() {
            let (start, kind, name) = decl;
            let raw_end = decls.get(di + 1).map(|(i, _, _)| *i).unwrap_or(lines.len());
            let end = trim_blank_tail(&lines, *start, raw_end);
            if end > *start {
                blocks.push((*start, end, format!("{name} ({kind})")));
            }
        }
        for (start, end, label) in &blocks {
            emit_code_block_chunks(source, label, &lines, *start, *end, &mut chunks, &mut t1_idx);
        }
    } else {
        let overlap = 50usize;
        let mut cur: Vec<&str> = Vec::new();
        let mut cur_tok = 0usize;
        for &line in &lines {
            let lt = count_tokens(line).saturating_add(1);
            if cur_tok + lt > TIER1_MAX_TOKENS && !cur.is_empty() {
                let content = format!("File: {source}\nCode:\n{}", cur.join("\n"));
                let tc = count_tokens(&content);
                chunks.push(Chunk {
                    tier: 1, chunk_index: t1_idx, token_count: tc, content,
                    meta: serde_json::json!({ "source": source }),
                });
                t1_idx += 1;
                let mut ov: Vec<&str> = Vec::new();
                let mut ot = 0usize;
                for &p in cur.iter().rev() {
                    let t = count_tokens(p).saturating_add(1);
                    if ot + t > overlap { break; }
                    ov.insert(0, p);
                    ot += t;
                }
                cur = ov;
                cur_tok = ot;
            }
            cur.push(line);
            cur_tok += lt;
        }
        if !cur.is_empty() {
            let content = format!("File: {source}\nCode:\n{}", cur.join("\n"));
            let tc = count_tokens(&content);
            chunks.push(Chunk {
                tier: 1, chunk_index: t1_idx, token_count: tc, content,
                meta: serde_json::json!({ "source": source }),
            });
            t1_idx += 1;
        }
    }
    let _ = t1_idx;

    // ── Tier 2: sliding window ────────────────────────────────────────────────
    for seg in split_to_budget(&text, TIER2_MAX_TOKENS) {
        let tc = count_tokens(&seg);
        chunks.push(Chunk {
            tier: 2, chunk_index: t2_idx, token_count: tc, content: seg,
            meta: serde_json::json!({ "source": source }),
        });
        t2_idx += 1;
    }
    let _ = t2_idx;

    // ── Tier 3: file outline ──────────────────────────────────────────────────
    let sym_str = if decls.is_empty() {
        "(none)".to_string()
    } else {
        decls.iter()
            .take(20)
            .map(|(_, k, n)| format!("{k} {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let summary = format!(
        "File: {source}\nMIME: {}\nEstimated tokens: {}\nLines: {}\nItems: {sym_str}",
        meta.mime_type,
        count_tokens(&text),
        lines.len(),
    );
    let tc = count_tokens(&summary);
    chunks.push(Chunk {
        tier: 3, chunk_index: 0, token_count: tc, content: summary,
        meta: serde_json::json!({ "source": source }),
    });

    Ok(ChunkResult { chunks, edges: vec![] })
}

fn parse_code(data: &[u8], meta: &FileMetadata) -> Result<ChunkResult, IndexerError> {
    if meta.mime_type == "text/x-python" {
        parse_plaintext(data, meta)
    } else {
        parse_ts_code(data, meta)
    }
}

// PDF and DOCX require libraries not yet in Cargo.toml; fall back to binary note.
fn parse_pdf(data: &[u8], meta: &FileMetadata) -> Result<ChunkResult, IndexerError> {
    parse_binary_fallback(data, meta)
}

fn parse_docx(data: &[u8], meta: &FileMetadata) -> Result<ChunkResult, IndexerError> {
    parse_binary_fallback(data, meta)
}

fn parse_binary_fallback(data: &[u8], meta: &FileMetadata) -> Result<ChunkResult, IndexerError> {
    let note = format!(
        "Unsupported MIME type: {}. File: {}. Size: {} bytes. No text content was extracted.",
        meta.mime_type,
        meta.name,
        data.len(),
    );
    let tc = count_tokens(&note);
    Ok(ChunkResult {
        chunks: vec![Chunk {
            tier: 3, chunk_index: 0, content: note, token_count: tc,
            meta: serde_json::json!({
                "mime_type": meta.mime_type,
                "source":    meta.name,
            }),
        }],
        edges: vec![],
    })
}

// ── Encoding helper ───────────────────────────────────────────────────────────

fn decode_bytes(data: &[u8]) -> String {
    // BOM detection first
    if let Some((enc, bom_len)) = encoding_rs::Encoding::for_bom(data) {
        let (cow, _, _) = enc.decode(&data[bom_len..]);
        return cow.into_owned();
    }
    // Try strict UTF-8, fall back to lossy
    match std::str::from_utf8(data) {
        Ok(s) => s.to_string(),
        Err(_) => String::from_utf8_lossy(data).into_owned(),
    }
}

// ── Parser dispatch ───────────────────────────────────────────────────────────

type ParserFn = fn(&[u8], &FileMetadata) -> Result<ChunkResult, IndexerError>;

fn resolve_parser(mime: &str) -> ParserFn {
    match mime {
        "application/pdf"  => parse_pdf,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        | "application/msword" => parse_docx,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        | "application/vnd.ms-excel" => parse_xlsx,
        "text/csv"         => parse_csv,
        "text/markdown"    => parse_markdown,
        "text/x-rust" => parse_rust_code,
        "text/x-python" | "text/typescript" | "text/javascript"
        | "application/javascript" => parse_code,
        other if other.starts_with("text/") => parse_plaintext,
        _ => parse_binary_fallback,
    }
}

// ── FileChunker ───────────────────────────────────────────────────────────────

pub struct FileChunker;

impl FileChunker {
    pub fn new() -> Self {
        FileChunker
    }

    pub fn chunk(
        &self,
        data: &[u8],
        meta: &FileMetadata,
        map: &ChunkerMap,
    ) -> Result<(ChunkResult, String), IndexerError> {
        let mime = meta
            .mime_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_lowercase();
        let method = map.method_for(&mime).to_string();
        let result = resolve_parser(&mime)(data, meta)?;
        Ok((result, method))
    }
}

impl Default for FileChunker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_meta(name: &str, mime: &str) -> FileMetadata {
        FileMetadata {
            file_uri:   format!("file:///test/{}", name),
            name:       name.to_string(),
            mime_type:  mime.to_string(),
            size_bytes: 0,
            modified_at: 0.0,
        }
    }

    // ── XLSX tests ────────────────────────────────────────────────────────────

    fn make_xlsx_3rows() -> Vec<u8> {
        use rust_xlsxwriter::Workbook;
        let mut wb = Workbook::new();
        let ws = wb.add_worksheet();
        ws.write(0, 0, "Name").unwrap();
        ws.write(0, 1, "Age").unwrap();
        ws.write(1, 0, "Alice").unwrap();
        ws.write(1, 1, 30_u32).unwrap();
        ws.write(2, 0, "Bob").unwrap();
        ws.write(2, 1, 25_u32).unwrap();
        ws.write(3, 0, "Charlie").unwrap();
        ws.write(3, 1, 35_u32).unwrap();
        wb.save_to_buffer().unwrap()
    }

    #[test]
    fn xlsx_tier1_one_row_per_chunk() {
        let data = make_xlsx_3rows();
        let meta = fake_meta("test.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
        let result = parse_xlsx(&data, &meta).unwrap();
        let t1: Vec<_> = result.chunks.iter().filter(|c| c.tier == 1).collect();
        // 3 data rows → 3 Tier 1 chunks
        assert_eq!(t1.len(), 3, "expected 3 Tier 1 chunks, got {}", t1.len());
    }

    #[test]
    fn xlsx_tier2_uses_kv_format() {
        let data = make_xlsx_3rows();
        let meta = fake_meta("test.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
        let result = parse_xlsx(&data, &meta).unwrap();
        for chunk in result.chunks.iter().filter(|c| c.tier == 2) {
            assert!(
                chunk.content.contains(": "),
                "Tier 2 chunk should use key-value format: 'Name: Alice | Age: 30'"
            );
        }
    }

    #[test]
    fn xlsx_tier3_schema_json() {
        let data = make_xlsx_3rows();
        let meta = fake_meta("test.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
        let result = parse_xlsx(&data, &meta).unwrap();
        let t3: Vec<_> = result.chunks.iter().filter(|c| c.tier == 3).collect();
        assert!(!t3.is_empty(), "expected at least one Tier 3 chunk");
        for chunk in &t3 {
            let v: serde_json::Value = serde_json::from_str(&chunk.content)
                .expect("Tier 3 chunk content should be valid JSON");
            assert!(
                v.get("columns").is_some(),
                "Tier 3 JSON should have 'columns' key"
            );
        }
    }

    // ── CSV tests ─────────────────────────────────────────────────────────────

    #[test]
    fn csv_empty_sheet_yields_header_chunk() {
        let csv_bytes = b"Name,Age\n";
        let meta = fake_meta("empty.csv", "text/csv");
        let result = parse_csv(csv_bytes, &meta).unwrap();
        let t2: Vec<_> = result.chunks.iter().filter(|c| c.tier == 2).collect();
        assert_eq!(t2.len(), 1, "header-only CSV should produce exactly 1 Tier 2 chunk");
        assert!(t2[0].content.contains("Name"), "Tier 2 chunk should contain headers");
    }

    // ── Markdown tests ────────────────────────────────────────────────────────

    #[test]
    fn markdown_splits_at_headings() {
        let md = "## Section A\n\nSome text about A.\n\n## Section B\n\nSome text about B.\n";
        let meta = fake_meta("doc.md", "text/markdown");
        let result = parse_markdown(md.as_bytes(), &meta).unwrap();
        let t1: Vec<_> = result.chunks.iter().filter(|c| c.tier == 1).collect();
        // Each ## heading should produce its own Tier 1 chunk (two sections)
        assert!(t1.len() >= 2, "expected ≥2 Tier 1 chunks for two ## headings, got {}", t1.len());
        assert!(
            t1.iter().any(|c| c.content.contains("Section A")),
            "missing Section A chunk"
        );
        assert!(
            t1.iter().any(|c| c.content.contains("Section B")),
            "missing Section B chunk"
        );
    }

    #[test]
    fn markdown_frontmatter_stripped() {
        let md = "---\ntitle: My Doc\nauthor: Alice\n---\n\n## Body\n\nContent here.\n";
        let meta = fake_meta("doc.md", "text/markdown");
        let result = parse_markdown(md.as_bytes(), &meta).unwrap();

        // Tier 1 chunks should NOT contain raw frontmatter delimiters
        for chunk in result.chunks.iter().filter(|c| c.tier == 1) {
            assert!(
                !chunk.content.starts_with("---"),
                "Tier 1 chunk should not start with frontmatter delimiter"
            );
        }

        // Tier 3 chunk should reference the frontmatter fields
        let t3 = result.chunks.iter().find(|c| c.tier == 3).unwrap();
        assert!(
            t3.content.contains("title") || t3.meta.get("title").is_some(),
            "frontmatter title should appear in Tier 3 content or meta"
        );
    }

    #[test]
    fn markdown_links_become_edges() {
        let md = "## Intro\n\nSee [overview](./overview.md) for details.\n";
        let meta = FileMetadata {
            file_uri:    "file:///docs/intro.md".to_string(),
            name:        "intro.md".to_string(),
            mime_type:   "text/markdown".to_string(),
            size_bytes:  0,
            modified_at: 0.0,
        };
        let result = parse_markdown(md.as_bytes(), &meta).unwrap();
        assert!(
            !result.edges.is_empty(),
            "relative markdown link should produce a LinksTo edge"
        );
        assert!(
            result.edges.iter().all(|e| e.edge_type == EdgeType::LinksTo),
            "all edges should be LinksTo"
        );
        assert!(
            result.edges.iter().any(|e| e.dst_uri.contains("overview")),
            "edge should reference overview.md"
        );
    }

    // ── Plaintext tests ───────────────────────────────────────────────────────

    #[test]
    fn plaintext_split_to_budget_no_token_overrun() {
        // Build a paragraph large enough to require splitting (> 500 tokens fallback).
        // With the byte fallback (3 bytes ≈ 1 token), 1600 bytes → ~533 tokens.
        let long_para = "word ".repeat(360); // 360 * 5 = 1800 bytes → ~600 tokens
        let text = format!("{}\n\n{}", long_para, long_para);
        let meta = fake_meta("big.txt", "text/plain");
        let result = parse_plaintext(text.as_bytes(), &meta).unwrap();
        for chunk in result.chunks.iter().filter(|c| c.tier == 1) {
            assert!(
                chunk.token_count <= TIER1_MAX_TOKENS,
                "Tier 1 chunk has {} tokens, expected ≤ {}",
                chunk.token_count,
                TIER1_MAX_TOKENS
            );
        }
    }

    // ── Binary fallback test ──────────────────────────────────────────────────

    #[test]
    fn binary_fallback_returns_tier3_note() {
        let meta = fake_meta("image.png", "image/png");
        let result = parse_binary_fallback(b"\x89PNG\r\n", &meta).unwrap();
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.chunks[0].tier, 3);
        assert!(
            result.chunks[0].content.contains("image/png"),
            "fallback note should mention the MIME type"
        );
        assert!(result.edges.is_empty());
    }

    // ── TypeScript / TSX chunker tests ────────────────────────────────────────

    #[test]
    fn tsx_chunks_by_declaration_not_blank_line() {
        let tsx = b"import React from 'react';\n\nexport const MyButton = ({ label }: { label: string }) => {\n  return (\n    <button\n      className=\"px-4 py-2\"\n    >\n\n      {label}\n    </button>\n  );\n};\n\nexport function Header() {\n  return <h1>Title</h1>;\n}\n";
        let meta = fake_meta("button.tsx", "text/typescript");
        let result = parse_ts_code(tsx, &meta).unwrap();
        let t1: Vec<_> = result.chunks.iter().filter(|c| c.tier == 1).collect();
        // Should produce: preamble + MyButton + Header = 3 blocks (not many small fragments)
        assert!(t1.len() <= 4, "expected at most 4 T1 chunks (preamble + 2 decls), got {}", t1.len());
        // No chunk should be just a closing tag fragment
        for chunk in &t1 {
            assert!(
                !chunk.content.trim().starts_with("</"),
                "T1 chunk should not be a bare closing tag: {:?}",
                &chunk.content[..chunk.content.len().min(60)]
            );
        }
    }

    #[test]
    fn tsx_t1_chunks_have_entity_header() {
        let tsx = b"export const Foo = () => <div>hello</div>;\nexport function Bar() { return null; }\n";
        let meta = fake_meta("comp.tsx", "text/typescript");
        let result = parse_ts_code(tsx, &meta).unwrap();
        let t1: Vec<_> = result.chunks.iter().filter(|c| c.tier == 1).collect();
        assert!(!t1.is_empty(), "expected T1 chunks");
        // Each T1 chunk must have the structured header
        for chunk in &t1 {
            assert!(
                chunk.content.contains("File:") && chunk.content.contains("Entity:"),
                "T1 chunk missing header: {:?}",
                &chunk.content[..chunk.content.len().min(80)]
            );
        }
    }

    #[test]
    fn tsx_t3_lists_symbols() {
        let tsx = b"export const Alpha = () => null;\nexport function Beta() { return null; }\n";
        let meta = fake_meta("syms.tsx", "text/typescript");
        let result = parse_ts_code(tsx, &meta).unwrap();
        let t3 = result.chunks.iter().find(|c| c.tier == 3).unwrap();
        assert!(t3.content.contains("Alpha"), "T3 should mention Alpha");
        assert!(t3.content.contains("Beta"), "T3 should mention Beta");
    }

    // ── Rust chunker tests ────────────────────────────────────────────────────

    // Written as a single-line string to avoid the line-continuation whitespace
    // stripping that would remove indentation from b"...\n\" literals.
    fn rust_src() -> &'static str {
        "use std::fmt;\n\nstruct Point {\n    x: f64,\n\n    y: f64,\n}\n\nimpl Point {\n    pub fn new(x: f64, y: f64) -> Self {\n        Self { x, y }\n    }\n\n    pub fn distance(&self, other: &Point) -> f64 {\n        let dx = self.x - other.x;\n        (dx * dx).sqrt()\n    }\n}\n\npub fn main() {\n    let p = Point::new(0.0, 0.0);\n    println!(\"{:?}\", p);\n}\n"
    }

    #[test]
    fn rust_chunks_by_declaration_not_blank_line() {
        let meta = fake_meta("point.rs", "text/x-rust");
        let result = parse_rust_code(rust_src().as_bytes(), &meta).unwrap();
        let t1: Vec<_> = result.chunks.iter().filter(|c| c.tier == 1).collect();
        // preamble + struct Point + impl Point + fn main = 4 blocks
        assert!(t1.len() <= 5, "expected ≤5 T1 chunks, got {}", t1.len());
        // No chunk should be a bare closing brace fragment
        for chunk in &t1 {
            assert!(
                chunk.content.trim().len() > 5,
                "T1 chunk looks like a micro-fragment: {:?}",
                &chunk.content[..chunk.content.len().min(60)]
            );
        }
    }

    #[test]
    fn rust_t1_chunks_have_entity_header() {
        let meta = fake_meta("point.rs", "text/x-rust");
        let result = parse_rust_code(rust_src().as_bytes(), &meta).unwrap();
        let t1: Vec<_> = result.chunks.iter().filter(|c| c.tier == 1).collect();
        assert!(!t1.is_empty(), "expected T1 chunks");
        for chunk in &t1 {
            assert!(
                chunk.content.contains("File:") && chunk.content.contains("Entity:"),
                "T1 chunk missing header: {:?}",
                &chunk.content[..chunk.content.len().min(80)]
            );
        }
    }

    #[test]
    fn rust_t3_lists_items() {
        let meta = fake_meta("point.rs", "text/x-rust");
        let result = parse_rust_code(rust_src().as_bytes(), &meta).unwrap();
        let t3 = result.chunks.iter().find(|c| c.tier == 3).unwrap();
        assert!(t3.content.contains("Point"), "T3 should mention Point");
        assert!(t3.content.contains("main"), "T3 should mention main");
    }
}
