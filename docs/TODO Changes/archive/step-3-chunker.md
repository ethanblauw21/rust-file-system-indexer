# COMPLETED at 5/13/26

# Step 3 — The Semantic Chunker (`chunker.rs`) ⏳

**Status:** Pending

## Goal
Implement the multi-modal, three-tier chunking pipeline using `calamine` for XLSX/CSV and `pulldown-cmark` for Markdown. The 3-Tier logic (Atomic / Contextual / Architectural) must remain intact.

## Python Reference
`C:\Users\edb\Documents\indexer\fileSystem\file_chunker.py` — `FileChunker` class.

## Tier Definitions
| Tier | Label | Max Tokens | Description |
|---|---|---|---|
| 1 | Atomic / Surgical | 500 | Paragraph, single spreadsheet row, heading section, AST symbol |
| 2 | Contextual / Component | 1500 | Full page, sliding window, multi-row table segment |
| 3 | Architectural | Unlimited | Structural summary / column schema / file outline |

## Output Types

```rust
pub struct Chunk {
    pub tier:        u8,        // 1 | 2 | 3
    pub chunk_index: usize,
    pub content:     String,
    pub token_count: usize,
    pub meta:        serde_json::Value,
}

pub struct Edge {
    pub src_chunk_index: usize,
    pub dst_uri:         String,
    pub edge_type:       EdgeType,  // enum: LinksTo | References | Embeds
    pub meta:            serde_json::Value,
}

pub struct ChunkResult {
    pub chunks: Vec<Chunk>,
    pub edges:  Vec<Edge>,
}
```

## Token Counting
Use the `tokenizers` crate (HuggingFace fast tokenizer) loaded with the `nomic-embed-text-v1.5` tokenizer vocabulary. This is more accurate than Python's `tiktoken cl100k_base` for the embedding model actually used.

Fallback: `max(1, bytes.len() / 3)` if tokenizer fails to load.

`split_to_budget(text, budget) -> Vec<String>` must split on token boundaries, never truncating a token mid-encode.

## Parser Dispatch
Dispatch by MIME type (never filename). Use a `match` on `&str`:

```rust
fn resolve_parser(mime: &str) -> fn(&[u8], &FileMetadata) -> Result<ChunkResult, IndexerError>
```

| MIME | Parser |
|---|---|
| `application/pdf` | `parse_pdf` (lopdf / pdf-extract) |
| `application/vnd...wordprocessingml.document` | `parse_docx` (docx crate) |
| `application/vnd...spreadsheetml.sheet` | `parse_xlsx` (calamine) |
| `application/vnd.ms-excel` | `parse_xlsx` |
| `text/csv` | `parse_csv` (calamine or std csv) |
| `text/markdown` | `parse_markdown` (pulldown-cmark) |
| `text/x-python`, `text/typescript`, `text/javascript` | `parse_code` (tree-sitter, fallback to plaintext) |
| `text/*` (catch-all) | `parse_plaintext` |
| anything else | `parse_binary_fallback` |

## Parser Notes

### XLSX (`calamine`)
- Use `calamine::open_workbook_auto_from_rs` on a `Cursor<&[u8]>` — no temp file needed
- `calamine` reads computed values directly; no need for the Python double-load hack (`data_only=True` + second load for formulas)
- Stream rows lazily with `worksheet.rows()` iterator — never `collect()` the entire sheet
- Tier 1: one self-contained Markdown mini-table per row (header block + single data row)
- Tier 2: accumulate rows until token budget exhausted, then flush; header repeated on each segment
- Tier 3: JSON column schema `{ sheet, columns, row_count }`
- Edges: scan cell string values for `!` cross-sheet references

### CSV
- Use `encoding_rs` to detect / transcode non-UTF-8 streams before parsing
- Same Tier 1/2/3 logic as XLSX but single-sheet

### Markdown (`pulldown-cmark`)
- Frontmatter extraction: regex `^---\n(.*?)\n---\n` (DOTALL) — no YAML lib needed
- Section splitting: walk `pulldown_cmark::Event` stream; emit a new section at each `Heading` event of level ≤ 3
- Merge heading-only sections forward (breadcrumb context)
- Edges: extract `[label](target)` links; skip `://` (external) and `#` (anchor-only); resolve relative paths against `file_uri` directory
- Tier 3: heading outline + frontmatter keys

### PDF (`lopdf`)
- Tier 1: paragraph-level splits on `\n{2,}` within each page
- Tier 2: full page text
- Tier 3: heuristic summary (heading lines, token count, page count)
- Edges: hyperlink annotations from page annotation dictionaries
- Design `PdfParser` behind a sub-trait so `poppler` (shell-out) can be swapped in if `lopdf` mis-parses a file

### Plaintext / Code
- Tier 1: split on `\n{2,}` paragraph boundaries, then `split_to_budget`
- Tier 2: sliding window over full file text
- Tier 3: structural summary (line count, token estimate, key headings heuristic)

### Binary fallback
- Return a single Tier 3 chunk with a human-readable note (MIME, filename, byte size)

## Key Rust Idioms to Apply
- Accept `&[u8]` not `Vec<u8>` in all parsers — callers pass a slice of the `Bytes` buffer from Step 1, zero-copy
- `Cow<'_, str>` for decoded text when the source is already UTF-8 (avoid allocation); fall back to `String` only when re-encoding is needed
- `pulldown-cmark` event iterator is lazy — do not collect into a `Vec<Event>` before processing
- `calamine` row iterator is lazy — same rule
- Token budget loop: accumulate into `String::with_capacity(estimated_bytes)` before each flush

## Tests to Write
- `xlsx_tier1_one_row_per_chunk` — 3-row sheet → 3 Tier 1 chunks
- `xlsx_tier2_header_repeated` — every Tier 2 segment starts with header block
- `xlsx_tier3_schema_json` — Tier 3 content is valid JSON with `columns` key
- `csv_empty_sheet_yields_header_chunk` — header-only sheet emits one Tier 2 chunk
- `markdown_splits_at_headings` — `## H2` boundary creates a new Tier 1 chunk
- `markdown_frontmatter_stripped` — frontmatter fields appear in Tier 3 meta, not Tier 1 content
- `markdown_links_become_edges` — `[label](./other.md)` produces a `LinksTo` edge
- `plaintext_split_to_budget_no_token_overrun` — every Tier 1 chunk ≤ 500 tokens
- `binary_fallback_returns_tier3_note` — unsupported MIME → single Tier 3 chunk with descriptive text
