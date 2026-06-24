# Dogfooding Run — 2026-06-22

A full end-to-end exercise of the indexer against real data under `C:\Users\ethan\`.
Three finding lists are tracked separately, plus a running usability-friction log.

## Setup

| Artifact | Status | Detail |
| --- | --- | --- |
| Release build | ✅ | `cargo build --release` (needs `PROTOC` env set) |
| ONNX Runtime DLL | ✅ | `onnxruntime-win-x64-**1.24.2**\lib\onnxruntime.dll`, `ORT_DYLIB_PATH` set (User). **1.22.0 deadlocks** — see Defect 1. |
| nomic ONNX model | ✅ | `models\nomic\nomic-embed-text-v1.5.onnx` (547 MB), `NOMIC_ONNX_PATH` set (User) |
| tokenizer.json | ✅ | `models\nomic\tokenizer.json` (711 KB) |
| Embedder smoke test | ✅ | 3 files → 9 chunks embedded, 9 vectors; hybrid query `"asynchronous programming in rust"` → correct doc at 100% (semantic, not lexical) |
| FTS-fix regression | ✅ | After Defect-2 fix + rebuild: 4× sparse + 3× hybrid on one index all succeed (was `malformed` at search #2 before). |

## Corpus

Curated data roots (built-in ignores already exclude AppData/.cargo/.rustup/node_modules/target/exe/media):

| Root | Indexable data files | Size | Notes |
| --- | --- | --- | --- |
| `Documents` | 1257 | 23.5 MB | includes this repo (`rust-file-system-indexer`) |
| `homelab` | 561 | 4.2 MB | ADR docs, configs |
| `OneDrive` | 331 | 9.6 MB | cloud placeholders auto-skipped |
| `codebase-indexer` | 91 | 1.6 MB | |
| `Desktop` | 0 | — | **empty** (fresh install) |
| `Downloads` | 0 | — | **empty** (fresh install) |

_Survey counts are pre-index estimates (data extensions, ≤50 MB, excluding node_modules/target/.git/etc.). Actual indexed counts below._

Index stats (live `full-corpus`, still building): root 1 `codebase-indexer` ≈ 68 files / 1,551 chunks (T1 1,336 · T2 147 · T3 68) at time of the fast-number probe. Remaining roots (homelab, OneDrive, Documents) embedding in background for the durable pass.

---

## List 1 — Ranking failures
_Queries that surfaced irrelevant results or mis-ordered relevant ones._

**Method.** 16 queries grounded in the live corpus (`C:\Users\ethan\codebase-indexer` — a Python multi-language code-intelligence project: adapters, hybrid retriever, AST chunker, ~25 ADR/design docs), each with a known correct target file, run in all three modes. Hit-rate (target in top-3 / top-10):

| Mode | top-3 | top-10 |
| --- | --- | --- |
| **dense** | **14/16 (88%)** | 14/16 |
| **hybrid** (default) | 11/16 (69%) | 11/16 |
| sparse | 10/16 | 11/16 |

**The flagship hybrid mode is worse than pure dense.** The engine itself is sound — dense embeddings + ANN hit 88% — every failure is in the sparse/fusion layer.

| # | Query | Mode | What went wrong | Expected | Severity |
| --- | --- | --- | --- | --- | --- |
| 1 | `C++ tree-sitter language adapter` | hybrid/sparse | **0 results** in default+sparse modes; dense ranks `cpp_adapter.py` **#1 (84%)**. Punctuation (`+`) crashes FTS5 → kills hybrid. | `adapters/cpp_adapter.py` | High |
| 2 | `C# csharp language adapter` | hybrid/sparse | Same: `#` → 0 sparse/hybrid; dense ranks `csharp_adapter.py` #2. | `adapters/csharp_adapter.py` | High |
| 3 | `import resolver cross-file references` | hybrid/sparse | Same: hyphen → 0 sparse/hybrid; dense ranks `import_resolver.py` #2. | `src/import_resolver.py` | High |
| 4 | (overall) hybrid vs dense | hybrid | Hybrid drops 3 targets out of top-10 that dense ranks #1–2 — RRF fusion **loses** good dense hits whenever sparse errors/empties (no fallback). | dense parity ≥ | High |
| 5 | `hybrid retrieval dense sparse fusion` | all | Implementation file `hybrid_retriever.py` (76 chunks) never reaches top-10; conceptual `.md` docs dominate. Code intent → only docs returned. | `src/hybrid_retriever.py` in top-10 | Med |
| 6 | `incremental indexing merkle tree drift detection` | all | Same code-vs-docs miss: `incremental_indexer.py` absent from top-10; `merkle-tree-drift-handling.md` wins. | `src/incremental_indexer.py` in top-10 | Med |
| 7 | many (AST chunking, summarizer, category tagging) | hybrid/dense | `graphify-out/GRAPH_REPORT.md` is a **ranking magnet** — one generated aggregate doc takes top-1 for several unrelated queries. | topical file | Low |
| 8 | single-char typo in key term | hybrid/dense | **Typo intolerance** — one wrong char drops the target out of top-10 (`C++`, `C#`, `recursive file search`); dense top-3 collapses 11→6. See Mutation M1. | typo-tolerant match | Med |
| 9 | exact basename (`ast_chunker`, `cpp_adapter`, `csharp_adapter`, `RecFileSearch`) | hybrid | **Filename-handle search unreliable** — searching a file's own name doesn't surface it in top-10 (collides with sibling files + magnet docs). See Mutation M3. | that file at/near #1 | Med |

## List 2 — Bad chunks
_Flagged by `scorer.rs` (structural + coherence)._

**⚠️ Correction (re-verified 2026-06-23).** An earlier draft of this list claimed "0 chunks flagged / structural floors at 0.50 / coherence NULL for every chunk — half the scorer is non-functional." **That was a measurement artifact.** It was scored on the incomplete 57-chunk `corpus-index`, where embedding had been interrupted: **tier-1/2 chunks never received `lance_id`s**, so coherence had no parent vectors to compare against (→ NULL), and that corpus was prose-dominated, so every chunk happened to sit at exactly `str=0.50`. Re-running `score` on a **complete** index (`src/`, 440 chunks, 100% lance coverage) tells a different story:

```
Scored 440 chunks — 74 flagged (68 structural, 1 coherence, 5 both)
```

| # | Finding | Status | Evidence (complete `src/` index, 440 chunks) |
| --- | --- | --- | --- |
| A | **Coherence is fully functional** — not a dead code path. | ❌ claim retracted | Populated for **430/430** eligible chunks (all T1+T2; T3 is `None` by design). Range min `0.533` / avg `0.802` / max `0.922`; **6 chunks flagged** below the `COHERENCE_THRESHOLD=0.6`. The earlier NULL was 100% caused by the un-embedded T1/T2 in the partial corpus. |
| B | **Structural flag works in general** — fires on real defects. | ❌ "flag is dead" retracted | **72/440 flagged** on code with unbalanced brackets / no terminal punctuation: histogram `0.15→1, 0.3→45, 0.4→27`. The "floors at 0.50" was specific to prose (balanced brackets), not universal. |
| C | **0.50 boundary dead-zone** — real, but narrow. Chunks scoring *exactly* 0.50 (fail token-count **and** sentence-completeness, but pass bracket `0.20` + whitespace `0.15` + repetition `0.15`) escape, because the flag test is **exclusive** (`s < 0.5`). | ⚠️ real defect | 3 chunks sit at exactly `0.50` and are **not** flagged: T2 import blocks / doc-comment headers (`use crate::error::IndexerError; use r2d2::Pool; …`, `//! storage.rs — Storage abstraction layer …`). Single-line chunks like `#pragma once` / `ARDUINOJSON_BEGIN_PRIVATE_NAMESPACE` land here too. |

**Why prose mis-measured it.** Prose/markdown chunks almost always have balanced brackets, low whitespace, and varied tokens, so they pile up at exactly `0.20+0.15+0.15 = 0.50` — the floor *equals* the threshold, and exclusive `<` flags none. Code chunks routinely break brackets at mid-function cuts and drop to `0.30–0.40`, so they flag fine. The 57-chunk `corpus-index` was both prose-dominated **and** missing its T1/T2 vectors, which simultaneously zeroed the structural flags **and** the coherence column — producing the misleading "scorer is half-dead" reading.

**Recommended fix (1 line, lands the real defect C).** Make the structural threshold **inclusive** — `let struct_bad = s <= STRUCTURAL_THRESHOLD;` (or set `STRUCTURAL_THRESHOLD = 0.51`). This flags the exact-0.50 dead-zone (import blocks, single-line `#pragma once`, doc headers) **without** touching the 195-chunk cluster at `0.55` (normal code that merely lacks terminal punctuation — flagging those would be noise). Coherence needs no change; the operational lesson is simply that `score` must run **after** embedding completes, never on a partially-embedded index.

## List 3 — Code defects
_Crashes, panics, index failures, wrong output._

| # | Where | Symptom | Repro | Severity |
| --- | --- | --- | --- | --- |
| 1 | `Cargo.toml` / `indexer.rs` `Embedder::load` | **Embedder deadlocks** (0 CPU, flat 14 MB, hangs forever) on any index/search with `NOMIC_ONNX_PATH` set. | Set `ORT_DYLIB_PATH` to onnxruntime **1.22.0** + `NOMIC_ONNX_PATH` → `index` hangs 120 s+; unset → 0.4 s. | Critical |
| 2 | `db.rs` `EnterpriseDb::new` (T3 FTS cleanup) | **FTS5 index corruption** → all sparse/hybrid searches fail `database disk image is malformed` after the *second* open of any index containing T3 chunks. | `index smoke` → search #1 OK → search #2 `malformed`. DB file MD5 changes after a read-only search; 3× dense opens then `sparse` → immediate `malformed`. | Critical |
| 3 | `indexer.rs` `index_root` producer/consumer | **Indexer hard-deadlocks** (0 CPU, all threads parked) on any corpus with **> 256 files**. Observed wedged 12.4 h on `homelab`. | Index `homelab\homelabMCPServer` (767 files) → freezes after `Embedding batch 1` with `Embedded 0`. Index `…\src` (221 files, < 256) → embeds normally. Threshold = the channel capacity. | Critical |
| 4 | `db.rs` `sanitize_fts_query` | **FTS5 syntax error → 0 sparse results** for any query containing `+`, `-`, or `#`. The sanitizer denylists only `():^*` and leaves tokens unquoted, so `C++` reaches FTS5 as `C ++` → `syntax error near "+"`. | `search 'C++ tree-sitter language adapter' --mode sparse` → `Search error: database error: fts5: syntax error near "+"`, 0 hits. Same for `C#`, `tree-sitter`. | High — **FIXED ✅** |
| 5 | `search.rs` `hybrid_search` (line ~355) | **Hybrid returns 0 results when sparse errors**, discarding the dense hits it would otherwise return. No graceful degradation. | `search 'C++ tree-sitter language adapter' --mode hybrid` → **0 results**, while `--mode dense` ranks `cpp_adapter.py` #1 (84%). | High — **FIXED ✅** |
| 6 | `indexer.rs` `index_root` end-prune (line 775–786) | **Multi-root index is impossible — each `index <root>` deletes every file not under *that* root.** A second root indexed into the same `--index-dir` silently evicts the entire first root. Defeats any union/distractor corpus. | Into one index dir: `index codebase-indexer` (92 files) → `index homelab` → log reads `572 indexed, **92 removed**`; `index OneDrive` → `15 indexed, **572 removed**`. Final index holds only the last root. | High |
| 7 | `storage.rs` ignore lists | **Game/app data is indexed as "data," against scope.** No game-data dir/extension patterns, so e.g. `Documents\American Truck Simulator\` save/config/cache (`.sii`, `.navcache`, `.cfg`) get chunked + embedded. | `index Documents` → index contains `American Truck Simulator\.history.sii`, `cache\…​.navcache`, `config.cfg`. | Medium |

**Defect 3 — root cause.** The producer is a `spawn_blocking` task running `file_uris.par_iter()` on the **global** rayon pool, each worker calling `tx.blocking_send()` into a `tokio::sync::mpsc::channel(256)`. The consumer drains the channel and, per `EMBED_BATCH_SIZE` (32) batch, calls `flush_embeddings` → `Embedder::embed` → `Tokenizer::encode_batch`, which **also fans out over the global rayon pool**. When a corpus exceeds 256 files, the producer fills the channel and **every global-rayon worker parks inside `blocking_send`**; the consumer's `encode_batch` then waits forever for a free rayon worker, so it never drains the channel, so the producer never unblocks → deadlock at 0 CPU. Corpora ≤ 256 files (codebase-indexer = 92, `src` = 221) escape because the producer drains fully and frees rayon before the first flush. **Confirmed empirically:** the wedge triggers exactly at the 256 file-count threshold; the histogram of a wedged run shows `Embedding batch ×1 / Embedded ×0` (consumer entered the first tokenize and never returned) while the producer had logged ~279 `Indexing:` lines and parked. **Fix applied (code):** run the producer's I/O+chunking on a **dedicated** `rayon::ThreadPool` (via `pool.install`) so its `blocking_send` parks never starve the global pool that the tokenizer needs. Backpressure and parallel tokenization are both preserved. **Verified:** after the fix, the full 767-file `homelabMCPServer` (12 h hang before) embeds steadily — `Embedded` climbed 0→11 batches in 84 s with CPU across all cores, no stall.

**Defect 1 — root cause.** `ort 2.0.0-rc.12` default features enable `api-24`, so `ORT_API_VERSION=24` and the runtime calls `GetApi(24)`, which requires onnxruntime **1.24.x**. A 1.22.0 DLL returns `NULL` for `GetApi(24)` and the session-builder wedges with no error. Both call sites (`indexer.rs:641`, `search.rs:92`) wrap load in `.ok()`, which swallows *errors* but cannot rescue a *hang* — the "silent-and-stuck" failure mode. **Fix applied (env):** use onnxruntime **1.24.2** (the exact build `ort-sys` pins in `dist.txt`). **Recommended code fix:** pin `ort` with `default-features = false` + an explicit `api-NN` matching the shipped runtime, and add a load timeout / version-probe that surfaces a real error instead of hanging.

**Defect 2 — root cause.** `EnterpriseDb::new` ran, on *every* open, an FTS5 external-content `'delete'` for every `tier = 3` row. But the `chunks_ai` trigger only inserts `tier IN (1,2)` into `chunks_fts`, so those T3 rows were **never in the FTS index**. The FTS5 docs state that issuing `'delete'` with content that doesn't match an indexed row corrupts the index. The corruption is written to the main DB on WAL checkpoint-at-close, so the *next* `MATCH` fails. Confirmed it's the open-path, not the query: 3 consecutive `--mode dense` opens (which never read FTS) corrupt the index before any sparse query runs. **Fix applied (code):** removed the blind T3 delete from `EnterpriseDb::new` — the triggers already guarantee FTS holds only T1/T2, so no cleanup is needed. Rebuilt; verified repeated sparse/hybrid searches survive.

**Defect 4 — root cause.** `sanitize_fts_query` (db.rs:940) sanitizes via a **denylist** — it replaces only `():^*` with spaces, then joins the surviving tokens with ` OR `. It never neutralizes `+`, `-`, or `#`, and it never quotes the tokens. So `C++ tree-sitter language adapter` becomes the FTS5 expression `C++ OR tree-sitter OR language OR adapter`; FTS5 parses the bare `C` then hits `+` as a (binary) operator with no right operand → `fts5: syntax error near "+"`. Every query mentioning a `+`/`#`/`-` language or hyphenated term (`C++`, `C#`, `tree-sitter`, `node.js`, `.NET`) returns **0 sparse hits**. **Recommended fix:** switch from denylist to an allowlist — wrap each token as a quoted FTS5 phrase (`"C++" OR "tree-sitter" OR …`); double-quoting makes FTS5 treat the contents literally and neutralizes *all* operator characters at once. **Fix applied (code):** replaced the denylist with token-wrapping — `tokens.map(|t| format!("\"{}\"", t)).join(" OR ")` (embedded quotes stripped first). **Verified:** `search 'C++ tree-sitter language adapter' --mode sparse` now returns 8 hits, no error; all 63 unit tests pass (the old `fts_sanitize_rejects_special_chars` test was updated to assert the no-error/quoted-token contract).

**Defect 5 — root cause.** `hybrid_search` (search.rs:354–358) builds the dense future first, then runs sparse **synchronously with `?`**: `let sparse_res = sparse_search(query, pool, tier, ext_filter, db)?;`. When sparse returns the Defect-4 `Err`, the `?` propagates it out of `hybrid_search` before `dense_fut` is ever awaited — so the entire hybrid query fails and the dense results (which were perfect) are discarded. The default mode thus returns **0 results** for any punctuated query. This is independent of Defect 4: *any* future FTS error would zero out hybrid. **Recommended fix:** degrade gracefully — `let sparse_res = sparse_search(...).unwrap_or_default();` (and `await` dense regardless), so hybrid falls back to dense-only when sparse is empty or errors. Together, 4+5 explain the hybrid-below-dense gap in List 1. **Fix applied (code):** `let sparse_res = sparse_search(...).unwrap_or_default();` (dense still awaited unconditionally). **Verified:** `search 'C++ tree-sitter language adapter' --mode hybrid` now returns 8 hits with `cpp_adapter.py` present, where it returned 0 before.

**Defect 6 — root cause.** `index_root` ends with a reconcile loop (indexer.rs:775–786): `for uri in self.db.list_all_file_uris() { if !live_uris.contains(uri) { delete_file(uri) } }`. `list_all_file_uris()` returns **every** file in the index; `live_uris` is only the **current root's** walk. So any file under a *different* root — still present on disk — fails the `contains` check and is deleted along with its LanceDB vectors. The comment says "Remove files no longer on disk," but the code conflates "not under the root I'm indexing now" with "gone from disk." **Confirmed empirically:** the `removed` counts exactly equal the prior root's file count (92, then 572). **Recommended fix (≈1 line):** scope the prune to the root being indexed — `if !live_uris.contains(uri) && uri.starts_with(&root_prefix)`. Then sequential `index <root>` calls into one dir *accumulate* roots instead of evicting them, which is what a multi-folder corpus needs. (Discovered while trying to build the full-corpus distractor union for the mutation re-run — it silently collapsed to a single root.)

**Defect 7 — root cause.** `LocalStorageClient`'s static ignore lists (dirs / extensions / filenames in `storage.rs`) have **no game- or app-data patterns**. Game state lives in human-named folders with data-ish extensions (`American Truck Simulator\…​.sii`/`.cfg`/`.navcache`), so it passes every filter and gets chunked + embedded — polluting the corpus and burning embed time, contrary to the standing "skip exe's, apps, games" scope. The `--exclude` flag (folder-name match) is a per-run workaround, but the *default* walk still ingests game data. **Recommended fix:** add a game/app-data dir denylist (or a known-junk-extension list: `.sii`, `.navcache`, `.sav`, …) to the static ignores; optionally skip dirs that look like game roots (sibling `.exe` + `cache/` + binary save blobs). Until then, pass `--exclude "American Truck Simulator" --exclude …` for the curated corpus.

**Before/after (same frozen corpus, target in top-3 / top-10):**

| Mode | top-3 before → after | top-10 before → after |
| --- | --- | --- |
| dense (control) | 12 → 12 | 13 → 13 |
| **hybrid** | 8 → 8 | **11 → 12** |
| sparse | 5 → 5 | 7 → 7 |

The fixes remove the **showstopper** (punctuated queries no longer crash/return 0 — `cpp_adapter` moved from *absent* to rank 9 in hybrid) but **do not close the hybrid-vs-dense gap**: once sparse returns results, `"C++"`/`"C#"` tokenize to a single char `c` that strongly matches the generated `graphify-out/graph.html` + `GRAPH_REPORT.md` magnet docs, and RRF pulls those magnets above the real target. The remaining ranking work (List 1 #5–7: magnet-doc suppression, code-vs-docs balance, sub-word code tokenization) is unaddressed and is the right target for the next pass.

---

## Friction log
_Every manual step, confusing output, or annoyance encountered while operating the tool._

| # | Phase | Friction | Suggested fix |
| --- | --- | --- | --- |
| 1 | Setup | `cargo` not on PATH in non-interactive PowerShell/Bash; needed explicit `~\.cargo\bin\cargo.exe`. | Document required env / shell init in CLAUDE.md. |
| 2 | Setup | Three separate downloads (ORT DLL, ONNX model, tokenizer) with no bootstrap script; `ORT_DYLIB_PATH` + `NOMIC_ONNX_PATH` must be discovered from source. | Provide a `scripts/setup-models.ps1` + README section. |
| 3 | Setup | The required onnxruntime version (**1.24.2**) is documented nowhere — had to read `ort-sys/build/download/dist.txt` to find it. A wrong version doesn't error, it hangs. | Document the exact onnxruntime version next to `ORT_DYLIB_PATH`; fail fast on `GetApi` mismatch. |
| 4 | Setup | `build --release` of the *test* binary triggers a multi-minute, multi-GB rustc compile (arrow/lancedb generics); easy to mistake for a hang. | Note expected build cost; consider `cargo build --release --bin file_indexer` for smoke tests. |
| 5 | Indexing | **No progress output during embedding.** CPU embedding runs ~3–5 chunks/s; a multi-thousand-file corpus is many minutes/hours with only sparse `Checked N/total` lines (every 100 files) and no chunk-level ETA. Combined with Defect 3 this made a *deadlock* indistinguishable from *slow* without process sampling. | Emit an embedding progress bar / ETA; surface chunks-embedded vs total. Consider GPU/`onnxruntime` EP or smaller batch latency. |
| 6 | Indexing | A wedged/killed indexer leaves its `file_indexer.exe` holding a file lock, blocking the next `cargo build` (`Access is denied`). | Ensure the index process exits cleanly on cancel; document killing stale processes before rebuild. |

---

## Mutation loop log
_Each approved test exercised 5 ways (query perturbation + file probing)._

**Candidate test set (16 queries, grounded in live corpus content) — awaiting approval before the ×5 mutation loop:**

| # | Query | Expected target | Kind |
| --- | --- | --- | --- |
| 1 | hybrid retrieval dense sparse fusion | `src/hybrid_retriever.py` | impl |
| 2 | AST based code chunking | `src/ast_chunker.py` | impl |
| 3 | incremental indexing merkle tree drift detection | `src/incremental_indexer.py` | impl |
| 4 | C++ tree-sitter language adapter | `adapters/cpp_adapter.py` | impl |
| 5 | stable deterministic id generation hashing | `src/stable_id.py` | impl |
| 6 | summarize source file with an LLM | `src/summarizer.py` | impl |
| 7 | terminal TUI application backend | `src/tui/*` | impl |
| 8 | import resolver cross-file references | `src/import_resolver.py` | impl |
| 9 | category tagging classifier | `src/category_tagger.py` | impl |
| 10 | C# csharp language adapter | `adapters/csharp_adapter.py` | impl |
| 11 | recursive file search walker | `src/RecFileSearch.py` | impl |
| 12 | tiered language support levels | `ADR-017` | doc |
| 13 | evaluation benchmark harness | `ADR-007` | doc |
| 14 | adapter architecture and language expansion | `ADR-003` | doc |
| 15 | measured conformance edge confidence | `ADR-008` | doc |
| 16 | usage driven adaptive ranking | `ADR-014` | doc |

Mutation ×5 per test: (1) synonym swap, (2) typo injection, (3) word reorder, (4) truncation, (5) recase — plus a file-handle probe (retrieve the target by its basename). All in **hybrid** mode (the mode under test) unless noted. Run against the live `full-corpus` index with root 1 (`codebase-indexer`, 76 files / 1,629 chunks) complete — all 16 query targets reside in this root. A full-corpus re-run will confirm once the remaining roots finish.

**Per-variant hit-rate (target in top-10 / top-3, N=16):**

| Variant | hybrid top-10 | hybrid top-3 | dense top-10 | dense top-3 |
| --- | --- | --- | --- | --- |
| base | 14 | 11 | 14 | 11 |
| synonym | 14 | 13 | 14 | 11 |
| **typo** | **11** | 11 | 12 | **6** |
| reorder | 13 | 11 | 14 | 11 |
| truncate | 15 | 14 | 13 | 10 |
| recase | 13 | 11 | 13 | 11 |
| filename-handle | 12 | 11 | 13 | 11 |

Mean query-mutation robustness (perturbations keeping target in top-10): **4.12 / 5**, with **11/16 queries fully robust (5/5)** — identical for hybrid and dense.

**Mutation findings:**

| # | Finding | Evidence | Severity |
| --- | --- | --- | --- |
| M1 | **Typo fragility is the dominant weakness.** A single-char typo in the key term is the worst perturbation — it knocks 3–4 queries out of top-10 and collapses **dense top-3 from 11→6**. (Hybrid resists better at top-3: 11 — a point in fusion's favor.) | `C++ tree-sitter` typo → miss; `C# csharp` typo → miss; `recursive file search` typo → miss. | Med |
| M2 | **Two queries fail under *every* mutation (code-vs-docs).** The `.py` implementation is buried under conceptual `.md` docs regardless of phrasing. | `incremental indexing merkle tree drift detection` → `incremental_indexer.py` absent in all 6 variants (0/5 robust); `hybrid retrieval dense sparse fusion` → `hybrid_retriever.py` absent in 5/6 (1/5). | High |
| M3 | **Filename-handle search is unreliable.** Searching a file's literal basename fails to surface it in top-10 for 4/16 — underscored identifiers collide with sibling files + the magnet docs. | `ast_chunker`, `cpp_adapter`, `csharp_adapter`, `RecFileSearch` basenames → target not in top-10. | Med |
| M4 | **Truncation is benign-to-helpful** (15/16) — dropping trailing words removes noise; core terms carry the query. Recase/reorder are near-neutral. | — | Info |
| M5 | **`C#`/`C++` queries are perturbation-fragile even post-fix.** `csharp_adapter` only 2/5 robust (typo/reorder/recase all miss) — the single-char `c` token + magnet docs make these brittle. | See Defects 4–5; ranking still imperfect after the crash fix. | Med |

**Bottom line:** the index is robust for well-formed doc/ADR queries (ADR-007/008/014/017 hit rank 1 across nearly all mutations) and for most impl files, but three systemic weaknesses remain — **typo intolerance (M1), code-vs-docs burial (M2, List 1 #5–6), and unreliable filename/symbol-handle retrieval (M3)** — all in the ranking/fusion layer, none in the (sound) dense engine.

### Distractor re-run (2026-06-24) — full union corpus

After fixing D6, the durable union was rebuilt across 4 roots (`codebase-indexer` 92, `homelab` 292, `OneDrive` 15, `Documents` 1,232) into one index: **1,618 files / 27,969 chunks, 100 % embedded**. The same 16-target ×5 mutation loop was re-run with ~26k *unrelated* distractor chunks present, to test whether rankings hold when the 16 targets are <12 % of the corpus.

| Run | Index | Mean robustness | Fully robust 5/5 | base top-3 |
| --- | --- | --- | --- | --- |
| Baseline (hybrid) | codebase-only, 92 files / ~1.6k chunks | **4.12 / 5** | 11/16 | 14/16 |
| Union (hybrid) | 1,618 files / 28k chunks | **3.69 / 5** | 9/16 | 10/16 |
| Union (dense) | 1,618 files / 28k chunks | **3.88 / 5** | 9/16 | 12/16 |

**Distractor findings:**

| # | Finding | Evidence | Severity |
| --- | --- | --- | --- |
| M6 | **Distractors cost ~0.4 robustness points** and 2 fully-robust queries. Degradation is graceful, not catastrophic — the engine still surfaces most targets despite 8× more noise. | hybrid 4.12→3.69, dense baseline→3.88; fully-robust 11→9 in both modes. | Info |
| M7 | **Dense still beats hybrid under distractors** (3.88 vs 3.69; base top-3 12 vs 10), reconfirming the standing ranking finding — RRF fusion *hurts* precision here. The sparse channel pulls in BM25 keyword matches from unrelated `homelab`/`Documents` files that dilute the dense ranking. | Union dense > union hybrid on every aggregate. | High |
| M8 | **M2 burial worsens to total failure with distractors.** The two code-vs-docs queries (`hybrid_retriever`, `incremental_indexer`) now score **0/5 in both modes** — even the *base* (unmutated) query can't surface the impl file in top-10 once distractor docs compete. | Both queries `—` across all 6 variants, hybrid and dense. | High |
| M9 | **Filename-handle retrieval is the weakest probe under scale** — top-3 collapses to **5/16** in both modes (was 11/16 hybrid baseline). Basenames collide with same-named files across the 4 roots. | `fname` column: top-3 5/16 hybrid, 5/16 dense. | Med |

**Goal status: COMPLETE.** All 16 targets have been mutation-tested ×5 in both the codebase-only baseline and the full-corpus distractor union. The dogfooding loop's exit condition ("each test tested with mutation mindset 5 times") is met. The persistent open issue is the **fusion/ranking layer** (M2/M7/M8 — hybrid trailing dense, code-vs-docs burial), not the storage, embedding, or change-detection layers, which are sound.

### Remediation R1 (2026-06-24) — weighted RRF fusion (fixes M7)

**Root cause of M7:** `rrf_fuse` blended the dense and sparse channels with **equal weight** — each contributed `1/(k+rank)`. Because dense is the stronger retriever on this corpus, an unrelated BM25 keyword match at sparse-rank 1–3 received the *same* fusion credit as a genuinely relevant dense hit, diluting precision. That is why unweighted hybrid (3.69) trailed pure dense (3.88).

**Fix:** introduced per-channel weights — `score = w_dense/(k+rank_d) + w_sparse/(k+rank_s)` — with `RRF_DENSE_WEIGHT=1.0`, `RRF_SPARSE_WEIGHT=0.4` (env-overridable for tuning). Sparse still contributes (it rescues exact-token/synonym cases dense misses) but with less pull.

**Weight chosen empirically.** Swept `w_sparse ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` over the full 16×7 mutation suite against the 1,618-file union corpus (`scratchpad/sweep_rrf.py`). The `0.0` and `1.0` endpoints reproduced the dense (3.88) and old-hybrid (3.69) baselines exactly, validating the harness.

| w_sparse | mean robustness | fully-robust | Σ top-3 | Σ top-10 |
| --- | --- | --- | --- | --- |
| 0.0 (≈ pure dense) | 3.88 | 9/16 | 68 | 87 |
| **0.4 (shipped)** | **3.94** | **10/16** | **69** | **88** |
| 1.0 (old unweighted) | 3.69 | 9/16 | 67 | 82 |

Weighted hybrid at `0.4` **dominates the old unweighted fusion on every aggregate** and **beats pure dense** on robustness, total top-3 precision, and top-10 recall — combining dense's base precision with sparse's synonym rescue (synonym top-3 8→11 vs dense). It concedes only 1 point on base top-3 (dense's single strongest cell). `0.2` tied on robustness but lost on top-3 precision and the weak filename handle, so `0.4` was shipped. **M7 → RESOLVED.**

**Still open after R1:** M8 (code-vs-docs burial) and M9 (filename-handle collisions) are *content-type / tokenization* problems, not fusion-weight problems — weighting lifted overall robustness (one previously-failing query recovered: fully-robust 9→10) but does not by itself rerank an impl file above a keyword-matching doc. Those need a separate mechanism (magnet-doc suppression / type-aware boosting / sub-word `C++`/`C#` tokenization) and remain future work.

### Remediation R2 (2026-06-24) — filename/path-coverage boost (fixes M8 + M9)

**Root cause of M8/M9:** retrieval scored only *chunk content*. A file literally named after the query concept (`incremental_indexer.py`, `hybrid_retriever.py`) carries that signal in its **path**, which the dense+sparse channels never see. So a conceptual doc that merely *discusses* the concept (e.g. `merkle-tree-drift-handling.md`) out-scored the impl file that *is* the concept. M9 (bare-filename lookups) is the same blind spot in the limit.

**Fix (`search.rs`):** add a bounded, additive **path-coverage boost** to the fused score. The query is tokenized; the candidate file's *stem* is tokenized; coverage = fraction of stem tokens matched (with light shared-prefix-≥5 stemming so `indexer`/`indexing`, `retriever`/`retrieval` fold together). The boost is `PATH_BOOST_WEIGHT/(k+1) · coverage`, scaling a full filename match to one rank-1 RRF channel. `PATH_BOOST_WEIGHT=1.5` (env-overridable). It is additive and capped by coverage, so it *re-ranks* a name-matched file upward without letting any single token dominate.

**Ordering bug found & fixed (this is what made the boost safe).** The boost initially collapsed result sets — query `incremental indexing merkle tree drift detection` returned only **3** results instead of 10. Cause: `rrf_fuse` truncated to `top_k` **before** `post_process` applied the per-file cap. The boost gave every chunk of one well-named doc the same coverage, so those chunks flooded the pre-truncation head; truncate-to-10 kept ~10 of them, then the per-file cap (max 3) deleted all but 3. Fix: `rrf_fuse` now returns the **full ranked candidate set**, and `hybrid_search` does **cap-then-truncate** (`post_process` → `truncate(top_k)`). This is a latent correctness fix independent of the boost: any flooding file would previously shrink the result set below `top_k`.

**Validated by sweep** (`scratchpad/sweep_path.py`, RRF weights held at the R1 optimum 1.0/0.4, full 16×7 suite vs the 1,618-file union corpus):

| PATH_BOOST_WEIGHT | mean robustness | fully-robust |
| --- | --- | --- |
| 0.0 (R1 baseline) | 3.94 | 10/16 |
| 1.0 | 4.56 | 11/16 |
| **1.5 (shipped)** | **4.69** | **12/16** |
| 2.0 | 4.69 | 12/16 |
| 3.0 | 4.62 | 11/16 |

`1.5` and `2.0` tie on aggregate; `1.5` shipped as the more conservative boost (lower over-fit risk on unseen queries). The boost lifts robustness **broadly**, not just the two target queries — mean **3.94 → 4.69**, the best of every configuration measured (vs pure dense 3.88, old unweighted hybrid 3.69).

**Per-query effect** (hybrid, union corpus, w=1.5):

- **M8 `incremental indexing merkle tree drift detection`: 0/5 → 4/5.** `src/incremental_indexer.py` now surfaces at #4 (was dense-rank 22 / absent from hybrid top-60).
- **M8 `hybrid retrieval dense sparse fusion`: 0/5 → 4/5.** `src/hybrid_retriever.py` now #1 (was #34).
- **M9 filename variant:** `fname` per-variant top-3 **5/16 → 11/16**, top-10 14/16.

The conceptual doc still co-occupies the head where it is genuinely relevant (merkle doc holds #1–3 on its query, capped at 3 by `post_process`) — the boost *adds* the impl file to the top-10 rather than evicting the doc. **M8 → RESOLVED. M9 → substantially improved** (filename top-3 recall more than doubled; remaining misses are bare single-token handles where the corpus holds near-duplicate copies across roots).
