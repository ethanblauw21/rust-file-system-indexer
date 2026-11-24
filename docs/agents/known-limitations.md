# Known limitations / recommended follow-ups

Moved from CLAUDE.md, verbatim except the line-number fixes noted in the PR. Read when working on this area. Line-number citations drift; re-check them against the code.

### Known Limitations / Recommended Follow-ups

- **Hybrid search is not actually parallel.** `hybrid_search` (`search.rs:506-536`) constructs the dense
  search `Future` (`dense_fut`, line 520), then calls `sparse_search(...)` *synchronously* and runs it
  to completion (line 524), and only then `.await`s `dense_fut` (line 525). Rust futures are lazy —
  nothing runs until polled — so despite the function being `async`, the sparse and dense channels
  execute strictly back-to-back on the calling task, not concurrently. Any doc that says dense and
  sparse "run in parallel" is describing an unshipped design, not the current code.
  **Recommended follow-up (not done this pass — a code change, out of scope for a docs-only edit):**
  wrap the blocking `sparse_search` call in `tokio::task::spawn_blocking` (it does synchronous SQLite
  I/O) and drive both channels concurrently with `tokio::join!(dense_fut, spawn_blocking(sparse_search))`.
  This would change *latency* (the two channels genuinely overlap) but not *result semantics* — fusion
  math is unaffected. Should go through `rust-implementer`, and any latency claim should be re-measured
  (e.g. by `search-eval-scientist`) rather than asserted.
- **`required_onnxruntime_version` is parsed but not enforced.** `config.rs`'s `RawEmbedderConfig`
  parses `[embedder].required_onnxruntime_version` so the field name is stable in `file_indexer.toml`,
  but nothing currently checks the loaded ORT dylib's actual version against it (`config.rs:33-38`,
  `#[allow(dead_code)]`). The version-mismatch protection that exists today is the load-timeout in
  `Embedder::load` (see the ONNX Runtime version pin bullet above), which detects a mismatch
  indirectly (by timing out) rather than by comparing version strings.
