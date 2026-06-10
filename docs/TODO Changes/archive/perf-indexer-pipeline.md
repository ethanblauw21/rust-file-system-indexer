# COMPLETED

# Perf — Indexer Pipeline Parallelism (`src/indexer.rs` + `src/main.rs`)

**Status:** Pending

## Goal

Three related inefficiencies in the indexing path:

1. **Phase barrier** — Phase 1 (rayon I/O + chunking) fully completes before Phase 2
   (DB writes + embedding) begins. The ONNX embedder sits idle during Phase 1; rayon
   sits idle during Phase 2. Pipelining them via a producer-consumer channel halves
   effective wall-clock time for large directories.

2. **Double directory traversal** — `run_index` in `main.rs` walks the directory tree
   once to get a file count for the header, then `index_root` walks it again internally.
   Two full filesystem traversals for the same data.

3. **Width computation coupling** — `run_index` captures `total_hint` from the pre-scan
   to format progress column widths. This is the only reason the pre-scan exists; the
   `on_progress` callback already receives `total` from inside `index_root`, so the
   pre-scan is entirely redundant.

---

## Files Affected

| File | Change |
|------|--------|
| `src/indexer.rs` | Add `on_start` callback; pipeline phases via `tokio::sync::mpsc`; wrap `stored_info` in `Arc` |
| `src/main.rs` | Remove pre-scan; use `on_start` for header; derive width from `total` in `on_progress` |

---

## Approach

### 1. `src/indexer.rs` — pipeline + `on_start`

#### New signature

```rust
pub async fn index_root(
    &self,
    root_uri:    &str,
    reindex:     bool,
    on_start:    Option<&dyn Fn(usize)>,        // ← new: fires with total file count
    on_progress: Option<&dyn Fn(usize, usize, &Stats)>,
) -> Result<Stats, IndexerError>
```

#### Fire `on_start` after listing files, before any work

```rust
let total = file_uris.len();
if let Some(f) = on_start { f(total); }
```

#### Wrap `stored_info` in `Arc`

`stored_info` is needed both inside the rayon closure (read: change detection) and after
Phase 2 (ghost-file removal). Wrapping it allows both sides to share ownership without
cloning the whole map.

```rust
let stored_info = Arc::new(stored_info);
let stored_info_bg = stored_info.clone();  // moved into spawn_blocking
```

The ghost-removal step after the pipeline uses `stored_info` (the Arc) unchanged — deref
works transparently.

#### Replace `par_iter().collect()` + serial for-loop with a channel pipeline

```rust
// Bounded channel; backpressure stops rayon outrunning the embedder.
let (tx, mut rx) = tokio::sync::mpsc::channel::<FileAction>(256);

let storage_bg = self.storage.clone();

// Phase 1: rayon I/O + chunking on a blocking thread pool.
// blocking_send is safe here — spawn_blocking threads are outside the tokio runtime.
let phase1 = tokio::task::spawn_blocking(move || {
    use rayon::prelude::*;
    file_uris.par_iter().for_each(|uri| {
        let info = stored_info_bg.get(uri.as_str());
        let action = process_file_sync(
            storage_bg.as_ref(), &FileChunker::new(), uri, info,
        );
        let _ = tx.blocking_send(action);
        // tx is dropped when the closure exits, closing the channel
    });
});

// Phase 2: async DB writes + embedding, runs concurrently with Phase 1.
let mut checked = 0usize;
while let Some(action) = rx.recv().await {
    checked += 1;
    match action {
        FileAction::Skip  => stats.skipped += 1,
        FileAction::Error => stats.errors  += 1,
        FileAction::Process { meta, hash, chunk_result, existing_file_id } => {
            if let Some(file_id) = existing_file_id {
                let old_lance_ids = self.db.get_lance_ids_for_file(file_id)?;
                self.vectors.remove_ids(&old_lance_ids).await?;
            }
            self.write_file(
                &meta, &hash, chunk_result,
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

phase1.await.map_err(|e| IndexerError::Other(e.to_string()))?;
```

Note: `checked` replaces `i + 1` from the old enumerated loop. Ordering from rayon is
non-deterministic, but `checked` only counts arrivals — it doesn't need to be in file order.

#### Update test call sites

Both tests pass `None` for `on_progress`; add a `None` for `on_start`:

```rust
indexer.index_root(&root_dir.path().to_string_lossy(), false, None, None).await.unwrap()
```

---

### 2. `src/main.rs` — remove pre-scan, use `on_start`

#### Remove the pre-scan

Delete:
```rust
let total_hint = storage.list_files(&root_str).count();
println!("Indexing {}  ({} files)", root_str, fmt_num(total_hint as i64));
```

#### Add `on_start` closure

```rust
let root_str_for_start = root_str.clone();
let on_start = move |total: usize| {
    println!("Indexing {}  ({} files)", root_str_for_start, fmt_num(total as i64));
};
```

#### Derive column width inside `on_progress`

`on_progress` already receives `total` from `index_root`, so the width is always
available without external state:

```rust
let on_progress = |checked: usize, total: usize, s: &Stats| {
    let width = fmt_num(total as i64).len();
    println!(
        "  [{:>width$} / {:>width$}]  indexed={:<8}  skipped={:<8}  errors={}",
        fmt_num(checked as i64), fmt_num(total as i64),
        fmt_num(s.indexed as i64), fmt_num(s.skipped as i64), s.errors,
        width = width,
    );
};
```

#### Update `index_root` call

```rust
match indexer.index_root(&root_str, reindex, Some(&on_start), Some(&on_progress)).await {
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| `spawn_blocking` error propagation | `phase1.await` returns `JoinError` on panic; map to `IndexerError::Other` |
| Channel backpressure stalls rayon | Capacity 256 gives ~1 batch of headroom; rayon blocks on `blocking_send` — this is correct behaviour, not a deadlock |
| `stored_info` Arc clone overhead | HashMap clone avoided entirely; only the Arc refcount is incremented |
| `on_start` / `on_progress` ordering | `on_start` is fired before `spawn_blocking`, so the header always prints before any progress lines |
| Test call-site arity mismatch | Two tests in `indexer.rs`; grep for `index_root(` to catch all sites |

---

## Verification Steps

1. `cargo build` — zero errors, zero warnings.
2. `cargo test` — all 36 existing tests pass.
3. `cargo run -- index --root <dir>` — header line appears immediately, then progress
   lines; no duplicate directory walk visible in `RUST_LOG=debug` output.
4. On a large directory (1 000+ files), confirm wall-clock time is lower than before
   (ONNX + rayon overlap should be visible in `--timings` or manual measurement).
5. `cargo run -- index --root <dir>` with `NOMIC_ONNX_PATH` unset — progress still
   works, no panic (embedder is `None`, `flush_embeddings` returns early).
