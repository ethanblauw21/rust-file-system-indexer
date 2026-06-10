# Step 6 — Live Index Progress & Elapsed Time (`indexer.rs` + `main.rs`)

**Status:** Complete

## Goal

The `index` subcommand is currently a black box for large directories: all progress
goes to `tracing::info!` (hidden unless `RUST_LOG` is set) and the user sees
nothing until the final done line appears — potentially minutes later.

This step surfaces progress directly to stdout and enriches the done summary with
elapsed time and final vector count.

## Desired Output

```
Indexing C:\Users\edb\Documents\  (3,241 files)
  [ 100 / 3,241]  indexed=12    skipped=88     errors=0
  [ 200 / 3,241]  indexed=31    skipped=169    errors=0
  [ 500 / 3,241]  indexed=78    skipped=422    errors=0
  [1000 / 3,241]  indexed=142   skipped=858    errors=0
  [2000 / 3,241]  indexed=286   skipped=1,714  errors=0
  [3,241 / 3,241]  indexed=312   skipped=2,929  errors=0

Done in 1m 43s  —  312 indexed, 2,929 skipped, 0 removed, 0 errors
  14,208 vectors in index
```

Progress lines print at: every 100 files, and on the final file.

## Changes Required

### 1. `src/indexer.rs` — `Stats` struct

Add `vec_total` so `index_root` can report the final vector count without the
caller needing to open a second `Searcher`:

```rust
#[derive(Debug, Default)]
pub struct Stats {
    pub indexed:   usize,
    pub skipped:   usize,
    pub removed:   usize,
    pub errors:    usize,
    pub vec_total: usize,   // ← new
}
```

Populate it at the end of `index_root`, after the removal phase:

```rust
stats.vec_total = self.vectors.ntotal().await?;
Ok(stats)
```

### 2. `src/indexer.rs` — `index_root` progress callback

Add an optional progress callback to `index_root`:

```rust
pub async fn index_root(
    &self,
    root_uri: &str,
    reindex:  bool,
    on_progress: Option<&dyn Fn(usize, usize, &Stats)>,  // (checked, total, running_stats)
) -> Result<Stats, IndexerError>
```

Inside the action-processing loop (around line 643), call the callback at the
existing checkpoint intervals:

```rust
if total >= 20 && (checked % 100 == 0 || checked == total) {
    tracing::info!(...);
    if let Some(f) = on_progress {
        f(checked, total, &stats);
    }
}
```

### 3. `src/main.rs` — `run_index`

Wire up the callback and print lines directly to stdout:

```rust
async fn run_index(root: PathBuf, index_dir: PathBuf, reindex: bool, exclude: Vec<String>) {
    // ...existing setup...

    let total_hint = storage.list_files(&root_str).count();   // optional pre-scan
    let t0 = std::time::Instant::now();

    let on_progress = |checked: usize, total: usize, s: &Stats| {
        println!(
            "  [{:>width$} / {:>width$}]  indexed={:<8}  skipped={:<8}  errors={}",
            fmt_num(checked as i64),
            fmt_num(total as i64),
            fmt_num(s.indexed as i64),
            fmt_num(s.skipped as i64),
            s.errors,
            width = fmt_num(total as i64).len(),
        );
    };

    match indexer.index_root(&root_str, reindex, Some(&on_progress)).await {
        Ok(stats) => {
            let elapsed = t0.elapsed();
            println!();
            println!(
                "Done in {}  —  {} indexed, {} skipped, {} removed, {} errors",
                format_elapsed(elapsed),
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
```

### 4. `src/main.rs` — `format_elapsed` helper

```rust
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
```

## Pre-scan consideration

The total file count (`3,241 files`) requires iterating the storage iterator once
before indexing begins. `LocalStorageClient::list_files` is a lazy DFS iterator,
so the pre-scan takes ~1–5 ms on local SSDs for typical directory sizes.

If the pre-scan is skipped, progress lines should omit the total and use a running
counter format: `[  100 checked]  indexed=12  skipped=88  errors=0`.

## Tests

No new tests required — this is purely observability. Existing `index_root_skips_unchanged_files`
test must be updated to pass `None` for `on_progress`.
