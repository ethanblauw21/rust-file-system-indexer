# Step 8 — Interactive TUI (`src/tui.rs` + `src/main.rs` + `Cargo.toml`)

**Status:** Pending

## Goal

Add a `tui` subcommand that launches a full-screen terminal UI backed by
`ratatui` + `crossterm`. The UI replaces the need to re-type `cargo run --
search` for every query and surfaces stats, explain, and index-trigger
functionality without leaving the terminal.

```
cargo run -- tui --index-dir .fileSystem-index
```

### Desired layout

```
┌─ File Indexer ── hybrid ──────────────────────────────────────────────────┐
│ Search: jwt middleware_                                                    │
├────────────────────────────────────────────────────────────────────────────┤
│   1  87%  T2  src/middleware.rs          pub fn validate_jwt(token: &str… │
│ ▶ 2  82%  T1  docs/auth.md              JWT middleware validates token…   │
│   3  74%  T1  src/auth/token.rs         const JWT_SECRET: &str = …       │
│                                                                            │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│ 10 results · ↑↓ navigate · ⏎ open · s stats · e explain · i index · q quit│
└────────────────────────────────────────────────────────────────────────────┘
```

Keybindings:
| Key | Action |
|-----|--------|
| Type | Update search query |
| ↑ / ↓ | Move selection |
| Enter | Open selected file (VS Code / default handler) |
| `s` | Toggle stats overlay |
| `e` | Toggle explain panel for selected result |
| `i` | Trigger `index_root` on the configured root; show live progress |
| Esc | Close overlay / clear query |
| `q` | Quit |

---

## Files Affected

| File | Change |
|------|--------|
| `Cargo.toml` | Add `ratatui = "0.30"`, `crossterm = "0.29"` |
| `src/main.rs` | New `Tui` variant in `Command` enum; `run_tui()` entry point |
| `src/tui.rs` | **New file** — all TUI logic (App struct, event loop, widgets) |

---

## Approach

### 1. `Cargo.toml` — new dependencies

```toml
ratatui   = "0.30"
crossterm = { version = "0.29", features = ["event-stream"] }
```

`event-stream` enables async crossterm event polling via a `tokio` stream,
avoiding a blocking thread.

### 2. `src/main.rs` — new subcommand

```rust
/// Launch the interactive TUI
Tui {
    #[arg(long, default_value = ".fileSystem-index")]
    index_dir: PathBuf,
    /// Root directory for the index trigger (i key)
    #[arg(long)]
    root: Option<PathBuf>,
},
```

`run_tui` just constructs a `Searcher`, hands it to `tui::App::new()`, and
calls `app.run()`.

### 3. `src/tui.rs` — App struct and state machine

```rust
enum Screen {
    Search,   // default: input box + results list
    Stats,    // overlay showing DbStats
    Explain,  // overlay showing chunks for selected result
    Indexing, // full-screen progress panel
}

struct App {
    searcher:    Searcher,
    query:       String,
    results:     Vec<SearchResult>,
    selected:    usize,
    screen:      Screen,
    stats_cache: Option<IndexStats>,
    explain_data: Option<ExplainData>,
    index_root:  Option<PathBuf>,   // set from CLI --root arg
    index_log:   Vec<String>,       // progress lines during indexing
    debounce_tx: mpsc::Sender<String>,
}
```

### 4. Event loop

Use `crossterm::event::EventStream` (async) inside a `tokio::select!` loop:

```
loop {
    select! {
        event = event_stream.next() => handle_key(event),
        result = debounce_rx.recv() => execute_search(result),
        progress = index_progress_rx.recv() => append_log(progress),
    }
}
```

- **Debounce**: on each keypress, send the current query into a
  `tokio::time::sleep`-gated channel. The search only fires after 150 ms of
  inactivity.
- **Index**: spawn a `tokio::task` that calls `index_root` with an
  `on_progress` closure that sends lines back over an `mpsc` channel.

### 5. Widgets (ratatui)

- **Search bar**: single-line `Paragraph` inside a `Block`, cursor drawn
  manually with a trailing `_`.
- **Results list**: `List` widget with `ListState` for selection highlight.
  Each item: `{rank}  {score}%  T{tier}  {source:<40}  {preview}`.
- **Status bar**: fixed 1-line `Paragraph` at the bottom with key hints.
- **Stats overlay**: centered `Block` with `Table` rows — mirrors the
  `--stats` output (file count, MIME breakdown, chunk/vector counts). Reuses
  `DbStats` directly; will automatically gain `mime_counts` once step 7 lands.
- **Explain overlay**: scrollable `List` of chunks — tier, index, token count,
  embedded indicator (`E`/`·`), first 60 chars of content.
- **Indexing panel**: replaces results area with a scrolling log of progress
  lines piped from `on_progress`.

### 6. Opening files

On Enter, call the same logic already used by the paths block:
```rust
std::process::Command::new("cmd")
    .args(["/c", "start", "", &path])
    .spawn()
    .ok();
```
This respects the user's default file association (VS Code for code files,
etc.) without hardcoding an editor.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Async + ratatui event loop complexity | Keep a single `tokio::select!` loop; no `std::thread`; debounce and index channels decouple long-running work |
| Windows terminal raw mode conflicts with other tools | `crossterm` handles Windows raw mode; restore terminal on panic via `std::panic::set_hook` |
| Index trigger blocks UI if run synchronously | Always spawn into a `tokio::task`; communicate back only via `mpsc` messages |
| ratatui version churn | Pin to `0.30`; the widget API stabilised in 0.26+ |
| Stats overlay incomplete until step 7 | Show available fields now; MIME breakdown and coverage rows render as `(pending step 7)` placeholders, replaced automatically once that step lands |

---

## Verification Steps

1. `cargo build` — zero errors, zero warnings.
2. `cargo run -- tui` — full-screen UI opens; type a query and results appear
   within ~200 ms.
3. `↑`/`↓` moves the selection highlight; `Enter` opens the file in VS Code.
4. `s` opens the stats overlay; counts match `cargo run -- search --stats`.
5. Select a result and press `e` — explain overlay shows the same chunks as
   `cargo run -- explain <file>`.
6. `i` (with `--root` provided) triggers indexing; progress lines scroll in
   the Indexing panel; UI returns to Search when done.
7. `q` restores the terminal cleanly (no stray escape sequences).
8. Pipe the binary output to a file (`cargo run -- tui > /dev/null`) — raw
   mode is not entered, the process exits immediately with a clear error.
