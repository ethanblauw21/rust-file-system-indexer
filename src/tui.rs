use crate::indexer::{IncrementalIndexer, Stats};
use crate::search::{ExplainData, IndexStats, SearchMode, SearchOptions, SearchResult, Searcher};
use crate::storage::LocalStorageClient;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use std::{
    io,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

// ── Palette ───────────────────────────────────────────────────────────────────

const NAVY: Color = Color::Rgb(29, 66, 138);
const GOLD: Color = Color::Rgb(240, 179, 35);
const DEBOUNCE_MS: u64 = 150;

// ── Screen state ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Screen {
    Search,
    Stats,
    Explain,
    ChunkPreview,
    Indexing,
    RootPrompt,
    Flagged,
    Scores,
}

// ── App ───────────────────────────────────────────────────────────────────────

pub struct App {
    searcher:          Searcher,
    index_dir:         PathBuf,
    query:             String,
    results:           Vec<SearchResult>,
    selected:          usize,
    list_offset:       usize,
    screen:            Screen,
    search_mode:       SearchMode,
    has_embedder:      bool,
    stats_data:        Option<IndexStats>,
    explain_data:      Option<ExplainData>,
    explain_scroll:    u16,
    explain_chunk_sel: usize,
    chunk_preview:     Option<String>,
    chunk_preview_scroll: u16,
    index_root:        Option<PathBuf>,
    index_log:         Vec<String>,
    index_log_scroll:  usize,
    root_input:        String,
    debounce_deadline: Option<Instant>,
    progress_rx:       Option<std::sync::mpsc::Receiver<String>>,
    score_rx:          Option<std::sync::mpsc::Receiver<String>>,
    status_msg:        Option<(String, Instant)>,
    flagged_data:      Option<Vec<crate::db::FlaggedSummaryRow>>,
    scores_data:       Option<Vec<crate::db::ChunkRow>>,
    scores_scroll:     usize,
    search_focused:    bool, // true = cursor in search bar, false = cursor on results
}

impl App {
    pub fn new(searcher: Searcher, index_dir: PathBuf, index_root: Option<PathBuf>) -> Self {
        let has_embedder = searcher.embedder.is_some();
        let search_mode = if has_embedder { SearchMode::Hybrid } else { SearchMode::Sparse };
        Self {
            searcher,
            index_dir,
            query: String::new(),
            results: Vec::new(),
            selected: 0,
            list_offset: 0,
            screen: Screen::Search,
            search_mode,
            has_embedder,
            stats_data: None,
            explain_data: None,
            explain_scroll: 0,
            explain_chunk_sel: 0,
            chunk_preview: None,
            chunk_preview_scroll: 0,
            index_root,
            index_log: Vec::new(),
            index_log_scroll: 0,
            root_input: String::new(),
            debounce_deadline: None,
            progress_rx: None,
            score_rx: None,
            status_msg: None,
            flagged_data: None,
            scores_data: None,
            scores_scroll: 0,
            search_focused: true,
        }
    }

    pub async fn run(&mut self) -> io::Result<()> {
        use std::io::IsTerminal;
        if !io::stdout().is_terminal() {
            eprintln!("error: TUI requires a terminal (stdout is redirected)");
            std::process::exit(1);
        }

        let orig_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let _ = execute!(io::stdout(), LeaveAlternateScreen);
            orig_hook(info);
        }));

        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = self.event_loop(&mut terminal).await;

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        result
    }

    // ── Main loop ─────────────────────────────────────────────────────────────

    async fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> io::Result<()> {
        loop {
            // Fire debounce if deadline passed
            if let Some(dl) = self.debounce_deadline
                && Instant::now() >= dl {
                    self.debounce_deadline = None;
                    self.execute_search().await;
                }

            // Drain index/score progress (non-blocking)
            self.drain_progress();
            self.drain_score_progress();

            terminal.draw(|f| self.render(f))?;

            // Compute poll timeout: short when debounce or indexing is active
            let timeout = if let Some(dl) = self.debounce_deadline {
                dl.saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(50))
            } else if self.progress_rx.is_some() || self.score_rx.is_some() {
                Duration::from_millis(50)
            } else {
                Duration::from_millis(200)
            };

            if event::poll(timeout)? {
                let ev = event::read()?;
                if self.handle_event(ev).await? {
                    break;
                }
            }
        }
        Ok(())
    }

    fn drain_progress(&mut self) {
        let Some(rx) = &self.progress_rx else { return };
        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    self.index_log.push(msg);
                    self.index_log_scroll = self.index_log.len().saturating_sub(1);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.progress_rx = None;
                    self.set_status("Indexing complete.");
                    self.screen = Screen::Search;
                    break;
                }
            }
        }
    }

    fn drain_score_progress(&mut self) {
        let Some(rx) = &self.score_rx else { return };
        let mut last_msg: Option<String> = None;
        let mut done = false;
        loop {
            match rx.try_recv() {
                Ok(msg)                                       => { last_msg = Some(msg); }
                Err(std::sync::mpsc::TryRecvError::Empty)    => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => { done = true; break; }
            }
        }
        if let Some(msg) = last_msg {
            self.set_status(msg);
        }
        if done {
            self.score_rx = None;
        }
    }

    fn start_scoring(&mut self, rescore: bool) {
        use crate::scorer::score_all;

        let (tx, rx) = std::sync::mpsc::sync_channel::<String>(256);
        self.score_rx = Some(rx);

        let db    = self.searcher.db.clone();
        let lance = self.searcher.vectors.clone();

        tokio::spawn(async move {
            let tx2 = tx.clone();
            let on_progress = move |checked: usize, total: usize| {
                let _ = tx2.try_send(format!("Scoring\u{2026} {checked} / {total} chunks"));
            };
            match score_all(&db, &lance, rescore, Some(&on_progress)).await {
                Ok(s) => {
                    let _ = tx.try_send(format!(
                        "Scored {} chunks \u{2014} {} flagged ({} structural, {} coherence, {} both)",
                        s.total, s.flagged, s.structural_only, s.coherence_only, s.both,
                    ));
                }
                Err(e) => { let _ = tx.try_send(format!("Score error: {e}")); }
            }
            // tx drops here — signals TUI that scoring is complete
        });

        self.set_status("Scoring\u{2026}");
    }

    fn handle_flagged_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Esc       => { self.screen = Screen::Search; self.flagged_data = None; }
            _ => {}
        }
        false
    }

    fn handle_scores_key(&mut self, key: KeyEvent) -> bool {
        let row_count = self.scores_data.as_ref().map(|r| r.len()).unwrap_or(0);
        match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Esc       => { self.screen = Screen::Search; self.scores_data = None; }
            KeyCode::Up        => { self.scores_scroll = self.scores_scroll.saturating_sub(1); }
            KeyCode::Down
                if self.scores_scroll + 1 < row_count => { self.scores_scroll += 1; }
            _ => {}
        }
        false
    }

    // ── Event dispatch ────────────────────────────────────────────────────────

    async fn handle_event(&mut self, event: Event) -> io::Result<bool> {
        if let Event::Key(key) = event {
            return Ok(self.handle_key(key).await);
        }
        Ok(false)
    }

    async fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return true;
        }
        match self.screen {
            Screen::Search       => self.handle_search_key(key).await,
            Screen::Stats        => self.handle_stats_key(key),
            Screen::Explain      => self.handle_explain_key(key),
            Screen::ChunkPreview => self.handle_chunk_preview_key(key),
            Screen::Indexing     => self.handle_indexing_key(key),
            Screen::RootPrompt   => self.handle_root_prompt_key(key).await,
            Screen::Flagged      => self.handle_flagged_key(key),
            Screen::Scores       => self.handle_scores_key(key),
        }
    }

    async fn handle_search_key(&mut self, key: KeyEvent) -> bool {
        if self.search_focused {
            // ── Search-bar focus: all chars type, Down moves cursor to results ──
            match key.code {
                KeyCode::Esc => {
                    if !self.query.is_empty() {
                        self.query.clear();
                        self.results.clear();
                        self.selected = 0;
                        self.list_offset = 0;
                        self.debounce_deadline = None;
                    }
                }
                KeyCode::Enter => {
                    let cmd = self.query.trim().to_ascii_lowercase();
                    if cmd == "/mode" || cmd.starts_with("/mode ") {
                        self.cycle_mode_with_arg(cmd.strip_prefix("/mode").unwrap_or("").trim());
                        self.query.clear();
                        self.results.clear();
                        self.selected = 0;
                        self.list_offset = 0;
                        self.debounce_deadline = None;
                    } else if cmd == "/score" || cmd == "/score --rescore" {
                        let rescore = cmd.contains("--rescore");
                        self.query.clear();
                        self.results.clear();
                        self.selected = 0;
                        self.list_offset = 0;
                        self.debounce_deadline = None;
                        self.start_scoring(rescore);
                    } else if cmd == "/scores" || cmd == "/scores --flagged" {
                        let flagged_only = cmd.contains("--flagged");
                        self.query.clear();
                        self.results.clear();
                        self.selected = 0;
                        self.list_offset = 0;
                        self.debounce_deadline = None;
                        match self.searcher.db.get_scored_chunks(200, flagged_only, None) {
                            Ok(rows) => {
                                if rows.is_empty() {
                                    self.set_status("No scored chunks — run /score first.");
                                } else {
                                    self.scores_data   = Some(rows);
                                    self.scores_scroll = 0;
                                    self.screen        = Screen::Scores;
                                }
                            }
                            Err(e) => self.set_status(format!("Scores error: {e}")),
                        }
                    }
                }
                KeyCode::Down if !self.results.is_empty() => {
                    self.search_focused = false;
                }
                KeyCode::Backspace => {
                    if self.query.pop().is_some() {
                        self.schedule_search();
                    }
                }
                KeyCode::Char(c) => {
                    self.query.push(c);
                    self.schedule_search();
                }
                _ => {}
            }
        } else {
            // ── Results focus: action keys active, Esc / Up-at-top return to bar ──
            match key.code {
                KeyCode::Char('q') => return true,

                KeyCode::Esc => self.search_focused = true,

                KeyCode::Char('s') => match self.searcher.stats().await {
                    Ok(s)  => { self.stats_data = Some(s); self.screen = Screen::Stats; }
                    Err(e) => self.set_status(format!("Stats error: {e}")),
                },

                KeyCode::Char('e') => {
                    let uri = self.results[self.selected].chunk.file_uri.clone();
                    match self.searcher.explain_full(std::path::Path::new(&uri)) {
                        Ok(Some(d)) => {
                            self.explain_data      = Some(d);
                            self.explain_scroll    = 0;
                            self.explain_chunk_sel = 0;
                            self.screen            = Screen::Explain;
                        }
                        Ok(None)    => self.set_status("File not in index."),
                        Err(e)      => self.set_status(format!("Explain error: {e}")),
                    }
                }

                KeyCode::Char('f') => {
                    match self.searcher.db.get_flagged_summary() {
                        Ok(rows) => {
                            self.flagged_data = Some(rows);
                            self.screen = Screen::Flagged;
                        }
                        Err(e) => self.set_status(format!("Flagged error: {e}")),
                    }
                }

                KeyCode::Char('i') => {
                    if self.index_root.is_some() {
                        self.start_indexing().await;
                    } else {
                        self.root_input.clear();
                        self.screen = Screen::RootPrompt;
                    }
                }

                KeyCode::Up => {
                    if self.selected > 0 {
                        self.selected -= 1;
                        if self.selected < self.list_offset {
                            self.list_offset = self.selected;
                        }
                    } else {
                        self.search_focused = true;
                    }
                }

                KeyCode::Down => {
                    if self.selected + 1 < self.results.len() {
                        self.selected += 1;
                    }
                }

                KeyCode::Enter => {
                    let uri = self.results[self.selected].chunk.file_uri.clone();
                    self.open_file(&uri);
                }

                _ => {}
            }
        }
        false
    }

    fn handle_stats_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Esc | KeyCode::Char('s') => self.screen = Screen::Search,
            _ => {}
        }
        false
    }

    fn handle_explain_key(&mut self, key: KeyEvent) -> bool {
        let chunk_count = self.explain_data.as_ref().map(|d| d.chunks.len()).unwrap_or(0);
        match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Esc => {
                self.screen = Screen::Search;
                self.explain_chunk_sel = 0;
            }
            KeyCode::Up => {
                if self.explain_chunk_sel > 0 {
                    self.explain_chunk_sel -= 1;
                    // keep selected chunk near top; header occupies ~4 lines
                    self.explain_scroll = (4 + self.explain_chunk_sel).saturating_sub(3) as u16;
                } else {
                    self.explain_scroll = self.explain_scroll.saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if self.explain_chunk_sel + 1 < chunk_count {
                    self.explain_chunk_sel += 1;
                    self.explain_scroll = (4 + self.explain_chunk_sel).saturating_sub(3) as u16;
                } else {
                    self.explain_scroll = self.explain_scroll.saturating_add(1);
                }
            }
            KeyCode::Char('e') => {
                if let Some(data) = &self.explain_data
                    && self.explain_chunk_sel < data.chunks.len() {
                        self.chunk_preview = Some(data.chunks[self.explain_chunk_sel].content.clone());
                        self.chunk_preview_scroll = 0;
                        self.screen = Screen::ChunkPreview;
                    }
            }
            _ => {}
        }
        false
    }

    fn handle_chunk_preview_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Esc       => self.screen = Screen::Explain,
            KeyCode::Up        => self.chunk_preview_scroll = self.chunk_preview_scroll.saturating_sub(1),
            KeyCode::Down      => self.chunk_preview_scroll = self.chunk_preview_scroll.saturating_add(1),
            _ => {}
        }
        false
    }

    fn handle_indexing_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Up => {
                self.index_log_scroll = self.index_log_scroll.saturating_sub(1);
            }
            KeyCode::Down => {
                let max = self.index_log.len().saturating_sub(1);
                if self.index_log_scroll < max {
                    self.index_log_scroll += 1;
                }
            }
            _ => {}
        }
        false
    }

    async fn handle_root_prompt_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Esc => self.screen = Screen::Search,
            KeyCode::Backspace => { self.root_input.pop(); }
            KeyCode::Enter => {
                let path = PathBuf::from(&self.root_input);
                if path.is_dir() {
                    self.index_root = Some(path);
                    self.screen = Screen::Search;
                    self.start_indexing().await;
                } else {
                    self.set_status(format!("Not a directory: {}", self.root_input));
                    self.screen = Screen::Search;
                }
            }
            KeyCode::Char(c) => { self.root_input.push(c); }
            _ => {}
        }
        false
    }

    // ── Search / index helpers ────────────────────────────────────────────────

    fn cycle_mode_with_arg(&mut self, arg: &str) {
        if !self.has_embedder {
            self.set_status("No embedder — sparse only (NOMIC_ONNX_PATH not set)");
            return;
        }
        self.search_mode = match arg {
            "hybrid" => SearchMode::Hybrid,
            "dense"  => SearchMode::Dense,
            "sparse" => SearchMode::Sparse,
            _ => match self.search_mode {
                SearchMode::Hybrid => SearchMode::Dense,
                SearchMode::Dense  => SearchMode::Sparse,
                SearchMode::Sparse => SearchMode::Hybrid,
            },
        };
        self.set_status(format!("Mode: {}", self.mode_label()));
    }

    fn schedule_search(&mut self) {
        self.debounce_deadline = Some(Instant::now() + Duration::from_millis(DEBOUNCE_MS));
    }

    async fn execute_search(&mut self) {
        if self.query.is_empty() {
            self.results.clear();
            self.selected = 0;
            self.list_offset = 0;
            return;
        }
        let mode = if !self.has_embedder { SearchMode::Sparse } else { self.search_mode.clone() };
        let opts = SearchOptions {
            mode,
            top_k: 20,
            candidate_pool: 100,
            max_per_file: 3,
            tier: None,
            ext_filter: None,
        };
        match self.searcher.search(&self.query, opts).await {
            Ok(results) => {
                self.results = results;
                self.selected = 0;
                self.list_offset = 0;
                if self.results.is_empty() {
                    self.search_focused = true;
                }
            }
            Err(e) => {
                self.set_status(format!("Search error: {e}"));
                self.results.clear();
            }
        }
    }

    async fn start_indexing(&mut self) {
        let root = match &self.index_root {
            Some(r) => r.clone(),
            None    => return,
        };
        self.index_log.clear();
        self.index_log_scroll = 0;
        self.screen = Screen::Indexing;

        let (tx, rx) = std::sync::mpsc::sync_channel::<String>(256);
        self.progress_rx = Some(rx);

        let root_str  = root.to_string_lossy().to_string();
        let index_dir = self.index_dir.clone();

        tokio::spawn(async move {
            let storage = Arc::new(LocalStorageClient::new());
            let indexer = match IncrementalIndexer::new(storage, &index_dir).await {
                Ok(i)  => i,
                Err(e) => { let _ = tx.try_send(format!("Error: {e}")); return; }
            };
            let tx2 = tx.clone();
            let on_progress = move |checked: usize, total: usize, s: &Stats| {
                let _ = tx2.try_send(format!(
                    "[{}/{}]  indexed={}  skipped={}  errors={}",
                    checked, total, s.indexed, s.skipped, s.errors,
                ));
            };
            match indexer.index_root(&root_str, false, None, Some(&on_progress)).await {
                Ok(s)  => { let _ = tx.try_send(format!("Done — {} indexed, {} skipped, {} errors", s.indexed, s.skipped, s.errors)); }
                Err(e) => { let _ = tx.try_send(format!("Error: {e}")); }
            }
        });
    }

    fn open_file(&self, uri: &str) {
        std::process::Command::new("cmd")
            .args(["/c", "start", "", uri])
            .spawn()
            .ok();
    }

    fn set_status(&mut self, msg: impl Into<String>) {
        self.status_msg = Some((msg.into(), Instant::now()));
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    fn render(&mut self, f: &mut Frame) {
        if let Some((_, ts)) = &self.status_msg
            && ts.elapsed().as_secs() >= 3 {
                self.status_msg = None;
            }

        let area  = f.area();
        let mode  = self.mode_label();
        let title = format!(" File Indexer \u{2500}\u{2500} {mode} ");

        let outer = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(NAVY))
            .title(Line::from(vec![
                Span::styled(title, Style::default().fg(GOLD).add_modifier(Modifier::BOLD)),
            ]));
        let inner = outer.inner(area);
        f.render_widget(outer, area);

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // query bar
                Constraint::Length(1), // separator
                Constraint::Min(1),    // content
                Constraint::Length(1), // separator
                Constraint::Length(1), // status bar
            ])
            .split(inner);

        // Query bar
        let query_text = match self.screen {
            Screen::RootPrompt => format!("Root: {}_", self.root_input),
            _ => if self.search_focused {
                format!("Search: {}_", self.query)
            } else {
                format!("Search: {}", self.query)
            },
        };
        f.render_widget(
            Paragraph::new(query_text).style(Style::default().fg(GOLD)),
            rows[0],
        );

        // Separators
        let sep = "\u{2500}".repeat(inner.width as usize);
        f.render_widget(Paragraph::new(sep.as_str()).style(Style::default().fg(NAVY)), rows[1]);
        f.render_widget(Paragraph::new(sep.as_str()).style(Style::default().fg(NAVY)), rows[3]);

        // Content
        if self.screen == Screen::Indexing {
            self.render_indexing(f, rows[2]);
        } else {
            self.render_results(f, rows[2]);
        }

        // Status bar
        self.render_status(f, rows[4]);

        // Overlays
        match self.screen {
            Screen::Stats        => self.render_stats_overlay(f, area),
            Screen::Explain      => self.render_explain_overlay(f, area),
            Screen::ChunkPreview => {
                self.render_explain_overlay(f, area);
                self.render_chunk_preview_overlay(f, area);
            }
            Screen::Flagged      => self.render_flagged_overlay(f, area),
            Screen::Scores       => self.render_scores_overlay(f, area),
            _ => {}
        }
    }

    fn render_results(&mut self, f: &mut Frame, area: Rect) {
        if self.results.is_empty() {
            let hint = if self.query.is_empty() { "Type to search\u{2026}" } else { "No results." };
            f.render_widget(
                Paragraph::new(hint).style(Style::default().fg(Color::DarkGray)),
                area,
            );
            return;
        }

        let visible = area.height as usize;

        // Keep selected item in view
        if self.selected >= self.list_offset + visible {
            self.list_offset = self.selected + 1 - visible;
        } else if self.selected < self.list_offset {
            self.list_offset = self.selected;
        }

        let items: Vec<ListItem> = self.results
            .iter()
            .enumerate()
            .skip(self.list_offset)
            .take(visible)
            .map(|(i, r)| {
                let path   = std::path::Path::new(&r.chunk.file_uri);
                let fname  = path.file_name().and_then(|n| n.to_str()).unwrap_or(&r.chunk.file_uri);
                let parent = path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("");
                let src_raw = if parent.is_empty() {
                    fname.to_string()
                } else {
                    format!("{parent}/{fname}")
                };
                let source: String = if src_raw.chars().count() > 38 {
                    let cut = src_raw.char_indices().nth(37).map(|(i, _)| i).unwrap_or(src_raw.len());
                    format!("{}\u{2026}", &src_raw[..cut])
                } else {
                    format!("{src_raw:<38}")
                };

                let pct = (r.normalized_score * 100.0).round() as u32;
                let score_color = if pct >= 80 {
                    Color::Green
                } else if pct >= 60 {
                    Color::Yellow
                } else {
                    Color::Red
                };

                let first = r.chunk.content
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("")
                    .trim_start();
                let cutoff = first.char_indices().nth(48).map(|(i, _)| i).unwrap_or(first.len());
                let preview = if first.len() > cutoff {
                    format!("{}\u{2026}", &first[..cutoff])
                } else {
                    first.to_string()
                };

                let is_sel = !self.search_focused && i == self.selected;
                let marker = if is_sel { "\u{25b6} " } else { "  " };

                let line = Line::from(vec![
                    Span::raw(format!("{marker}{:>2}  ", i + 1)),
                    Span::styled(format!("{pct:>3}%"), Style::default().fg(score_color)),
                    Span::raw(format!("  T{}  ", r.chunk.tier)),
                    Span::styled(source, Style::default().fg(GOLD)),
                    Span::raw(format!("  {preview}")),
                ]);

                if is_sel {
                    ListItem::new(line).style(Style::default().bg(Color::Rgb(20, 40, 80)))
                } else {
                    ListItem::new(line)
                }
            })
            .collect();

        f.render_widget(List::new(items), area);
    }

    fn render_indexing(&self, f: &mut Frame, area: Rect) {
        let visible = area.height as usize;
        let skip    = self.index_log_scroll.min(self.index_log.len().saturating_sub(1));
        let items: Vec<ListItem> = self.index_log
            .iter()
            .skip(skip)
            .take(visible)
            .map(|line| ListItem::new(line.as_str()))
            .collect();
        f.render_widget(List::new(items), area);
    }

    fn render_stats_overlay(&self, f: &mut Frame, area: Rect) {
        let Some(stats) = &self.stats_data else { return };
        let popup = centered_rect(55, 70, area);
        f.render_widget(Clear, popup);

        let block = Block::default()
            .title(Line::from(Span::styled(
                " Index Statistics ",
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            )))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(NAVY));
        let inner = block.inner(popup);
        f.render_widget(block, popup);

        let s       = &stats.db_stats;
        let pct     = if s.chunks > 0 { 100 * s.chunks_embedded / s.chunks } else { 100 };
        let pending = s.chunks - s.chunks_embedded;

        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled("Files", Style::default().fg(GOLD))),
            Line::from(format!("  {:>10}  total", s.files)),
        ];
        // prefix "  {count:>8}  " = 12 chars; truncate MIME to fit inner width
        let mime_max = inner.width.saturating_sub(12) as usize;
        if s.mime_counts.is_empty() {
            lines.push(Line::from("  (none)"));
        } else {
            for (mime, count) in &s.mime_counts {
                lines.push(Line::from(format!(
                    "  {count:>8}  {}",
                    truncate_mid(mime, mime_max)
                )));
            }
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Chunks", Style::default().fg(GOLD))));
        lines.push(Line::from(format!("  {:>10}  total",               s.chunks)));
        lines.push(Line::from(format!("  {:>10}  embedded ({pct}%)",   s.chunks_embedded)));
        if pending > 0 {
            lines.push(Line::from(format!("  {:>10}  pending",         pending)));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(format!("  {:>10}  edges",    s.edges)));
        lines.push(Line::from(format!("  {:>10}  FTS docs", s.chunks_fts_docs)));
        lines.push(Line::from(format!("  {:>10}  vectors",  stats.vec_total)));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Esc / s  to close",
            Style::default().fg(Color::DarkGray),
        )));

        f.render_widget(Paragraph::new(lines), inner);
    }

    fn render_flagged_overlay(&self, f: &mut Frame, area: Rect) {
        let popup = centered_rect(72, 75, area);
        f.render_widget(Clear, popup);

        let block = Block::default()
            .title(Line::from(Span::styled(
                " Flagged Chunks ",
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            )))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(NAVY));
        let inner = block.inner(popup);
        f.render_widget(block, popup);

        let rows = self.flagged_data.as_deref().unwrap_or(&[]);

        // Column widths relative to available inner width.
        // mime(~28) method(~12) flagged(7) total(7) pct(5)
        let w = inner.width as usize;
        let mime_w   = w.saturating_sub(12 + 7 + 7 + 5 + 6).max(12);
        let sep = "\u{2500}".repeat(mime_w + 12 + 7 + 7 + 5 + 5);

        let header = Line::from(vec![
            Span::styled(
                format!("{:<mime_w$}  {:<12}  {:>7}  {:>5}  {:>4}", "MIME type", "Method", "Flagged", "Total", "%"),
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            ),
        ]);

        let mut lines: Vec<Line> = vec![
            header,
            Line::from(Span::styled(sep, Style::default().fg(NAVY))),
        ];

        if rows.is_empty() {
            lines.push(Line::from(Span::styled(
                "  No flagged chunks.",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            for row in rows {
                let pct = if row.total > 0 {
                    row.flagged * 100 / row.total
                } else {
                    0
                };
                let pct_color = if pct >= 20 { Color::Red } else if pct >= 10 { Color::Yellow } else { Color::Green };
                lines.push(Line::from(vec![
                    Span::raw(format!(
                        "{:<mime_w$}  {:<12}  {:>7}  {:>5}",
                        truncate_mid(&row.mime_type, mime_w),
                        truncate_mid(&row.method, 12),
                        row.flagged,
                        row.total,
                    )),
                    Span::styled(
                        format!("  {:>3}%", pct),
                        Style::default().fg(pct_color),
                    ),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Esc: close  \u{00b7}  q: quit",
            Style::default().fg(Color::DarkGray),
        )));

        f.render_widget(Paragraph::new(lines), inner);
    }

    fn render_scores_overlay(&self, f: &mut Frame, area: Rect) {
        let rows = match &self.scores_data {
            Some(r) => r,
            None    => return,
        };

        let popup = centered_rect(92, 88, area);
        f.render_widget(Clear, popup);

        let block = Block::default()
            .title(Line::from(Span::styled(
                " Chunk Scores — worst first ",
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            )))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(NAVY));
        let inner = block.inner(popup);
        f.render_widget(block, popup);

        let w        = inner.width as usize;
        // fixed columns: rank(4) str(5) coh(5) flag(1) tier(2) spaces+sep ≈ 22; rest for source+preview
        let rest     = w.saturating_sub(26);
        let src_w    = (rest / 2).clamp(16, 36);
        let prev_w   = rest.saturating_sub(src_w).max(10);

        let sep = "\u{2500}".repeat(w.min(6 + 6 + 2 + 3 + src_w + 2 + prev_w));
        let header = Line::from(Span::styled(
            format!("{:>4}  {:>5}  {:>5}  F  T  {:<src_w$}  {}", "#", "str", "coh", "Source", "Preview"),
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
        ));

        let mut lines: Vec<Line> = vec![
            header,
            Line::from(Span::styled(sep, Style::default().fg(NAVY))),
        ];

        let visible = (inner.height as usize).saturating_sub(4); // header + sep + footer
        let skip    = self.scores_scroll.min(rows.len().saturating_sub(1));

        for (abs_i, c) in rows.iter().enumerate().skip(skip).take(visible) {
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
            let source = truncate_mid(&src_raw, src_w);

            let first = c.content.lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("")
                .trim_start();
            let cutoff  = first.char_indices().nth(prev_w).map(|(i, _)| i).unwrap_or(first.len());
            let preview = if first.len() > cutoff {
                format!("{}\u{2026}", &first[..cutoff])
            } else {
                first.to_string()
            };

            let str_val = c.structural_score.unwrap_or(0.0);
            let str_s   = format!("{str_val:.2}");
            let coh_s   = c.coherence_score
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "  \u{2014}  ".to_string());
            let flag    = if c.is_flagged { "\u{26a0}" } else { " " };

            let str_color = if str_val < 0.5 { Color::Red } else if str_val < 0.75 { Color::Yellow } else { Color::Green };
            let coh_color = match c.coherence_score {
                Some(v) if v < 0.6 => Color::Red,
                Some(_)            => Color::Green,
                None               => Color::DarkGray,
            };
            let flag_color = if c.is_flagged { Color::Red } else { Color::DarkGray };

            let line = Line::from(vec![
                Span::raw(format!("{:>4}  ", abs_i + 1)),
                Span::styled(format!("{str_s:>5}"), Style::default().fg(str_color)),
                Span::raw("  "),
                Span::styled(format!("{coh_s:>5}"), Style::default().fg(coh_color)),
                Span::raw("  "),
                Span::styled(flag.to_string(), Style::default().fg(flag_color)),
                Span::raw(format!("  {}  {:<src_w$}  {}", c.tier, source, preview)),
            ]);
            lines.push(line);
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Esc: close  \u{00b7}  \u{2191}\u{2193}: scroll  \u{00b7}  q: quit",
            Style::default().fg(Color::DarkGray),
        )));

        f.render_widget(Paragraph::new(lines), inner);
    }

    fn render_explain_overlay(&self, f: &mut Frame, area: Rect) {
        let Some(data) = &self.explain_data else { return };
        let popup = centered_rect(80, 80, area);
        f.render_widget(Clear, popup);

        let file_uri = data.chunks.first().map(|c| c.file_uri.as_str()).unwrap_or("");
        let fname = std::path::Path::new(file_uri)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(file_uri);

        let block = Block::default()
            .title(Line::from(Span::styled(
                format!(" {fname} "),
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            )))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(NAVY));
        let inner = block.inner(popup);
        f.render_widget(block, popup);

        let mut lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled("Chunks: ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!(
                    "{}T1 / {}T2 / {}T3",
                    data.detail.t1_count, data.detail.t2_count, data.detail.t3_count,
                )),
            ]),
            Line::from(vec![
                Span::styled("Size: ", Style::default().fg(Color::DarkGray)),
                Span::raw(format_bytes(data.detail.size_bytes)),
                Span::raw("  "),
                Span::styled("MIME: ", Style::default().fg(Color::DarkGray)),
                Span::raw(data.detail.mime_type.clone()),
            ]),
            Line::from(""),
            Line::from(Span::styled("Chunk structure", Style::default().fg(GOLD))),
        ];

        for (ci, chunk) in data.chunks.iter().enumerate() {
            let tok = chunk.token_count
                .map(|t| format!("{t:>4} tok"))
                .unwrap_or_else(|| "       \u{2014}".to_string());
            let emb   = if chunk.lance_id.is_some() { "E" } else { "\u{00b7}" };
            let first = chunk.content
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("")
                .trim_start();
            let cutoff = first.char_indices().nth(32).map(|(i, _)| i).unwrap_or(first.len());
            let sel    = ci == self.explain_chunk_sel;
            let marker = if sel { "\u{25b6}" } else { " " };
            let prefix = format!(
                " {marker} T{}  #{:<3}  {tok}  {emb}  ",
                chunk.tier, chunk.chunk_index,
            );

            // Score spans — coloured when not selected, plain gold when selected.
            let str_text = chunk.structural_score
                .map(|s| format!("str={:.2}", s))
                .unwrap_or_else(|| "str=\u{2014}   ".to_string());
            let coh_text = chunk.coherence_score
                .map(|s| format!(" coh={:.2}", s))
                .unwrap_or_else(|| " coh=\u{2014}  ".to_string());
            let flag_text = if chunk.is_flagged { " \u{26a0}" } else { "   " };

            let preview = format!("  {}", &first[..cutoff]);

            let line = if sel {
                Line::from(vec![
                    Span::raw(prefix),
                    Span::raw(str_text),
                    Span::raw(coh_text),
                    Span::raw(flag_text),
                    Span::raw(preview),
                ]).style(Style::default().bg(Color::Rgb(20, 40, 80)).fg(GOLD))
            } else {
                let str_color = match chunk.structural_score {
                    Some(s) if s < 0.5 => Color::Red,
                    Some(_)            => Color::Green,
                    None               => Color::DarkGray,
                };
                let coh_color = match chunk.coherence_score {
                    Some(c) if c < 0.6 => Color::Red,
                    Some(_)            => Color::Green,
                    None               => Color::DarkGray,
                };
                let flag_color = if chunk.is_flagged { Color::Red } else { Color::DarkGray };
                Line::from(vec![
                    Span::raw(prefix),
                    Span::styled(str_text,  Style::default().fg(str_color)),
                    Span::styled(coh_text,  Style::default().fg(coh_color)),
                    Span::styled(flag_text, Style::default().fg(flag_color)),
                    Span::raw(preview),
                ])
            };
            lines.push(line);
        }

        if !data.outgoing.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("References", Style::default().fg(GOLD))));
            for e in &data.outgoing {
                let name = std::path::Path::new(&e.dst_uri)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&e.dst_uri);
                lines.push(Line::from(format!("  \u{2192}  {name}  ({})", e.edge_type)));
            }
        }

        if !data.incoming.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("Referenced By", Style::default().fg(GOLD))));
            for e in &data.incoming {
                let name = std::path::Path::new(&e.src_uri)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&e.src_uri);
                lines.push(Line::from(format!("  \u{2190}  {name}  ({})", e.edge_type)));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Esc: close  \u{00b7}  \u{2191}\u{2193}: select chunk  \u{00b7}  e: preview chunk  \u{00b7}  q: quit",
            Style::default().fg(Color::DarkGray),
        )));

        f.render_widget(
            Paragraph::new(lines)
                .scroll((self.explain_scroll, 0))
                .wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn render_chunk_preview_overlay(&self, f: &mut Frame, area: Rect) {
        let Some(content) = &self.chunk_preview else { return };
        let Some(data)    = &self.explain_data  else { return };

        let popup = centered_rect(88, 88, area);
        f.render_widget(Clear, popup);

        let chunk = &data.chunks[self.explain_chunk_sel];
        let tok_label = chunk.token_count
            .map(|t| format!("  {t} tok"))
            .unwrap_or_default();
        let title = format!(" T{}  #{}{tok_label} ", chunk.tier, chunk.chunk_index);

        let block = Block::default()
            .title(Line::from(Span::styled(
                title,
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            )))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(NAVY));
        let inner = block.inner(popup);
        f.render_widget(block, popup);

        let lines: Vec<Line> = content.lines()
            .map(|l| Line::from(l.to_string()))
            .collect();

        f.render_widget(
            Paragraph::new(lines)
                .scroll((self.chunk_preview_scroll, 0))
                .wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn render_status(&self, f: &mut Frame, area: Rect) {
        let widget = if let Some((msg, _)) = &self.status_msg {
            Paragraph::new(msg.as_str()).style(Style::default().fg(Color::Yellow))
        } else {
            let warn = if !self.has_embedder { "  \u{26a0} sparse only" } else { "" };
            let text = if self.search_focused {
                let nav_hint = if !self.results.is_empty() {
                    "  \u{00b7}  \u{2193} select results"
                } else {
                    ""
                };
                format!("Type to search  \u{00b7}  /mode  /score  /scores  \u{00b7}  Ctrl-C: quit{nav_hint}{warn}")
            } else {
                let count = self.results.len();
                format!(
                    "{count} result{}  \u{00b7}  \u{2191}\u{2193} navigate  \u{00b7}  \u{23ce} open  \u{00b7}  s: stats  \u{00b7}  f: flagged  \u{00b7}  e: explain  \u{00b7}  i: index  \u{00b7}  Esc: back  \u{00b7}  q: quit{warn}",
                    if count == 1 { "" } else { "s" },
                )
            };
            Paragraph::new(text).style(Style::default().fg(Color::DarkGray))
        };
        f.render_widget(widget, area);
    }

    fn mode_label(&self) -> &'static str {
        match self.search_mode {
            SearchMode::Hybrid => "hybrid",
            SearchMode::Dense  => "dense",
            SearchMode::Sparse => "sparse",
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn centered_rect(pct_x: u16, pct_y: u16, r: Rect) -> Rect {
    let vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - pct_y) / 2),
            Constraint::Percentage(pct_y),
            Constraint::Percentage((100 - pct_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - pct_x) / 2),
            Constraint::Percentage(pct_x),
            Constraint::Percentage((100 - pct_x) / 2),
        ])
        .split(vert[1])[1]
}

// Shorten a string to max_chars, replacing the middle with … if needed.
fn truncate_mid(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars || max_chars < 5 {
        return s.to_string();
    }
    let half = (max_chars - 1) / 2;
    let left: String  = chars[..half].iter().collect();
    let right: String = chars[chars.len() - (max_chars - 1 - half)..].iter().collect();
    format!("{left}\u{2026}{right}")
}

fn format_bytes(bytes: i64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    }
}
