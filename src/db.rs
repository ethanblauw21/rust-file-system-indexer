use crate::error::IndexerError;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, Connection};
use serde_json;
use std::collections::HashMap;
use std::path::Path;

// ── Public row / result types ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub file_id:        i64,
    pub modified_at:    Option<f64>,
    pub content_hash:   String,
    /// True if any chunks for this file have lance_id IS NULL (vectors not yet written to LanceDB).
    pub has_unembedded: bool,
}

#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub id: i64,
    #[allow(dead_code)]
    pub file_id: i64,
    pub tier: u8,
    pub chunk_index: i64,
    pub content: String,
    pub token_count: Option<i64>,
    #[allow(dead_code)]
    pub meta: serde_json::Value,
    pub lance_id: Option<i64>,
    pub file_uri: String,
    pub mime_type: String,
    pub is_flagged: bool,
    pub chunker_method: Option<String>,
    pub structural_score: Option<f32>,
    pub coherence_score: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct FlaggedSummaryRow {
    pub mime_type: String,
    pub method: String,
    pub flagged: i64,
    pub total: i64,
}

#[derive(Debug, Clone)]
pub struct FtsResult {
    pub id: i64,
    pub file_id: i64,
    pub tier: u8,
    pub content: String,
    pub meta: serde_json::Value,
    pub bm25_score: f64,
    pub file_uri: String,
}

#[derive(Debug, Clone)]
pub struct DbStats {
    pub files:           i64,
    pub chunks:          i64,
    pub chunks_embedded: i64,
    pub edges:           i64,
    pub chunks_fts_docs: i64,
    pub mime_counts:     Vec<(String, i64)>,
}

#[derive(Debug, Clone)]
pub struct FileDetail {
    pub mime_type:   String,
    pub size_bytes:  i64,
    pub indexed_at:  f64,
    pub modified_at: Option<f64>,
    pub t1_count:    i64,
    pub t2_count:    i64,
    pub t3_count:    i64,
}

#[derive(Debug, Clone)]
pub struct EdgeRow {
    pub dst_uri:   String,
    pub edge_type: String,
}

#[derive(Debug, Clone)]
pub struct IncomingEdge {
    pub src_uri:   String,
    pub edge_type: String,
}

/// Input type for `insert_chunks`.
#[derive(Debug)]
pub struct ChunkInput {
    pub tier: u8,
    pub chunk_index: usize,
    pub content: String,
    pub token_count: Option<usize>,
    pub meta: serde_json::Value,
    pub chunker_method: Option<String>,
}

/// Input type for `insert_edges`.
#[derive(Debug)]
pub struct EdgeInput {
    pub src_chunk: i64,
    pub dst_uri: String,
    pub edge_type: EdgeType,
    pub meta: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeType {
    LinksTo,
    References,
    Embeds,
}

impl EdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::LinksTo    => "LINKS_TO",
            EdgeType::References => "REFERENCES",
            EdgeType::Embeds     => "EMBEDS",
        }
    }
}

// ── Schema DDL ────────────────────────────────────────────────────────────────

const DDL_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file_uri     TEXT    UNIQUE NOT NULL,
    mime_type    TEXT    NOT NULL DEFAULT 'application/octet-stream',
    content_hash TEXT    NOT NULL,
    size_bytes   INTEGER NOT NULL DEFAULT 0,
    indexed_at   REAL    NOT NULL,
    modified_at  REAL    DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_uri ON files(file_uri);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    tier        INTEGER NOT NULL CHECK (tier IN (1, 2, 3)),
    chunk_index INTEGER NOT NULL,
    content     TEXT    NOT NULL,
    token_count INTEGER,
    meta        TEXT    NOT NULL DEFAULT '{}',
    lance_id    INTEGER DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file_tier ON chunks(file_id, tier);
CREATE INDEX IF NOT EXISTS idx_chunks_tier      ON chunks(tier);
CREATE INDEX IF NOT EXISTS idx_chunks_lance_id  ON chunks(lance_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(
        content,
        content       = 'chunks',
        content_rowid = 'id',
        tokenize      = 'porter unicode61'
    );

DROP TRIGGER IF EXISTS chunks_ai;
CREATE TRIGGER chunks_ai
AFTER INSERT ON chunks
WHEN new.tier IN (1, 2) BEGIN
    INSERT INTO chunks_fts(rowid, content)
    VALUES (new.id, new.content);
END;

DROP TRIGGER IF EXISTS chunks_au;
CREATE TRIGGER chunks_au
AFTER UPDATE ON chunks
WHEN old.tier IN (1, 2) OR new.tier IN (1, 2) BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content)
    SELECT new.id, new.content WHERE new.tier IN (1, 2);
END;

DROP TRIGGER IF EXISTS chunks_ad;
CREATE TRIGGER chunks_ad
AFTER DELETE ON chunks
WHEN old.tier IN (1, 2) BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TABLE IF NOT EXISTS edges (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    src_chunk  INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    dst_uri    TEXT    NOT NULL,
    edge_type  TEXT    NOT NULL CHECK (edge_type IN ('LINKS_TO', 'REFERENCES', 'EMBEDS')),
    meta       TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_chunk);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_uri);
"#;

// ── Transaction helper ────────────────────────────────────────────────────────

fn with_tx<F, T>(conn: &Connection, f: F) -> Result<T, IndexerError>
where
    F: FnOnce(&Connection) -> Result<T, IndexerError>,
{
    conn.execute_batch("BEGIN")?;
    match f(conn) {
        Ok(val) => {
            conn.execute_batch("COMMIT")?;
            Ok(val)
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

// ── EnterpriseDb ──────────────────────────────────────────────────────────────

pub struct EnterpriseDb {
    pool: Pool<SqliteConnectionManager>,
}

impl EnterpriseDb {
    pub fn new(path: &Path) -> Result<Self, IndexerError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| IndexerError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        let manager = SqliteConnectionManager::file(path).with_init(|conn| {
            conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous   = NORMAL;
                 PRAGMA foreign_keys  = ON;
                 PRAGMA cache_size    = -65536;
                 PRAGMA temp_store    = MEMORY;
                 PRAGMA mmap_size     = 2147483648;
                 PRAGMA threads       = 4;",
            )
        });

        let pool = Pool::new(manager).map_err(|e| IndexerError::Other(e.into()))?;

        {
            let conn = pool.get().map_err(|e| IndexerError::Other(e.into()))?;
            conn.execute_batch(DDL_SQL)?;
            // Idempotent column migrations — ignore "duplicate column" errors
            for ddl in &[
                "ALTER TABLE files  ADD COLUMN modified_at       REAL    DEFAULT NULL",
                "ALTER TABLE chunks ADD COLUMN lance_id          INTEGER DEFAULT NULL",
                "ALTER TABLE chunks ADD COLUMN structural_score  REAL",
                "ALTER TABLE chunks ADD COLUMN coherence_score   REAL",
                "ALTER TABLE chunks ADD COLUMN is_flagged        INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE chunks ADD COLUMN chunker_method    TEXT",
            ] {
                let _ = conn.execute_batch(ddl);
            }
            // Remove any existing T3 entries from the FTS index (T3 are now excluded from FTS).
            // Uses the FTS5 external-content delete protocol; no-op if they're already absent.
            let _ = conn.execute_batch(
                "INSERT INTO chunks_fts(chunks_fts, rowid, content) \
                 SELECT 'delete', c.id, c.content FROM chunks c WHERE c.tier = 3",
            );
            let _ = conn.execute_batch("PRAGMA optimize");
        }

        Ok(Self { pool })
    }

    fn conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>, IndexerError> {
        self.pool.get().map_err(|e| IndexerError::Other(e.into()))
    }

    // ── File operations ───────────────────────────────────────────────────────

    pub fn get_all_file_info(&self) -> Result<HashMap<String, FileInfo>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT f.id, f.file_uri, f.modified_at, f.content_hash, \
             EXISTS(SELECT 1 FROM chunks c WHERE c.file_id = f.id AND c.lance_id IS NULL) \
             FROM files f",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<f64>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, bool>(4)?,
            ))
        })?;
        let mut map = HashMap::new();
        for row in rows {
            let (file_id, uri, modified_at, content_hash, has_unembedded) = row?;
            map.insert(uri, FileInfo { file_id, modified_at, content_hash, has_unembedded });
        }
        Ok(map)
    }

    pub fn get_lance_ids_for_file(&self, file_id: i64) -> Result<Vec<i64>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT lance_id FROM chunks WHERE file_id = ?1 AND lance_id IS NOT NULL",
        )?;
        let ids = stmt
            .query_map(params![file_id], |r| r.get(0))?
            .collect::<Result<Vec<i64>, _>>()?;
        Ok(ids)
    }

    pub fn upsert_file(
        &self,
        file_uri: &str,
        mime_type: &str,
        content_hash: &str,
        size_bytes: i64,
        indexed_at: f64,
        modified_at: Option<f64>,
    ) -> Result<i64, IndexerError> {
        let conn = self.conn()?;
        with_tx(&conn, |c| {
            c.execute(
                "INSERT INTO files (file_uri, mime_type, content_hash, size_bytes, indexed_at, modified_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(file_uri) DO UPDATE SET
                     mime_type    = excluded.mime_type,
                     content_hash = excluded.content_hash,
                     size_bytes   = excluded.size_bytes,
                     indexed_at   = excluded.indexed_at,
                     modified_at  = excluded.modified_at",
                params![file_uri, mime_type, content_hash, size_bytes, indexed_at, modified_at],
            )?;
            let id: i64 = c.query_row(
                "SELECT id FROM files WHERE file_uri = ?1",
                params![file_uri],
                |row| row.get(0),
            )?;
            Ok(id)
        })
    }

    pub fn delete_file(&self, file_uri: &str) -> Result<(), IndexerError> {
        let conn = self.conn()?;
        with_tx(&conn, |c| {
            // Delete chunks explicitly so the FTS5 sync triggers fire (SQLite
            // does not guarantee triggers fire for rows removed by FK CASCADE).
            c.execute(
                "DELETE FROM chunks WHERE file_id = (SELECT id FROM files WHERE file_uri = ?1)",
                params![file_uri],
            )?;
            c.execute("DELETE FROM files WHERE file_uri = ?1", params![file_uri])?;
            Ok(())
        })
    }

    pub fn list_all_file_uris(&self) -> Result<Vec<String>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare("SELECT file_uri FROM files")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        let mut uris = Vec::new();
        for row in rows {
            uris.push(row?);
        }
        Ok(uris)
    }

    /// Return the Tier-3 (whole-file summary) chunk content for `file_uri`,
    /// or `None` if the file is not indexed or has no T3 chunk.
    pub fn get_tier3_content(&self, file_uri: &str) -> Result<Option<String>, IndexerError> {
        let conn = self.conn()?;
        match conn.query_row(
            "SELECT c.content FROM chunks c
             JOIN files f ON c.file_id = f.id
             WHERE f.file_uri = ?1 AND c.tier = 3
             ORDER BY c.chunk_index LIMIT 1",
            params![file_uri],
            |row| row.get::<_, String>(0),
        ) {
            Ok(content)                               => Ok(Some(content)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e)                                    => Err(e.into()),
        }
    }

    /// File metadata + per-tier chunk counts and total tokens in a single query.
    pub fn get_file_detail(&self, file_uri: &str) -> Result<Option<FileDetail>, IndexerError> {
        let conn = self.conn()?;
        match conn.query_row(
            "SELECT f.mime_type, f.size_bytes, f.indexed_at, f.modified_at,
                    COALESCE(SUM(CASE WHEN c.tier = 1 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN c.tier = 2 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN c.tier = 3 THEN 1 ELSE 0 END), 0)
             FROM files f
             LEFT JOIN chunks c ON c.file_id = f.id
             WHERE f.file_uri = ?1
             GROUP BY f.id",
            params![file_uri],
            |row| Ok(FileDetail {
                mime_type:   row.get(0)?,
                size_bytes:  row.get(1)?,
                indexed_at:  row.get(2)?,
                modified_at: row.get(3)?,
                t1_count:    row.get(4)?,
                t2_count:    row.get(5)?,
                t3_count:    row.get(6)?,
            }),
        ) {
            Ok(d)                                     => Ok(Some(d)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e)                                    => Err(e.into()),
        }
    }

    /// All chunks for a file ordered by tier then chunk_index.
    pub fn get_chunks_for_file(&self, file_uri: &str) -> Result<Vec<ChunkRow>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT c.id, c.file_id, c.tier, c.chunk_index,
                    c.content, c.token_count, c.meta, c.lance_id,
                    f.mime_type, c.is_flagged, c.chunker_method,
                    c.structural_score, c.coherence_score
             FROM chunks c
             JOIN files f ON c.file_id = f.id
             WHERE f.file_uri = ?1
             ORDER BY c.tier, c.chunk_index",
        )?;
        let rows = stmt.query_map(params![file_uri], |row| {
            let meta_str: String = row.get(6)?;
            Ok(ChunkRow {
                id:               row.get(0)?,
                file_id:          row.get(1)?,
                tier:             row.get::<_, i64>(2)? as u8,
                chunk_index:      row.get(3)?,
                content:          row.get(4)?,
                token_count:      row.get(5)?,
                meta: serde_json::from_str(&meta_str)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                lance_id:         row.get(7)?,
                file_uri:         file_uri.to_string(),
                mime_type:        row.get(8)?,
                is_flagged:       row.get::<_, i64>(9)? != 0,
                chunker_method:   row.get(10)?,
                structural_score: row.get(11)?,
                coherence_score:  row.get(12)?,
            })
        })?;
        let mut chunks = Vec::new();
        for row in rows { chunks.push(row?); }
        Ok(chunks)
    }

    /// Edges where the source chunk belongs to `file_uri` (deduplicated by dst + type).
    pub fn get_outgoing_edges(&self, file_uri: &str) -> Result<Vec<EdgeRow>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT DISTINCT e.dst_uri, e.edge_type
             FROM edges e
             JOIN chunks c ON e.src_chunk = c.id
             JOIN files f ON c.file_id = f.id
             WHERE f.file_uri = ?1
             ORDER BY e.edge_type, e.dst_uri",
        )?;
        let rows = stmt.query_map(params![file_uri], |row| {
            Ok(EdgeRow { dst_uri: row.get(0)?, edge_type: row.get(1)? })
        })?;
        let mut edges = Vec::new();
        for row in rows { edges.push(row?); }
        Ok(edges)
    }

    /// Edges where `dst_uri` equals `file_uri`, resolved back to source file URIs.
    pub fn get_incoming_edges(&self, file_uri: &str) -> Result<Vec<IncomingEdge>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT DISTINCT f.file_uri, e.edge_type
             FROM edges e
             JOIN chunks c ON e.src_chunk = c.id
             JOIN files f ON c.file_id = f.id
             WHERE e.dst_uri = ?1
             ORDER BY e.edge_type, f.file_uri",
        )?;
        let rows = stmt.query_map(params![file_uri], |row| {
            Ok(IncomingEdge { src_uri: row.get(0)?, edge_type: row.get(1)? })
        })?;
        let mut edges = Vec::new();
        for row in rows { edges.push(row?); }
        Ok(edges)
    }

    // ── Chunk operations ──────────────────────────────────────────────────────

    pub fn insert_chunks(
        &self,
        file_id: i64,
        chunks: &[ChunkInput],
    ) -> Result<Vec<i64>, IndexerError> {
        let conn = self.conn()?;
        with_tx(&conn, |c| {
            let mut ids = Vec::with_capacity(chunks.len());
            let mut stmt = c.prepare(
                "INSERT INTO chunks (file_id, tier, chunk_index, content, token_count, meta, lance_id, chunker_method)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, ?7)",
            )?;
            for chunk in chunks {
                stmt.execute(params![
                    file_id,
                    chunk.tier as i64,
                    chunk.chunk_index as i64,
                    &chunk.content,
                    chunk.token_count.map(|n| n as i64),
                    serde_json::to_string(&chunk.meta)
                        .unwrap_or_else(|_| "{}".to_string()),
                    chunk.chunker_method.as_deref(),
                ])?;
                ids.push(c.last_insert_rowid());
            }
            Ok(ids)
        })
    }

    pub fn delete_chunks_for_file(&self, file_id: i64) -> Result<(), IndexerError> {
        let conn = self.conn()?;
        with_tx(&conn, |c| {
            c.execute("DELETE FROM chunks WHERE file_id = ?1", params![file_id])?;
            Ok(())
        })
    }

    pub fn delete_all_chunks(&self) -> Result<(), IndexerError> {
        let conn = self.conn()?;
        with_tx(&conn, |c| {
            c.execute("DELETE FROM chunks", [])?;
            Ok(())
        })
    }

    pub fn set_chunk_lance_ids(&self, map: &HashMap<i64, i64>) -> Result<(), IndexerError> {
        if map.is_empty() {
            return Ok(());
        }
        let conn = self.conn()?;
        with_tx(&conn, |c| {
            for (chunk_id, lance_id) in map {
                c.execute(
                    "UPDATE chunks SET lance_id = ?1 WHERE id = ?2",
                    params![lance_id, chunk_id],
                )?;
            }
            Ok(())
        })
    }

    /// Fetch chunk rows keyed by their `lance_id`.  IDs are i64 values from
    /// our own system — inlining them into the SQL is safe (no user input).
    pub fn get_chunks_by_lance_ids(
        &self,
        lance_ids: &[i64],
    ) -> Result<HashMap<i64, ChunkRow>, IndexerError> {
        if lance_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let id_list: String = lance_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT c.id, c.file_id, c.tier, c.chunk_index,
                    c.content, c.token_count, c.meta, c.lance_id,
                    f.file_uri, f.mime_type, c.is_flagged, c.chunker_method,
                    c.structural_score, c.coherence_score
             FROM   chunks c
             JOIN   files  f ON c.file_id = f.id
             WHERE  c.lance_id IN ({})",
            id_list
        );
        let conn = self.conn()?;
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            let meta_str: String = row.get(6)?;
            Ok(ChunkRow {
                id:               row.get(0)?,
                file_id:          row.get(1)?,
                tier:             row.get::<_, i64>(2)? as u8,
                chunk_index:      row.get(3)?,
                content:          row.get(4)?,
                token_count:      row.get(5)?,
                meta: serde_json::from_str(&meta_str)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                lance_id:         row.get(7)?,
                file_uri:         row.get(8)?,
                mime_type:        row.get(9)?,
                is_flagged:       row.get::<_, i64>(10)? != 0,
                chunker_method:   row.get(11)?,
                structural_score: row.get(12)?,
                coherence_score:  row.get(13)?,
            })
        })?;
        let mut map = HashMap::new();
        for row in rows {
            let chunk = row?;
            if let Some(lid) = chunk.lance_id {
                map.insert(lid, chunk);
            }
        }
        Ok(map)
    }

    // ── Scoring helpers ───────────────────────────────────────────────────────

    pub fn set_chunk_scores(
        &self,
        chunk_id: i64,
        structural: f32,
        coherence: Option<f32>,
        is_flagged: bool,
    ) -> Result<(), IndexerError> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE chunks
             SET structural_score = ?1,
                 coherence_score  = ?2,
                 is_flagged       = ?3
             WHERE id = ?4",
            params![
                structural as f64,
                coherence.map(|v| v as f64),
                is_flagged as i64,
                chunk_id,
            ],
        )?;
        Ok(())
    }

    pub fn get_flagged_summary(&self) -> Result<Vec<FlaggedSummaryRow>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT f.mime_type,
                    COALESCE(c.chunker_method, '_default') AS method,
                    SUM(c.is_flagged)                      AS flagged,
                    COUNT(*)                               AS total
             FROM chunks c
             JOIN files f ON c.file_id = f.id
             GROUP BY f.mime_type, method
             ORDER BY flagged DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(FlaggedSummaryRow {
                mime_type: row.get(0)?,
                method:    row.get(1)?,
                flagged:   row.get(2)?,
                total:     row.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows { out.push(row?); }
        Ok(out)
    }

    pub fn get_unscored_chunks(&self) -> Result<Vec<ChunkRow>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT c.id, c.file_id, c.tier, c.chunk_index,
                    c.content, c.token_count, c.meta, c.lance_id,
                    f.file_uri, f.mime_type, c.is_flagged, c.chunker_method,
                    c.structural_score, c.coherence_score
             FROM chunks c
             JOIN files f ON c.file_id = f.id
             WHERE c.structural_score IS NULL
             ORDER BY c.id",
        )?;
        let rows = stmt.query_map([], |row| {
            let meta_str: String = row.get(6)?;
            Ok(ChunkRow {
                id:               row.get(0)?,
                file_id:          row.get(1)?,
                tier:             row.get::<_, i64>(2)? as u8,
                chunk_index:      row.get(3)?,
                content:          row.get(4)?,
                token_count:      row.get(5)?,
                meta: serde_json::from_str(&meta_str)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                lance_id:         row.get(7)?,
                file_uri:         row.get(8)?,
                mime_type:        row.get(9)?,
                is_flagged:       row.get::<_, i64>(10)? != 0,
                chunker_method:   row.get(11)?,
                structural_score: row.get(12)?,
                coherence_score:  row.get(13)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows { out.push(row?); }
        Ok(out)
    }

    pub fn get_chunks_for_rescoring(&self) -> Result<Vec<ChunkRow>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT c.id, c.file_id, c.tier, c.chunk_index,
                    c.content, c.token_count, c.meta, c.lance_id,
                    f.file_uri, f.mime_type, c.is_flagged, c.chunker_method,
                    c.structural_score, c.coherence_score
             FROM chunks c
             JOIN files f ON c.file_id = f.id
             ORDER BY c.id",
        )?;
        let rows = stmt.query_map([], |row| {
            let meta_str: String = row.get(6)?;
            Ok(ChunkRow {
                id:               row.get(0)?,
                file_id:          row.get(1)?,
                tier:             row.get::<_, i64>(2)? as u8,
                chunk_index:      row.get(3)?,
                content:          row.get(4)?,
                token_count:      row.get(5)?,
                meta: serde_json::from_str(&meta_str)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                lance_id:         row.get(7)?,
                file_uri:         row.get(8)?,
                mime_type:        row.get(9)?,
                is_flagged:       row.get::<_, i64>(10)? != 0,
                chunker_method:   row.get(11)?,
                structural_score: row.get(12)?,
                coherence_score:  row.get(13)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows { out.push(row?); }
        Ok(out)
    }

    /// Returns scored chunks ordered by structural_score ASC (worst first).
    /// `flagged_only` restricts to is_flagged = 1; `tier` filters by tier.
    pub fn get_scored_chunks(
        &self,
        limit:        usize,
        flagged_only: bool,
        tier:         Option<u8>,
    ) -> Result<Vec<ChunkRow>, IndexerError> {
        let conn = self.conn()?;

        let mut filters = vec!["c.structural_score IS NOT NULL"];
        let flagged_clause  = "c.is_flagged = 1";
        let tier_val;
        let tier_clause;
        if flagged_only { filters.push(flagged_clause); }
        if let Some(t) = tier {
            tier_val    = t;
            tier_clause = format!("c.tier = {tier_val}");
            filters.push(&tier_clause);
        }
        let sql = format!(
            "SELECT c.id, c.file_id, c.tier, c.chunk_index,
                    c.content, c.token_count, c.meta, c.lance_id,
                    f.file_uri, f.mime_type, c.is_flagged, c.chunker_method,
                    c.structural_score, c.coherence_score
             FROM chunks c
             JOIN files f ON c.file_id = f.id
             WHERE {}
             ORDER BY c.structural_score ASC
             LIMIT {}",
            filters.join(" AND "),
            limit,
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            let meta_str: String = row.get(6)?;
            Ok(ChunkRow {
                id:               row.get(0)?,
                file_id:          row.get(1)?,
                tier:             row.get::<_, i64>(2)? as u8,
                chunk_index:      row.get(3)?,
                content:          row.get(4)?,
                token_count:      row.get(5)?,
                meta: serde_json::from_str(&meta_str)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                lance_id:         row.get(7)?,
                file_uri:         row.get(8)?,
                mime_type:        row.get(9)?,
                is_flagged:       row.get::<_, i64>(10)? != 0,
                chunker_method:   row.get(11)?,
                structural_score: row.get(12)?,
                coherence_score:  row.get(13)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows { out.push(row?); }
        Ok(out)
    }

    /// Returns distinct `(file_uri, mime_type, stored_method)` for every flagged file.
    /// One row per unique `(file_uri, mime_type, chunker_method)` combination.
    pub fn get_flagged_files_with_methods(
        &self,
    ) -> Result<Vec<(String, String, String)>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT DISTINCT f.file_uri, f.mime_type, COALESCE(c.chunker_method, '_default')
             FROM chunks c
             JOIN files f ON c.file_id = f.id
             WHERE c.is_flagged = 1",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })?;
        let mut out = Vec::new();
        for row in rows { out.push(row?); }
        Ok(out)
    }

    /// Returns the `lance_id` values of all embedded chunks with the given file + tier.
    pub fn get_lance_ids_for_tier(&self, file_id: i64, tier: u8) -> Result<Vec<i64>, IndexerError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT lance_id FROM chunks
             WHERE file_id = ?1 AND tier = ?2 AND lance_id IS NOT NULL",
        )?;
        let rows = stmt.query_map(params![file_id, tier as i64], |row| row.get::<_, i64>(0))?;
        let mut out = Vec::new();
        for row in rows { out.push(row?); }
        Ok(out)
    }

    // ── Edge operations ───────────────────────────────────────────────────────

    pub fn insert_edges(&self, edges: &[EdgeInput]) -> Result<(), IndexerError> {
        if edges.is_empty() {
            return Ok(());
        }
        let conn = self.conn()?;
        with_tx(&conn, |c| {
            for edge in edges {
                c.execute(
                    "INSERT INTO edges (src_chunk, dst_uri, edge_type, meta)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![
                        edge.src_chunk,
                        &edge.dst_uri,
                        edge.edge_type.as_str(),
                        serde_json::to_string(&edge.meta)
                            .unwrap_or_else(|_| "{}".to_string()),
                    ],
                )?;
            }
            Ok(())
        })
    }

    // ── Hybrid search ─────────────────────────────────────────────────────────

    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<FtsResult>, IndexerError> {
        let safe_query = Self::sanitize_fts_query(query);
        if safe_query == "\"\"" {
            return Ok(vec![]);
        }
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT c.id, c.file_id, c.tier, c.content, c.meta,
                    COALESCE(bm25(chunks_fts), 0.0) AS bm25_score,
                    f.file_uri
             FROM   chunks_fts
             JOIN   chunks c ON chunks_fts.rowid = c.id
             JOIN   files  f ON c.file_id = f.id
             WHERE  chunks_fts MATCH ?1
             ORDER  BY bm25_score
             LIMIT  ?2",
        )?;
        let rows = stmt.query_map(params![safe_query, limit as i64], |row| {
            let meta_str: String = row.get(4)?;
            Ok(FtsResult {
                id:         row.get(0)?,
                file_id:    row.get(1)?,
                tier:       row.get::<_, i64>(2)? as u8,
                content:    row.get(3)?,
                meta: serde_json::from_str(&meta_str)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                bm25_score: row.get(5)?,
                file_uri:   row.get(6)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    pub fn stats(&self) -> Result<DbStats, IndexerError> {
        let conn = self.conn()?;
        let files: i64 =
            conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        let chunks: i64 =
            conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        let chunks_embedded: i64 = conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE lance_id IS NOT NULL", [], |r| r.get(0),
        )?;
        let edges: i64 =
            conn.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
        let chunks_fts_docs: i64 =
            conn.query_row("SELECT COUNT(*) FROM chunks_fts", [], |r| r.get(0))?;

        let mut stmt = conn.prepare(
            "SELECT mime_type, COUNT(*) AS n FROM files GROUP BY mime_type ORDER BY n DESC",
        )?;
        let mime_counts: Vec<(String, i64)> = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(DbStats { files, chunks, chunks_embedded, edges, chunks_fts_docs, mime_counts })
    }

    // ── FTS5 query sanitisation ───────────────────────────────────────────────

    const STOP_WORDS: &'static [&'static str] = &[
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "in", "on", "at", "to", "for", "of",
        "and", "or", "but", "if", "with", "by", "from", "as", "it", "its",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "his",
        "her", "they", "their", "what", "which", "who", "how", "when",
        "where", "why", "this", "that", "these", "those",
    ];

    /// Strip FTS5 operator characters `( ) : ^ *`, balance quotes for phrase
    /// search, filter common stop words, then join with OR.  Falls back to all
    /// tokens when every token is a stop word (e.g. a single-word query "the").
    /// Double-quotes pass through so callers can use FTS5 phrase syntax.
    pub fn sanitize_fts_query(query: &str) -> String {
        let sanitized: String = query
            .chars()
            .map(|c| if "():^*".contains(c) { ' ' } else { c })
            .collect();
        // FTS5 requires balanced double-quotes; drop the last one if odd.
        let sanitized = if sanitized.chars().filter(|&c| c == '"').count() % 2 != 0 {
            let last = sanitized.rfind('"').unwrap();
            let mut s = sanitized;
            s.remove(last);
            s
        } else {
            sanitized
        };
        let all_tokens: Vec<&str> = sanitized.split_whitespace().collect();
        if all_tokens.is_empty() {
            return "\"\"".to_string();
        }
        let filtered: Vec<&str> = all_tokens
            .iter()
            .copied()
            .filter(|t| !Self::STOP_WORDS.contains(&t.to_lowercase().as_str()))
            .collect();
        let tokens = if filtered.is_empty() { all_tokens } else { filtered };
        tokens.join(" OR ")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_test_db() -> (EnterpriseDb, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = EnterpriseDb::new(&path).unwrap();
        (db, dir)
    }

    #[test]
    fn schema_creates_all_tables() {
        let (db, _dir) = open_test_db();
        let conn = db.conn().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT name FROM sqlite_master \
                 WHERE type IN ('table', 'trigger') ORDER BY name",
            )
            .unwrap();
        let names: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();

        for expected in &[
            "files", "chunks", "edges", "chunks_fts",
            "chunks_ai", "chunks_au", "chunks_ad",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "sqlite_master missing: {expected}"
            );
        }
    }

    #[test]
    fn upsert_and_get_file() {
        let (db, _dir) = open_test_db();

        let id1 = db
            .upsert_file("file:///test/foo.txt", "text/plain", "abc123", 100, 1.0, Some(0.5))
            .unwrap();
        assert!(id1 > 0);

        let info = db.get_all_file_info().unwrap();
        let fi = info.get("file:///test/foo.txt").unwrap();
        assert_eq!(fi.content_hash, "abc123");
        assert_eq!(fi.modified_at, Some(0.5));

        // Update — must keep the same rowid
        let id2 = db
            .upsert_file("file:///test/foo.txt", "text/plain", "def456", 200, 2.0, None)
            .unwrap();
        assert_eq!(id1, id2);

        let info2 = db.get_all_file_info().unwrap();
        assert_eq!(info2["file:///test/foo.txt"].content_hash, "def456");
    }

    #[test]
    fn insert_chunks_and_fts_sync() {
        let (db, _dir) = open_test_db();
        let file_id = db
            .upsert_file("file:///test/a.md", "text/markdown", "h1", 50, 1.0, None)
            .unwrap();

        let chunks = vec![
            ChunkInput {
                tier: 1,
                chunk_index: 0,
                content: "hello world".to_string(),
                token_count: Some(2),
                meta: serde_json::json!({}),
                chunker_method: None,
            },
            ChunkInput {
                tier: 2,
                chunk_index: 0,
                content: "foo bar baz".to_string(),
                token_count: Some(3),
                meta: serde_json::json!({}),
                chunker_method: None,
            },
        ];

        let ids = db.insert_chunks(file_id, &chunks).unwrap();
        assert_eq!(ids.len(), 2);

        let stats = db.stats().unwrap();
        assert_eq!(stats.chunks, 2);
        assert_eq!(stats.chunks_fts_docs, 2);
    }

    #[test]
    fn delete_file_cascades() {
        let (db, _dir) = open_test_db();
        let file_id = db
            .upsert_file("file:///test/b.md", "text/markdown", "h2", 50, 1.0, None)
            .unwrap();

        let chunk_ids = db
            .insert_chunks(
                file_id,
                &[ChunkInput {
                    tier: 1,
                    chunk_index: 0,
                    content: "chunk one".to_string(),
                    token_count: None,
                    meta: serde_json::json!({}),
                    chunker_method: None,
                }],
            )
            .unwrap();

        db.insert_edges(&[EdgeInput {
            src_chunk: chunk_ids[0],
            dst_uri: "file:///other.md".to_string(),
            edge_type: EdgeType::LinksTo,
            meta: serde_json::json!({}),
        }])
        .unwrap();

        let before = db.stats().unwrap();
        assert_eq!(before.files, 1);
        assert_eq!(before.chunks, 1);
        assert_eq!(before.edges, 1);

        db.delete_file("file:///test/b.md").unwrap();

        let after = db.stats().unwrap();
        assert_eq!(after.files, 0);
        assert_eq!(after.chunks, 0);
        assert_eq!(after.edges, 0);
        assert_eq!(after.chunks_fts_docs, 0);
    }

    #[test]
    fn fts_search_returns_ranked_results() {
        let (db, _dir) = open_test_db();
        let file_id = db
            .upsert_file("file:///test/c.md", "text/markdown", "h3", 50, 1.0, None)
            .unwrap();

        db.insert_chunks(
            file_id,
            &[
                ChunkInput {
                    tier: 1,
                    chunk_index: 0,
                    content: "quarterly revenue report shows growth".to_string(),
                    token_count: None,
                    meta: serde_json::json!({}),
                    chunker_method: None,
                },
                ChunkInput {
                    tier: 1,
                    chunk_index: 1,
                    content: "unrelated document about weather".to_string(),
                    token_count: None,
                    meta: serde_json::json!({}),
                    chunker_method: None,
                },
                ChunkInput {
                    tier: 1,
                    chunk_index: 2,
                    content: "quarterly earnings and revenue summary".to_string(),
                    token_count: None,
                    meta: serde_json::json!({}),
                    chunker_method: None,
                },
            ],
        )
        .unwrap();

        let results = db.fts_search("quarterly revenue", 10).unwrap();
        assert!(!results.is_empty(), "FTS search returned no results");
        assert!(results.iter().any(|r| r.content.contains("growth")));
        assert!(results.iter().any(|r| r.content.contains("earnings")));
        // Unrelated document should not appear (no match on "quarterly revenue")
        assert!(!results.iter().any(|r| r.content.contains("weather")));
    }

    #[test]
    fn stats_mime_counts_and_coverage() {
        let (db, _dir) = open_test_db();
        let fid1 = db.upsert_file("file:///a.md",  "text/markdown", "h1", 10, 1.0, None).unwrap();
        let fid2 = db.upsert_file("file:///b.md",  "text/markdown", "h2", 10, 1.0, None).unwrap();
        let _fid3 = db.upsert_file("file:///c.txt", "text/plain",   "h3", 10, 1.0, None).unwrap();

        let ids = db.insert_chunks(fid1, &[ChunkInput {
            tier: 1, chunk_index: 0, content: "a".into(), token_count: None,
            meta: serde_json::json!({}), chunker_method: None,
        }]).unwrap();
        db.set_chunk_lance_ids(&std::collections::HashMap::from([(ids[0], 999i64)])).unwrap();
        db.insert_chunks(fid2, &[ChunkInput {
            tier: 1, chunk_index: 0, content: "b".into(), token_count: None,
            meta: serde_json::json!({}), chunker_method: None,
        }]).unwrap();

        let s = db.stats().unwrap();
        assert_eq!(s.files, 3);
        assert_eq!(s.chunks_embedded, 1);
        assert_eq!(s.chunks - s.chunks_embedded, 1);

        assert_eq!(s.mime_counts[0].0, "text/markdown");
        assert_eq!(s.mime_counts[0].1, 2);
        assert_eq!(s.mime_counts[1].0, "text/plain");
    }

    #[test]
    fn fts_sanitize_rejects_special_chars() {
        let (db, _dir) = open_test_db();
        // Must not panic or return an error on FTS5 operator characters
        let result = db.fts_search("error: \"null\"", 10);
        assert!(result.is_ok(), "fts_search panicked on special chars");

        let s = EnterpriseDb::sanitize_fts_query("error: \"null\"");
        assert!(!s.contains(':'), "colon should be stripped");
        // Double-quotes now pass through for FTS5 phrase search, but must be balanced
        let qcount = s.chars().filter(|&c| c == '"').count();
        assert_eq!(qcount % 2, 0, "quotes must be balanced for FTS5");

        // Empty input after stripping → sentinel
        assert_eq!(EnterpriseDb::sanitize_fts_query("()^*:\"\""), "\"\"");
    }
}
