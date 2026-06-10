//! error.rs — Unified error type for the entire indexer.
//!
//! All modules return `Result<T, IndexerError>`.  The `thiserror` crate
//! generates `std::error::Error` + `Display` impls from the `#[error(...)]`
//! annotations, replacing Python's bare `except Exception` patterns with
//! typed, inspectable variants.

use std::path::PathBuf;
use thiserror::Error;

#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum IndexerError {
    /// Filesystem I/O failure (read, stat, walk).
    #[error("I/O error on {path}: {source}")]
    Io {
        path:   PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// File exceeds `MAX_FILE_SIZE`; rejected before allocation.
    #[error("file too large: {path} is {size} bytes (limit {limit})")]
    FileTooLarge {
        path:  PathBuf,
        size:  u64,
        limit: u64,
    },

    /// A file was referenced but does not exist.
    #[error("file not found: {0}")]
    NotFound(PathBuf),

    /// MIME type has no registered parser.
    #[error("unsupported MIME type: {0}")]
    UnsupportedMime(String),

    /// Parsing failed for a supported format (e.g. malformed XLSX).
    #[error("parse error in {file}: {message}")]
    Parse {
        file:    String,
        message: String,
    },

    /// SQLite / database layer errors.
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// ONNX Runtime embedding errors.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Vector database (LanceDB) errors.
    #[error("vector store error: {0}")]
    VectorStore(String),

    /// Any other error not covered above (escape hatch for third-party crates).
    #[error("unexpected error: {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}
