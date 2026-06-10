use crate::error::IndexerError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

const MAP_FILENAME: &str = "chunker_map.toml";

const DEFAULT_METHODS: &[(&str, &str)] = &[
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx/v1"),
    ("text/markdown",       "markdown/v1"),
    ("text/plain",          "plaintext/v1"),
    ("application/pdf",     "pdf/v1"),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx/v1"),
    ("text/csv",            "csv/v1"),
    ("text/typescript",     "typescript/v1"),
    ("text/javascript",     "typescript/v1"),
    ("application/javascript", "typescript/v1"),
    ("text/x-rust",        "rust/v1"),
    ("text/x-python",      "plaintext/v1"),
    ("_default",            "generic/v1"),
];

#[derive(Debug, Serialize, Deserialize)]
struct ChunkerMapFile {
    methods: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ChunkerMap {
    methods: HashMap<String, String>,
}

impl ChunkerMap {
    /// Load from `<index_dir>/chunker_map.toml`, creating the file with
    /// defaults if it doesn't exist.
    pub fn load_or_create(index_dir: &Path) -> Result<Self, IndexerError> {
        let path = index_dir.join(MAP_FILENAME);
        if path.exists() {
            let text = std::fs::read_to_string(&path).map_err(|e| IndexerError::Io {
                path: path.clone(),
                source: e,
            })?;
            let file: ChunkerMapFile = toml::from_str(&text)
                .map_err(|e| IndexerError::Other(e.into()))?;
            Ok(Self { methods: file.methods })
        } else {
            let methods: HashMap<String, String> = DEFAULT_METHODS
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            let file = ChunkerMapFile { methods: methods.clone() };
            let text = toml::to_string_pretty(&file)
                .map_err(|e| IndexerError::Other(e.into()))?;
            std::fs::write(&path, text).map_err(|e| IndexerError::Io {
                path: path.clone(),
                source: e,
            })?;
            Ok(Self { methods })
        }
    }

    /// Return the version-tagged method string for `mime`, falling back to
    /// `_default` when the MIME type has no explicit entry.
    pub fn method_for(&self, mime: &str) -> &str {
        self.methods
            .get(mime)
            .or_else(|| self.methods.get("_default"))
            .map(String::as_str)
            .unwrap_or("generic/v1")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tmp() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn creates_toml_on_first_run() {
        let dir = tmp();
        let map = ChunkerMap::load_or_create(dir.path()).unwrap();
        assert!(dir.path().join("chunker_map.toml").exists());
        assert_eq!(map.method_for("text/markdown"), "markdown/v1");
    }

    #[test]
    fn loads_existing_toml() {
        let dir = tmp();
        // First run creates it
        ChunkerMap::load_or_create(dir.path()).unwrap();
        // Second run loads it
        let map = ChunkerMap::load_or_create(dir.path()).unwrap();
        assert_eq!(map.method_for("text/plain"), "plaintext/v1");
    }

    #[test]
    fn unknown_mime_falls_back_to_default() {
        let dir = tmp();
        let map = ChunkerMap::load_or_create(dir.path()).unwrap();
        assert_eq!(map.method_for("application/x-unknown"), "generic/v1");
    }

    #[test]
    fn edited_toml_is_picked_up() {
        let dir = tmp();
        ChunkerMap::load_or_create(dir.path()).unwrap();
        let path = dir.path().join("chunker_map.toml");
        let text = std::fs::read_to_string(&path).unwrap();
        let text = text.replace("xlsx/v1", "xlsx/v2");
        std::fs::write(&path, text).unwrap();
        let map = ChunkerMap::load_or_create(dir.path()).unwrap();
        assert_eq!(
            map.method_for(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            "xlsx/v2"
        );
    }
}
