//! config.rs — file_indexer.toml loader.
//!
//! Resolution order for the config file itself: `<index_dir>/file_indexer.toml`
//! first, then `./file_indexer.toml` (cwd) as a fallback, else built-in defaults
//! (no error, no auto-created file — unlike `chunker_map.rs`, config absence is
//! normal). Per-setting resolution within the loaded config follows CLI flag >
//! config file > built-in default, except where noted (fusion weights keep their
//! existing env-var override on TOP of this chain, documented as tuning-only —
//! see `search.rs::FusionWeights::resolve`; `[indexer].extra_ignore_dirs` is
//! UNIONED with `--exclude`, not overridden, since more ignoring is always safe).

use crate::error::IndexerError;
use serde::Deserialize;
use std::path::Path;

const CONFIG_FILENAME: &str = "file_indexer.toml";

#[derive(Debug, Deserialize, Default, Clone)]
pub struct RawConfig {
    #[serde(default)]
    pub embedder: RawEmbedderConfig,
    #[serde(default)]
    pub fusion: RawFusionConfig,
    #[serde(default)]
    pub indexer: RawIndexerConfig,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct RawEmbedderConfig {
    pub onnx_model_dir: Option<String>,
    /// ONNX model filename within `onnx_model_dir`, e.g. `model_fp16.onnx` to
    /// select the fp16 export for the CUDA/GPU execution provider (ADR-006 —
    /// the default int8-quantized export forces ~156 CPU-fallback memcpy nodes
    /// on the CUDA EP, ~20x slower). Falls back to the `NOMIC_ONNX_FILE` env
    /// var, then to `"nomic-embed-text-v1.5.onnx"`. The tokenizer filename
    /// (`tokenizer.json`) is fixed — it's shared across precisions.
    pub onnx_model_file: Option<String>,
    pub ort_dylib_path: Option<String>,
    pub embedding_dim: Option<usize>,
    /// Reserved: not yet enforced anywhere (would require probing the loaded
    /// ORT dylib's version at `Embedder::load` time). Parsed now so the field
    /// name is stable in `file_indexer.toml` when a future phase wires it up.
    #[allow(dead_code)]
    pub required_onnxruntime_version: Option<String>,
    pub load_timeout_secs: Option<u64>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct RawFusionConfig {
    pub dense_weight: Option<f64>,
    pub sparse_weight: Option<f64>,
    pub path_weight: Option<f64>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct RawIndexerConfig {
    pub extra_ignore_dirs: Option<Vec<String>>,
    pub embed_batch_size: Option<usize>,
}

impl RawConfig {
    /// Load `file_indexer.toml`, trying `<index_dir>/file_indexer.toml` first,
    /// then `./file_indexer.toml` (cwd), else returns `RawConfig::default()`
    /// (all fields `None` — no error, config absence is the normal case).
    pub fn load(index_dir: &Path) -> Result<Self, IndexerError> {
        let candidates = [index_dir.join(CONFIG_FILENAME), Path::new(CONFIG_FILENAME).to_path_buf()];
        for path in &candidates {
            if path.exists() {
                let text = std::fs::read_to_string(path).map_err(|e| IndexerError::Io {
                    path: path.clone(),
                    source: e,
                })?;
                let cfg: RawConfig = toml::from_str(&text)
                    .map_err(|e| IndexerError::Config(format!(
                        "failed to parse {}: {}", path.display(), e
                    )))?;
                return Ok(cfg);
            }
        }
        Ok(Self::default())
    }

    /// Hard-error if the file declares an `embedding_dim` that disagrees with
    /// this binary's compiled dimension. `EMBEDDING_DIM` is not yet a live
    /// runtime value (see module doc), so a mismatch here means the config is
    /// describing a different build than the one running — silently ignoring
    /// it would let the file lie about the corpus's actual vector width.
    pub fn validate_embedding_dim(&self, compiled_dim: usize) -> Result<(), IndexerError> {
        if let Some(configured) = self.embedder.embedding_dim
            && configured != compiled_dim
        {
            return Err(IndexerError::Config(format!(
                    "file_indexer.toml [embedder].embedding_dim = {} does not match this build's \
                     compiled EMBEDDING_DIM = {} (src/indexer.rs). These must agree — the compiled \
                     binary always uses EMBEDDING_DIM regardless of what the config file says, so a \
                     mismatch means the config is describing a different build than the one running. \
                     Fix embedding_dim in file_indexer.toml to {}, or rebuild the binary with \
                     EMBEDDING_DIM changed to match.",
                configured, compiled_dim, compiled_dim
            )));
        }
        Ok(())
    }
}

/// Generic CLI > file > default precedence resolver. `cli` wins if `Some`,
/// else `file` wins if `Some`, else `default`.
pub fn resolve<T: Clone>(cli: Option<T>, file: Option<T>, default: T) -> T {
    cli.or(file).unwrap_or(default)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn resolve_prefers_cli_over_file_over_default() {
        assert_eq!(resolve(Some(3usize), Some(7usize), 32usize), 3);
    }

    #[test]
    fn resolve_falls_back_to_file_when_no_cli() {
        assert_eq!(resolve(None, Some(7usize), 32usize), 7);
    }

    #[test]
    fn resolve_falls_back_to_default_when_nothing_set() {
        assert_eq!(resolve::<usize>(None, None, 32), 32);
    }

    #[test]
    fn config_load_reads_index_dir_file_over_cwd() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("file_indexer.toml"),
            "[indexer]\nembed_batch_size = 7\n",
        ).unwrap();
        let cfg = RawConfig::load(dir.path()).unwrap();
        assert_eq!(cfg.indexer.embed_batch_size, Some(7));
    }

    #[test]
    fn config_load_defaults_when_absent() {
        let dir = TempDir::new().unwrap();
        let cfg = RawConfig::load(dir.path()).unwrap();
        assert!(cfg.embedder.onnx_model_dir.is_none());
        assert!(cfg.indexer.embed_batch_size.is_none());
    }

    #[test]
    fn config_load_reads_onnx_model_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("file_indexer.toml"),
            "[embedder]\nonnx_model_file = \"model_fp16.onnx\"\n",
        ).unwrap();
        let cfg = RawConfig::load(dir.path()).unwrap();
        assert_eq!(cfg.embedder.onnx_model_file.as_deref(), Some("model_fp16.onnx"));
    }

    #[test]
    fn validate_embedding_dim_errors_on_mismatch() {
        let mut cfg = RawConfig::default();
        cfg.embedder.embedding_dim = Some(768);
        let result = cfg.validate_embedding_dim(256);
        assert!(matches!(result, Err(IndexerError::Config(_))));
    }

    #[test]
    fn validate_embedding_dim_ok_when_absent_or_matching() {
        let cfg = RawConfig::default(); // embedding_dim = None
        assert!(cfg.validate_embedding_dim(256).is_ok());

        let mut cfg2 = RawConfig::default();
        cfg2.embedder.embedding_dim = Some(256);
        assert!(cfg2.validate_embedding_dim(256).is_ok());
    }
}
