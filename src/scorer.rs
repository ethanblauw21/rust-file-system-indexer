use crate::db::{ChunkRow, EnterpriseDb};
use crate::error::IndexerError;
use crate::indexer::LanceStore;
use std::collections::HashSet;

pub struct ScoreStats {
    pub total:            usize,
    pub flagged:          usize,
    pub structural_only:  usize,
    pub coherence_only:   usize,
    pub both:             usize,
}

const STRUCTURAL_THRESHOLD: f32 = 0.5;
const COHERENCE_THRESHOLD:  f32 = 0.6;

const TIER1_LIMIT: i64 = 500;
const TIER2_LIMIT: i64 = 1500;

pub fn structural_score(chunk: &ChunkRow) -> f32 {
    let content      = &chunk.content;
    let token_count  = chunk.token_count.unwrap_or(0);

    // Check 1 — token count in healthy range (weight 0.25)
    let tier_limit = match chunk.tier {
        1 => TIER1_LIMIT,
        2 => TIER2_LIMIT,
        _ => i64::MAX, // tier 3 has no upper limit
    };
    let upper_ok  = tier_limit == i64::MAX || token_count < tier_limit * 99 / 100;
    let score1 = if token_count >= 15 && upper_ok { 0.25_f32 } else { 0.0 };

    // Check 2 — sentence completeness: ends with . ? ! : " ` (weight 0.25)
    let trimmed = content.trim_end();
    let score2 = if trimmed.ends_with(['.', '?', '!', ':', '"', '`']) { 0.25 } else { 0.0 };

    // Check 3 — bracket balance: (, [, {, < all closed (weight 0.20)
    let score3 = if is_bracket_balanced(content) { 0.20 } else { 0.0 };

    // Check 4 — whitespace density ≤ 40% (weight 0.15)
    let total_chars = content.chars().count();
    let ws_chars    = content.chars().filter(|c| c.is_whitespace()).count();
    let score4 = if total_chars == 0 || ws_chars as f32 / total_chars as f32 <= 0.40 {
        0.15
    } else {
        0.0
    };

    // Check 5 — repetition ratio: unique/total ≥ 0.3 (weight 0.15)
    let tokens: Vec<&str>  = content.split_whitespace().collect();
    let total_toks          = tokens.len();
    let unique_toks: HashSet<&str> = tokens.iter().copied().collect();
    let score5 = if total_toks == 0 || unique_toks.len() as f32 / total_toks as f32 >= 0.3 {
        0.15
    } else {
        0.0
    };

    score1 + score2 + score3 + score4 + score5
}

fn is_bracket_balanced(s: &str) -> bool {
    // Only track (), [], {} — angle brackets are comparison operators or generics
    // in most languages (Rust `->`, TypeScript `=>`, generics) and produce too many
    // false negatives when included.
    let mut stack: Vec<char> = Vec::new();
    for c in s.chars() {
        match c {
            '(' | '[' | '{' => stack.push(c),
            ')' => { if stack.last() == Some(&'(') { stack.pop(); } else { return false; } }
            ']' => { if stack.last() == Some(&'[') { stack.pop(); } else { return false; } }
            '}' => { if stack.last() == Some(&'{') { stack.pop(); } else { return false; } }
            _ => {}
        }
    }
    stack.is_empty()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot:    f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

/// Returns the cosine similarity of this chunk's embedding vs the mean of its
/// parent-tier (tier+1) embeddings.  Returns `None` for Tier-3 chunks, chunks
/// without a `lance_id`, or files with no parent-tier vectors.
async fn coherence_score(chunk: &ChunkRow, db: &EnterpriseDb, lance: &LanceStore) -> Option<f32> {
    if chunk.tier >= 3 { return None; }

    let my_id      = chunk.lance_id?;
    let parent_ids = db.get_lance_ids_for_tier(chunk.file_id, chunk.tier + 1).ok()?;
    if parent_ids.is_empty() { return None; }

    // Fetch this chunk's vector + parent vectors in one round-trip.
    let mut all_ids = vec![my_id];
    all_ids.extend_from_slice(&parent_ids);
    let vectors = lance.fetch_vectors_by_ids(&all_ids).await.ok()?;

    let my_vec = vectors.get(&my_id)?;
    let parent_vecs: Vec<&Vec<f32>> = parent_ids.iter().filter_map(|id| vectors.get(id)).collect();
    if parent_vecs.is_empty() { return None; }

    // Mean of parent vectors (not unit-norm), then full cosine sim.
    let dim = my_vec.len();
    let mut mean = vec![0.0f32; dim];
    for pv in &parent_vecs {
        for (i, &v) in pv.iter().enumerate() { mean[i] += v; }
    }
    let n = parent_vecs.len() as f32;
    for v in &mut mean { *v /= n; }

    Some(cosine_sim(my_vec, &mean))
}

/// Scores all unscored chunks (or all chunks when `rescore = true`).
/// Structural heuristics are always computed; embedding coherence is computed
/// for chunks that have a `lance_id` and a parent tier in the vector store.
/// `on_progress(checked, total)` is called every 50 chunks when provided.
pub async fn score_all(
    db:          &EnterpriseDb,
    lance:       &LanceStore,
    rescore:     bool,
    on_progress: Option<&(dyn Fn(usize, usize) + Sync)>,
) -> Result<ScoreStats, IndexerError> {
    let chunks: Vec<ChunkRow> = if rescore {
        db.get_chunks_for_rescoring()?
    } else {
        db.get_unscored_chunks()?
    };

    let total = chunks.len();
    let mut flagged         = 0usize;
    let mut structural_only = 0usize;
    let mut coherence_only  = 0usize;
    let mut both            = 0usize;

    for (i, chunk) in chunks.iter().enumerate() {
        let s = structural_score(chunk);
        let c = coherence_score(chunk, db, lance).await;

        // `<=`, not `<`: chunks that fail token-count + sentence checks but pass
        // bracket/whitespace/repetition score *exactly* 0.50 (import blocks,
        // single-line `#pragma once`). An exclusive test let that dead-zone
        // escape flagging; inclusive catches it without snaring the 0.55 cluster.
        let struct_bad = s <= STRUCTURAL_THRESHOLD;
        let coher_bad  = c.is_some_and(|v| v < COHERENCE_THRESHOLD);
        let is_flagged = struct_bad || coher_bad;

        db.set_chunk_scores(chunk.id, s, c, is_flagged)?;

        if is_flagged {
            flagged += 1;
            match (struct_bad, coher_bad) {
                (true,  true)  => both            += 1,
                (true,  false) => structural_only += 1,
                (false, true)  => coherence_only  += 1,
                (false, false) => unreachable!(),
            }
        }

        if let Some(f) = on_progress
            && (i % 50 == 49 || i + 1 == total) {
                f(i + 1, total);
            }
    }

    Ok(ScoreStats { total, flagged, structural_only, coherence_only, both })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ChunkRow;

    fn make_chunk(content: &str, tier: u8, tokens: i64) -> ChunkRow {
        ChunkRow {
            id:               1,
            file_id:          1,
            tier,
            chunk_index:      0,
            content:          content.to_string(),
            token_count:      Some(tokens),
            meta:             serde_json::Value::Object(Default::default()),
            lance_id:         None,
            file_uri:         String::new(),
            mime_type:        String::new(),
            is_flagged:       false,
            chunker_method:   None,
            structural_score: None,
            coherence_score:  None,
        }
    }

    #[test]
    fn cosine_sim_identical_is_one() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn perfect_chunk_scores_one() {
        // 20 tokens, balanced, good sentence, normal whitespace, varied
        let text = "The quick brown fox jumps over the lazy dog, right here today. \
                    A second sentence follows with more unique words and content.";
        let chunk = make_chunk(text, 1, 20);
        let score = structural_score(&chunk);
        assert!((score - 1.0).abs() < 1e-6, "expected 1.0, got {score}");
    }

    #[test]
    fn too_few_tokens_penalised() {
        let chunk = make_chunk("Short.", 1, 5);
        let score = structural_score(&chunk);
        assert!(score < 1.0);
        // all other checks pass → 0.75
        assert!((score - 0.75).abs() < 1e-6, "expected 0.75, got {score}");
    }

    #[test]
    fn incomplete_sentence_penalised() {
        let text = "The quick brown fox jumps over the lazy dog and then continued";
        let chunk = make_chunk(text, 1, 20);
        let score = structural_score(&chunk);
        // token ok, no sentence end, balanced, low ws, varied → 0.75
        assert!((score - 0.75).abs() < 1e-6, "expected 0.75, got {score}");
    }

    #[test]
    fn unmatched_bracket_penalised() {
        let text = "The function foo(bar, baz returns a value.";
        let chunk = make_chunk(text, 1, 20);
        let score = structural_score(&chunk);
        assert!(score < 1.0);
        // token ok, sentence ok, bracket bad, ws ok, varied → 0.80
        assert!((score - 0.80).abs() < 1e-6, "expected 0.80, got {score}");
    }

    #[test]
    fn high_whitespace_penalised() {
        // >40% whitespace
        let text = "word   \n\n\n   word  \t\t  word     \n\n\n   word   thing.";
        let chunk = make_chunk(text, 1, 20);
        let score = structural_score(&chunk);
        assert!(score < 1.0);
    }

    #[test]
    fn repetitive_content_penalised() {
        // unique/total < 0.3: "the" repeated 10x out of 12 tokens
        let text = "the the the the the the the the the the dog cat.";
        let chunk = make_chunk(text, 1, 20);
        let score = structural_score(&chunk);
        assert!(score < 1.0);
    }

    #[test]
    fn below_threshold_is_flagged() {
        // 5 tokens (token fails) + no sentence end (sentence fails) + unmatched ( (bracket fails)
        // passes only ws + rep → 0.0 + 0.0 + 0.0 + 0.15 + 0.15 = 0.30
        let chunk = make_chunk("hello world foo (bar", 1, 5);
        let score = structural_score(&chunk);
        assert!(score < STRUCTURAL_THRESHOLD, "score {score} should be below threshold");
    }
}
