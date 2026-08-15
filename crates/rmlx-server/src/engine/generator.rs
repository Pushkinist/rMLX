//! `Generator` trait — the seam between HTTP routes and inference engines.
//! `NotReadyGenerator` — stage-1 placeholder returning 503.

use std::pin::Pin;

use futures::stream::{self, Stream};
use rmlx_core::Error;

use super::types::{GenerationRequest, GenerationToken};

/// Async stream of tokens. Cancellable by dropping the stream.
pub trait Generator: Send + Sync {
    /// Begin token generation for `req` and return a `Pin<Box<dyn Stream>>` of produced tokens.
    fn generate(
        &self,
        req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>>;

    /// Return current prompt-cache hit/miss/bytes stats for this generator.
    ///
    /// Returns `None` when the generator does not maintain a prompt cache
    /// (e.g. `NotReadyGenerator`) or the cache has not been used yet.
    fn cache_stats(&self) -> Option<rmlx_models::CacheStats> {
        None
    }

    /// Return the KV-cache bytes from the last completed request for this generator.
    ///
    /// Reads the counter on the model instance this generator serves, written
    /// by `generate_greedy` at request boundary (N16). Returns 0 when no
    /// request has completed yet or the arch does not track KV-cache bytes.
    ///
    /// This is a **display** surface: it hands back the last-known count with
    /// no generation boundary to check it against, so it may be a previous
    /// request's figure. Do not record from it — a recording path brackets the
    /// generation with `Architecture::kv_cache_bytes_sample` and refuses when
    /// the store sequence did not advance.
    fn kv_cache_bytes(&self) -> u64 {
        0
    }

    /// Return the load-phase timing from the most recent `load_model` call (N17).
    ///
    /// Returns `None` when no model has been loaded yet or the generator does
    /// not participate in arch-dispatch load (e.g. `NotReadyGenerator`).
    fn load_phases(&self) -> Option<rmlx_models::LoadPhases> {
        None
    }

    /// Effective per-process maximum prompt-context length for this generator
    /// (A2 — context_length_exceeded guard).
    ///
    /// Captured at load time as `min(--max-ctx override, model
    /// max_position_embeddings, KV_MAX_SEQ_DEFAULT=4096)` — i.e. the
    /// actually-allocated KV-cache capacity, not the model's theoretical max.
    /// The route handler compares `prompt_tokens.len()` against this value and
    /// returns HTTP 400 `context_length_exceeded` when the prompt overflows.
    ///
    /// Default `usize::MAX` makes the guard a no-op for generators that don't
    /// participate in KV-cache sizing (e.g. `NotReadyGenerator`); the
    /// fall-through 503 path still catches actual runtime overflows there.
    fn effective_max_ctx(&self) -> usize {
        usize::MAX
    }
}

// ── NotReadyGenerator ────────────────────────────────────────────────────────

/// Stage-1 placeholder. Returns a single `Err(Error::Other("generator not
/// ready"))` item; the route layer translates this to 503.
#[derive(Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "unit struct placeholder — no fields; adding state would make it a real generator, not a placeholder"
)]
pub struct NotReadyGenerator;

impl Generator for NotReadyGenerator {
    fn generate(
        &self,
        _req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        Box::pin(stream::once(async {
            Err(Error::Other("generator not ready".to_owned()))
        }))
    }
}
