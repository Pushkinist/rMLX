//! Per-session KV-reuse registry (N2).
//!
//! ## Purpose
//!
//! Multi-turn conversations share a session ID (`X-Session-Id` header). Each
//! turn's chat template renders the **full** conversation history, so turn 2's
//! `prompt_tokens` starts with exactly the same tokens as turn 1. The
//! per-arch `PromptCache` (N1) already handles prefix matching — N2's job is
//! to ensure the prior-turn's PromptCache slot is not evicted before the next
//! turn arrives.
//!
//! ## How slot reservation works
//!
//! `SessionCache` tracks active sessions and reports `active_count()`. The
//! caller (`ArchGenerator::generate`) passes
//! `base_slots + session_cache.active_count()` as `prompt_cache_slots` to
//! `generate_greedy`. This increases the per-arch PromptCache capacity so
//! there is always headroom for every active session, preventing FIFO eviction
//! from clobbering a live session's snapshot.
//!
//! ## LRU eviction
//!
//! When the number of entries hits `max_sessions`, the session with the oldest
//! `last_used` timestamp is dropped before inserting the new one. Dropped
//! sessions lose their slot-reservation benefit on subsequent turns — the next
//! request will still get an N1 cache hit *if* the PromptCache slot has not been
//! overwritten (best-effort).
//!
//! Default `max_sessions`: `--session-cache-max-sessions` CLI flag (env: `RMLX_SESSION_CACHE_MAX_SESSIONS`), default 64.
//!
//! ## Cross-session safety
//!
//! Session state is purely the last-used timestamp and the prior prompt length.
//! KV tensors live inside the per-arch `PromptCache` global; they are never
//! copied across session entries. Two session IDs always produce distinct
//! PromptCache lookups keyed by different token sequences.

use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// SessionEntry
// ---------------------------------------------------------------------------

/// Bookkeeping record for one live session.
///
/// Does NOT hold KV tensors — those live in the per-arch PromptCache global.
/// We track only the timestamp (for LRU ordering) and the prompt length of
/// the last request (for diagnostics / future slot affinity).
#[derive(Debug)]
struct SessionEntry {
    /// Wall-clock time of the last request in this session.
    last_used: Instant,
    /// Number of prompt tokens in the last request (diagnostics only).
    ///
    /// Not read back in this crate — kept for future slot-affinity work (M29).
    #[allow(dead_code)]
    last_prompt_len: usize,
}

// ---------------------------------------------------------------------------
// SessionKey
// ---------------------------------------------------------------------------

/// Composite key: (model_id, session_id).
///
/// Prevents cross-model key collisions when the server serves multiple models.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed key struct — two fields are the complete session-cache key contract; adding a field requires updating all SessionKey construction sites and Hash impl"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionKey {
    /// Registry model id this session belongs to.
    pub model_id: String,
    /// Client-supplied session identifier from the `X-Session-Id` header.
    pub session_id: String,
}

// ---------------------------------------------------------------------------
// SessionCache
// ---------------------------------------------------------------------------

/// LRU registry of active sessions.
///
/// `Mutex<SessionCache>` is held in `AppState` and accessed from
/// `ArchGenerator::generate` (inside `spawn_blocking`).
pub struct SessionCache {
    entries: HashMap<SessionKey, SessionEntry>,
    max_sessions: usize,
}

impl std::fmt::Debug for SessionCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionCache")
            .field("active", &self.entries.len())
            .field("max", &self.max_sessions)
            .finish()
    }
}

impl SessionCache {
    /// Create a new cache.
    ///
    /// `max_sessions = 0` is treated as 1 (must allow at least one session).
    pub fn new(max_sessions: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_sessions: max_sessions.max(1),
        }
    }

    /// Number of currently tracked sessions.
    ///
    /// Used to compute the effective `prompt_cache_slots` value passed to
    /// `generate_greedy`: `base_slots + session_cache.active_count()`.
    pub fn active_count(&self) -> usize {
        self.entries.len()
    }

    /// Record a request for `key` with `prompt_len` tokens.
    ///
    /// If the session is new and we are at capacity, the oldest session
    /// (minimum `last_used`) is evicted first. Returns `true` if this was a
    /// **returning** session (hit), `false` if it was freshly created (miss).
    pub fn touch(&mut self, key: SessionKey, prompt_len: usize) -> bool {
        let is_hit = self.entries.contains_key(&key);

        if !is_hit && self.entries.len() >= self.max_sessions {
            // LRU eviction: drop the entry with the oldest last_used.
            // `min_by_key` is O(n) over max_sessions entries.
            // At the default max=64 this is negligible.
            if let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, v)| v.last_used)
                .map(|(k, _)| k.clone())
            {
                tracing::debug!(
                    evicted_session = %oldest_key.session_id,
                    model_id = %oldest_key.model_id,
                    "SessionCache: evicted oldest session (LRU)"
                );
                self.entries.remove(&oldest_key);
            }
        }

        self.entries.insert(
            key,
            SessionEntry {
                last_used: Instant::now(),
                last_prompt_len: prompt_len,
            },
        );

        is_hit
    }

    /// Remove a session explicitly (e.g. model unload). No-op if not present.
    #[allow(dead_code)]
    pub fn remove(&mut self, key: &SessionKey) {
        self.entries.remove(key);
    }

    /// Remove all sessions for a given model (e.g. model swap).
    pub fn remove_model(&mut self, model_id: &str) {
        self.entries.retain(|k, _| k.model_id != model_id);
    }
}

/// Prompt-cache slot count for a request that carries an `X-Session-Id`.
///
/// The configured base widened by one slot per active session, so the FIFO
/// `PromptCache` never evicts a live session's snapshot. `None` means "leave the
/// generator's own setting alone".
///
/// Two properties this exists to hold, both of which an inline expression at
/// each route got wrong:
///
/// - **A base of 0 stays 0.** Zero slots is the disabled cache; a request header
///   must not be able to turn on a cache the operator switched off. It would
///   also alternate capacities across mixed traffic, and `ensure` rebuilds — and
///   resets the hit/miss counters — every time the capacity changes.
/// - **The base is the operator's, not a literal.** Widening a hard-coded 4
///   hands someone who configured 8 slots a smaller cache than they asked for.
#[must_use]
pub const fn effective_prompt_cache_slots(base: usize, active_sessions: usize) -> Option<usize> {
    if base == 0 {
        None
    } else {
        Some(base + active_sessions)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "session_cache_tests.rs"]
mod tests;
