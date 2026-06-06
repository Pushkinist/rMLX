//! Model keep-alive + auto-unload TTL.
//!
//! This module owns the **per-model** keep-alive policy: parse Go-style
//! duration strings (`30s`, `15m`, `2h`, `24h`), resolve the precedence
//! chain (per-request, then env `RMLX_KEEP_ALIVE`, then CLI
//! `--idle-timeout-secs`, then the default 15 min), and expose the
//! active-decode lease that prevents an unload from racing a running
//! generation.
//!
//! The reaper is **per-model**, not a global poll loop: when a request
//! resolves a [`LoadedModel`] in the registry slot, the previously armed
//! `JoinHandle` is cancelled and a fresh `sleep(ttl).then(unload)` is spawned.
//! Reset is O(1) per request.
//!
//! ## Decode lease
//!
//! Every generation path (chat, embeddings, audio transcription, audio TTS)
//! acquires a [`DecodeLeaseGuard`] before the first decode step and holds it
//! through the entire response (including streaming). When the unload timer
//! fires, it checks the lease count; if non-zero the unload is suppressed
//! and re-armed for one TTL period (so the model stays resident long enough
//! to finish the active decode). This guarantees:
//!
//! 1. Streaming responses always complete — TTL never tears down a model
//!    mid-decode.
//! 2. Unloads are idempotent and safe to call from either the TTL timer or
//!    the LRU / cooperative-evict path.

#![allow(clippy::module_name_repetitions)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ── Policy enum ──────────────────────────────────────────────────────────────

/// Resolved keep-alive policy for one model slot.
///
/// Constructed by [`KeepAlivePolicy::resolve`] from the precedence chain
/// (per-request > env > flag > default) at request entry and at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "internal policy enum; adding a variant requires updating arm/reset call sites"
)]
pub enum KeepAlivePolicy {
    /// Pin forever — never unload via TTL. Set by negative input
    /// (`-1`, `-1s`) or the explicit "pin" request field.
    Pin,
    /// Unload immediately after the current response finishes. Set by `0`
    /// (zero seconds). The timer is armed with a near-zero TTL so the
    /// decode-lease guard's drop is the actual trigger.
    UnloadAfter,
    /// Idle TTL — unload after this many seconds of no requests. Set by
    /// positive integers (`30`, `15m`, `2h`) or the default 15 min.
    Idle(Duration),
}

impl KeepAlivePolicy {
    /// Default TTL when nothing else is configured: 15 minutes.
    ///
    /// Matches Ollama's `OLLAMA_KEEP_ALIVE` default (5 min) sized up for
    /// MLX's larger model loads, which take seconds-to-minutes to reload.
    pub const DEFAULT_TTL_SECS: u64 = 15 * 60;

    /// The duration to wait before unloading, or `None` for `Pin`.
    ///
    /// `UnloadAfter` returns a near-zero duration (1 ms) so the timer fires
    /// immediately after the decode-lease guard drops.
    #[must_use]
    pub fn ttl(self) -> Option<Duration> {
        match self {
            KeepAlivePolicy::Pin => None,
            KeepAlivePolicy::UnloadAfter => Some(Duration::from_millis(1)),
            KeepAlivePolicy::Idle(d) => Some(d),
        }
    }

    /// Human-readable seconds count for logs / events.
    ///
    /// `Pin` and `UnloadAfter` both report `0` — the field is informational and
    /// the policy variant is the source of truth for behavior.
    #[must_use]
    pub fn ttl_secs_for_log(self) -> u64 {
        match self {
            KeepAlivePolicy::Pin | KeepAlivePolicy::UnloadAfter => 0,
            KeepAlivePolicy::Idle(d) => d.as_secs(),
        }
    }

    /// Resolve the precedence chain.
    ///
    /// Order: per-request > env > CLI flag > default.
    /// Each input is `Option<Self>`; the first `Some` wins.
    #[must_use]
    pub fn resolve(
        per_request: Option<KeepAlivePolicy>,
        env: Option<KeepAlivePolicy>,
        flag: Option<KeepAlivePolicy>,
    ) -> KeepAlivePolicy {
        per_request
            .or(env)
            .or(flag)
            .unwrap_or(KeepAlivePolicy::Idle(Duration::from_secs(
                Self::DEFAULT_TTL_SECS,
            )))
    }
}

// ── Duration parsing ─────────────────────────────────────────────────────────

/// Parse a Go-style duration spec into a [`KeepAlivePolicy`].
///
/// Accepted shapes:
/// - `-1`, `-1s`, `-30m` → [`KeepAlivePolicy::Pin`] (any negative).
/// - `0`, `0s` → [`KeepAlivePolicy::UnloadAfter`].
/// - `30`, `30s` → `Idle(30s)`.
/// - `15m` → `Idle(900s)`.
/// - `2h` → `Idle(7200s)`.
/// - `24h` → `Idle(86400s)`.
///
/// Bare integers are treated as seconds (Ollama-compatible).
/// Returns `Err(reason)` for unparseable input.
///
/// # Errors
///
/// Returns `Err` when the spec is empty, contains a non-`s|m|h` suffix, or
/// the numeric part fails to parse.
pub fn parse_duration_spec(spec: &str) -> Result<KeepAlivePolicy, String> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return Err("empty duration spec".to_owned());
    }

    // Detect sign explicitly so we can return Pin for any negative shape.
    let (sign_negative, body) = if let Some(rest) = trimmed.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = trimmed.strip_prefix('+') {
        (false, rest)
    } else {
        (false, trimmed)
    };

    if body.is_empty() {
        return Err(format!("duration spec '{spec}' missing numeric body"));
    }

    // Split body into numeric prefix + optional unit suffix.
    let (num_str, unit) = match body.chars().last() {
        Some(c) if c.is_ascii_alphabetic() => {
            let split_at = body.len() - c.len_utf8();
            let (n, u) = body.split_at(split_at);
            (n, Some(u))
        }
        _ => (body, None),
    };

    if num_str.is_empty() {
        return Err(format!("duration spec '{spec}' missing numeric value"));
    }

    let n: u64 = num_str
        .parse::<u64>()
        .map_err(|e| format!("duration spec '{spec}' has invalid number '{num_str}': {e}"))?;

    if sign_negative {
        return Ok(KeepAlivePolicy::Pin);
    }
    if n == 0 {
        return Ok(KeepAlivePolicy::UnloadAfter);
    }

    let secs = match unit {
        None | Some("s") => n,
        Some("m") => n.saturating_mul(60),
        Some("h") => n.saturating_mul(3600),
        Some(other) => {
            return Err(format!(
                "duration spec '{spec}' has unknown unit '{other}' (expected s | m | h)"
            ));
        }
    };

    Ok(KeepAlivePolicy::Idle(Duration::from_secs(secs)))
}

/// Convenience: parse the `RMLX_KEEP_ALIVE` environment variable if set.
///
/// Returns `Ok(None)` if the env var is absent. Returns `Err` if the var
/// is set but unparseable — caller decides whether to fail-loud or fall
/// back to the next precedence level.
///
/// # Errors
///
/// Returns `Err` when the env var is set to an unparseable spec.
pub fn parse_env_keep_alive() -> Result<Option<KeepAlivePolicy>, String> {
    match std::env::var("RMLX_KEEP_ALIVE") {
        Ok(s) => parse_duration_spec(&s).map(Some),
        Err(_) => Ok(None),
    }
}

/// Convenience: map an `Option<i64>` per-request `keep_alive` field to a policy.
///
/// Ollama / LM Studio body field shape: `-1` = pin, `0` = unload-after, `N>0` = idle TTL seconds.
/// Returns `None` when the field is absent.
#[must_use]
pub fn policy_from_request_field(field: Option<i64>) -> Option<KeepAlivePolicy> {
    field.map(|n| match n {
        n if n < 0 => KeepAlivePolicy::Pin,
        0 => KeepAlivePolicy::UnloadAfter,
        // n > 0 here, so try_from is always Ok — fallback 0 is defensive.
        n => KeepAlivePolicy::Idle(Duration::from_secs(u64::try_from(n).unwrap_or(0))),
    })
}

// ── Decode lease ─────────────────────────────────────────────────────────────

/// Active-decode lease counter for a model slot.
///
/// Stored on [`crate::LoadedModel`] (one per resident slot). Incremented at
/// the start of every generation by acquiring a [`DecodeLeaseGuard`] (RAII)
/// and decremented on guard drop — including drops from streaming-stream
/// teardown, error paths, and task panics.
///
/// The unload timer checks `count() > 0` before tearing down the model;
/// when busy, the unload is suppressed and re-armed for one TTL period.
pub type DecodeLease = Arc<AtomicUsize>;

/// RAII guard that increments the decode lease on construction and
/// decrements on drop.
///
/// Cloning is not supported — each `acquire` returns one guard; pass it by
/// move into the streaming stream / blocking task so its drop signals
/// "decode done" exactly once.
#[derive(Debug)]
pub struct DecodeLeaseGuard {
    lease: DecodeLease,
}

impl DecodeLeaseGuard {
    /// Acquire a new lease — bumps the counter by 1.
    #[must_use]
    pub fn acquire(lease: DecodeLease) -> Self {
        lease.fetch_add(1, Ordering::AcqRel);
        Self { lease }
    }

    /// Current lease count (for tests + diagnostics).
    #[must_use]
    pub fn count(&self) -> usize {
        self.lease.load(Ordering::Acquire)
    }
}

impl Drop for DecodeLeaseGuard {
    fn drop(&mut self) {
        // AcqRel + fetch_sub: ensure prior writes happen-before any observer of
        // the decremented count (e.g. the unload timer about to fire).
        self.lease.fetch_sub(1, Ordering::AcqRel);
    }
}

// ── Guarded stream wrapper ───────────────────────────────────────────────────

/// Streaming-response wrapper that holds a [`DecodeLeaseGuard`] until
/// the inner stream is dropped.
///
/// SSE responses return an async stream whose lifetime extends past the
/// handler future. To keep the decode lease alive until the final byte has
/// been delivered (or the client disconnects and axum drops the body),
/// wrap the SSE stream in this struct and move the guard into it.
///
/// When the stream is dropped — either via normal end-of-stream or
/// connection close — `Self::drop` runs and releases the guard, allowing
/// the keep-alive timer to proceed with unload.
///
/// Requires `S: Stream + Unpin` — SSE streams used in the server are boxed
/// (`Box<dyn Stream + Send>`) and therefore Unpin.
#[allow(
    missing_debug_implementations,
    reason = "inner stream `S` is often a boxed trait object"
)]
pub struct GuardedStream<S> {
    inner: S,
    // Optional because callers pass `None` when the resident state has no
    // lease (e.g. test stubs without a slot). When `Some`, the guard's
    // drop fires when this struct drops with the stream.
    #[allow(dead_code, reason = "held purely for Drop side-effect")]
    lease_guard: Option<DecodeLeaseGuard>,
}

impl<S> GuardedStream<S> {
    /// Wrap `inner` with a lease guard. The guard drops when the stream is dropped.
    pub fn new(inner: S, guard: Option<DecodeLeaseGuard>) -> Self {
        Self {
            inner,
            lease_guard: guard,
        }
    }
}

impl<S: futures::Stream + Unpin> futures::Stream for GuardedStream<S> {
    type Item = S::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(test)]
#[path = "keep_alive_tests.rs"]
mod keep_alive_tests;
