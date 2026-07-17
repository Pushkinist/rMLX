//! Recurrent state cache for GatedDeltaNet (linear-attention) layers.
//!
//! [`LinearAttnCache`] mirrors mlx-lm's `ArraysCache(size=2)`. Unlike
//! [`super::kvcache::KvCache`], which holds K/V tensors that grow along the
//! sequence axis, this cache holds two fixed-shape recurrent states that are
//! **replaced** on every decode step:
//!
//! - `conv_state` — `[B, kernel_size - 1, conv_dim]` — the depthwise-conv1d
//!   tail, carried forward so streaming conv output matches full-sequence output.
//! - `delta_state` — `[B, Hv, Dv, Dk]` f32 — the GatedDeltaNet recurrent state.
//!
//! # Public API
//!
//! - [`LinearAttnCache`] — the recurrent state holder.
//!
//! # See also
//!
//! - [`super::kvcache::KvCache`] — standard KV cache for full-attention layers.

use rmlx_core::error::Result;
use rmlx_mlx::Array;

// ── LinearAttnCache (GatedDeltaNet recurrent state) ──────────────────────────
//
// Mirrors mlx-lm's `ArraysCache(size=2)` for linear-attention layers. Unlike
// `KvCache` (which holds K/V tensors growing along the sequence axis), this
// cache holds two fixed-shape recurrent states that are *replaced* every step:
//
// - `conv_state` shape `[B, kernel_size - 1, conv_dim]`
// The trailing `(kernel-1)` tokens of the depthwise-conv1d input — used
// instead of zero-padding on the next call so the streaming conv1d output
// matches the full-sequence conv output token-for-token.
//
// - `delta_state` shape `[B, Hv, Dv, Dk]` f32
// The recurrent delta state at the end of the last call. Carried forward
// so the per-step `gated_delta_ops` recurrence picks up where it left off.
//
// Both fields start `None`; `GatedDeltaNet::forward` initialises them from the
// model dtype on first use, then overwrites them every call. There is no
// quantization here — the state is small (one chunk per layer) and lives only
// during a single decode session.

/// Per-layer recurrent state for a `GatedDeltaNet` (linear-attention) block.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed cache struct — fields are the complete GatedDeltaNet recurrent-state contract; adding a field requires updating all GDN layer constructors and hydrate paths"
)]
#[allow(missing_debug_implementations)]
pub struct LinearAttnCache {
    /// `[B, kernel - 1, conv_dim]` — the depthwise-conv1d tail. `None` until
    /// the first prefill call, in which case the layer pads with zeros.
    pub conv_state: Option<Array>,
    /// `[B, Hv, Dv, Dk]` f32 — the delta-state at the end of the last call.
    /// `None` until the first prefill call, in which case the layer starts at
    /// zero.
    pub delta_state: Option<Array>,
}

impl LinearAttnCache {
    /// Empty cache. Both fields `None` — first forward call initialises them.
    pub fn new() -> Self {
        Self {
            conv_state: None,
            delta_state: None,
        }
    }

    /// Drop both states so the next forward call starts from zero.
    pub fn reset(&mut self) {
        self.conv_state = None;
        self.delta_state = None;
    }

    /// Conceptual "truncate to sequence position N" — NOT implementable by slicing.
    ///
    /// # Why truncate_to cannot work for LinearAttnCache
    ///
    /// `KvCache::truncate_to(n)` works because K/V tensors have an explicit
    /// sequence axis: positions `[0..n]` are simply sliced off and returned.
    ///
    /// `LinearAttnCache` has **no sequence axis**: `conv_state` and `delta_state`
    /// are the *compressed* recurrent state at the end of the most recent call.
    /// There is no way to recover the state at an earlier position `n` by slicing —
    /// the recurrence has consumed and discarded the intermediate states.
    ///
    /// The correct rollback mechanism is [`snapshot`] + [`restore_snapshot`]:
    /// take a deep clone of the state before the speculative draft round, then
    /// call `restore_snapshot` on partial rejection.
    ///
    /// This method exists so that callers can discover the semantic gap at
    /// compile time. It panics unconditionally. Do NOT call it on a live cache.
    ///
    /// # L36 context
    ///
    /// The spec decoder must call `snapshot()` before each draft round and
    /// `restore_snapshot()` on partial acceptance. See
    /// `docs/research/L36-spec-decoding-design.md` §1.3.
    #[allow(dead_code)]
    #[allow(
        clippy::panic,
        reason = "LinearAttnCache::truncate_to is structurally non-implementable: the GDN \
                  recurrent state has no sequence axis to truncate. Any caller must use \
                  snapshot()/restore_snapshot() instead; this panic surfaces misuse at \
                  call sites that have not yet been ported."
    )]
    pub fn truncate_to(&self, _n: i32) {
        panic!(
            "LinearAttnCache::truncate_to is not implementable: GDN recurrent state has no \
             sequence axis. Use snapshot() + restore_snapshot() for speculative decoding rollback. \
             See docs/research/L36-spec-decoding-design.md §1.3."
        );
    }

    /// Resident RAM held by this recurrent state, in bytes.
    ///
    /// Both `conv_state` and `delta_state` are fixed-shape tensors (independent
    /// of sequence length). Each buffer's size comes from its own shape ×
    /// dtype, so a state that is promoted to a wider dtype reports the truth
    /// with nothing to update here — this total feeds the same `kv_bytes` sum
    /// as the attention caches, and a hard-coded item size is how such a sum
    /// silently drifts away from the memory it claims to measure.
    ///
    /// Returns 0 if neither field has been populated yet.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    pub fn resident_bytes(&self) -> u64 {
        let Self {
            conv_state,
            delta_state,
        } = self;
        crate::bytes::opt_array_bytes(conv_state.as_ref())
            + crate::bytes::opt_array_bytes(delta_state.as_ref())
    }

    /// Snapshot the recurrent state at the start of a speculative round.
    ///
    /// Returns a deep clone of both `conv_state` and `delta_state` as they
    /// stand at the current end-of-sequence position. The caller must store
    /// this snapshot and call `restore_snapshot` if the spec round is partially
    /// rejected.
    ///
    /// # M23 / L36 design note
    ///
    /// `LinearAttnCache` holds fixed-shape recurrent state, not per-position
    /// tensors. There is no sequence-position axis to slice into — state at
    /// position N is produced by running the GDN recurrence from 0 to N.
    ///
    /// dflash's `RecurrentRollbackCache` solves this by recording a QKV/K/G
    /// *tape* during the spec draft round and then replaying the recurrence
    /// forward from the pre-round snapshot via a Metal `tape_replay_kernel`.
    /// Porting that approach to rMLX requires (a) making the GDN kernel
    /// callable from inside `LinearAttnCache` and (b) plumbing tape recording
    /// through the decode hot path — both are multi-day L36 work.
    ///
    /// For now (M23 audit result), spec decoding on Qwen3.5MoE GDN layers is
    /// deferred. Gemma4 (which has NO GDN layers) uses `KvCache::truncate_to`
    /// for spec rollback and is unaffected by this gap. This `snapshot` /
    /// `restore_snapshot` pair is the correct future interface; it will be
    /// wired through the speculative decoder in L36.
    pub fn snapshot(&self) -> Result<Self> {
        self.try_deep_clone()
    }

    /// Materialize this cache's GPU `Array` buffers on the calling (inference)
    /// thread so the SSD-spill drain thread can serialize them without a Metal
    /// stream. See `KvCache::eval_for_spill`.
    pub fn eval_for_spill(&self) -> Result<()> {
        if let Some(a) = &self.conv_state {
            a.eval()?;
        }
        if let Some(a) = &self.delta_state {
            a.eval()?;
        }
        Ok(())
    }

    /// Restore previously snapshotted recurrent state (see [`snapshot`]).
    ///
    /// Replaces both `conv_state` and `delta_state` with the values from
    /// `snap`. The snapshot is consumed.
    pub fn restore_snapshot(&mut self, snap: Self) {
        self.conv_state = snap.conv_state;
        self.delta_state = snap.delta_state;
    }

    /// Deep clone of the recurrent state (used by prompt cache).
    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(Self {
            conv_state: match &self.conv_state {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            delta_state: match &self.delta_state {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
        })
    }
}

impl Default for LinearAttnCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "linear_attn_tests.rs"]
mod tests;
