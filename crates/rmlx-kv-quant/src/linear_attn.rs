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
//! - [`GdnTape`] / [`GdnTapeSegment`] — the round tape a speculative loop arms
//!   so a partly-accepted round can be rolled back without a second forward.
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
    /// The recurrence inputs of the forwards taken since [`LinearAttnCache::arm_tape`],
    /// or `None` when nothing is recording. See [`GdnTape`].
    pub tape: Option<GdnTape>,
}

impl LinearAttnCache {
    /// Empty cache. Both fields `None` — first forward call initialises them.
    pub fn new() -> Self {
        Self {
            conv_state: None,
            delta_state: None,
            tape: None,
        }
    }

    /// Drop both states so the next forward call starts from zero.
    pub fn reset(&mut self) {
        self.conv_state = None;
        self.delta_state = None;
        self.tape = None;
    }

    /// Start recording a round tape, discarding any the previous round left.
    ///
    /// While a tape is armed every GDN forward through this cache appends its
    /// recurrence inputs to it, which is what makes [`GdnTape`] able to rebuild
    /// the state at an interior position. Arm it once per speculative round,
    /// before the forwards that round takes.
    pub fn arm_tape(&mut self) {
        self.tape = Some(GdnTape::new());
    }

    /// Stop recording and hand back what was recorded, if anything.
    pub fn take_tape(&mut self) -> Option<GdnTape> {
        self.tape.take()
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
    /// Returns 0 if neither state has been populated yet.
    ///
    /// An armed [`GdnTape`] counts: it holds real buffers for as long as a
    /// speculative round is open, and a total that ignored them would read low
    /// for exactly the rounds that allocate the most.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    pub fn resident_bytes(&self) -> u64 {
        let Self {
            conv_state,
            delta_state,
            tape,
        } = self;
        crate::bytes::opt_array_bytes(conv_state.as_ref())
            + crate::bytes::opt_array_bytes(delta_state.as_ref())
            + tape.as_ref().map_or(0, GdnTape::resident_bytes)
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

    /// Deep clone of the recurrent state (used by prompt cache).
    ///
    /// The clone carries no tape: a tape describes one speculative round of one
    /// live cache, and a stored prompt-cache entry is neither.
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
            tape: None,
        })
    }
}

impl Default for LinearAttnCache {
    fn default() -> Self {
        Self::new()
    }
}

// -- GdnTape (speculative-round rollback) ------------------------------------

/// One GDN forward's recurrence inputs, in the shapes the recurrence kernel
/// takes them.
///
/// Every field is a handle on an array the forward had already built, so
/// recording a segment copies nothing.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed record - the fields are exactly the recurrence kernel's inputs; adding one means the kernel takes another argument"
)]
#[allow(missing_debug_implementations)]
pub struct GdnTapeSegment {
    /// `[B, T, Hk, Dk]` - query, normalised and scaled.
    pub q: Array,
    /// `[B, T, Hk, Dk]` - key, normalised and scaled.
    pub k: Array,
    /// `[B, T, Hv, Dv]` - value.
    pub v: Array,
    /// `[B, T, Hv]` f32 - the per-position decay.
    pub g: Array,
    /// `[B, T, Hv]` - the per-position update weight.
    pub beta: Array,
    /// `[B, kernel - 1 + T, conv_dim]` - the depthwise-conv1d input this call
    /// ran over, the tail carried in from the previous call included.
    pub conv_input: Array,
    /// Positions this call consumed: `T`.
    pub len: usize,
}

impl GdnTapeSegment {
    /// Resident RAM this segment's handles keep alive, in bytes.
    ///
    /// The exhaustive destructure is the drift guard, as in
    /// [`LinearAttnCache::resident_bytes`].
    pub fn resident_bytes(&self) -> u64 {
        let Self {
            q,
            k,
            v,
            g,
            beta,
            conv_input,
            len: _,
        } = self;
        [q, k, v, g, beta, conv_input]
            .into_iter()
            .map(crate::bytes::array_bytes)
            .sum()
    }
}

/// The recurrence inputs of every GDN forward a speculative round took, plus
/// the state the round started from.
///
/// # Why a tape rather than a snapshot
///
/// The recurrent state has no sequence axis, so a partly-accepted round cannot
/// slice it back to the accepted position the way a K/V cache is truncated. The
/// rollback used to restore a pre-round snapshot and re-run the accepted prefix
/// through the whole layer stack - a second full read of the model's weights,
/// paid on every partial round.
///
/// The inputs the recurrence consumes are causal and per-position: the value
/// the forward computed at position `i` does not depend on any position after
/// it, so the ones a shorter forward would have produced are exactly the ones
/// this forward already produced. Keeping them lets the accepted prefix be
/// re-folded by the recurrence kernel alone, reading no weights at all.
///
/// # Accumulating across forwards
///
/// A round is not always one forward. A two-model loop's drafter takes one
/// forward per drafted token and rolls back across the lot of them, so segments
/// accumulate in call order and the prefix to refold may span several. The
/// pre-round state is recorded once, at the first segment.
#[allow(missing_debug_implementations)]
pub struct GdnTape {
    state_in: Option<Array>,
    segments: Vec<GdnTapeSegment>,
}

impl GdnTape {
    /// An armed, empty tape.
    pub fn new() -> Self {
        Self {
            state_in: None,
            segments: Vec::new(),
        }
    }

    /// Record one forward's recurrence inputs.
    ///
    /// `state_in` is the recurrent state that forward started from; it is kept
    /// from the first call only, because that is the state the round started
    /// from and every later segment starts where its predecessor ended.
    pub fn push(&mut self, state_in: &Array, segment: GdnTapeSegment) -> Result<()> {
        if self.state_in.is_none() {
            self.state_in = Some(state_in.try_clone()?);
        }
        self.segments.push(segment);
        Ok(())
    }

    /// Positions recorded, over every segment.
    pub fn positions(&self) -> usize {
        self.segments.iter().map(|s| s.len).sum()
    }

    /// The recurrent state the round started from, or `None` on an empty tape.
    pub fn state_in(&self) -> Option<&Array> {
        self.state_in.as_ref()
    }

    /// The recorded forwards, in the order they ran.
    pub fn segments(&self) -> &[GdnTapeSegment] {
        &self.segments
    }

    /// Resident RAM this tape keeps alive, in bytes.
    pub fn resident_bytes(&self) -> u64 {
        crate::bytes::opt_array_bytes(self.state_in.as_ref())
            + self
                .segments
                .iter()
                .map(GdnTapeSegment::resident_bytes)
                .sum::<u64>()
    }
}

impl Default for GdnTape {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "linear_attn_tests.rs"]
mod tests;
