//! What every speculative round loop sets up the same way.

// kv-layer-quants: uniform — speculative scratch stack. The drafter/verifier
// caches a round builds live for that round only: they are never pushed to the
// prompt cache, never spilled, and never keyed by `layout_key`, so no on-disk
// description has to match them. Applying the boundary promotion here would
// change the codec of a stack whose only reader is the round that built it.

use rmlx_core::error::Result;
use rmlx_kv_quant::{KvCache, KvQuant};

use crate::arch::Architecture;

/// The KV codec, context ceiling and per-layer cache stack a round loop runs
/// its verifier on.
///
/// The codec is the request's override or the same constant the verifier's own
/// decode path resolves — a spec pair must not run two different caches — and
/// the ceiling comes from the verifier's context limits, so an over-capacity
/// `--max-ctx` is refused here rather than overflowing a cache mid-round. Each
/// layer's cache carries that layer's sliding window, the ceiling, and whether
/// the verifier stack's layers read each other's K/V, which is what decides
/// whether Mixed/RotK keep their bf16 mirror.
///
/// # Errors
///
/// [`rmlx_core::error::Error::ContextCeilingExceeded`] when `max_ctx_override`
/// is above the verifier's positional capacity.
pub(crate) fn verifier_cache_stack(
    verifier: &Architecture,
    kv_quant_override: Option<KvQuant>,
    max_ctx_override: Option<i32>,
) -> Result<(KvQuant, i32, Vec<KvCache>)> {
    let kv_quant = kv_quant_override.unwrap_or(crate::kv_cache::DEFAULT_KV_QUANT);
    let max_seq = super::verifier_context(verifier, max_ctx_override)?.ceiling;
    Ok((kv_quant, max_seq, cache_stack(verifier, kv_quant, max_seq)))
}

/// One model's per-layer cache stack for one speculative request.
///
/// The two-model loops build a second one of these for the draft model, at the
/// verifier's codec and the verifier's ceiling: the verifier owns the KV
/// geometry of a pair, so the draft's stack is the same stack against a
/// different layer count.
pub(crate) fn cache_stack(arch: &Architecture, kv_quant: KvQuant, max_seq: i32) -> Vec<KvCache> {
    (0..arch.num_hidden_layers())
        .map(|i| {
            let window = arch.layer_sliding_window(i);
            KvCache::with_quant_max_seq_window(kv_quant, max_seq, window)
                .with_max_seq_ceiling(max_seq)
                .with_layer_idx(i)
                .with_shares_kv(arch.shares_kv_across_layers())
        })
        .collect()
}
