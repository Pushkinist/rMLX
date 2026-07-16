//! Cross-layer KV sharing: what a producer layer hands to its consumers.
//!
//! Some architectures declare a **shared-KV topology**: a run of layers do not
//! project their own K/V but attend over the K/V a designated earlier
//! ("producer") layer accumulated. The producer holds the [`KvCache`]; the
//! consumers hold none.
//!
//! A consumer does not need bf16 *tensors* — it needs **access to the
//! producer's K/V**. The quantized store already provides that. Handing over a
//! dequantized tensor forces the producer to materialise bf16 K/V, which is
//! exactly what a fused flash-decode-over-quant kernel exists to avoid: it
//! would strand every fused codec on any shared-KV model.
//!
//! [`SharedKv`] therefore lets the **codec** decide which of the two it can
//! offer, and the consumer runs the matching attention either way. The model on
//! top never chooses — it only reports the sharing topology.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{scaled_dot_product_attention, Array, Device};

use crate::kvcache::KvCache;

/// The producer layer's K/V, as offered to shared-KV consumer layers.
///
/// Produced by [`KvCache::update_and_sdpa_shared_source`], consumed by
/// [`SharedKv::sdpa`]. Which variant a step yields is decided by the codec's
/// own decode dispatch, never by the architecture on top.
#[allow(missing_debug_implementations)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed protocol enum — the two ways a producer can offer its K/V. Deliberately \
              exhaustive so a third kind breaks compilation at every consumer match instead of \
              silently falling into a wildcard arm that reads the wrong K/V"
)]
pub enum SharedKv {
    /// The producer's own SDPA already materialised bf16 `(K, V)`, so consumers
    /// reuse those exact tensors — one materialisation, N consumers.
    ///
    /// This is what every codec without a live fused decode kernel yields, plus
    /// every prefill step and every rotating (SWA) layer.
    Bf16(Array, Array),
    /// The producer ran a fused decode kernel straight off its quant store and
    /// never materialised bf16 K/V. Consumers re-run the same fused kernel
    /// against that store via [`SharedKv::sdpa`].
    ///
    /// `kv_len` is the number of K/V positions the producer attended — the
    /// value a consumer must size its mask's key dim from, exactly as it would
    /// read `k.shape()[2]` off a [`SharedKv::Bf16`] tensor.
    Store {
        /// Accumulated K/V positions held by the producer's store.
        kv_len: i32,
    },
}

impl SharedKv {
    /// Number of K/V positions a consumer will attend over.
    ///
    /// A consumer's mask is built from the **producer's** length, never from
    /// the model-wide offset: the two can legitimately disagree after a
    /// speculative partial-accept rollback.
    pub fn kv_len(&self) -> Result<i32> {
        match self {
            Self::Bf16(k, _) => k.shape().get(2).copied().ok_or_else(|| {
                Error::Mlx(format!(
                    "SharedKv::Bf16: K shape {:?} has no seq axis",
                    k.shape()
                ))
            }),
            Self::Store { kv_len } => Ok(*kv_len),
        }
    }

    /// Run a consumer layer's SDPA over the producer's K/V.
    ///
    /// `producer` is the cache the K/V lives in. It is only read for
    /// [`Self::Store`]; pass `None` on cacheless forwards (which can only ever
    /// yield [`Self::Bf16`]).
    ///
    /// The `Store` arm re-enters the producer's own fused decode kernel through
    /// [`KvCache::sdpa_shared`] — no bf16 materialisation, no second append.
    pub fn sdpa(
        &self,
        producer: Option<&KvCache>,
        queries: &Array,
        scale: f32,
        mask_mode: &str,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Array> {
        match self {
            Self::Bf16(k, v) => {
                scaled_dot_product_attention(queries, k, v, scale, mask_mode, additive_mask, device)
            }
            Self::Store { kv_len } => {
                let cache = producer.ok_or_else(|| {
                    Error::Mlx(
                        "SharedKv::Store: no producer cache supplied — a store-backed share can \
                         only be produced by a cache-holding layer"
                            .to_owned(),
                    )
                })?;
                cache.sdpa_shared(queries, scale, additive_mask, *kv_len, device)
            }
        }
    }

    /// Materialise the producer's K/V as tensors.
    ///
    /// For callers that genuinely need *tensors* rather than attention output —
    /// e.g. handing a verifier's representative K/V to a separate drafter
    /// model. **Not** for the attention hot path: on a [`Self::Store`] share
    /// this pays the full-prefix dequant the fused kernel exists to avoid.
    ///
    /// The pair comes back at the activation-stream dtype (bf16 in production);
    /// K and V always agree, so a consumer's attention is never promoted to a
    /// wider dtype by one half of the pair. See
    /// [`KvCache::materialise_shared_kv`].
    pub fn materialise(
        &self,
        producer: Option<&KvCache>,
        device: Device,
    ) -> Result<(Array, Array)> {
        match self {
            Self::Bf16(k, v) => Ok((k.try_clone()?, v.try_clone()?)),
            Self::Store { kv_len } => {
                let cache = producer.ok_or_else(|| {
                    Error::Mlx(
                        "SharedKv::Store: no producer cache supplied — cannot materialise K/V \
                         without the store it lives in"
                            .to_owned(),
                    )
                })?;
                cache.materialise_shared_kv(*kv_len, device)
            }
        }
    }
}
