//! SDPA dispatch paths: Mixed, RotKTq4V, standard, and K8V4 flash.
#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::too_many_lines
)]

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{
    add, dequantize, matmul, scaled_dot_product_attention, softmax_precise, Array, Device, Dtype,
};

use crate::mixed_quant::{mixed_quantized_sdpa, rot_k_tq4v_sdpa, MixedTuple};
use crate::planar_flash_decode_msl::{planar_flash_decode_enabled, planar_flash_decode_sdpa};
use crate::planar_fused_qk::planar_fused_qk_enabled;
use crate::planar_fused_qk_msl::planar_fused_qk;
use crate::storage::KvStorage;

use super::helpers::storage_variant_name;
use super::KvCache;

/// Shared-KV payload surfaced by [`KvCache::update_and_sdpa_returning_shared_kv`]
/// for cross-layer-KV architectures (Gemma4).
///
/// Most codecs (and every prefill step) surface dequantized **bf16** K/V — the
/// historic contract a consumer layer attends with a plain SDPA. The `Mixed`
/// codec on the GLOBAL / full-attention decode path instead surfaces the live
/// quant 3-tuples so the consumer can run a fused quantized SDPA directly over
/// the quant store — no full-length bf16 mirror is materialised, so the global
/// KV stays at quant-store size instead of re-inflating to bf16.
#[allow(
    clippy::exhaustive_enums,
    reason = "the Gemma4 arch wiring matches this exhaustively on purpose: \
              adding a shared-KV payload variant should force every consumer \
              dispatch to be updated rather than silently take a wildcard arm"
)]
pub enum SharedKvOut {
    /// Dequantized bf16 K/V — `(k_full, v_full)`. The legacy shared-KV contract.
    Bf16(Array, Array),
    /// Fused-quant decode tuples for the consumer's `mixed_quantized_sdpa`.
    MixedQuant {
        /// Quantized K (codes/scales/biases), full accumulated length.
        k: MixedTuple,
        /// Quantized V (codes/scales/biases), full accumulated length.
        v: MixedTuple,
        /// Optional K-rotation the consumer must apply to Q (RotK; `None` for
        /// plain Mixed).
        k_rotation: Option<Array>,
        /// K quantization bit-width.
        k_bits: i32,
        /// V quantization bit-width.
        v_bits: i32,
        /// K quantization group size.
        k_group_size: i32,
        /// V quantization group size.
        v_group_size: i32,
    },
}

impl std::fmt::Debug for SharedKvOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // mlx-rs `Array` carries no meaningful Debug; print only the variant
        // and the scalar quant params so logs stay readable.
        match self {
            Self::Bf16(_, _) => f.write_str("SharedKvOut::Bf16"),
            Self::MixedQuant {
                k_bits,
                v_bits,
                k_group_size,
                v_group_size,
                k_rotation,
                ..
            } => f
                .debug_struct("SharedKvOut::MixedQuant")
                .field("k_bits", k_bits)
                .field("v_bits", v_bits)
                .field("k_group_size", k_group_size)
                .field("v_group_size", v_group_size)
                .field("k_rotation", &k_rotation.is_some())
                .finish(),
        }
    }
}

impl KvCache {
    /// Mixed-precision one-shot append + quantized SDPA.
    ///
    /// Steps (mirror mlx-lm-turboquant's Qwen3 Attention forward):
    /// 1. Append `new_k`/`new_v` to the cache as `mx.quantize` 3-tuples
    /// 2. Build an additive mask if `seq > 1` (prefill / chunked prefill);
    /// 3. Run two `mx.quantized_matmul` + softmax inside
    ///
    /// Returns the SDPA output `[B, n_q_heads, L, D]`.
    ///
    /// `additive_mask` is the explicit additive mask used in the rMLX
    /// `"array"` mask mode (caller's responsibility to construct via
    /// `build_chunked_prefill_mask`). Pass `None` for `mask_mode == ""` (decode)
    /// or `mask_mode == "causal"` and rely on the caller to substitute a
    /// causal additive mask for `seq > 1` prefill.
    pub fn update_and_sdpa_mixed(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Array> {
        let (out, _) = self.update_and_sdpa_mixed_inner(
            queries,
            new_k,
            new_v,
            scale,
            additive_mask,
            false,
            false,
            device,
        )?;
        Ok(out)
    }

    /// Inner Mixed-precision path shared by [`KvCache::update_and_sdpa_mixed`]
    /// (the non-shared-KV hot path) and [`KvCache::update_and_sdpa_returning_kv`]
    /// (cross-layer-KV archs such as Gemma4).
    ///
    /// `want_kv` controls whether the accumulated **bf16** K/V is surfaced for a
    /// shared-KV consumer (dequant-before-share). When `false` the cache
    /// behaves exactly as before — no extra bf16 accumulator maintenance — so
    /// the Bonsai / Qwen3 Mixed decode hot path is unchanged. When `true`:
    /// - **prefill** returns the fp16 prefill-raw accumulator (already
    ///   materialised by `update_prefill_raw`);
    /// - **decode** mirrors `new_k`/`new_v` into `decode_fp16_k/v` (same
    ///   mechanism K8V4-flash uses, `update_decode_fp16`) and returns the full
    ///   bf16 prefix. These are the SAME values the fused quantized SDPA was
    ///   computed from (a dequant of the identical fp16 tokens), so the
    ///   downstream sharing layer sees consistent KV.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    fn update_and_sdpa_mixed_inner(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        want_kv: bool,
        surface_quant: bool,
        device: Device,
    ) -> Result<(Array, Option<SharedKvOut>)> {
        let (k_bits, v_bits, k_group_size, v_group_size) =
            self.quant.mixed_params().ok_or_else(|| {
                Error::Mlx("update_and_sdpa_mixed called on non-Mixed-path cache".into())
            })?;

        let new_seq = new_k.shape()[2];
        let prev_offset = self.offset;
        self.offset = prev_offset + new_seq;

        // Mixed prefill path: during prefill accumulate fp16 K/V via
        // update_prefill_raw (same buffer used by K8V4/K8V8 prefill), then
        // dispatch the fast fp16 SDPA. Bulk quantization to Mixed happens in
        // exit_prefill. This avoids per-chunk mx.quantize calls during the
        // 3770-token prefill, bringing cold TTFT in line with k8v4.
        if self.in_prefill {
            let (k_full, v_full) = self.update_prefill_raw(new_k, new_v, device)?;
            // mask_mode: "" for single-token decode (never here), "causal" for
            // first prefill chunk (prev_offset == 0), "array" for chunked
            // continuations. The caller already built the additive mask for
            // "array" mode; for "causal" we pass None and let MLX build the
            // causal mask internally.
            let mask_mode = if additive_mask.is_some() {
                "array"
            } else if new_seq > 1 {
                "causal"
            } else {
                ""
            };
            let out = scaled_dot_product_attention(
                queries,
                &k_full,
                &v_full,
                scale,
                mask_mode,
                additive_mask,
                device,
            )?;
            // the prefill-raw bf16 buffers ARE the accumulator a shared-KV
            // consumer needs — return them directly (no extra work). The quant
            // store is not built until `exit_prefill`, so even on the
            // `surface_quant` path the consumer attends bf16 during prefill;
            // the fused-quant decode path only kicks in post-prefill (decode
            // branch below), which is where the O(ctx) inflation lives.
            let kv = if want_kv {
                Some(SharedKvOut::Bf16(k_full, v_full))
            } else {
                None
            };
            return Ok((out, kv));
        }

        let state = match &mut self.storage {
            KvStorage::Mixed { state, .. } => state,
            _ => {
                return Err(Error::KvStorageMismatch {
                    expected: "Mixed",
                    got: storage_variant_name(&self.storage),
                })
            }
        };

        let (k_tuple, v_tuple) = state.update_and_fetch(new_k, new_v, device)?;

        // pass the K-rotation (if any) so SDPA pre-rotates Q. `k_tuple` /
        // `v_tuple` are owned, so the mutable borrow of `state` above has ended;
        // re-borrow `k_rotation` immutably here.
        let k_rotation = state
            .k_rotation
            .as_ref()
            .map(Array::try_clone)
            .transpose()?;

        let out = mixed_quantized_sdpa(
            queries,
            &k_tuple,
            &v_tuple,
            scale,
            additive_mask,
            k_group_size,
            k_bits,
            v_group_size,
            v_bits,
            k_rotation.as_ref(),
            device,
        )?;

        // Surface KV for cross-layer-KV consumers (Gemma4). Two modes:
        //
        // * `surface_quant` — hand the consumer the live quant 3-tuples
        //   (`k_tuple`/`v_tuple`, already the full accumulated length) so it can
        //   run a fused quantized SDPA directly. NO bf16 mirror is built, so the
        //   global KV stays at quant-store size (the whole point — drop the
        //   O(ctx) bf16 re-inflation). The consumer applies `k_rotation` to Q
        //   exactly as `mixed_quantized_sdpa` does here.
        //
        // * bf16 (legacy) — maintain the full-length bf16 accumulator via
        //   `update_decode_fp16` and surface that. `self.offset` was already
        //   advanced to `prev_offset + new_seq` above, so the new token is
        //   sliced in at `[prev_offset:offset]` — identical bookkeeping to K8V4.
        //
        // The non-shared Mixed decode hot path (Bonsai / Qwen3, want_kv=false)
        // takes neither branch and pays nothing.
        let kv = if want_kv && surface_quant {
            Some(SharedKvOut::MixedQuant {
                k: k_tuple.try_clone()?,
                v: v_tuple.try_clone()?,
                k_rotation,
                k_bits,
                v_bits,
                k_group_size,
                v_group_size,
            })
        } else if want_kv {
            let max_seq = match &self.storage {
                KvStorage::Mixed { max_seq, .. } => *max_seq,
                _ => {
                    return Err(Error::KvStorageMismatch {
                        expected: "Mixed",
                        got: storage_variant_name(&self.storage),
                    })
                }
            };
            let (k_full, v_full) = self.update_decode_fp16(new_k, new_v, max_seq, device)?;
            Some(SharedKvOut::Bf16(k_full, v_full))
        } else {
            None
        };
        Ok((out, kv))
    }

    /// RotKTq4V hybrid — one-shot K update (rotated affine) + V update (tq4) + SDPA.
    ///
    /// During **prefill** the same `update_prefill_raw` → `exit_prefill` path as
    /// other quantized modes is used; this method is only called at **decode** time
    /// (offset > 0, `in_prefill` false).
    ///
    /// Steps:
    /// 1. Bump `self.offset += new_seq`.
    /// 2. During prefill: accumulate via `update_prefill_raw` + fp16 SDPA.
    /// 3. At decode: append K via `MixedKvState::update_k_and_fetch`, append V via `update_v`.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn update_and_sdpa_rot_k_tq4v(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Array> {
        let new_seq = new_k.shape()[2];
        let prev_offset = self.offset;
        self.offset = prev_offset + new_seq;

        // Prefill: accumulate raw fp16 and run fast SDPA (same as Mixed Part-B).
        // Bulk quantize into RotKTq4V storage happens in exit_prefill.
        if self.in_prefill {
            let (k_full, v_full) = self.update_prefill_raw(new_k, new_v, device)?;
            let mask_mode = if additive_mask.is_some() {
                "array"
            } else if new_seq > 1 {
                "causal"
            } else {
                ""
            };
            return scaled_dot_product_attention(
                queries,
                &k_full,
                &v_full,
                scale,
                mask_mode,
                additive_mask,
                device,
            );
        }

        // Decode: append K to MixedKvState, append V to QuantV, then SDPA.
        let KvStorage::RotKTq4V { k_state, v, .. } = &mut self.storage else {
            // -review HIGH 2: typed error instead of unreachable!() panic.
            let variant = storage_variant_name(&self.storage);
            return Err(Error::Mlx(format!(
                "RotKTq4V dispatch: storage variant mismatch (quant=RotKTq4V, \
                 storage={variant}); cache may have been hydrated from incompatible spill"
            )));
        };

        // Append and fetch K (rotated + affine quantized 3-tuple).
        let k_tuple = k_state.update_k_and_fetch(new_k, device)?;

        // Borrow k_rotation for Q pre-rotation (after mutable borrow of k_state ends).
        let k_rotation = k_state
            .k_rotation
            .as_ref()
            .map(Array::try_clone)
            .transpose()?;

        // Append V token to QuantV and dequantize the full prefix to bf16.
        let v_shape = new_v.shape();
        // For GPU path QuantV::append uses the Array directly (v_f32 is unused).
        let v_f32: Vec<f32> = if device == Device::Gpu {
            Vec::new()
        } else {
            // CPU path: convert bf16 → f32 for scalar turboquant encoder.
            let bytes = new_v.to_bytes()?;
            bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits(u32::from(bits) << 16)
                })
                .collect()
        };

        let qv = v.as_mut().ok_or_else(|| {
            Error::Mlx("RotKTq4V: V buffer not initialised (exit_prefill not called?)".into())
        })?;
        // -review HIGH 1: use append_uncapped — RotKTq4V decode tokens
        // accumulate past max_seq after a full-prefill bulk-init.
        qv.append_uncapped(&v_f32, &v_shape, new_v, device)?;

        // Dequantize the full V prefix.
        let v_bf16 = {
            let (v_cpu, v_gpu) = qv.dequantize_choice(device, new_v.dtype())?;
            if let Some(arr) = v_gpu {
                arr
            } else {
                // CPU path: rebuild bf16 from f32 vec and shape.
                let bytes: Vec<u8> = v_cpu
                    .iter()
                    .flat_map(|&f| {
                        let bits = (f.to_bits() >> 16) as u16;
                        bits.to_le_bytes()
                    })
                    .collect();
                Array::from_bytes(&bytes, &qv.shape, Dtype::Bf16)
                    .map_err(|e| Error::Mlx(e.to_string()))?
            }
        };

        rot_k_tq4v_sdpa(
            queries,
            &k_tuple,
            &v_bf16,
            scale,
            additive_mask,
            64, // k_group_size fixed for RotKTq4V
            8,  // k_bits fixed for RotKTq4V
            k_rotation.as_ref(),
            device,
        )
    }

    /// -review CRITICAL 1: `update_and_sdpa_returning_kv` variant for
    /// RotKTq4V. Runs the same K/V update and SDPA as
    /// `update_and_sdpa_rot_k_tq4v` but additionally surfaces the dequantized
    /// bf16 `(K, V)` so Gemma4 shared-KV consumer layers can receive them.
    ///
    /// During prefill: returns the raw fp16 accumulator K/V (same as the Mixed
    /// path) — no quantize/dequantize round-trip on the prefill chunk.
    /// During decode: dequantizes K from the `MixedKvState` 3-tuple and V from
    /// `QuantV`, then runs SDPA.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn update_and_sdpa_rot_k_tq4v_returning_kv(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        mask_mode: &str,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<(Array, Array, Array)> {
        let new_seq = new_k.shape()[2];
        let prev_offset = self.offset;
        self.offset = prev_offset + new_seq;

        // Prefill: accumulate raw fp16 and run fast SDPA — identical to the
        // `update_and_sdpa_rot_k_tq4v` prefill path. The caller receives the
        // raw fp16 K/V (no quant round-trip on prefill chunks).
        if self.in_prefill {
            let (k_full, v_full) = self.update_prefill_raw(new_k, new_v, device)?;
            let out = scaled_dot_product_attention(
                queries,
                &k_full,
                &v_full,
                scale,
                mask_mode,
                additive_mask,
                device,
            )?;
            return Ok((out, k_full, v_full));
        }

        // Decode: append K to MixedKvState, append V to QuantV, run SDPA,
        // and also return the dequantized bf16 K/V for shared-KV consumers.
        let KvStorage::RotKTq4V { k_state, v, .. } = &mut self.storage else {
            let variant = storage_variant_name(&self.storage);
            return Err(Error::Mlx(format!(
                "RotKTq4V dispatch: storage variant mismatch (quant=RotKTq4V, \
                 storage={variant}); cache may have been hydrated from incompatible spill"
            )));
        };

        // Append and fetch K 3-tuple.
        let k_tuple = k_state.update_k_and_fetch(new_k, device)?;
        let k_rotation = k_state
            .k_rotation
            .as_ref()
            .map(Array::try_clone)
            .transpose()?;
        let k_group_size = k_state.k_group_size;
        let k_bits = k_state.k_bits;

        // Append V and dequantize the full prefix.
        let v_shape = new_v.shape();
        let v_f32: Vec<f32> = if device == Device::Gpu {
            Vec::new()
        } else {
            let bytes = new_v.to_bytes()?;
            bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits(u32::from(bits) << 16)
                })
                .collect()
        };
        let qv = v.as_mut().ok_or_else(|| {
            Error::Mlx("RotKTq4V: V buffer not initialised (exit_prefill not called?)".into())
        })?;
        qv.append_uncapped(&v_f32, &v_shape, new_v, device)?;

        let v_bf16 = {
            let (v_cpu, v_gpu) = qv.dequantize_choice(device, new_v.dtype())?;
            if let Some(arr) = v_gpu {
                arr
            } else {
                let bytes: Vec<u8> = v_cpu
                    .iter()
                    .flat_map(|&f| {
                        let bits = (f.to_bits() >> 16) as u16;
                        bits.to_le_bytes()
                    })
                    .collect();
                Array::from_bytes(&bytes, &qv.shape, Dtype::Bf16)
                    .map_err(|e| Error::Mlx(e.to_string()))?
            }
        };

        // Dequantize K from the affine 3-tuple (K was stored rotated).
        let k_bf16 = dequantize(
            &k_tuple.codes,
            &k_tuple.scales,
            Some(&k_tuple.biases),
            k_group_size,
            k_bits,
            "affine",
            device,
        )?;

        // Run SDPA with pre-rotated Q (same as rot_k_tq4v_sdpa step 1-4,
        // but K is already dequantized above so we call SDPA directly).
        let queries_owned;
        let q_ref: &Array = match k_rotation.as_ref() {
            Some(r) => {
                let d = *queries.shape().last().unwrap_or(&0) as usize;
                let try_fused =
                    crate::rot_k_msl::rot_k_fused_enabled() && crate::rot_k_msl::is_supported_d(d);
                queries_owned = if try_fused {
                    match crate::rot_k_msl::rot_k_fwht_rotate_gpu(queries, device) {
                        Ok(q_rot) => q_rot,
                        Err(e) => {
                            tracing::warn!(
                                reason = %e,
                                "rot_k_fwht_rotate_gpu failed in returning_kv; \
                                 falling back to v1 matmul Q rotation"
                            );
                            crate::rot_k::rotate_last_axis(queries, r, device)?
                        }
                    }
                } else {
                    crate::rot_k::rotate_last_axis(queries, r, device)?
                };
                &queries_owned
            }
            None => queries,
        };

        let attn_mask_mode = if additive_mask.is_some() {
            "array"
        } else {
            "causal"
        };
        let out = scaled_dot_product_attention(
            q_ref,
            &k_bf16,
            &v_bf16,
            scale,
            attn_mask_mode,
            additive_mask,
            device,
        )?;

        Ok((out, k_bf16, v_bf16))
    }

    /// Universal dispatch wrapper. Replaces the three call patterns used today
    /// by attention layers (Mixed → `update_and_sdpa_mixed`; K8V4 TurboFlash →
    /// `sdpa_dispatch`; legacy → `update` + `scaled_dot_product_attention`).
    ///
    /// Dispatch order:
    /// 1. `KvQuant::Mixed` → delegate to [`KvCache::update_and_sdpa_mixed`].
    /// 2. Else try [`KvCache::sdpa_dispatch`]; `Some(out)` is the K8V4
    /// TurboFlash result, `None` means fall through.
    /// 3. Legacy fallback: [`KvCache::update`] consumes `new_k`/`new_v` and
    /// returns the accumulated `(k_full, v_full)`; those — NOT the
    /// per-step `new_k`/`new_v` — are fed to
    /// [`scaled_dot_product_attention`].
    ///
    /// `mask_mode` is consumed only by the legacy fallback branch. The Mixed
    /// and K8V4-flash paths build their own causal masks internally and ignore
    /// the wrapper's `mask_mode`.
    ///
    /// Task 9: wrapped with a debug-level span so per-layer per-token timing
    /// is visible at `--log debug`. Tensor args skipped to avoid log bloat.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        level = "debug",
        name = "update_and_sdpa",
        skip(self, queries, new_k, new_v, additive_mask, mask_mode, scale, device),
        fields(
            kv_quant = %self.quant,
            offset = self.offset,
            path = tracing::field::Empty,
        )
    )]
    pub fn update_and_sdpa(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        mask_mode: &str,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Array> {
        // SWA (rotating) layers stay bf16 for EVERY quant. The Mixed
        // short-circuit below uses `update_prefill_raw` (full uncapped K),
        // which disagrees with the ring-capped SWA attention mask on the
        // window-crossing chunk. Route rotating caches through the legacy
        // `update()` path (ring first) so the K shape matches the mask — same
        // fix as `update_and_sdpa_returning_kv`. See `with_quant_max_seq_window`.
        if self.rotating.is_some() {
            tracing::Span::current().record("path", "rotating");
            let (k_full, v_full) = self.update(new_k, new_v, device)?;
            let out = scaled_dot_product_attention(
                queries,
                &k_full,
                &v_full,
                scale,
                mask_mode,
                additive_mask,
                device,
            )?;
            tracing::trace!(
                kv_bytes = self.approx_bytes(),
                offset = self.offset,
                "kv cache bytes"
            );
            return Ok(out);
        }

        // 1a. : RotKTq4V hybrid path (rotated K affine + tq4 V).
        if self.quant.uses_rot_k_tq4v_path() {
            tracing::Span::current().record("path", "rot_k_tq4v");
            let out = self.update_and_sdpa_rot_k_tq4v(
                queries,
                new_k,
                new_v,
                scale,
                additive_mask,
                device,
            )?;
            tracing::trace!(
                kv_bytes = self.approx_bytes(),
                offset = self.offset,
                "kv cache bytes"
            );
            return Ok(out);
        }

        // 1. Mixed-precision quantized SDPA path (Mixed + RotK).
        if self.quant.uses_mixed_path() {
            tracing::Span::current().record("path", "mixed");
            let out =
                self.update_and_sdpa_mixed(queries, new_k, new_v, scale, additive_mask, device)?;
            // Emit kv_bytes event for multi-turn growth tracking.
            // Demoted to trace! (per-layer per-step; opt in via --log verbose).
            tracing::trace!(
                kv_bytes = self.approx_bytes(),
                offset = self.offset,
                "kv cache bytes"
            );
            return Ok(out);
        }

        // 1b. PlanarK fused-QK fast path.  When K is stored as
        // PlanarQuant-packed (`KvStorage::PlanarK`) and the CLI toggle is on
        // (default), compute pre-softmax QK from the packed K directly via
        // the `planar_fused_qk` MSL kernel — skipping the full K dequant
        // write+read round-trip that dominates decode-step bandwidth.
        //
        // The decode-only gate (q_seq == 1) is checked HERE — not inside the
        // helper — because the helper mutates cache state (`ks.append`,
        // `update_decode_fp16`) before it knows whether it can complete the
        // SDPA, and the legacy fallback would then double-append.
        //
        // Warm-TTFT gate: every other quant (K8V4/K8V8/Planar/Mixed/
        // K8VTurbo*/Iso*/Rotor*/TurboSym*) uses the bf16 K seed materialised in
        // `decode_fp16_k` by `exit_prefill` for the entire post-prefill decode
        // window — `update_<arch>` checks `self.decode_fp16_k.is_some()` and
        // shortcuts to `update_decode_fp16`. Without this gate PlanarK was the
        // SOLE codec that re-encoded K through the lossy 4-bit Lloyd-Max +
        // Givens rotation kernel on every decode step, while K8V4 etc. silently
        // stayed in bf16. On Bonsai NIAH 8k d50 that asymmetry caused
        // `niah_pflash_bonsai_8k_d50` to fail retrieval (decoded text =
        // `"9. The secret. The grass…"`) while `niah_bonsai_8k_d50` (K8V4 storage,
        // bf16-warm K) passes. Honour the same warm-TTFT contract here: when the
        // bf16 K seed is live, fall through to the legacy bf16 SDPA path. The
        // explicit `--planar-fused-qk on` CLI override still reaches the fused
        // kernel via the next branch's fall-through (the K seed is only set on
        // the first `KvCache` of a request; later requests with a fresh cache
        // hit `decode_fp16_k.is_none()` and the fused path fires).
        //
        // See `docs/reports/planar-chunked-prefill-fix.md` § "Followups"
        // for the open question of whether warm-TTFT-as-default is the intended
        // steady-state design for the entire quantised-KV surface.
        if matches!(self.storage, KvStorage::PlanarK { .. })
            && planar_fused_qk_enabled()
            && device == Device::Gpu
            && queries.shape().get(2).copied().unwrap_or(0) == 1
        {
            if self.decode_fp16_k.is_some() {
                // Seed is live: PlanarK fused-QK / flash-decode kernels stay
                // dormant. Operator can distinguish this from a gate miss by
                // searching for this event in the JSONL log.
                tracing::debug!(
                    target: "rmlx_kv_quant::warm_ttft",
                    path = "warm_ttft_bypass",
                    codec = "PlanarK",
                    offset = self.offset,
                    "PlanarK SDPA dispatcher skipping fused-QK / \
                     flash-decode kernels — bf16 K seed is live (warm-TTFT)"
                );
                // Fall through to legacy below.
            } else if let Some(out) = self.update_and_sdpa_planar_k_fused(
                queries,
                new_k,
                new_v,
                scale,
                additive_mask,
                device,
            )? {
                tracing::Span::current().record("path", "planar_k_fused");
                tracing::trace!(
                    kv_bytes = self.approx_bytes(),
                    offset = self.offset,
                    "kv cache bytes"
                );
                return Ok(out);
                // Falls through to legacy when fused path is not yet eligible
                // (e.g. prefill chunk before any GPU buffer exists, or CPU device).
            }
        }

        // 2. K8V4 TurboFlash path (returns None when not eligible).
        if let Some(out) =
            self.sdpa_dispatch(queries, new_k, new_v, scale, additive_mask, device)?
        {
            tracing::Span::current().record("path", "flash");
            // Demoted to trace! (per-layer per-step; opt in via --log verbose).
            tracing::trace!(
                kv_bytes = self.approx_bytes(),
                offset = self.offset,
                "kv cache bytes"
            );
            return Ok(out);
        }

        // 2b. Head-major fused-QK shadow path (q8/turbo3/turbo4 today;
        // iso/rotor are HOLD until GPU encoders ship). Decode-only,
        // gated by `RMLX_FUSED_QK=1`. Returns `None` to fall through to the
        // legacy bf16 SDPA path when any gate is off, when the codec has
        // no GPU encoder yet, or when the bf16 mirror is not yet seeded
        // (cold-prefill window).
        if let Some(out) =
            self.try_fused_qk_dispatch(queries, new_k, new_v, scale, additive_mask, device)?
        {
            tracing::Span::current().record("path", "fused_qk");
            tracing::trace!(
                kv_bytes = self.approx_bytes(),
                offset = self.offset,
                "kv cache bytes"
            );
            return Ok(out);
        }

        // 3. Legacy fallback: accumulated K/V from update(), then SDPA. Passing
        // `new_k`/`new_v` here instead of the accumulated buffers would
        // silently break chunked prefill and every decode step after the
        // first.
        tracing::Span::current().record("path", "legacy");
        let (k_full, v_full) = self.update(new_k, new_v, device)?;
        let out = scaled_dot_product_attention(
            queries,
            &k_full,
            &v_full,
            scale,
            mask_mode,
            additive_mask,
            device,
        )?;
        // Demoted to trace! (per-layer per-step; opt in via --log verbose).
        tracing::trace!(
            kv_bytes = self.approx_bytes(),
            offset = self.offset,
            "kv cache bytes"
        );
        Ok(out)
    }

    /// Sibling of [`KvCache::update_and_sdpa`] for archs with **cross-layer KV
    /// sharing** (e.g. Gemma3 / Gemma4). Returns the accumulated post-update
    /// `(K, V)` alongside the SDPA output so the caller can hand the K/V
    /// downstream to shared-KV consumer layers.
    ///
    /// For [`KvQuant::Mixed`] the fused quantized SDPA stores K/V as
    /// quant 3-tuples, so this method routes through
    /// [`KvCache::update_and_sdpa_mixed_inner`] with `want_kv = true`, which
    /// surfaces the accumulated **bf16** K/V (the prefill-raw buffer during
    /// prefill, the maintained `decode_fp16_k/v` during decode) — the same
    /// values the quantized SDPA was computed from. All other quants (`None`,
    /// `K8V4`, `K8V8`, `Planar`) route through [`KvCache::update`], which
    /// dequantises and returns the accumulated bf16/fp32 K/V the wrapper needs.
    ///
    /// The SWA `rotating.is_some()` short-circuit inside [`KvCache::update`]
    /// still applies — SWA layers stay bf16 even when `self.quant` requests a
    /// non-rotation codec. No SWA-specific branching is needed here.
    ///
    /// Task 9: wrapped with a debug-level span (sibling of `update_and_sdpa`).
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        level = "debug",
        name = "update_and_sdpa_returning_kv",
        skip(self, queries, new_k, new_v, additive_mask, mask_mode, scale, device),
        fields(
            kv_quant = %self.quant,
            offset = self.offset,
            path = "returning_kv_legacy",
        )
    )]
    pub fn update_and_sdpa_returning_kv(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        mask_mode: &str,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<(Array, Array, Array)> {
        // SWA (rotating) layers stay bf16 for EVERY quant, including
        // Mixed/RotK (mlx-lm `RotatingKVCache.to_quantized` raises
        // NotImplementedError; see `with_quant_max_seq_window`). The
        // `uses_mixed_path()` short-circuit below routes through
        // `update_prefill_raw` (full uncapped K) during chunked prefill, but
        // the Gemma4 attention mask for an SWA layer is sized to the ring's
        // window-capped K (`offset.min(window-1) + seq`). On the chunk that
        // crosses the window the two disagree (full K = offset+seq vs capped
        // mask), producing `add: [broadcast_shapes] (…,512,1024) and
        // (…,512,1023)`. Routing rotating caches through `update()` (which
        // honors the ring first) keeps Mixed consistent with None/K8V8/K8V4/
        // Planar, all of which already reach the ring via `update()`.
        if self.rotating.is_some() {
            let (k_full, v_full) = self.update(new_k, new_v, device)?;
            let out = scaled_dot_product_attention(
                queries,
                &k_full,
                &v_full,
                scale,
                mask_mode,
                additive_mask,
                device,
            )?;
            return Ok((out, k_full, v_full));
        }

        // Mixed dequant-before-share. The fused quantized SDPA stores K/V
        // as quant 3-tuples, but `update_and_sdpa_mixed_inner(want_kv=true)`
        // surfaces the accumulated bf16 K/V (prefill-raw buffer during prefill;
        // maintained `decode_fp16_k/v` during decode) — the same values the
        // SDPA was computed from. Cross-layer-KV archs (Gemma4) hand those
        // bf16 tensors to shared-KV consumer layers. RotK rides the
        // same path; the shared-KV consumer sees unrotated bf16 K (the SDPA
        // helper rotates Q internally, never the surfaced K).
        if self.quant.uses_mixed_path() {
            let (out, kv) = self.update_and_sdpa_mixed_inner(
                queries,
                new_k,
                new_v,
                scale,
                additive_mask,
                true,
                false,
                device,
            )?;
            let (k_full, v_full) = match kv {
                Some(SharedKvOut::Bf16(k, v)) => (k, v),
                _ => {
                    return Err(Error::Mlx(
                        "update_and_sdpa_returning_kv: Mixed path returned no bf16 K/V \
                         (want_kv=true, surface_quant=false must surface the bf16 accumulator)"
                            .into(),
                    ))
                }
            };
            return Ok((out, k_full, v_full));
        }

        // / -review CRITICAL 1: RotKTq4V must be explicitly routed
        // here before the `update()` fallthrough. `update()` returns an error for
        // RotKTq4V ("Contract violation"), crashing Gemma4 shared-KV layers that
        // call this method. Mirror the Mixed branch: run the hybrid SDPA and
        // surface the dequantized bf16 K/V for downstream shared-KV consumers.
        if self.quant.uses_rot_k_tq4v_path() {
            let (out, k_full, v_full) = self.update_and_sdpa_rot_k_tq4v_returning_kv(
                queries,
                new_k,
                new_v,
                scale,
                mask_mode,
                additive_mask,
                device,
            )?;
            return Ok((out, k_full, v_full));
        }

        // TurboFlash + fused-QK dispatch hooks for cross-layer-KV producer
        // layers (Gemma4). The dispatch chain mirrors the one in
        // `update_and_sdpa`; we surface bf16 (K, V) by slicing the
        // `decode_fp16_k/v` mirrors that those dispatch paths already
        // maintain. Returns `None` when no dispatch arm is eligible — the
        // legacy bf16 fallback below then runs unchanged.
        if let Some((out, k_full, v_full)) =
            self.try_dispatch_returning_kv(queries, new_k, new_v, scale, additive_mask, device)?
        {
            tracing::Span::current().record("path", "returning_kv_dispatch");
            return Ok((out, k_full, v_full));
        }

        let (k_full, v_full) = self.update(new_k, new_v, device)?;
        let out = scaled_dot_product_attention(
            queries,
            &k_full,
            &v_full,
            scale,
            mask_mode,
            additive_mask,
            device,
        )?;
        Ok((out, k_full, v_full))
    }

    /// Cross-layer-KV producer SDPA that can surface the **quant store** for the
    /// consumer instead of a bf16 mirror.
    ///
    /// This is the fused-quant-decode variant of
    /// [`KvCache::update_and_sdpa_returning_kv`]. On the `Mixed` codec DECODE
    /// path it surfaces [`SharedKvOut::MixedQuant`] — the live quant 3-tuples —
    /// so the shared-KV consumer runs a fused quantized SDPA directly and NO
    /// full-length bf16 mirror is materialised (the global KV stays at
    /// quant-store size). Every other path (SWA rotating, prefill, non-Mixed
    /// codecs, the dispatch chain, the legacy `update()` fallthrough) surfaces
    /// [`SharedKvOut::Bf16`] — identical to `update_and_sdpa_returning_kv`.
    ///
    /// The legacy `update_and_sdpa_returning_kv` is left byte-for-byte
    /// unchanged for callers that do not opt into the fused-quant-decode share.
    #[allow(clippy::too_many_arguments)]
    pub fn update_and_sdpa_returning_shared_kv(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        mask_mode: &str,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<(Array, SharedKvOut)> {
        // Mixed (and RotK) DECODE: surface the quant tuples. SWA rotating layers
        // never reach here as Mixed (they are forced bf16 by the rotating
        // short-circuit inside `update_and_sdpa_mixed_inner`'s prefill branch
        // and by `with_quant_max_seq_window`); the `surface_quant` mode only
        // takes effect on the decode (post-prefill) branch, so prefill still
        // surfaces bf16.
        if self.quant.uses_mixed_path() && self.rotating.is_none() {
            let (out, kv) = self.update_and_sdpa_mixed_inner(
                queries,
                new_k,
                new_v,
                scale,
                additive_mask,
                true,
                true,
                device,
            )?;
            let kv = kv.ok_or_else(|| {
                Error::Mlx(
                    "update_and_sdpa_returning_shared_kv: Mixed path returned no shared KV \
                     (want_kv=true must always surface a payload)"
                        .into(),
                )
            })?;
            return Ok((out, kv));
        }

        // Everything else: delegate to the bf16-surfacing legacy path and wrap.
        let (out, k_full, v_full) = self.update_and_sdpa_returning_kv(
            queries,
            new_k,
            new_v,
            scale,
            mask_mode,
            additive_mask,
            device,
        )?;
        Ok((out, SharedKvOut::Bf16(k_full, v_full)))
    }

    /// Run the TurboFlash / fused-QK dispatch chain in the cross-layer-KV
    /// (`update_and_sdpa_returning_kv`) context and surface the bf16 `(K, V)`
    /// consumed by shared-KV consumer layers.
    ///
    /// The chain mirrors `update_and_sdpa`:
    ///
    /// 1. TurboFlash K8V4 (`sdpa_dispatch_no_lock` — forces lock-on OFF so
    ///    the bf16 mirror stays current; see
    ///    [`Self::update_and_sdpa_k8v4_flash_no_lock`]).
    /// 2. Head-major fused-QK (`try_fused_qk_dispatch` — already maintains
    ///    the bf16 mirror unconditionally via `update_decode_fp16`).
    ///
    /// On a successful dispatch the bf16 K/V is sliced from
    /// `decode_fp16_k/v` to `[0:self.offset]` — those are the exact bf16
    /// tensors the dispatch kernels read from / quantised from, so the
    /// downstream consumer sees consistent KV.
    ///
    /// Returns `Ok(None)` to fall through to the legacy path when no arm
    /// fires (e.g. codec not eligible, kv_seq below threshold, device CPU,
    /// gates off). This preserves Gemma4's default (gates OFF) behaviour
    /// bit-for-bit.
    ///
    /// General mechanism: nothing here is Gemma4-specific. Any future
    /// architecture that uses `update_and_sdpa_returning_kv` (medgemma,
    /// Laguna, etc.) reaches the same dispatch chain.
    #[allow(clippy::too_many_arguments)]
    fn try_dispatch_returning_kv(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Option<(Array, Array, Array)>> {
        // TurboFlash K8V4 — lock-on disabled so the bf16 mirror stays current.
        if let Some(out) =
            self.sdpa_dispatch_no_lock(queries, new_k, new_v, scale, additive_mask, device)?
        {
            tracing::debug!(
                target: "rmlx_kv_quant::returning_kv",
                kernel = "turbo_flash",
                offset = self.offset,
                "TurboFlash dispatched on returning_kv path"
            );
            let (k_full, v_full) = self.slice_decode_fp16_for_consumer(new_k, new_v, device)?;
            return Ok(Some((out, k_full, v_full)));
        }

        // Head-major fused-QK shadow (q8/turbo3/turbo4 codecs today;
        // iso/rotor HOLD pending GPU encoders).
        if let Some(out) =
            self.try_fused_qk_dispatch(queries, new_k, new_v, scale, additive_mask, device)?
        {
            tracing::debug!(
                target: "rmlx_kv_quant::returning_kv",
                kernel = "fused_qk",
                offset = self.offset,
                "fused-QK dispatched on returning_kv path"
            );
            let (k_full, v_full) = self.slice_decode_fp16_for_consumer(new_k, new_v, device)?;
            return Ok(Some((out, k_full, v_full)));
        }

        Ok(None)
    }

    /// Slice the bf16 K/V mirror to the current `self.offset` so a
    /// shared-KV consumer layer sees the same bf16 prefix the dispatch
    /// kernels operated on.
    ///
    /// Assumes the caller has just successfully dispatched a kernel from
    /// `try_dispatch_returning_kv`. Both dispatch arms maintain
    /// `decode_fp16_k` / `decode_fp16_v` (TurboFlash via `update_decode_fp16`
    /// at `update_and_sdpa_k8v4_flash_inner`; fused-QK via
    /// `update_decode_fp16` inside `try_fused_qk_dispatch`).
    ///
    /// `new_k` / `new_v` are the step tensors passed to the caller; their
    /// B, kv_h, and head_dim axes are validated against the bf16 mirror to
    /// catch layout regressions before slicing.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by the rank-4 check immediately above"
    )]
    fn slice_decode_fp16_for_consumer(
        &self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let k_buf = self.decode_fp16_k.as_ref().ok_or_else(|| {
            Error::Mlx("decode_fp16_k missing after dispatch — internal invariant violated".into())
        })?;
        let v_buf = self.decode_fp16_v.as_ref().ok_or_else(|| {
            Error::Mlx("decode_fp16_v missing after dispatch — internal invariant violated".into())
        })?;
        let k_shape = k_buf.shape();
        let v_shape = v_buf.shape();
        if k_shape.len() != 4 || v_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "decode_fp16_k/v rank != 4 (k={k_shape:?}, v={v_shape:?})"
            )));
        }
        // Validate B, kv_h, head_dim of the incoming step tensors against the
        // bf16 mirror. A future regression that swaps the mirror storage layout
        // could silently corrupt consumer-layer K/V if we skip these checks.
        let nk_shape = new_k.shape();
        let nv_shape = new_v.shape();
        if nk_shape.len() == 4 {
            if k_shape[0] != nk_shape[0] {
                return Err(Error::Mlx(format!(
                    "decode_fp16_k B axis mismatch: mirror={}, new_k={}",
                    k_shape[0], nk_shape[0]
                )));
            }
            if k_shape[1] != nk_shape[1] {
                return Err(Error::Mlx(format!(
                    "decode_fp16_k kv_h axis mismatch: mirror={}, new_k={}",
                    k_shape[1], nk_shape[1]
                )));
            }
            if k_shape[3] != nk_shape[3] {
                return Err(Error::Mlx(format!(
                    "decode_fp16_k head_dim axis mismatch: mirror={}, new_k={}",
                    k_shape[3], nk_shape[3]
                )));
            }
        }
        if nv_shape.len() == 4 {
            if v_shape[0] != nv_shape[0] {
                return Err(Error::Mlx(format!(
                    "decode_fp16_v B axis mismatch: mirror={}, new_v={}",
                    v_shape[0], nv_shape[0]
                )));
            }
            if v_shape[1] != nv_shape[1] {
                return Err(Error::Mlx(format!(
                    "decode_fp16_v kv_h axis mismatch: mirror={}, new_v={}",
                    v_shape[1], nv_shape[1]
                )));
            }
            if v_shape[3] != nv_shape[3] {
                return Err(Error::Mlx(format!(
                    "decode_fp16_v head_dim axis mismatch: mirror={}, new_v={}",
                    v_shape[3], nv_shape[3]
                )));
            }
        }
        let kv_seq = self.offset;
        if kv_seq > k_shape[2] || kv_seq > v_shape[2] {
            return Err(Error::Mlx(format!(
                "offset {} exceeds bf16 mirror length (k_seq={}, v_seq={})",
                kv_seq, k_shape[2], v_shape[2]
            )));
        }
        let strides = [1_i32; 4];
        let k_stop = [k_shape[0], k_shape[1], kv_seq, k_shape[3]];
        let v_stop = [v_shape[0], v_shape[1], kv_seq, v_shape[3]];
        let zero = [0_i32; 4];
        let k_full = k_buf.slice(&zero, &k_stop, &strides, device)?;
        let v_full = v_buf.slice(&zero, &v_stop, &strides, device)?;
        Ok((k_full, v_full))
    }

    /// Fused-QK + manual SV dispatch for `KvStorage::PlanarK`.
    ///
    /// Returns `Some(out)` when the fused path runs successfully, `None` when
    /// the caller should fall through to the legacy dequant + SDPA path
    /// (e.g. GPU buffers not yet populated — first prefill chunk).
    ///
    /// Steps:
    ///   1. Append `new_k` into the `QuantPlanarK` buffer (no dequant).
    ///   2. Append `new_v` into the bf16 decode-fp16 accumulator (V is bf16).
    ///   3. Slice GPU packed K (codes / scales / rot32) to current `S`.
    ///   4. Run [`planar_fused_qk`] → scores `[B, n_q_heads, 1, S]`.
    ///   5. Add the additive mask (if any), softmax (precise), reshape probs
    ///      for GQA broadcast, matmul with V, reshape back.
    ///
    /// The K dequant is skipped — this is the bandwidth win.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "all index uses validated by the shape contracts at function entry"
    )]
    fn update_and_sdpa_planar_k_fused(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Option<Array>> {
        use crate::storage::QuantPlanarK;

        // Extract & validate storage variant.
        let KvStorage::PlanarK { k, max_seq } = &mut self.storage else {
            return Ok(None);
        };
        let max_seq = *max_seq;

        // Append K (packed) — same as the legacy update_planar_k path but
        // SKIPPING the K dequant (the bandwidth win).
        let new_shape = new_k.shape();
        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantPlanarK::new(init_shape, max_seq));
        }
        let Some(ks) = k.as_mut() else {
            return Ok(None);
        };
        // GPU-only fused path; CPU appends would need a separate dequant path.
        let k_f32_empty: Vec<f32> = Vec::new();
        ks.append(&k_f32_empty, &new_shape, new_k, device, max_seq)?;

        // GPU packed view — `None` means buffers not populated (CPU path or
        // very first call corner cases); fall through to legacy.
        let Some((codes_arr, scales_arr, rot32_arr)) = ks.gpu_packed_view(device)? else {
            return Ok(None);
        };
        let k_shape = ks.shape.clone();
        // k_shape = [B, kv_h, S, head_dim].
        if k_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "planar_k_fused: K shape rank != 4, got {k_shape:?}"
            )));
        }
        let b = k_shape[0];
        let kv_h = k_shape[1];
        let kv_seq = k_shape[2];
        let head_dim = k_shape[3];

        // CRITICAL: advance `self.offset` BEFORE calling
        // `update_decode_fp16_v_only`.  The V-only helper (and the full
        // `update_decode_fp16`) compute the write window as
        // `prev_offset = self.offset - new_seq` and `new_offset = self.offset`.
        // If `self.offset` is not pre-incremented, every decode step writes V
        // at `[offset-1, offset)` — overwriting the LAST written position
        // instead of appending — and the V accumulator ends up permanently
        // one-step lagged.  All other `update_and_sdpa_*` variants
        // (`update_and_sdpa_mixed_inner` line 99-100,
        // `update_and_sdpa_rot_k_tq4v` line 223-224) advance offset before
        // calling the bf16 helpers; `KvCache::update` itself advances offset
        // at line 48 before dispatching variant updates.  The original
        // fused-QK landing missed this step.
        let new_seq = new_k.shape()[2];
        self.offset += new_seq;

        // Append V to bf16 buffer via the V-only helper (must NOT touch
        // `decode_fp16_k`).  The fused K path doesn't need the dequant K
        // shadow, and allocating it via the full `update_decode_fp16` would
        // waste an O(seq) bf16 K buffer every decode step.  Mirrors the
        // pattern from `update_iso_k_only_3`/`_4`.
        let v_full = self.update_decode_fp16_v_only(new_v, max_seq, device)?;

        // Q shape: [B, n_q_heads, q_seq, head_dim].  GQA: heads_per_kv =
        // n_q_heads / kv_h (must divide evenly).
        let q_shape = queries.shape();
        if q_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "planar_k_fused: Q shape rank != 4, got {q_shape:?}"
            )));
        }
        let q_seq = q_shape[2];
        let n_q_heads = q_shape[1];
        // Fused-QK kernel is decode-step only.  The call-site
        // already gates q_seq==1 BEFORE any cache mutation; this is a
        // defence-in-depth assertion.
        if q_seq != 1 {
            return Err(Error::Mlx(format!(
                "planar_k_fused: q_seq must be 1 (decode-only), got {q_seq}"
            )));
        }
        if n_q_heads % kv_h != 0 {
            return Err(Error::Mlx(format!(
                "planar_k_fused: n_q_heads={n_q_heads} not divisible by kv_h={kv_h}"
            )));
        }
        let heads_per_kv = n_q_heads / kv_h;

        // ── Planar flash decode (single-pass fused QK + softmax + SV) ──
        // When `RMLX_PLANAR_FLASH_DECODE=1` is set (or the CLI flag resolves
        // to On / Auto-on), bypass the fused-QK chain entirely — the flash
        // kernel does scores + softmax + SV in two Metal dispatches and emits
        // the final output directly. Head_dim must be a power of two (the
        // kernel tree-reduction relies on it); non-pow-2 dims fall through
        // to the fused-QK chain.
        if planar_flash_decode_enabled() && (head_dim as u32).is_power_of_two() {
            tracing::Span::current().record("path", "planar_flash_decode");
            let flash_out = planar_flash_decode_sdpa(
                queries,
                &codes_arr,
                &scales_arr,
                &rot32_arr,
                &v_full,
                additive_mask,
                b,
                kv_h,
                kv_seq,
                head_dim,
                heads_per_kv,
                4,
                scale,
                device,
            )?;
            let out = if flash_out.dtype() == queries.dtype() {
                flash_out
            } else {
                flash_out.astype(queries.dtype(), device)?
            };
            return Ok(Some(out));
        }

        // Pre-softmax scores via fused QK kernel — bits=4 (K-side PlanarK is
        // currently 4-bit only; the 3-bit kernel exists for V-axis / future K).
        let scores = planar_fused_qk(
            queries,
            &codes_arr,
            &scales_arr,
            &rot32_arr,
            b,
            kv_h,
            kv_seq,
            head_dim,
            heads_per_kv,
            4,
            scale,
            device,
        )?;

        // Add the additive mask if present, then softmax along the K-seq axis.
        let scores_masked = match additive_mask {
            Some(m) => add(&scores, m, device)?,
            None => scores,
        };
        let probs = softmax_precise(&scores_masked, -1, device)?;

        // SV path: probs `[B, n_q_heads, 1, kv_seq]` × V `[B, kv_h, kv_seq, head_dim]`.
        // GQA broadcast: reshape probs to `[B, kv_h, heads_per_kv, 1, kv_seq]`
        // and V to `[B, kv_h, 1, kv_seq, head_dim]`, then matmul along the
        // last two dims (the kv_seq contraction), then reshape back.
        let probs_g = probs.reshape(&[b, kv_h, heads_per_kv, 1, kv_seq], device)?;
        let v_g = v_full.reshape(&[b, kv_h, 1, kv_seq, head_dim], device)?;
        let out_g = matmul(&probs_g, &v_g, device)?;
        let out = out_g.reshape(&[b, n_q_heads, 1, head_dim], device)?;
        // Match the legacy SDPA output dtype (bf16/f16 callers); softmax may
        // have promoted to f32.
        let out = if out.dtype() == queries.dtype() {
            out
        } else {
            out.astype(queries.dtype(), device)?
        };
        Ok(Some(out))
    }
}
