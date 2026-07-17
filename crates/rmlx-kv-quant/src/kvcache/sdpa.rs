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

use crate::iso_flash_decode_msl::{iso_flash_decode_sdpa, ISO_FLASH_HEAD_DIM_MAX};
use crate::mixed_quant::{mixed_quantized_sdpa, rot_k_tq4v_sdpa};
use crate::planar_flash_decode_msl::{planar_flash_decode_enabled, planar_flash_decode_sdpa};
use crate::planar_fused_qk::planar_fused_qk_enabled;
use crate::planar_fused_qk_msl::planar_fused_qk;
use crate::rotor_flash_decode_msl::{rotor_flash_decode_sdpa, ROTOR_FLASH_HEAD_DIM_MAX};
use crate::storage::{KvStorage, ISO_K3_GROUP_SIZE};

use super::helpers::{f32_vec_to_array, storage_variant_name};
use super::shared_kv::SharedKv;
use super::KvCache;

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
            device,
        )?;
        Ok(out)
    }

    /// Inner Mixed-precision path shared by [`KvCache::update_and_sdpa_mixed`]
    /// (the non-shared-KV hot path) and [`KvCache::update_and_sdpa_shared_source`]
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
        device: Device,
    ) -> Result<(Array, Option<(Array, Array)>)> {
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
            // consumer needs — return them directly (no extra work).
            let kv = if want_kv {
                Some((k_full, v_full))
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

        // maintain the bf16 accumulator for cross-layer-KV consumers.
        // Only on the want_kv path so the non-shared Mixed decode hot path
        // (Bonsai / Qwen3) pays nothing. `self.offset` was already advanced to
        // `prev_offset + new_seq` above, so `update_decode_fp16` slices the new
        // token in at `[prev_offset:offset]` — identical bookkeeping to K8V4.
        let kv = if want_kv {
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
            Some((k_full, v_full))
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

    /// -review CRITICAL 1: `update_and_sdpa_shared_source` variant for
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
    fn update_and_sdpa_rot_k_tq4v_shared_source(
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
                                "rot_k_fwht_rotate_gpu failed on the shared-KV producer path; \
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
        // fix as `update_and_sdpa_shared_source`. See `with_quant_max_seq_window`.
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

        // 1c. Rotor K-only flash-decode fast path. Without it this codec
        // CPU-dequants the entire K prefix on every decode step
        // (`QuantRotorK{3,4}::dequant()` → `Vec<f32>` → re-upload), which is
        // O(seq) host work per token with the GPU idle. The kernel reads the
        // packed rotor store directly instead.
        //
        // The decode-only gate (q_seq == 1) is checked HERE, not in the helper:
        // the helper mutates cache state (store append, bf16 V accumulate)
        // before it knows whether it can complete the SDPA, so a late fallback
        // would double-append.
        //
        // QJL gate: the 1-bit QJL residual is a per-token back-projection
        // through a dense [head_dim, head_dim] matrix. Reproducing it in the
        // flash inner loop would cost more bandwidth than the kernel saves, so
        // a QJL-carrying store keeps the CPU dequant path. `--rotor-qjl off`
        // reaches the kernel. Same reasoning as the fused-QK shadow path.
        if matches!(
            self.storage,
            KvStorage::RotorKOnly3 { .. } | KvStorage::RotorKOnly4 { .. }
        ) && device == Device::Gpu
            && !rotor_k_store_uses_qjl(&self.storage)
            && queries.shape().get(2).copied().unwrap_or(0) == 1
        {
            if let Some(out) = self.update_and_sdpa_rotor_k_fused(
                queries,
                new_k,
                new_v,
                scale,
                additive_mask,
                device,
            )? {
                tracing::trace!(
                    kv_bytes = self.approx_bytes(),
                    offset = self.offset,
                    "kv cache bytes"
                );
                return Ok(out);
            }
        }

        // 1d. Iso K-only flash-decode fast path. Same shape as the rotor arm
        // above and for the same reason: without it this codec CPU-dequants the
        // entire K prefix on every decode step (`QuantIsoK{3,4}::dequant()` →
        // `Vec<f32>` → re-upload), which is O(seq) host work per token with the
        // GPU idle. The kernel reads the packed iso ring directly instead.
        //
        // Iso has no QJL sideband — that is a rotor-only residual — so there is
        // no equivalent gate here.
        //
        // The decode-only gate (q_seq == 1) is checked HERE, not in the helper:
        // the helper mutates cache state (store append, bf16 V accumulate)
        // before it knows whether it can complete the SDPA, so a late fallback
        // would double-append.
        if matches!(
            self.storage,
            KvStorage::IsoKOnly3 { .. } | KvStorage::IsoKOnly4 { .. }
        ) && device == Device::Gpu
            && queries.shape().get(2).copied().unwrap_or(0) == 1
        {
            if let Some(out) = self.update_and_sdpa_iso_k_fused(
                queries,
                new_k,
                new_v,
                scale,
                additive_mask,
                device,
            )? {
                tracing::trace!(
                    kv_bytes = self.approx_bytes(),
                    offset = self.offset,
                    "kv cache bytes"
                );
                return Ok(out);
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

        // 2b. Head-major fused-QK shadow path (q8/turbo3/turbo4/iso/rotor).
        // Decode-only, gated by `RMLX_FUSED_QK=1`. Returns `None` to fall
        // through to the legacy bf16 SDPA path when any gate is off, when the
        // codec has no GPU encoder, or when the bf16 mirror is not yet seeded
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
    /// sharing**: runs this producer layer's own update + SDPA and reports what
    /// its downstream consumer layers can attend over, as a [`SharedKv`].
    ///
    /// The dispatch chain mirrors [`KvCache::update_and_sdpa`] arm for arm —
    /// that symmetry is the contract. A codec that has a fused decode kernel
    /// must reach it here too; if it does not, the producer is pushed onto the
    /// legacy bf16 path and the fused kernel is silently dead for every model
    /// with a shared-KV topology.
    ///
    /// Which [`SharedKv`] variant a step yields is decided by the codec, not by
    /// the arch:
    ///
    /// * The fused-over-quant-store arms ([`Self::update_and_sdpa_planar_k_fused`],
    ///   [`Self::update_and_sdpa_rotor_k_fused`]) never materialise bf16 K/V, so
    ///   they yield [`SharedKv::Store`] and consumers re-enter the same kernel
    ///   via [`Self::sdpa_shared`].
    /// * Every other arm — rotating (SWA) rings, [`KvQuant::Mixed`] / RotK
    ///   (which surface the bf16 accumulator the quantized SDPA was computed
    ///   from), the TurboFlash / fused-QK dispatches (which maintain a bf16
    ///   mirror anyway), and the legacy [`KvCache::update`] fallthrough — has a
    ///   bf16 `(K, V)` in hand already and yields [`SharedKv::Bf16`], which all
    ///   consumers share without a second materialisation.
    ///
    /// The SWA `rotating.is_some()` short-circuit inside [`KvCache::update`]
    /// still applies — SWA layers stay bf16 even when `self.quant` requests a
    /// non-rotation codec. No SWA-specific branching is needed here.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        level = "debug",
        name = "update_and_sdpa_shared_source",
        skip(self, queries, new_k, new_v, additive_mask, mask_mode, scale, device),
        fields(
            kv_quant = %self.quant,
            offset = self.offset,
            path = tracing::field::Empty,
        )
    )]
    pub fn update_and_sdpa_shared_source(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        mask_mode: &str,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<(Array, SharedKv)> {
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
            return Ok((out, SharedKv::Bf16(k_full, v_full)));
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
            tracing::Span::current().record("path", "mixed");
            let (out, kv) = self.update_and_sdpa_mixed_inner(
                queries,
                new_k,
                new_v,
                scale,
                additive_mask,
                true,
                device,
            )?;
            let (k_full, v_full) = kv.ok_or_else(|| {
                Error::Mlx(
                    "update_and_sdpa_shared_source: Mixed path returned no bf16 K/V \
                     (want_kv=true must always surface the accumulator)"
                        .into(),
                )
            })?;
            return Ok((out, SharedKv::Bf16(k_full, v_full)));
        }

        // RotKTq4V must be explicitly routed here before the `update()`
        // fallthrough. `update()` returns an error for RotKTq4V ("Contract
        // violation"), crashing shared-KV layers that call this method. Mirror
        // the Mixed branch: run the hybrid SDPA and surface the dequantized
        // bf16 K/V for downstream shared-KV consumers.
        if self.quant.uses_rot_k_tq4v_path() {
            tracing::Span::current().record("path", "rot_k_tq4v");
            let (out, k_full, v_full) = self.update_and_sdpa_rot_k_tq4v_shared_source(
                queries,
                new_k,
                new_v,
                scale,
                mask_mode,
                additive_mask,
                device,
            )?;
            return Ok((out, SharedKv::Bf16(k_full, v_full)));
        }

        // Fused-over-quant-store arms. These read the packed store directly and
        // deliberately never materialise a bf16 K prefix, so the share they
        // hand downstream is the store itself. Same gates and same order as the
        // matching arms in `update_and_sdpa` — a consumer of this cache
        // re-enters the identical kernel through `sdpa_shared`.
        if let Some((out, kv_len)) =
            self.try_dispatch_shared_store(queries, new_k, new_v, scale, additive_mask, device)?
        {
            return Ok((out, SharedKv::Store { kv_len }));
        }

        // TurboFlash + fused-QK dispatch hooks. These maintain the bf16
        // `decode_fp16_k/v` mirrors regardless, so the share is those exact
        // tensors sliced to the current offset. Returns `None` when no arm is
        // eligible — the legacy bf16 fallback below then runs unchanged.
        if let Some((out, k_full, v_full)) =
            self.try_dispatch_shared_bf16(queries, new_k, new_v, scale, additive_mask, device)?
        {
            return Ok((out, SharedKv::Bf16(k_full, v_full)));
        }

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
        Ok((out, SharedKv::Bf16(k_full, v_full)))
    }

    /// Run the fused-over-quant-store dispatch arms for a cross-layer-KV
    /// producer layer, returning `(out, kv_len)` on a hit.
    ///
    /// Mirrors arms 1b / 1c of [`Self::update_and_sdpa`] — including their
    /// pre-mutation gates — so a shared-KV producer reaches exactly the kernels
    /// a non-sharing model reaches. `kv_len` is the store length the kernel
    /// attended, which downstream consumers size their mask from.
    ///
    /// Returns `Ok(None)` when no arm is eligible, leaving the cache untouched
    /// for the caller's next arm.
    ///
    /// General mechanism: nothing here is arch-specific. Every gate is keyed on
    /// the codec's storage variant and the step's shape.
    #[allow(clippy::too_many_arguments)]
    fn try_dispatch_shared_store(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Option<(Array, i32)>> {
        let q_seq_is_decode = queries.shape().get(2).copied().unwrap_or(0) == 1;

        // PlanarK fused-QK / flash-decode. The warm-TTFT gate mirrors
        // `update_and_sdpa`: while the bf16 K seed is live every other codec
        // decodes off it, so PlanarK must not be the sole codec re-encoding K
        // through its lossy kernel every step.
        if matches!(self.storage, KvStorage::PlanarK { .. })
            && planar_fused_qk_enabled()
            && device == Device::Gpu
            && q_seq_is_decode
        {
            if self.decode_fp16_k.is_some() {
                tracing::debug!(
                    target: "rmlx_kv_quant::warm_ttft",
                    path = "warm_ttft_bypass",
                    codec = "PlanarK",
                    offset = self.offset,
                    "PlanarK shared-source dispatcher skipping fused-QK / \
                     flash-decode kernels — bf16 K seed is live (warm-TTFT)"
                );
            } else if let Some(out) = self.update_and_sdpa_planar_k_fused(
                queries,
                new_k,
                new_v,
                scale,
                additive_mask,
                device,
            )? {
                tracing::Span::current().record("path", "planar_k_fused");
                tracing::debug!(
                    target: "rmlx_kv_quant::shared_kv",
                    kernel = "planar_k_fused",
                    offset = self.offset,
                    "fused kernel dispatched on the shared-KV producer path — \
                     consumers attend the quant store"
                );
                let kv_len = planar_k_accumulated_seq(&self.storage)?;
                return Ok(Some((out, kv_len)));
            }
        }

        // Rotor K-only flash-decode. Without it this codec CPU-dequants the
        // whole K prefix every decode step with the GPU idle.
        if matches!(
            self.storage,
            KvStorage::RotorKOnly3 { .. } | KvStorage::RotorKOnly4 { .. }
        ) && device == Device::Gpu
            && !rotor_k_store_uses_qjl(&self.storage)
            && q_seq_is_decode
        {
            if let Some(out) = self.update_and_sdpa_rotor_k_fused(
                queries,
                new_k,
                new_v,
                scale,
                additive_mask,
                device,
            )? {
                tracing::debug!(
                    target: "rmlx_kv_quant::shared_kv",
                    kernel = "rotor_flash",
                    offset = self.offset,
                    "fused kernel dispatched on the shared-KV producer path — \
                     consumers attend the quant store"
                );
                let kv_len = rotor_k_accumulated_seq(&self.storage)?;
                return Ok(Some((out, kv_len)));
            }
        }

        // Iso K-only flash-decode. Mirrors arm 1d of `update_and_sdpa`; without
        // it a shared-KV model (Gemma4) would push this codec onto the legacy
        // bf16 path and the kernel would be silently dead there.
        if matches!(
            self.storage,
            KvStorage::IsoKOnly3 { .. } | KvStorage::IsoKOnly4 { .. }
        ) && device == Device::Gpu
            && q_seq_is_decode
        {
            if let Some(out) = self.update_and_sdpa_iso_k_fused(
                queries,
                new_k,
                new_v,
                scale,
                additive_mask,
                device,
            )? {
                tracing::debug!(
                    target: "rmlx_kv_quant::shared_kv",
                    kernel = "iso_flash",
                    offset = self.offset,
                    "fused kernel dispatched on the shared-KV producer path — \
                     consumers attend the quant store"
                );
                let kv_len = iso_k_accumulated_seq(&self.storage)?;
                return Ok(Some((out, kv_len)));
            }
        }

        Ok(None)
    }

    /// Run the TurboFlash / fused-QK dispatch chain in the cross-layer-KV
    /// (`update_and_sdpa_shared_source`) context and surface the bf16 `(K, V)`
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
    /// General mechanism: nothing here is arch-specific. Any architecture that
    /// uses `update_and_sdpa_shared_source` reaches the same dispatch chain.
    #[allow(clippy::too_many_arguments)]
    fn try_dispatch_shared_bf16(
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
            tracing::Span::current().record("path", "flash");
            tracing::debug!(
                target: "rmlx_kv_quant::shared_kv",
                kernel = "turbo_flash",
                offset = self.offset,
                "TurboFlash dispatched on the shared-KV producer path"
            );
            let (k_full, v_full) = self.slice_decode_fp16_for_consumer(new_k, new_v, device)?;
            return Ok(Some((out, k_full, v_full)));
        }

        // Head-major fused-QK shadow (q8/turbo3/turbo4/iso/rotor codecs).
        if let Some(out) =
            self.try_fused_qk_dispatch(queries, new_k, new_v, scale, additive_mask, device)?
        {
            tracing::Span::current().record("path", "fused_qk");
            tracing::debug!(
                target: "rmlx_kv_quant::shared_kv",
                kernel = "fused_qk",
                offset = self.offset,
                "fused-QK dispatched on the shared-KV producer path"
            );
            let (k_full, v_full) = self.slice_decode_fp16_for_consumer(new_k, new_v, device)?;
            return Ok(Some((out, k_full, v_full)));
        }

        Ok(None)
    }

    /// Consumer-side SDPA over the K/V a **producer** layer accumulated in this
    /// cache, for a cross-layer-KV (shared-KV) topology.
    ///
    /// Read-only: no append, no offset advance. The producer already ran its
    /// own `update_and_sdpa_shared_source` for this step; this only re-enters
    /// the same fused kernel with the consumer's own `queries`.
    ///
    /// Only reachable when the producer reported [`SharedKv::Store`], i.e. the
    /// codec ran a fused-over-quant-store kernel. `kv_len` is the length the
    /// producer reported; it is checked against the store rather than trusted,
    /// so a producer/consumer desync errors instead of silently attending the
    /// wrong prefix.
    ///
    /// An unrecognised storage variant is an error, never a fallback: falling
    /// back would answer with a different codec's numbers under the same name.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the wildcard arm returns Err — it never selects a concrete codec, so a new \
                  storage variant fails loudly here instead of being silently decoded as another"
    )]
    pub fn sdpa_shared(
        &self,
        queries: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        kv_len: i32,
        device: Device,
    ) -> Result<Array> {
        match &self.storage {
            KvStorage::RotorKOnly3 { .. } | KvStorage::RotorKOnly4 { .. } => {
                let kv_seq = rotor_k_accumulated_seq(&self.storage)?;
                Self::check_shared_kv_len(kv_len, kv_seq)?;
                let v_full = self.slice_decode_fp16_v(kv_seq, device)?;
                self.rotor_k_flash_over_store(
                    queries,
                    &v_full,
                    scale,
                    additive_mask,
                    kv_seq,
                    device,
                )
            }
            KvStorage::IsoKOnly3 { .. } | KvStorage::IsoKOnly4 { .. } => {
                let kv_seq = iso_k_accumulated_seq(&self.storage)?;
                Self::check_shared_kv_len(kv_len, kv_seq)?;
                let v_full = self.slice_decode_fp16_v(kv_seq, device)?;
                self.iso_k_flash_over_store(queries, &v_full, scale, additive_mask, kv_seq, device)
            }
            KvStorage::PlanarK { .. } => {
                let kv_seq = planar_k_accumulated_seq(&self.storage)?;
                Self::check_shared_kv_len(kv_len, kv_seq)?;
                let v_full = self.slice_decode_fp16_v(kv_seq, device)?;
                self.planar_k_flash_over_store(
                    queries,
                    &v_full,
                    scale,
                    additive_mask,
                    kv_seq,
                    device,
                )
            }
            other => Err(Error::Mlx(format!(
                "sdpa_shared: storage variant {} has no fused-over-store consumer path — a \
                 producer must not report a store-backed share for it",
                storage_variant_name(other)
            ))),
        }
    }

    /// Materialise the `(K, V)` held by a store-backed share, as tensors.
    ///
    /// Cold path only — for callers that need K/V *tensors* rather than
    /// attention output (e.g. handing a verifier's representative K/V to a
    /// separate drafter model). This pays the full-prefix dequant the fused
    /// kernel exists to avoid, so never call it per decode step.
    ///
    /// # Dtype
    ///
    /// The pair comes back at the **V mirror's dtype** — the activation-stream
    /// dtype the model pushed into this cache (bf16 in production). K is cast to
    /// match it.
    ///
    /// That cast is load-bearing, in both directions:
    ///
    /// * The rotor stores dequantise through a `Vec<f32>`, so K arrives F32.
    ///   Handing an F32 K to a downstream model's attention alongside a bf16 V
    ///   promotes that model's whole stream to f32 and doubles its KV residency.
    /// * Hard-coding bf16 instead would be the mirror-image bug on a cache whose
    ///   stream is wider: K would be silently downcast away from what the cache
    ///   actually holds. Following V keeps the pair faithful to the stream.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the wildcard arm returns Err — it never selects a concrete codec, so a new \
                  storage variant fails loudly here instead of being silently decoded as another"
    )]
    pub fn materialise_shared_kv(&self, kv_len: i32, device: Device) -> Result<(Array, Array)> {
        let v_full = self.slice_decode_fp16_v(kv_len, device)?;
        let stream_dtype = v_full.dtype();
        let k_full = match &self.storage {
            KvStorage::RotorKOnly3 { k: Some(ks), .. } => {
                Self::check_shared_kv_len(kv_len, rotor_k_accumulated_seq(&self.storage)?)?;
                f32_vec_to_array(&ks.dequant()?, &ks.shape)?
            }
            KvStorage::RotorKOnly4 { k: Some(ks), .. } => {
                Self::check_shared_kv_len(kv_len, rotor_k_accumulated_seq(&self.storage)?)?;
                f32_vec_to_array(&ks.dequant()?, &ks.shape)?
            }
            KvStorage::IsoKOnly3 { k: Some(ks), .. } => {
                Self::check_shared_kv_len(kv_len, iso_k_accumulated_seq(&self.storage)?)?;
                f32_vec_to_array(&ks.dequant()?, &ks.shape)?
            }
            KvStorage::IsoKOnly4 { k: Some(ks), .. } => {
                Self::check_shared_kv_len(kv_len, iso_k_accumulated_seq(&self.storage)?)?;
                f32_vec_to_array(&ks.dequant()?, &ks.shape)?
            }
            KvStorage::PlanarK { k: Some(ks), .. } => {
                Self::check_shared_kv_len(kv_len, planar_k_accumulated_seq(&self.storage)?)?;
                let (k_recon, k_arr) = ks.dequantize_choice(device, stream_dtype)?;
                match k_arr {
                    Some(arr) => arr,
                    None => f32_vec_to_array(&k_recon, &ks.shape)?,
                }
            }
            other => {
                return Err(Error::Mlx(format!(
                    "materialise_shared_kv: storage variant {} holds no store-backed share",
                    storage_variant_name(other)
                )))
            }
        };
        let k_full = if k_full.dtype() == stream_dtype {
            k_full
        } else {
            k_full.astype(stream_dtype, device)?
        };
        // Defence-in-depth: K and V go to the same downstream attention, so a
        // dtype split between them would promote its stream to the wider of the
        // two. The cast above should make this unreachable.
        if k_full.dtype() != v_full.dtype() {
            return Err(Error::Mlx(format!(
                "materialise_shared_kv: K dtype {:?} != V dtype {:?} — a shared K/V pair \
                 must be one dtype or it promotes the consumer's attention stream",
                k_full.dtype(),
                v_full.dtype()
            )));
        }
        Ok((k_full, v_full))
    }

    /// Reject a producer/consumer length desync rather than attend the wrong
    /// prefix. The two are written and read one step apart on the same cache,
    /// so a mismatch is an internal invariant break.
    fn check_shared_kv_len(reported: i32, actual: i32) -> Result<()> {
        if reported == actual {
            Ok(())
        } else {
            Err(Error::Mlx(format!(
                "shared-KV length desync: producer reported kv_len={reported}, store holds \
                 {actual}"
            )))
        }
    }

    /// Read-only slice of the bf16 V accumulator to `kv_seq` positions.
    ///
    /// Matches what `update_decode_fp16_v_only` returns to the producer on the
    /// same step, so producer and consumer attend the identical V window.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by the rank-4 check immediately above"
    )]
    fn slice_decode_fp16_v(&self, kv_seq: i32, device: Device) -> Result<Array> {
        let v_buf = self.decode_fp16_v.as_ref().ok_or_else(|| {
            Error::Mlx(
                "decode_fp16_v missing on a store-backed share — the fused arms maintain it on \
                 every append, so this is an internal invariant violation"
                    .to_owned(),
            )
        })?;
        let v_shape = v_buf.shape();
        if v_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "decode_fp16_v rank != 4, got {v_shape:?}"
            )));
        }
        if kv_seq > v_shape[2] {
            return Err(Error::Mlx(format!(
                "shared-KV length {kv_seq} exceeds the bf16 V mirror (v_seq={})",
                v_shape[2]
            )));
        }
        v_buf.slice(
            &[0_i32; 4],
            &[v_shape[0], v_shape[1], kv_seq, v_shape[3]],
            &[1_i32; 4],
            device,
        )
    }

    /// Slice the bf16 K/V mirror to the current `self.offset` so a
    /// shared-KV consumer layer sees the same bf16 prefix the dispatch
    /// kernels operated on.
    ///
    /// Assumes the caller has just successfully dispatched a kernel from
    /// `try_dispatch_shared_bf16`. Both dispatch arms maintain
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

        if !matches!(self.storage, KvStorage::PlanarK { .. }) {
            return Ok(None);
        }
        // Provision this step before `max_seq` is read: it caps both the packed
        // K append and the bf16 V mirror below. Ahead of the first mutation, so
        // an over-ceiling request is a clean error rather than a torn cache.
        self.ensure_decode_capacity(self.offset + new_k.shape()[2])?;

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

        // GPU buffers not populated (CPU path or very first call corner cases):
        // fall through to legacy BEFORE any further mutation.
        if ks.gpu_packed_view(device)?.is_none() {
            return Ok(None);
        }
        let k_shape = ks.shape.clone();
        // k_shape = [B, kv_h, S, head_dim].
        if k_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "planar_k_fused: K shape rank != 4, got {k_shape:?}"
            )));
        }
        let kv_seq = k_shape[2];

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

        // Past this point the cache is already mutated (store appended, offset
        // advanced, bf16 V accumulated), so `Ok(None)` is no longer available —
        // it would send the caller into the legacy `update()` path, which
        // appends a second time. Every not-eligible condition is screened
        // above.
        let out =
            self.planar_k_flash_over_store(queries, &v_full, scale, additive_mask, kv_seq, device)?;
        Ok(Some(out))
    }

    /// Run the PlanarK fused-QK / flash-decode chain over the K already packed
    /// in this cache's store — no append, no offset advance.
    ///
    /// Split out of [`Self::update_and_sdpa_planar_k_fused`] so a shared-KV
    /// **consumer** layer can attend the producer's store through the exact
    /// same kernels the producer used, instead of the producer having to
    /// materialise bf16 K/V for it.
    ///
    /// `kv_seq` is the accumulated store length and `v_full` the matching bf16
    /// V prefix; both come from the caller so producer and consumer read the
    /// identical window.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "all index uses validated by the shape contracts at function entry"
    )]
    fn planar_k_flash_over_store(
        &self,
        queries: &Array,
        v_full: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        kv_seq: i32,
        device: Device,
    ) -> Result<Array> {
        let KvStorage::PlanarK { k, .. } = &self.storage else {
            return Err(Error::KvStorageMismatch {
                expected: "PlanarK",
                got: storage_variant_name(&self.storage),
            });
        };
        let Some(ks) = k.as_ref() else {
            return Err(Error::Mlx(
                "planar_k_fused: K store absent after a maintained append — internal invariant \
                 violated"
                    .to_owned(),
            ));
        };
        let Some((codes_arr, scales_arr, rot32_arr)) = ks.gpu_packed_view(device)? else {
            return Err(Error::Mlx(
                "planar_k_fused: GPU packed view absent after a maintained append — internal \
                 invariant violated"
                    .to_owned(),
            ));
        };
        let k_shape = &ks.shape;
        if k_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "planar_k_fused: K shape rank != 4, got {k_shape:?}"
            )));
        }
        // The store the packed view was written from is the one source of truth
        // for the attended length. A divergence would silently attend the wrong
        // prefix rather than error.
        Self::check_shared_kv_len(kv_seq, k_shape[2])?;
        let b = k_shape[0];
        let kv_h = k_shape[1];
        let head_dim = k_shape[3];

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
                v_full,
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
            return if flash_out.dtype() == queries.dtype() {
                Ok(flash_out)
            } else {
                flash_out.astype(queries.dtype(), device)
            };
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
        if out.dtype() == queries.dtype() {
            Ok(out)
        } else {
            out.astype(queries.dtype(), device)
        }
    }

    /// Flash-decode dispatch for the rotor K-only storage variants
    /// (`RotorKOnly3` / `RotorKOnly4`).
    ///
    /// Returns `Some(out)` when the fused kernel ran, `None` to fall through to
    /// the legacy `update()` + SDPA path (which CPU-dequants the whole K
    /// prefix every step).
    ///
    /// Steps:
    ///   1. Append `new_k` into the rotor store — GPU encode into the packed
    ///      ring, no dequant.
    ///   2. Append `new_v` into the bf16 accumulator (V is bf16 for this codec).
    ///   3. Take the ring's GPU packed view at the current `S`.
    ///   4. Run [`rotor_flash_decode_sdpa`] — QK over the packed K + online
    ///      softmax + bf16-V SV in two Metal dispatches.
    ///
    /// The full-prefix `QuantRotorK{3,4}::dequant()` is skipped — that is the
    /// entire point: it is O(seq) CPU work per decode step.
    ///
    /// Caller must have already gated `q_seq == 1` and `device == Gpu` BEFORE
    /// any cache mutation: this helper mutates cache state (append, bf16 V
    /// accumulate) before it can know whether the kernel is eligible, so a
    /// late fallback would double-append.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "all index uses validated by the shape contracts at function entry"
    )]
    fn update_and_sdpa_rotor_k_fused(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Option<Array>> {
        let new_shape = new_k.shape();
        if new_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "rotor_k_fused: new_k rank != 4, got {new_shape:?}"
            )));
        }
        let new_seq = new_shape[2];

        // Shape gates — checked BEFORE any mutation so a reject is a clean
        // fall-through to the legacy path.
        if !Self::rotor_flash_shape_ok(&new_shape) {
            return Ok(None);
        }

        // Append K into the rotor store (GPU encode → packed ring) and V into
        // the bf16 mirror. `update_rotor_k_only_{3,4}` also runs the O(seq)
        // dequant, so go through the storage append directly.
        let prev_seq = self.offset;
        // Not a rotor K-only cache: the caller's gate should have kept us out.
        // Fall through rather than mutate.
        if !matches!(
            self.storage,
            KvStorage::RotorKOnly3 { .. } | KvStorage::RotorKOnly4 { .. }
        ) {
            return Ok(None);
        }
        // Provision this step before the first mutation: both the packed ring
        // below and the bf16 V mirror further down are capped by the storage
        // `max_seq`, so it has to cover `prev_seq + new_seq` first.
        self.ensure_decode_capacity(prev_seq + new_seq)?;
        self.rotor_k_gpu_append(new_k, &new_shape, device)?;

        // CRITICAL: advance `self.offset` BEFORE `update_decode_fp16_v_only` —
        // the V-only helper computes its write window as
        // `[self.offset - new_seq, self.offset)`. Without the pre-increment
        // every decode step overwrites the last V position instead of
        // appending. Same ordering as `update_and_sdpa_planar_k_fused`.
        self.offset = prev_seq + new_seq;
        let max_seq = rotor_k_max_seq(&self.storage)?;
        let v_full = self.update_decode_fp16_v_only(new_v, max_seq, device)?;

        // Take `kv_seq` from the store the ring was written from, not from
        // `self.offset` — one source of truth. They must agree; a divergence
        // would silently attend over the wrong prefix length rather than error.
        // Same precedent as `update_and_sdpa_planar_k_fused` (`k_shape[2]`).
        let kv_seq = rotor_k_accumulated_seq(&self.storage)?;
        debug_assert_eq!(
            kv_seq, self.offset,
            "rotor_k_fused: store seq {kv_seq} != cache offset {} — the ring write \
             and the attention length disagree",
            self.offset
        );
        // Past this point the cache is already mutated (store appended, offset
        // advanced, bf16 V accumulated), so `Ok(None)` is NOT available: it
        // would send the caller into the legacy `update()` path, which appends
        // K/V a second time and advances `offset` again. Every not-eligible
        // condition is screened at the call-site gate and by
        // `rotor_flash_shape_ok` BEFORE any mutation; reaching here without a
        // ring means an internal invariant broke, so fail loudly.
        let out =
            self.rotor_k_flash_over_store(queries, &v_full, scale, additive_mask, kv_seq, device)?;
        Ok(Some(out))
    }

    /// Run the rotor flash-decode kernel over the K already packed in this
    /// cache's store — no append, no offset advance.
    ///
    /// Split out of [`Self::update_and_sdpa_rotor_k_fused`] so a shared-KV
    /// **consumer** layer can attend the producer's store through the exact
    /// same kernel the producer used, instead of the producer having to
    /// materialise bf16 K/V for it.
    ///
    /// `kv_seq` is the accumulated store length and `v_full` the matching bf16
    /// V prefix; both come from the caller so producer and consumer read the
    /// identical window.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "all index uses validated by the shape contracts at function entry"
    )]
    fn rotor_k_flash_over_store(
        &self,
        queries: &Array,
        v_full: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        kv_seq: i32,
        device: Device,
    ) -> Result<Array> {
        // Select the bit width from the live storage variant. No wildcard arm:
        // an unexpected variant must not be decoded with another codec's
        // kernel.
        let bits: u8 = if matches!(self.storage, KvStorage::RotorKOnly3 { .. }) {
            3
        } else if matches!(self.storage, KvStorage::RotorKOnly4 { .. }) {
            4
        } else {
            return Err(Error::KvStorageMismatch {
                expected: "RotorKOnly3 | RotorKOnly4",
                got: storage_variant_name(&self.storage),
            });
        };
        let Some((codes, scales, norms, rotors)) = self.rotor_k_packed_view(kv_seq, device)? else {
            return Err(Error::Mlx(format!(
                "rotor_k_fused: GPU ring absent after a maintained append \
                 (kv_seq={kv_seq}, bits={bits}) — internal invariant violated"
            )));
        };

        let v_shape = v_full.shape();
        if v_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "rotor_k_fused: V rank != 4, got {v_shape:?}"
            )));
        }
        let b = v_shape[0];
        let kv_h = v_shape[1];
        let head_dim = v_shape[3];

        let q_shape = queries.shape();
        if q_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "rotor_k_fused: Q shape rank != 4, got {q_shape:?}"
            )));
        }
        let q_seq = q_shape[2];
        let n_q_heads = q_shape[1];
        // Defence-in-depth: the call site already gates decode-only.
        if q_seq != 1 {
            return Err(Error::Mlx(format!(
                "rotor_k_fused: q_seq must be 1 (decode-only), got {q_seq}"
            )));
        }
        if n_q_heads % kv_h != 0 {
            return Err(Error::Mlx(format!(
                "rotor_k_fused: n_q_heads={n_q_heads} not divisible by kv_h={kv_h}"
            )));
        }
        let heads_per_kv = n_q_heads / kv_h;

        tracing::Span::current().record("path", "rotor_flash_decode");
        // Select the bit width explicitly. A wildcard arm here would map any
        // unexpected `bits` onto one width's kernel and decode the other's codes
        // at the wrong unpack stride — silently wrong K, no error. `bits` is a
        // plain integer, so no exhaustiveness lint would catch that.
        let flash_out = match bits {
            3 => rotor_flash_decode_sdpa::<3>(
                queries,
                &codes,
                &scales,
                &norms,
                &rotors,
                v_full,
                additive_mask,
                b,
                kv_h,
                kv_seq,
                head_dim,
                heads_per_kv,
                scale,
                device,
            )?,
            4 => rotor_flash_decode_sdpa::<4>(
                queries,
                &codes,
                &scales,
                &norms,
                &rotors,
                v_full,
                additive_mask,
                b,
                kv_h,
                kv_seq,
                head_dim,
                heads_per_kv,
                scale,
                device,
            )?,
            other => {
                return Err(Error::Quant(format!(
                    "rotor_k_fused: unsupported bits={other} (only 3 and 4); \
                     refusing to decode with another width's kernel"
                )))
            }
        };
        if flash_out.dtype() == queries.dtype() {
            Ok(flash_out)
        } else {
            flash_out.astype(queries.dtype(), device)
        }
    }

    /// Shape gates for the rotor flash-decode kernel, evaluated before any
    /// cache mutation.
    ///
    /// Keyed off shape only — never an arch name. `b == 1` because the packed
    /// ring's per-step stride does not interleave batch (one `KvCache` per
    /// request); a batched cache falls back rather than read a layout the
    /// kernel would misinterpret.
    fn rotor_flash_shape_ok(new_shape: &[i32]) -> bool {
        let (Some(&b), Some(&kv_h), Some(&head_dim)) =
            (new_shape.first(), new_shape.get(1), new_shape.get(3))
        else {
            return false;
        };
        if b != 1 || kv_h <= 0 {
            return false;
        }
        if head_dim <= 0 || head_dim > ROTOR_FLASH_HEAD_DIM_MAX {
            return false;
        }
        // Tree reduction over `head_dim` threads.
        (head_dim as u32).is_power_of_two()
    }

    /// GPU-append `new_k` into whichever rotor K-only store is active.
    fn rotor_k_gpu_append(
        &mut self,
        new_k: &Array,
        new_shape: &[i32],
        device: Device,
    ) -> Result<()> {
        if matches!(self.storage, KvStorage::RotorKOnly3 { .. }) {
            super::update::rotor3_k_only_gpu_append(self, new_k, new_shape, device)
        } else if matches!(self.storage, KvStorage::RotorKOnly4 { .. }) {
            super::update::rotor4_k_only_gpu_append(self, new_k, new_shape, device)
        } else {
            Err(Error::KvStorageMismatch {
                expected: "RotorKOnly3 | RotorKOnly4",
                got: storage_variant_name(&self.storage),
            })
        }
    }

    /// `(codes, scales, norms, rotors)` GPU view of the active rotor K store at
    /// `kv_seq`, or `None` when the ring is not live.
    fn rotor_k_packed_view(
        &self,
        kv_seq: i32,
        device: Device,
    ) -> Result<Option<(Array, Array, Array, Array)>> {
        let (view, rotors) = if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = &self.storage {
            (ks.gpu_packed_view(kv_seq, device)?, &ks.rotors)
        } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = &self.storage {
            (ks.gpu_packed_view(kv_seq, device)?, &ks.rotors)
        } else {
            return Ok(None);
        };
        let Some((codes, scales, norms)) = view else {
            return Ok(None);
        };
        let rotors_arr = crate::rotorquant_msl::rotor_table_to_array(rotors)?;
        Ok(Some((codes, scales, norms, rotors_arr)))
    }

    /// Flash-decode dispatch for the iso K-only storage variants
    /// (`IsoKOnly3` / `IsoKOnly4`).
    ///
    /// Sibling of [`Self::update_and_sdpa_rotor_k_fused`] — see it for the full
    /// rationale. Same shape, same ordering constraints:
    ///
    ///   1. Append `new_k` into the iso store — GPU encode into the packed ring,
    ///      no dequant.
    ///   2. Append `new_v` into the bf16 accumulator (V is bf16 for this codec).
    ///   3. Run [`iso_flash_decode_sdpa`] over the ring's packed view.
    ///
    /// The full-prefix `QuantIsoK{3,4}::dequant()` is skipped — that is the
    /// entire point: it is O(seq) CPU work per decode step.
    ///
    /// Caller must have already gated `q_seq == 1` and `device == Gpu` BEFORE
    /// any cache mutation.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "all index uses validated by the shape contracts at function entry"
    )]
    fn update_and_sdpa_iso_k_fused(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Option<Array>> {
        let new_shape = new_k.shape();
        if new_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "iso_k_fused: new_k rank != 4, got {new_shape:?}"
            )));
        }
        let new_seq = new_shape[2];

        // Shape gates — checked BEFORE any mutation so a reject is a clean
        // fall-through to the legacy path.
        if !Self::iso_flash_shape_ok(&new_shape) {
            return Ok(None);
        }

        let prev_seq = self.offset;
        // Not an iso K-only cache: the caller's gate should have kept us out.
        // Fall through rather than mutate.
        if !matches!(
            self.storage,
            KvStorage::IsoKOnly3 { .. } | KvStorage::IsoKOnly4 { .. }
        ) {
            return Ok(None);
        }
        // Provision this step before the first mutation: both the packed ring
        // below and the bf16 V mirror further down are capped by the storage
        // `max_seq`, so it has to cover `prev_seq + new_seq` first.
        self.ensure_decode_capacity(prev_seq + new_seq)?;
        self.iso_k_gpu_append(new_k, &new_shape, device)?;

        // CRITICAL: advance `self.offset` BEFORE `update_decode_fp16_v_only` —
        // the V-only helper computes its write window as
        // `[self.offset - new_seq, self.offset)`. Without the pre-increment
        // every decode step overwrites the last V position instead of
        // appending. Same ordering as `update_and_sdpa_rotor_k_fused`.
        self.offset = prev_seq + new_seq;
        let max_seq = iso_k_max_seq(&self.storage)?;
        let v_full = self.update_decode_fp16_v_only(new_v, max_seq, device)?;

        // Take `kv_seq` from the store the ring was written from, not from
        // `self.offset` — one source of truth.
        let kv_seq = iso_k_accumulated_seq(&self.storage)?;
        debug_assert_eq!(
            kv_seq, self.offset,
            "iso_k_fused: store seq {kv_seq} != cache offset {} — the ring write \
             and the attention length disagree",
            self.offset
        );
        // Past this point the cache is already mutated, so `Ok(None)` is NOT
        // available: it would send the caller into the legacy `update()` path,
        // which appends K/V a second time and advances `offset` again. Every
        // not-eligible condition is screened at the call-site gate and by
        // `iso_flash_shape_ok` BEFORE any mutation.
        let out =
            self.iso_k_flash_over_store(queries, &v_full, scale, additive_mask, kv_seq, device)?;
        Ok(Some(out))
    }

    /// Run the iso flash-decode kernel over the K already packed in this cache's
    /// store — no append, no offset advance.
    ///
    /// Split out of [`Self::update_and_sdpa_iso_k_fused`] so a shared-KV
    /// **consumer** layer can attend the producer's store through the exact same
    /// kernel the producer used. Sibling of
    /// [`Self::rotor_k_flash_over_store`].
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "all index uses validated by the shape contracts at function entry"
    )]
    fn iso_k_flash_over_store(
        &self,
        queries: &Array,
        v_full: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        kv_seq: i32,
        device: Device,
    ) -> Result<Array> {
        // Select the bit width from the live storage variant. No wildcard arm:
        // an unexpected variant must not be decoded with another codec's kernel.
        let bits: u8 = if matches!(self.storage, KvStorage::IsoKOnly3 { .. }) {
            3
        } else if matches!(self.storage, KvStorage::IsoKOnly4 { .. }) {
            4
        } else {
            return Err(Error::KvStorageMismatch {
                expected: "IsoKOnly3 | IsoKOnly4",
                got: storage_variant_name(&self.storage),
            });
        };
        let Some((codes, scales, norms)) = self.iso_k_packed_view(kv_seq, device)? else {
            return Err(Error::Mlx(format!(
                "iso_k_fused: GPU ring absent after a maintained append \
                 (kv_seq={kv_seq}, bits={bits}) — internal invariant violated"
            )));
        };

        let v_shape = v_full.shape();
        if v_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "iso_k_fused: V rank != 4, got {v_shape:?}"
            )));
        }
        let b = v_shape[0];
        let kv_h = v_shape[1];
        let head_dim = v_shape[3];

        let q_shape = queries.shape();
        if q_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "iso_k_fused: Q shape rank != 4, got {q_shape:?}"
            )));
        }
        let q_seq = q_shape[2];
        let n_q_heads = q_shape[1];
        // Defence-in-depth: the call site already gates decode-only.
        if q_seq != 1 {
            return Err(Error::Mlx(format!(
                "iso_k_fused: q_seq must be 1 (decode-only), got {q_seq}"
            )));
        }
        if n_q_heads % kv_h != 0 {
            return Err(Error::Mlx(format!(
                "iso_k_fused: n_q_heads={n_q_heads} not divisible by kv_h={kv_h}"
            )));
        }
        let heads_per_kv = n_q_heads / kv_h;

        tracing::Span::current().record("path", "iso_flash_decode");
        // Select the bit width explicitly. A wildcard arm here would map any
        // unexpected `bits` onto one width's kernel and decode the other's codes
        // at the wrong unpack stride — silently wrong K, no error. `bits` is a
        // plain integer, so no exhaustiveness lint would catch that.
        let flash_out = match bits {
            3 => iso_flash_decode_sdpa::<3>(
                queries,
                &codes,
                &scales,
                &norms,
                v_full,
                additive_mask,
                b,
                kv_h,
                kv_seq,
                head_dim,
                heads_per_kv,
                scale,
                device,
            )?,
            4 => iso_flash_decode_sdpa::<4>(
                queries,
                &codes,
                &scales,
                &norms,
                v_full,
                additive_mask,
                b,
                kv_h,
                kv_seq,
                head_dim,
                heads_per_kv,
                scale,
                device,
            )?,
            other => {
                return Err(Error::Quant(format!(
                    "iso_k_fused: unsupported bits={other} (only 3 and 4); \
                     refusing to decode with another width's kernel"
                )))
            }
        };
        if flash_out.dtype() == queries.dtype() {
            Ok(flash_out)
        } else {
            flash_out.astype(queries.dtype(), device)
        }
    }

    /// Shape gates for the iso flash-decode kernel, evaluated before any cache
    /// mutation.
    ///
    /// Keyed off shape only — never an arch name. `b == 1` because the packed
    /// ring's per-step stride does not interleave batch (one `KvCache` per
    /// request); a batched cache falls back rather than read a layout the kernel
    /// would misinterpret.
    fn iso_flash_shape_ok(new_shape: &[i32]) -> bool {
        let (Some(&b), Some(&kv_h), Some(&head_dim)) =
            (new_shape.first(), new_shape.get(1), new_shape.get(3))
        else {
            return false;
        };
        if b != 1 || kv_h <= 0 {
            return false;
        }
        if head_dim <= 0 || head_dim > ISO_FLASH_HEAD_DIM_MAX {
            return false;
        }
        // One quaternion block per group — a partial trailing group has no
        // defined decode.
        if !(head_dim as usize).is_multiple_of(ISO_K3_GROUP_SIZE) {
            return false;
        }
        // Tree reduction over `head_dim` threads.
        (head_dim as u32).is_power_of_two()
    }

    /// GPU-append `new_k` into whichever iso K-only store is active.
    fn iso_k_gpu_append(&mut self, new_k: &Array, new_shape: &[i32], device: Device) -> Result<()> {
        if matches!(self.storage, KvStorage::IsoKOnly3 { .. }) {
            super::update::iso3_k_only_gpu_append(self, new_k, new_shape, device)
        } else if matches!(self.storage, KvStorage::IsoKOnly4 { .. }) {
            super::update::iso4_k_only_gpu_append(self, new_k, new_shape, device)
        } else {
            Err(Error::KvStorageMismatch {
                expected: "IsoKOnly3 | IsoKOnly4",
                got: storage_variant_name(&self.storage),
            })
        }
    }

    /// `(codes, scales, norms)` GPU view of the active iso K store at `kv_seq`,
    /// or `None` when the ring is not live.
    fn iso_k_packed_view(
        &self,
        kv_seq: i32,
        device: Device,
    ) -> Result<Option<(Array, Array, Array)>> {
        if let KvStorage::IsoKOnly3 { k: Some(ks), .. } = &self.storage {
            ks.gpu_packed_view(kv_seq, device)
        } else if let KvStorage::IsoKOnly4 { k: Some(ks), .. } = &self.storage {
            ks.gpu_packed_view(kv_seq, device)
        } else {
            Ok(None)
        }
    }
}

/// Whether the active rotor K-only store carries the QJL residual sideband.
///
/// Reads the **store's** decision rather than the global
/// `rotor_qjl_enabled()` toggle. The codec fixes QJL at first append and never
/// adds or drops the sideband mid-stream, so the store is the authority: a
/// toggle flipped after the store was built must not change how its existing
/// bytes are read. Before the first append there is no store yet, so the global
/// toggle — the value the store is about to be built with — is the answer.
fn rotor_k_store_uses_qjl(storage: &KvStorage) -> bool {
    if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = storage {
        ks.use_qjl()
    } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = storage {
        ks.use_qjl()
    } else {
        crate::rotor_qjl::rotor_qjl_enabled()
    }
}

/// Accumulated sequence length held by the active rotor K-only store.
///
/// The store's own `shape[2]` — the length the GPU ring was written against —
/// so callers do not have to trust that `KvCache::offset` still agrees.
fn rotor_k_accumulated_seq(storage: &KvStorage) -> Result<i32> {
    let shape = if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = storage {
        &ks.shape
    } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = storage {
        &ks.shape
    } else {
        return Err(Error::KvStorageMismatch {
            expected: "RotorKOnly3 | RotorKOnly4 with a live K buffer",
            got: storage_variant_name(storage),
        });
    };
    shape.get(2).copied().ok_or_else(|| {
        Error::Mlx(format!(
            "rotor_k_fused: rotor K store shape {shape:?} has no seq axis"
        ))
    })
}

/// Accumulated sequence length held by the active PlanarK store.
///
/// Sibling of [`rotor_k_accumulated_seq`], and for the same reason: the store's
/// own `shape[2]` is the length the packed buffers were written against, so it —
/// not [`KvCache::offset`] — is what the kernel attends and what a shared-KV
/// consumer must size its mask from.
fn planar_k_accumulated_seq(storage: &KvStorage) -> Result<i32> {
    let KvStorage::PlanarK { k: Some(ks), .. } = storage else {
        return Err(Error::KvStorageMismatch {
            expected: "PlanarK with a live K buffer",
            got: storage_variant_name(storage),
        });
    };
    ks.shape.get(2).copied().ok_or_else(|| {
        Error::Mlx(format!(
            "planar_k_fused: PlanarK store shape {:?} has no seq axis",
            ks.shape
        ))
    })
}

/// Accumulated sequence length held by the active iso K-only store.
///
/// Sibling of [`rotor_k_accumulated_seq`]: the store's own `shape[2]` is the
/// length the GPU ring was written against, so it — not [`KvCache::offset`] —
/// is what the kernel attends and what a shared-KV consumer sizes its mask
/// from.
fn iso_k_accumulated_seq(storage: &KvStorage) -> Result<i32> {
    let shape = if let KvStorage::IsoKOnly3 { k: Some(ks), .. } = storage {
        &ks.shape
    } else if let KvStorage::IsoKOnly4 { k: Some(ks), .. } = storage {
        &ks.shape
    } else {
        return Err(Error::KvStorageMismatch {
            expected: "IsoKOnly3 | IsoKOnly4 with a live K buffer",
            got: storage_variant_name(storage),
        });
    };
    shape.get(2).copied().ok_or_else(|| {
        Error::Mlx(format!(
            "iso_k_fused: iso K store shape {shape:?} has no seq axis"
        ))
    })
}

/// `max_seq` of the active iso K-only storage variant.
///
/// Read from the live `KvStorage` variant — which `ensure_decode_capacity` has
/// just grown for this step — never from the store struct's own inert
/// `max_seq` field, which is a prefill-time snapshot.
fn iso_k_max_seq(storage: &KvStorage) -> Result<i32> {
    if let KvStorage::IsoKOnly3 { max_seq, .. } | KvStorage::IsoKOnly4 { max_seq, .. } = storage {
        Ok(*max_seq)
    } else {
        Err(Error::KvStorageMismatch {
            expected: "IsoKOnly3 | IsoKOnly4",
            got: storage_variant_name(storage),
        })
    }
}

/// `max_seq` of the active rotor K-only storage variant.
fn rotor_k_max_seq(storage: &KvStorage) -> Result<i32> {
    if let KvStorage::RotorKOnly3 { max_seq, .. } | KvStorage::RotorKOnly4 { max_seq, .. } = storage
    {
        Ok(*max_seq)
    } else {
        Err(Error::KvStorageMismatch {
            expected: "RotorKOnly3 | RotorKOnly4",
            got: storage_variant_name(storage),
        })
    }
}
