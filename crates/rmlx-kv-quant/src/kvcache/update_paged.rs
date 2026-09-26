//! Paged KV update path.
//!
//! Holds the one update-side body that only the paged storage uses. The
//! `KvStorage` dispatch and the helpers with more than one family caller stay
//! in [`super::update`].

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::storage::KvStorage;
use crate::KvQuant;

use super::KvCache;

impl KvCache {
    /// PagedAttention decode-step update (paged KV path, `--paged-kv`).
    ///
    /// Steps:
    /// 1. Quantize `new_k` (q8_0) and `new_v` (TurboQuant V4 / q8_0 / Planar)
    /// 2. Append into the block-table page allocator (creates/fills pages).
    /// 3. Gather all filled pages into contiguous flat arrays.
    /// 4. Dequantize and return the full accumulated `(K, V)`.
    ///
    /// Falls back to the warm-TTFT fp16 decode seed path if `decode_fp16_k` is
    /// set (same as the non-paged quant paths).
    #[allow(
        clippy::unreachable,
        reason = "all sites guard KvStorage::Paged invariant or impossible paged_quant arms; \
                  each is reachable only via a construction-time BUG in cache allocation"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub(super) fn update_paged(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        use crate::paged::{
            paged_kv_page_tokens, PagedKStorage, PagedPlanarVStorage, PagedVStorage,
        };
        use crate::planarquant_msl::{planar_dequantize_v4_gpu, planar_quantize_v4_gpu};
        use crate::q8_msl::{q8_dequantize_gpu, q8_quantize_gpu};
        use crate::turboquant_msl::{turbo_dequantize_v4_gpu, turbo_quantize_v4_gpu};

        // Extract quant + max_seq without holding the storage borrow.
        let (paged_quant, max_seq) = match &self.storage {
            KvStorage::Paged { quant, .. } => (*quant, self.storage.max_seq()),
            _ => unreachable!("update_paged called on non-Paged storage"),
        };

        // Warm-TTFT fp16 seed path (same as non-paged quant paths).
        // decode_fp16_k is set after exit_prefill; first decode step uses it.
        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        // GPU only: paged path only activates on GPU; CPU falls back to decode_fp16.
        if device != Device::Gpu {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();
        let page_tokens = paged_kv_page_tokens();
        let n_pages = ((max_seq + page_tokens - 1) / page_tokens) as usize;

        // The page slabs store `words_per_token` (all heads) per token slot, so
        // the physical layout is sequence-major: per token, all heads are
        // contiguous. `new_k`/`new_v` arrive head-major (`[B, kv_h, S, D]`);
        // quantizing them directly emits head-major codes that the per-token
        // page write then mis-indexes (the `prev_seq * words_per_seq` class of
        // the QuantK / QuantV layout fix). Reorder to sequence-major
        // `[B, S, kv_h, D]` and materialize (`contiguous`) — the q8 / TurboQuant
        // / PlanarQuant MSL kernels read their input by raw linear offset and
        // ignore MLX strides. `gather` then yields a sequence-major prefix;
        // dequant reshapes sequence-major and transposes back to the logical
        // `[B, kv_h, S, D]`.
        let new_k_sm = new_k.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;
        let new_v_sm = new_v.transpose(&[0, 2, 1, 3], device)?.contiguous(device)?;

        // --- K (always q8_0) ---
        let (codes_k, scales_k) = q8_quantize_gpu(&new_k_sm, device)?;
        let KvStorage::Paged { k, .. } = &mut self.storage else {
            unreachable!()
        };
        if k.is_none() {
            *k = Some(PagedKStorage::new(max_seq, page_tokens, n_pages));
        }
        let pk = k.as_mut().unwrap();
        pk.append(&new_shape, codes_k, scales_k, device)?;
        let k_shape = pk.shape.clone();
        let total_k = pk.total_tokens;
        let n_pages_k = pk.block_table.len();
        let (gathered_codes_k, gathered_scales_k) = pk.gather(device)?;
        // `pk.shape` is the accumulated head-major `[B, kv_h, S, D]`; the page
        // buffer is sequence-major, so dequant into `[B, S, kv_h, D]` then
        // transpose heads↔seq back to the logical head-major shape.
        let k_sm_shape = [k_shape[0], k_shape[2], k_shape[1], k_shape[3]];
        let k_full = q8_dequantize_gpu(
            &gathered_codes_k,
            &gathered_scales_k,
            &k_sm_shape,
            new_k.dtype(),
            device,
        )?
        .transpose(&[0, 2, 1, 3], device)?
        .contiguous(device)?;

        // --- V (mode-dependent) ---
        let v_full = match paged_quant {
            KvQuant::K8V8 => {
                let (codes_v, scales_v) = q8_quantize_gpu(&new_v_sm, device)?;
                let KvStorage::Paged { v_k8, .. } = &mut self.storage else {
                    unreachable!()
                };
                if v_k8.is_none() {
                    *v_k8 = Some(Box::new(PagedVStorage::new(
                        max_seq,
                        page_tokens,
                        n_pages,
                        8,
                    )));
                }
                let pv = v_k8.as_mut().unwrap();
                pv.append(&new_shape, codes_v, scales_v, device)?;
                let v_shape = pv.shape.clone();
                let v_sm_shape = [v_shape[0], v_shape[2], v_shape[1], v_shape[3]];
                let (gathered_codes_v, gathered_scales_v) = pv.gather(device)?;
                q8_dequantize_gpu(
                    &gathered_codes_v,
                    &gathered_scales_v,
                    &v_sm_shape,
                    new_v.dtype(),
                    device,
                )?
                .transpose(&[0, 2, 1, 3], device)?
                .contiguous(device)?
            }
            KvQuant::K8V4 => {
                let (codes_v, scales_v) = turbo_quantize_v4_gpu(&new_v_sm, device)?;
                let KvStorage::Paged { v_k8, .. } = &mut self.storage else {
                    unreachable!()
                };
                if v_k8.is_none() {
                    *v_k8 = Some(Box::new(PagedVStorage::new(
                        max_seq,
                        page_tokens,
                        n_pages,
                        4,
                    )));
                }
                let pv = v_k8.as_mut().unwrap();
                pv.append(&new_shape, codes_v, scales_v, device)?;
                let v_shape = pv.shape.clone();
                let v_sm_shape = [v_shape[0], v_shape[2], v_shape[1], v_shape[3]];
                let (gathered_codes_v, gathered_scales_v) = pv.gather(device)?;
                turbo_dequantize_v4_gpu(
                    &gathered_codes_v,
                    &gathered_scales_v,
                    &v_sm_shape,
                    new_v.dtype(),
                    device,
                )?
                .transpose(&[0, 2, 1, 3], device)?
                .contiguous(device)?
            }
            KvQuant::Planar => {
                let (codes_v, scales_v, rotations_v) = planar_quantize_v4_gpu(&new_v_sm, device)?;
                let KvStorage::Paged { v_planar, .. } = &mut self.storage else {
                    unreachable!()
                };
                if v_planar.is_none() {
                    *v_planar = Some(Box::new(PagedPlanarVStorage::new(
                        max_seq,
                        page_tokens,
                        n_pages,
                    )));
                }
                let pv = v_planar.as_mut().unwrap();
                pv.append(&new_shape, codes_v, scales_v, rotations_v, device)?;
                let v_shape = pv.shape.clone();
                let v_sm_shape = [v_shape[0], v_shape[2], v_shape[1], v_shape[3]];
                let (gathered_codes_v, gathered_scales_v, gathered_rotations_v) =
                    pv.gather(device)?;
                planar_dequantize_v4_gpu(
                    &gathered_codes_v,
                    &gathered_scales_v,
                    &gathered_rotations_v,
                    &v_sm_shape,
                    new_v.dtype(),
                    device,
                )?
                .transpose(&[0, 2, 1, 3], device)?
                .contiguous(device)?
            }
            _ => {
                return Err(Error::Mlx(
                    "update_paged: unexpected quant mode (None/Mixed not routed here)".into(),
                ));
            }
        };

        tracing::trace!(
            seq = new_shape[2],
            total = total_k,
            pages = n_pages_k,
            "paged KV append"
        );

        Ok((k_full, v_full))
    }
}
