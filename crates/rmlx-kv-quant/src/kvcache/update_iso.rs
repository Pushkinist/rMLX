// LOC-exempt: the iso family carries three storage spellings — V, symmetric
// and K-only — at two code widths, and each needs its own GPU encode, ring
// sync and prefill bulk-encode path. The six prefill bodies joined the decode
// bodies here when the `exit_prefill` arms were extracted. Splitting the file
// by verb instead would put a ring sync in a different file from the appender
// that maintains it, which is the drift the family grouping exists to prevent.
//! Iso KV update path.
//!
//! Holds every update-side body that only the iso storage types use: the
//! per-variant `update_iso_*` decode entries, the `exit_prefill_iso*` prefill
//! bulk-encode bodies, the GPU encode and ring-sync helpers, the chunk
//! appenders and the materialise-tail path. The `KvStorage` dispatch and the
//! helpers with more than one family caller stay in [`super::update`].

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::storage::{iso_n_groups_for, IsoBlocks, KvStorage, QuantIsoK, QuantIsoV, QuantK};

use super::helpers::{array_to_f32_vec, arrays_to_f32, f32_vec_to_array};
use super::update::{
    accumulated_seq, b_kv_h_new_seq, bump_ring_k_shape, collapse_group_norms_to_token,
    head_dim_from_shape, is_ring_only_append, packed_k_chunk_seq_major, storage_mismatch,
    warn_if_width_disagrees, PackedKEncodedGpu, RingFeed,
};
use super::KvCache;

/// Push a pre-built [`IsoBlocks`] onto an iso K buffer and update `ks.shape`
/// the same way [`QuantIsoK::append`] does.
#[allow(
    clippy::indexing_slicing,
    reason = "ks.shape rank-4 guard above each indexing site; new_shape rank validated by upstream encoder helper"
)]
pub(super) fn push_iso_k_block<const BITS: u8>(
    ks: &mut QuantIsoK<BITS>,
    block: IsoBlocks,
    new_shape: &[i32],
) {
    ks.blocks.push(block);
    if ks.shape.len() != 4 || ks.shape[0] == 0 {
        ks.shape = new_shape.to_vec();
    } else {
        ks.shape[2] += new_shape[2];
    }
}
/// Append one iso K chunk at `BITS`: GPU encode → CPU blocks (+ the GPU ring
/// when `feed` is [`RingFeed::Maintain`]).
///
/// Used by the K side of [`iso_sym_update`] and by [`iso_k_only_k_side`]
/// (both `RingFeed::Skip` — they dequant the whole prefix anyway), and by
/// [`iso_k_only_gpu_append_at`] on the flash-decode path
/// (`RingFeed::MaintainRingOnly`).
///
/// The chunk is reordered to sequence-major before encoding. That is not
/// optional: `QuantIsoK::append` (the CPU path) transposes, and
/// `QuantIsoK::dequant{,_gpu}` transpose *back* — so encoding head-major here
/// would hand `dequant` head-major blocks it reads as sequence-major and
/// scramble heads for any `kv_h > 1` chunk longer than one token. For the
/// decode step (`S == 1`) the reorder is the identity.
pub(super) fn iso_gpu_append_into_k_blocks<const BITS: u8>(
    ks: &mut QuantIsoK<BITS>,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim = head_dim_from_shape(new_shape, "iso_gpu_append_into_k_blocks")?;
    let seq_major = packed_k_chunk_seq_major(new_k, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        // Ring-only tail: feed the GPU ring, advance `shape[2]`, and skip the
        // per-step host download + CPU block push. The ring is the source of
        // truth for the decode tail; the blocks are rebuilt on demand at a
        // `dequant()` / SSD-spill boundary (`synced_iso_v_blocks`).
        let gpu = iso_gpu_encode_ring_only(&seq_major, new_shape, BITS)?;
        iso_k_sync_ring(
            ks,
            &gpu,
            RingFeed::Maintain,
            new_shape,
            head_dim,
            max_seq,
            device,
        )?;
        bump_ring_k_shape(&mut ks.shape, new_shape);
        return Ok(());
    }
    // Block path: prefill / non-fused decode (Maintain) and the K-side of the
    // legacy sym fallback (Skip). A ring-only feed that reaches here is a `b > 1`
    // chunk — normalise it to Maintain so the shared ring feeder clears the ring
    // for the un-representable batch and the CPU block carries the data.
    ks.reconcile_ring(device, crate::storage::RingDisposition::Keep)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = iso_gpu_encode_block_retaining(&seq_major, new_shape, BITS)?;
    iso_k_sync_ring(ks, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    push_iso_k_block(ks, block, new_shape);
    Ok(())
}
/// GPU-encode one iso K chunk, returning the CPU block **and** the GPU arrays
/// the ring feed consumes.
///
/// Mirror of [`rotor_gpu_encode_block_retaining`] for the iso codec. The CPU
/// download is kept for the same reason it is there: [`IsoBlocks`] is the
/// source of truth for `dequant()` and the SSD spill, and `shape[2]` is bumped
/// in lockstep with the block push, so skipping the block would leave `blocks`
/// shorter than `shape[2]` — a gap `dequant()` silently zero-pads and a spill
/// would persist as a truncated store.
///
/// `bits` selects the encode kernel and the readback through
/// [`crate::isoquant_msl_dispatch`]; anything but 3 or 4 is an error rather
/// than a silent fall-through to one width's kernel over the other's codes.
pub(super) fn iso_gpu_encode_block_retaining(
    new_k: &Array,
    new_shape: &[i32],
    bits: u8,
) -> Result<(IsoBlocks, PackedKEncodedGpu)> {
    let head_dim = head_dim_from_shape(new_shape, "iso_gpu_encode_block_retaining")?;
    let (b, kv_h, s) = b_kv_h_new_seq(new_shape)?;
    let n_tokens_total = (b as usize) * (kv_h as usize) * (s as usize);
    // Strict form: `iso_n_groups_for` truncates, so a `head_dim` that is not a
    // whole number of quaternion blocks would silently drop the trailing partial
    // group. The helper the deleted iso4 encoder used rejected that; keep the
    // rejection now that this is the shared path.
    let n_groups = usize::try_from(crate::storage::iso_n_groups_i32(
        head_dim as i32,
        "iso_gpu_encode_block_retaining",
    )?)
    .map_err(|_| Error::Quant("iso_gpu_encode_block_retaining: n_groups negative".to_owned()))?;

    let (codes_arr, scales_arr, quats_arr, norms_arr) =
        crate::isoquant_msl_dispatch::iso_quantize_gpu(
            new_k,
            head_dim,
            bits,
            "iso_gpu_encode_block_retaining",
            Device::Gpu,
        )?;

    let (codes, scales, quats, norms) = crate::isoquant_msl_dispatch::iso_gpu_outputs_to_cpu(
        &codes_arr,
        &scales_arr,
        &quats_arr,
        &norms_arr,
        n_tokens_total,
        n_groups,
        bits,
        "iso_gpu_encode_block_retaining",
    )?;

    // The encode kernel emits norms **per group** (`[n_tokens * n_groups]`, the
    // same per-token L2 replicated across a token's groups). The ring stores the
    // per-token form the decode kernel indexes (`norms[tok_idx]`), so collapse
    // it on the GPU to keep the ring feed off the host.
    let norms_per_token = collapse_group_norms_to_token(&norms_arr, n_tokens_total, n_groups)?;

    Ok((
        IsoBlocks {
            codes,
            scales,
            quaternions: quats,
            norms,
            n_tokens: n_tokens_total,
        },
        PackedKEncodedGpu {
            codes: codes_arr,
            scales: scales_arr,
            norms: norms_per_token,
        },
    ))
}
/// GPU-encode one iso K/V chunk for the **ring-only** append path: feed the GPU
/// ring without paying the per-step host download that
/// [`iso_gpu_encode_block_retaining`] does.
///
/// The fused decode kernel reads the ring, never the CPU `IsoBlocks`, so
/// downloading and materialising a block per decode step is pure host work in
/// the path whose purpose is removing host work. Skipping it leaves the ring the
/// sole source of truth for the decode tail; the CPU blocks are rebuilt from the
/// ring on demand at a `dequant()` / SSD-spill boundary (`synced_iso_v_blocks`),
/// and the `blocks`-vs-`shape[2]` invariant is enforced loudly there — never
/// zero-padded. Mirror of [`rotor_gpu_encode_ring_only`].
///
/// `bits` selects the encode kernel through [`crate::isoquant_msl_dispatch`];
/// anything but 3 or 4 is an error.
pub(super) fn iso_gpu_encode_ring_only(
    new_kv: &Array,
    new_shape: &[i32],
    bits: u8,
) -> Result<PackedKEncodedGpu> {
    let head_dim = head_dim_from_shape(new_shape, "iso_gpu_encode_ring_only")?;
    let (b, kv_h, s) = b_kv_h_new_seq(new_shape)?;
    let n_tokens_total = (b as usize) * (kv_h as usize) * (s as usize);
    let n_groups = iso_n_groups_for(head_dim);
    if n_groups == 0 {
        return Err(Error::Quant(format!(
            "iso_gpu_encode_ring_only: head_dim={head_dim} yields no quaternion groups"
        )));
    }
    let (codes_arr, scales_arr, _quats_arr, norms_arr) =
        crate::isoquant_msl_dispatch::iso_quantize_gpu(
            new_kv,
            head_dim,
            bits,
            "iso_gpu_encode_ring_only",
            Device::Gpu,
        )?;
    // Collapse per-group norms to the per-token form the ring stores (GPU-side,
    // no host round-trip).
    let norms_per_token = collapse_group_norms_to_token(&norms_arr, n_tokens_total, n_groups)?;
    Ok(PackedKEncodedGpu {
        codes: codes_arr,
        scales: scales_arr,
        norms: norms_per_token,
    })
}
/// Append `new_k` into a live iso K-only store's GPU ring (+ CPU blocks) at
/// `BITS`, lazily creating the store on first use. No dequant — the fused iso
/// K-only flash-decode SDPA path reaches this through
/// [`iso_k_only_gpu_append`]. `variant` is the storage spelling the caller
/// resolved, used only in the "buffer absent after init" diagnostic.
pub(super) fn iso_k_only_gpu_append_at<const BITS: u8>(
    k: &mut Option<QuantIsoK<BITS>>,
    max_seq: i32,
    variant: &'static str,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    if k.is_none() {
        let mut init_shape = new_shape.to_vec();
        if let Some(s) = init_shape.get_mut(2) {
            *s = 0;
        }
        *k = Some(QuantIsoK::<BITS>::new(init_shape, max_seq));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx(format!("{variant} K buffer absent after init")));
    };
    // Ring-only tail: the fused iso K-only flash decode reads the ring, never the
    // CPU blocks, so skip the per-step host download; the blocks are rebuilt from
    // the ring on demand at a `dequant()` / SSD-spill boundary.
    iso_gpu_append_into_k_blocks(
        ks,
        new_k,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    drop_blocks_when_ring_live_iso_k(ks);
    Ok(())
}
/// Append `new_k` into whichever iso K-only store is active — `IsoKOnly3` at 3
/// bits, `IsoKOnly4` at 4 — lazily creating it on first use. This is the entry
/// point the iso flash-decode SDPA path uses.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is neither
/// `IsoKOnly3` nor `IsoKOnly4`, and forwards encode / ring errors.
pub(super) fn iso_k_only_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let max_seq = cache.max_seq;
    let (KvStorage::IsoKOnly3 { .. } | KvStorage::IsoKOnly4 { .. }) = &cache.storage else {
        return Err(storage_mismatch("IsoKOnly3 | IsoKOnly4", &cache.storage));
    };

    if let KvStorage::IsoKOnly3 { k, .. } = &mut cache.storage {
        iso_k_only_gpu_append_at::<3>(k, max_seq, "IsoKOnly3", new_k, new_shape, device)
    } else if let KvStorage::IsoKOnly4 { k, .. } = &mut cache.storage {
        iso_k_only_gpu_append_at::<4>(k, max_seq, "IsoKOnly4", new_k, new_shape, device)
    } else {
        // Unreachable: the width read above accepted no other variant.
        Err(storage_mismatch("IsoKOnly3 | IsoKOnly4", &cache.storage))
    }
}
/// Append `new_k` / `new_v` into a live iso symmetric store's GPU rings
/// (ring-only tail on both axes) at `BITS`, lazily creating the stores on first
/// use. No dequant on either axis — the fused iso symmetric quant-V
/// flash-decode SDPA path reaches this through [`iso_sym_gpu_append`].
/// `variant` is the storage spelling the caller resolved, used only in the
/// "buffer absent after init" diagnostics.
pub(super) fn iso_sym_gpu_append_at<const BITS: u8>(
    k: &mut Option<QuantIsoK<BITS>>,
    v: &mut Option<QuantIsoV<BITS>>,
    max_seq: i32,
    variant: &'static str,
    new_k: &Array,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let mut init_shape = new_shape.to_vec();
    if let Some(s) = init_shape.get_mut(2) {
        *s = 0;
    }
    if k.is_none() {
        *k = Some(QuantIsoK::<BITS>::new(init_shape.clone(), max_seq));
    }
    if v.is_none() {
        *v = Some(QuantIsoV::<BITS>::new(init_shape));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx(format!("{variant} K buffer absent after init")));
    };
    iso_gpu_append_into_k_blocks(
        ks,
        new_k,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    // Ring is now the sole resident store for K — drop the redundant CPU blocks
    // (the prefill prefix, seeded into the ring on the first fused-decode step).
    drop_blocks_when_ring_live_iso_k(ks);
    let Some(vs) = v.as_mut() else {
        return Err(Error::Mlx(format!("{variant} V buffer absent after init")));
    };
    iso_gpu_append_into_v_blocks(
        vs,
        new_v,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    // The GPU ring holds the full prefix; the fused decode reads the ring, and
    // `dequant` / SSD spill / clone / truncate rebuild the CPU blocks on demand
    // from it (`synced_iso_v_blocks`).
    drop_blocks_when_ring_live_iso_v(vs);
    Ok(())
}
/// Append `new_k` / `new_v` into whichever iso symmetric store is active —
/// `IsoSym3` at 3 bits, `IsoSym4` at 4 — lazily creating the stores on first
/// use. This is the entry point the iso symmetric quant-V flash-decode SDPA
/// path uses.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is neither
/// `IsoSym3` nor `IsoSym4`, and forwards encode / ring errors.
pub(super) fn iso_sym_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let max_seq = cache.max_seq;
    let (KvStorage::IsoSym3 { .. } | KvStorage::IsoSym4 { .. }) = &cache.storage else {
        return Err(storage_mismatch("IsoSym3 | IsoSym4", &cache.storage));
    };

    if let KvStorage::IsoSym3 { k, v, .. } = &mut cache.storage {
        iso_sym_gpu_append_at::<3>(k, v, max_seq, "IsoSym3", new_k, new_v, new_shape, device)
    } else if let KvStorage::IsoSym4 { k, v, .. } = &mut cache.storage {
        iso_sym_gpu_append_at::<4>(k, v, max_seq, "IsoSym4", new_k, new_v, new_shape, device)
    } else {
        // Unreachable: the width read above accepted no other variant.
        Err(storage_mismatch("IsoSym3 | IsoSym4", &cache.storage))
    }
}
/// Drop an iso K store's CPU blocks once its GPU ring is live — the ring is then
/// the sole resident copy. No-op until the ring is allocated or for any store
/// that never feeds a ring.
pub(super) fn drop_blocks_when_ring_live_iso_k<const BITS: u8>(ks: &mut QuantIsoK<BITS>) {
    if ks.gpu.is_allocated() {
        ks.blocks.clear();
        ks.blocks.shrink_to_fit();
    }
}
/// Drop an iso V store's CPU blocks once its GPU ring is live.
pub(super) fn drop_blocks_when_ring_live_iso_v<const BITS: u8>(vs: &mut QuantIsoV<BITS>) {
    if vs.gpu.is_allocated() {
        vs.blocks.clear();
        vs.blocks.shrink_to_fit();
    }
}
/// V-side append with a ring-only branch at `BITS` — mirror of
/// [`iso_gpu_append_into_k_blocks`]. The iso codec is axis-agnostic, so V
/// encodes through the same kernel as K.
pub(super) fn iso_gpu_append_into_v_blocks<const BITS: u8>(
    vs: &mut QuantIsoV<BITS>,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim = head_dim_from_shape(new_shape, "iso_gpu_append_into_v_blocks")?;
    let seq_major = packed_k_chunk_seq_major(new_v, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        let gpu = iso_gpu_encode_ring_only(&seq_major, new_shape, BITS)?;
        iso_v_sync_ring(
            vs,
            &gpu,
            RingFeed::Maintain,
            new_shape,
            head_dim,
            max_seq,
            device,
        )?;
        bump_ring_k_shape(&mut vs.shape, new_shape);
        return Ok(());
    }
    vs.reconcile_ring(device, crate::storage::RingDisposition::Keep)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = iso_gpu_encode_block_retaining(&seq_major, new_shape, BITS)?;
    iso_v_sync_ring(vs, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    vs.blocks.push(block);
    bump_ring_k_shape(&mut vs.shape, new_shape);
    Ok(())
}
/// Test seam onto the iso V append, at the width and feed the caller states.
///
/// The appender is private to this module and so is [`RingFeed`]; the
/// orientation test lives with the other iso dispatch tests, one module up.
/// Exposing the call rather than duplicating it is what keeps the test pinned
/// to the production path.
#[cfg(test)]
pub(super) fn iso_v_gpu_append_for_test<const BITS: u8>(
    vs: &mut QuantIsoV<BITS>,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
    max_seq: i32,
) -> Result<()> {
    iso_gpu_append_into_v_blocks(vs, new_v, new_shape, device, RingFeed::Skip, max_seq)
}
/// V-side mirror of [`iso_k_sync_ring`].
pub(super) fn iso_v_sync_ring<const BITS: u8>(
    vs: &mut QuantIsoV<BITS>,
    gpu: &PackedKEncodedGpu,
    feed: RingFeed,
    new_shape: &[i32],
    head_dim: usize,
    max_seq: i32,
    device: Device,
) -> Result<()> {
    let (b, kv_h, new_seq) = b_kv_h_new_seq(new_shape)?;
    if feed != RingFeed::Maintain || b != 1 {
        vs.gpu.clear();
        return Ok(());
    }
    let prev_seq = accumulated_seq(&vs.shape);
    vs.gpu_append(
        &gpu.codes,
        &gpu.scales,
        &gpu.norms,
        kv_h,
        head_dim as i32,
        prev_seq,
        new_seq,
        max_seq,
        device,
    )
}
/// Feed one encoded iso K chunk into the store's GPU ring.
///
/// `b > 1` is a skip for the same reason [`rotor_k_sync_ring`] skips it:
/// [`crate::storage::QuantKGpuRing`]'s per-step stride does not interleave
/// batch. A skipped feed *clears* rather than leaving a stale ring — see the
/// [`RingFeed`] invariant.
pub(super) fn iso_k_sync_ring<const BITS: u8>(
    ks: &mut QuantIsoK<BITS>,
    gpu: &PackedKEncodedGpu,
    feed: RingFeed,
    new_shape: &[i32],
    head_dim: usize,
    max_seq: i32,
    device: Device,
) -> Result<()> {
    let (b, kv_h, new_seq) = b_kv_h_new_seq(new_shape)?;
    if feed != RingFeed::Maintain || b != 1 {
        ks.gpu.clear();
        return Ok(());
    }
    // Feed BEFORE `push_iso_k_block` — the push bumps `ks.shape[2]`, and the
    // ring append needs `prev_seq`, the length before this chunk.
    let prev_seq = accumulated_seq(&ks.shape);
    ks.gpu_append(
        &gpu.codes,
        &gpu.scales,
        &gpu.norms,
        kv_h,
        head_dim as i32,
        prev_seq,
        new_seq,
        max_seq,
        device,
    )
}
/// Encode one V chunk into an iso V store and return the V rows attention
/// receives — the whole V axis of [`iso_v_update`] and [`iso_sym_update`].
///
/// Lazily creates the store, takes the GPU encode when `device` is
/// `Device::Gpu` and the CPU one otherwise, and decodes through the matching
/// entry. Each phase emits one structured `trace!` event, off by default; opt
/// in with `--log verbose` or `RUST_LOG=rmlx_kv_quant=trace`. `variant` is the
/// storage spelling the caller resolved: it names the store in the
/// "buffer absent after init" diagnostic and in every trace event, which is
/// what tells the two callers' events apart.
///
/// `v_f32` is read on the CPU route only; the GPU route takes `new_v`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction"
)]
pub(super) fn iso_v_encode_decode<const BITS: u8>(
    v: &mut Option<QuantIsoV<BITS>>,
    new_v: &Array,
    v_f32: &[f32],
    new_shape: &[i32],
    max_seq: i32,
    variant: &'static str,
    device: Device,
) -> Result<Array> {
    if v.is_none() {
        let mut init_shape = new_shape.to_vec();
        init_shape[2] = 0;
        *v = Some(QuantIsoV::<BITS>::new(init_shape));
    }
    let Some(vs) = v.as_mut() else {
        return Err(Error::Mlx(format!("{variant} V buffer absent after init")));
    };
    let kv_h = new_shape[1];
    let head_dim = new_shape[3];
    let t_enc = std::time::Instant::now();
    if device == Device::Gpu {
        // `QuantIsoV::append_gpu` retains the encode outputs in a
        // pre-allocated per-struct buffer so `dequant_gpu` below can skip the
        // CPU-staged `Array::from_bytes` upload on every step.
        vs.append_gpu(new_v, new_shape, max_seq, device)?;
    } else {
        vs.append(v_f32, new_shape)?;
    }
    let s_total = vs.shape[2];
    tracing::trace!(
        phase = "iso_encode",
        bits = BITS,
        variant,
        ms = t_enc.elapsed().as_secs_f64() * 1e3,
        s_total = s_total,
        kv_h,
        head_dim,
        "iso hot-path"
    );
    // On GPU, skip the CPU dequant + vec_to_array round-trip and dispatch the
    // dequant kernel over the packed plane directly. Single-pass GPU side, no
    // intermediate Vec<f32> materialisation.
    if device == Device::Gpu {
        let t_deq = std::time::Instant::now();
        let arr = vs.dequant_gpu(device)?;
        tracing::trace!(
            phase = "iso_dequant_gpu",
            bits = BITS,
            variant,
            ms = t_deq.elapsed().as_secs_f64() * 1e3,
            s_total = s_total,
            kv_h,
            head_dim,
            "iso hot-path"
        );
        return Ok(arr);
    }
    let v_shape = vs.shape.clone();
    let t_deq = std::time::Instant::now();
    let v_recon_f32 = vs.dequant_on(device)?;
    tracing::trace!(
        phase = "iso_dequant_cpu",
        bits = BITS,
        variant,
        ms = t_deq.elapsed().as_secs_f64() * 1e3,
        s_total = s_total,
        kv_h,
        head_dim,
        "iso hot-path"
    );
    let t_mat = std::time::Instant::now();
    let arr = f32_vec_to_array(&v_recon_f32, &v_shape)?;
    tracing::trace!(
        phase = "iso_vec_to_array",
        bits = BITS,
        variant,
        ms = t_mat.elapsed().as_secs_f64() * 1e3,
        s_total = s_total,
        kv_h,
        head_dim,
        "iso hot-path"
    );
    Ok(arr)
}
/// Decode update for `IsoV3` / `IsoV4`: K = affine q8_0, V = IsoQuant at
/// `BITS`.
///
/// The V side routes its encode through the iso MSL kernel when
/// `device == Device::Gpu` and decodes with `dequant_gpu`; CPU encode and
/// `dequant_on` are the fallback. A cache that went through prefill never
/// reaches this body: `exit_prefill` builds no iso V store and decode reads the
/// bf16 mirror. It runs only on a cache with no mirror.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn iso_v_update<const BITS: u8>(
    k: &mut Option<QuantK>,
    v: &mut Option<QuantIsoV<BITS>>,
    max_seq: i32,
    variant: &'static str,
    new_k: &Array,
    new_v: &Array,
    device: Device,
) -> Result<(Array, Array)> {
    let new_shape = new_k.shape();

    let k_f32 = if device == Device::Gpu {
        Vec::new()
    } else {
        array_to_f32_vec(new_k, device)?
    };
    let v_f32 = if device == Device::Gpu {
        Vec::new()
    } else {
        array_to_f32_vec(new_v, Device::Cpu)?
    };

    if k.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        *k = Some(QuantK {
            codes: Vec::new(),
            scales: Vec::new(),
            gpu_codes_buf: None,
            gpu_scales_buf: None,
            gpu_words_per_step: 0,
            gpu_scales_per_step: 0,
            gpu_capacity: 0,
            shape: init_shape,
            max_seq,
        });
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx(format!("{variant} K buffer absent after init")));
    };
    ks.append(&k_f32, &new_shape, new_k, device, max_seq)?;
    let k_shape = ks.shape.clone();
    let (k_recon_f32, k_arr_opt) = ks.dequantize_choice(device, new_k.dtype())?;
    let k_full = match k_arr_opt {
        Some(arr) => arr,
        None => f32_vec_to_array(&k_recon_f32, &k_shape)?,
    };

    let v_full =
        iso_v_encode_decode::<BITS>(v, new_v, &v_f32, &new_shape, max_seq, variant, device)?;

    Ok((k_full, v_full))
}
/// Decode update for `IsoSym3` / `IsoSym4`: K and V both quantize through
/// IsoQuant at `BITS`.
///
/// The kernel is axis-agnostic, so K and V share one dispatch. `variant` is
/// the storage spelling the caller resolved, used only in the "buffer absent
/// after init" diagnostics.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction"
)]
pub(super) fn iso_sym_update<const BITS: u8>(
    k: &mut Option<QuantIsoK<BITS>>,
    v: &mut Option<QuantIsoV<BITS>>,
    max_seq: i32,
    variant: &'static str,
    new_k: &Array,
    new_v: &Array,
    device: Device,
) -> Result<(Array, Array)> {
    let new_shape = new_k.shape();
    let k_f32 = if device == Device::Gpu {
        Vec::new()
    } else {
        array_to_f32_vec(new_k, Device::Cpu)?
    };
    let v_f32 = if device == Device::Gpu {
        Vec::new()
    } else {
        array_to_f32_vec(new_v, Device::Cpu)?
    };

    if k.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        *k = Some(QuantIsoK::<BITS>::new(init_shape, max_seq));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx(format!("{variant} K buffer absent after init")));
    };
    if device == Device::Gpu {
        iso_gpu_append_into_k_blocks(ks, new_k, &new_shape, device, RingFeed::Skip, max_seq)?;
    } else {
        ks.append(&k_f32, &new_shape)?;
    }
    let k_shape = ks.shape.clone();
    let k_full = if device == Device::Gpu {
        ks.dequant_gpu(device)?
    } else {
        let k_recon_f32 = ks.dequant_on(device)?;
        f32_vec_to_array(&k_recon_f32, &k_shape)?
    };

    let v_full =
        iso_v_encode_decode::<BITS>(v, new_v, &v_f32, &new_shape, max_seq, variant, device)?;

    Ok((k_full, v_full))
}
/// The K side of the `IsoKOnly3` / `IsoKOnly4` decode update at `BITS`.
///
/// The V side is the caller's: it stays bf16 and must go through
/// `update_decode_fp16_v_only` — see [`KvCache::update_iso_k_only`].
/// `variant` is the storage spelling the caller resolved, used only in the
/// "buffer absent after init" diagnostic.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction"
)]
pub(super) fn iso_k_only_k_side<const BITS: u8>(
    k: &mut Option<QuantIsoK<BITS>>,
    max_seq: i32,
    variant: &'static str,
    new_k: &Array,
    device: Device,
) -> Result<Array> {
    let new_shape = new_k.shape();
    let k_f32 = if device == Device::Gpu {
        Vec::new()
    } else {
        array_to_f32_vec(new_k, Device::Cpu)?
    };

    if k.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        *k = Some(QuantIsoK::<BITS>::new(init_shape, max_seq));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx(format!("{variant} K buffer absent after init")));
    };
    // Per-phase trace instrumentation for iso K. The V side is bf16, not iso,
    // so no phase events on V.
    let kv_h = new_shape[1];
    let head_dim = new_shape[3];
    let t_enc = std::time::Instant::now();
    if device == Device::Gpu {
        iso_gpu_append_into_k_blocks(ks, new_k, &new_shape, device, RingFeed::Skip, max_seq)?;
    } else {
        ks.append(&k_f32, &new_shape)?;
    }
    let s_total = ks.shape[2];
    tracing::trace!(
        phase = "iso_encode",
        bits = BITS,
        ms = t_enc.elapsed().as_secs_f64() * 1e3,
        s_total = s_total,
        kv_h,
        head_dim,
        "iso hot-path (K-only)"
    );
    let k_shape = ks.shape.clone();
    if device == Device::Gpu {
        let t_deq = std::time::Instant::now();
        let arr = ks.dequant_gpu(device)?;
        tracing::trace!(
            phase = "iso_dequant_gpu",
            bits = BITS,
            ms = t_deq.elapsed().as_secs_f64() * 1e3,
            s_total = s_total,
            kv_h,
            head_dim,
            "iso hot-path (K-only)"
        );
        return Ok(arr);
    }
    let t_deq = std::time::Instant::now();
    let k_recon_f32 = ks.dequant_on(device)?;
    tracing::trace!(
        phase = "iso_dequant_cpu",
        bits = BITS,
        ms = t_deq.elapsed().as_secs_f64() * 1e3,
        s_total = s_total,
        kv_h,
        head_dim,
        "iso hot-path (K-only)"
    );
    let t_mat = std::time::Instant::now();
    let arr = f32_vec_to_array(&k_recon_f32, &k_shape)?;
    tracing::trace!(
        phase = "iso_vec_to_array",
        bits = BITS,
        ms = t_mat.elapsed().as_secs_f64() * 1e3,
        s_total = s_total,
        kv_h,
        head_dim,
        "iso hot-path (K-only)"
    );
    Ok(arr)
}

// ── Iso prefill bulk-encode bodies, one per store shape over both widths ─────
//
// The three `KvCache::exit_prefill_iso_*` entries below resolve their storage
// variant to a code width and hand the stores here; a body below is the one
// bulk encode both widths of a family share, with the width as the store's
// `BITS`.

/// Iso V prefill bulk encode: K affine q8_0 (the K8V4 / K8V8 path), V
/// IsoQuant at `BITS` on CPU.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
)]
pub(super) fn iso_v_bulk_encode<const BITS: u8>(
    k: &mut Option<QuantK>,
    v: &mut Option<QuantIsoV<BITS>>,
    max_seq: i32,
    k_full: &Array,
    v_full: &Array,
    device: Device,
    total_seq: i32,
) -> Result<()> {
    tracing::debug!(
        total_seq,
        bits = BITS,
        "exit_prefill iso V: bulk-quantizing K (q8_0) + V (iso CPU)"
    );
    let new_shape = k_full.shape();
    let (k_f32, v_f32) = arrays_to_f32(k_full, v_full, device)?;
    let mut init_shape = new_shape.clone();
    init_shape[2] = 0;
    let mut qk = QuantK {
        codes: Vec::new(),
        scales: Vec::new(),
        gpu_codes_buf: None,
        gpu_scales_buf: None,
        gpu_words_per_step: 0,
        gpu_scales_per_step: 0,
        gpu_capacity: 0,
        shape: init_shape.clone(),
        max_seq,
    };
    let mut qv = QuantIsoV::<BITS>::new(init_shape);
    qk.append(&k_f32, &new_shape, k_full, device, max_seq)?;
    let kv_h = new_shape[1];
    let head_dim = new_shape[3];
    let t_enc = std::time::Instant::now();
    qv.append(&v_f32, &new_shape)?;
    // `docs/PERF_BASELINE.md` names `iso3_encode` as this site's trace phase;
    // the 4-bit width has none. It does not fire in a served run:
    // `exit_prefill` builds no iso V store, so it never reaches this body.
    if BITS == 3 {
        tracing::trace!(
            phase = "iso3_encode",
            ms = t_enc.elapsed().as_secs_f64() * 1e3,
            s_total = new_shape[2],
            kv_h,
            head_dim,
            site = "exit_prefill",
            "iso3 hot-path"
        );
    }
    *k = Some(qk);
    *v = Some(qv);
    Ok(())
}

/// Iso symmetric prefill bulk encode: K and V both IsoQuant at `BITS`. The
/// codec is axis-agnostic; only the role on the SDPA path differs.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
)]
pub(super) fn iso_sym_bulk_encode<const BITS: u8>(
    k: &mut Option<QuantIsoK<BITS>>,
    v: &mut Option<QuantIsoV<BITS>>,
    max_seq: i32,
    k_full: &Array,
    v_full: &Array,
    device: Device,
    total_seq: i32,
) -> Result<()> {
    tracing::debug!(
        total_seq,
        bits = BITS,
        "exit_prefill iso symmetric: bulk-quantizing K + V (both iso CPU)"
    );
    let new_shape = k_full.shape();
    let (k_f32, v_f32) = arrays_to_f32(k_full, v_full, device)?;
    let mut init_shape = new_shape.clone();
    init_shape[2] = 0;
    let mut qk = QuantIsoK::<BITS>::new(init_shape.clone(), max_seq);
    let mut qv = QuantIsoV::<BITS>::new(init_shape);
    qk.append(&k_f32, &new_shape)?;
    qv.append(&v_f32, &new_shape)?;
    *k = Some(qk);
    *v = Some(qv);
    Ok(())
}

/// Iso K-only prefill bulk encode: K IsoQuant at `BITS`; V stays bf16 on the
/// caller's `decode_fp16_pair`, the same machinery as PlanarK.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: the prefill shape is rank-4 and `init_shape` is its clone"
)]
pub(super) fn iso_k_only_bulk_encode<const BITS: u8>(
    k: &mut Option<QuantIsoK<BITS>>,
    max_seq: i32,
    k_full: &Array,
    device: Device,
    total_seq: i32,
) -> Result<()> {
    tracing::debug!(
        total_seq,
        bits = BITS,
        "exit_prefill iso K-only: bulk-quantizing K (iso CPU); V stays bf16"
    );
    let new_shape = k_full.shape();
    let k_f32 = array_to_f32_vec(k_full, device)?;
    let mut init_shape = new_shape.clone();
    init_shape[2] = 0;
    let mut qk = QuantIsoK::<BITS>::new(init_shape, max_seq);
    qk.append(&k_f32, &new_shape)?;
    *k = Some(qk);
    Ok(())
}

impl KvCache {
    /// Iso V decode update — K = affine q8_0, V = IsoQuant (quaternion SO(4)
    /// rotation + Lloyd-Max codebook) at the code width the active storage
    /// variant carries: `IsoV3` is 3 bits, `IsoV4` is 4.
    ///
    /// The body is [`iso_v_update`] at that width; this entry resolves the
    /// storage variant and takes the warm-TTFT bf16 shortcut.
    pub(super) fn update_iso_v(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let max_seq = self.max_seq;
        let (KvStorage::IsoV3 { .. } | KvStorage::IsoV4 { .. }) = &self.storage else {
            return Err(storage_mismatch("IsoV3 | IsoV4", &self.storage));
        };

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        if let KvStorage::IsoV3 { k, v, .. } = &mut self.storage {
            iso_v_update::<3>(k, v, max_seq, "IsoV3", new_k, new_v, device)
        } else if let KvStorage::IsoV4 { k, v, .. } = &mut self.storage {
            iso_v_update::<4>(k, v, max_seq, "IsoV4", new_k, new_v, device)
        } else {
            // Unreachable: the width read above accepted no other variant.
            Err(storage_mismatch("IsoV3 | IsoV4", &self.storage))
        }
    }
    /// Iso symmetric decode update: K and V both quantize through IsoQuant at
    /// the code width the active storage variant carries — `IsoSym3` is 3
    /// bits, `IsoSym4` is 4.
    ///
    /// The body is [`iso_sym_update`] at that width; this entry resolves the
    /// storage variant and takes the warm-TTFT bf16 shortcut.
    pub(super) fn update_iso_sym(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let max_seq = self.max_seq;
        let (KvStorage::IsoSym3 { .. } | KvStorage::IsoSym4 { .. }) = &self.storage else {
            return Err(storage_mismatch("IsoSym3 | IsoSym4", &self.storage));
        };

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        if let KvStorage::IsoSym3 { k, v, .. } = &mut self.storage {
            iso_sym_update::<3>(k, v, max_seq, "IsoSym3", new_k, new_v, device)
        } else if let KvStorage::IsoSym4 { k, v, .. } = &mut self.storage {
            iso_sym_update::<4>(k, v, max_seq, "IsoSym4", new_k, new_v, device)
        } else {
            // Unreachable: the width read above accepted no other variant.
            Err(storage_mismatch("IsoSym3 | IsoSym4", &self.storage))
        }
    }
    /// Iso K-only decode update. K is IsoQuant at the code width the active
    /// storage variant carries (`IsoKOnly3` is 3 bits, `IsoKOnly4` is 4); V
    /// stays bf16 on `decode_fp16_v` (same machinery as `KvStorage::None` /
    /// `KvStorage::PlanarK`).
    ///
    /// **CRITICAL** (HIGH bug guard): uses
    /// [`Self::update_decode_fp16_v_only`] for the V side, NOT
    /// `update_decode_fp16`. The latter populates `self.decode_fp16_k` as a
    /// side-effect, which causes the `decode_fp16_k.is_some()` early-return
    /// guard to short-circuit the K codec on the *next* decode step (silent
    /// bf16-K regression).
    ///
    /// The K side is [`iso_k_only_k_side`] at the resolved width.
    pub(super) fn update_iso_k_only(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let max_seq = self.max_seq;
        let (KvStorage::IsoKOnly3 { .. } | KvStorage::IsoKOnly4 { .. }) = &self.storage else {
            return Err(storage_mismatch("IsoKOnly3 | IsoKOnly4", &self.storage));
        };

        let k_full = if let KvStorage::IsoKOnly3 { k, .. } = &mut self.storage {
            iso_k_only_k_side::<3>(k, max_seq, "IsoKOnly3", new_k, device)?
        } else if let KvStorage::IsoKOnly4 { k, .. } = &mut self.storage {
            iso_k_only_k_side::<4>(k, max_seq, "IsoKOnly4", new_k, device)?
        } else {
            // Unreachable: the width read above accepted no other variant.
            return Err(storage_mismatch("IsoKOnly3 | IsoKOnly4", &self.storage));
        };

        // V-side: bf16 via the V-only helper (must NOT touch decode_fp16_k).
        let v_full = self.update_decode_fp16_v_only(new_v, max_seq, device)?;
        Ok((k_full, v_full))
    }

    /// Iso V prefill bulk encode — K affine q8_0, V IsoQuant at the code width
    /// the active storage variant carries (`IsoV3` is 3 bits, `IsoV4` is 4).
    /// The body is [`iso_v_bulk_encode`]; this entry resolves the storage
    /// variant.
    pub(super) fn exit_prefill_iso_v(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        let quant_bits = self.quant.approx_code_bits().1;
        let max_seq = self.max_seq;
        if let KvStorage::IsoV3 { k, v, .. } = &mut self.storage {
            warn_if_width_disagrees(self.quant, quant_bits, 3);
            iso_v_bulk_encode::<3>(k, v, max_seq, k_full, v_full, device, total_seq)
        } else if let KvStorage::IsoV4 { k, v, .. } = &mut self.storage {
            warn_if_width_disagrees(self.quant, quant_bits, 4);
            iso_v_bulk_encode::<4>(k, v, max_seq, k_full, v_full, device, total_seq)
        } else {
            Err(storage_mismatch("IsoV3 | IsoV4", &self.storage))
        }
    }

    /// Iso symmetric prefill bulk encode at the width the active storage
    /// variant carries (`IsoSym3` is 3 bits, `IsoSym4` is 4). The body is
    /// [`iso_sym_bulk_encode`]; this entry resolves the storage variant.
    pub(super) fn exit_prefill_iso_sym(
        &mut self,
        k_full: &Array,
        v_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        let quant_bits = self.quant.approx_code_bits().0;
        let max_seq = self.max_seq;
        if let KvStorage::IsoSym3 { k, v, .. } = &mut self.storage {
            warn_if_width_disagrees(self.quant, quant_bits, 3);
            iso_sym_bulk_encode::<3>(k, v, max_seq, k_full, v_full, device, total_seq)
        } else if let KvStorage::IsoSym4 { k, v, .. } = &mut self.storage {
            warn_if_width_disagrees(self.quant, quant_bits, 4);
            iso_sym_bulk_encode::<4>(k, v, max_seq, k_full, v_full, device, total_seq)
        } else {
            Err(storage_mismatch("IsoSym3 | IsoSym4", &self.storage))
        }
    }

    /// Iso K-only prefill bulk encode at the width the active storage variant
    /// carries (`IsoKOnly3` is 3 bits, `IsoKOnly4` is 4); V stays bf16. The
    /// body is [`iso_k_only_bulk_encode`]; this entry resolves the storage
    /// variant.
    pub(super) fn exit_prefill_iso_k_only(
        &mut self,
        k_full: &Array,
        device: Device,
        total_seq: i32,
    ) -> Result<()> {
        let quant_bits = self.quant.approx_code_bits().0;
        let max_seq = self.max_seq;
        if let KvStorage::IsoKOnly3 { k, .. } = &mut self.storage {
            warn_if_width_disagrees(self.quant, quant_bits, 3);
            iso_k_only_bulk_encode::<3>(k, max_seq, k_full, device, total_seq)
        } else if let KvStorage::IsoKOnly4 { k, .. } = &mut self.storage {
            warn_if_width_disagrees(self.quant, quant_bits, 4);
            iso_k_only_bulk_encode::<4>(k, max_seq, k_full, device, total_seq)
        } else {
            Err(storage_mismatch("IsoKOnly3 | IsoKOnly4", &self.storage))
        }
    }
}
