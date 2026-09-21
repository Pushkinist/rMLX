// LOC-exempt: the rotor family carries four storage spellings — V, symmetric,
// K-only and K-asymmetric — and each needs its own GPU encode, ring sync and
// materialise-tail path. Splitting the file by verb instead would put a ring
// sync in a different file from the appender that maintains it, which is the
// drift the family grouping exists to prevent.
//! Rotor KV update path.
//!
//! Holds every update-side body that only the rotor storage types use: the
//! per-variant `update_rotor_*` entries, the GPU encode and ring-sync
//! helpers, the chunk appenders and the materialise-tail path. The
//! `KvStorage` dispatch and the helpers with more than one family caller stay
//! in [`super::update`].

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::storage::{
    KvStorage, QuantK, QuantRotorK, QuantRotorV, QuantV, RotorBlocks, RotorKBlocks,
};

use super::helpers::{array_to_f32_vec, f32_vec_to_array, storage_variant_name};
use super::update::{
    accumulated_seq, b_kv_h_new_seq, bump_ring_k_shape, collapse_group_norms_to_token,
    head_dim_from_shape, is_ring_only_append, layer_idx_u32, packed_k_chunk_seq_major,
    storage_mismatch, PackedKEncodedGpu, RingFeed,
};
use super::KvCache;

/// Feed the legacy rotor symmetric / asymmetric `update_*` entries pass.
///
/// These dequantize the whole prefix on the same step, so the CPU blocks must
/// carry everything and the ring is dropped. Named rather than repeated at each
/// call site because the "blocks are the only copy" property that follows from
/// it is what makes a mid-block truncation unrecoverable there — see
/// `crate::storage::truncate_plan`.
const LEGACY_ROTOR_SYM_FEED: RingFeed = RingFeed::Skip;

/// Feed the legacy rotor K-only `update_*` entries pass.
///
/// Unlike the sym path these keep the ring, so a ring-only tail survives a
/// fallback step; the CPU block is pushed alongside because the same step
/// dequantizes.
const LEGACY_ROTOR_K_ONLY_FEED: RingFeed = RingFeed::Maintain;

/// GPU-encode one rotor K/V chunk, returning the raw kernel arrays plus the
/// `(n_tokens_total, n_groups)` geometry the CPU download and ring feed both
/// need. Shared by the block-retaining and ring-only encode wrappers.
pub(super) fn rotor_gpu_encode_arrays(
    new_kv: &Array,
    new_shape: &[i32],
    rotors: &[f32],
    bits: u8,
) -> Result<(Array, Array, Array, usize, usize)> {
    if new_shape.len() != 4 {
        return Err(Error::Mlx(format!(
            "rotor_gpu_encode_block: expected 4D new_shape, got {new_shape:?}"
        )));
    }
    // `.get()` rather than `[]` — rank-4 verified immediately above.
    let dim = |i: usize| -> Result<usize> {
        new_shape.get(i).map(|&d| d as usize).ok_or_else(|| {
            Error::Mlx(format!(
                "rotor_gpu_encode_block: shape len {} mismatched rank-4 guard \
                 (internal invariant)",
                new_shape.len()
            ))
        })
    };
    let b = dim(0)?;
    let kv_h = dim(1)?;
    let s = dim(2)?;
    let head_dim = dim(3)?;
    let n_tokens_total = b * kv_h * s;
    let n_groups = head_dim.div_ceil(crate::rotorquant::ROTOR3_GROUP_SIZE);

    let rotors_arr = crate::rotorquant_msl::rotor_table_to_array(rotors)?;

    let (codes_arr, scales_arr, norms_arr) = if bits == crate::rotorquant::ROTOR3_BITS {
        crate::rotorquant_msl::rotor_quantize_v3_gpu(new_kv, &rotors_arr, head_dim, Device::Gpu)?
    } else {
        crate::rotorquant_msl::rotor_quantize_v4_gpu(new_kv, &rotors_arr, head_dim, Device::Gpu)?
    };
    Ok((codes_arr, scales_arr, norms_arr, n_tokens_total, n_groups))
}
/// [`rotor_gpu_encode_block`] that also hands back the GPU arrays.
///
/// Used by the CPU-authoritative append paths (prefill, and the non-fused
/// decode fallback): the `RotorBlocks` is the source of truth for `dequant()`
/// and the SSD spill/hydrate round-trip, while the K-side ring feed reuses the
/// pre-download GPU arrays so keeping the store GPU-resident costs no extra
/// encode work.
///
/// The fused decode path does **not** use this — it uses
/// [`rotor_gpu_encode_ring_only`], which skips the per-step host download
/// entirely (a ring-only tail; the blocks are rebuilt from the ring on demand
/// when `dequant()` / an SSD spill actually needs them).
pub(super) fn rotor_gpu_encode_block_retaining(
    new_kv: &Array,
    new_shape: &[i32],
    rotors: &[f32],
    bits: u8,
) -> Result<(RotorBlocks, PackedKEncodedGpu)> {
    let (codes_arr, scales_arr, norms_arr, n_tokens_total, n_groups) =
        rotor_gpu_encode_arrays(new_kv, new_shape, rotors, bits)?;

    let head_dim = new_shape.get(3).copied().unwrap_or(0).max(0) as usize;
    let (codes, scales, norms) = crate::rotorquant_msl::rotor_gpu_outputs_to_cpu(
        &codes_arr,
        &scales_arr,
        &norms_arr,
        n_tokens_total,
        n_groups,
        crate::rotorquant::row_words_for(head_dim, bits),
    )?;

    // The encode kernel emits norms **per group** (`[n_tokens * n_groups]`,
    // the same per-token L2 replicated across a token's groups);
    // `rotor_gpu_outputs_to_cpu` collapses that to per-token by taking every
    // `n_groups`-th element. The ring stores the per-token form the decode
    // kernel indexes (`norms[tok_idx]`), so collapse it here too — on the GPU,
    // to keep the ring feed off the host.
    let norms_per_token = collapse_group_norms_to_token(&norms_arr, n_tokens_total, n_groups)?;

    Ok((
        RotorBlocks {
            codes,
            scales,
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
/// GPU-encode one rotor K chunk for the **ring-only** append path: feed the GPU
/// ring without paying the per-step host download that
/// [`rotor_gpu_encode_block_retaining`] does.
///
/// The fused decode kernel reads the ring, never the CPU `RotorBlocks`, so
/// downloading and materialising a block per decode step is pure host work in
/// the path whose purpose is removing host work. Skipping it leaves the ring
/// the sole source of truth for the decode tail; the CPU blocks are rebuilt
/// from the ring on demand at a `dequant()` / SSD-spill boundary
/// (`synced_rotor_k_blocks`), and the `blocks`-vs-`shape[2]` invariant is
/// enforced loudly there — never zero-padded.
pub(super) fn rotor_gpu_encode_ring_only(
    new_kv: &Array,
    new_shape: &[i32],
    rotors: &[f32],
    bits: u8,
) -> Result<PackedKEncodedGpu> {
    let (codes_arr, scales_arr, norms_arr, n_tokens_total, n_groups) =
        rotor_gpu_encode_arrays(new_kv, new_shape, rotors, bits)?;
    // Collapse per-group norms to the per-token form the ring stores (GPU-side,
    // no host round-trip).
    let norms_per_token = collapse_group_norms_to_token(&norms_arr, n_tokens_total, n_groups)?;
    Ok(PackedKEncodedGpu {
        codes: codes_arr,
        scales: scales_arr,
        norms: norms_per_token,
    })
}
/// Ensure `vs.rotors` is initialised for `head_dim` (mirrors the lazy-init
/// branch inside [`QuantRotorV::append`]).
///
/// The CPU `append` lazy-inits the rotor table on first call; the GPU path
/// bypasses that, so the helpers below seed the table once before dispatch.
///
/// The group size is the same at every rotor width, so this reads the same
/// constant whatever `BITS` is.
pub(super) fn ensure_rotor_v_table<const BITS: u8>(vs: &mut QuantRotorV<BITS>, head_dim: usize) {
    if vs.rotors.is_empty() {
        let n_groups = head_dim.div_ceil(crate::rotorquant::ROTOR3_GROUP_SIZE);
        vs.rotors = crate::clifford::make_rotor_table(vs.layer_idx, vs.head_idx, n_groups);
    }
}
/// The CPU `QuantRotorK::append` couples rotor-table init with QJL
/// projection-matrix init under a single `if self.rotors.is_empty()` guard.
/// The GPU encode path bypasses that `append`, so seed both here too —
/// otherwise a mid-run CPU fallback after a GPU first-chunk would find
/// `rotors` non-empty, skip QJL init, and silently encode without QJL. We
/// always seed when `rotor_qjl_enabled()` is true; the GPU encode itself
/// ignores the JL projection, so this is harmless on the pure-GPU path.
pub(super) fn ensure_rotor_k_table<const BITS: u8>(ks: &mut QuantRotorK<BITS>, head_dim: usize) {
    if ks.rotors.is_empty() {
        let n_groups = head_dim.div_ceil(crate::rotorquant::ROTOR3_GROUP_SIZE);
        ks.rotors = crate::clifford::make_rotor_table(ks.layer_idx, ks.head_idx, n_groups);
        if crate::rotor_qjl::rotor_qjl_enabled() && ks.qjl_s_matrix.is_none() {
            ks.qjl_s_matrix = Some(crate::rotorquant::make_qjl_projection(head_dim));
        }
    }
}
/// Push a pre-built [`RotorBlocks`] onto a [`QuantRotorV`] buffer and update
/// `vs.shape` the same way [`QuantRotorV::append`] does.
#[allow(
    clippy::indexing_slicing,
    reason = "vs.shape rank-4 guard above each indexing site; new_shape rank validated by upstream encoder helper"
)]
pub(super) fn push_rotor_v_block<const BITS: u8>(
    vs: &mut QuantRotorV<BITS>,
    block: RotorBlocks,
    new_shape: &[i32],
) {
    vs.blocks.push(block);
    if vs.shape.len() != 4 || vs.shape[0] == 0 {
        vs.shape = new_shape.to_vec();
    } else {
        vs.shape[2] += new_shape[2];
    }
}
/// Push a pre-built [`RotorBlocks`] onto a [`QuantRotorK`] buffer (QJL OFF
/// path — caller has already verified [`crate::rotor_qjl::rotor_qjl_enabled`]
/// is `false`).
pub(super) fn push_rotor_k_block<const BITS: u8>(
    ks: &mut QuantRotorK<BITS>,
    block: RotorBlocks,
    new_shape: &[i32],
) {
    ks.blocks.push(RotorKBlocks {
        codes: block.codes,
        scales: block.scales,
        norms: block.norms,
        qjl_codes: Vec::new(),
        qjl_norms: Vec::new(),
        n_tokens: block.n_tokens,
    });
    bump_ring_k_shape(&mut ks.shape, new_shape);
}
/// V-side convenience wrapper: GPU-encode + push onto a [`QuantRotorV`]
/// buffer at the store's own width. Lazy-inits the rotor table on first call.
///
/// `feed` decides whether the GPU ring is maintained — see [`RingFeed`]. Only
/// the symmetric codecs have a kernel that reads the V ring; the V-only rotor
/// variants pass `Skip`.
///
/// `max_seq` is the window the cache is currently provisioned for, read from the
/// active `KvStorage` variant by the caller and forwarded to the ring — same
/// contract as the K-side sibling. Ignored when `feed` is `Skip`.
///
/// The chunk is reordered head-major -> sequence-major before encoding, exactly
/// as the CPU `QuantRotorV::append` and the K GPU path already do: the ring and
/// `dequant()` both read sequence-major, so a multi-token chunk with `kv_h > 1`
/// must not be encoded head-major. For the decode step (`S == 1`) the reorder is
/// the identity.
pub(super) fn rotor_gpu_append_into_v_blocks<const BITS: u8>(
    vs: &mut QuantRotorV<BITS>,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim =
        head_dim_from_shape(new_shape, &format!("rotor{BITS}_gpu_append_into_v_blocks"))?;
    ensure_rotor_v_table(vs, head_dim);
    let seq_major = packed_k_chunk_seq_major(new_v, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        // Ring-only tail: feed the GPU ring, advance `shape[2]`, and skip the
        // per-step host download + CPU block push. The ring is the source of
        // truth for the decode tail; the blocks are rebuilt on demand at a
        // `dequant()` / SSD-spill boundary (`synced_rotor_v_blocks`). Mirror of
        // the K-side `rotor_gpu_append_into_k_blocks`.
        let gpu = rotor_gpu_encode_ring_only(&seq_major, new_shape, &vs.rotors, BITS)?;
        rotor_v_sync_ring(
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
    // Block path: prefill / non-fused decode (Maintain) and the V-only variants
    // (Skip). A ring-only feed that reaches here is a `b > 1` chunk — normalise
    // it to Maintain so the shared ring feeder clears the ring for the
    // un-representable batch and the CPU block carries the data.
    materialize_rotor_v_ring_tail(vs, device)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = rotor_gpu_encode_block_retaining(&seq_major, new_shape, &vs.rotors, BITS)?;
    rotor_v_sync_ring(vs, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    push_rotor_v_block(vs, block, new_shape);
    Ok(())
}
/// Reconcile a pre-existing ring-only decode tail into `vs.blocks` before a
/// block-path append — mirror of [`materialize_rotor_k_ring_tail`]. No-op when
/// `blocks` already cover `shape[2]` (the common prefill / V-only case — reads
/// no GPU).
pub(super) fn materialize_rotor_v_ring_tail<const BITS: u8>(
    vs: &mut QuantRotorV<BITS>,
    device: Device,
) -> Result<()> {
    if !vs.gpu.is_allocated() {
        return Ok(());
    }
    let rebuilt = match crate::storage::synced_rotor_v_blocks(
        &vs.blocks, &vs.shape, &vs.gpu, vs.bits, device,
    )? {
        std::borrow::Cow::Owned(full) => Some(full),
        std::borrow::Cow::Borrowed(_) => None,
    };
    if let Some(full) = rebuilt {
        vs.blocks = full;
    }
    Ok(())
}
/// V-side mirror of [`rotor_k_sync_ring`] for [`QuantRotorV`].
///
/// Same invariant, same `b > 1` skip, same self-healing re-seed — the ring type
/// and its contract are axis-agnostic. Only ever receives `Maintain` (ring-only
/// tail and block path both feed the ring) or `Skip` (V-only), so anything other
/// than `Maintain` clears.
pub(super) fn rotor_v_sync_ring<const BITS: u8>(
    vs: &mut QuantRotorV<BITS>,
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
    // Feed BEFORE `push_rotor_v_block` — the push bumps `vs.shape[2]`, and the
    // ring append needs `prev_seq`, the length before this chunk.
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
/// Keep the GPU ring consistent with the CPU blocks for one append.
///
/// **Invariant: the ring either tracks `blocks` exactly, or it does not exist.**
/// A stale ring — blocks grown, ring not — is the dangerous state: the next
/// `gpu_append` would take `prev_seq` from the (longer) `shape` and write past
/// the ring's filled region, leaving the gap zeroed and attention silently
/// wrong. So a skipped feed *clears*; it never just leaves the ring behind. A
/// cleared ring is re-seeded from `blocks` on the next maintained append, so
/// this is self-healing rather than a one-way door.
///
/// `b > 1` is a skip: [`crate::storage::QuantKGpuRing`]'s per-step stride is
/// `kv_h * n_groups` and does not interleave batch, so a batched chunk cannot be
/// laid into it. The CPU blocks (which do handle `b > 1`) stay the source of
/// truth and the flash dispatcher's own `b == 1` gate keeps the kernel away.
pub(super) fn rotor_k_sync_ring<const BITS: u8>(
    ks: &mut QuantRotorK<BITS>,
    gpu: &PackedKEncodedGpu,
    feed: RingFeed,
    new_shape: &[i32],
    head_dim: usize,
    max_seq: i32,
    device: Device,
) -> Result<()> {
    let (b, kv_h, new_seq) = b_kv_h_new_seq(new_shape)?;
    if feed == RingFeed::Skip || b != 1 {
        ks.gpu.clear();
        return Ok(());
    }
    // Feed BEFORE `push_*_k_block` — the push bumps `ks.shape[2]`, and the ring
    // append needs `prev_seq`, the length before this chunk.
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
/// K-side convenience wrapper, QJL off only. Caller MUST check
/// [`crate::rotor_qjl::rotor_qjl_enabled`] returns `false` before invoking
/// this; with QJL enabled, fall back to the CPU [`QuantRotorK::append`] path
/// (the K-side QJL residual is not implemented in MSL — see
/// `rotorquant_msl.rs`).
///
/// `feed` decides whether the GPU ring is maintained — see [`RingFeed`].
/// `max_seq` is the window the cache is currently provisioned for, read from
/// the active `KvStorage` variant by the caller and forwarded to the ring.
pub(super) fn rotor_gpu_append_into_k_blocks<const BITS: u8>(
    ks: &mut QuantRotorK<BITS>,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim =
        head_dim_from_shape(new_shape, &format!("rotor{BITS}_gpu_append_into_k_blocks"))?;
    ensure_rotor_k_table(ks, head_dim);
    // qjl_s_matrix is seeded inside `ensure_rotor_k_table` when `rotor_qjl_enabled()`
    // is true (mid-run CPU fallback after a GPU first-chunk needs it). The GPU
    // encode itself ignores the JL projection — this path is QJL-off-only by
    // dispatcher contract.
    let seq_major = packed_k_chunk_seq_major(new_k, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        // Ring-only tail: feed the GPU ring, advance `shape[2]`, and skip the
        // per-step host download + CPU block push. The ring is the source of
        // truth for the decode tail; the blocks are rebuilt on demand at a
        // `dequant()` / SSD-spill boundary.
        let gpu = rotor_gpu_encode_ring_only(&seq_major, new_shape, &ks.rotors, BITS)?;
        rotor_k_sync_ring(
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
    // Block path: prefill / non-fused decode (Maintain), asym (Skip), and the
    // `b > 1` fallback of a ring-only append. A ring-only feed that reaches here
    // is a `b > 1` chunk — normalise it to Maintain so the shared ring feeder
    // clears the ring for the un-representable batch and the CPU block carries
    // the data.
    materialize_rotor_k_ring_tail(ks, device)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = rotor_gpu_encode_block_retaining(&seq_major, new_shape, &ks.rotors, BITS)?;
    rotor_k_sync_ring(ks, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    push_rotor_k_block(ks, block, new_shape);
    Ok(())
}
/// Reconcile a pre-existing ring-only decode tail into `ks.blocks` before a
/// block-path append, so `blocks` stay a contiguous prefix after the push.
///
/// Both rotor-K mutators must reconcile the ring: `truncate_to` keeps the ring
/// (the tail lives there), and this block-path append materialises the tail
/// into `blocks` so a later push does not leave a non-contiguous
/// prefix-then-gap-then-tail. No-op when `blocks` already cover `shape[2]` (the
/// common prefill / asym case — reads no GPU).
///
/// **Reachability.** This is a live path, not defensive scaffolding. The fused
/// decode entry is gated on `q_seq == 1`, so any multi-token append on a cache
/// that has already run a fused decode step — a speculative verify chunk, or a
/// continuation turn appending prompt tokens against a warm cache — falls
/// through to the legacy `update()` entry and lands here with `blocks` empty
/// (the fused path drops them once the ring is live). The `b > 1` normalisation
/// below reaches it too, but that shape does not occur while the batch dim is
/// fixed per request.
///
/// **Cost.** The rebuild is a full-prefix host readback. It is not additive:
/// every block-path call site dequantizes the whole prefix on the same step, and
/// `dequant` would take exactly this readback itself if `blocks` were left
/// short. Doing it here instead repairs `blocks` once, so the following chunks
/// of the same multi-chunk append pay nothing.
pub(super) fn materialize_rotor_k_ring_tail<const BITS: u8>(
    ks: &mut QuantRotorK<BITS>,
    device: Device,
) -> Result<()> {
    if !ks.gpu.is_allocated() {
        return Ok(());
    }
    let rebuilt = match crate::storage::synced_rotor_k_blocks(
        &ks.blocks, &ks.shape, &ks.gpu, ks.bits, device,
    )? {
        std::borrow::Cow::Owned(full) => Some(full),
        std::borrow::Cow::Borrowed(_) => None,
    };
    if let Some(full) = rebuilt {
        ks.blocks = full;
    }
    Ok(())
}
/// Append `new_k` into a live rotor K-only store's GPU ring (+ CPU blocks) at
/// `BITS`, lazily creating the store on first use. No dequant — the fused
/// rotor flash-decode SDPA path reaches this through
/// [`rotor_k_only_gpu_append`]. `variant` is the storage spelling the caller
/// resolved, used only in the "buffer absent after init" diagnostic.
pub(super) fn rotor_k_only_gpu_append_at<const BITS: u8>(
    k: &mut Option<QuantRotorK<BITS>>,
    max_seq: i32,
    layer_idx: u32,
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
        *k = Some(QuantRotorK::<BITS>::new(init_shape, layer_idx));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx(format!("{variant} K buffer absent after init")));
    };
    rotor_gpu_append_into_k_blocks::<BITS>(
        ks,
        new_k,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    // Ring is now the sole resident store for K — drop the redundant CPU blocks,
    // exactly as the symmetric append does. Without this the prefill prefix stays
    // resident in `blocks` for the whole request on top of the ring that already
    // holds the same packed bytes, inflating the codec's resident KV well above
    // what its own layout costs.
    drop_blocks_when_ring_live_rotor_k(ks);
    Ok(())
}
/// Append `new_k` into whichever rotor K-only store is active — `RotorKOnly3`
/// at 3 bits, `RotorKOnly4` at 4 — lazily creating it on first use. This is
/// the entry point the rotor flash-decode SDPA path uses.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is neither
/// `RotorKOnly3` nor `RotorKOnly4`, and forwards encode / ring errors.
pub(super) fn rotor_k_only_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let layer_idx = layer_idx_u32(cache.layer_idx);
    let (KvStorage::RotorKOnly3 { max_seq, .. } | KvStorage::RotorKOnly4 { max_seq, .. }) =
        &cache.storage
    else {
        return Err(Error::KvStorageMismatch {
            expected: "RotorKOnly3 | RotorKOnly4",
            got: storage_variant_name(&cache.storage),
        });
    };
    let max_seq = *max_seq;

    if let KvStorage::RotorKOnly3 { k, .. } = &mut cache.storage {
        rotor_k_only_gpu_append_at::<3>(
            k,
            max_seq,
            layer_idx,
            "RotorKOnly3",
            new_k,
            new_shape,
            device,
        )
    } else if let KvStorage::RotorKOnly4 { k, .. } = &mut cache.storage {
        rotor_k_only_gpu_append_at::<4>(
            k,
            max_seq,
            layer_idx,
            "RotorKOnly4",
            new_k,
            new_shape,
            device,
        )
    } else {
        // Unreachable: the width read above accepted no other variant.
        Err(Error::KvStorageMismatch {
            expected: "RotorKOnly3 | RotorKOnly4",
            got: storage_variant_name(&cache.storage),
        })
    }
}
/// Drop a rotor K store's CPU blocks once its GPU ring is live — the ring is
/// then the sole resident copy. No-op until the ring is allocated (before the
/// first GPU append) or for any store that never feeds a ring.
pub(super) fn drop_blocks_when_ring_live_rotor_k<const BITS: u8>(ks: &mut QuantRotorK<BITS>) {
    if ks.gpu.is_allocated() {
        ks.blocks.clear();
        ks.blocks.shrink_to_fit();
    }
}
/// Drop a rotor V store's CPU blocks once its GPU ring is live.
pub(super) fn drop_blocks_when_ring_live_rotor_v<const BITS: u8>(vs: &mut QuantRotorV<BITS>) {
    if vs.gpu.is_allocated() {
        vs.blocks.clear();
        vs.blocks.shrink_to_fit();
    }
}
/// Append `new_k` / `new_v` into a live rotor symmetric store's GPU rings
/// (+ CPU blocks) at `BITS`, lazily creating the stores on first use. No
/// dequant on either axis. Both axes maintain their ring: the quant-V kernel
/// reads K's *and* V's. `variant` is the storage spelling the caller resolved,
/// used only in the "buffer absent after init" diagnostics.
#[allow(
    clippy::too_many_arguments,
    reason = "the fused append carries both stores, the ring geometry and the \
              width the caller resolved; a parameter struct would exist for \
              this one call"
)]
pub(super) fn rotor_sym_gpu_append_at<const BITS: u8>(
    k: &mut Option<QuantRotorK<BITS>>,
    v: &mut Option<QuantRotorV<BITS>>,
    max_seq: i32,
    layer_idx: u32,
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
        *k = Some(QuantRotorK::<BITS>::new(init_shape.clone(), layer_idx));
    }
    if v.is_none() {
        *v = Some(QuantRotorV::<BITS>::new(init_shape, max_seq, layer_idx));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx(format!("{variant} K buffer absent after init")));
    };
    rotor_gpu_append_into_k_blocks::<BITS>(
        ks,
        new_k,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    // Ring is now the sole resident store for K — drop the redundant CPU blocks
    // (the prefill prefix, seeded into the ring on the first fused-decode step).
    // See the note in the V append below; this is what turns the codec's ~34%
    // logical compression into a resident-RAM win.
    drop_blocks_when_ring_live_rotor_k(ks);
    let Some(vs) = v.as_mut() else {
        return Err(Error::Mlx(format!("{variant} V buffer absent after init")));
    };
    rotor_gpu_append_into_v_blocks::<BITS>(
        vs,
        new_v,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    // The GPU ring holds the full prefix; the fused decode reads the ring, and
    // `dequant` / SSD spill / clone / truncate rebuild the CPU blocks on demand
    // from it (`synced_rotor_v_blocks`). A later GPU chunk (spec-verify) takes
    // the block path, which materialises the ring tail before its Skip clear, so
    // no ring-clear ever loses data; a CPU-only run never allocates the ring.
    drop_blocks_when_ring_live_rotor_v(vs);
    Ok(())
}
/// Append `new_k` / `new_v` into whichever rotor symmetric store is active —
/// `RotorSym3` at 3 bits, `RotorSym4` at 4 — lazily creating the stores on
/// first use. This is the entry point the rotor symmetric quant-V
/// flash-decode SDPA path uses.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is neither
/// `RotorSym3` nor `RotorSym4`, and forwards encode / ring errors.
pub(super) fn rotor_sym_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let layer_idx = layer_idx_u32(cache.layer_idx);
    let (KvStorage::RotorSym3 { max_seq, .. } | KvStorage::RotorSym4 { max_seq, .. }) =
        &cache.storage
    else {
        return Err(Error::KvStorageMismatch {
            expected: "RotorSym3 | RotorSym4",
            got: storage_variant_name(&cache.storage),
        });
    };
    let max_seq = *max_seq;

    if let KvStorage::RotorSym3 { k, v, .. } = &mut cache.storage {
        rotor_sym_gpu_append_at::<3>(
            k,
            v,
            max_seq,
            layer_idx,
            "RotorSym3",
            new_k,
            new_v,
            new_shape,
            device,
        )
    } else if let KvStorage::RotorSym4 { k, v, .. } = &mut cache.storage {
        rotor_sym_gpu_append_at::<4>(
            k,
            v,
            max_seq,
            layer_idx,
            "RotorSym4",
            new_k,
            new_v,
            new_shape,
            device,
        )
    } else {
        // Unreachable: the width read above accepted no other variant.
        Err(Error::KvStorageMismatch {
            expected: "RotorSym3 | RotorSym4",
            got: storage_variant_name(&cache.storage),
        })
    }
}
/// Decode update for `RotorV3` / `RotorV4`: K = affine q8_0, V = rotor at
/// `BITS` (Cl(3,0) Clifford rotor sandwich + Lloyd-Max codebook).
///
/// Structurally mirrors [`KvCache::update_iso4`] with the V side bound to
/// [`QuantRotorV`] instead of `QuantIsoV4`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Option is Some by construction immediately above this fn body's assignments"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the rotor update bodies carry both stores, the ring geometry and the \
              width the caller resolved; a parameter struct would exist for \
              this one call"
)]
pub(super) fn rotor_v_update<const BITS: u8>(
    k: &mut Option<QuantK>,
    v: &mut Option<QuantRotorV<BITS>>,
    max_seq: i32,
    layer_idx: u32,
    new_k: &Array,
    new_v: &Array,
    device: Device,
) -> Result<(Array, Array)> {
    let new_shape = new_k.shape();

    // K-side: GPU-capable affine q8_0 (same as iso3 / iso4 / K8V4).
    // V-side: routes encode through the rotor MSL kernel when
    // `device == Device::Gpu`; CPU encode remains the fallback. This
    // hot path is shadowed by the warm-TTFT bf16 seed: the GPU encode
    // fires once at exit_prefill (large `new_v` slice), not per decode step.
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
    let ks = k.as_mut().unwrap();
    ks.append(&k_f32, &new_shape, new_k, device, max_seq)?;
    let k_shape = ks.shape.clone();
    let (k_recon_f32, k_arr_opt) = ks.dequantize_choice(device, new_k.dtype())?;
    let k_full = match k_arr_opt {
        Some(arr) => arr,
        None => f32_vec_to_array(&k_recon_f32, &k_shape)?,
    };

    if v.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        // Thread the real model-layer index into the rotor seed so each layer
        // gets a distinct rotor table. The rotor table is deterministic and
        // persists via SSD round-trip.
        *v = Some(QuantRotorV::<BITS>::new(init_shape, max_seq, layer_idx));
    }
    let vs = v.as_mut().unwrap();
    if device == Device::Gpu {
        rotor_gpu_append_into_v_blocks(vs, new_v, &new_shape, device, RingFeed::Skip, max_seq)?;
    } else {
        vs.append(&v_f32, &new_shape)?;
    }
    let v_shape = vs.shape.clone();
    let v_recon_f32 = vs.dequant()?;
    let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

    Ok((k_full, v_full))
}
/// Decode update for `RotorSym3` / `RotorSym4`: K and V both quantize through
/// the rotor codec at `BITS`.
///
/// Structurally mirrors [`KvCache::update_iso3_sym`] with the codec bound to
/// [`QuantRotorK`] / [`QuantRotorV`] instead of the iso K/V types. `variant` is
/// the storage spelling the caller resolved, used only in the "buffer absent
/// after init" diagnostic.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the rotor update bodies carry both stores, the ring geometry and the \
              width the caller resolved; a parameter struct would exist for \
              this one call"
)]
pub(super) fn rotor_sym_update<const BITS: u8>(
    k: &mut Option<QuantRotorK<BITS>>,
    v: &mut Option<QuantRotorV<BITS>>,
    max_seq: i32,
    layer_idx: u32,
    variant: &'static str,
    new_k: &Array,
    new_v: &Array,
    device: Device,
) -> Result<(Array, Array)> {
    let new_shape = new_k.shape();
    // GPU encode for both K and V when device == GPU and the store carries
    // no QJL residual. The QJL decision is sticky to the store (fixed at
    // first append) — a later process-env toggle must not reinterpret bytes
    // already written; read the store's own flag, same as the sdpa fast path
    // and the K-only body. Env is the fallback only before the store exists.
    // With QJL on the K-side falls back to CPU (the GPU kernel cannot
    // replicate the QJL residual — see rotor_fused_qk_msl.rs).
    let store_uses_qjl = match k.as_ref() {
        Some(ks) => ks.use_qjl(),
        None => crate::rotor_qjl::rotor_qjl_enabled(),
    };
    let use_gpu = device == Device::Gpu;
    let gpu_k_ok = use_gpu && !store_uses_qjl;
    let k_f32 = if gpu_k_ok {
        Vec::new()
    } else {
        array_to_f32_vec(new_k, Device::Cpu)?
    };
    let v_f32 = if use_gpu {
        Vec::new()
    } else {
        array_to_f32_vec(new_v, Device::Cpu)?
    };

    if k.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        *k = Some(QuantRotorK::<BITS>::new(init_shape, layer_idx));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx(format!("{variant} K buffer absent after init")));
    };
    if gpu_k_ok {
        rotor_gpu_append_into_k_blocks(
            ks,
            new_k,
            &new_shape,
            device,
            LEGACY_ROTOR_SYM_FEED,
            max_seq,
        )?;
    } else {
        ks.append(&k_f32, &new_shape)?;
    }
    let k_shape = ks.shape.clone();
    let k_recon_f32 = ks.dequant()?;
    let k_full = f32_vec_to_array(&k_recon_f32, &k_shape)?;

    if v.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        *v = Some(QuantRotorV::<BITS>::new(init_shape, max_seq, layer_idx));
    }
    let Some(vs) = v.as_mut() else {
        return Err(Error::Mlx(format!("{variant} V buffer absent after init")));
    };
    if use_gpu {
        rotor_gpu_append_into_v_blocks(vs, new_v, &new_shape, device, RingFeed::Skip, max_seq)?;
    } else {
        vs.append(&v_f32, &new_shape)?;
    }
    let v_shape = vs.shape.clone();
    let v_recon_f32 = vs.dequant()?;
    let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

    Ok((k_full, v_full))
}
/// K side of the `RotorKOnly3` / `RotorKOnly4` decode update: rotor K at
/// `BITS`, returning the reconstructed K. The V side stays bf16 and is the
/// caller's, because it needs `&mut self` (see the entries above).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction"
)]
pub(super) fn rotor_k_only_k_side<const BITS: u8>(
    k: &mut Option<QuantRotorK<BITS>>,
    max_seq: i32,
    layer_idx: u32,
    variant: &'static str,
    new_k: &Array,
    device: Device,
) -> Result<Array> {
    let new_shape = new_k.shape();
    // GPU encode for K when device == GPU AND this store was written without
    // QJL. The QJL decision is sticky to the store — fixed at first append —
    // so a later process-env toggle must not reinterpret bytes already
    // written. Read the store's own flag, the same source the sdpa fused
    // fast path consults; fall back to the env only before the store exists,
    // i.e. the value the store is about to be built with.
    let store_uses_qjl = match k.as_ref() {
        Some(ks) => ks.use_qjl(),
        None => crate::rotor_qjl::rotor_qjl_enabled(),
    };
    let gpu_k_ok = device == Device::Gpu && !store_uses_qjl;
    let k_f32 = if gpu_k_ok {
        Vec::new()
    } else {
        array_to_f32_vec(new_k, Device::Cpu)?
    };

    if k.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        *k = Some(QuantRotorK::<BITS>::new(init_shape, layer_idx));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx(format!("{variant} K buffer absent after init")));
    };
    if gpu_k_ok {
        rotor_gpu_append_into_k_blocks(
            ks,
            new_k,
            &new_shape,
            device,
            LEGACY_ROTOR_K_ONLY_FEED,
            max_seq,
        )?;
    } else {
        ks.append(&k_f32, &new_shape)?;
    }
    let k_shape = ks.shape.clone();
    let k_recon_f32 = ks.dequant()?;
    f32_vec_to_array(&k_recon_f32, &k_shape)
}
/// Decode update for `RotorKAsym3` / `RotorKAsym4`: K is rotor at `BITS`, V is
/// MLX affine `v_bits` (reuses [`QuantV`]).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Option is Some by construction immediately above this fn body's assignments"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the rotor update bodies carry both stores, the ring geometry and the \
              width the caller resolved; a parameter struct would exist for \
              this one call"
)]
pub(super) fn rotor_k_asym_update<const BITS: u8>(
    k: &mut Option<QuantRotorK<BITS>>,
    v: &mut Option<QuantV>,
    max_seq: i32,
    v_bits: u8,
    layer_idx: u32,
    new_k: &Array,
    new_v: &Array,
    device: Device,
) -> Result<(Array, Array)> {
    let new_shape = new_k.shape();
    // Gate the GPU K encode on the store's sticky QJL flag (fixed at first
    // append), not the live env — a later toggle must not reinterpret bytes
    // already written. Env is the fallback only before the store exists.
    let store_uses_qjl = match k.as_ref() {
        Some(ks) => ks.use_qjl(),
        None => crate::rotor_qjl::rotor_qjl_enabled(),
    };
    let gpu_k_ok = device == Device::Gpu && !store_uses_qjl;
    let k_f32 = if gpu_k_ok {
        Vec::new()
    } else {
        array_to_f32_vec(new_k, Device::Cpu)?
    };
    let v_f32 = if device == Device::Gpu {
        Vec::new()
    } else {
        array_to_f32_vec(new_v, device)?
    };

    if k.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        *k = Some(QuantRotorK::<BITS>::new(init_shape, layer_idx));
    }
    let ks = k.as_mut().unwrap();
    if gpu_k_ok {
        rotor_gpu_append_into_k_blocks(
            ks,
            new_k,
            &new_shape,
            device,
            LEGACY_ROTOR_SYM_FEED,
            max_seq,
        )?;
    } else {
        ks.append(&k_f32, &new_shape)?;
    }
    let k_shape = ks.shape.clone();
    let k_recon_f32 = ks.dequant()?;
    let k_full = f32_vec_to_array(&k_recon_f32, &k_shape)?;

    if v.is_none() {
        let mut init_shape = new_shape.clone();
        init_shape[2] = 0;
        *v = Some(QuantV::new_affine_decode(init_shape, v_bits, max_seq));
    }
    let vs = v.as_mut().unwrap();
    vs.append(&v_f32, &new_shape, new_v, device, max_seq)?;
    let v_shape = vs.shape.clone();
    let (v_recon_f32, v_arr_opt) = vs.dequantize_choice(device, new_v.dtype())?;
    let v_full = match v_arr_opt {
        Some(arr) => arr,
        None => f32_vec_to_array(&v_recon_f32, &v_shape)?,
    };

    Ok((k_full, v_full))
}

impl KvCache {
    /// Rotor V decode update — K = affine q8_0, V = rotor (Cl(3,0) Clifford
    /// rotor sandwich + Lloyd-Max codebook) at the code width the active
    /// storage variant carries: `RotorV3` is 3 bits, `RotorV4` is 4.
    ///
    /// The body is [`rotor_v_update`] at that width; this entry resolves the
    /// storage variant and takes the warm-TTFT bf16 shortcut.
    pub(super) fn update_rotor_v(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let layer_idx = layer_idx_u32(self.layer_idx);
        let (KvStorage::RotorV3 { max_seq, .. } | KvStorage::RotorV4 { max_seq, .. }) =
            &self.storage
        else {
            return Err(storage_mismatch("RotorV3 | RotorV4", &self.storage));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        if let KvStorage::RotorV3 { k, v, .. } = &mut self.storage {
            rotor_v_update::<3>(k, v, max_seq, layer_idx, new_k, new_v, device)
        } else if let KvStorage::RotorV4 { k, v, .. } = &mut self.storage {
            rotor_v_update::<4>(k, v, max_seq, layer_idx, new_k, new_v, device)
        } else {
            // Unreachable: the width read above accepted no other variant.
            Err(storage_mismatch("RotorV3 | RotorV4", &self.storage))
        }
    }
    /// Rotor symmetric decode update: K and V both quantize through the rotor
    /// (Cl(3,0) Clifford rotor) codec at the code width the active storage
    /// variant carries — `RotorSym3` is 3 bits, `RotorSym4` is 4. The K side
    /// carries the optional 1-bit QJL residual sideband when
    /// [`crate::rotor_qjl::rotor_qjl_enabled`] is `true` at first append.
    ///
    /// The body is [`rotor_sym_update`] at that width; this entry resolves the
    /// storage variant and takes the warm-TTFT bf16 shortcut.
    pub(super) fn update_rotor_sym(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let layer_idx = layer_idx_u32(self.layer_idx);
        let (KvStorage::RotorSym3 { max_seq, .. } | KvStorage::RotorSym4 { max_seq, .. }) =
            &self.storage
        else {
            return Err(storage_mismatch("RotorSym3 | RotorSym4", &self.storage));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        if let KvStorage::RotorSym3 { k, v, .. } = &mut self.storage {
            rotor_sym_update::<3>(k, v, max_seq, layer_idx, "RotorSym3", new_k, new_v, device)
        } else if let KvStorage::RotorSym4 { k, v, .. } = &mut self.storage {
            rotor_sym_update::<4>(k, v, max_seq, layer_idx, "RotorSym4", new_k, new_v, device)
        } else {
            // Unreachable: the width read above accepted no other variant.
            Err(storage_mismatch("RotorSym3 | RotorSym4", &self.storage))
        }
    }
    /// Rotor K-only decode update. K is rotor at the code width the active
    /// storage variant carries (`RotorKOnly3` is 3 bits, `RotorKOnly4` is 4);
    /// V stays bf16 on `decode_fp16_v`.
    ///
    /// **CRITICAL** (HIGH bug guard): uses
    /// [`Self::update_decode_fp16_v_only`] for the V side, NOT
    /// `update_decode_fp16`. The latter populates `self.decode_fp16_k` as a
    /// side-effect, which causes the `decode_fp16_k.is_some()` early-return
    /// guard to short-circuit the K codec on the *next* decode step (silent
    /// bf16-K regression).
    ///
    /// The K side is [`rotor_k_only_k_side`] at the resolved width.
    pub(super) fn update_rotor_k_only(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let layer_idx = layer_idx_u32(self.layer_idx);
        let (KvStorage::RotorKOnly3 { max_seq, .. } | KvStorage::RotorKOnly4 { max_seq, .. }) =
            &self.storage
        else {
            return Err(Error::KvStorageMismatch {
                expected: "RotorKOnly3 | RotorKOnly4",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;

        let k_full = if let KvStorage::RotorKOnly3 { k, .. } = &mut self.storage {
            rotor_k_only_k_side::<3>(k, max_seq, layer_idx, "RotorKOnly3", new_k, device)?
        } else if let KvStorage::RotorKOnly4 { k, .. } = &mut self.storage {
            rotor_k_only_k_side::<4>(k, max_seq, layer_idx, "RotorKOnly4", new_k, device)?
        } else {
            // Unreachable: the width read above accepted no other variant.
            return Err(Error::KvStorageMismatch {
                expected: "RotorKOnly3 | RotorKOnly4",
                got: storage_variant_name(&self.storage),
            });
        };

        // V-side: bf16 via the V-only helper (must NOT touch decode_fp16_k).
        let v_full = self.update_decode_fp16_v_only(new_v, max_seq, device)?;
        Ok((k_full, v_full))
    }
    /// Rotor asymmetric decode update. K is rotor at the code width the active
    /// storage variant carries (`RotorKAsym3` is 3 bits, `RotorKAsym4` is 4);
    /// V is MLX affine `v_bits` / `v_group_size` (reuses [`QuantV`]).
    ///
    /// Mirrors [`Self::update_rotor_k_only`] for K and [`Self::update_k8v4`]
    /// for V (affine path) on the **seedless** path only.
    ///
    /// **Warm-TTFT.** Unlike `RotorKOnly{3,4}`, these asym variants DO carry
    /// the `decode_fp16_k.is_some()` shortcut (below): once the bf16 seed is
    /// live (always, post-`exit_prefill` — see `exit_prefill`'s
    /// generic seed tail), the entire decode step routes through
    /// [`Self::update_decode_fp16`] and serves **both** K and V from bf16. The
    /// rotor-K and affine-V codecs are quiescent for the whole decode window;
    /// they re-encode only at `exit_prefill` or on a seedless cache. This is
    /// the universal warm-TTFT decode contract documented in
    /// `docs/KV_CACHE.md` §9.6.
    ///
    /// NB: `RotorKOnly{3,4}` (no asym V) is the opposite — that entry has
    /// **no** seed shortcut in its body, so its rotor-K codec runs every
    /// decode step (K-only family). Do not assume the two share K-side decode
    /// semantics.
    ///
    /// The body is [`rotor_k_asym_update`] at the resolved width.
    pub(super) fn update_rotor_k_asym(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let layer_idx = layer_idx_u32(self.layer_idx);
        let (KvStorage::RotorKAsym3 {
            max_seq, v_bits, ..
        }
        | KvStorage::RotorKAsym4 {
            max_seq, v_bits, ..
        }) = &self.storage
        else {
            return Err(Error::KvStorageMismatch {
                expected: "RotorKAsym3 | RotorKAsym4",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;
        let v_bits = *v_bits;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        if let KvStorage::RotorKAsym3 { k, v, .. } = &mut self.storage {
            rotor_k_asym_update::<3>(k, v, max_seq, v_bits, layer_idx, new_k, new_v, device)
        } else if let KvStorage::RotorKAsym4 { k, v, .. } = &mut self.storage {
            rotor_k_asym_update::<4>(k, v, max_seq, v_bits, layer_idx, new_k, new_v, device)
        } else {
            // Unreachable: the width read above accepted no other variant.
            Err(Error::KvStorageMismatch {
                expected: "RotorKAsym3 | RotorKAsym4",
                got: storage_variant_name(&self.storage),
            })
        }
    }
}

#[cfg(test)]
#[path = "ring_feed_routing_tests.rs"]
mod ring_feed_routing_tests;
