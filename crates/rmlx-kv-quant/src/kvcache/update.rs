//! Update paths: `update`, prefill, GPU state management, and storage-specific appenders.
#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::too_many_lines
)]

use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::storage::{
    iso_n_groups_for, IsoBlocks, KvStorage, QuantIsoK3, QuantIsoK4, QuantIsoV3, QuantIsoV4, QuantK,
    QuantKTurbo3, QuantKTurbo4, QuantPlanarK, QuantPlanarV, QuantRotorV3, QuantRotorV4, QuantV,
    RotorBlocks, RotorKBlocks, ISO_K3_BITS, ISO_K4_BITS,
};
use crate::turbo_flash_msl::{turbo_flash_sdpa, turbo_flash_should_run};
use crate::KvQuant;

use super::helpers::{array_to_f32_vec, arrays_to_f32, f32_vec_to_array, storage_variant_name};
use super::KvCache;

/// Narrow `layer_idx` (`usize`) to `u32` for rotor-seed APIs.
///
/// Centralizes the cast so the `cast_possible_truncation` lint allow is
/// scoped to a single spot rather than the whole module. `layer_idx` is a
/// model-layer count (≤ thousands), well within `u32::MAX`. Taking the value
/// by argument (not via `&self`) avoids partial-borrow conflicts at call
/// sites that already hold a mutable borrow on `self.storage`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "layer_idx is a model-layer count (≤ thousands), fits u32"
)]
#[inline]
fn layer_idx_u32(layer_idx: usize) -> u32 {
    layer_idx as u32
}

/// Model-agnostic bf16 floor for the unquantised (`KvQuant::None`) / warm-TTFT
/// cache store boundary.
///
/// The `decode_fp16_k/v` mirror is bf16 by contract (that is what every sibling
/// MLX backend stores for unquantised KV, and the only sensible value), but the
/// incoming K/V inherit whatever dtype the model's attention stream happened to
/// produce. A single f32 scalar leaking into that stream upstream silently
/// promotes K/V to f32 and doubles resident KV. Casting here at the store
/// boundary caps the memory damage regardless of arch — defence-in-depth on top
/// of the per-arch source fixes, not a replacement for them (upstream f32
/// compute stays f32; this only floors what the cache *stores*).
///
/// Idempotent: when the input is already bf16 (the steady state after the
/// per-arch fixes) this returns `None` after a cheap dtype check — no `astype`
/// launch on the decode hot path. Only a non-bf16 input materialises the cast.
/// Callers fold the result with `result.as_ref().unwrap_or(new_k)`.
#[inline]
fn cast_store_bf16(arr: &Array, device: Device) -> Result<Option<Array>> {
    if arr.dtype() == Dtype::Bf16 {
        Ok(None)
    } else {
        Ok(Some(arr.astype(Dtype::Bf16, device)?))
    }
}

/// Push a pre-built [`IsoBlocks`] onto a [`QuantIsoK4`] buffer and update
/// `ks.shape` the same way [`QuantIsoK4::append`] does.
#[allow(
    clippy::indexing_slicing,
    reason = "ks.shape rank-4 guard above each indexing site; new_shape rank validated by upstream encoder helper"
)]
fn push_iso4_k_block(ks: &mut QuantIsoK4, block: IsoBlocks, new_shape: &[i32]) {
    ks.blocks.push(block);
    if ks.shape.len() != 4 || ks.shape[0] == 0 {
        ks.shape = new_shape.to_vec();
    } else {
        ks.shape[2] += new_shape[2];
    }
}

/// Mirror of [`iso3_gpu_append_into_k_blocks`] for the iso4 codec — see it for
/// the `feed` contract and why the chunk is reordered to sequence-major.
fn iso4_gpu_append_into_k_blocks(
    ks: &mut QuantIsoK4,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim = head_dim_from_shape(new_shape, "iso4_gpu_append_into_k_blocks")?;
    let seq_major = packed_k_chunk_seq_major(new_k, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        // Ring-only tail — see [`iso3_gpu_append_into_k_blocks`].
        let gpu = iso_gpu_encode_ring_only(&seq_major, new_shape, ISO_K4_BITS)?;
        iso4_sync_ring(
            ks,
            &gpu,
            RingFeed::Maintain,
            new_shape,
            head_dim,
            max_seq,
            device,
        )?;
        bump_rotor_k_shape(&mut ks.shape, new_shape);
        return Ok(());
    }
    ks.reconcile_ring(device, crate::storage::RingDisposition::Keep)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = iso_gpu_encode_block_retaining(&seq_major, new_shape, ISO_K4_BITS)?;
    iso4_sync_ring(ks, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    push_iso4_k_block(ks, block, new_shape);
    Ok(())
}

// ── Iso3 MSL helpers (mirror of iso4) ────────────────────────────────────────
//
// The iso3 K-side encode itself lives in `iso_gpu_encode_block_retaining`,
// which serves both bit widths and hands back the GPU arrays the ring needs.

// `push_iso3_v_block` was removed along with the V-side helper —
// `QuantIsoV3::append_gpu` now owns shape bookkeeping for the V path. The
// K-side equivalent (`push_iso3_k_block`) is still in use below.

/// Push a pre-built [`IsoBlocks`] onto a [`QuantIsoK3`] buffer and update
/// `ks.shape` the same way [`QuantIsoK3::append`] does.
#[allow(
    clippy::indexing_slicing,
    reason = "ks.shape rank-4 guard above each indexing site; new_shape rank validated by upstream encoder helper"
)]
fn push_iso3_k_block(ks: &mut QuantIsoK3, block: IsoBlocks, new_shape: &[i32]) {
    ks.blocks.push(block);
    if ks.shape.len() != 4 || ks.shape[0] == 0 {
        ks.shape = new_shape.to_vec();
    } else {
        ks.shape[2] += new_shape[2];
    }
}

// The V-side iso3 helper
// `iso3_gpu_append_into_blocks(vs, new_v, &new_shape)` was inlined into
// `QuantIsoV3::append_gpu` (which also writes the GPU-resident mirror). The
// K-side wrapper below remains in use.

/// Append one iso3 K chunk: GPU encode → CPU blocks (+ the GPU ring when
/// `feed` is [`RingFeed::Maintain`]).
///
/// Used by [`KvCache::update_iso_k_only_3`] and the K-side of
/// [`KvCache::update_iso3_sym`] (both `RingFeed::Skip` — they dequant the whole
/// prefix anyway), and by [`iso3_k_only_gpu_append`] on the flash-decode path
/// (`RingFeed::Maintain`).
///
/// The chunk is reordered to sequence-major before encoding. That is not
/// optional: `QuantIsoK3::append` (the CPU path) transposes, and
/// `QuantIsoK3::dequant{,_gpu}` transpose *back* — so encoding head-major here
/// would hand `dequant` head-major blocks it reads as sequence-major and
/// scramble heads for any `kv_h > 1` chunk longer than one token. For the
/// decode step (`S == 1`) the reorder is the identity.
fn iso3_gpu_append_into_k_blocks(
    ks: &mut QuantIsoK3,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim = head_dim_from_shape(new_shape, "iso3_gpu_append_into_k_blocks")?;
    let seq_major = packed_k_chunk_seq_major(new_k, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        // Ring-only tail: feed the GPU ring, advance `shape[2]`, and skip the
        // per-step host download + CPU block push. The ring is the source of
        // truth for the decode tail; the blocks are rebuilt on demand at a
        // `dequant()` / SSD-spill boundary (`synced_iso_v_blocks`).
        let gpu = iso_gpu_encode_ring_only(&seq_major, new_shape, ISO_K3_BITS)?;
        iso3_sync_ring(
            ks,
            &gpu,
            RingFeed::Maintain,
            new_shape,
            head_dim,
            max_seq,
            device,
        )?;
        bump_rotor_k_shape(&mut ks.shape, new_shape);
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
    let (block, gpu) = iso_gpu_encode_block_retaining(&seq_major, new_shape, ISO_K3_BITS)?;
    iso3_sync_ring(ks, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    push_iso3_k_block(ks, block, new_shape);
    Ok(())
}

// ── Rotor3 / rotor4 MSL helpers ───────────────────────────────────────────────

/// The GPU-resident `(codes, scales, norms)` triple a packed-K encode produced
/// (rotor or iso), retained so a caller can push it straight into a GPU ring
/// instead of re-uploading the downloaded CPU copy.
///
/// `norms` is always the **per-token** form the decode kernels index
/// (`norms[tok_idx]`), not the per-group form the encode kernels emit.
struct PackedKEncodedGpu {
    codes: Array,
    scales: Array,
    norms: Array,
}

/// GPU-encode one rotor K/V chunk, returning the raw kernel arrays plus the
/// `(n_tokens_total, n_groups)` geometry the CPU download and ring feed both
/// need. Shared by the block-retaining and ring-only encode wrappers.
fn rotor_gpu_encode_arrays(
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
fn rotor_gpu_encode_block_retaining(
    new_kv: &Array,
    new_shape: &[i32],
    rotors: &[f32],
    bits: u8,
) -> Result<(RotorBlocks, PackedKEncodedGpu)> {
    let (codes_arr, scales_arr, norms_arr, n_tokens_total, n_groups) =
        rotor_gpu_encode_arrays(new_kv, new_shape, rotors, bits)?;

    let (codes, scales, norms) = crate::rotorquant_msl::rotor_gpu_outputs_to_cpu(
        &codes_arr,
        &scales_arr,
        &norms_arr,
        n_tokens_total,
        n_groups,
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
fn rotor_gpu_encode_ring_only(
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

/// Collapse a packed-K encode kernel's per-group norms (`[n_tokens, n_groups]`,
/// each row a repeat of that token's L2) to the per-token `[n_tokens]` form.
///
/// Shared by the rotor and iso encode paths — both kernels emit the per-group
/// form and both rings store per-token. GPU-side equivalent of the
/// `norms_per_group[tok * n_groups]` pick in
/// [`crate::rotorquant_msl::rotor_gpu_outputs_to_cpu`] /
/// [`crate::isoquant_msl::iso3_gpu_outputs_to_cpu`] — column 0 of each row.
fn collapse_group_norms_to_token(
    norms_per_group: &Array,
    n_tokens: usize,
    n_groups: usize,
) -> Result<Array> {
    let n_tokens_i32 = i32::try_from(n_tokens).map_err(|_| {
        Error::Quant(format!(
            "packed-K norms: n_tokens={n_tokens} exceeds i32::MAX"
        ))
    })?;
    let n_groups_i32 = i32::try_from(n_groups).map_err(|_| {
        Error::Quant(format!(
            "packed-K norms: n_groups={n_groups} exceeds i32::MAX"
        ))
    })?;
    norms_per_group
        .reshape(&[n_tokens_i32, n_groups_i32], Device::Gpu)?
        .slice(&[0, 0], &[n_tokens_i32, 1], &[1, 1], Device::Gpu)?
        .reshape(&[n_tokens_i32], Device::Gpu)
}

/// Ensure `vs.rotors` is initialised for `head_dim` (mirrors the lazy-init
/// branch inside [`QuantRotorV3::append`] / [`QuantRotorV4::append`]).
///
/// The CPU `append` lazy-inits the rotor table on first call; the GPU path
/// bypasses that, so the helpers below seed the table once before dispatch.
fn ensure_v3_rotors(vs: &mut QuantRotorV3, head_dim: usize) {
    if vs.rotors.is_empty() {
        let n_groups = head_dim.div_ceil(crate::rotorquant::ROTOR3_GROUP_SIZE);
        vs.rotors = crate::clifford::make_rotor_table(vs.layer_idx, vs.head_idx, n_groups);
    }
}

fn ensure_v4_rotors(vs: &mut QuantRotorV4, head_dim: usize) {
    if vs.rotors.is_empty() {
        let n_groups = head_dim.div_ceil(crate::rotorquant::ROTOR3_GROUP_SIZE);
        vs.rotors = crate::clifford::make_rotor_table(vs.layer_idx, vs.head_idx, n_groups);
    }
}

/// The CPU `QuantRotorK3::append` couples rotor-table init with QJL
/// projection-matrix init under a single `if self.rotors.is_empty()` guard.
/// The GPU encode path bypasses that `append`, so seed both here too —
/// otherwise a mid-run CPU fallback after a GPU first-chunk would find
/// `rotors` non-empty, skip QJL init, and silently encode without QJL. We
/// always seed when `rotor_qjl_enabled()` is true; the GPU encode itself
/// ignores the JL projection, so this is harmless on the pure-GPU path.
fn ensure_k3_rotors(ks: &mut crate::storage::QuantRotorK3, head_dim: usize) {
    if ks.rotors.is_empty() {
        let n_groups = head_dim.div_ceil(crate::rotorquant::ROTOR3_GROUP_SIZE);
        ks.rotors = crate::clifford::make_rotor_table(ks.layer_idx, ks.head_idx, n_groups);
        if crate::rotor_qjl::rotor_qjl_enabled() && ks.qjl_s_matrix.is_none() {
            ks.qjl_s_matrix = Some(crate::rotorquant::make_qjl_projection(head_dim));
        }
    }
}

/// See [`ensure_k3_rotors`] for the QJL-init coupling rationale.
fn ensure_k4_rotors(ks: &mut crate::storage::QuantRotorK4, head_dim: usize) {
    if ks.rotors.is_empty() {
        let n_groups = head_dim.div_ceil(crate::rotorquant::ROTOR3_GROUP_SIZE);
        ks.rotors = crate::clifford::make_rotor_table(ks.layer_idx, ks.head_idx, n_groups);
        if crate::rotor_qjl::rotor_qjl_enabled() && ks.qjl_s_matrix.is_none() {
            ks.qjl_s_matrix = Some(crate::rotorquant::make_qjl_projection(head_dim));
        }
    }
}

/// Push a pre-built [`RotorBlocks`] onto a [`QuantRotorV3`] buffer and update
/// `vs.shape` the same way [`QuantRotorV3::append`] does.
#[allow(
    clippy::indexing_slicing,
    reason = "vs.shape rank-4 guard above each indexing site; new_shape rank validated by upstream encoder helper"
)]
fn push_rotor3_v_block(vs: &mut QuantRotorV3, block: RotorBlocks, new_shape: &[i32]) {
    vs.blocks.push(block);
    if vs.shape.len() != 4 || vs.shape[0] == 0 {
        vs.shape = new_shape.to_vec();
    } else {
        vs.shape[2] += new_shape[2];
    }
}

#[allow(
    clippy::indexing_slicing,
    reason = "vs.shape rank-4 guard above each indexing site; new_shape rank validated by upstream encoder helper"
)]
fn push_rotor4_v_block(vs: &mut QuantRotorV4, block: RotorBlocks, new_shape: &[i32]) {
    vs.blocks.push(block);
    if vs.shape.len() != 4 || vs.shape[0] == 0 {
        vs.shape = new_shape.to_vec();
    } else {
        vs.shape[2] += new_shape[2];
    }
}

/// Advance a rotor K store's accumulated `shape` by one appended chunk,
/// matching the `QuantRotorK{3,4}::append` bookkeeping. Shared by the
/// block-pushing and ring-only append paths so `shape[2]` advances identically
/// whether or not a CPU block is materialised.
#[allow(
    clippy::indexing_slicing,
    reason = "shape rank-4 guard above each indexing site; new_shape rank validated by upstream encoder helper"
)]
fn bump_rotor_k_shape(shape: &mut Vec<i32>, new_shape: &[i32]) {
    if shape.len() != 4 || shape[0] == 0 {
        *shape = new_shape.to_vec();
    } else {
        shape[2] += new_shape[2];
    }
}

/// Push a pre-built [`RotorBlocks`] onto a [`QuantRotorK3`] buffer (QJL OFF
/// path — caller has already verified [`crate::rotor_qjl::rotor_qjl_enabled`]
/// is `false`).
fn push_rotor3_k_block(
    ks: &mut crate::storage::QuantRotorK3,
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
    bump_rotor_k_shape(&mut ks.shape, new_shape);
}

fn push_rotor4_k_block(
    ks: &mut crate::storage::QuantRotorK4,
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
    bump_rotor_k_shape(&mut ks.shape, new_shape);
}

/// Recover `head_dim` from a 4-D new_shape without silently swallowing
/// malformed shapes (previous `.get(3).unwrap_or(0)` pattern). Returns
/// `Error::Mlx` on rank mismatch.
fn head_dim_from_shape(new_shape: &[i32], ctx: &str) -> Result<usize> {
    if new_shape.len() != 4 {
        return Err(Error::Mlx(format!(
            "{ctx}: expected 4D new_shape, got {new_shape:?}"
        )));
    }
    // `.get(3)` rather than `[3]` to avoid the `clippy::indexing_slicing`
    // allow; rank-4 guard above guarantees the value is present.
    let d = new_shape.get(3).copied().ok_or_else(|| {
        Error::Mlx(format!(
            "{ctx}: shape len {} mismatched rank-4 guard (internal invariant)",
            new_shape.len()
        ))
    })?;
    Ok(d as usize)
}

/// V-side convenience wrapper: GPU-encode + push onto a [`QuantRotorV3`]
/// buffer. Lazy-inits the rotor table on first call.
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
/// as the CPU `QuantRotorV3::append` and the K GPU path already do: the ring and
/// `dequant()` both read sequence-major, so a multi-token chunk with `kv_h > 1`
/// must not be encoded head-major. For the decode step (`S == 1`) the reorder is
/// the identity.
fn rotor3_gpu_append_into_blocks(
    vs: &mut QuantRotorV3,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim = head_dim_from_shape(new_shape, "rotor3_gpu_append_into_blocks")?;
    ensure_v3_rotors(vs, head_dim);
    let seq_major = packed_k_chunk_seq_major(new_v, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        // Ring-only tail: feed the GPU ring, advance `shape[2]`, and skip the
        // per-step host download + CPU block push. The ring is the source of
        // truth for the decode tail; the blocks are rebuilt on demand at a
        // `dequant()` / SSD-spill boundary (`synced_rotor_v_blocks`). Mirror of
        // the K-side `rotor3_gpu_append_into_k_blocks`.
        let gpu = rotor_gpu_encode_ring_only(
            &seq_major,
            new_shape,
            &vs.rotors,
            crate::rotorquant::ROTOR3_BITS,
        )?;
        rotor3_v_sync_ring(
            vs,
            &gpu,
            RingFeed::Maintain,
            new_shape,
            head_dim,
            max_seq,
            device,
        )?;
        bump_rotor_k_shape(&mut vs.shape, new_shape);
        return Ok(());
    }
    // Block path: prefill / non-fused decode (Maintain) and the V-only variants
    // (Skip). A ring-only feed that reaches here is a `b > 1` chunk — normalise
    // it to Maintain so the shared ring feeder clears the ring for the
    // un-representable batch and the CPU block carries the data.
    materialize_rotor_v3_ring_tail(vs, device)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = rotor_gpu_encode_block_retaining(
        &seq_major,
        new_shape,
        &vs.rotors,
        crate::rotorquant::ROTOR3_BITS,
    )?;
    rotor3_v_sync_ring(vs, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    push_rotor3_v_block(vs, block, new_shape);
    Ok(())
}

/// Reconcile a pre-existing ring-only decode tail into `vs.blocks` before a
/// block-path append — mirror of [`materialize_rotor_k3_ring_tail`]. No-op when
/// `blocks` already cover `shape[2]` (the common prefill / V-only case — reads
/// no GPU).
fn materialize_rotor_v3_ring_tail(vs: &mut QuantRotorV3, device: Device) -> Result<()> {
    if !vs.gpu.is_allocated() {
        return Ok(());
    }
    let rebuilt =
        match crate::storage::synced_rotor_v_blocks(&vs.blocks, &vs.shape, &vs.gpu, device)? {
            std::borrow::Cow::Owned(full) => Some(full),
            std::borrow::Cow::Borrowed(_) => None,
        };
    if let Some(full) = rebuilt {
        vs.blocks = full;
    }
    Ok(())
}

/// Mirror of [`rotor3_gpu_append_into_blocks`] for [`QuantRotorV4`].
fn rotor4_gpu_append_into_blocks(
    vs: &mut QuantRotorV4,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim = head_dim_from_shape(new_shape, "rotor4_gpu_append_into_blocks")?;
    ensure_v4_rotors(vs, head_dim);
    let seq_major = packed_k_chunk_seq_major(new_v, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        // Ring-only tail — see [`rotor3_gpu_append_into_blocks`].
        let gpu = rotor_gpu_encode_ring_only(
            &seq_major,
            new_shape,
            &vs.rotors,
            crate::rotorquant::ROTOR4_BITS,
        )?;
        rotor4_v_sync_ring(
            vs,
            &gpu,
            RingFeed::Maintain,
            new_shape,
            head_dim,
            max_seq,
            device,
        )?;
        bump_rotor_k_shape(&mut vs.shape, new_shape);
        return Ok(());
    }
    // Block path — see [`rotor3_gpu_append_into_blocks`].
    materialize_rotor_v4_ring_tail(vs, device)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = rotor_gpu_encode_block_retaining(
        &seq_major,
        new_shape,
        &vs.rotors,
        crate::rotorquant::ROTOR4_BITS,
    )?;
    rotor4_v_sync_ring(vs, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    push_rotor4_v_block(vs, block, new_shape);
    Ok(())
}

/// Mirror of [`materialize_rotor_v3_ring_tail`] for [`QuantRotorV4`].
fn materialize_rotor_v4_ring_tail(vs: &mut QuantRotorV4, device: Device) -> Result<()> {
    if !vs.gpu.is_allocated() {
        return Ok(());
    }
    let rebuilt =
        match crate::storage::synced_rotor_v_blocks(&vs.blocks, &vs.shape, &vs.gpu, device)? {
            std::borrow::Cow::Owned(full) => Some(full),
            std::borrow::Cow::Borrowed(_) => None,
        };
    if let Some(full) = rebuilt {
        vs.blocks = full;
    }
    Ok(())
}

/// V-side mirror of [`rotor3_sync_ring`] for [`QuantRotorV3`].
///
/// Same invariant, same `b > 1` skip, same self-healing re-seed — the ring type
/// and its contract are axis-agnostic. Only ever receives `Maintain` (ring-only
/// tail and block path both feed the ring) or `Skip` (V-only), so anything other
/// than `Maintain` clears.
fn rotor3_v_sync_ring(
    vs: &mut QuantRotorV3,
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
    // Feed BEFORE `push_rotor3_v_block` — the push bumps `vs.shape[2]`, and the
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

/// Mirror of [`rotor3_v_sync_ring`] for [`QuantRotorV4`].
fn rotor4_v_sync_ring(
    vs: &mut QuantRotorV4,
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

/// Whether a rotor K GPU append should also maintain the store's GPU ring.
///
/// Only the K-only codecs (`RotorKOnly3` / `RotorKOnly4`) have a kernel that
/// reads the ring — `Rotor{3,4}Sym` and `RotorK{3,4}Asym` quantize V as well and
/// the flash kernel takes bf16 V only, so a ring built for them is never read.
/// It is not free: `capacity * kv_h * n_groups * 8 + capacity * kv_h * 4` bytes
/// per layer, growing with context, which at 4k over a 36-layer model is on the
/// order of a few hundred MB of pure waste.
///
/// Passed down from the caller rather than inferred here, so eligibility lives
/// with the dispatcher that knows it.
///
/// Both K-only paths maintain — the prefill-time `update_rotor_k_only_*` as well
/// as the fused decode entry. They only reach a GPU append when QJL is off,
/// which is exactly when the flash kernel is eligible, so the ring they build is
/// the one decode reads. Letting prefill fill it incrementally also avoids
/// making the first decode step re-seed the whole prefix from the CPU blocks
/// (`seed_from_cpu` still covers the paths that genuinely start cold: SSD
/// hydrate, deep clone, and a mid-run fall back to the CPU append).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RingFeed {
    /// Maintain the ring **and** push a CPU block — the ring is the source of
    /// truth for the flash-decode kernel, the block for `dequant()` / SSD spill.
    /// Used by prefill and the non-fused decode fallback, which `dequant()` the
    /// whole prefix on the same step and so need the block immediately.
    Maintain,
    /// Maintain the ring **without** pushing a CPU block — a **ring-only tail**.
    /// The fused decode path never reads the block, so skipping the per-step
    /// host download is the win; the blocks are rebuilt from the ring on demand
    /// at a `dequant()` / SSD-spill boundary. `shape[2]` still advances so the
    /// ring and the attention length stay in lockstep.
    MaintainRingOnly,
    /// Skip the ring. Any live ring is dropped (see the invariant below).
    Skip,
}

/// Feed the legacy rotor symmetric / asymmetric `update_*` entries pass.
///
/// These dequantize the whole prefix on the same step, so the CPU blocks must
/// carry everything and the ring is dropped. Named rather than repeated at each
/// call site because the "blocks are the only copy" property that follows from
/// it is what makes a mid-block truncation unrecoverable there — see
/// `crate::storage::truncate_plan`.
const LEGACY_ROTOR_SYM_FEED: RingFeed = RingFeed::Skip;

/// Feed the legacy iso V `update_iso4` / `update_iso4_sym` entries pass.
///
/// Named for the same reason as the rotor constants: these entries dequantize
/// the whole prefix on the same step, so the CPU blocks must carry everything
/// and the ring is dropped. Passing it also routes the append through
/// [`iso4_gpu_append_into_v_blocks`], which reconciles a pre-existing ring-only
/// tail **and** stores the chunk sequence-major — the orientation the CPU
/// `QuantIsoV4::append` and `QuantIsoV4::dequant` both assume.
const LEGACY_ISO4_V_FEED: RingFeed = RingFeed::Skip;

/// Feed the legacy rotor K-only `update_*` entries pass.
///
/// Unlike the sym path these keep the ring, so a ring-only tail survives a
/// fallback step; the CPU block is pushed alongside because the same step
/// dequantizes.
const LEGACY_ROTOR_K_ONLY_FEED: RingFeed = RingFeed::Maintain;

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
fn rotor3_sync_ring(
    ks: &mut crate::storage::QuantRotorK3,
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

/// Mirror of [`rotor3_sync_ring`] for [`crate::storage::QuantRotorK4`].
fn rotor4_sync_ring(
    ks: &mut crate::storage::QuantRotorK4,
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
/// this; with QJL enabled, fall back to the CPU [`QuantRotorK3::append`] path
/// (the K-side QJL residual is not implemented in MSL — see
/// `rotorquant_msl.rs`).
///
/// `feed` decides whether the GPU ring is maintained — see [`RingFeed`].
/// `max_seq` is the window the cache is currently provisioned for, read from
/// the active `KvStorage` variant by the caller and forwarded to the ring.
fn rotor3_gpu_append_into_k_blocks(
    ks: &mut crate::storage::QuantRotorK3,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim = head_dim_from_shape(new_shape, "rotor3_gpu_append_into_k_blocks")?;
    ensure_k3_rotors(ks, head_dim);
    // qjl_s_matrix is seeded inside `ensure_k3_rotors` when `rotor_qjl_enabled()`
    // is true (mid-run CPU fallback after a GPU first-chunk needs it). The GPU
    // encode itself ignores the JL projection — this path is QJL-off-only by
    // dispatcher contract.
    let seq_major = packed_k_chunk_seq_major(new_k, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        // Ring-only tail: feed the GPU ring, advance `shape[2]`, and skip the
        // per-step host download + CPU block push. The ring is the source of
        // truth for the decode tail; the blocks are rebuilt on demand at a
        // `dequant()` / SSD-spill boundary.
        let gpu = rotor_gpu_encode_ring_only(
            &seq_major,
            new_shape,
            &ks.rotors,
            crate::rotorquant::ROTOR3_BITS,
        )?;
        rotor3_sync_ring(
            ks,
            &gpu,
            RingFeed::Maintain,
            new_shape,
            head_dim,
            max_seq,
            device,
        )?;
        bump_rotor_k_shape(&mut ks.shape, new_shape);
        return Ok(());
    }
    // Block path: prefill / non-fused decode (Maintain), asym (Skip), and the
    // `b > 1` fallback of a ring-only append. A ring-only feed that reaches here
    // is a `b > 1` chunk — normalise it to Maintain so the shared ring feeder
    // clears the ring for the un-representable batch and the CPU block carries
    // the data.
    materialize_rotor_k3_ring_tail(ks, device)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = rotor_gpu_encode_block_retaining(
        &seq_major,
        new_shape,
        &ks.rotors,
        crate::rotorquant::ROTOR3_BITS,
    )?;
    rotor3_sync_ring(ks, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    push_rotor3_k_block(ks, block, new_shape);
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
fn materialize_rotor_k3_ring_tail(
    ks: &mut crate::storage::QuantRotorK3,
    device: Device,
) -> Result<()> {
    if !ks.gpu.is_allocated() {
        return Ok(());
    }
    let rebuilt =
        match crate::storage::synced_rotor_k_blocks(&ks.blocks, &ks.shape, &ks.gpu, device)? {
            std::borrow::Cow::Owned(full) => Some(full),
            std::borrow::Cow::Borrowed(_) => None,
        };
    if let Some(full) = rebuilt {
        ks.blocks = full;
    }
    Ok(())
}

/// Mirror of [`materialize_rotor_k3_ring_tail`] for [`crate::storage::QuantRotorK4`].
fn materialize_rotor_k4_ring_tail(
    ks: &mut crate::storage::QuantRotorK4,
    device: Device,
) -> Result<()> {
    if !ks.gpu.is_allocated() {
        return Ok(());
    }
    let rebuilt =
        match crate::storage::synced_rotor_k_blocks(&ks.blocks, &ks.shape, &ks.gpu, device)? {
            std::borrow::Cow::Owned(full) => Some(full),
            std::borrow::Cow::Borrowed(_) => None,
        };
    if let Some(full) = rebuilt {
        ks.blocks = full;
    }
    Ok(())
}

/// Mirror of [`rotor3_gpu_append_into_k_blocks`] for `QuantRotorK4`.
fn rotor4_gpu_append_into_k_blocks(
    ks: &mut crate::storage::QuantRotorK4,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim = head_dim_from_shape(new_shape, "rotor4_gpu_append_into_k_blocks")?;
    ensure_k4_rotors(ks, head_dim);
    let seq_major = packed_k_chunk_seq_major(new_k, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        // Ring-only tail — see [`rotor3_gpu_append_into_k_blocks`].
        let gpu = rotor_gpu_encode_ring_only(
            &seq_major,
            new_shape,
            &ks.rotors,
            crate::rotorquant::ROTOR4_BITS,
        )?;
        rotor4_sync_ring(
            ks,
            &gpu,
            RingFeed::Maintain,
            new_shape,
            head_dim,
            max_seq,
            device,
        )?;
        bump_rotor_k_shape(&mut ks.shape, new_shape);
        return Ok(());
    }
    // Block path — see [`rotor3_gpu_append_into_k_blocks`] (b>1 ring-only
    // fallback normalises to Maintain).
    materialize_rotor_k4_ring_tail(ks, device)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = rotor_gpu_encode_block_retaining(
        &seq_major,
        new_shape,
        &ks.rotors,
        crate::rotorquant::ROTOR4_BITS,
    )?;
    rotor4_sync_ring(ks, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    push_rotor4_k_block(ks, block, new_shape);
    Ok(())
}

/// Append `new_k` into a live `RotorKOnly3` store's GPU ring (+ CPU blocks),
/// lazily creating the store on first use. No dequant — this is the entry
/// point the rotor flash-decode SDPA path uses.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is not
/// `RotorKOnly3`, and forwards encode / ring errors.
pub(super) fn rotor3_k_only_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let KvStorage::RotorKOnly3 { k, max_seq } = &mut cache.storage else {
        return Err(Error::KvStorageMismatch {
            expected: "RotorKOnly3",
            got: storage_variant_name(&cache.storage),
        });
    };
    let max_seq = *max_seq;
    if k.is_none() {
        let mut init_shape = new_shape.to_vec();
        if let Some(s) = init_shape.get_mut(2) {
            *s = 0;
        }
        *k = Some(crate::storage::QuantRotorK3::new(
            init_shape,
            layer_idx_u32(cache.layer_idx),
        ));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx("RotorKOnly3 K buffer absent after init".into()));
    };
    rotor3_gpu_append_into_k_blocks(
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
    drop_blocks_when_ring_live_k3(ks);
    Ok(())
}

/// Mirror of [`rotor3_k_only_gpu_append`] for `RotorKOnly4`.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is not
/// `RotorKOnly4`, and forwards encode / ring errors.
pub(super) fn rotor4_k_only_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let KvStorage::RotorKOnly4 { k, max_seq } = &mut cache.storage else {
        return Err(Error::KvStorageMismatch {
            expected: "RotorKOnly4",
            got: storage_variant_name(&cache.storage),
        });
    };
    let max_seq = *max_seq;
    if k.is_none() {
        let mut init_shape = new_shape.to_vec();
        if let Some(s) = init_shape.get_mut(2) {
            *s = 0;
        }
        *k = Some(crate::storage::QuantRotorK4::new(
            init_shape,
            layer_idx_u32(cache.layer_idx),
        ));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx("RotorKOnly4 K buffer absent after init".into()));
    };
    rotor4_gpu_append_into_k_blocks(
        ks,
        new_k,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    // Ring is the sole resident store — see [`rotor3_k_only_gpu_append`].
    drop_blocks_when_ring_live_k4(ks);
    Ok(())
}

/// Append `new_k` / `new_v` into a live `RotorSym3` store's GPU rings (+ CPU
/// blocks), lazily creating the stores on first use. No dequant on either axis —
/// this is the entry point the rotor symmetric quant-V flash-decode SDPA path
/// uses.
///
/// Both axes maintain their ring: the quant-V kernel reads K's *and* V's.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is not
/// `RotorSym3`, and forwards encode / ring errors.
pub(super) fn rotor3_sym_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let layer_idx = layer_idx_u32(cache.layer_idx);
    let KvStorage::RotorSym3 { k, v, max_seq } = &mut cache.storage else {
        return Err(Error::KvStorageMismatch {
            expected: "RotorSym3",
            got: storage_variant_name(&cache.storage),
        });
    };
    let max_seq = *max_seq;
    let mut init_shape = new_shape.to_vec();
    if let Some(s) = init_shape.get_mut(2) {
        *s = 0;
    }
    if k.is_none() {
        *k = Some(crate::storage::QuantRotorK3::new(
            init_shape.clone(),
            layer_idx,
        ));
    }
    if v.is_none() {
        *v = Some(QuantRotorV3::new(init_shape, max_seq, layer_idx));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx("RotorSym3 K buffer absent after init".into()));
    };
    rotor3_gpu_append_into_k_blocks(
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
    drop_blocks_when_ring_live_k3(ks);
    let Some(vs) = v.as_mut() else {
        return Err(Error::Mlx("RotorSym3 V buffer absent after init".into()));
    };
    rotor3_gpu_append_into_blocks(
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
    drop_blocks_when_ring_live_v3(vs);
    Ok(())
}

/// Drop a rotor K store's CPU blocks once its GPU ring is live — the ring is
/// then the sole resident copy. No-op until the ring is allocated (before the
/// first GPU append) or for any store that never feeds a ring.
fn drop_blocks_when_ring_live_k3(ks: &mut crate::storage::QuantRotorK3) {
    if ks.gpu.is_allocated() {
        ks.blocks.clear();
        ks.blocks.shrink_to_fit();
    }
}

/// Mirror of [`drop_blocks_when_ring_live_k3`] for [`QuantRotorK4`].
fn drop_blocks_when_ring_live_k4(ks: &mut crate::storage::QuantRotorK4) {
    if ks.gpu.is_allocated() {
        ks.blocks.clear();
        ks.blocks.shrink_to_fit();
    }
}

/// Drop a rotor V store's CPU blocks once its GPU ring is live.
fn drop_blocks_when_ring_live_v3(vs: &mut QuantRotorV3) {
    if vs.gpu.is_allocated() {
        vs.blocks.clear();
        vs.blocks.shrink_to_fit();
    }
}

/// Mirror of [`drop_blocks_when_ring_live_v3`] for [`QuantRotorV4`].
fn drop_blocks_when_ring_live_v4(vs: &mut QuantRotorV4) {
    if vs.gpu.is_allocated() {
        vs.blocks.clear();
        vs.blocks.shrink_to_fit();
    }
}

/// Mirror of [`rotor3_sym_gpu_append`] for `RotorSym4`.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is not
/// `RotorSym4`, and forwards encode / ring errors.
pub(super) fn rotor4_sym_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let layer_idx = layer_idx_u32(cache.layer_idx);
    let KvStorage::RotorSym4 { k, v, max_seq } = &mut cache.storage else {
        return Err(Error::KvStorageMismatch {
            expected: "RotorSym4",
            got: storage_variant_name(&cache.storage),
        });
    };
    let max_seq = *max_seq;
    let mut init_shape = new_shape.to_vec();
    if let Some(s) = init_shape.get_mut(2) {
        *s = 0;
    }
    if k.is_none() {
        *k = Some(crate::storage::QuantRotorK4::new(
            init_shape.clone(),
            layer_idx,
        ));
    }
    if v.is_none() {
        *v = Some(QuantRotorV4::new(init_shape, max_seq, layer_idx));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx("RotorSym4 K buffer absent after init".into()));
    };
    rotor4_gpu_append_into_k_blocks(
        ks,
        new_k,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    drop_blocks_when_ring_live_k4(ks);
    let Some(vs) = v.as_mut() else {
        return Err(Error::Mlx("RotorSym4 V buffer absent after init".into()));
    };
    rotor4_gpu_append_into_blocks(
        vs,
        new_v,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    drop_blocks_when_ring_live_v4(vs);
    Ok(())
}

/// Accumulated sequence length held by a rotor K storage `shape`
/// (`[B, kv_h, S, D]`), or 0 for a not-yet-shaped buffer.
fn accumulated_seq(shape: &[i32]) -> i32 {
    if shape.len() != 4 {
        return 0;
    }
    shape.get(2).copied().unwrap_or(0).max(0)
}

/// Whether an append should take the ring-only tail path (no CPU block push).
///
/// Only `feed == MaintainRingOnly` **and** `b == 1` qualify: the ring's
/// per-step stride does not interleave batch, so a `b > 1` chunk cannot be laid
/// into it. Because the ring-only path pushes no CPU block, a `b > 1` chunk
/// here would clear the ring in `rotor*_sync_ring` and silently drop the chunk
/// while still advancing `shape[2]` — the ring-vs-blocks divergence the
/// invariant forbids. So `b > 1` falls back to the block-pushing path, which
/// handles batch and keeps the CPU blocks the source of truth. (Per request the
/// batch dim is fixed, so a `b > 1` cache never builds a ring-only tail to
/// lose.)
///
/// `new_seq` is deliberately **not** consulted: a ring-only feed carries a
/// multi-token chunk as happily as a single decode step. What routes a
/// multi-token append to the block path is the `feed` its caller chose — the
/// legacy `update_*` entries pass `Maintain` / `Skip`, and they are where a
/// `q_seq > 1` forward lands once the fused decode gate (`q_seq == 1`) rejects
/// it.
fn is_ring_only_append(feed: RingFeed, new_shape: &[i32]) -> bool {
    feed == RingFeed::MaintainRingOnly && matches!(b_kv_h_new_seq(new_shape), Ok((1, _, _)))
}

/// `(b, kv_h, new_seq)` from a rank-4 `[B, kv_h, S, D]` shape.
fn b_kv_h_new_seq(new_shape: &[i32]) -> Result<(i32, i32, i32)> {
    match (new_shape.first(), new_shape.get(1), new_shape.get(2)) {
        (Some(&b), Some(&kv_h), Some(&s)) if new_shape.len() == 4 => Ok((b, kv_h, s)),
        _ => Err(Error::Mlx(format!(
            "rotor K append: expected 4D new_shape, got {new_shape:?}"
        ))),
    }
}

/// Reorder a head-major `[B, kv_h, S, D]` K chunk to the sequence-major
/// `[B, S, kv_h, D]` element order the packed K stores (rotor / iso)
/// accumulate in.
///
/// The CPU `QuantRotorK{3,4}::append` / `QuantIsoK{3,4}::append` transpose
/// before encoding, so the GPU encode must too or the two produce different
/// block layouts for a multi-token chunk with `kv_h > 1`. For the decode step
/// (`S == 1`) this is the identity.
///
/// `transpose` yields a strided view and the MSL kernels read by raw linear
/// offset (they ignore MLX lazy-transpose strides), so the permutation is
/// materialised here.
fn packed_k_chunk_seq_major(new_k: &Array, new_shape: &[i32], device: Device) -> Result<Array> {
    let (_b, _kv_h, new_seq) = b_kv_h_new_seq(new_shape)?;
    if new_seq == 1 {
        // Identity permutation — skip the copy on the decode hot path.
        return new_k.try_clone();
    }
    new_k.transpose(&[0, 2, 1, 3], device)?.contiguous(device)
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
/// `bits` selects the encode kernel explicitly; anything but 3 or 4 is an error
/// rather than a silent fall-through to one width's kernel over the other's
/// codes.
fn iso_gpu_encode_block_retaining(
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

    let (codes_arr, scales_arr, quats_arr, norms_arr) = match bits {
        ISO_K3_BITS => crate::isoquant_msl::iso_quantize_v3_gpu(new_k, head_dim, Device::Gpu)?,
        ISO_K4_BITS => crate::isoquant_msl_v4::iso_quantize_v4_gpu(new_k, head_dim, Device::Gpu)?,
        other => {
            return Err(Error::Quant(format!(
                "iso_gpu_encode_block_retaining: unsupported bits={other} (only 3 and 4); \
                 refusing to encode with another width's kernel"
            )))
        }
    };

    // Select the readback by width explicitly. A `_` arm here would send a
    // future width's codes through one of these two unpackers and silently
    // produce a wrong CPU block — `bits` is a plain integer, so no
    // exhaustiveness lint would catch it.
    let (codes, scales, quats, norms) = match bits {
        ISO_K3_BITS => crate::isoquant_msl::iso3_gpu_outputs_to_cpu(
            &codes_arr,
            &scales_arr,
            &quats_arr,
            &norms_arr,
            n_tokens_total,
            n_groups,
        )?,
        ISO_K4_BITS => crate::isoquant_msl_v4::iso4_gpu_outputs_to_cpu(
            &codes_arr,
            &scales_arr,
            &quats_arr,
            &norms_arr,
            n_tokens_total,
            n_groups,
        )?,
        other => {
            return Err(Error::Quant(format!(
                "iso_gpu_encode_block_retaining: unsupported bits={other} on readback \
                 (only 3 and 4)"
            )))
        }
    };

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
/// `bits` selects the encode kernel explicitly; anything but 3 or 4 is an error.
fn iso_gpu_encode_ring_only(
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
    let (codes_arr, scales_arr, _quats_arr, norms_arr) = match bits {
        ISO_K3_BITS => crate::isoquant_msl::iso_quantize_v3_gpu(new_kv, head_dim, Device::Gpu)?,
        ISO_K4_BITS => crate::isoquant_msl_v4::iso_quantize_v4_gpu(new_kv, head_dim, Device::Gpu)?,
        other => {
            return Err(Error::Quant(format!(
                "iso_gpu_encode_ring_only: unsupported bits={other} (only 3 and 4); \
                 refusing to encode with another width's kernel"
            )))
        }
    };
    // Collapse per-group norms to the per-token form the ring stores (GPU-side,
    // no host round-trip).
    let norms_per_token = collapse_group_norms_to_token(&norms_arr, n_tokens_total, n_groups)?;
    Ok(PackedKEncodedGpu {
        codes: codes_arr,
        scales: scales_arr,
        norms: norms_per_token,
    })
}

/// Append `new_k` into a live `IsoKOnly3` store's GPU ring (+ CPU blocks),
/// lazily creating the store on first use. No dequant — this is the entry point
/// the iso flash-decode SDPA path uses.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is not
/// `IsoKOnly3`, and forwards encode / ring errors.
pub(super) fn iso3_k_only_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let KvStorage::IsoKOnly3 { k, max_seq } = &mut cache.storage else {
        return Err(Error::KvStorageMismatch {
            expected: "IsoKOnly3",
            got: storage_variant_name(&cache.storage),
        });
    };
    let max_seq = *max_seq;
    if k.is_none() {
        let mut init_shape = new_shape.to_vec();
        if let Some(s) = init_shape.get_mut(2) {
            *s = 0;
        }
        *k = Some(QuantIsoK3::new(init_shape, max_seq));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx("IsoKOnly3 K buffer absent after init".into()));
    };
    // Ring-only tail: the fused iso K-only flash decode reads the ring, never the
    // CPU blocks, so skip the per-step host download; the blocks are rebuilt from
    // the ring on demand at a `dequant()` / SSD-spill boundary.
    iso3_gpu_append_into_k_blocks(
        ks,
        new_k,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    drop_blocks_when_ring_live_iso_k3(ks);
    Ok(())
}

/// Mirror of [`iso3_k_only_gpu_append`] for `IsoKOnly4`.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is not
/// `IsoKOnly4`, and forwards encode / ring errors.
pub(super) fn iso4_k_only_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let KvStorage::IsoKOnly4 { k, max_seq } = &mut cache.storage else {
        return Err(Error::KvStorageMismatch {
            expected: "IsoKOnly4",
            got: storage_variant_name(&cache.storage),
        });
    };
    let max_seq = *max_seq;
    if k.is_none() {
        let mut init_shape = new_shape.to_vec();
        if let Some(s) = init_shape.get_mut(2) {
            *s = 0;
        }
        *k = Some(QuantIsoK4::new(init_shape, max_seq));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx("IsoKOnly4 K buffer absent after init".into()));
    };
    // Ring-only tail — see [`iso3_k_only_gpu_append`].
    iso4_gpu_append_into_k_blocks(
        ks,
        new_k,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    drop_blocks_when_ring_live_iso_k4(ks);
    Ok(())
}

/// Append `new_k` / `new_v` into a live `IsoSym3` store's GPU rings (ring-only
/// tail on both axes), lazily creating the stores on first use. No dequant on
/// either axis — this is the entry point the iso symmetric quant-V flash-decode
/// SDPA path uses. Mirror of [`rotor3_sym_gpu_append`].
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is not
/// `IsoSym3`, and forwards encode / ring errors.
pub(super) fn iso3_sym_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let KvStorage::IsoSym3 { k, v, max_seq } = &mut cache.storage else {
        return Err(Error::KvStorageMismatch {
            expected: "IsoSym3",
            got: storage_variant_name(&cache.storage),
        });
    };
    let max_seq = *max_seq;
    let mut init_shape = new_shape.to_vec();
    if let Some(s) = init_shape.get_mut(2) {
        *s = 0;
    }
    if k.is_none() {
        *k = Some(QuantIsoK3::new(init_shape.clone(), max_seq));
    }
    if v.is_none() {
        *v = Some(QuantIsoV3::new(init_shape));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx("IsoSym3 K buffer absent after init".into()));
    };
    iso3_gpu_append_into_k_blocks(
        ks,
        new_k,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    // Ring is now the sole resident store for K — drop the redundant CPU blocks
    // (the prefill prefix, seeded into the ring on the first fused-decode step).
    drop_blocks_when_ring_live_iso_k3(ks);
    let Some(vs) = v.as_mut() else {
        return Err(Error::Mlx("IsoSym3 V buffer absent after init".into()));
    };
    iso3_gpu_append_into_v_blocks(
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
    drop_blocks_when_ring_live_iso_v3(vs);
    Ok(())
}

/// Mirror of [`iso3_sym_gpu_append`] for `IsoSym4`.
///
/// # Errors
///
/// Returns [`Error::KvStorageMismatch`] when the active storage is not
/// `IsoSym4`, and forwards encode / ring errors.
pub(super) fn iso4_sym_gpu_append(
    cache: &mut KvCache,
    new_k: &Array,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<()> {
    let KvStorage::IsoSym4 { k, v, max_seq } = &mut cache.storage else {
        return Err(Error::KvStorageMismatch {
            expected: "IsoSym4",
            got: storage_variant_name(&cache.storage),
        });
    };
    let max_seq = *max_seq;
    let mut init_shape = new_shape.to_vec();
    if let Some(s) = init_shape.get_mut(2) {
        *s = 0;
    }
    if k.is_none() {
        *k = Some(QuantIsoK4::new(init_shape.clone(), max_seq));
    }
    if v.is_none() {
        *v = Some(QuantIsoV4::new(init_shape, max_seq));
    }
    let Some(ks) = k.as_mut() else {
        return Err(Error::Mlx("IsoSym4 K buffer absent after init".into()));
    };
    iso4_gpu_append_into_k_blocks(
        ks,
        new_k,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    drop_blocks_when_ring_live_iso_k4(ks);
    let Some(vs) = v.as_mut() else {
        return Err(Error::Mlx("IsoSym4 V buffer absent after init".into()));
    };
    iso4_gpu_append_into_v_blocks(
        vs,
        new_v,
        new_shape,
        device,
        RingFeed::MaintainRingOnly,
        max_seq,
    )?;
    drop_blocks_when_ring_live_iso_v4(vs);
    Ok(())
}

/// Drop an iso K store's CPU blocks once its GPU ring is live — the ring is then
/// the sole resident copy. No-op until the ring is allocated or for any store
/// that never feeds a ring.
fn drop_blocks_when_ring_live_iso_k3(ks: &mut QuantIsoK3) {
    if ks.gpu.is_allocated() {
        ks.blocks.clear();
        ks.blocks.shrink_to_fit();
    }
}

/// Mirror of [`drop_blocks_when_ring_live_iso_k3`] for [`QuantIsoK4`].
fn drop_blocks_when_ring_live_iso_k4(ks: &mut QuantIsoK4) {
    if ks.gpu.is_allocated() {
        ks.blocks.clear();
        ks.blocks.shrink_to_fit();
    }
}

/// Drop an iso V store's CPU blocks once its GPU ring is live.
fn drop_blocks_when_ring_live_iso_v3(vs: &mut QuantIsoV3) {
    if vs.gpu.is_allocated() {
        vs.blocks.clear();
        vs.blocks.shrink_to_fit();
    }
}

/// Mirror of [`drop_blocks_when_ring_live_iso_v3`] for [`QuantIsoV4`].
fn drop_blocks_when_ring_live_iso_v4(vs: &mut QuantIsoV4) {
    if vs.gpu.is_allocated() {
        vs.blocks.clear();
        vs.blocks.shrink_to_fit();
    }
}

/// V-side append with a ring-only branch — mirror of
/// [`iso3_gpu_append_into_k_blocks`] for [`QuantIsoV3`]. The iso codec is
/// axis-agnostic, so V encodes through the same kernel as K.
fn iso3_gpu_append_into_v_blocks(
    vs: &mut QuantIsoV3,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim = head_dim_from_shape(new_shape, "iso3_gpu_append_into_v_blocks")?;
    let seq_major = packed_k_chunk_seq_major(new_v, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        let gpu = iso_gpu_encode_ring_only(&seq_major, new_shape, ISO_K3_BITS)?;
        iso3_v_sync_ring(
            vs,
            &gpu,
            RingFeed::Maintain,
            new_shape,
            head_dim,
            max_seq,
            device,
        )?;
        bump_rotor_k_shape(&mut vs.shape, new_shape);
        return Ok(());
    }
    vs.reconcile_ring(device, crate::storage::RingDisposition::Keep)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = iso_gpu_encode_block_retaining(&seq_major, new_shape, ISO_K3_BITS)?;
    iso3_v_sync_ring(vs, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    vs.blocks.push(block);
    bump_rotor_k_shape(&mut vs.shape, new_shape);
    Ok(())
}

/// Test seam onto the iso4 V append the legacy entries use.
///
/// The appender is private to this module; the orientation test lives with the
/// other iso dispatch tests, one module up. Exposing the call rather than
/// duplicating it is what keeps the test pinned to the production path.
#[cfg(test)]
pub(super) fn iso4_v_gpu_append_for_test(
    vs: &mut QuantIsoV4,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
    max_seq: i32,
) -> Result<()> {
    iso4_gpu_append_into_v_blocks(vs, new_v, new_shape, device, LEGACY_ISO4_V_FEED, max_seq)
}

/// Mirror of [`iso3_gpu_append_into_v_blocks`] for [`QuantIsoV4`].
fn iso4_gpu_append_into_v_blocks(
    vs: &mut QuantIsoV4,
    new_v: &Array,
    new_shape: &[i32],
    device: Device,
    feed: RingFeed,
    max_seq: i32,
) -> Result<()> {
    let head_dim = head_dim_from_shape(new_shape, "iso4_gpu_append_into_v_blocks")?;
    let seq_major = packed_k_chunk_seq_major(new_v, new_shape, device)?;
    if is_ring_only_append(feed, new_shape) {
        let gpu = iso_gpu_encode_ring_only(&seq_major, new_shape, ISO_K4_BITS)?;
        iso4_v_sync_ring(
            vs,
            &gpu,
            RingFeed::Maintain,
            new_shape,
            head_dim,
            max_seq,
            device,
        )?;
        bump_rotor_k_shape(&mut vs.shape, new_shape);
        return Ok(());
    }
    vs.reconcile_ring(device, crate::storage::RingDisposition::Keep)?;
    let block_feed = if feed == RingFeed::Skip {
        RingFeed::Skip
    } else {
        RingFeed::Maintain
    };
    let (block, gpu) = iso_gpu_encode_block_retaining(&seq_major, new_shape, ISO_K4_BITS)?;
    iso4_v_sync_ring(vs, &gpu, block_feed, new_shape, head_dim, max_seq, device)?;
    vs.blocks.push(block);
    bump_rotor_k_shape(&mut vs.shape, new_shape);
    Ok(())
}

/// V-side mirror of [`iso3_sync_ring`] for [`QuantIsoV3`].
fn iso3_v_sync_ring(
    vs: &mut QuantIsoV3,
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

/// Mirror of [`iso3_v_sync_ring`] for [`QuantIsoV4`].
fn iso4_v_sync_ring(
    vs: &mut QuantIsoV4,
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

/// Feed one encoded iso3 chunk into the store's GPU ring.
///
/// `b > 1` is a skip for the same reason [`rotor3_sync_ring`] skips it:
/// [`crate::storage::QuantKGpuRing`]'s per-step stride does not interleave
/// batch. A skipped feed *clears* rather than leaving a stale ring — see the
/// [`RingFeed`] invariant.
fn iso3_sync_ring(
    ks: &mut QuantIsoK3,
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
    // Feed BEFORE `push_iso3_k_block` — the push bumps `ks.shape[2]`, and the
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

/// Mirror of [`iso3_sync_ring`] for [`QuantIsoK4`].
fn iso4_sync_ring(
    ks: &mut QuantIsoK4,
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

impl KvCache {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    /// Append one decode step's K/V tensors (or a prefill chunk) to the cache.
    pub fn update(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        // Rotating SWA path. The RotatingState owns its own offset
        // (mirrors mlx-lm `RotatingKVCache.offset`), so we sync `self.offset`
        // from it after the update. Skip the prefill_raw machinery (the
        // rotating buffer has its own growth+rotate semantics that the
        // pre-allocated prefill_raw buffer would conflict with).
        if let Some(ref mut rot) = self.rotating {
            let (k_full, v_full) = rot.update_and_fetch(new_k, new_v, device)?;
            self.offset = rot.offset;
            return Ok((k_full, v_full));
        }

        let new_seq = new_k.shape()[2];
        // Provision the step before any mutation: `update_prefill_raw` runs the
        // prefill-side check itself, so this covers the decode dispatch below,
        // whose stores all cap their capacity at the storage `max_seq`.
        if !self.in_prefill {
            self.ensure_decode_capacity(self.offset + new_seq)?;
        }
        self.offset += new_seq;

        if self.in_prefill {
            return self.update_prefill_raw(new_k, new_v, device);
        }

        // Dispatch on the actual storage variant, not self.quant.
        //
        // Using self.quant here was the bug: after SSD hydration, SWA
        // layers are stored with tag "none" → KvStorage::None, but
        // from_storage() sets self.quant to the model's global KvQuant (e.g.
        // K8V8). The quant-based dispatch then routed to update_k8v8(), which
        // pattern-matched self.storage expecting KvStorage::K8V8 and hit the
        // unreachable!(). Dispatching on self.storage is the ground truth:
        // it reflects what data is actually in the cache regardless of the
        // declared KvQuant, and it already covers the Paged variant which was
        // the only exception before this fix.
        match &self.storage {
            KvStorage::K8V4 { .. } => self.update_k8v4(new_k, new_v, device),
            KvStorage::K8V8 { .. } => self.update_k8v8(new_k, new_v, device),
            KvStorage::Planar { .. } => self.update_planar(new_k, new_v, device),
            KvStorage::None { .. } => self.update_none(new_k, new_v, device),
            KvStorage::Paged { .. } => self.update_paged(new_k, new_v, device),
            KvStorage::Mixed { .. } | KvStorage::RotKTq4V { .. } => Err(Error::Mlx(
                "Contract violation: KvCache::update called on a Mixed/RotKTq4V cache. \
                     These caches MUST be driven through KvCache::update_and_sdpa (universal \
                     wrapper). Direct update() bypasses the quantized SDPA and leaves the cache \
                     in an inconsistent state."
                    .into(),
            )),
            // K8VTurbo3 decode update — same structure as K8V4 but bits=3 on V.
            KvStorage::K8VTurbo3 { .. } => self.update_k8vturbo3(new_k, new_v, device),
            // TurboSym3 decode update — symmetric WHT-3 K + turbo3 V.
            // K side uses the GPU turbo3 MSL kernel; V side forced CPU
            // (same K8VTurbo3 precedent: GPU V-side dispatch regressed −2% TPS gate).
            KvStorage::TurboSym3 { .. } => self.update_tsym3(new_k, new_v, device),
            // TurboSym4 decode update — symmetric WHT-4 K + tq4 V.
            KvStorage::TurboSym4 { .. } => self.update_tsym4(new_k, new_v, device),
            // PlanarK decode update — K is PlanarQuant 4-bit, V bf16.
            KvStorage::PlanarK { .. } => self.update_planar_k(new_k, new_v, device),
            // K8VTurbo2 decode update — same structure as K8V4 but bits=2 on V.
            KvStorage::K8VTurbo2 { .. } => self.update_k8vturbo2(new_k, new_v, device),
            // IsoV3 decode update — K = affine q8_0, V = IsoQuant 3-bit (CPU).
            KvStorage::IsoV3 { .. } => self.update_iso3(new_k, new_v, device),
            // IsoV4 decode update — K = affine q8_0, V = IsoQuant 4-bit (CPU).
            KvStorage::IsoV4 { .. } => self.update_iso4(new_k, new_v, device),
            // RotorV3 decode update — K = affine q8_0, V = rotor3 (CPU).
            KvStorage::RotorV3 { .. } => self.update_rotor3(new_k, new_v, device),
            // RotorV4 decode update — K = affine q8_0, V = rotor4 (CPU).
            KvStorage::RotorV4 { .. } => self.update_rotor4(new_k, new_v, device),
            // K8VTurbo3Tcq decode update — same code path as
            // K8VTurbo3 but with Viterbi trellis encode-side assignment.
            KvStorage::K8VTurbo3Tcq { .. } => self.update_k8vturbo3_tcq(new_k, new_v, device),
            // K8VTurbo2Tcq decode update — same code path as
            // K8VTurbo2 but with Viterbi trellis encode-side assignment.
            KvStorage::K8VTurbo2Tcq { .. } => self.update_k8vturbo2_tcq(new_k, new_v, device),
            // Iso symmetric / K-only decode updates.
            KvStorage::IsoSym3 { .. } => self.update_iso3_sym(new_k, new_v, device),
            KvStorage::IsoSym4 { .. } => self.update_iso4_sym(new_k, new_v, device),
            KvStorage::IsoKOnly3 { .. } => self.update_iso_k_only_3(new_k, new_v, device),
            KvStorage::IsoKOnly4 { .. } => self.update_iso_k_only_4(new_k, new_v, device),
            // Symmetric / K-only rotor variants.
            KvStorage::RotorSym3 { .. } => self.update_rotor3_sym(new_k, new_v, device),
            KvStorage::RotorSym4 { .. } => self.update_rotor4_sym(new_k, new_v, device),
            KvStorage::RotorKOnly3 { .. } => self.update_rotor_k_only_3(new_k, new_v, device),
            KvStorage::RotorKOnly4 { .. } => self.update_rotor_k_only_4(new_k, new_v, device),
            // Asymmetric rotor K + affine V variants.
            KvStorage::RotorKAsym3 { .. } => self.update_rotor_k_asym_3(new_k, new_v, device),
            KvStorage::RotorKAsym4 { .. } => self.update_rotor_k_asym_4(new_k, new_v, device),
        }
    }

    /// Decode-step update for the unquantised (`KvQuant::None`) cache.
    ///
    /// Reuses `update_decode_fp16` — the same pre-allocated bf16 buffer machinery
    /// already used as the warm-TTFT fp16 decode seed for the quantised paths.
    /// On the first call, `update_decode_fp16` allocates the
    /// `[B, kv_h, max_seq, head_dim]` buffer; subsequent calls slice_update at
    /// the current offset.
    fn update_none(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::None { max_seq } = &self.storage else {
            return Err(Error::KvStorageMismatch {
                expected: "None",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;
        self.update_decode_fp16(new_k, new_v, max_seq, device)
    }

    /// Enforce the optional `RMLX_KV_MAX_SEQ_HARD_CAP` and grow the
    /// per-layer raw prefill buffer when the next chunk would overflow it.
    ///
    /// `needed_seq` is the post-append sequence length (`prev_offset + new_seq`).
    /// When the existing `[B, kv_h, max_seq, head_dim]` buffer is too small,
    /// a new buffer of the next power-of-two capacity is allocated and the
    /// filled prefix `[..prev_offset]` is copied forward via `slice_update`.
    /// The storage variant's `max_seq` is bumped to the new capacity so
    /// downstream `exit_prefill` quantised buffers honour the same size.
    ///
    /// # Resumed-cache guard
    ///
    /// The "quantised payload not allocated yet" doc-claim only holds on the
    /// **first** prefill of a layer. `KvCache::enter_prefill` only flips
    /// `in_prefill=true` and clears `prefill_raw_k/v`; it does NOT reset
    /// payload buffers or `MixedKvState.offset`. On any resumed-cache path
    /// (chunked-prefill resume across calls, SSD-hydrate seed, branched
    /// generation), the storage already carries on-axis buffers sized to the
    /// old `max_seq`. Bumping the scalar would let it disagree with the
    /// payload shape and produce a shape assert or silent truncation in
    /// `exit_prefill`. The guard below detects payload presence and fails
    /// loudly with a typed error instead.
    #[allow(
        clippy::too_many_arguments,
        reason = "internal helper; carries the prefill shape derived from new_k/new_v in the caller"
    )]
    fn ensure_prefill_capacity(
        &mut self,
        needed_seq: i32,
        prev_offset: i32,
        b: i32,
        kv_h: i32,
        head_dim: i32,
        k_dtype: Dtype,
        v_dtype: Dtype,
        device: Device,
    ) -> Result<()> {
        // Hard cap: opt-in via `RMLX_KV_MAX_SEQ_HARD_CAP`. Unset → no cap.
        if let Some(cap) = kv_hard_cap() {
            if needed_seq > cap {
                tracing::warn!(
                    requested = needed_seq,
                    cap,
                    "KV hard cap exceeded — rejecting prefill request"
                );
                return Err(Error::KvHardCapExceeded {
                    requested: needed_seq,
                    cap,
                });
            }
        }

        // Virtual ceiling: the resolved `--max-ctx`. The ring grows lazily up
        // to it; a prefill needing more is rejected before any allocation so a
        // server started with a large ceiling pays no long-context tax on short
        // requests. Checked after the env hard cap (both bound the same axis).
        if let Some(ceiling) = self.max_seq_ceiling {
            if needed_seq > ceiling {
                tracing::warn!(
                    requested = needed_seq,
                    ceiling,
                    "KV max-ctx ceiling exceeded — rejecting prefill request"
                );
                return Err(Error::KvCeilingExceeded {
                    requested: needed_seq,
                    ceiling,
                });
            }
        }

        let current_max_seq = storage_max_seq(&self.storage);
        if needed_seq <= current_max_seq {
            return Ok(());
        }

        // A grow on a resumed cache (any payload currently materialised)
        // would leave the storage `max_seq` scalar disagreeing with the
        // on-axis payload buffer length sized to the old max_seq → shape
        // assert or silent truncation downstream. Detect and fail loudly
        // instead of corrupting the buffers. The grow path is only legal on
        // a fresh layer / between `enter_prefill` and the very first
        // `exit_prefill`, where no quantised payload exists yet.
        //
        // Invariant: callers (chunked-prefill resume, SSD-hydrate seed,
        // branched generation) MUST size the cache to fit the full prompt
        // at construction time. Hitting this branch in production means the
        // caller mis-sized the cache. Documented as a non-fatal typed error
        // so the runtime degrades cleanly; the test
        // [`update_prefill_raw_rejects_grow_on_resumed_cache`]
        // exercises this exact path.
        if self.storage_has_materialised_payload() {
            return Err(Error::Mlx(
                "grow not legal after exit_prefill — current_max_seq exceeded on \
                 resumed cache; raise --max-ctx or RMLX_KV_MAX_SEQ_HARD_CAP"
                    .into(),
            ));
        }

        // Grow to next power-of-two ≥ needed. Doubling avoids per-chunk
        // churn during multi-chunk prefill while keeping the buffer within
        // a single doubling of the request. When a virtual ceiling is set,
        // clamp the doubled size to it (never allocate past `--max-ctx`):
        // `needed_seq <= ceiling` is guaranteed by the reject check above, so
        // the clamped capacity still fits the request.
        let new_max_seq = match self.max_seq_ceiling {
            Some(ceiling) => next_pow2_seq(needed_seq).min(ceiling),
            None => next_pow2_seq(needed_seq),
        };

        tracing::info!(
            from = current_max_seq,
            to = new_max_seq,
            needed_seq,
            "KV prefill buffer grow"
        );

        // Bump max_seq on the storage variant first so subsequent reads see
        // the new capacity (single-source-of-truth for exit_prefill).
        set_storage_max_seq(&mut self.storage, new_max_seq);

        // If the raw prefill buffer was already allocated, copy the filled
        // prefix `[..prev_offset]` into a fresh, larger buffer. If not
        // allocated, the lazy path in `update_prefill_raw` will pick up the
        // new max_seq below.
        //
        // `prev_offset` is computed upstream as `self.offset - new_seq` after
        // `self.offset += new_seq`, so it is structurally non-negative at this
        // call site. Pin the invariant with a `debug_assert!` instead of
        // laundering it through `.max(0)`.
        debug_assert!(
            prev_offset >= 0,
            "prev_offset invariant — offset accounting upstream"
        );
        let filled = prev_offset;
        // K/V copy hoists the three slice descriptors above the K/V copy
        // blocks. The descriptors are identical for both axes (same
        // `[B, kv_h, filled, head_dim]` window), so sharing them
        // is the correct deduplication and does not require a helper.
        if let (Some(old_k), Some(old_v)) = (self.prefill_raw_k.take(), self.prefill_raw_v.take()) {
            let new_shape = [b, kv_h, new_max_seq, head_dim];
            let new_k_buf = zeros(&new_shape, k_dtype, device)?;
            let new_v_buf = zeros(&new_shape, v_dtype, device)?;

            let slice_start = vec![0i32; 4];
            let slice_stop: Vec<i32> = [b, kv_h, filled, head_dim].into();
            let strides = vec![1i32; 4];

            let new_k_buf = if filled > 0 {
                let prefix_k = old_k.slice(&slice_start, &slice_stop, &strides, device)?;
                new_k_buf.slice_update(&prefix_k, &slice_start, &slice_stop, &strides, device)?
            } else {
                new_k_buf
            };
            let new_v_buf = if filled > 0 {
                let prefix_v = old_v.slice(&slice_start, &slice_stop, &strides, device)?;
                new_v_buf.slice_update(&prefix_v, &slice_start, &slice_stop, &strides, device)?
            } else {
                new_v_buf
            };
            let _ = new_k_buf.async_eval();
            let _ = new_v_buf.async_eval();
            self.prefill_raw_k = Some(new_k_buf);
            self.prefill_raw_v = Some(new_v_buf);
        }

        Ok(())
    }

    /// Does the active `KvStorage` already carry quantised (or fp16-seeded)
    /// payload buffers from a prior `exit_prefill`?
    ///
    /// Returns true if any K/V payload Option is `Some` on the active variant,
    /// or — for variants that store K/V on `MixedKvState` rather than as
    /// per-axis Options — when `state.offset > 0`. Also returns true when the
    /// parent cache holds a decode fp16 seed (`KvStorage::None`, `PlanarK` V,
    /// K-only variants, Paged seed), since those routes materialise their
    /// "payload" outside the storage variant.
    fn storage_has_materialised_payload(&self) -> bool {
        if self.decode_fp16_k.is_some() || self.decode_fp16_v.is_some() {
            return true;
        }
        match &self.storage {
            KvStorage::K8V4 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::K8V8 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::Planar { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::None { .. } => false,
            KvStorage::Mixed { state, .. } => state.offset > 0,
            KvStorage::Paged {
                k, v_k8, v_planar, ..
            } => k.is_some() || v_k8.is_some() || v_planar.is_some(),
            KvStorage::RotKTq4V { k_state, v, .. } => k_state.offset > 0 || v.is_some(),
            KvStorage::K8VTurbo3 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::TurboSym3 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::TurboSym4 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::PlanarK { k, .. } => k.is_some(),
            KvStorage::K8VTurbo2 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::IsoV3 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::IsoV4 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::RotorV3 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::RotorV4 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::K8VTurbo3Tcq { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::K8VTurbo2Tcq { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::IsoSym3 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::IsoSym4 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::IsoKOnly3 { k, .. } => k.is_some(),
            KvStorage::IsoKOnly4 { k, .. } => k.is_some(),
            KvStorage::RotorSym3 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::RotorSym4 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::RotorKOnly3 { k, .. } => k.is_some(),
            KvStorage::RotorKOnly4 { k, .. } => k.is_some(),
            // RotorKAsym3 / RotorKAsym4 — either side materialised.
            KvStorage::RotorKAsym3 { k, v, .. } => k.is_some() || v.is_some(),
            KvStorage::RotorKAsym4 { k, v, .. } => k.is_some() || v.is_some(),
        }
    }

    /// Grow the provisioned `max_seq` when the next **decode** append would
    /// overflow it.
    ///
    /// `max_seq` is provisioned lazily: it starts at the small default and
    /// [`Self::ensure_prefill_capacity`] grows it as the prompt fills. Decode
    /// then appends one token per step, so a sequence that crosses the
    /// provisioned bound has to grow it too. Without this, `max_seq` freezes at
    /// whatever the prompt happened to need and every store that caps its own
    /// capacity at `max_seq` stops accepting appends mid-stream — each in a
    /// different way, none of them good:
    ///
    /// * the packed GPU rings raise a shape error and abort the decode step;
    /// * the paged code buffers clamp their capacity and slice to zero length,
    ///   surfacing as a downstream reshape failure;
    /// * the bf16 mirrors stop expanding, so the append `slice_update`s out of
    ///   bounds — depending on the shape either an MLX error or a **silent
    ///   no-op** that drops the token while `offset` marches on.
    ///
    /// Growing here keeps one provisioning rule for both phases, so the stores
    /// stay on the paged/realloc growth paths they already implement.
    ///
    /// Unlike the prefill path there is no "payload already materialised" guard:
    /// at decode the payload always exists, and raising `max_seq` is precisely
    /// what lets each store's own grow path extend it. That is safe wherever the
    /// window is read per-append (`QuantK`, `QuantPlanarK`, `QuantRotorK{3,4}`,
    /// `QuantIsoV3`, the bf16 mirrors), which is every store this function can
    /// reach: each tracks its own capacity and copies the filled prefix forward
    /// on realloc, so the scalar and the buffers cannot disagree.
    ///
    /// **Head-major flash buffers.** These latch their window into
    /// `flash_max_seq` at allocation, so raising `max_seq` here does not, by
    /// itself, extend them. The TurboFlash path is opt-in and has its own
    /// dispatch (`update_and_sdpa_k8v4_flash_inner`), which bypasses
    /// `KvCache::update` — so it calls this function directly before its append
    /// and re-sizes the latched buffers via `grow_flash_buffers` when the window
    /// grew past `flash_max_seq`. That keeps the one provisioning rule covering
    /// the flash path too, instead of letting the append walk off the frozen
    /// window at the next power-of-two boundary.
    ///
    /// The hard cap and the `--max-ctx` ceiling still bound the growth: a
    /// request that genuinely cannot fit is rejected loudly rather than
    /// truncated.
    pub(super) fn ensure_decode_capacity(&mut self, needed_seq: i32) -> Result<()> {
        // Hard cap: opt-in via `RMLX_KV_MAX_SEQ_HARD_CAP`. Unset → no cap.
        if let Some(cap) = kv_hard_cap() {
            if needed_seq > cap {
                tracing::warn!(
                    requested = needed_seq,
                    cap,
                    "KV hard cap exceeded — rejecting decode step"
                );
                return Err(Error::KvHardCapExceeded {
                    requested: needed_seq,
                    cap,
                });
            }
        }

        // Virtual ceiling: the resolved `--max-ctx`. A decode step that would
        // carry the sequence past it cannot be served by growing, so reject it
        // with the same typed error the prefill path uses.
        if let Some(ceiling) = self.max_seq_ceiling {
            if needed_seq > ceiling {
                tracing::warn!(
                    requested = needed_seq,
                    ceiling,
                    "KV max-ctx ceiling exceeded — rejecting decode step"
                );
                return Err(Error::KvCeilingExceeded {
                    requested: needed_seq,
                    ceiling,
                });
            }
        }

        let current_max_seq = storage_max_seq(&self.storage);
        if needed_seq <= current_max_seq {
            return Ok(());
        }

        // Same next-power-of-two policy as the prefill grow, so one rule covers
        // both phases. This writes a scalar; it reallocs nothing by itself. What
        // it buys differs per store: the bf16 mirrors size themselves straight
        // off `max_seq`, so they realloc once per doubling, while the packed
        // rings round to whole `KV_PAGE_SIZE` pages and realloc on their own
        // page cadence regardless of the doubling. `needed_seq <= ceiling` is
        // guaranteed above, so the clamp still fits the request.
        let new_max_seq = match self.max_seq_ceiling {
            Some(ceiling) => next_pow2_seq(needed_seq).min(ceiling),
            None => next_pow2_seq(needed_seq),
        };

        // info!, matching the prefill grow: a capacity commit fires O(log n)
        // times per request (not per step) and re-sizes the bf16 mirrors, so it
        // belongs in the default run log.
        tracing::info!(
            from = current_max_seq,
            to = new_max_seq,
            needed_seq,
            "KV decode buffer grow"
        );

        set_storage_max_seq(&mut self.storage, new_max_seq);
        Ok(())
    }

    /// Append a prefill K/V chunk into the per-layer raw prefill buffer.
    ///
    /// # Buffer-grow contract
    ///
    /// The raw prefill buffer is allocated lazily on first call with shape
    /// `[B, kv_h, max_seq, head_dim]`, where `max_seq` is the value recorded
    /// on the active `KvStorage` variant. Before every write we check whether
    /// `prev_offset + new_seq` fits in the current buffer; if not, we grow
    /// the buffer to the next power-of-two ≥ needed (so subsequent chunks
    /// also fit without churn) and copy the existing filled prefix forward.
    /// The storage variant's `max_seq` field is bumped in lockstep so
    /// `exit_prefill` allocates downstream quantised buffers with the same
    /// new capacity.
    ///
    /// # Hard cap
    ///
    /// `RMLX_KV_MAX_SEQ_HARD_CAP` (env var) is an opt-in hard cap on the
    /// total prefill length. When set and exceeded by `needed_seq`, the
    /// call returns [`Error::KvHardCapExceeded`] **before** any allocation
    /// happens. When the env var is unset there is no cap.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(super) fn update_prefill_raw(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        // bf16 floor: the raw prefill buffer becomes the warm-TTFT decode seed
        // (the unquantised K/V the cache stores). Cast at the store boundary so
        // an upstream f32 leak cannot double resident KV — the capacity grow,
        // the buffer alloc, and the slice_update below all then see bf16.
        // Idempotent: a no-op when already bf16.
        let k_cast = cast_store_bf16(new_k, device)?;
        let v_cast = cast_store_bf16(new_v, device)?;
        let new_k = k_cast.as_ref().unwrap_or(new_k);
        let new_v = v_cast.as_ref().unwrap_or(new_v);

        let shape = new_k.shape();
        let b = shape[0];
        let kv_h = shape[1];
        let new_seq = shape[2];
        let head_dim = shape[3];

        let prev_offset = self.offset - new_seq;
        let new_offset = self.offset;

        // Enforce the optional hard cap before any allocation, then grow the
        // per-layer raw prefill buffer if the new chunk would overflow it.
        // The storage variant's `max_seq` is bumped in lockstep so the
        // downstream `exit_prefill` quantised buffers see the new capacity.
        self.ensure_prefill_capacity(
            new_offset,
            prev_offset,
            b,
            kv_h,
            head_dim,
            new_k.dtype(),
            new_v.dtype(),
            device,
        )?;

        let max_seq = storage_max_seq(&self.storage);

        if self.prefill_raw_k.is_none() {
            let buf_shape = [b, kv_h, max_seq, head_dim];
            self.prefill_raw_k = Some(zeros(&buf_shape, new_k.dtype(), device)?);
            self.prefill_raw_v = Some(zeros(&buf_shape, new_v.dtype(), device)?);
        }

        let k_buf = self.prefill_raw_k.as_mut().unwrap();
        let v_buf = self.prefill_raw_v.as_mut().unwrap();

        let ndim = 4usize;
        let mut start = vec![0i32; ndim];
        start[2] = prev_offset;
        let mut stop: Vec<i32> = [b, kv_h, 0i32, head_dim].into();
        stop[2] = new_offset;
        let strides = vec![1i32; ndim];

        let k_updated = k_buf.slice_update(new_k, &start, &stop, &strides, device)?;
        let v_updated = v_buf.slice_update(new_v, &start, &stop, &strides, device)?;
        *k_buf = k_updated;
        *v_buf = v_updated;
        let _ = k_buf.async_eval();
        let _ = v_buf.async_eval();

        let slice_start = vec![0i32; ndim];
        let slice_stop: Vec<i32> = [b, kv_h, new_offset, head_dim].into();
        let slice_strides = vec![1i32; ndim];
        let k_full = k_buf.slice(&slice_start, &slice_stop, &slice_strides, device)?;
        let v_full = v_buf.slice(&slice_start, &slice_stop, &slice_strides, device)?;

        Ok((k_full, v_full))
    }

    /// True while the cache sits between `enter_prefill` and `exit_prefill`.
    ///
    /// Observable for the sweep invariant: every cache that entered prefill must
    /// run `exit_prefill` before its prefill helper returns, on the failure path
    /// as well as the success path. A cache left in prefill keeps un-finalized
    /// state (no decode seed / un-quantized storage), and the next decode on it
    /// errors or corrupts KV. Rotating caches never enter prefill (the ring is
    /// the authority), so this stays false for them throughout.
    pub fn in_prefill(&self) -> bool {
        self.in_prefill
    }

    /// Switch the cache into prefill mode (accumulates raw K/V before quantizing).
    pub fn enter_prefill(&mut self) {
        // Rotating cache handles prefill via update_concat directly.
        // Skip the prefill_raw scaffolding so the ring buffer is the authority.
        if self.rotating.is_some() {
            return;
        }
        // Mixed cache also uses the fp16 prefill_raw scaffolding — store K/V
        // as fp16 during prefill, bulk-quantize to Mixed (k8/v4/g64) in
        // exit_prefill. This mirrors K8V4's pattern and avoids calling
        // mx.quantize on every 256-token prefill chunk, which
        // was the bottleneck causing 2077 ms cold TTFT vs 1353 ms champion.
        self.in_prefill = true;
        self.prefill_raw_k = None;
        self.prefill_raw_v = None;
    }

    #[allow(
        clippy::unreachable,
        reason = "each arm is guarded by a prior `match self.quant` that guarantees storage \
                  variant alignment; mismatch is a construction-time BUG, not a runtime condition"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    #[allow(
        unused_qualifications,
        reason = "pre-existing `crate::storage::QuantIsoK4::new` call sites are intentionally kept fully qualified to keep the dispatcher diff surgical; `QuantIsoK4` is imported for the new helper code only"
    )]
    /// Finalize prefill: quantize the accumulated raw K/V into the storage buffers.
    pub fn exit_prefill(&mut self, device: Device) -> Result<()> {
        // Snapshot before the storage borrows below; the bulk-quantize arms
        // hand it to the K encoders.
        let policy = self.policy;
        if self.rotating.is_some() {
            // No-op: rotating prefill writes go straight into the ring buffer.
            return Ok(());
        }

        // Paged storage uses the decode_fp16 seed for prefill (same as
        // `KvQuant::None` path) — the page allocator is populated lazily during
        // decode steps via `update_paged`. Materialise the compact fp16 seed
        // here and return early.
        if matches!(self.storage, KvStorage::Paged { .. }) {
            self.in_prefill = false;
            if let (Some(raw_k), Some(raw_v)) =
                (self.prefill_raw_k.take(), self.prefill_raw_v.take())
            {
                // Compact seed: clone filled slice only.
                let shape = raw_k.shape();
                let b = shape[0];
                let kv_h = shape[1];
                let head_dim = shape[3];
                let total_seq = self.offset;
                let sl_start = vec![0i32; 4];
                let sl_stop: Vec<i32> = [b, kv_h, total_seq, head_dim].into();
                let sl_strides = vec![1i32; 4];
                // `contiguous`: a bare slice stays a strided view over the raw
                // prefill buffer, which pins the parent and mis-serialises. See
                // the seed materialisation in the main path below.
                let k_buf = raw_k
                    .slice(&sl_start, &sl_stop, &sl_strides, device)?
                    .contiguous(device)?;
                let v_buf = raw_v
                    .slice(&sl_start, &sl_stop, &sl_strides, device)?
                    .contiguous(device)?;
                k_buf.eval()?;
                v_buf.eval()?;
                self.decode_fp16_k = Some(k_buf);
                self.decode_fp16_v = Some(v_buf);
            }
            return Ok(());
        }

        self.in_prefill = false;

        let (raw_k, raw_v) = match (self.prefill_raw_k.take(), self.prefill_raw_v.take()) {
            (Some(k), Some(v)) => (k, v),
            _ => return Ok(()),
        };

        // Quant paths: slice out the filled portion for quantization, and keep
        // a compact fp16 decode seed (compact seed, not full max_seq).
        //
        // Previously the full `max_seq`-sized raw buffer was cloned as the seed,
        // wasting up to 64 MB/layer when max_seq >> total_seq. Now we store
        // only the filled portion (`total_seq` tokens). `update_decode_fp16`
        // already handles compact seeds via the `needs_expand` path, expanding
        // to `max_seq` lazily on the first decode step.
        let shape = raw_k.shape();
        let b = shape[0];
        let kv_h = shape[1];
        let head_dim = shape[3];
        let total_seq = self.offset;

        let slice_start = vec![0i32; 4];
        let slice_stop: Vec<i32> = [b, kv_h, total_seq, head_dim].into();
        let slice_strides = vec![1i32; 4];
        let k_full = raw_k.slice(&slice_start, &slice_stop, &slice_strides, device)?;
        let v_full = raw_v.slice(&slice_start, &slice_stop, &slice_strides, device)?;

        // Compact seed: clone just the filled slice (total_seq tokens), not the
        // full max_seq-sized raw buffer. Saves up to (max_seq - total_seq) ×
        // B × kv_h × head_dim × 2 bytes per layer during the prefill phase.
        //
        // Each bf16 seed is only materialised when a consumer actually reads
        // it — either the `KvStorage::None` bf16 fallback below (which IS these
        // buffers, so it needs both) or a quant arm whose decode path honours
        // the `decode_fp16_{k,v}` shortcut. The K-only family reads no K seed;
        // the fused rotor symmetric codecs read neither, because their decode is
        // a flash kernel over both packed rings. An unread seed is not a small
        // waste: it is `total_seq * B * kv_h * head_dim * 2` bytes per layer per
        // axis, the dominant residency term at long context.
        //
        // `contiguous` and not `try_clone`: `k_full` is a slice of the raw
        // prefill buffer along the sequence axis, and MLX keeps that as a
        // strided view over the parent allocation — `eval` does not flatten it.
        // Two things follow that the seed's own contract denies. It is not
        // compact: the whole `max_seq` parent stays resident behind it until the
        // first decode step expands the mirror. And anything that reads the
        // buffer by raw linear offset rather than by stride — `Array::to_bytes`,
        // which is how the SSD tier serialises it — sees the parent's leading
        // bytes under the slice's shape, i.e. head 0's whole row window in place
        // of every head. Materialising row-major here, on the inference thread
        // that owns the Metal stream, is what makes the seed mean what its shape
        // says for every later reader.
        let is_bf16_storage = matches!(self.storage, KvStorage::None { .. });
        let need_k_seed = is_bf16_storage || self.quant.feeds_bf16_k_at_decode();
        let need_v_seed = is_bf16_storage || self.quant.feeds_bf16_v_at_decode();
        let k_buf = if need_k_seed {
            let k = k_full.contiguous(device)?;
            k.eval()?;
            Some(k)
        } else {
            None
        };
        let v_buf = if need_v_seed {
            let v = v_full.contiguous(device)?;
            v.eval()?;
            Some(v)
        } else {
            None
        };
        tracing::debug!(
            total_seq,
            max_seq = shape[2],
            k_seeded = need_k_seed,
            v_seeded = need_v_seed,
            "exit_prefill: compact fp16 decode seed materialised (total_seq tokens)"
        );
        let decode_fp16_pair = Some((k_buf, v_buf));

        // Guard: when the actual storage is KvStorage::None — which happens for
        // SWA layers that were hydrated from the SSD tier (they are stored as
        // tag "none" since the rotating bf16 ring cannot be serialised, but
        // from_storage() sets self.quant to the model's global KvQuant) — take
        // the bf16 path regardless of self.quant. The quantised dispatch arms
        // all pattern-match self.storage expecting their specific variant and hit
        // unreachable!() when they find KvStorage::None instead.
        if matches!(self.storage, KvStorage::None { .. }) {
            if let Some((k_seed, v_seed)) = decode_fp16_pair {
                // `is_bf16_storage` is true on this path, so both clones above
                // were materialised — these buffers *are* this codec's storage.
                self.decode_fp16_k = k_seed;
                self.decode_fp16_v = v_seed;
            }
            return Ok(());
        }

        // Bulk encode only what something will read. A codec that feeds both
        // axes from the bf16 mirror and has no decode path over its packed
        // store would write that store once here and then never touch it again
        // — a second full copy of the context, per layer, live for the whole
        // decode window, on top of a mirror that is already bf16-sized. The
        // classification is the codec's own
        // (`KvQuant::decode_reads_packed_store`), so this closes the class
        // rather than a codec at a time; the store is still built for the
        // families whose decode reads it, and a hydrated cache (which has no
        // mirror) still gets its store from the SSD block it was read from.
        if !self.quant.materialises_packed_store() {
            tracing::debug!(
                kv_quant = %self.quant,
                total_seq,
                "exit_prefill: packed store skipped — decode reads the bf16 mirror only"
            );
            // Not just "build nothing" — drop anything already there. Every
            // arm below *replaces* the payload, so before this gate a second
            // prefill on a cache that arrived carrying one (an SSD-hydrated
            // entry, deep-cloned and tail-extended; `enter_prefill` does not
            // clear `storage`) overwrote it. Returning early without clearing
            // would leave a store of the old length beside a mirror of the new
            // one, and the spill writer prefers the store — so the block would
            // be written under the full prompt's hash while holding only the
            // prefix.
            self.storage.clear_payload();
            if let Some((k_seed, v_seed)) = decode_fp16_pair {
                self.decode_fp16_k = k_seed;
                self.decode_fp16_v = v_seed;
            }
            return Ok(());
        }

        // REACHABILITY, as of the gate above: only the arms for codecs whose
        // `materialises_packed_store()` is true run. That is `Mixed`, `RotK`,
        // `RotKTq4V`, `IsoKOnly3/4`, `RotorKOnly3/4`, `Iso3Sym`, `Iso4Sym`,
        // `Rotor3Sym`, `Rotor4Sym` — nine of the arms below. The rest are the
        // bf16-mirror family and the gate returns before them.
        //
        // They are kept, not deleted, because they ARE the re-enable path: a
        // codec that grows a decode kernel over its own packed store flips one
        // arm in `decode_reads_packed_store` and this bulk encode is what then
        // fills the buffer that kernel reads (see `docs/KV_CACHE.md` §9.6 —
        // `planar_flash_decode` and the fused quant-decode work are both
        // waiting on exactly that flip). The hazard that creates is real: a
        // flipped predicate re-arms code that has had no execution since. The
        // pairing guard is
        // `warm_ttft_cross_codec_tests::exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`,
        // which sweeps every variant and fails the moment a codec's arm and its
        // classification disagree.
        match self.quant {
            // ── mirror-family group: NOT reachable today ────────────────────
            // The arms from here down that belong to the bf16-mirror family
            // (`K8V8`, `K8V4`, `Planar*`, `PlanarK`, `K8VTurbo*`, `TurboSym*`,
            // `Iso3/4`, `Rotor3/4`, `RotorK*Asym`) are behind the gate above.
            // The nine listed there are the ones that still run.
            KvQuant::K8V8 => {
                let max_seq = match &self.storage {
                    KvStorage::K8V8 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::KvStorageMismatch {
                            expected: "K8V8",
                            got: storage_variant_name(&self.storage),
                        })
                    }
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = if device == Device::Gpu {
                    (Vec::new(), Vec::new())
                } else {
                    arrays_to_f32(&k_full, &v_full, device)?
                };

                let KvStorage::K8V8 { k, v, .. } = &mut self.storage else {
                    return Err(Error::KvStorageMismatch {
                        expected: "K8V8",
                        got: storage_variant_name(&self.storage),
                    });
                };
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
                let mut qv = QuantK {
                    codes: Vec::new(),
                    scales: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape,
                    max_seq,
                };
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                qv.append(&v_f32, &new_shape, &v_full, device, max_seq)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            KvQuant::K8V4 => {
                let max_seq = match &self.storage {
                    KvStorage::K8V4 { max_seq, .. } => *max_seq,
                    _ => unreachable!(),
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = if device == Device::Gpu {
                    (Vec::new(), Vec::new())
                } else {
                    arrays_to_f32(&k_full, &v_full, device)?
                };

                let KvStorage::K8V4 { k, v, .. } = &mut self.storage else {
                    unreachable!("KvQuant::K8V4 but storage is not K8V4");
                };
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
                let mut qv = QuantV {
                    blocks: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape,
                    bits: 4,
                    max_seq,
                    high_precision_indices: None,
                    value_codebook: None,
                    value_codebook_gpu: None,
                    use_tcq: false,
                };
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                qv.append(&v_f32, &new_shape, &v_full, device, max_seq)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            KvQuant::None => {
                // BF16 KV path. The `raw_k`/`raw_v` buffers — already
                // sized to `[B, kv_h, max_seq, head_dim]` by `update_prefill_raw`
                // — are exactly the decode buffers we need. Promote them
                // directly into `decode_fp16_k`/`decode_fp16_v`; subsequent
                // `update_none` calls hit `update_decode_fp16` and slice_update
                // at the current offset. No quantize/dequantize work.
                self.decode_fp16_k = Some(raw_k.try_clone()?);
                self.decode_fp16_v = Some(raw_v.try_clone()?);
                return Ok(());
            }
            KvQuant::Planar | KvQuant::Planar3 => {
                let (max_seq, v_bits) = match &self.storage {
                    KvStorage::Planar { max_seq, bits, .. } => (*max_seq, *bits),
                    _ => unreachable!(),
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = if device == Device::Gpu {
                    (Vec::new(), Vec::new())
                } else {
                    arrays_to_f32(&k_full, &v_full, device)?
                };

                let KvStorage::Planar { k, v, .. } = &mut self.storage else {
                    unreachable!("KvQuant::Planar/Planar3 but storage is not Planar");
                };
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
                let mut qpv = QuantPlanarV {
                    blocks: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_rotations_buf: None,
                    gpu_codes_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_rotations_words_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape,
                    max_seq,
                    bits: v_bits,
                };
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                qpv.append(&v_f32, &new_shape, &v_full, device, max_seq)?;
                *k = Some(qk);
                *v = Some(qpv);
            }
            // Mixed cache uses the fp16 prefill_raw scaffolding. Bulk-quantize
            // the accumulated fp16 K/V (total_seq tokens) into the Mixed state
            // directly (skips the zero-alloc + 6×slice_update round-trip that
            // the per-token path pays on large prefill prefixes).
            KvQuant::Mixed { .. } | KvQuant::RotK { .. } => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill Mixed/RotK: bulk-quantizing fp16 prefill K/V"
                );
                let state = match &mut self.storage {
                    KvStorage::Mixed { state, .. } => state,
                    _ => unreachable!("Mixed-path quant but storage is not Mixed"),
                };
                // Reset so bulk_init_from_fp16 starts clean. reset() preserves
                // `k_rotation` so RotK keeps rotating K post-prefill.
                state.reset();
                // Directly quantize and store — no zero-buffer alloc, no
                // write_at, no slice_seq_to. offset is set to total_seq.
                state.bulk_init_from_fp16(&k_full, &v_full, device, policy)?;
                // The compact fp16 seed for warm-TTFT decode is the same buffer
                // already materialised above (decode_fp16_pair).
            }
            // RotKTq4V — bulk-quantize K into the rotated MixedKvState,
            // and V into TurboFlash QuantV, from the accumulated fp16 prefix.
            KvQuant::RotKTq4V => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill RotKTq4V: bulk-quantizing rotated K + tq4 V"
                );
                let new_shape = k_full.shape();
                // _k_f32 unused: bulk_init_k_from_fp16 takes the Array directly.
                // v_f32 is for QuantV::append CPU path.
                let (_k_f32, v_f32) = if device == Device::Gpu {
                    (Vec::new(), Vec::new())
                } else {
                    arrays_to_f32(&k_full, &v_full, device)?
                };
                let KvStorage::RotKTq4V {
                    k_state,
                    v,
                    max_seq,
                } = &mut self.storage
                else {
                    // -review HIGH 2: typed error instead of unreachable!() panic.
                    let variant = storage_variant_name(&self.storage);
                    return Err(Error::Mlx(format!(
                        "RotKTq4V dispatch: storage variant mismatch (quant=RotKTq4V, \
                         storage={variant}); cache may have been hydrated from incompatible spill"
                    )));
                };
                let max_seq = *max_seq;
                // K-side: use MixedKvState bulk init (rotate + affine-quantize K).
                k_state.reset();
                // For bulk K init we only need the K side — call bulk_init_from_fp16
                // with a dummy zero V (it quantizes V too but we replace V below).
                // Instead, use rotate_k_and_quantize directly + store K.
                let (k_codes, k_scales, k_biases) =
                    k_state.bulk_init_k_from_fp16(&k_full, device, policy)?;
                k_state.keys = Some(crate::mixed_quant::MixedTuple {
                    codes: k_codes,
                    scales: k_scales,
                    biases: k_biases,
                });
                k_state.offset = new_shape[2];

                // V-side: quantize into TurboFlash QuantV.
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qv = QuantV {
                    blocks: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape,
                    bits: 4,
                    max_seq,
                    high_precision_indices: None,
                    value_codebook: None,
                    value_codebook_gpu: None,
                    use_tcq: false,
                };
                qv.append(&v_f32, &new_shape, &v_full, device, max_seq)?;
                *v = Some(qv);
            }
            // K8VTurbo3 — bulk-quantize K (affine q8_0) + V (TurboQuant 3-bit).
            // CPU dequant only: no GPU path for 3-bit V (no MSL kernel this pass).
            KvQuant::K8VTurbo3 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill K8VTurbo3: bulk-quantizing K (q8_0) + V (turbo3)"
                );
                let max_seq = match &self.storage {
                    KvStorage::K8VTurbo3 { max_seq, .. } => *max_seq,
                    _ => unreachable!("KvQuant::K8VTurbo3 but storage is not K8VTurbo3"),
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::K8VTurbo3 { k, v, .. } = &mut self.storage else {
                    unreachable!("KvQuant::K8VTurbo3 but storage is not K8VTurbo3");
                };
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
                let mut qv = QuantV {
                    blocks: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape,
                    bits: 3,
                    max_seq,
                    high_precision_indices: None,
                    value_codebook: None,
                    value_codebook_gpu: None,
                    use_tcq: false,
                };
                // K-side: GPU affine q8_0 (same path as K8V4/K8V8).
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                // V-side: CPU TurboQuant 3-bit (GPU path not yet available for bits=3).
                qv.append(&v_f32, &new_shape, &v_full, Device::Cpu, max_seq)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // TurboSym3 — symmetric WHT-3 K + turbo3 V.
            // K side uses the GPU turbo3 MSL kernel (Decision B); V side forced CPU
            // (K8VTurbo3 precedent: GPU V-side dispatch regressed −2% TPS gate).
            KvQuant::TurboSym3 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill TurboSym3: bulk-quantizing K (turbo3/GPU) + V (turbo3/CPU)"
                );
                let max_seq = match &self.storage {
                    KvStorage::TurboSym3 { max_seq, .. } => *max_seq,
                    _ => unreachable!("KvQuant::TurboSym3 but storage is not TurboSym3"),
                };
                let new_shape = k_full.shape();
                // K GPU-capable; V CPU-forced.
                let k_f32 = if device == Device::Gpu {
                    Vec::new()
                } else {
                    array_to_f32_vec(&k_full, device)?
                };
                let v_f32 = array_to_f32_vec(&v_full, Device::Cpu)?;

                let KvStorage::TurboSym3 { k, v, .. } = &mut self.storage else {
                    unreachable!("KvQuant::TurboSym3 but storage is not TurboSym3");
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk = QuantKTurbo3::new(init_shape.clone(), max_seq);
                let mut qv = QuantV {
                    blocks: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape,
                    bits: 3,
                    max_seq,
                    high_precision_indices: None,
                    value_codebook: None,
                    value_codebook_gpu: None,
                    use_tcq: false,
                };
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                // V-side: force CPU path for 3-bit (GPU kernel wired but disabled;
                // see update_k8vturbo3 doc-comment for the −2% gate fail).
                qv.append(&v_f32, &new_shape, &v_full, Device::Cpu, max_seq)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // TurboSym4 — symmetric WHT-4 K + tq4 V. Both axes are
            // bulk-quantized via the same MSL kernel (axis-agnostic).
            KvQuant::TurboSym4 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill TurboSym4: bulk-quantizing K (tq4) + V (tq4)"
                );
                let max_seq = match &self.storage {
                    KvStorage::TurboSym4 { max_seq, .. } => *max_seq,
                    _ => unreachable!("KvQuant::TurboSym4 but storage is not TurboSym4"),
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = if device == Device::Gpu {
                    (Vec::new(), Vec::new())
                } else {
                    arrays_to_f32(&k_full, &v_full, device)?
                };

                let KvStorage::TurboSym4 { k, v, .. } = &mut self.storage else {
                    unreachable!("KvQuant::TurboSym4 but storage is not TurboSym4");
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk = QuantKTurbo4 {
                    blocks: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape.clone(),
                    bits: 4,
                    max_seq,
                };
                let mut qv = QuantV {
                    blocks: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape,
                    bits: 4,
                    max_seq,
                    high_precision_indices: None,
                    value_codebook: None,
                    value_codebook_gpu: None,
                    use_tcq: false,
                };
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                qv.append(&v_f32, &new_shape, &v_full, device, max_seq)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // PlanarK — bulk-quantize K via QuantPlanarK; V stays
            // bf16 (materialised by decode_fp16_pair below).
            KvQuant::PlanarK => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill PlanarK: bulk-quantizing K (planar4); V stays bf16"
                );
                let max_seq = match &self.storage {
                    KvStorage::PlanarK { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::KvStorageMismatch {
                            expected: "PlanarK",
                            got: storage_variant_name(&self.storage),
                        })
                    }
                };
                let new_shape = k_full.shape();
                let k_f32 = if device == Device::Gpu {
                    Vec::new()
                } else {
                    array_to_f32_vec(&k_full, device)?
                };

                let KvStorage::PlanarK { k, .. } = &mut self.storage else {
                    return Err(Error::KvStorageMismatch {
                        expected: "PlanarK",
                        got: storage_variant_name(&self.storage),
                    });
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qpk = QuantPlanarK::new(init_shape, max_seq);
                qpk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                *k = Some(qpk);
                // V side: bf16 — materialized by decode_fp16_pair below.
            }
            // K8VTurbo2 — same shape as K8VTurbo3 with bits=2.
            // CPU dequant only (the turbo2 MSL kernel is a future-reference
            // hook, mirroring K8VTurbo3; see `turbo2_v_msl.rs`).
            KvQuant::K8VTurbo2 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill K8VTurbo2: bulk-quantizing K (q8_0) + V (turbo2)"
                );
                let max_seq = match &self.storage {
                    KvStorage::K8VTurbo2 { max_seq, .. } => *max_seq,
                    _ => return Err(Error::Mlx(
                        "K8VTurbo2 exit_prefill: storage mismatch (KvQuant::K8VTurbo2 but storage is not K8VTurbo2)".into()
                    )),
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::K8VTurbo2 { k, v, .. } = &mut self.storage else {
                    return Err(Error::Mlx(
                        "K8VTurbo2 exit_prefill: storage mismatch (KvQuant::K8VTurbo2 but storage is not K8VTurbo2)".into()
                    ));
                };
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
                let mut qv = QuantV {
                    blocks: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape,
                    bits: 2,
                    max_seq,
                    high_precision_indices: None,
                    value_codebook: None,
                    value_codebook_gpu: None,
                    use_tcq: false,
                };
                // K-side: GPU affine q8_0 (same path as K8V4/K8V8).
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                // V-side: CPU TurboQuant 2-bit (GPU path not wired on hot path).
                qv.append(&v_f32, &new_shape, &v_full, Device::Cpu, max_seq)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // Iso4 — bulk-quantize K (affine q8_0) + V (IsoQuant 4-bit, CPU only).
            KvQuant::Iso4 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill Iso4: bulk-quantizing K (q8_0) + V (iso4 CPU)"
                );
                let max_seq = match &self.storage {
                    KvStorage::IsoV4 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::Mlx(
                            "Iso4 exit_prefill: storage mismatch (KvQuant::Iso4 but storage is not IsoV4)".into()
                        ))
                    }
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::IsoV4 { k, v, .. } = &mut self.storage else {
                    return Err(Error::Mlx("Iso4 exit_prefill: storage mismatch".into()));
                };
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
                let mut qv = QuantIsoV4::new(init_shape, max_seq);
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                qv.append(&v_f32, &new_shape)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // Iso3 — bulk-quantize K (affine q8_0) + V (IsoQuant 3-bit, CPU only).
            KvQuant::Iso3 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill Iso3: bulk-quantizing K (q8_0) + V (iso3 CPU)"
                );
                let max_seq = match &self.storage {
                    KvStorage::IsoV3 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::Mlx(
                            "Iso3 exit_prefill: storage mismatch (KvQuant::Iso3 but storage is not IsoV3)".into()
                        ))
                    }
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::IsoV3 { k, v, .. } = &mut self.storage else {
                    return Err(Error::Mlx("Iso3 exit_prefill: storage mismatch".into()));
                };
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
                let mut qv = QuantIsoV3::new(init_shape);
                // K-side: GPU affine q8_0 (same path as K8V4/K8V8).
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                // V-side: CPU IsoQuant 3-bit (no GPU path until T11d).
                //
                // Per-phase trace instrumentation for the iso3 V
                // exit_prefill encode (where the V-side codec actually runs
                // in the current build — decode hot-path is shadowed by the
                // bf16 warm-TTFT seed, see docs/PERF_BASELINE.md).
                let kv_h = new_shape[1];
                let head_dim = new_shape[3];
                let t_enc = std::time::Instant::now();
                qv.append(&v_f32, &new_shape)?;
                tracing::trace!(
                    phase = "iso3_encode",
                    ms = t_enc.elapsed().as_secs_f64() * 1e3,
                    s_total = new_shape[2],
                    kv_h,
                    head_dim,
                    site = "exit_prefill",
                    "iso3 hot-path"
                );
                *k = Some(qk);
                *v = Some(qv);
            }
            // Rotor3 — bulk-quantize K (affine q8_0) + V (rotor3, CPU only).
            KvQuant::Rotor3 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill Rotor3: bulk-quantizing K (q8_0) + V (rotor3 CPU)"
                );
                let max_seq = match &self.storage {
                    KvStorage::RotorV3 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::Mlx(
                            "Rotor3 exit_prefill: storage mismatch (KvQuant::Rotor3 but storage is not RotorV3)".into()
                        ))
                    }
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::RotorV3 { k, v, .. } = &mut self.storage else {
                    return Err(Error::Mlx("Rotor3 exit_prefill: storage mismatch".into()));
                };
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
                let mut qv = QuantRotorV3::new(init_shape, max_seq, layer_idx_u32(self.layer_idx));
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                qv.append(&v_f32, &new_shape)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // Rotor4 — bulk-quantize K (affine q8_0) + V (rotor4, CPU only).
            KvQuant::Rotor4 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill Rotor4: bulk-quantizing K (q8_0) + V (rotor4 CPU)"
                );
                let max_seq = match &self.storage {
                    KvStorage::RotorV4 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::Mlx(
                            "Rotor4 exit_prefill: storage mismatch (KvQuant::Rotor4 but storage is not RotorV4)".into()
                        ))
                    }
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::RotorV4 { k, v, .. } = &mut self.storage else {
                    return Err(Error::Mlx("Rotor4 exit_prefill: storage mismatch".into()));
                };
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
                let mut qv = QuantRotorV4::new(init_shape, max_seq, layer_idx_u32(self.layer_idx));
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                qv.append(&v_f32, &new_shape)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // K8VTurbo3Tcq — bulk-quantize K (affine q8_0) + V
            // (TurboQuant 3-bit with Viterbi assignment). Mirrors the K8VTurbo3
            // arm with `use_tcq = true` on the V slot so the encode dispatch in
            // QuantV::append picks the Viterbi path.
            KvQuant::K8VTurbo3Tcq => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill K8VTurbo3Tcq: bulk-quantizing K (q8_0) + V (turbo3 + Viterbi)"
                );
                let max_seq = match &self.storage {
                    KvStorage::K8VTurbo3Tcq { max_seq, .. } => *max_seq,
                    _ => unreachable!("KvQuant::K8VTurbo3Tcq but storage is not K8VTurbo3Tcq"),
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::K8VTurbo3Tcq { k, v, .. } = &mut self.storage else {
                    unreachable!("KvQuant::K8VTurbo3Tcq but storage is not K8VTurbo3Tcq");
                };
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
                let mut qv = QuantV {
                    blocks: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape,
                    bits: 3,
                    max_seq,
                    high_precision_indices: None,
                    value_codebook: None,
                    value_codebook_gpu: None,
                    use_tcq: true,
                };
                // K-side: GPU affine q8_0 (same path as K8V4/K8V8).
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                // V-side: CPU TurboQuant 3-bit with Viterbi assignment.
                qv.append(&v_f32, &new_shape, &v_full, Device::Cpu, max_seq)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // K8VTurbo2Tcq — bulk-quantize K (affine q8_0) + V
            // (TurboQuant 2-bit with Viterbi assignment). Mirrors K8VTurbo3Tcq
            // with bits=2 and the `turbo2_tcq` max_compression preset.
            KvQuant::K8VTurbo2Tcq => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill K8VTurbo2Tcq: bulk-quantizing K (q8_0) + V (turbo2 + Viterbi)"
                );
                let max_seq = match &self.storage {
                    KvStorage::K8VTurbo2Tcq { max_seq, .. } => *max_seq,
                    _ => unreachable!("KvQuant::K8VTurbo2Tcq but storage is not K8VTurbo2Tcq"),
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::K8VTurbo2Tcq { k, v, .. } = &mut self.storage else {
                    unreachable!("KvQuant::K8VTurbo2Tcq but storage is not K8VTurbo2Tcq");
                };
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
                let mut qv = QuantV {
                    blocks: Vec::new(),
                    gpu_codes_buf: None,
                    gpu_scales_buf: None,
                    gpu_words_per_step: 0,
                    gpu_scales_per_step: 0,
                    gpu_capacity: 0,
                    shape: init_shape,
                    bits: 2,
                    max_seq,
                    high_precision_indices: None,
                    value_codebook: None,
                    value_codebook_gpu: None,
                    use_tcq: true,
                };
                // K-side: GPU affine q8_0 (same path as K8V4/K8V8).
                qk.append(&k_f32, &new_shape, &k_full, device, max_seq)?;
                // V-side: CPU TurboQuant 2-bit with Viterbi assignment.
                qv.append(&v_f32, &new_shape, &v_full, Device::Cpu, max_seq)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // Iso3Sym — bulk-quantize K (iso3, CPU) + V (iso3, CPU).
            // The codec is axis-agnostic; only the role on the SDPA path differs.
            KvQuant::Iso3Sym => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill Iso3Sym: bulk-quantizing K + V (both iso3 CPU)"
                );
                let max_seq = match &self.storage {
                    KvStorage::IsoSym3 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::Mlx(
                            "Iso3Sym exit_prefill: storage mismatch (KvQuant::Iso3Sym but storage is not IsoSym3)".into()
                        ))
                    }
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::IsoSym3 { k, v, .. } = &mut self.storage else {
                    return Err(Error::Mlx("Iso3Sym exit_prefill: storage mismatch".into()));
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk = crate::storage::QuantIsoK3::new(init_shape.clone(), max_seq);
                let mut qv = QuantIsoV3::new(init_shape);
                qk.append(&k_f32, &new_shape)?;
                qv.append(&v_f32, &new_shape)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // Iso4Sym — bulk-quantize K (iso4) + V (iso4).
            KvQuant::Iso4Sym => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill Iso4Sym: bulk-quantizing K + V (both iso4 CPU)"
                );
                let max_seq = match &self.storage {
                    KvStorage::IsoSym4 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::Mlx(
                            "Iso4Sym exit_prefill: storage mismatch (KvQuant::Iso4Sym but storage is not IsoSym4)".into()
                        ))
                    }
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::IsoSym4 { k, v, .. } = &mut self.storage else {
                    return Err(Error::Mlx("Iso4Sym exit_prefill: storage mismatch".into()));
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk = crate::storage::QuantIsoK4::new(init_shape.clone(), max_seq);
                let mut qv = QuantIsoV4::new(init_shape, max_seq);
                qk.append(&k_f32, &new_shape)?;
                qv.append(&v_f32, &new_shape)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // IsoKOnly3 — bulk-quantize K (iso3, CPU); V stays bf16
            // (materialised by `decode_fp16_pair` below, same machinery as PlanarK).
            KvQuant::IsoKOnly3 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill IsoKOnly3: bulk-quantizing K (iso3 CPU); V stays bf16"
                );
                let max_seq = match &self.storage {
                    KvStorage::IsoKOnly3 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::Mlx(
                            "IsoKOnly3 exit_prefill: storage mismatch (KvQuant::IsoKOnly3 but storage is not IsoKOnly3)".into()
                        ))
                    }
                };
                let new_shape = k_full.shape();
                let k_f32 = array_to_f32_vec(&k_full, device)?;

                let KvStorage::IsoKOnly3 { k, .. } = &mut self.storage else {
                    return Err(Error::Mlx(
                        "IsoKOnly3 exit_prefill: storage mismatch".into(),
                    ));
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk = crate::storage::QuantIsoK3::new(init_shape, max_seq);
                qk.append(&k_f32, &new_shape)?;
                *k = Some(qk);
                // V-side: bf16 — materialised by decode_fp16_pair below.
            }
            // IsoKOnly4 — bulk-quantize K (iso4 CPU); V stays bf16.
            KvQuant::IsoKOnly4 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill IsoKOnly4: bulk-quantizing K (iso4 CPU); V stays bf16"
                );
                let max_seq = match &self.storage {
                    KvStorage::IsoKOnly4 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::Mlx(
                            "IsoKOnly4 exit_prefill: storage mismatch (KvQuant::IsoKOnly4 but storage is not IsoKOnly4)".into()
                        ))
                    }
                };
                let new_shape = k_full.shape();
                let k_f32 = array_to_f32_vec(&k_full, device)?;

                let KvStorage::IsoKOnly4 { k, .. } = &mut self.storage else {
                    return Err(Error::Mlx(
                        "IsoKOnly4 exit_prefill: storage mismatch".into(),
                    ));
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk = crate::storage::QuantIsoK4::new(init_shape, max_seq);
                qk.append(&k_f32, &new_shape)?;
                *k = Some(qk);
            }
            // Rotor3Sym — bulk-quantize K (rotor3 CPU, optional QJL)
            // + V (rotor3 CPU). Mirrors `KvQuant::Iso3Sym` with rotor3 codecs.
            KvQuant::Rotor3Sym => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill Rotor3Sym: bulk-quantizing K + V (both rotor3 CPU)"
                );
                let max_seq = match &self.storage {
                    KvStorage::RotorSym3 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::Mlx(
                            "Rotor3Sym exit_prefill: storage mismatch (KvQuant::Rotor3Sym but storage is not RotorSym3)".into()
                        ))
                    }
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::RotorSym3 { k, v, .. } = &mut self.storage else {
                    return Err(Error::Mlx(
                        "Rotor3Sym exit_prefill: storage mismatch".into(),
                    ));
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk = crate::storage::QuantRotorK3::new(
                    init_shape.clone(),
                    layer_idx_u32(self.layer_idx),
                );
                let mut qv = QuantRotorV3::new(init_shape, max_seq, layer_idx_u32(self.layer_idx));
                qk.append(&k_f32, &new_shape)?;
                qv.append(&v_f32, &new_shape)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // Rotor4Sym — bulk-quantize K (rotor4 CPU, optional QJL)
            // + V (rotor4 CPU). Mirrors `KvQuant::Iso4Sym`.
            KvQuant::Rotor4Sym => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill Rotor4Sym: bulk-quantizing K + V (both rotor4 CPU)"
                );
                let max_seq = match &self.storage {
                    KvStorage::RotorSym4 { max_seq, .. } => *max_seq,
                    _ => {
                        return Err(Error::Mlx(
                            "Rotor4Sym exit_prefill: storage mismatch (KvQuant::Rotor4Sym but storage is not RotorSym4)".into()
                        ))
                    }
                };
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = arrays_to_f32(&k_full, &v_full, device)?;

                let KvStorage::RotorSym4 { k, v, .. } = &mut self.storage else {
                    return Err(Error::Mlx(
                        "Rotor4Sym exit_prefill: storage mismatch".into(),
                    ));
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk = crate::storage::QuantRotorK4::new(
                    init_shape.clone(),
                    layer_idx_u32(self.layer_idx),
                );
                let mut qv = QuantRotorV4::new(init_shape, max_seq, layer_idx_u32(self.layer_idx));
                qk.append(&k_f32, &new_shape)?;
                qv.append(&v_f32, &new_shape)?;
                *k = Some(qk);
                *v = Some(qv);
            }
            // RotorKOnly3 — bulk-quantize K (rotor3 CPU, optional QJL);
            // V stays bf16 on `decode_fp16_v`.
            KvQuant::RotorKOnly3 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill RotorKOnly3: bulk-quantizing K (rotor3 CPU); V stays bf16"
                );
                let new_shape = k_full.shape();
                let k_f32 = array_to_f32_vec(&k_full, device)?;

                let KvStorage::RotorKOnly3 { k, .. } = &mut self.storage else {
                    return Err(Error::Mlx(
                        "RotorKOnly3 exit_prefill: storage mismatch".into(),
                    ));
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk =
                    crate::storage::QuantRotorK3::new(init_shape, layer_idx_u32(self.layer_idx));
                qk.append(&k_f32, &new_shape)?;
                *k = Some(qk);
            }
            // RotorKOnly4 — bulk-quantize K (rotor4 CPU, optional QJL);
            // V stays bf16.
            KvQuant::RotorKOnly4 => {
                tracing::debug!(
                    total_seq,
                    "exit_prefill RotorKOnly4: bulk-quantizing K (rotor4 CPU); V stays bf16"
                );
                let new_shape = k_full.shape();
                let k_f32 = array_to_f32_vec(&k_full, device)?;

                let KvStorage::RotorKOnly4 { k, .. } = &mut self.storage else {
                    return Err(Error::Mlx(
                        "RotorKOnly4 exit_prefill: storage mismatch".into(),
                    ));
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk =
                    crate::storage::QuantRotorK4::new(init_shape, layer_idx_u32(self.layer_idx));
                qk.append(&k_f32, &new_shape)?;
                *k = Some(qk);
            }
            // RotorK3Asym — bulk-quantize K (rotor3 CPU, optional QJL)
            // and V (affine `v_bits` / `v_group_size` via the existing QuantV
            // path). Mirrors the RotorKOnly3 K side + the K8V4 V side.
            KvQuant::RotorK3Asym { .. } => {
                let (max_seq, v_bits, v_group_size) = match &self.storage {
                    KvStorage::RotorKAsym3 {
                        max_seq,
                        v_bits,
                        v_group_size,
                        ..
                    } => (*max_seq, *v_bits, *v_group_size),
                    _ => {
                        return Err(Error::Mlx(
                            "RotorK3Asym exit_prefill: storage mismatch (KvQuant::RotorK3Asym but storage is not RotorKAsym3)".into()
                        ))
                    }
                };
                tracing::debug!(
                    total_seq,
                    v_bits,
                    v_group_size,
                    "exit_prefill RotorK3Asym: bulk-quantizing K (rotor3 CPU) + V (affine)"
                );
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = if device == Device::Gpu {
                    (array_to_f32_vec(&k_full, device)?, Vec::new())
                } else {
                    arrays_to_f32(&k_full, &v_full, device)?
                };

                let KvStorage::RotorKAsym3 { k, v, .. } = &mut self.storage else {
                    return Err(Error::KvStorageMismatch {
                        expected: "RotorKAsym3",
                        got: storage_variant_name(&self.storage),
                    });
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk = crate::storage::QuantRotorK3::new(
                    init_shape.clone(),
                    layer_idx_u32(self.layer_idx),
                );
                let mut qv = QuantV::new_affine_decode(init_shape, v_bits, max_seq);
                qk.append(&k_f32, &new_shape)?;
                qv.append(&v_f32, &new_shape, &v_full, device, max_seq)?;
                *k = Some(qk);
                *v = Some(qv);
                let _ = v_group_size;
            }
            // RotorK4Asym — mirror of RotorK3Asym with rotor4 K.
            KvQuant::RotorK4Asym { .. } => {
                let (max_seq, v_bits, v_group_size) = match &self.storage {
                    KvStorage::RotorKAsym4 {
                        max_seq,
                        v_bits,
                        v_group_size,
                        ..
                    } => (*max_seq, *v_bits, *v_group_size),
                    _ => {
                        return Err(Error::Mlx(
                            "RotorK4Asym exit_prefill: storage mismatch (KvQuant::RotorK4Asym but storage is not RotorKAsym4)".into()
                        ))
                    }
                };
                tracing::debug!(
                    total_seq,
                    v_bits,
                    v_group_size,
                    "exit_prefill RotorK4Asym: bulk-quantizing K (rotor4 CPU) + V (affine)"
                );
                let new_shape = k_full.shape();
                let (k_f32, v_f32) = if device == Device::Gpu {
                    (array_to_f32_vec(&k_full, device)?, Vec::new())
                } else {
                    arrays_to_f32(&k_full, &v_full, device)?
                };

                let KvStorage::RotorKAsym4 { k, v, .. } = &mut self.storage else {
                    return Err(Error::KvStorageMismatch {
                        expected: "RotorKAsym4",
                        got: storage_variant_name(&self.storage),
                    });
                };
                let mut init_shape = new_shape.clone();
                init_shape[2] = 0;
                let mut qk = crate::storage::QuantRotorK4::new(
                    init_shape.clone(),
                    layer_idx_u32(self.layer_idx),
                );
                let mut qv = QuantV::new_affine_decode(init_shape, v_bits, max_seq);
                qk.append(&k_f32, &new_shape)?;
                qv.append(&v_f32, &new_shape, &v_full, device, max_seq)?;
                *k = Some(qk);
                *v = Some(qv);
                let _ = v_group_size;
            }
        }

        // Warm-TTFT seed: the shortcut quant arms get the bf16 K+V decode
        // mirror that `update_decode_fp16` reads via the
        // `decode_fp16_k.is_some()` shortcut. See docs/KV_CACHE.md §9.6.
        //
        // For the K-only family (IsoKOnly3/4, RotorKOnly3/4) the K codec runs
        // every decode step and never reads `decode_fp16_k`, so populating the
        // bf16 K seed was dead memory; they still read the bf16 **V** seed via
        // `update_decode_fp16_v_only`. The fused rotor symmetric codecs
        // (Rotor3Sym/Rotor4Sym) read neither — their decode is a flash kernel
        // over both packed rings — so both seeds are dropped. Pure RAM reclaim;
        // output unchanged.
        if let Some((k_buf, v_buf)) = decode_fp16_pair {
            // Each is `Some` iff the matching `feeds_bf16_{k,v}_at_decode()`
            // said this codec's decode actually reads it.
            self.decode_fp16_k = k_buf;
            self.decode_fp16_v = v_buf;
        }

        Ok(())
    }

    /// Live-inference KV resident bytes held by this cache at the call-site.
    ///
    /// Reports the bytes of the K/V that actually serves decode — the *filled*
    /// prefix of each buffer, not its pre-allocated capacity. The bf16 decode
    /// mirrors (`decode_fp16_k/v`) are sized to the max-context ceiling, so the
    /// seq-scaled buffers are counted by their filled length (`offset`,
    /// clamped to the buffer's capacity) rather than the whole allocation.
    /// This keeps the figure consistent across contexts and configurations so
    /// bytes-per-KV-token is comparable: a run with a large ceiling no longer
    /// reports more KV than its active cache. Sums:
    ///
    /// - **Quantized storage** (`KvStorage::resident_bytes`): packed codes,
    ///   scales, rotation buffers, etc. for all codec variants. Already
    ///   compacted to the filled length at `exit_prefill`. Returns 0 for
    ///   `KvStorage::None` (bf16 buffers live in `decode_fp16_k/v` below).
    /// - **fp16 decode seeds** (`decode_fp16_k`, `decode_fp16_v`): warm-TTFT
    ///   bf16 mirrors present on quantized paths; also the sole storage for
    ///   `KvStorage::None` (bf16). Counted by filled length, not capacity.
    /// - **Rotating SWA ring** (`rotating`): pre-allocated bf16 `[B, kv_h,
    ///   window, D]` buffer used by SWA layers; counted by filled length
    ///   (≤ the sliding window).
    /// - **TurboFlash head-major buffers** (`flash_k_codes/scales`,
    ///   `flash_v_codes/scales`): lazy copy of the K/V cache in head-major
    ///   layout for the TurboFlash MSL SDPA kernel.
    /// - **Fused-QK shadow** (`fused_qk_shadow`): head-major K shadow used by
    ///   fused-QK dispatch (q8, turbo3/4-sym, iso3/4-sym, rotor3/4-sym, …).
    ///
    /// The per-position size comes from the actual `Array` shape × dtype of
    /// each live buffer (picking up per-layer head_dim differences, e.g.
    /// windowed vs full-attention layers), scaled by the filled length. There
    /// is deliberately no quant-bit formula anywhere in this total: a codec's
    /// bytes are whatever its store actually allocated, which only the store
    /// can answer.
    ///
    /// **Cost: O(blocks).** Asking the store is not free — the block-based
    /// codecs walk a `Vec` that grows by one entry per decode step. Call this
    /// at request boundaries (as the `kv_bytes` event does), not per-layer
    /// per-decode-step: that would make a generation quadratic in context.
    /// Do not "fix" the cost with a cached running counter — a byte total kept
    /// alongside the buffers instead of read from them is the drifting mirror
    /// this accounting exists to avoid.
    ///
    /// Returns 0 when the cache has never been used (`offset == 0` and no
    /// buffers are allocated). Safe to call at any point — no FFI eval, no
    /// data read, no mutation.
    ///
    /// The exhaustive destructure below is the drift guard: a new buffer cannot
    /// be added to `KvCache` without this failing to compile.
    pub fn resident_bytes(&self) -> u64 {
        use crate::bytes::{array_bytes, filled_seq_bytes};

        // Naming every field is what makes this total drift-proof: a buffer
        // added to `KvCache` cannot slip past the accounting unnoticed, because
        // the pattern stops compiling until it is classified here.
        let Self {
            storage,
            offset,
            decode_fp16_k,
            decode_fp16_v,
            rotating,
            flash_k_codes,
            flash_k_scales,
            flash_v_codes,
            flash_v_scales,
            fused_qk_shadow,
            // Transient prefill staging: borrowed views handed straight to the
            // encoder and dropped at `exit_prefill`, not cache residency.
            prefill_raw_k: _,
            prefill_raw_v: _,
            // Configuration / bookkeeping, not allocations.
            quant: _,
            layer_idx: _,
            in_prefill: _,
            stream_dtype: _,
            flash_max_seq: _,
            flash_filled: _,
            max_seq_ceiling: _,
            policy: _,
        } = self;

        let offset = (*offset).max(0) as u64;

        // 1. Quantized storage (codec-specific buffers; None → 0). The store
        //    owns its own byte total, GPU rings and mirrors included.
        let mut total = storage.resident_bytes();

        // 2. fp16 decode seeds (KvQuant::None bf16 storage lives here too).
        //    Count only the filled prefix, not the ceiling-sized allocation.
        if let Some(k) = decode_fp16_k {
            total += filled_seq_bytes(k, offset);
        }
        if let Some(v) = decode_fp16_v {
            total += filled_seq_bytes(v, offset);
        }

        // 3. Rotating SWA ring buffer (bf16, sized to the sliding window). The
        //    ring owns its own byte total: reaching its buffers by field access
        //    from here would let a buffer added to it go uncounted.
        if let Some(rot) = rotating {
            total += rot.byte_size(offset);
        }

        // 4. TurboFlash head-major K/V buffers (lazy; only for K8V4 path).
        if let Some(c) = flash_k_codes {
            total += array_bytes(c);
        }
        if let Some(s) = flash_k_scales {
            total += array_bytes(s);
        }
        if let Some(c) = flash_v_codes {
            total += array_bytes(c);
        }
        if let Some(s) = flash_v_scales {
            total += array_bytes(s);
        }

        // 5. Fused-QK shadow (head-major K shadow for fused-QK MSL kernels).
        if let Some(shadow) = fused_qk_shadow {
            total += shadow.byte_size();
        }

        total
    }

    /// Reset the cache to an empty state (offset → 0, buffers zeroed).
    pub fn reset(&mut self) {
        if let Some(ref mut rot) = self.rotating {
            rot.reset();
        }
        self.storage.reset();
        self.offset = 0;
        self.in_prefill = false;
        self.prefill_raw_k = None;
        self.prefill_raw_v = None;
        self.decode_fp16_k = None;
        self.decode_fp16_v = None;
        // Drop the head-major TurboFlash buffers. They will be
        // re-seeded from the next request's prefill on first decode dispatch.
        self.flash_k_codes = None;
        self.flash_k_scales = None;
        self.flash_v_codes = None;
        self.flash_v_scales = None;
        self.flash_max_seq = 0;
        self.flash_filled = 0;
        // Drop the head-major fused-QK shadow. Re-seeded from the
        // next request's prefill on first fused-QK decode dispatch.
        self.fused_qk_shadow = None;
    }

    /// Truncate the cache to `n` positions.
    ///
    /// Drops accumulated K/V state past position `n` so that the next
    /// `update` call writes at offset `n`. Preserves the decode_fp16
    /// buffers — they are pre-allocated to `max_seq` length and
    /// positions `[0..n]` remain valid; positions `[n..]` are stale
    /// but won't be sliced since SDPA reads `[0..offset]`.
    ///
    /// For `KvQuant::None` (bf16 KV path) the decode_fp16
    /// buffers ARE the storage; dropping them would lose prefill data.
    /// For quantised paths (K8V4/K8V8/Planar) the decode_fp16 buffers
    /// are the warm-TTFT seed; the quantised storage's
    /// `shape[2]` is also lowered to `n` via `storage.truncate_to`.
    pub fn truncate_to(&mut self, n: i32) {
        debug_assert!(
            n <= self.offset,
            "KvCache::truncate_to: n={n} > offset={}",
            self.offset
        );
        // Roll the fused-QK shadow's filled count back on BOTH the rotating
        // and non-rotating paths. Today `try_fused_qk_dispatch` gates
        // rotating storage out via `storage_max_seq_for_fused_qk`, so the
        // rotating branch should never have a shadow allocated — but the
        // assertion below makes
        // that explicit and the truncate call keeps the shadow filled
        // count in sync if a future change ever wires rotating into the
        // shadow path.
        if let Some(ref mut shadow) = self.fused_qk_shadow {
            debug_assert!(
                self.rotating.is_none(),
                "rotating cache should never have a fused-QK shadow allocated \
                 (storage_max_seq_for_fused_qk returns None for rotating variants)"
            );
            shadow.truncate_to(n);
        }
        // Port mlx-lm `RotatingKVCache.trim` semantics (cache.py:542-549).
        // mlx-lm's `trim_prompt_cache(cache, num)` decrements offset/idx
        // losslessly when `is_trimmable` (offset < max_size); otherwise
        // silently returns 0 (cache.py:109-111). Speculative decoding
        // tolerates this — the Gemma 31b spec test in mlx-lm relies on it.
        if let Some(ref mut rot) = self.rotating {
            let delta = self.offset - n;
            let _trimmed = rot.trim_lossless(delta);
            // Mirror rot's offset back onto KvCache.offset. If `trimmed == 0`
            // (post-rotation), `rot.offset` is unchanged and KvCache.offset
            // must stay where it was — matches mlx-lm's no-op behaviour.
            self.offset = rot.offset;
            return;
        }
        self.storage.truncate_to(n);
        self.offset = n;
        self.in_prefill = false;
        // Keep the GPU buffer allocation but mark only `n` tokens valid.
        // Subsequent `update_and_sdpa_k8v4_flash` calls will overwrite
        // positions `[n..]` head-major as new tokens stream in.
        if self.flash_filled > n {
            self.flash_filled = n;
        }
    }

    /// Force evaluation of any pending MLX lazy operations in the KV buffers.
    pub fn eval_gpu_state(&self) -> Result<()> {
        // Rotating ring buffer holds K/V on its own arrays.
        if let Some(ref rot) = self.rotating {
            if let Some(k) = &rot.keys {
                k.eval()?;
            }
            if let Some(v) = &rot.values {
                v.eval()?;
            }
            return Ok(());
        }
        match &self.storage {
            KvStorage::K8V4 { k, v, .. } => {
                if let Some(qk) = k {
                    if let Some(codes) = &qk.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qk.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
                if let Some(qv) = v {
                    if let Some(codes) = &qv.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qv.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
            }
            KvStorage::K8V8 { k, v, .. } => {
                if let Some(qk) = k {
                    if let Some(codes) = &qk.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qk.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
                if let Some(qv) = v {
                    if let Some(codes) = &qv.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qv.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
            }
            KvStorage::Planar { k, v, .. } => {
                if let Some(qk) = k {
                    if let Some(codes) = &qk.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qk.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
                if let Some(qv) = v {
                    if let Some(codes) = &qv.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qv.gpu_scales_buf {
                        scales.eval()?;
                    }
                    if let Some(rotations) = &qv.gpu_rotations_buf {
                        rotations.eval()?;
                    }
                }
            }
            KvStorage::None { .. } => {
                // BF16 KV — buffers live on `decode_fp16_k`/`decode_fp16_v`
                // and are eval'd by the trailing block below.
            }
            KvStorage::Mixed { state, .. } => {
                state.eval_gpu_state()?;
            }
            // Paged storage — the active page arrays live inside PageSlab::pool.
            // They are already async_eval'd by the slice_update chain inside write_page.
            // No additional flush needed here beyond the decode_fp16 trailing block.
            KvStorage::Paged { .. } => {}
            // RotKTq4V — flush the MixedKvState (affine K) and the QuantV.
            // -review MEDIUM 1: flush codes/scales independently so a mid-
            // allocation window (one Some, one None) still flushes the available buf.
            KvStorage::RotKTq4V { k_state, v, .. } => {
                k_state.eval_gpu_state()?;
                if let Some(qv) = v {
                    if let Some(codes) = &qv.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qv.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
            }
            // K8VTurbo3 — flush K (QuantK) and V (QuantV, bits=3, CPU-dequant only).
            KvStorage::K8VTurbo3 { k, v, .. } => {
                if let Some(qk) = k {
                    if let Some(codes) = &qk.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qk.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
                if let Some(qv) = v {
                    if let Some(codes) = &qv.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qv.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
            }
            // TurboSym4 — flush both TurboQuant K + V GPU buffers.
            KvStorage::TurboSym4 { k, v, .. } => {
                if let Some(qk) = k {
                    if let Some(codes) = &qk.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qk.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
                if let Some(qv) = v {
                    if let Some(codes) = &qv.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qv.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
            }
            // PlanarK — flush K (codes/scales/rotations); V is bf16
            // (decode_fp16_v) and is flushed by the trailing block below.
            KvStorage::PlanarK { k, .. } => {
                if let Some(qk) = k {
                    if let Some(codes) = &qk.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qk.gpu_scales_buf {
                        scales.eval()?;
                    }
                    if let Some(rotations) = &qk.gpu_rotations_buf {
                        rotations.eval()?;
                    }
                }
            }
            // K8VTurbo2 — K is QuantK (codes/scales), V is QuantV (CPU-only).
            KvStorage::K8VTurbo2 { k, v, .. } => {
                if let Some(qk) = k {
                    if let Some(codes) = &qk.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qk.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
                if let Some(qv) = v {
                    if let Some(codes) = &qv.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qv.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
            }
            // IsoV3 / IsoV4 / RotorV3 / RotorV4 — K is QuantK (GPU-capable
            // q8_0); V is CPU-only payload.
            KvStorage::IsoV3 { k, .. }
            | KvStorage::IsoV4 { k, .. }
            | KvStorage::RotorV3 { k, .. }
            | KvStorage::RotorV4 { k, .. } => {
                // K is GPU-capable q8_0 (`QuantK`); V is CPU-only so no GPU
                // buffers to flush on the V side.
                if let Some(qk) = k {
                    if let Some(codes) = &qk.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qk.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
            }
            // K8VTurbo3Tcq — flush like K8VTurbo3 (K is GPU-capable q8_0,
            // V is CPU-only QuantV with Viterbi encode; gpu_codes_buf
            // may exist from a hydrated cache).
            KvStorage::K8VTurbo3Tcq { k, v, .. } => {
                if let Some(qk) = k {
                    if let Some(codes) = &qk.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qk.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
                if let Some(qv) = v {
                    if let Some(codes) = &qv.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qv.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
            }
            // IsoSym3 / IsoSym4 / IsoKOnly3 / IsoKOnly4 — CPU-only
            // codecs (no GPU buffers on either axis). V-side IsoKOnly* is bf16,
            // flushed by the trailing decode_fp16 block below.
            KvStorage::IsoSym3 { .. }
            | KvStorage::IsoSym4 { .. }
            | KvStorage::IsoKOnly3 { .. }
            | KvStorage::IsoKOnly4 { .. } => {}
            // Rotor symmetric / K-only — CPU-only codecs, no GPU buffers on
            // either axis. V-side RotorKOnly* is bf16, flushed by the trailing
            // decode_fp16 block below.
            KvStorage::RotorSym3 { .. }
            | KvStorage::RotorSym4 { .. }
            | KvStorage::RotorKOnly3 { .. }
            | KvStorage::RotorKOnly4 { .. } => {}
            // RotorKAsym3 / RotorKAsym4 — K rotor is CPU-only; V is
            // affine QuantV with optional GPU codes/scales buffers (flushed
            // like the K8V4 V side).
            KvStorage::RotorKAsym3 { v, .. } | KvStorage::RotorKAsym4 { v, .. } => {
                if let Some(qv) = v {
                    if let Some(codes) = &qv.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qv.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
            }
            // TurboSym3 — K is GPU-capable (QuantKTurbo3 shares the same flush
            // pattern as TurboSym4 K side); V is CPU-only (no GPU buffers).
            // NOTE: No explicit eval() needed here. MLX is lazy — QuantKTurbo3::append
            // builds the compute graph; the Metal encoder is flushed when the K buffer
            // is first read (e.g. dequantize_choice). Omitting eval() is correct and
            // consistent with how IsoSym3/RotorSym3 K-side GPU buffers are handled.
            KvStorage::TurboSym3 { .. } => {}
            // K8VTurbo2Tcq — flush like K8VTurbo2 / K8VTurbo3Tcq.
            KvStorage::K8VTurbo2Tcq { k, v, .. } => {
                if let Some(qk) = k {
                    if let Some(codes) = &qk.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qk.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
                if let Some(qv) = v {
                    if let Some(codes) = &qv.gpu_codes_buf {
                        codes.eval()?;
                    }
                    if let Some(scales) = &qv.gpu_scales_buf {
                        scales.eval()?;
                    }
                }
            }
        }
        if let Some(buf) = &self.decode_fp16_k {
            buf.eval()?;
        }
        if let Some(buf) = &self.decode_fp16_v {
            buf.eval()?;
        }
        if let Some(buf) = &self.prefill_raw_k {
            buf.eval()?;
        }
        if let Some(buf) = &self.prefill_raw_v {
            buf.eval()?;
        }
        Ok(())
    }

    /// Force materialization of just the prefill raw buffers, the cheap subset
    /// of `eval_gpu_state` used to flush the Metal command buffer between
    /// prefill chunks. Skips the lm_head logits projection by not touching
    /// the forward pass output.
    pub fn eval_prefill_state(&self) -> Result<()> {
        // Flush rotating ring buffer between prefill chunks.
        if let Some(ref rot) = self.rotating {
            if let Some(k) = &rot.keys {
                k.eval()?;
            }
            if let Some(v) = &rot.values {
                v.eval()?;
            }
            return Ok(());
        }
        // Flush mixed-quant 3-tuple buffers between prefill chunks.
        if let KvStorage::Mixed { state, .. } = &self.storage {
            state.eval_gpu_state()?;
            return Ok(());
        }
        if let Some(buf) = &self.prefill_raw_k {
            buf.eval()?;
        }
        if let Some(buf) = &self.prefill_raw_v {
            buf.eval()?;
        }
        Ok(())
    }

    /// Warm-TTFT decode step (`0806148`). Serves K **and** V from the
    /// pre-expanded bf16 mirror (`decode_fp16_k`/`decode_fp16_v`) via one
    /// `slice_update` per step — no per-token quantize/dequant.
    ///
    /// This is the universal decode shortcut: every quantized `update_<codec>`
    /// (K8V*, Mixed, Planar*, Turbo*, Iso*Sym, Rotor*Sym, RotorK*Asym, …)
    /// early-returns here when `self.decode_fp16_k.is_some()` (always, post
    /// `exit_prefill`). Decode-phase K/V are bf16 and the packed store is not
    /// consulted at decode-read time — which is why, for a codec that reads no
    /// store at all ([`KvQuant::materialises_packed_store`] is `false`),
    /// `exit_prefill` does not build one and these mirrors are the cache's
    /// entire residency. The codecs that *do* read their store never reach
    /// here for the axes they read. See the architectural contract + per-codec
    /// audit table in `docs/KV_CACHE.md` §9.6.
    ///
    /// Exceptions: the K-only family (`IsoKOnly*`, `RotorKOnly*`) does NOT
    /// route here for K — it quantizes K every decode step and uses
    /// [`Self::update_decode_fp16_v_only`] for V instead (calling this full
    /// helper would re-arm the K shortcut and silently drop K to bf16).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub(super) fn update_decode_fp16(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        max_seq: i32,
        device: Device,
    ) -> Result<(Array, Array)> {
        // bf16 floor: the unquantised / warm-TTFT decode mirror is bf16 by
        // contract. Cast at the store boundary so an upstream f32 leak can never
        // double resident KV (idempotent — a no-op when already bf16). The
        // resulting `dtype` below then sizes the resident buffer in bf16 too.
        let k_cast = cast_store_bf16(new_k, device)?;
        let v_cast = cast_store_bf16(new_v, device)?;
        let new_k = k_cast.as_ref().unwrap_or(new_k);
        let new_v = v_cast.as_ref().unwrap_or(new_v);

        let shape = new_k.shape();
        let b = shape[0];
        let kv_h = shape[1];
        let new_seq = shape[2];
        let head_dim = shape[3];
        let dtype = new_k.dtype();

        let mut prev_offset = self.offset - new_seq;
        let mut new_offset = self.offset;

        let buf_shape = [b, kv_h, max_seq, head_dim];
        let needs_expand = match &self.decode_fp16_k {
            None => true,
            Some(k) => k.shape()[2] < max_seq,
        };

        // SWA hydration guard: KvStorage::None layers after SSD hydration carry
        // the block's seq_len as self.offset (needed for RoPE base_offset on the
        // model side) but hold no actual K/V data (the rotating ring buffer cannot
        // be spilled). On the first decode step after hydration, decode_fp16_k is
        // None AND prev_offset may exceed max_seq (e.g. prev_offset=1023 vs
        // max_seq=512 for Gemma4-e2b SWA layers). Writing at position 1023 into a
        // buffer of size 512 produces a broadcast-shape error in mlx slice_update.
        //
        // Fix: when there is no existing K data AND prev_offset would overflow the
        // buffer, reset prev_offset/new_offset relative to 0. The SWA cache
        // effectively starts fresh — the phantom prefix tokens were never spilled.
        if needs_expand && self.decode_fp16_k.is_none() && prev_offset >= max_seq {
            // H5: tracing event for SWA hydration offset reset (emit before mutation).
            tracing::warn!(
                layer_max_seq = max_seq,
                old_offset = prev_offset,
                new_offset = new_seq,
                "SWA hydration: resetting out-of-range offset (phantom prefix discarded)"
            );
            // Also update self.offset so subsequent decode steps use correct positions.
            self.offset = new_seq;
            prev_offset = 0;
            new_offset = new_seq;
        }
        if needs_expand {
            // Task 12: kv_alloc event — fires on first allocation (first_step)
            // and on lazy expansion from compact seed to full max_seq buffer
            // (grow). Does NOT fire on normal decode steps — only when the
            // buffer is newly allocated or expanded.
            let cause = if self.decode_fp16_k.is_none() {
                "first_step"
            } else {
                "grow"
            };
            // bf16 per element = 2 bytes; K + V = 2 arrays.
            let kv_bytes_allocated =
                b as u64 * kv_h as u64 * max_seq as u64 * head_dim as u64 * 2 * 2;
            tracing::debug!(
                kv_bytes_allocated,
                cause,
                offset = self.offset,
                max_seq,
                "kv_alloc"
            );
            let k_zeros = zeros(&buf_shape, dtype, device)?;
            let v_zeros = zeros(&buf_shape, dtype, device)?;
            let k_buf = if let Some(seed) = self.decode_fp16_k.take() {
                let seed_seq = seed.shape()[2];
                let mut seed_start = vec![0i32; 4];
                seed_start[2] = 0;
                let seed_stop: Vec<i32> = [b, kv_h, seed_seq, head_dim].into();
                let seed_strides = vec![1i32; 4];
                k_zeros.slice_update(&seed, &seed_start, &seed_stop, &seed_strides, device)?
            } else {
                k_zeros
            };
            let v_buf = if let Some(seed) = self.decode_fp16_v.take() {
                let seed_seq = seed.shape()[2];
                let mut seed_start = vec![0i32; 4];
                seed_start[2] = 0;
                let seed_stop: Vec<i32> = [b, kv_h, seed_seq, head_dim].into();
                let seed_strides = vec![1i32; 4];
                v_zeros.slice_update(&seed, &seed_start, &seed_stop, &seed_strides, device)?
            } else {
                v_zeros
            };
            self.decode_fp16_k = Some(k_buf);
            self.decode_fp16_v = Some(v_buf);
        }

        let k_buf = self.decode_fp16_k.as_mut().unwrap();
        let v_buf = self.decode_fp16_v.as_mut().unwrap();

        let ndim = 4usize;
        let mut start = vec![0i32; ndim];
        start[2] = prev_offset;
        let mut stop: Vec<i32> = [b, kv_h, 0i32, head_dim].into();
        stop[2] = new_offset;
        let strides = vec![1i32; ndim];

        let k_updated = k_buf.slice_update(new_k, &start, &stop, &strides, device)?;
        let v_updated = v_buf.slice_update(new_v, &start, &stop, &strides, device)?;
        *k_buf = k_updated;
        *v_buf = v_updated;
        let _ = k_buf.async_eval();
        let _ = v_buf.async_eval();

        let slice_start = vec![0i32; ndim];
        let slice_stop: Vec<i32> = [b, kv_h, new_offset, head_dim].into();
        let slice_strides = vec![1i32; ndim];
        let k_full = k_buf.slice(&slice_start, &slice_stop, &slice_strides, device)?;
        let v_full = v_buf.slice(&slice_start, &slice_stop, &slice_strides, device)?;

        Ok((k_full, v_full))
    }

    /// V-only bf16 update helper for IsoKOnly variants.
    ///
    /// Mirrors [`Self::update_decode_fp16`] but **only** manages the V side
    /// (`decode_fp16_v`). It deliberately does NOT touch `self.decode_fp16_k`.
    ///
    /// This is the correct helper for `update_iso_k_only_3` /
    /// `update_iso_k_only_4`: those codecs own K in the ISO quantized buffer
    /// while V stays bf16. Calling the full `update_decode_fp16` would
    /// populate `self.decode_fp16_k` as a side-effect, causing the
    /// `decode_fp16_k.is_some()` early-return guard in `update_iso3_sym` /
    /// `update_iso4_sym` to short-circuit the codec on the *next* decode step
    /// (silent bf16-K regression guarded by regression test).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "decode_fp16_v is Some by construction: the needs_expand branch above always sets it, and this path is only reached after that guard"
    )]
    pub(super) fn update_decode_fp16_v_only(
        &mut self,
        new_v: &Array,
        max_seq: i32,
        device: Device,
    ) -> Result<Array> {
        let shape = new_v.shape();
        let b = shape[0];
        let kv_h = shape[1];
        let new_seq = shape[2];
        let head_dim = shape[3];
        let dtype = new_v.dtype();

        let mut prev_offset = self.offset - new_seq;
        let mut new_offset = self.offset;

        let buf_shape = [b, kv_h, max_seq, head_dim];
        let needs_expand = match &self.decode_fp16_v {
            None => true,
            Some(v) => v.shape()[2] < max_seq,
        };

        // Mirror the SWA hydration guard from update_decode_fp16: when there is
        // no existing V data AND prev_offset overflows max_seq (e.g. SWA layer
        // hydrated with phantom-prefix offset), reset to a fresh position.
        if needs_expand && self.decode_fp16_v.is_none() && prev_offset >= max_seq {
            self.offset = new_seq;
            prev_offset = 0;
            new_offset = new_seq;
        }
        if needs_expand {
            let v_zeros = zeros(&buf_shape, dtype, device)?;
            let v_buf = if let Some(seed) = self.decode_fp16_v.take() {
                let seed_seq = seed.shape()[2];
                let mut seed_start = vec![0i32; 4];
                seed_start[2] = 0;
                let seed_stop: Vec<i32> = [b, kv_h, seed_seq, head_dim].into();
                let seed_strides = vec![1i32; 4];
                v_zeros.slice_update(&seed, &seed_start, &seed_stop, &seed_strides, device)?
            } else {
                v_zeros
            };
            self.decode_fp16_v = Some(v_buf);
        }

        let v_buf = self.decode_fp16_v.as_mut().unwrap();

        let ndim = 4usize;
        let mut start = vec![0i32; ndim];
        start[2] = prev_offset;
        let mut stop: Vec<i32> = [b, kv_h, 0i32, head_dim].into();
        stop[2] = new_offset;
        let strides = vec![1i32; ndim];

        let v_updated = v_buf.slice_update(new_v, &start, &stop, &strides, device)?;
        *v_buf = v_updated;
        let _ = v_buf.async_eval();

        let slice_start = vec![0i32; ndim];
        let slice_stop: Vec<i32> = [b, kv_h, new_offset, head_dim].into();
        let slice_strides = vec![1i32; ndim];
        let v_full = v_buf.slice(&slice_start, &slice_stop, &slice_strides, device)?;

        Ok(v_full)
    }

    #[allow(
        clippy::unreachable,
        reason = "storage variant is guaranteed by the `match &self.storage` dispatch in \
                  KvCache::update() (KvStorage::K8V4 arm); \
                  mismatch is a construction-time BUG"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn update_k8v4(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::K8V4 { k, v, max_seq } = &mut self.storage else {
            unreachable!("storage mismatch: expected K8V4");
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        let (k_f32, v_f32) = if device == Device::Gpu {
            (Vec::new(), Vec::new())
        } else {
            arrays_to_f32(new_k, new_v, device)?
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
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 4,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: false,
            });
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

    /// K8VTurbo3 decode update — K = affine q8_0, V = TurboQuant 3-bit
    /// Lloyd-Max (codebook from `turboquant::lloyd_gaussian_codebook(3)`).
    ///
    /// Mirrors `update_k8v4` but with `bits=3` on the V side. added a
    /// Metal 3-bit kernel (`k8vturbo3_append_msl`) but bench showed it
    /// regresses Gemma4-e4b by ~3.5% and Gemma4-26b by ~6.9% vs the
    /// `Mixed{v_bits:3}` affine baseline — both fail the −2% TPS
    /// gate. The GPU dispatch wiring was therefore reverted; the V side
    /// stays on CPU here, exactly as in . The MSL kernel source is
    /// retained in `k8vturbo3_append_msl.rs` as a future-reference hook
    /// (still unit-tested for bit-equivalence) — see
    /// `docs/research/turboquant_v3_vs_affine_v3.md` "Second pass" for the
    /// bench numbers.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn update_k8vturbo3(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::K8VTurbo3 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected K8VTurbo3, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as K8V4).
        // V-side: force CPU for 3-bit (: GPU dispatch failed the −2% gate;
        // dispatch was reverted, kernel source kept as future-reference hook).
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };
        let v_f32 = array_to_f32_vec(new_v, Device::Cpu)?;

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
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 3,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: false,
            });
        }
        let vs = v.as_mut().unwrap();
        // V-side: force CPU path for 3-bit (GPU kernel wired but disabled;
        // see `update_k8vturbo3` doc-comment + research doc for the −2% gate fail).
        vs.append(&v_f32, &new_shape, new_v, Device::Cpu, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, _) = vs.dequantize_choice(Device::Cpu, new_v.dtype())?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// TurboSym3 decode update — K = `QuantKTurbo3` (GPU-capable
    /// turbo3 MSL kernel), V = `QuantV` (bits=3, **CPU-forced**; GPU V-side
    /// dispatch failed the −2% TPS gate on K8VTurbo3, see `update_k8vturbo3`
    /// doc-comment).
    ///
    /// Mirrors `update_tsym4` for the K side with `bits=3`; mirrors
    /// `update_k8vturbo3` for the V side (forced `Device::Cpu`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn update_tsym3(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::TurboSym3 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected TurboSym3, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        tracing::trace!(quant = "tsym3", "update_tsym3: decode step");

        let new_shape = new_k.shape();

        // K-side: GPU-capable turbo3 MSL kernel (Decision B: reuse existing
        // k8vturbo3 MSL kernel, axis-agnostic). V-side: force CPU for 3-bit
        // (GPU V-side dispatch regressed −2% TPS gate on K8VTurbo3).
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };
        let v_f32 = array_to_f32_vec(new_v, Device::Cpu)?;

        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantKTurbo3::new(init_shape, max_seq));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("TurboSym3 K buffer absent after init".into()));
        };
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
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 3,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: false,
            });
        }
        let Some(vs) = v.as_mut() else {
            return Err(Error::Mlx("TurboSym3 V buffer absent after init".into()));
        };
        // V-side: force CPU path for 3-bit (GPU kernel wired but disabled;
        // see doc-comment + K8VTurbo3 precedent for the −2% gate fail).
        vs.append(&v_f32, &new_shape, new_v, Device::Cpu, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, _) = vs.dequantize_choice(Device::Cpu, new_v.dtype())?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// K8VTurbo3Tcq decode update — K = affine q8_0,
    /// V = TurboQuant 3-bit with Viterbi trellis (TCQ) assignment.
    ///
    /// Mirrors [`Self::update_k8vturbo3`] structurally — same K-side q8_0
    /// dispatch, same CPU V-side path; only `QuantV::use_tcq` is set so that
    /// `QuantV::append` calls the Viterbi encoder
    /// ([`crate::tcq::tcq_quantize_v3`]) instead of nearest-centroid. The
    /// decoder is shared with plain turbo3 (TCQ output layout is byte-for-byte
    /// identical), so `dequantize_choice` is unchanged.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn update_k8vturbo3_tcq(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::K8VTurbo3Tcq { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected K8VTurbo3Tcq, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as K8V4 / K8VTurbo3).
        // V-side: force CPU — the TCQ MSL kernel ships as a future-ref hook
        // (see `tcq_v_msl.rs` Dispatch status).
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };
        let v_f32 = array_to_f32_vec(new_v, Device::Cpu)?;

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
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 3,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: true,
            });
        }
        let vs = v.as_mut().unwrap();
        // Force CPU on the TCQ encode path (Viterbi over the trellis).
        vs.append(&v_f32, &new_shape, new_v, Device::Cpu, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, _) = vs.dequantize_choice(Device::Cpu, new_v.dtype())?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// K8VTurbo2Tcq decode update — K = affine q8_0,
    /// V = TurboQuant **2-bit** with Viterbi trellis (TCQ) assignment.
    ///
    /// Structurally identical to [`Self::update_k8vturbo3_tcq`] — same K-side
    /// GPU-capable affine q8_0 path (same as K8V4 / K8VTurbo3), same forced-CPU
    /// V-side. TCQ is an **encode-side** Viterbi trellis assignment only; decode
    /// is bit-identical to plain turbo and reuses the turbo GPU dequant. The
    /// production V-encode runs forced-CPU here: the 2-bit TCQ encode has no
    /// wired MSL kernel (its hook was removed), while the 3-bit encode kernel in
    /// `tcq_v_msl.rs` stays parked. Only the V-side `bits` changes from 3 to 2.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Option is Some by construction: if k/v.is_none() block above guarantees Some on both branches"
    )]
    fn update_k8vturbo2_tcq(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::K8VTurbo2Tcq { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected K8VTurbo2Tcq, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as K8V4 / K8VTurbo3).
        // V-side: force CPU — the TCQ MSL kernel ships as a future-ref hook.
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };
        let v_f32 = array_to_f32_vec(new_v, Device::Cpu)?;

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
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 2,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: true,
            });
        }
        let vs = v.as_mut().unwrap();
        // Force CPU on the 2-bit TCQ encode path (Viterbi over the trellis).
        vs.append(&v_f32, &new_shape, new_v, Device::Cpu, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, _) = vs.dequantize_choice(Device::Cpu, new_v.dtype())?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// TurboSym4 decode update — symmetric WHT-4 K + tq4 V.
    ///
    /// Mirrors `update_k8v4` but the K side uses [`QuantKTurbo4`] (TurboQuant
    /// 4-bit) instead of `QuantK` (q8_0). Both K and V dispatch through the
    /// axis-agnostic `turbo_quantize_v4_gpu` / `turbo_dequantize_v4_gpu` MSL
    /// kernel — no kernel fork.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn update_tsym4(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::TurboSym4 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected TurboSym4, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        tracing::trace!(quant = "tsym4", "update_tsym4: decode step");

        let new_shape = new_k.shape();
        let (k_f32, v_f32) = if device == Device::Gpu {
            (Vec::new(), Vec::new())
        } else {
            arrays_to_f32(new_k, new_v, device)?
        };

        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantKTurbo4 {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 4,
                max_seq,
            });
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("TurboSym4 K buffer absent after init".into()));
        };
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
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 4,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: false,
            });
        }
        let Some(vs) = v.as_mut() else {
            return Err(Error::Mlx("TurboSym4 V buffer absent after init".into()));
        };
        vs.append(&v_f32, &new_shape, new_v, device, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, v_arr_opt) = vs.dequantize_choice(device, new_v.dtype())?;
        let v_full = match v_arr_opt {
            Some(arr) => arr,
            None => f32_vec_to_array(&v_recon_f32, &v_shape)?,
        };

        Ok((k_full, v_full))
    }

    /// K8VTurbo2 decode update — K = affine q8_0, V = TurboQuant
    /// 2-bit Lloyd-Max (codebook from `turboquant::lloyd_gaussian_codebook(2)`).
    ///
    /// Mirrors [`update_k8vturbo3`](Self::update_k8vturbo3) byte-for-byte with
    /// `bits=2` on the V side. CPU dequant only on the hot path: the MSL kernel
    /// in `turbo2_v_msl.rs` is wired as a future-reference hook (unit-tested
    /// for bit-exact CPU↔GPU parity) but never dispatched. The naïve Lloyd-Max
    /// 2-bit codebook ships without outlier-mask; outlier-mask deferred pending
    /// calibration loader. See `docs/KV_QUANT.md`
    /// for the gap-vs-mtq quantification.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn update_k8vturbo2(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::K8VTurbo2 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected K8VTurbo2, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as K8V4 / K8VTurbo3).
        // V-side: force CPU for 2-bit (no GPU dispatch on the hot path; the
        // MSL kernel is a future-reference hook in `turbo2_v_msl.rs`).
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };
        let v_f32 = array_to_f32_vec(new_v, Device::Cpu)?;

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
        // SAFETY: k is Some by construction — the `if k.is_none()` block above
        // assigns Some on both branches (initial + pre-existing), so this unwrap
        // cannot fail.
        #[allow(
            clippy::unwrap_used,
            reason = "Option is Some by construction: if k.is_none() block above guarantees Some on both branches"
        )]
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
            *v = Some(QuantV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                bits: 2,
                max_seq,
                high_precision_indices: None,
                value_codebook: None,
                value_codebook_gpu: None,
                use_tcq: false,
            });
        }
        // SAFETY: v is Some by construction — the `if v.is_none()` block above
        // assigns Some on both branches (initial + pre-existing), so this unwrap
        // cannot fail.
        #[allow(
            clippy::unwrap_used,
            reason = "Option is Some by construction: if v.is_none() block above guarantees Some on both branches"
        )]
        let vs = v.as_mut().unwrap();
        // V-side: force CPU path for 2-bit (no GPU dispatch on the hot path;
        // see `update_k8vturbo2` doc-comment + `turbo2_v_msl.rs` "Dispatch status").
        vs.append(&v_f32, &new_shape, new_v, Device::Cpu, max_seq)?;
        let v_shape = vs.shape.clone();
        let (v_recon_f32, _) = vs.dequantize_choice(Device::Cpu, new_v.dtype())?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// TurboFlash K8V4 path.
    ///
    /// Like `update_k8v4`, but when `turbo_flash_should_run()` is true AND the
    /// GPU buffers are available, dispatches `turbo_flash_sdpa` directly on the
    /// raw quantized K/V buffers — skipping the dequantize → SDPA → re-quantize
    /// round-trip that the standard path pays.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(output))` — TurboFlash ran; `output` is `[B, n_q_heads, 1, D]`.
    /// - `Ok(None)` — TurboFlash did not run (conditions not met or default-OFF);
    ///   caller should fall through to the standard `update_k8v4` + SDPA path.
    ///
    /// # Fallback conditions
    ///
    /// - `DispatchPolicy::turbo_flash` unset (default).
    /// - Smoke-probe forced fallback (corruption detected).
    /// - q_seq != 1 (prefill path — only decode is supported).
    /// - `kv_seq <= DispatchPolicy::turbo_flash_min_kv_seq` (below the split-K
    ///   crossover).
    /// - GPU buffers not yet populated (first call, before alloc).
    /// - head_dim ∉ {128, 256} (kernel register-array sizing constraint).
    ///
    /// **CAVEAT**: TheTom's original TurboFlash is default-OFF on Apple10 (M5+)
    /// due to corruption (commit `67f076f2e`, a default-flip — no upstream
    /// kernel fix exists). Empirically reproduced the M5 Max failure on rMLX's
    /// adaptation: a hard `SIGSEGV`/`KERN_INVALID_ADDRESS` (null
    /// `Buffer::raw_ptr()` in the kernel-output `to_bytes`) at 32k ctx on
    /// Qwen3.6-35B-A3B-8bit (head_dim=256) — worse than TheTom's
    /// garbage-token corruption (it crashes the server). Stays default-OFF.
    /// Setting `DispatchPolicy::turbo_flash` will crash on that cell. See
    /// `docs/reports/B1-turboflash-m5-validation.md`.
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
    pub fn update_and_sdpa_k8v4_flash(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Option<Array>> {
        self.update_and_sdpa_k8v4_flash_inner(
            queries,
            new_k,
            new_v,
            scale,
            additive_mask,
            false,
            device,
        )
    }

    /// TurboFlash entry point for cross-layer-KV consumers (Gemma4). Identical
    /// behaviour to [`Self::update_and_sdpa_k8v4_flash`] except
    /// `DispatchPolicy::turbo_flash_lock` is ignored: the bf16 `decode_fp16_k/v`
    /// mirror MUST stay current every decode step, because the caller
    /// (`update_and_sdpa_shared_source`) slices it to surface bf16 (K, V) for
    /// shared-KV consumer layers. Lock-on would freeze the mirror at the
    /// prefill prefix and silently drop decode tokens from the surfaced K/V.
    pub(super) fn update_and_sdpa_k8v4_flash_no_lock(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Option<Array>> {
        self.update_and_sdpa_k8v4_flash_inner(
            queries,
            new_k,
            new_v,
            scale,
            additive_mask,
            true,
            device,
        )
    }

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
    #[allow(clippy::too_many_arguments)]
    fn update_and_sdpa_k8v4_flash_inner(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        force_lock_off: bool,
        device: Device,
    ) -> Result<Option<Array>> {
        if !self.is_k8v4() {
            return Ok(None);
        }

        let q_shape = queries.shape();
        let q_seq = q_shape[2];
        let head_dim = q_shape[3];
        let new_seq = new_k.shape()[2];

        // Gating (BEFORE any state update so the caller can drive `c.update()`
        // cleanly on the fallback path).
        //
        // `kv_seq_after_update` is the offset the caller's `update()` would
        // produce — used for the `kv_seq > turbo_flash_min_kv_seq` gate.
        let kv_seq_after_update = self.offset + new_seq;
        if !turbo_flash_should_run(&self.policy, q_seq, kv_seq_after_update) {
            return Ok(None);
        }
        if head_dim != 128 && head_dim != 256 {
            tracing::debug!(
                "TurboFlash: head_dim={head_dim} not in {{128, 256}}, falling back to standard SDPA"
            );
            return Ok(None);
        }
        if device != Device::Gpu {
            return Ok(None);
        }

        // Keep state in lock-step with what `update()` would do:
        // 1. Bump `self.offset += new_seq`.
        // 2. Mirror the new K/V into `decode_fp16_k/v` (so a future fallback
        // step sees this token in the bf16 fast-path buffer).
        //
        // The head-major persistent K8V4 buffer (P2.A.1) is updated below, AFTER
        // we know the prefill seed has been materialised in `decode_fp16_k/v`.
        //
        // Cold-decode (no fp16 seed) is impossible in practice because the gate
        // `kv_seq_after_update > 4096` implies a prior prefill of >4K tokens,
        // and prefill's `exit_prefill` always populates `decode_fp16_k/v` for
        // K8V4. Bail before any state mutation so the caller's `c.update()`
        // fallback runs cleanly without double-counting.
        if self.decode_fp16_k.is_none() {
            return Ok(None);
        }
        // Grow the provisioned decode window before the head-major append, the
        // same rule the legacy `update()` path applies via `ensure_decode_capacity`.
        // This flash dispatch bypasses `update()`, so without growing here the
        // storage `max_seq` (and the bf16 mirror + latched flash buffers sized
        // off it) freeze at the prefill length; the append then walks off the
        // end at the next power-of-two boundary and slices an empty tensor
        // (surfacing downstream as a `reshape … size 0`). A request that
        // genuinely cannot fit is rejected loudly here (ceiling / hard cap)
        // rather than crashing mid-append.
        self.ensure_decode_capacity(kv_seq_after_update)?;
        let max_seq = match &self.storage {
            KvStorage::K8V4 { max_seq, .. } => *max_seq,
            _ => return Ok(None),
        };
        let prev_offset = self.offset;
        self.offset += new_seq;

        // ── Lock-on skip of `update_decode_fp16`
        //
        // When `DispatchPolicy::turbo_flash_lock` is set AND the persistent flash buffers are
        // already seeded (`flash_k_codes.is_some()`), the bf16 mirror is no
        // longer read by anyone — the kernel reads `flash_*` directly and the
        // request has opted out of standard-SDPA fallback. Skipping the bf16
        // maintenance call eliminates the single largest dispatch on the hot
        // path (one full `slice_update` over `[B, kv_h, max_seq, D]` bf16).
        //
        // First dispatch still pays the bf16 update because the seed for
        // `flash_*` is quantised from `decode_fp16_k/v`. After that the
        // mirror is frozen at the prefill prefix.
        // Cross-layer-KV consumers (Gemma4) require the bf16 mirror to be
        // updated every step so `update_and_sdpa_shared_source` can slice it
        // back to the consumer. `force_lock_off` short-circuits the lock-on
        // optimisation in that case. Non-cross-layer-KV callers (the public
        // entry point) keep the optimisation.
        let lock_on =
            !force_lock_off && self.policy.turbo_flash_lock && self.flash_k_codes.is_some();
        if !lock_on {
            self.update_decode_fp16(new_k, new_v, max_seq, device)?;
        }

        let kv_seq = self.offset;

        // ── Head-major persistent K8V4 storage ───────────────────────────────
        //
        // First TurboFlash dispatch on this cache: allocate the persistent
        // 4-D buffer pair `[B, kv_h, max_seq, D/.]` and seed it by quantising
        // the prefill prefix `[B, kv_h, prev_offset, D]` from `decode_fp16_k/v`
        // in one shot. Subsequent dispatches: per-decode-token, quantise the
        // single new token and `slice_update` head-major at the new row.
        //
        // This eliminates the prior O(prefix) re-quant per dispatch, which was
        // the entire reason TurboFlash regressed vs OFF at 16K-128K (P1.A.1
        // bench: -60% to -80% TPS). After this fix, per-decode write traffic
        // is `B*kv_h*D` (a few KB per layer).
        let (b, kv_h, _) = {
            let s = new_k.shape();
            (s[0], s[1], s[3])
        };

        if self.flash_k_codes.is_none() {
            // First dispatch: allocate persistent buffers and seed from prefix.
            self.alloc_flash_buffers(b, kv_h, head_dim, max_seq, device)?;
            if prev_offset > 0 {
                // Seed: quantise the prefill prefix [B, kv_h, prev_offset, D]
                // from decode_fp16_k/v and slice_update head-major into the
                // persistent buffers at [:, :, 0:prev_offset, :].
                self.append_flash_buffers_from_fp16(0, prev_offset, device)?;
            }
            // First-dispatch new chunk still comes from decode_fp16_k/v
            // (the mirror was just updated above so it contains the new token).
            self.append_flash_buffers_from_fp16(prev_offset, new_seq, device)?;
        } else {
            // Grow the latched head-major buffers when the storage window has
            // grown past what they were allocated for — a power-of-two boundary
            // crossed mid-decode. The buffers latch their capacity into
            // `flash_max_seq` at allocation, so without re-sizing them here the
            // append below overflows the frozen window.
            if self.flash_max_seq < max_seq {
                self.grow_flash_buffers(b, kv_h, head_dim, max_seq, device)?;
            }
            if lock_on {
                // Subsequent dispatch under lock-on: quantise `new_k`/`new_v`
                // directly into the persistent flash buffers — no bf16 round-trip.
                self.append_flash_buffers_from_new(new_k, new_v, prev_offset, new_seq, device)?;
            } else {
                // Subsequent dispatch, lock OFF: read the new chunk back through
                // `decode_fp16_k/v` (which was just updated above). Preserves the
                // original behaviour bit-for-bit when lock is not requested.
                self.append_flash_buffers_from_fp16(prev_offset, new_seq, device)?;
            }
        }

        // Pull the persistent buffers as 1-D flat views for the kernel
        // (contiguous reshape — zero copy on the MLX side).
        let k_codes_buf = self.flash_k_codes.as_ref().unwrap();
        let k_scales_buf = self.flash_k_scales.as_ref().unwrap();
        let v_codes_buf = self.flash_v_codes.as_ref().unwrap();
        let v_scales_buf = self.flash_v_scales.as_ref().unwrap();
        let k_codes_total: i32 = k_codes_buf.shape().iter().product();
        let k_scales_total: i32 = k_scales_buf.shape().iter().product();
        let v_codes_total: i32 = v_codes_buf.shape().iter().product();
        let v_scales_total: i32 = v_scales_buf.shape().iter().product();
        let k_codes_flat = k_codes_buf.reshape(&[k_codes_total], device)?;
        let k_scales_flat = k_scales_buf.reshape(&[k_scales_total], device)?;
        let v_codes_flat = v_codes_buf.reshape(&[v_codes_total], device)?;
        let v_scales_flat = v_scales_buf.reshape(&[v_scales_total], device)?;

        let q_shape = queries.shape();
        let b = q_shape[0];
        let n_q_heads = q_shape[1];

        // Scale the queries (turbo_flash_sdpa expects pre-scaled Q).
        let q_scaled = {
            use rmlx_mlx::{multiply, scalar_f32};
            // Canonical guarded form: `astype` to the same dtype is a no-op in
            // MLX, so this is identical to branching on it, and the guard sits
            // on the same statement where `check-no-scalar-f32-leak` can see
            // it. An f32 scalar multiplied into bf16 queries promotes Q, and
            // the kernel's whole output behind it.
            let sc = scalar_f32(scale).astype(queries.dtype(), device)?;
            multiply(queries, &sc, device)?
        };

        let t_stride = self.flash_max_seq;
        let out = turbo_flash_sdpa(
            &q_scaled,
            &k_codes_flat,
            &k_scales_flat,
            &v_codes_flat,
            &v_scales_flat,
            additive_mask,
            b,
            n_q_heads,
            kv_h,
            kv_seq,
            t_stride,
            head_dim,
            device,
        );

        match out {
            Ok(arr) => Ok(Some(arr)),
            Err(e) => {
                tracing::warn!("TurboFlash: kernel error, falling back to standard SDPA: {e}");
                Ok(None)
            }
        }
    }

    /// Allocate the 4-D head-major persistent K8V4 buffers.
    ///
    /// Layout (per documented K/V format):
    /// K codes: u32 `[B, kv_h, max_seq, head_dim/4]` — q8_0, 4 i8/u32
    /// K scales: f32 `[B, kv_h, max_seq, head_dim/Q8_GROUP]` — q8_0 scales
    /// V codes: u32 `[B, kv_h, max_seq, head_dim/8]` — turbo4, 8 nibbles/u32
    /// V scales: f32 `[B, kv_h, max_seq, head_dim/TQ4_GROUP]` — turbo4 scales
    ///
    /// Buffers are zero-init via `zeros()`; slots `[kv_seq..max_seq)` are
    /// never read by the kernel (it iterates `t < t_active`). Total RAM:
    /// for Qwen35B max_seq=128K, B=1, kv_h=8, head_dim=128 ≈ 33 MB per layer
    /// across both K and V — same order as the existing K8V4 storage which
    /// would also size to max_seq when paged growth filled, so the residency
    /// uplift is bounded.
    fn alloc_flash_buffers(
        &mut self,
        b: i32,
        kv_h: i32,
        head_dim: i32,
        max_seq: i32,
        device: Device,
    ) -> Result<()> {
        use crate::q8_msl::Q8_GROUP_SIZE;
        use crate::turboquant::GROUP_SIZE as TQ4_GROUP;

        let k_codes_shape = [b, kv_h, max_seq, head_dim / 4];
        let k_scales_shape = [b, kv_h, max_seq, head_dim / Q8_GROUP_SIZE as i32];
        let v_codes_shape = [b, kv_h, max_seq, head_dim / 8];
        let v_scales_shape = [b, kv_h, max_seq, head_dim / TQ4_GROUP as i32];

        self.flash_k_codes = Some(zeros(&k_codes_shape, Dtype::U32, device)?);
        self.flash_k_scales = Some(zeros(&k_scales_shape, Dtype::F32, device)?);
        self.flash_v_codes = Some(zeros(&v_codes_shape, Dtype::U32, device)?);
        self.flash_v_scales = Some(zeros(&v_scales_shape, Dtype::F32, device)?);
        self.flash_max_seq = max_seq;
        self.flash_filled = 0;
        Ok(())
    }

    /// Grow the head-major persistent K8V4 buffers to a larger `max_seq`,
    /// preserving every already-written slot.
    ///
    /// The flash buffers latch their capacity into `flash_max_seq` at
    /// [`Self::alloc_flash_buffers`] and never revisit it. When decode crosses a
    /// power-of-two boundary the storage window grows (via
    /// [`Self::ensure_decode_capacity`]) but these buffers do not — so the next
    /// head-major append would walk off the frozen window. This reallocates each
    /// buffer at the new capacity and copies the existing `[.., 0..old_max_seq, .]`
    /// content forward, matching the copy-prefix-forward contract every other
    /// grow path uses. It is codec-general (keyed off buffer shape, never an
    /// arch) and lock-state agnostic — under `turbo_flash_lock` the flash
    /// buffers are the sole K/V store, so copying them (rather than re-seeding
    /// from the frozen bf16 mirror) is what preserves the decode tail.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "the four flash buffers are Some by the caller's is_none() guard; the grow path only runs after a first dispatch allocated them"
    )]
    fn grow_flash_buffers(
        &mut self,
        b: i32,
        kv_h: i32,
        head_dim: i32,
        new_max_seq: i32,
        device: Device,
    ) -> Result<()> {
        use crate::q8_msl::Q8_GROUP_SIZE;
        use crate::turboquant::GROUP_SIZE as TQ4_GROUP;

        let old_max_seq = self.flash_max_seq;
        if new_max_seq <= old_max_seq {
            return Ok(());
        }

        let k_codes_shape = [b, kv_h, new_max_seq, head_dim / 4];
        let k_scales_shape = [b, kv_h, new_max_seq, head_dim / Q8_GROUP_SIZE as i32];
        let v_codes_shape = [b, kv_h, new_max_seq, head_dim / 8];
        let v_scales_shape = [b, kv_h, new_max_seq, head_dim / TQ4_GROUP as i32];

        let sl_start = [0i32; 4];
        let sl_strides = [1i32; 4];

        let kc_old = self.flash_k_codes.take().unwrap();
        let ks_old = self.flash_k_scales.take().unwrap();
        let vc_old = self.flash_v_codes.take().unwrap();
        let vs_old = self.flash_v_scales.take().unwrap();

        let kc_stop = [b, kv_h, old_max_seq, head_dim / 4];
        let ks_stop = [b, kv_h, old_max_seq, head_dim / Q8_GROUP_SIZE as i32];
        let vc_stop = [b, kv_h, old_max_seq, head_dim / 8];
        let vs_stop = [b, kv_h, old_max_seq, head_dim / TQ4_GROUP as i32];

        let kc_new = zeros(&k_codes_shape, Dtype::U32, device)?.slice_update(
            &kc_old,
            &sl_start,
            &kc_stop,
            &sl_strides,
            device,
        )?;
        let ks_new = zeros(&k_scales_shape, Dtype::F32, device)?.slice_update(
            &ks_old,
            &sl_start,
            &ks_stop,
            &sl_strides,
            device,
        )?;
        let vc_new = zeros(&v_codes_shape, Dtype::U32, device)?.slice_update(
            &vc_old,
            &sl_start,
            &vc_stop,
            &sl_strides,
            device,
        )?;
        let vs_new = zeros(&v_scales_shape, Dtype::F32, device)?.slice_update(
            &vs_old,
            &sl_start,
            &vs_stop,
            &sl_strides,
            device,
        )?;

        self.flash_k_codes = Some(kc_new);
        self.flash_k_scales = Some(ks_new);
        self.flash_v_codes = Some(vc_new);
        self.flash_v_scales = Some(vs_new);
        self.flash_max_seq = new_max_seq;
        tracing::info!(
            from = old_max_seq,
            to = new_max_seq,
            "TurboFlash buffer grow"
        );
        Ok(())
    }

    /// Quantise `[B, kv_h, n, D]` from `decode_fp16_k/v` starting
    /// at token `start` and `slice_update` it head-major into the persistent
    /// flash buffers at `[:, :, start:start+n, :]`.
    ///
    /// Both seed (prefill prefix, `n = prev_offset`) and per-decode-step
    /// append (`n = 1`) flow through this single helper.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn append_flash_buffers_from_fp16(&mut self, start: i32, n: i32, device: Device) -> Result<()> {
        if n <= 0 {
            return Ok(());
        }
        let fp16_k = self
            .decode_fp16_k
            .as_ref()
            .ok_or_else(|| Error::Mlx("flash append: decode_fp16_k missing".into()))?;
        let fp16_v = self
            .decode_fp16_v
            .as_ref()
            .ok_or_else(|| Error::Mlx("flash append: decode_fp16_v missing".into()))?;
        let k_shape = fp16_k.shape();
        let b = k_shape[0];
        let kv_h = k_shape[1];
        let d = k_shape[3];

        // Slice [B, kv_h, n, D] from decode_fp16 starting at `start`.
        let sl_start = [0i32, 0, start, 0];
        let sl_stop = [b, kv_h, start + n, d];
        let sl_strides = [1i32; 4];
        let k_chunk = fp16_k.slice(&sl_start, &sl_stop, &sl_strides, device)?;
        let v_chunk = fp16_v.slice(&sl_start, &sl_stop, &sl_strides, device)?;
        // q8_quantize_gpu / turbo_quantize_v4_gpu expect f32 input.
        let k_f32 = if k_chunk.dtype() == Dtype::F32 {
            k_chunk
        } else {
            k_chunk.astype(Dtype::F32, device)?
        };
        let v_f32 = if v_chunk.dtype() == Dtype::F32 {
            v_chunk
        } else {
            v_chunk.astype(Dtype::F32, device)?
        };

        let (k_codes, k_scales) = crate::q8_msl::q8_quantize_gpu(&k_f32, device)?;
        let (v_codes, v_scales) = crate::turboquant_msl::turbo_quantize_v4_gpu(&v_f32, device)?;

        // The quantize kernels return flat arrays whose underlying data is
        // `[B, kv_h, n, D/.]` row-major (they preserve element order).
        // Reshape to 4-D so the slice_update target stride matches.
        use crate::q8_msl::Q8_GROUP_SIZE;
        use crate::turboquant::GROUP_SIZE as TQ4_GROUP;
        let k_codes_4d = k_codes.reshape(&[b, kv_h, n, d / 4], device)?;
        let k_scales_4d = k_scales.reshape(&[b, kv_h, n, d / Q8_GROUP_SIZE as i32], device)?;
        let v_codes_4d = v_codes.reshape(&[b, kv_h, n, d / 8], device)?;
        let v_scales_4d = v_scales.reshape(&[b, kv_h, n, d / TQ4_GROUP as i32], device)?;

        // 4-D slice_update at [:, :, start:start+n, :] into each persistent buffer.
        let kc_buf = self.flash_k_codes.take().unwrap();
        let ks_buf = self.flash_k_scales.take().unwrap();
        let vc_buf = self.flash_v_codes.take().unwrap();
        let vs_buf = self.flash_v_scales.take().unwrap();

        let max_seq = self.flash_max_seq;
        let kc_stop = [b, kv_h, start + n, d / 4];
        let ks_stop = [b, kv_h, start + n, d / Q8_GROUP_SIZE as i32];
        let vc_stop = [b, kv_h, start + n, d / 8];
        let vs_stop = [b, kv_h, start + n, d / TQ4_GROUP as i32];
        // The flash buffers latch their window at allocation, so an append past
        // it cannot be served by growing here. It must not be waved through: the
        // `slice_update` below would land out of bounds and silently no-op,
        // dropping the token while `offset` advances. `debug_assert!` is not
        // enough — the perf/release profiles compile it out, which is exactly
        // where this would bite. Fail loudly instead.
        if start + n > max_seq {
            return Err(Error::Quant(format!(
                "flash append: needed={} exceeds the allocated flash window \
                 max_seq={max_seq} — the buffers were sized for a shorter \
                 sequence and cannot be extended in place",
                start + n
            )));
        }

        let kc_new = kc_buf.slice_update(&k_codes_4d, &sl_start, &kc_stop, &sl_strides, device)?;
        let ks_new = ks_buf.slice_update(&k_scales_4d, &sl_start, &ks_stop, &sl_strides, device)?;
        let vc_new = vc_buf.slice_update(&v_codes_4d, &sl_start, &vc_stop, &sl_strides, device)?;
        let vs_new = vs_buf.slice_update(&v_scales_4d, &sl_start, &vs_stop, &sl_strides, device)?;

        self.flash_k_codes = Some(kc_new);
        self.flash_k_scales = Some(ks_new);
        self.flash_v_codes = Some(vc_new);
        self.flash_v_scales = Some(vs_new);
        if start + n > self.flash_filled {
            self.flash_filled = start + n;
        }
        Ok(())
    }

    /// Quantise `new_k`/`new_v` directly into the persistent flash buffers at
    /// `[:, :, start:start+n, :]`, bypassing the bf16 `decode_fp16_k/v` mirror.
    ///
    /// Used by the `DispatchPolicy::turbo_flash_lock` path after the initial seed has
    /// populated `flash_*`. Algorithmically equivalent to
    /// `append_flash_buffers_from_fp16` if the caller updated `decode_fp16_*`
    /// with the same `new_k`/`new_v` first — but we skip that update so the
    /// kernel can drop one slice_update dispatch per layer per decode step.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn append_flash_buffers_from_new(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        start: i32,
        n: i32,
        device: Device,
    ) -> Result<()> {
        if n <= 0 {
            return Ok(());
        }
        let k_shape = new_k.shape();
        let b = k_shape[0];
        let kv_h = k_shape[1];
        debug_assert_eq!(
            k_shape[2], n,
            "append_flash_buffers_from_new: new_k.shape[2] must equal n"
        );
        let d = k_shape[3];

        // q8_quantize_gpu / turbo_quantize_v4_gpu expect f32 input.
        let k_f32_owned;
        let k_f32: &Array = if new_k.dtype() == Dtype::F32 {
            new_k
        } else {
            k_f32_owned = new_k.astype(Dtype::F32, device)?;
            &k_f32_owned
        };
        let v_f32_owned;
        let v_f32: &Array = if new_v.dtype() == Dtype::F32 {
            new_v
        } else {
            v_f32_owned = new_v.astype(Dtype::F32, device)?;
            &v_f32_owned
        };

        let (k_codes, k_scales) = crate::q8_msl::q8_quantize_gpu(k_f32, device)?;
        let (v_codes, v_scales) = crate::turboquant_msl::turbo_quantize_v4_gpu(v_f32, device)?;

        use crate::q8_msl::Q8_GROUP_SIZE;
        use crate::turboquant::GROUP_SIZE as TQ4_GROUP;
        let k_codes_4d = k_codes.reshape(&[b, kv_h, n, d / 4], device)?;
        let k_scales_4d = k_scales.reshape(&[b, kv_h, n, d / Q8_GROUP_SIZE as i32], device)?;
        let v_codes_4d = v_codes.reshape(&[b, kv_h, n, d / 8], device)?;
        let v_scales_4d = v_scales.reshape(&[b, kv_h, n, d / TQ4_GROUP as i32], device)?;

        let sl_start = [0i32, 0, start, 0];
        let sl_strides = [1i32; 4];
        let kc_buf = self.flash_k_codes.take().unwrap();
        let ks_buf = self.flash_k_scales.take().unwrap();
        let vc_buf = self.flash_v_codes.take().unwrap();
        let vs_buf = self.flash_v_scales.take().unwrap();

        let max_seq = self.flash_max_seq;
        let kc_stop = [b, kv_h, start + n, d / 4];
        let ks_stop = [b, kv_h, start + n, d / Q8_GROUP_SIZE as i32];
        let vc_stop = [b, kv_h, start + n, d / 8];
        let vs_stop = [b, kv_h, start + n, d / TQ4_GROUP as i32];
        // The flash buffers latch their window at allocation, so an append past
        // it cannot be served by growing here. It must not be waved through: the
        // `slice_update` below would land out of bounds and silently no-op,
        // dropping the token while `offset` advances. `debug_assert!` is not
        // enough — the perf/release profiles compile it out, which is exactly
        // where this would bite. Fail loudly instead.
        if start + n > max_seq {
            return Err(Error::Quant(format!(
                "flash append: needed={} exceeds the allocated flash window \
                 max_seq={max_seq} — the buffers were sized for a shorter \
                 sequence and cannot be extended in place",
                start + n
            )));
        }

        let kc_new = kc_buf.slice_update(&k_codes_4d, &sl_start, &kc_stop, &sl_strides, device)?;
        let ks_new = ks_buf.slice_update(&k_scales_4d, &sl_start, &ks_stop, &sl_strides, device)?;
        let vc_new = vc_buf.slice_update(&v_codes_4d, &sl_start, &vc_stop, &sl_strides, device)?;
        let vs_new = vs_buf.slice_update(&v_scales_4d, &sl_start, &vs_stop, &sl_strides, device)?;

        self.flash_k_codes = Some(kc_new);
        self.flash_k_scales = Some(ks_new);
        self.flash_v_codes = Some(vc_new);
        self.flash_v_scales = Some(vs_new);
        if start + n > self.flash_filled {
            self.flash_filled = start + n;
        }
        Ok(())
    }

    /// Returns true when this cache holds K8V4 quantization (q8_0 K + turbo4 V).
    pub fn is_k8v4(&self) -> bool {
        matches!(self.quant, KvQuant::K8V4)
    }

    /// Single dispatch entry point for all K8V4 attention layers.
    ///
    /// Encapsulates the TurboFlash opt-in check so each arch attention layer
    /// calls one function instead of repeating the env-var / seq-len logic.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(output))` — TurboFlash ran; output is `[B, n_q_heads, 1, D]`.
    /// - `Ok(None)` — not K8V4, the gate is off, seq too short, or a prefill
    ///   step; caller falls through to standard `cache.update()` + SDPA.
    ///
    /// # Dispatch rule
    ///
    /// ```text
    /// if policy.turbo_flash AND is_k8v4()
    ///    AND kv_seq_after_update > policy.turbo_flash_min_kv_seq {
    /// update_and_sdpa_k8v4_flash(...) // split-K FA, no dequant round-trip
    /// } else {
    /// None // caller does standard update() + scaled_dot_product_attention()
    /// }
    /// ```
    ///
    /// The threshold is checked inside `update_and_sdpa_k8v4_flash` via
    /// `turbo_flash_should_run`; this wrapper delegates entirely to that
    /// function, keeping the check in one place.
    ///
    /// Callers that are NOT K8V4 pay only the `is_k8v4()` bool check and
    /// immediately get `Ok(None)` — zero overhead for other quant modes.
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa_dispatch(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Option<Array>> {
        if !self.is_k8v4() {
            return Ok(None);
        }
        self.update_and_sdpa_k8v4_flash(queries, new_k, new_v, scale, additive_mask, device)
    }

    /// Sibling of [`Self::sdpa_dispatch`] used by
    /// `update_and_sdpa_shared_source` (cross-layer-KV producers). Forces the
    /// TurboFlash lock-on optimisation OFF so the bf16 mirror stays current.
    pub(super) fn sdpa_dispatch_no_lock(
        &mut self,
        queries: &Array,
        new_k: &Array,
        new_v: &Array,
        scale: f32,
        additive_mask: Option<&Array>,
        device: Device,
    ) -> Result<Option<Array>> {
        if !self.is_k8v4() {
            return Ok(None);
        }
        self.update_and_sdpa_k8v4_flash_no_lock(queries, new_k, new_v, scale, additive_mask, device)
    }

    #[allow(
        clippy::unreachable,
        reason = "storage variant is guaranteed by the `match &self.storage` dispatch in \
                  KvCache::update() (KvStorage::K8V8 arm); \
                  mismatch is a construction-time BUG"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn update_k8v8(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::K8V8 { k, v, max_seq } = &mut self.storage else {
            unreachable!("storage mismatch: expected K8V8");
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();
        let (k_f32, v_f32) = if device == Device::Gpu {
            (Vec::new(), Vec::new())
        } else {
            arrays_to_f32(new_k, new_v, device)?
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
            *v = Some(QuantK {
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

    #[allow(
        clippy::unreachable,
        reason = "storage variant is guaranteed by the `match &self.storage` dispatch in \
                  KvCache::update() (KvStorage::Planar arm); \
                  mismatch is a construction-time BUG"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn update_planar(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::Planar {
            k,
            v,
            max_seq,
            bits,
        } = &mut self.storage
        else {
            unreachable!("storage mismatch: expected Planar");
        };
        let max_seq = *max_seq;
        let v_bits = *bits;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();
        let (k_f32, v_f32) = if device == Device::Gpu {
            (Vec::new(), Vec::new())
        } else {
            arrays_to_f32(new_k, new_v, device)?
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
            *v = Some(QuantPlanarV {
                blocks: Vec::new(),
                gpu_codes_buf: None,
                gpu_scales_buf: None,
                gpu_rotations_buf: None,
                gpu_codes_words_per_step: 0,
                gpu_scales_per_step: 0,
                gpu_rotations_words_per_step: 0,
                gpu_capacity: 0,
                shape: init_shape,
                max_seq,
                bits: v_bits,
            });
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

    /// PlanarK decode update. K is quantized through `QuantPlanarK`
    /// (shared MSL kernel with `Planar` V-side — PlanarQuant is axis-agnostic).
    /// V stays bf16 in `decode_fp16_v` (same machinery as `KvStorage::None`).
    ///
    /// Warm-TTFT shortcut: when `decode_fp16_k` is present (set by
    /// `exit_prefill`), route through `update_decode_fp16` so the bf16 K seed
    /// is used for the rest of the request. This matches every other
    /// `update_<arch>` (K8V4/K8V8/Planar/Mixed/K8VTurbo*/Iso*/Rotor*/
    /// TurboSym*). Before this fix, PlanarK was the **sole** codec that
    /// re-encoded K through the lossy 4-bit Lloyd-Max + Givens kernel on every
    /// decode step while every other variant silently stayed in bf16 K, and
    /// that asymmetry surfaced as the Bonsai PlanarK NIAH retrieval failure.
    ///
    /// See `docs/reports/planar-chunked-prefill-fix.md` § "Followups"
    /// for the open question of whether warm-TTFT-as-default is the intended
    /// steady-state design for the entire quantised-KV surface.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn update_planar_k(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::PlanarK { k, max_seq } = &mut self.storage else {
            return Err(Error::KvStorageMismatch {
                expected: "PlanarK",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;

        // Warm-TTFT bf16 K seed path. Same shortcut every other
        // `update_<arch>` honours; before this fix PlanarK lacked it and was
        // the only codec exercising the per-decode-step 4-bit K encode.
        //
        // See `docs/reports/planar-chunked-prefill-fix.md` § "Followups"
        // for the open question of whether warm-TTFT-as-default is the intended
        // steady-state design for the entire quantised-KV surface.
        if self.decode_fp16_k.is_some() {
            tracing::debug!(
                target: "rmlx_kv_quant::warm_ttft",
                path = "warm_ttft_bypass",
                codec = "PlanarK",
                offset = self.offset,
                "PlanarK update routing through warm-TTFT bf16 K seed; \
                 4-bit codec stays quiescent for this decode step"
            );
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, device)?
        };

        // ── K side: QuantPlanarK encode + dequant for SDPA input ─────────────
        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantPlanarK::new(init_shape, max_seq));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("PlanarK K buffer absent after init".into()));
        };
        ks.append(&k_f32, &new_shape, new_k, device, max_seq)?;
        let k_shape = ks.shape.clone();
        let (k_recon_f32, k_arr_opt) = ks.dequantize_choice(device, new_k.dtype())?;
        let k_full = match k_arr_opt {
            Some(arr) => arr,
            None => f32_vec_to_array(&k_recon_f32, &k_shape)?,
        };

        // ── V side: bf16 via update_decode_fp16 (same as KvStorage::None V).
        // We pass the original `new_k` shadow for the unused K side; the V
        // returned is the one we keep. K from quant codec is what SDPA sees.
        let (_k_shadow, v_full) = self.update_decode_fp16(new_k, new_v, max_seq, device)?;
        Ok((k_full, v_full))
    }

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
    fn update_paged(
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
            KvStorage::Paged { quant, max_seq, .. } => (*quant, *max_seq),
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

    /// Materialize all GPU `Array` buffers this cache holds to host-readable
    /// memory, on the **calling** thread.
    ///
    /// Called on the inference thread by the spill sink right after the
    /// refcount-clone, so that the background spill drain thread (which has no
    /// access to the Metal stream that built these arrays) can serialize the
    /// already-evaluated bytes without re-evaluating the lazy graph. Without
    /// this, the drain thread's serialize fails with
    /// `There is no Stream(gpu, N) in current thread`.
    pub fn eval_for_spill(&self) -> Result<()> {
        // Delegate to the complete GPU-state materializer (handles every storage
        // variant + rotating ring + decode_fp16/prefill_raw scratch). Called on
        // the inference thread by the spill sinks so the drain thread —
        // which has no Metal stream — only copies already-evaluated host bytes.
        self.eval_gpu_state()
    }

    /// Deep-clone this cache: creates new MLX arrays for every stored tensor.
    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(Self {
            storage: self.storage.try_deep_clone()?,
            offset: self.offset,
            quant: self.quant,
            layer_idx: self.layer_idx,
            prefill_raw_k: match &self.prefill_raw_k {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            prefill_raw_v: match &self.prefill_raw_v {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            in_prefill: self.in_prefill,
            decode_fp16_k: match &self.decode_fp16_k {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            decode_fp16_v: match &self.decode_fp16_v {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            // The clone serves the same model, so it inherits the stream dtype
            // rather than re-learning it on its first append.
            stream_dtype: self.stream_dtype,
            rotating: match &self.rotating {
                Some(r) => Some(r.try_deep_clone()?),
                None => None,
            },
            flash_k_codes: match &self.flash_k_codes {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            flash_k_scales: match &self.flash_k_scales {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            flash_v_codes: match &self.flash_v_codes {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            flash_v_scales: match &self.flash_v_scales {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            flash_max_seq: self.flash_max_seq,
            flash_filled: self.flash_filled,
            // The fused-QK shadow holds purely transient decode-time state
            // (re-seeded on every fresh decode). A deep clone for request
            // branching simply drops it; the next decode dispatch on the
            // cloned cache will reallocate from the bf16 prefix.
            fused_qk_shadow: None,
            // Preserve the virtual ceiling across a branch clone so the cloned
            // cache enforces the same --max-ctx bound on further prefill.
            max_seq_ceiling: self.max_seq_ceiling,
            // A branch clone continues the same request and must dispatch
            // through the same kernel paths, so it inherits the policy rather
            // than re-reading the process default.
            policy: self.policy,
        })
    }

    /// Iso3 decode update — K = affine q8_0, V = IsoQuant 3-bit
    /// (quaternion SO(4) rotation + Lloyd-Max codebook).
    ///
    /// Mirrors [`Self::update_k8vturbo3`] with the V side replaced by
    /// [`QuantIsoV3`].
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Option is Some by construction immediately above this fn body's assignments"
    )]
    fn update_iso3(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::IsoV3 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected IsoV3, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as K8V4 / K8VTurbo3).
        // V-side: routes encode through the iso3 MSL kernel when
        // `device == Device::Gpu`; CPU encode remains the fallback. This
        // hot path is shadowed by the warm-TTFT bf16 seed: the GPU encode
        // fires once at exit_prefill (large `new_v` slice), not per decode
        // step (the bf16 V buffer absorbs decode-step appends).
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
            *v = Some(QuantIsoV3::new(init_shape));
        }
        let vs = v.as_mut().unwrap();
        // Per-phase trace instrumentation for the iso3 V hot path.
        // Each phase emits one structured `trace!` event. Off by default; opt
        // in with `--log verbose` or `RUST_LOG=rmlx_kv_quant=trace`. Per
        // CLAUDE.md §Debug mode: never enable trace at info level.
        let kv_h = new_shape[1];
        let head_dim = new_shape[3];
        let t_enc = std::time::Instant::now();
        if device == Device::Gpu {
            // GPU-resident mirror update. `QuantIsoV3::append_gpu` retains
            // the encode outputs in a pre-allocated per-struct buffer so
            // `dequant_gpu` below can skip the CPU-staged `Array::from_bytes`
            // upload on every step.
            vs.append_gpu(new_v, &new_shape, max_seq, device)?;
        } else {
            vs.append(&v_f32, &new_shape)?;
        }
        let s_total = vs.shape[2];
        tracing::trace!(
            phase = "iso3_encode",
            ms = t_enc.elapsed().as_secs_f64() * 1e3,
            s_total = s_total,
            kv_h,
            head_dim,
            "iso3 hot-path"
        );
        // On GPU, skip the CPU dequant + vec_to_array round-trip and upload
        // codes/scales/quaternions/norms directly via Array::from_bytes
        // → dispatch iso_dequantize_v3_gpu. Single-pass GPU side, no
        // intermediate Vec<f32> materialisation.
        let v_shape = vs.shape.clone();
        let v_full = if device == Device::Gpu {
            let t_deq = std::time::Instant::now();
            let arr = vs.dequant_gpu(device)?;
            tracing::trace!(
                phase = "iso3_dequant_gpu",
                codec = "iso3_gpu",
                ms = t_deq.elapsed().as_secs_f64() * 1e3,
                s_total = s_total,
                kv_h,
                head_dim,
                "iso3 hot-path"
            );
            arr
        } else {
            let t_deq = std::time::Instant::now();
            let v_recon_f32 = vs.dequant_on(device)?;
            tracing::trace!(
                phase = "iso3_dequant_cpu",
                ms = t_deq.elapsed().as_secs_f64() * 1e3,
                s_total = s_total,
                kv_h,
                head_dim,
                "iso3 hot-path"
            );
            let t_mat = std::time::Instant::now();
            let arr = f32_vec_to_array(&v_recon_f32, &v_shape)?;
            tracing::trace!(
                phase = "iso3_vec_to_array",
                ms = t_mat.elapsed().as_secs_f64() * 1e3,
                s_total = s_total,
                kv_h,
                head_dim,
                "iso3 hot-path"
            );
            arr
        };

        Ok((k_full, v_full))
    }

    /// Iso4 decode update — K = affine q8_0, V = IsoQuant 4-bit
    /// (quaternion SO(4) rotation + 4-bit Lloyd-Max codebook).
    ///
    /// Structurally identical to [`Self::update_iso3`] with the V side bound
    /// to `QuantIsoV4` instead of `QuantIsoV3`. The iso4 MSL kernel wires the
    /// V-encode step when `device == Device::Gpu`; CPU dequant remains primary
    /// for the returned Array (warm-TTFT bf16 shortcut is active at decode).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Option is Some by construction immediately above this fn body's assignments"
    )]
    fn update_iso4(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::IsoV4 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected IsoV4, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as K8V4 / K8VTurbo3).
        // V-side: routes encode through the iso4 MSL kernel when
        // `device == Device::Gpu`; CPU encode remains the fallback. This
        // hot path is shadowed by the warm-TTFT bf16 seed: the GPU encode
        // fires once at exit_prefill (large `new_v` slice), not per decode
        // step (the bf16 V buffer absorbs decode-step appends).
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
            *v = Some(QuantIsoV4::new(init_shape, max_seq));
        }
        let vs = v.as_mut().unwrap();
        if device == Device::Gpu {
            iso4_gpu_append_into_v_blocks(
                vs,
                new_v,
                &new_shape,
                device,
                LEGACY_ISO4_V_FEED,
                max_seq,
            )?;
        } else {
            vs.append(&v_f32, &new_shape)?;
        }
        let v_shape = vs.shape.clone();
        let v_recon_f32 = vs.dequant()?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// Iso3Sym decode update: K and V both quantize through
    /// IsoQuant 3-bit (CPU-only). Structurally mirrors [`Self::update_iso3`]
    /// with the K side bound to `QuantIsoK3` instead of `QuantK`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction"
    )]
    fn update_iso3_sym(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::IsoSym3 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected IsoSym3, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();
        // K and V encode through the iso3 MSL kernel when
        // `device == Device::Gpu`. The kernel is axis-agnostic — K-side and
        // V-side share the same dispatch. This hot path is warm-TTFT-shadowed:
        // the GPU encode fires once at exit_prefill, not per decode step.
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
            *k = Some(QuantIsoK3::new(init_shape, max_seq));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("IsoSym3 K buffer absent after init".into()));
        };
        if device == Device::Gpu {
            iso3_gpu_append_into_k_blocks(ks, new_k, &new_shape, device, RingFeed::Skip, max_seq)?;
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

        if v.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *v = Some(QuantIsoV3::new(init_shape));
        }
        let Some(vs) = v.as_mut() else {
            return Err(Error::Mlx("IsoSym3 V buffer absent after init".into()));
        };
        // Per-phase trace instrumentation for iso3 V (sym path).
        let kv_h = new_shape[1];
        let head_dim = new_shape[3];
        let t_enc = std::time::Instant::now();
        if device == Device::Gpu {
            // Same GPU-resident mirror pattern as `update_iso3`.
            vs.append_gpu(new_v, &new_shape, max_seq, device)?;
        } else {
            vs.append(&v_f32, &new_shape)?;
        }
        let s_total = vs.shape[2];
        tracing::trace!(
            phase = "iso3_encode",
            ms = t_enc.elapsed().as_secs_f64() * 1e3,
            s_total = s_total,
            kv_h,
            head_dim,
            "iso3 hot-path (sym V)"
        );
        let v_shape = vs.shape.clone();
        let v_full = if device == Device::Gpu {
            let t_deq = std::time::Instant::now();
            let arr = vs.dequant_gpu(device)?;
            tracing::trace!(
                phase = "iso3_dequant_gpu",
                codec = "iso3_gpu",
                ms = t_deq.elapsed().as_secs_f64() * 1e3,
                s_total = s_total,
                kv_h,
                head_dim,
                "iso3 hot-path (sym V)"
            );
            arr
        } else {
            let t_deq = std::time::Instant::now();
            let v_recon_f32 = vs.dequant_on(device)?;
            tracing::trace!(
                phase = "iso3_dequant_cpu",
                ms = t_deq.elapsed().as_secs_f64() * 1e3,
                s_total = s_total,
                kv_h,
                head_dim,
                "iso3 hot-path (sym V)"
            );
            let t_mat = std::time::Instant::now();
            let arr = f32_vec_to_array(&v_recon_f32, &v_shape)?;
            tracing::trace!(
                phase = "iso3_vec_to_array",
                ms = t_mat.elapsed().as_secs_f64() * 1e3,
                s_total = s_total,
                kv_h,
                head_dim,
                "iso3 hot-path (sym V)"
            );
            arr
        };

        Ok((k_full, v_full))
    }

    /// Iso4Sym decode update: K and V both quantize through
    /// IsoQuant 4-bit (CPU-only). Mirrors [`Self::update_iso3_sym`] with bits=4.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction"
    )]
    fn update_iso4_sym(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::IsoSym4 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected IsoSym4, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();
        // Both K and V encode through the iso4 MSL kernel when
        // `device == Device::Gpu`. The kernel is axis-agnostic — K-side
        // and V-side share the same dispatch. This path is warm-TTFT-shadowed:
        // GPU encode fires once at exit_prefill, not per step.
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
            *k = Some(QuantIsoK4::new(init_shape, max_seq));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("IsoSym4 K buffer absent after init".into()));
        };
        if device == Device::Gpu {
            iso4_gpu_append_into_k_blocks(ks, new_k, &new_shape, device, RingFeed::Skip, max_seq)?;
        } else {
            ks.append(&k_f32, &new_shape)?;
        }
        let k_shape = ks.shape.clone();
        let k_recon_f32 = ks.dequant()?;
        let k_full = f32_vec_to_array(&k_recon_f32, &k_shape)?;

        if v.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *v = Some(QuantIsoV4::new(init_shape, max_seq));
        }
        let Some(vs) = v.as_mut() else {
            return Err(Error::Mlx("IsoSym4 V buffer absent after init".into()));
        };
        if device == Device::Gpu {
            iso4_gpu_append_into_v_blocks(
                vs,
                new_v,
                &new_shape,
                device,
                LEGACY_ISO4_V_FEED,
                max_seq,
            )?;
        } else {
            vs.append(&v_f32, &new_shape)?;
        }
        let v_shape = vs.shape.clone();
        let v_recon_f32 = vs.dequant()?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// IsoKOnly3 decode update. K is IsoQuant 3-bit (CPU); V stays
    /// bf16 on `decode_fp16_v` (same machinery as `KvStorage::None` /
    /// `KvStorage::PlanarK`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction"
    )]
    fn update_iso_k_only_3(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::IsoKOnly3 { k, max_seq } = &mut self.storage else {
            return Err(Error::KvStorageMismatch {
                expected: "IsoKOnly3",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;

        let new_shape = new_k.shape();
        // K encode routes through the iso3 MSL kernel when
        // `device == Device::Gpu`. Warm-TTFT-shadowed: GPU encode fires
        // once at exit_prefill, not per decode step.
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, Device::Cpu)?
        };

        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantIsoK3::new(init_shape, max_seq));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("IsoKOnly3 K buffer absent after init".into()));
        };
        // Per-phase trace instrumentation for iso3 K (K-only path).
        // V side is bf16, not iso3, so no phase events on V.
        let kv_h = new_shape[1];
        let head_dim = new_shape[3];
        let t_enc = std::time::Instant::now();
        if device == Device::Gpu {
            iso3_gpu_append_into_k_blocks(ks, new_k, &new_shape, device, RingFeed::Skip, max_seq)?;
        } else {
            ks.append(&k_f32, &new_shape)?;
        }
        let s_total = ks.shape[2];
        tracing::trace!(
            phase = "iso3_encode",
            ms = t_enc.elapsed().as_secs_f64() * 1e3,
            s_total = s_total,
            kv_h,
            head_dim,
            "iso3 hot-path (K-only)"
        );
        let k_shape = ks.shape.clone();
        let k_full = if device == Device::Gpu {
            let t_deq = std::time::Instant::now();
            let arr = ks.dequant_gpu(device)?;
            tracing::trace!(
                phase = "iso3_dequant_gpu",
                codec = "iso3_gpu",
                ms = t_deq.elapsed().as_secs_f64() * 1e3,
                s_total = s_total,
                kv_h,
                head_dim,
                "iso3 hot-path (K-only)"
            );
            arr
        } else {
            let t_deq = std::time::Instant::now();
            let k_recon_f32 = ks.dequant()?;
            tracing::trace!(
                phase = "iso3_dequant_cpu",
                ms = t_deq.elapsed().as_secs_f64() * 1e3,
                s_total = s_total,
                kv_h,
                head_dim,
                "iso3 hot-path (K-only)"
            );
            let t_mat = std::time::Instant::now();
            let arr = f32_vec_to_array(&k_recon_f32, &k_shape)?;
            tracing::trace!(
                phase = "iso3_vec_to_array",
                ms = t_mat.elapsed().as_secs_f64() * 1e3,
                s_total = s_total,
                kv_h,
                head_dim,
                "iso3 hot-path (K-only)"
            );
            arr
        };

        // V-side: bf16 via the V-only helper (must NOT touch decode_fp16_k).
        // Calling the full update_decode_fp16 here would populate
        // self.decode_fp16_k as a side-effect, causing the
        // decode_fp16_k.is_some() early-return guard to short-circuit the
        // ISO codec on the next decode step (silent bf16-K regression).
        let v_full = self.update_decode_fp16_v_only(new_v, max_seq, device)?;
        Ok((k_full, v_full))
    }

    /// IsoKOnly4 decode update. K is IsoQuant 4-bit (CPU); V bf16.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction"
    )]
    fn update_iso_k_only_4(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::IsoKOnly4 { k, max_seq } = &mut self.storage else {
            return Err(Error::KvStorageMismatch {
                expected: "IsoKOnly4",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;

        let new_shape = new_k.shape();
        // K encode routes through the iso4 MSL kernel when
        // `device == Device::Gpu`. Warm-TTFT-shadowed: GPU encode fires
        // once at exit_prefill, not per decode step.
        let k_f32 = if device == Device::Gpu {
            Vec::new()
        } else {
            array_to_f32_vec(new_k, Device::Cpu)?
        };

        if k.is_none() {
            let mut init_shape = new_shape.clone();
            init_shape[2] = 0;
            *k = Some(QuantIsoK4::new(init_shape, max_seq));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("IsoKOnly4 K buffer absent after init".into()));
        };
        if device == Device::Gpu {
            iso4_gpu_append_into_k_blocks(ks, new_k, &new_shape, device, RingFeed::Skip, max_seq)?;
        } else {
            ks.append(&k_f32, &new_shape)?;
        }
        let k_shape = ks.shape.clone();
        let k_recon_f32 = ks.dequant()?;
        let k_full = f32_vec_to_array(&k_recon_f32, &k_shape)?;

        // V-side: bf16 via the V-only helper (must NOT touch decode_fp16_k).
        let v_full = self.update_decode_fp16_v_only(new_v, max_seq, device)?;
        Ok((k_full, v_full))
    }

    /// Rotor3 decode update — K = affine q8_0, V = rotor3
    /// (Cl(3,0) Clifford rotor sandwich + 3-bit Lloyd-Max codebook).
    ///
    /// Structurally mirrors [`Self::update_iso4`] with the V side bound to
    /// `QuantRotorV3` instead of `QuantIsoV4`. CPU dequant only — no MSL
    /// kernel for rotor3 (deferred, see [`crate::rotorquant`] module docs).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Option is Some by construction immediately above this fn body's assignments"
    )]
    fn update_rotor3(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::RotorV3 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected RotorV3, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as iso3 / iso4 / K8V4).
        // V-side: routes encode through the rotor3 MSL kernel when
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
            // Thread the real model-layer index into the rotor3 seed
            // so each layer gets a distinct rotor table. The rotor table is
            // deterministic and persists via SSD round-trip.
            *v = Some(QuantRotorV3::new(
                init_shape,
                max_seq,
                layer_idx_u32(self.layer_idx),
            ));
        }
        let vs = v.as_mut().unwrap();
        if device == Device::Gpu {
            rotor3_gpu_append_into_blocks(vs, new_v, &new_shape, device, RingFeed::Skip, max_seq)?;
        } else {
            vs.append(&v_f32, &new_shape)?;
        }
        let v_shape = vs.shape.clone();
        let v_recon_f32 = vs.dequant()?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// Rotor4 decode update — K = affine q8_0, V = rotor4
    /// (Cl(3,0) Clifford rotor sandwich + 4-bit Lloyd-Max codebook).
    ///
    /// Structurally mirrors [`Self::update_rotor3`] with the V side bound to
    /// `QuantRotorV4` instead of `QuantRotorV3`. CPU dequant only — no MSL
    /// kernel for rotor4 (deferred, same rationale as rotor3).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Option is Some by construction immediately above this fn body's assignments"
    )]
    fn update_rotor4(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::RotorV4 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected RotorV4, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();

        // K-side: GPU-capable affine q8_0 (same as rotor3 / iso3 / iso4 / K8V4).
        // V-side: routes encode through the rotor4 MSL kernel when
        // `device == Device::Gpu`; CPU encode remains the fallback. This
        // hot path is shadowed by the warm-TTFT bf16 seed: the GPU encode
        // fires once at exit_prefill, not per decode step.
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
            // Thread the real model-layer index into the rotor4 seed
            // so each layer gets a distinct rotor table. The rotor table is
            // deterministic and persists via SSD round-trip.
            *v = Some(QuantRotorV4::new(
                init_shape,
                max_seq,
                layer_idx_u32(self.layer_idx),
            ));
        }
        let vs = v.as_mut().unwrap();
        if device == Device::Gpu {
            rotor4_gpu_append_into_blocks(vs, new_v, &new_shape, device, RingFeed::Skip, max_seq)?;
        } else {
            vs.append(&v_f32, &new_shape)?;
        }
        let v_shape = vs.shape.clone();
        let v_recon_f32 = vs.dequant()?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// Rotor3Sym decode update: K and V both quantize through the rotor3
    /// (Cl(3,0) Clifford rotor) codec (CPU-only). K-side carries the optional
    /// 1-bit QJL residual sideband when
    /// [`crate::rotor_qjl::rotor_qjl_enabled`] is `true` at first append.
    ///
    /// Structurally mirrors [`Self::update_iso3_sym`] with the codec
    /// bound to `QuantRotorK3` / `QuantRotorV3` instead of the iso K/V types.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction"
    )]
    fn update_rotor3_sym(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::RotorSym3 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected RotorSym3, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();
        // GPU encode for both K and V when device == GPU and the store carries
        // no QJL residual. The QJL decision is sticky to the store (fixed at
        // first append) — a later process-env toggle must not reinterpret bytes
        // already written; read the store's own flag, same as the sdpa fast path
        // and `update_rotor_k_only_*`. Env is the fallback only before the store
        // exists. With QJL on the K-side falls back to CPU (the GPU kernel cannot
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
            *k = Some(crate::storage::QuantRotorK3::new(
                init_shape,
                layer_idx_u32(self.layer_idx),
            ));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("RotorSym3 K buffer absent after init".into()));
        };
        if gpu_k_ok {
            rotor3_gpu_append_into_k_blocks(
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
            *v = Some(QuantRotorV3::new(
                init_shape,
                max_seq,
                layer_idx_u32(self.layer_idx),
            ));
        }
        let Some(vs) = v.as_mut() else {
            return Err(Error::Mlx("RotorSym3 V buffer absent after init".into()));
        };
        if use_gpu {
            rotor3_gpu_append_into_blocks(vs, new_v, &new_shape, device, RingFeed::Skip, max_seq)?;
        } else {
            vs.append(&v_f32, &new_shape)?;
        }
        let v_shape = vs.shape.clone();
        let v_recon_f32 = vs.dequant()?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// Rotor4Sym decode update. Mirror of
    /// [`Self::update_rotor3_sym`] with `bits=4`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction"
    )]
    fn update_rotor4_sym(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::RotorSym4 { k, v, max_seq } = &mut self.storage else {
            return Err(Error::Mlx(format!(
                "storage mismatch: expected RotorSym4, got {}",
                storage_variant_name(&self.storage)
            )));
        };
        let max_seq = *max_seq;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

        let new_shape = new_k.shape();
        // GPU encode when device == GPU. K-side opts out when the store carries
        // the QJL residual — read the store's sticky flag, not the live env (see
        // rotor3_sym mirror).
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
            *k = Some(crate::storage::QuantRotorK4::new(
                init_shape,
                layer_idx_u32(self.layer_idx),
            ));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("RotorSym4 K buffer absent after init".into()));
        };
        if gpu_k_ok {
            rotor4_gpu_append_into_k_blocks(
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
            *v = Some(QuantRotorV4::new(
                init_shape,
                max_seq,
                layer_idx_u32(self.layer_idx),
            ));
        }
        let Some(vs) = v.as_mut() else {
            return Err(Error::Mlx("RotorSym4 V buffer absent after init".into()));
        };
        if use_gpu {
            rotor4_gpu_append_into_blocks(vs, new_v, &new_shape, device, RingFeed::Skip, max_seq)?;
        } else {
            vs.append(&v_f32, &new_shape)?;
        }
        let v_shape = vs.shape.clone();
        let v_recon_f32 = vs.dequant()?;
        let v_full = f32_vec_to_array(&v_recon_f32, &v_shape)?;

        Ok((k_full, v_full))
    }

    /// RotorKOnly3 decode update. K is rotor3 (CPU); V stays
    /// bf16 on `decode_fp16_v`.
    ///
    /// **CRITICAL** (HIGH bug guard): uses
    /// [`Self::update_decode_fp16_v_only`] for the V side, NOT
    /// `update_decode_fp16`. The latter populates `self.decode_fp16_k` as a
    /// side-effect, which causes the `decode_fp16_k.is_some()` early-return
    /// guard to short-circuit the K codec on the *next* decode step (silent
    /// bf16-K regression).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction"
    )]
    fn update_rotor_k_only_3(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::RotorKOnly3 { k, max_seq } = &mut self.storage else {
            return Err(Error::KvStorageMismatch {
                expected: "RotorKOnly3",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;

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
            *k = Some(crate::storage::QuantRotorK3::new(
                init_shape,
                layer_idx_u32(self.layer_idx),
            ));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("RotorKOnly3 K buffer absent after init".into()));
        };
        if gpu_k_ok {
            rotor3_gpu_append_into_k_blocks(
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
        let k_full = f32_vec_to_array(&k_recon_f32, &k_shape)?;

        // V-side: bf16 via the V-only helper (must NOT touch decode_fp16_k).
        let v_full = self.update_decode_fp16_v_only(new_v, max_seq, device)?;
        Ok((k_full, v_full))
    }

    /// RotorKOnly4 decode update. Mirror of
    /// [`Self::update_rotor_k_only_3`] with `bits=4`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction"
    )]
    fn update_rotor_k_only_4(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::RotorKOnly4 { k, max_seq } = &mut self.storage else {
            return Err(Error::KvStorageMismatch {
                expected: "RotorKOnly4",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;

        let new_shape = new_k.shape();
        // See `update_rotor_k_only_3`: gate the GPU encode on the store's sticky
        // QJL flag (fixed at first append), not the current process env, so a
        // later toggle cannot reinterpret bytes already written. Env is the
        // fallback only before the store exists.
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
            *k = Some(crate::storage::QuantRotorK4::new(
                init_shape,
                layer_idx_u32(self.layer_idx),
            ));
        }
        let Some(ks) = k.as_mut() else {
            return Err(Error::Mlx("RotorKOnly4 K buffer absent after init".into()));
        };
        if gpu_k_ok {
            rotor4_gpu_append_into_k_blocks(
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
        let k_full = f32_vec_to_array(&k_recon_f32, &k_shape)?;

        // V-side: bf16 via the V-only helper (must NOT touch decode_fp16_k).
        let v_full = self.update_decode_fp16_v_only(new_v, max_seq, device)?;
        Ok((k_full, v_full))
    }

    /// RotorK3Asym decode update. K is rotor3 (CPU); V is MLX
    /// affine `v_bits` / `v_group_size` (reuses [`QuantV`]).
    ///
    /// Mirrors [`Self::update_rotor_k_only_3`] for K and [`Self::update_k8v4`]
    /// for V (affine path) on the **seedless** path only.
    ///
    /// **Warm-TTFT.** Unlike `RotorKOnly3`, this asym variant DOES carry the
    /// `decode_fp16_k.is_some()` shortcut (below): once the bf16 seed is live
    /// (always, post-`exit_prefill` — see `exit_prefill`'s
    /// generic seed tail), the entire decode step routes through
    /// [`Self::update_decode_fp16`] and serves **both** K and V from bf16. The
    /// rotor-K and affine-V codecs are quiescent for the whole decode window;
    /// they re-encode only at `exit_prefill` or on a seedless cache. This is
    /// the universal warm-TTFT decode contract documented in
    /// `docs/KV_CACHE.md` §9.6.
    ///
    /// NB: `RotorKOnly3` (no asym V) is the opposite — it has **no** seed
    /// shortcut in its body, so its rotor-K codec runs every decode step
    /// (K-only family). Do not assume the two share K-side decode semantics.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Option is Some by construction immediately above this fn body's assignments"
    )]
    fn update_rotor_k_asym_3(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::RotorKAsym3 {
            k,
            v,
            max_seq,
            v_bits,
            v_group_size,
        } = &mut self.storage
        else {
            return Err(Error::KvStorageMismatch {
                expected: "RotorKAsym3",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;
        let v_bits = *v_bits;
        let _ = v_group_size;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

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
            *k = Some(crate::storage::QuantRotorK3::new(
                init_shape,
                layer_idx_u32(self.layer_idx),
            ));
        }
        let ks = k.as_mut().unwrap();
        if gpu_k_ok {
            rotor3_gpu_append_into_k_blocks(
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

    /// RotorK4Asym decode update. Mirror of
    /// [`Self::update_rotor_k_asym_3`] with rotor4 K.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Option is Some by construction immediately above this fn body's assignments"
    )]
    fn update_rotor_k_asym_4(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let KvStorage::RotorKAsym4 {
            k,
            v,
            max_seq,
            v_bits,
            v_group_size,
        } = &mut self.storage
        else {
            return Err(Error::KvStorageMismatch {
                expected: "RotorKAsym4",
                got: storage_variant_name(&self.storage),
            });
        };
        let max_seq = *max_seq;
        let v_bits = *v_bits;
        let _ = v_group_size;

        if self.decode_fp16_k.is_some() {
            return self.update_decode_fp16(new_k, new_v, max_seq, device);
        }

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
            *k = Some(crate::storage::QuantRotorK4::new(
                init_shape,
                layer_idx_u32(self.layer_idx),
            ));
        }
        let ks = k.as_mut().unwrap();
        if gpu_k_ok {
            rotor4_gpu_append_into_k_blocks(
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
}

// ── KV hard-cap helpers ──────────────────────────────────────────────────────

/// Cached value of the `RMLX_KV_MAX_SEQ_HARD_CAP` env var. Resolved once
/// per process. `None` = no cap; `Some(cap)` = reject prefill requests
/// whose total length exceeds `cap`.
static KV_HARD_CAP: OnceLock<Option<i32>> = OnceLock::new();

/// Returns the configured hard cap on KV prefill length, if any.
fn kv_hard_cap() -> Option<i32> {
    *KV_HARD_CAP.get_or_init(|| {
        let raw = std::env::var("RMLX_KV_MAX_SEQ_HARD_CAP").ok()?;
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }
        match raw.parse::<i32>() {
            Ok(v) if v > 0 => {
                // Downgrade to debug — the OnceLock is resolved lazily on
                // first call (mid-decode), not at startup, so an info-level
                // event would appear out-of-band in the run log. Parse-error
                // branches stay at warn.
                tracing::debug!(cap = v, "KV hard cap enabled via RMLX_KV_MAX_SEQ_HARD_CAP");
                Some(v)
            }
            Ok(v) => {
                tracing::warn!(
                    value = v,
                    "RMLX_KV_MAX_SEQ_HARD_CAP must be a positive i32; ignoring"
                );
                None
            }
            Err(e) => {
                tracing::warn!(
                    value = raw,
                    error = %e,
                    "RMLX_KV_MAX_SEQ_HARD_CAP failed to parse as i32; ignoring"
                );
                None
            }
        }
    })
}

/// Next power-of-two ≥ `needed`, clamped to the closest power-of-two
/// representable as `i32`. `needed <= 0` is treated as 1.
pub(super) fn next_pow2_seq(needed: i32) -> i32 {
    if needed <= 1 {
        return 1;
    }
    // Largest i32 power-of-two is 1 << 30 (2^30 = 1_073_741_824).
    // 2^31 overflows i32. Saturate there.
    let max_pow2: i32 = 1 << 30;
    if needed >= max_pow2 {
        return max_pow2;
    }
    let n = needed as u32;
    // next_power_of_two on u32; safe because n <= max_pow2 < 2^31.
    let p = n.next_power_of_two();
    p as i32
}

/// Read the `max_seq` recorded on whichever `KvStorage` variant is active.
pub(super) fn storage_max_seq(storage: &KvStorage) -> i32 {
    match storage {
        KvStorage::K8V4 { max_seq, .. } => *max_seq,
        KvStorage::K8V8 { max_seq, .. } => *max_seq,
        KvStorage::Planar { max_seq, .. } => *max_seq,
        KvStorage::None { max_seq } => *max_seq,
        KvStorage::Mixed { max_seq, .. } => *max_seq,
        KvStorage::Paged { max_seq, .. } => *max_seq,
        KvStorage::RotKTq4V { max_seq, .. } => *max_seq,
        KvStorage::K8VTurbo3 { max_seq, .. } => *max_seq,
        KvStorage::TurboSym3 { max_seq, .. } => *max_seq,
        KvStorage::TurboSym4 { max_seq, .. } => *max_seq,
        KvStorage::PlanarK { max_seq, .. } => *max_seq,
        KvStorage::K8VTurbo2 { max_seq, .. } => *max_seq,
        KvStorage::IsoV3 { max_seq, .. } => *max_seq,
        KvStorage::IsoV4 { max_seq, .. } => *max_seq,
        KvStorage::RotorV3 { max_seq, .. } => *max_seq,
        KvStorage::RotorV4 { max_seq, .. } => *max_seq,
        KvStorage::K8VTurbo3Tcq { max_seq, .. } => *max_seq,
        KvStorage::K8VTurbo2Tcq { max_seq, .. } => *max_seq,
        KvStorage::IsoSym3 { max_seq, .. } => *max_seq,
        KvStorage::IsoSym4 { max_seq, .. } => *max_seq,
        KvStorage::IsoKOnly3 { max_seq, .. } => *max_seq,
        KvStorage::IsoKOnly4 { max_seq, .. } => *max_seq,
        KvStorage::RotorSym3 { max_seq, .. } => *max_seq,
        KvStorage::RotorSym4 { max_seq, .. } => *max_seq,
        KvStorage::RotorKOnly3 { max_seq, .. } => *max_seq,
        KvStorage::RotorKOnly4 { max_seq, .. } => *max_seq,
        KvStorage::RotorKAsym3 { max_seq, .. } => *max_seq,
        KvStorage::RotorKAsym4 { max_seq, .. } => *max_seq,
    }
}

/// Bump the `max_seq` recorded on the active storage variant. This is the
/// single source of truth read by `update_prefill_raw`, `exit_prefill`, and the
/// per-axis `QuantK::append` / `QuantV::append` capacity caps.
///
/// Two callers, with different payload states — neither needs bytes migrated
/// here:
///
/// * [`KvCache::ensure_prefill_capacity`] — no quantised payload exists yet
///   (`storage_has_materialised_payload` refuses the grow otherwise). The
///   storage is materialised later, in `exit_prefill`, from the now-larger raw
///   prefill buffer, and reads `max_seq` from here.
/// * [`KvCache::ensure_decode_capacity`] — the payload always exists. Raising
///   the scalar is exactly what lets each store's own grow path extend itself
///   on the next append: capacity is tracked per-store and the filled prefix is
///   copied forward on realloc.
///
/// In both cases this writes a scalar only; no buffer is resized here.
fn set_storage_max_seq(storage: &mut KvStorage, new_max_seq: i32) {
    match storage {
        KvStorage::K8V4 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::K8V8 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::Planar { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::None { max_seq } => *max_seq = new_max_seq,
        KvStorage::Mixed { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::Paged { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::RotKTq4V { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::K8VTurbo3 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::TurboSym3 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::TurboSym4 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::PlanarK { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::K8VTurbo2 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::IsoV3 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::IsoV4 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::RotorV3 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::RotorV4 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::K8VTurbo3Tcq { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::K8VTurbo2Tcq { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::IsoSym3 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::IsoSym4 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::IsoKOnly3 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::IsoKOnly4 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::RotorSym3 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::RotorSym4 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::RotorKOnly3 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::RotorKOnly4 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::RotorKAsym3 { max_seq, .. } => *max_seq = new_max_seq,
        KvStorage::RotorKAsym4 { max_seq, .. } => *max_seq = new_max_seq,
    }
}

#[cfg(test)]
#[path = "ring_feed_routing_tests.rs"]
mod ring_feed_routing_tests;
