// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel
// input data (dims arrays) for MSL dispatch.
#![allow(unsafe_code)]

//! `iso_flash_decode_symv`: fused QK over iso-quant K + online softmax +
//! **iso-quant V** SV in a single MSL pass per (B, H, tile).
//!
//! # What this is
//!
//! The all-quant sibling of [`crate::iso_flash_decode_msl`]. That kernel decodes
//! K from the iso store but still reads V out of a bf16 mirror; this one reads
//! **both** axes straight from their packed iso rings, so the symmetric iso
//! codecs ([`crate::storage::KvStorage::IsoSym3`] / `IsoSym4`) need no bf16 K or
//! V buffer at all.
//!
//! # What it replaced
//!
//! `Iso{3,4}Sym` quantised both axes at `exit_prefill` and then decoded from a
//! full bf16 K+V mirror (`decode_fp16_k` / `decode_fp16_v`), which
//! `update_iso{3,4}_sym` short-circuits to on its first line. The codec was
//! dormant: the packed store was written, never read. That bf16 mirror is the
//! whole KV cost of the codec — a codec advertising ~3 bits/axis actually
//! carried bf16 K + bf16 V *plus* its codes, i.e. **more** than plain bf16, the
//! heaviest cell in the KV matrix (~3.43×). Dropping the mirror is what turns
//! the codec's advertised compression into a real resident-byte win.
//!
//! Two Metal dispatches per call:
//! * **Pass 1** (`rmlx_iso_flash_decode_symv_p1_b{3,4}`): per-tile threadgroups
//!   compute partial outputs + per-tile LSE state `(tile_max, tile_sum_exp)`.
//!   Each threadgroup runs `head_dim` threads over `TILE_SIZE` tokens.
//! * **Pass 2** (`rmlx_iso_flash_decode_symv_p2`): one threadgroup per
//!   `(B, head)` merges the per-tile partials via log-sum-exp. The body is the
//!   codec-agnostic `metal/flash_decode_merge_p2.metal`, shared with
//!   `iso_flash_decode_msl` / `rotor_flash_decode_msl` / `planar_flash_decode_msl`.
//!
//! # Reuse of the K-decode half
//!
//! [`crate::iso_flash_decode_msl`] emits its per-lane iso decode into the
//! **header** as the MSL function `if_decode_k_lane(...)` precisely so a
//! quantized-V kernel could call it unchanged. This kernel does exactly that,
//! for both axes:
//!
//! * The header is [`crate::iso_flash_decode_msl::build_iso_flash_header`],
//!   reused verbatim — no second copy of the Lloyd-Max codebook or the fixed
//!   golden-ratio quaternion, and no second probe snapshot to keep in sync.
//! * The body calls `if_decode_k_lane` twice per token: once over the K ring,
//!   once over the V ring.
//!
//! That works because the iso codec is **axis-agnostic**: `iso_encode_fast`
//! quantises K and V the same way (`r = q * v_unit`, a single left Hamilton
//! product, decoded by `q̄ * r`). The per-lane decode is self-contained (lane
//! `d` reads only its own group's word / scale plus the token's L2 norm), so
//! unlike the rotor codec's Cl(3,0) block the iso V unpack needs neither
//! cross-lane exchange nor a threadgroup stage in the SV loop.
//!
//! # Fixed quaternion
//!
//! Both axes rotate every group by the same golden-ratio unit quaternion
//! ([`crate::isoquant::FIXED_QUAT`]), baked into the header as a constant. The
//! rings therefore do not carry a quaternion table on either axis;
//! [`crate::iso_flash_decode_msl::assert_fixed_quat_blocks`] rejects a store
//! whose per-group quaternions are not the fixed constant, so a future
//! per-group-quaternion encoder fails loudly at the dispatch boundary rather
//! than decoding against the wrong rotation.
//!
//! # Bit-width parameterisation
//!
//! `bits ∈ {3, 4}` is a template parameter carried by the header, so one
//! `metal/iso_flash_decode_symv_p1.metal` serves both variants. Selection is
//! explicit — any other `bits` is an `Err`, never a silent fallback to the wrong
//! unpack width. Both axes of a symmetric codec share one width by construction
//! (`Iso3Sym` → 3/3, `Iso4Sym` → 4/4), so a single `IF_BITS` covers K and V.
//!
//! # Codec contract
//!
//! Bit-exact with [`crate::isoquant::iso_decode_fast`] on **both** axes. Per
//! axis:
//!
//! * `codes`  u32 `[B * S_kv * kv_h * n_groups]` — 1 u32 per group of 4
//!   quaternion components (both widths fit one group in one word).
//! * `scales` f32 `[B * S_kv * kv_h * n_groups]` — one f32 per group of 4.
//! * `norms`  f32 `[B * S_kv * kv_h]` — per-token L2 norm.
//!
//! Both rings are **sequence-major** (`[B, S, kv_h, ...]`).
//!
//! # Single-MLX claim
//!
//! Per CLAUDE.md "Single MLX process per Mac", callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.
//!
//! # Pattern reference
//!
//! * [`crate::iso_flash_decode_msl`] — the bf16-V sibling; this kernel is its
//!   shell with the SV read swapped for an iso unpack.
//! * [`crate::rotor_flash_decode_symv_msl`] — the rotor quant-V sibling (the
//!   same shell over the Cl(3,0) codec).
//! * [`crate::isoquant::iso_decode_fast`] — CPU reference (Rust).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

use crate::flash_decode_common::pad_norms_to_device_floor;
use crate::iso_flash_decode_msl::{build_iso_flash_header, ISO_FLASH_HEAD_DIM_MAX, TILE_SIZE};
use crate::storage::{iso_n_groups_for, ISO_QUAT_BLOCK_SIZE};

// ── Dispatch counters ─────────────────────────────────────────────────────────

/// Incremented once per `iso_flash_decode_symv_sdpa::<3>` P1 enqueue.
static ISO3_SYMV_FLASH_DECODE_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Incremented once per `iso_flash_decode_symv_sdpa::<4>` P1 enqueue.
static ISO4_SYMV_FLASH_DECODE_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Process-lifetime count of iso3 quant-V flash-decode P1 dispatches.
///
/// Tests assert `delta > 0` to prove the MSL kernel actually fired rather than
/// the caller silently falling back to the CPU dequant path. Production code
/// does not consult this counter.
pub fn iso3_symv_flash_decode_dispatch_count() -> u64 {
    ISO3_SYMV_FLASH_DECODE_DISPATCHES.load(Ordering::Relaxed)
}

/// Process-lifetime count of iso4 quant-V flash-decode P1 dispatches.
pub fn iso4_symv_flash_decode_dispatch_count() -> u64 {
    ISO4_SYMV_FLASH_DECODE_DISPATCHES.load(Ordering::Relaxed)
}

/// Combined process-lifetime count (3-bit + 4-bit).
pub fn iso_symv_flash_decode_dispatch_count() -> u64 {
    iso3_symv_flash_decode_dispatch_count() + iso4_symv_flash_decode_dispatch_count()
}

// ── MSL sources ───────────────────────────────────────────────────────────────
//
// Grid: (n_tiles * head_dim, B * n_q_heads, 1).  Threadgroup: (head_dim, 1, 1).
//
// P1 buffer layout (must match `add_input` order in
// `iso_flash_decode_symv_sdpa`). BOTH iso rings are SEQUENCE-major
// (`[B, S, kv_h, ...]`) — there is no head-major bf16 V here.
// 0.  query     : f32  [B * n_q_heads * head_dim]
// 1.  k_codes   : u32  [B * kv_seq * kv_h * n_groups]
// 2.  k_scales  : f32  [B * kv_seq * kv_h * n_groups]
// 3.  k_norms   : f32  [B * kv_seq * kv_h]
// 4.  v_codes   : u32  [B * kv_seq * kv_h * n_groups]
// 5.  v_scales  : f32  [B * kv_seq * kv_h * n_groups]
// 6.  v_norms   : f32  [B * kv_seq * kv_h]
// 7.  mask_flat : f32  [B * n_q_heads * kv_seq] or [1] dummy when no mask
// 8.  scale_arr : f32  [1]
// 9.  dims      : u32  [8] — {head_dim, kv_seq, n_bh, kv_h, heads_per_kv,
//                             n_tiles, has_mask, n_groups}
//
// P1 outputs:
// 0. partial_o     : f32 [n_tiles * n_bh * head_dim]
// 1. tile_max      : f32 [n_tiles * n_bh]
// 2. tile_sum_exp  : f32 [n_tiles * n_bh]
//
// One body serves both bit widths — the unpack width arrives via the header's
// IF_BITS / IF_MASK.
const P1_SOURCE: &str = include_str!("metal/iso_flash_decode_symv_p1.metal");

// P2 buffer layout:
// 0. partial_o    : f32 [n_tiles * n_bh * head_dim]
// 1. tile_max     : f32 [n_tiles * n_bh]
// 2. tile_sum_exp : f32 [n_tiles * n_bh]
// 3. dims_p2      : u32 [3] — {head_dim, n_tiles, n_bh}
// Output:
// 0. dst          : f32 [n_bh * head_dim]
//
// Codec-agnostic; shared with `iso_flash_decode_msl` / `rotor_flash_decode_msl`.
const P2_SOURCE: &str = include_str!("metal/flash_decode_merge_p2.metal");

// ── Kernel singletons (one P1 per BITS variant) ───────────────────────────────

static P1_KERNEL_B3: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P1_KERNEL_B4: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P2_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();

fn p1_kernel(bits: u8) -> Result<&'static MetalKernel> {
    let (cell, name) = match bits {
        3 => (&P1_KERNEL_B3, "rmlx_iso_flash_decode_symv_p1_b3"),
        4 => (&P1_KERNEL_B4, "rmlx_iso_flash_decode_symv_p1_b4"),
        _ => {
            return Err(Error::Quant(format!(
                "iso_flash_decode_symv: bits must be 3 or 4, got {bits}"
            )))
        }
    };
    cell.get_or_init(|| {
        // Same header as the bf16-V sibling: IF_BITS / IF_MASK / ISO_CB / the
        // fixed quaternion / `if_decode_k_lane` / tile + head-dim sizing. Both
        // axes of a symmetric codec share one bit width, so one header covers
        // the K and the V unpack.
        let header = build_iso_flash_header(bits).map_err(|e| format!("{e}"))?;
        MetalKernel::new(
            name,
            &header,
            P1_SOURCE,
            &[
                "query",
                "k_codes",
                "k_scales",
                "k_norms",
                "v_codes",
                "v_scales",
                "v_norms",
                "mask_flat",
                "scale_arr",
                "dims",
            ],
            &["partial_o", "tile_max", "tile_sum_exp"],
        )
        .map_err(|e| format!("{e}"))
    })
    .as_ref()
    .map_err(|e| {
        Error::Mlx(format!(
            "iso_flash_decode_symv P1(bits={bits}) kernel init: {e}"
        ))
    })
}

fn p2_kernel() -> Result<&'static MetalKernel> {
    P2_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_iso_flash_decode_symv_p2",
                "", // No header — P2 is generic over (head_dim, n_tiles, n_bh).
                P2_SOURCE,
                &["partial_o", "tile_max", "tile_sum_exp", "dims_p2"],
                &["dst"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("iso_flash_decode_symv P2 kernel init: {e}")))
}

/// The packed iso payload of one axis, as the dispatcher takes it.
///
/// Grouping the three buffers keeps `iso_flash_decode_symv_sdpa` from taking six
/// positional `&Array`s in a row, where a K/V transposition at a call site would
/// type-check and decode the wrong store.
#[derive(Debug, Clone, Copy)]
#[allow(
    clippy::exhaustive_structs,
    reason = "plain parameter bundle — literal construction at the call site IS the intended use, \
              and the field set is the codec's on-disk contract (codes/scales/norms). A new field \
              would be a new codec layout, which must break every call site anyway"
)]
pub struct IsoPackedAxis<'a> {
    /// Packed iso codes, flat `u32 [B * kv_seq * kv_h * n_groups]`
    /// (sequence-major).
    pub codes: &'a Array,
    /// Per-group f32 scales, same flat length as `codes`.
    pub scales: &'a Array,
    /// Per-token f32 L2 norms, flat `[B * kv_seq * kv_h]`.
    pub norms: &'a Array,
}

/// Shape metadata for one decode step.
#[derive(Debug, Clone, Copy)]
#[allow(
    clippy::exhaustive_structs,
    reason = "plain parameter bundle — literal construction at the call site IS the intended use; \
              every field is a required shape the dispatcher validates, so there is no default a \
              non-exhaustive builder could supply"
)]
pub struct IsoFlashShape {
    /// Batch size. Must be 1 — the rings do not interleave batch in their
    /// per-step stride.
    pub b: i32,
    /// KV head count.
    pub kv_h: i32,
    /// Accumulated KV positions to attend.
    pub kv_seq: i32,
    /// Per-head dimension. Power of two, multiple of the quaternion block size,
    /// `<= ISO_FLASH_HEAD_DIM_MAX`.
    pub head_dim: i32,
    /// Query heads per KV head (GQA fan-out).
    pub heads_per_kv: i32,
}

// ── Public dispatcher ─────────────────────────────────────────────────────────

/// Run the iso symmetric flash-decode kernel (fused QK over iso-quant K + online
/// softmax + **iso-quant V** SV).
///
/// # Inputs
///
/// * `query` — Q for the new token, shape `[B, n_q_heads, 1, head_dim]`.
/// * `k` / `v` — the two packed iso axes ([`IsoPackedAxis`]).
/// * `additive_mask` — optional `f32 [B, n_q_heads, 1, kv_seq]`.
/// * `shape` — [`IsoFlashShape`].
/// * `scale` — softmax pre-scale (typically `1/sqrt(head_dim)`).
/// * `device` — MLX device (must be GPU).
///
/// # Output
///
/// `f32` array of shape `[B, n_q_heads, 1, head_dim]`.
///
/// # Errors
///
/// * [`Error::Quant`] for `BITS` outside `{3, 4}`, shape-contract violations
///   (`head_dim` not a power of two, not a multiple of the quaternion block
///   size, above [`ISO_FLASH_HEAD_DIM_MAX`], non-positive shapes), or grid
///   overflow.
/// * [`Error::Mlx`] for kernel build / dispatch failures.
#[allow(clippy::too_many_lines)]
pub fn iso_flash_decode_symv_sdpa<const BITS: u8>(
    query: &Array,
    k: IsoPackedAxis<'_>,
    v: IsoPackedAxis<'_>,
    additive_mask: Option<&Array>,
    shape: IsoFlashShape,
    scale: f32,
    device: Device,
) -> Result<Array> {
    if BITS != 3 && BITS != 4 {
        return Err(Error::Quant(format!(
            "iso_flash_decode_symv: unsupported BITS={BITS} (only 3 and 4)"
        )));
    }
    let IsoFlashShape {
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
    } = shape;

    if head_dim <= 0 {
        return Err(Error::Quant(format!(
            "iso_flash_decode_symv: head_dim={head_dim} must be positive"
        )));
    }
    if !(head_dim as u32).is_power_of_two() {
        return Err(Error::Quant(format!(
            "iso_flash_decode_symv: head_dim={head_dim} must be a power of two for the \
             tree reduction; caller falls back to the CPU dequant path"
        )));
    }
    if head_dim > ISO_FLASH_HEAD_DIM_MAX {
        let max = ISO_FLASH_HEAD_DIM_MAX;
        return Err(Error::Quant(format!(
            "iso_flash_decode_symv: head_dim={head_dim} exceeds \
             ISO_FLASH_HEAD_DIM_MAX={max}; raise the static threadgroup-array sizes in \
             metal/iso_flash_decode_symv_p1.metal to support it"
        )));
    }
    if !(head_dim as usize).is_multiple_of(ISO_QUAT_BLOCK_SIZE) {
        return Err(Error::Quant(format!(
            "iso_flash_decode_symv: head_dim={head_dim} must be a multiple of the quaternion \
             block size {ISO_QUAT_BLOCK_SIZE}"
        )));
    }
    if heads_per_kv <= 0 {
        return Err(Error::Quant(format!(
            "iso_flash_decode_symv: heads_per_kv must be > 0, got {heads_per_kv}"
        )));
    }
    if b <= 0 || kv_seq <= 0 || kv_h <= 0 {
        return Err(Error::Quant(format!(
            "iso_flash_decode_symv: b={b}, kv_h={kv_h}, kv_seq={kv_seq} must all be > 0"
        )));
    }

    let n_q_heads = kv_h * heads_per_kv;
    let n_bh = b * n_q_heads;
    let n_tiles = (kv_seq + TILE_SIZE - 1) / TILE_SIZE;
    let n_groups = iso_n_groups_for(head_dim as usize);
    let n_groups_i64 = n_groups as i64;

    // ── Flatten Q to [n_bh * head_dim] f32 ────────────────────────────────
    let q_total: i64 = i64::from(n_bh) * i64::from(head_dim);
    let q_flat = query.reshape(&[q_total as i32], device)?;
    let q_f32 = if q_flat.dtype() == Dtype::F32 {
        q_flat
    } else {
        q_flat.astype(Dtype::F32, device)?
    };

    // ── Flatten both packed axes ──────────────────────────────────────────
    let tok_count: i64 = i64::from(b) * i64::from(kv_h) * i64::from(kv_seq);
    let codes_total: i64 = tok_count * n_groups_i64;

    let flatten_axis = |axis: IsoPackedAxis<'_>| -> Result<(Array, Array, Array)> {
        Ok((
            axis.codes.reshape(&[codes_total as i32], device)?,
            axis.scales.reshape(&[codes_total as i32], device)?,
            axis.norms.reshape(&[tok_count as i32], device)?,
        ))
    };
    let (k_codes, k_scales, k_norms) = flatten_axis(k)?;
    let (v_codes, v_scales, v_norms) = flatten_axis(v)?;

    // GPU-native small-`norms`-buffer floor — see
    // [`crate::flash_decode_common::pad_norms_to_device_floor`].
    let k_norms = pad_norms_to_device_floor(k_norms, tok_count, device)?;
    let v_norms = pad_norms_to_device_floor(v_norms, tok_count, device)?;

    // ── Mask ──────────────────────────────────────────────────────────────
    let (mask_flat, has_mask) = if let Some(m) = additive_mask {
        let flat_len: i64 = i64::from(n_bh) * i64::from(kv_seq);
        let m_f = if m.dtype() == Dtype::F32 {
            m.reshape(&[flat_len as i32], device)?
        } else {
            m.astype(Dtype::F32, device)?
                .reshape(&[flat_len as i32], device)?
        };
        (m_f, 1u32)
    } else {
        let zero_bytes = [0u8; 4];
        Array::from_bytes(&zero_bytes, &[1], Dtype::F32)
            .map(|a| (a, 0u32))
            .map_err(|e| Error::Mlx(format!("iso_flash_decode_symv dummy mask: {e}")))?
    };

    // ── scale_arr ─────────────────────────────────────────────────────────
    let scale_arr = {
        let bytes = scale.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::F32)?
    };

    // ── dims (8 u32) ──────────────────────────────────────────────────────
    // `bits` is not carried — the dispatcher selects the per-BITS kernel and
    // its header supplies IF_BITS / IF_MASK.
    let dims_arr = {
        let dims: [u32; 8] = [
            head_dim as u32,
            kv_seq as u32,
            n_bh as u32,
            kv_h as u32,
            heads_per_kv as u32,
            n_tiles as u32,
            has_mask,
            n_groups as u32,
        ];
        // SAFETY:
        // * `dims` is a stack-local `[u32; 8]` fully initialised above.
        // * `u32` has stricter alignment than `u8`, so the cast is sound.
        // * The byte length `8 * 4` equals `size_of::<[u32; 8]>()`.
        // * The borrow is bounded by the enclosing block; `Array::from_bytes`
        //   copies into mlx storage before this scope ends.
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(dims.as_ptr().cast::<u8>(), 8 * 4) };
        Array::from_bytes(bytes, &[8], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("iso_flash_decode_symv dims: {e}")))?
    };

    // Materialise inputs to flush any pending lazy ops before kernel dispatch.
    // The MSL kernels read by raw linear offset and ignore MLX lazy-transpose
    // strides, so a pending permutation must be resolved here.
    q_f32.eval()?;
    for arr in [&k_codes, &k_scales, &k_norms, &v_codes, &v_scales, &v_norms] {
        arr.eval()?;
    }
    if has_mask == 1 {
        mask_flat.eval()?;
    }
    scale_arr.eval()?;
    dims_arr.eval()?;

    // ── P1 dispatch ───────────────────────────────────────────────────────
    let kern_p1 = p1_kernel(BITS)?;
    let mut inv_p1 = MetalKernelInvoke::new();
    inv_p1.add_input(&q_f32)?;
    inv_p1.add_input(&k_codes)?;
    inv_p1.add_input(&k_scales)?;
    inv_p1.add_input(&k_norms)?;
    inv_p1.add_input(&v_codes)?;
    inv_p1.add_input(&v_scales)?;
    inv_p1.add_input(&v_norms)?;
    inv_p1.add_input(&mask_flat)?;
    inv_p1.add_input(&scale_arr)?;
    inv_p1.add_input(&dims_arr)?;

    let partial_o_len: i64 = i64::from(n_tiles) * i64::from(n_bh) * i64::from(head_dim);
    let tile_meta_len: i64 = i64::from(n_tiles) * i64::from(n_bh);
    if partial_o_len > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "iso_flash_decode_symv: partial_o length {partial_o_len} exceeds i32::MAX"
        )));
    }
    inv_p1.add_output_shape(&[partial_o_len as i32], Dtype::F32)?;
    inv_p1.add_output_shape(&[tile_meta_len as i32], Dtype::F32)?;
    inv_p1.add_output_shape(&[tile_meta_len as i32], Dtype::F32)?;

    // Grid X: n_tiles threadgroups × head_dim threads each.
    let grid_x: i64 = i64::from(n_tiles) * i64::from(head_dim);
    let grid_y: i64 = i64::from(n_bh);
    if grid_x > i64::from(i32::MAX) || grid_y > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "iso_flash_decode_symv: grid dimensions exceed i32::MAX (x={grid_x}, y={grid_y})"
        )));
    }
    inv_p1.set_grid(grid_x as i32, grid_y as i32, 1)?;
    inv_p1.set_thread_group(head_dim, 1, 1)?;

    // Counter increment at the actual P1 enqueue point — after every
    // validation gate, immediately before `.apply()`.
    if BITS == 3 {
        ISO3_SYMV_FLASH_DECODE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        ISO4_SYMV_FLASH_DECODE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    }
    tracing::trace!(
        bits = BITS,
        b,
        n_q_heads,
        kv_h,
        kv_seq,
        head_dim,
        has_mask,
        n_groups,
        n_tiles,
        "iso_flash_decode_symv_sdpa: dispatch"
    );

    let mut p1_outs = kern_p1.apply(inv_p1, device)?;
    if p1_outs.len() < 3 {
        return Err(Error::Mlx(
            "iso_flash_decode_symv P1: expected 3 outputs".into(),
        ));
    }
    let partial_o = p1_outs.remove(0);
    let tile_max = p1_outs.remove(0);
    let tile_sum_exp = p1_outs.remove(0);

    // ── P2 dispatch ───────────────────────────────────────────────────────
    let dims_p2_arr = {
        let dims_p2: [u32; 3] = [head_dim as u32, n_tiles as u32, n_bh as u32];
        // SAFETY: same reasoning as the `dims` cast above — stack-local
        // `[u32; 3]`, `u32` alignment ≥ `u8`, byte length `3 * 4` equals
        // `size_of::<[u32; 3]>()`, and `Array::from_bytes` copies before the
        // borrow ends.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(dims_p2.as_ptr().cast::<u8>(), 3 * 4) };
        Array::from_bytes(bytes, &[3], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("iso_flash_decode_symv dims_p2: {e}")))?
    };

    let kern_p2 = p2_kernel()?;
    let mut inv_p2 = MetalKernelInvoke::new();
    inv_p2.add_input(&partial_o)?;
    inv_p2.add_input(&tile_max)?;
    inv_p2.add_input(&tile_sum_exp)?;
    inv_p2.add_input(&dims_p2_arr)?;

    let dst_len: i64 = i64::from(n_bh) * i64::from(head_dim);
    inv_p2.add_output_shape(&[dst_len as i32], Dtype::F32)?;

    let p2_grid_x: i64 = i64::from(n_bh) * i64::from(head_dim);
    if p2_grid_x > i64::from(i32::MAX) {
        return Err(Error::Quant(format!(
            "iso_flash_decode_symv P2: grid x {p2_grid_x} exceeds i32::MAX"
        )));
    }
    inv_p2.set_grid(p2_grid_x as i32, 1, 1)?;
    inv_p2.set_thread_group(head_dim, 1, 1)?;

    let mut p2_outs = kern_p2.apply(inv_p2, device)?;
    if p2_outs.is_empty() {
        return Err(Error::Mlx(
            "iso_flash_decode_symv P2: expected 1 output".into(),
        ));
    }
    let dst_flat = p2_outs.remove(0);

    // Reshape to canonical SDPA output.
    dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)
}

#[cfg(test)]
#[path = "iso_flash_decode_symv_msl_tests.rs"]
mod iso_flash_decode_symv_msl_tests;
