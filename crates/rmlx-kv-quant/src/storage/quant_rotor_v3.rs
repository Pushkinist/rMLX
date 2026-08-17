// Rotor3 (Cl(3,0) Clifford sandwich) 3-bit V storage.
//
// Mirrors `quant_iso_v.rs` (`QuantIsoV3`) for the per-token bookkeeping but
// holds a **static per-(layer, head)** rotor table — generated once on first
// append, never per-token. Tokens carry only (codes, scales, norms).
#![allow(
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::indexing_slicing,
    clippy::doc_lazy_continuation
)]
//! Quantized V buffer: `QuantRotorV3` (rotor3 V codec).

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::clifford::make_rotor_table;
use crate::rotorquant::{
    n_groups_for, rotor3_decode, rotor3_encode, RotorQuantError, ROTOR3_BITS, ROTOR3_GROUP_SIZE,
};

use super::QuantKGpuRing;

/// Bit-width of the rotor3 V codec (fixed at 3-bit — see
/// [`crate::rotorquant::rotor3_encode`]).
pub const ROTOR3_V_BITS: u8 = ROTOR3_BITS;

/// Multivector group size (3 grade-1 elements per group; one rotor).
pub const ROTOR3_V_GROUP_SIZE: usize = ROTOR3_GROUP_SIZE;

/// One token-batch's rotor3 payload: codes + per-group scales + per-token L2 norm.
///
/// The rotor table is **not** stored per-block — it lives on the parent
/// `QuantRotorV3` (one table per layer/head). Each `RotorBlocks` is one
/// append-call's worth of tokens.
#[derive(Debug, Clone)]
pub struct RotorBlocks {
    /// Packed 3-bit codes; pack convention = 10 vals/u32 (planar3 / iso3).
    /// Length per block = `n_tokens * n_groups * 1` u32 words.
    pub codes: Vec<u32>,
    /// Per-group scale: `n_tokens * n_groups` f32 entries (flat).
    pub scales: Vec<f32>,
    /// Per-token L2 norm: `n_tokens` f32 entries.
    pub norms: Vec<f32>,
    /// Number of tokens this block represents.
    pub n_tokens: usize,
}

impl RotorBlocks {
    /// Heap bytes this block holds.
    ///
    /// The exhaustive destructure is the drift guard: a new payload field
    /// cannot be added without this failing to compile. See [`crate::bytes`].
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            codes,
            scales,
            norms,
            // Inline metadata (no heap payload).
            n_tokens: _,
        } = self;
        crate::bytes::vec_bytes(codes)
            + crate::bytes::vec_bytes(scales)
            + crate::bytes::vec_bytes(norms)
    }
}

impl super::BlockRows for RotorBlocks {
    /// The exhaustive destructure is the drift guard: a new payload field
    /// cannot be added without this failing to compile, which is what stops a
    /// buffer from surviving a mid-block truncation at its full length.
    fn retain_rows(&mut self, rows: usize) -> bool {
        let Self {
            codes,
            scales,
            norms,
            n_tokens,
        } = self;
        let lengths = [codes.len(), scales.len(), norms.len()];
        if !super::rows_split_ok(&lengths, *n_tokens, rows) {
            return false;
        }
        super::retain_rows_in(codes, *n_tokens, rows);
        super::retain_rows_in(scales, *n_tokens, rows);
        super::retain_rows_in(norms, *n_tokens, rows);
        *n_tokens = rows;
        true
    }
}

/// Accumulated rotor3 V cache.
///
/// Holds:
///   * `rotors` — static `[n_groups, 4]` table generated once at first append
///     (or supplied by SSD hydrate). Reseeded via
///     [`crate::clifford::make_rotor_table`] using (`layer_idx`, `head_idx`).
///   * `blocks` — per-append payload (`RotorBlocks`).
///   * `shape` — accumulated `[B, kv_h, S_total, D]`.
///
/// Carries both forms of the payload:
///
/// * `blocks` — CPU `RotorBlocks`, the source of truth for `dequant()` and the
///   SSD spill/hydrate round-trip.
/// * `gpu` — the optional GPU-resident packed ring ([`QuantKGpuRing`]),
///   populated by `gpu_append`. When present it lets a fused flash-decode kernel
///   read the V quant store directly instead of attending a bf16 mirror of V.
///
/// The ring type is named for the K side it landed with, but its payload
/// (codes / per-group scales / per-token L2 norms) is exactly the rotor codec's
/// own — the codec is axis-agnostic, so the V side reuses it verbatim.
pub struct QuantRotorV3 {
    /// Static rotor table for this layer/head, flat `[n_groups * 4]` f32 in
    /// `[s, b12, b13, b23]` per-rotor order. Initialised lazily on first
    /// `append`; never replaced.
    pub rotors: Vec<f32>,
    /// GPU-resident packed ring. Empty until the first `gpu_append`.
    pub gpu: QuantKGpuRing,
    /// Accumulated per-token blocks (one entry per append call; `dequant`
    /// flattens them).
    pub blocks: Vec<RotorBlocks>,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Maximum sequence length the storage was provisioned for.
    pub max_seq: i32,
    /// Layer index (0-based) — used to seed the rotor table.
    pub layer_idx: u32,
    /// Head index (0-based) — used to seed the rotor table.
    /// Currently always `0` (one rotor table per layer; per-head decorrelation
    /// is via the codec's per-token scale + norm, not per-head rotors).
    pub head_idx: u32,
    /// Bit-width tag (always [`ROTOR3_V_BITS`] for this codec).
    pub bits: u8,
}

impl std::fmt::Debug for QuantRotorV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantRotorV3")
            .field("n_rotors", &(self.rotors.len() / 4))
            .field("gpu_resident", &self.gpu.is_allocated())
            .field("n_blocks", &self.blocks.len())
            .field("shape", &self.shape)
            .field("max_seq", &self.max_seq)
            .field("layer_idx", &self.layer_idx)
            .field("head_idx", &self.head_idx)
            .field("bits", &self.bits)
            .finish()
    }
}

impl QuantRotorV3 {
    /// Construct an empty `QuantRotorV3` for `init_shape = [B, kv_h, 0, D]`.
    ///
    /// `rotors` is left empty — it is generated lazily on the first `append`
    /// call once `head_dim = init_shape[3]` is known. Use
    /// [`Self::with_rotors`] (or [`Self::from_cpu_blocks`]) if a pre-computed
    /// rotor table is on hand (SSD hydrate path).
    #[must_use]
    pub fn new(init_shape: Vec<i32>, max_seq: i32, layer_idx: u32) -> Self {
        Self {
            rotors: Vec::new(),
            gpu: QuantKGpuRing::default(),
            blocks: Vec::new(),
            shape: init_shape,
            max_seq,
            layer_idx,
            head_idx: 0,
            bits: ROTOR3_V_BITS,
        }
    }

    /// Construct with a pre-supplied rotor table (used by SSD hydrate +
    /// round-trip tests).
    #[must_use]
    pub fn with_rotors(
        rotors: Vec<f32>,
        init_shape: Vec<i32>,
        max_seq: i32,
        layer_idx: u32,
    ) -> Self {
        Self {
            rotors,
            gpu: QuantKGpuRing::default(),
            blocks: Vec::new(),
            shape: init_shape,
            max_seq,
            layer_idx,
            head_idx: 0,
            bits: ROTOR3_V_BITS,
        }
    }

    /// Build a `QuantRotorV3` from pre-computed CPU blocks (SSD hydrate path).
    #[must_use]
    pub fn from_cpu_blocks(
        rotors: Vec<f32>,
        blocks: Vec<RotorBlocks>,
        shape: Vec<i32>,
        layer_idx: u32,
    ) -> Self {
        debug_assert!(
            shape.len() == 4,
            "QuantRotorV3::from_cpu_blocks expects a 4-element [B, kv_h, S, D] shape, got {shape:?}"
        );
        let max_seq = if shape.len() >= 3 { shape[2] } else { 0 };
        Self {
            rotors,
            // Hydrated caches start CPU-only; the ring is rebuilt lazily from
            // the next GPU append.
            gpu: QuantKGpuRing::default(),
            blocks,
            shape,
            max_seq,
            layer_idx,
            head_idx: 0,
            bits: ROTOR3_V_BITS,
        }
    }

    /// Append one V slice (CPU path).
    ///
    /// On the first call the rotor table is generated via
    /// [`make_rotor_table`] using the stored `layer_idx` / `head_idx`.
    /// Subsequent calls reuse the same table.
    ///
    /// # Errors
    ///
    /// Forwards any [`RotorQuantError`] from [`rotor3_encode`].
    pub fn append(&mut self, f32_data: &[f32], new_shape: &[i32]) -> Result<()> {
        if new_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantRotorV3::append: expected 4D new_shape, got {new_shape:?}"
            )));
        }
        let b = new_shape[0] as usize;
        let kv_h = new_shape[1] as usize;
        let new_seq = new_shape[2] as usize;
        let head_dim = new_shape[3] as usize;
        let n_tokens_total = b * kv_h * new_seq;

        if self.rotors.is_empty() {
            let n_groups = n_groups_for(head_dim);
            self.rotors = make_rotor_table(self.layer_idx, self.head_idx, n_groups);
        }

        // Store each chunk sequence-major (`[B, new_seq, kv_h, D]`) so the
        // per-append blocks share one layout; a head-major store transposes
        // heads across multi-append GQA caches (kv_h>1) when `dequant` reshapes
        // head-major over the full sequence. The static rotor table is indexed
        // by group position within `head_dim` (not by token), so the reorder
        // leaves it correctly associated. See [`super::QuantIsoV3::append`].
        let seq_major =
            super::seq_layout::transpose_heads_seq(f32_data, b, kv_h, new_seq, head_dim);

        let (codes, scales, norms) = rotor3_encode(&seq_major, &self.rotors, head_dim)
            .map_err(|e: RotorQuantError| Error::Mlx(format!("rotor3 encode: {e}")))?;

        self.blocks.push(RotorBlocks {
            codes,
            scales,
            norms,
            n_tokens: n_tokens_total,
        });

        // A CPU append does not touch the GPU ring, so any live ring is now a
        // stale prefix. Drop it; the next `gpu_append` re-seeds from `blocks`.
        self.gpu.clear();

        if self.shape.len() != 4 || self.shape[0] == 0 {
            self.shape = new_shape.to_vec();
        } else {
            self.shape[2] += new_shape[2];
        }
        Ok(())
    }

    /// Reset the accumulated sequence to zero. Buffers are kept for reuse.
    /// Does **not** touch the rotor table — that is layer-static.
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.gpu.clear();
        if self.shape.len() >= 4 {
            self.shape[2] = 0;
        }
    }

    /// Truncate the accumulated sequence to `n` tokens.
    ///
    /// Drops trailing blocks past `n` and **splits** the block the cut lands
    /// inside (block `n_tokens` counts rows, not sequence positions — see
    /// [`super::truncate_plan`]), then lowers `shape[2]` to `n`. Does **not**
    /// touch the rotor table.
    ///
    /// The GPU ring is **kept**, not cleared — mirror of the K store's
    /// `truncate_to`. Lowering `shape[2]` to `n` makes the ring's logical fill
    /// `n`; the stale `[n, prev)` capacity is overwritten by the next append and
    /// never read (`packed_view` slices to `shape[2]`). This preserves any
    /// ring-only decode tail up to `n`, so `dequant` / an SSD spill can still
    /// rebuild it via `synced_rotor_v_blocks`. Clearing the ring here would
    /// discard the tail (the only copy of `[frozen_prefix, n)`), leaving `blocks`
    /// short of `shape[2]` with no ring — the divergent state `dequant` rejects
    /// loudly, which would abort generation on the speculative-decode rollback
    /// path.
    pub fn truncate_to(&mut self, n: i32) {
        let n = n.max(0);
        let plan = super::truncate_plan(self.blocks.iter().map(|blk| blk.n_tokens), &self.shape, n);
        super::apply_truncate_plan(&mut self.blocks, &plan);
        // NB: no `self.gpu.clear()` — the ring is the source of truth for a
        // ring-only decode tail; see the doc comment above.
        if self.shape.len() >= 4 {
            self.shape[2] = n;
        }
    }

    /// Deep-clone.
    ///
    /// Materialises any ring-only decode tail into complete CPU blocks first:
    /// the clone starts CPU-only (the ring is not cloned), and both the
    /// prompt-cache snapshot and the SSD spill clone route through here, so this
    /// is the single point where a store leaving the live decode loop reconciles
    /// its blocks with the ring. A short-blocks clone with no ring would silently
    /// truncate the store — refused loudly by `synced_rotor_v_blocks` instead.
    ///
    /// # Errors
    ///
    /// Forwards a [`synced_rotor_v_blocks`] reconciliation error (blocks over-run
    /// `shape[2]`, or a ring-only tail exists but the ring is absent / too short).
    pub fn try_deep_clone(&self) -> Result<Self> {
        let blocks =
            synced_rotor_v_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?.into_owned();
        Ok(Self {
            rotors: self.rotors.clone(),
            // The clone starts CPU-only: `blocks` carries the full payload, so
            // the ring re-seeds from them on the clone's first GPU append.
            // Sharing the source's Arrays would alias one ring across two
            // independent caches.
            gpu: QuantKGpuRing::default(),
            blocks,
            shape: self.shape.clone(),
            max_seq: self.max_seq,
            layer_idx: self.layer_idx,
            head_idx: self.head_idx,
            bits: self.bits,
        })
    }

    /// Concatenate the accumulated CPU blocks into flat sequence-major
    /// `(codes, scales, norms)` — the form [`QuantKGpuRing::seed_from_cpu`]
    /// wants.
    fn flatten_blocks(&self) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        // Exact capacities — the prefill seed concatenates the whole prefix
        // (millions of entries at long context), so growing from empty would
        // realloc+memcpy repeatedly.
        let (n_codes, n_scales, n_norms) = self.blocks.iter().fold((0, 0, 0), |(c, s, n), blk| {
            (
                c + blk.codes.len(),
                s + blk.scales.len(),
                n + blk.norms.len(),
            )
        });
        let mut codes = Vec::with_capacity(n_codes);
        let mut scales = Vec::with_capacity(n_scales);
        let mut norms = Vec::with_capacity(n_norms);
        for blk in &self.blocks {
            codes.extend_from_slice(&blk.codes);
            scales.extend_from_slice(&blk.scales);
            norms.extend_from_slice(&blk.norms);
        }
        (codes, scales, norms)
    }

    /// Push one GPU-encoded chunk into the GPU ring, seeding the ring from the
    /// accumulated CPU blocks first when it is not yet live.
    ///
    /// `codes` / `scales` / `norms` are the rotor encode kernel's GPU outputs
    /// for a sequence-major chunk; `prev_seq` is the accumulated sequence length
    /// **before** this chunk.
    ///
    /// This only maintains the ring — the caller still pushes the matching CPU
    /// block, which stays the source of truth for `dequant()` and SSD spill.
    ///
    /// `max_seq` is the window the cache is provisioned for **right now**, read
    /// from the active `KvStorage` variant by the caller. It is a parameter
    /// rather than the `self.max_seq` field so it cannot go stale as the window
    /// grows during decode — the same contract the K-side ring uses.
    ///
    /// # Errors
    ///
    /// Forwards [`QuantKGpuRing::seed_from_cpu`] / [`QuantKGpuRing::append_encoded`]
    /// errors.
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_append(
        &mut self,
        codes: &Array,
        scales: &Array,
        norms: &Array,
        kv_h: i32,
        head_dim: i32,
        prev_seq: i32,
        new_seq: i32,
        max_seq: i32,
        device: Device,
    ) -> Result<()> {
        // The ring is codec-agnostic and takes `n_groups`; the rotor group rule
        // stays here, in the codec's own store.
        let n_groups = i32::try_from(n_groups_for(usize::try_from(head_dim.max(0)).unwrap_or(0)))
            .map_err(|_| {
            Error::Quant(format!(
                "QuantRotorV3::gpu_append: n_groups for head_dim={head_dim} exceeds i32::MAX"
            ))
        })?;
        if !self.gpu.is_allocated() && prev_seq > 0 {
            let (c, s, n) = self.flatten_blocks();
            self.gpu
                .seed_from_cpu(&c, &s, &n, kv_h, n_groups, prev_seq, max_seq, device)?;
        }
        self.gpu.append_encoded(
            codes, scales, norms, kv_h, n_groups, prev_seq, new_seq, max_seq, device,
        )
    }

    /// GPU packed view of the first `kv_seq` positions, or `None` when the ring
    /// is not live (CPU path — caller falls back to `dequant`).
    ///
    /// # Errors
    ///
    /// Forwards [`QuantKGpuRing::packed_view`] errors.
    pub fn gpu_packed_view(
        &self,
        kv_seq: i32,
        device: Device,
    ) -> Result<Option<(Array, Array, Array)>> {
        self.gpu.packed_view(kv_seq, device)
    }

    /// Resident bytes held by this store.
    ///
    /// Counts the rotor table **once** (it is layer-static) plus all
    /// accumulated per-token block buffers and the GPU ring when live.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            rotors,
            gpu,
            blocks,
            // Geometry / tags, not allocations.
            shape: _,
            max_seq: _,
            layer_idx: _,
            head_idx: _,
            bits: _,
        } = self;
        crate::bytes::vec_bytes(rotors)
            + blocks.iter().map(RotorBlocks::byte_size).sum::<u64>()
            + gpu.byte_size()
    }

    /// Dequantize all accumulated V slices into one flat f32 vector of length
    /// `prod(shape)`.
    ///
    /// # Errors
    ///
    /// Returns an `Error::Mlx` if the underlying [`rotor3_decode`] fails for any
    /// block, if `rotors` is empty (no append happened yet), or a
    /// [`synced_rotor_v_blocks`] reconciliation error.
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantRotorV3::dequant: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        let total_elems: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out: Vec<f32> = Vec::with_capacity(total_elems);

        // Reconcile the CPU blocks with the GPU ring: on the fused symmetric
        // decode path the decode tail lives only in the ring (`blocks` trail
        // `shape[2]`), and this rebuilds it on demand rather than decoding a
        // short prefix and zero-padding the gap. Loud on any unrecoverable
        // disagreement.
        let blocks = synced_rotor_v_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?;

        if blocks.is_empty() {
            // No tokens written yet — return zeros padded to the declared shape.
            out.resize(total_elems, 0.0);
            return Ok(out);
        }

        if self.rotors.is_empty() {
            return Err(Error::Mlx(
                "QuantRotorV3::dequant: rotor table is empty but blocks were appended".into(),
            ));
        }

        for blk in blocks.iter() {
            let dec = rotor3_decode(&blk.codes, &blk.scales, &blk.norms, &self.rotors, head_dim)
                .map_err(|e: RotorQuantError| Error::Mlx(format!("rotor3 decode: {e}")))?;
            out.extend_from_slice(&dec);
        }
        // `synced_rotor_v_blocks` guarantees the blocks cover `shape[2]`, so a
        // length mismatch here is an internal invariant break — surface it loudly
        // rather than zero-padding or truncating a decoded prefix.
        if out.len() != total_elems {
            return Err(Error::Mlx(format!(
                "QuantRotorV3::dequant: decoded {} elems but shape {:?} implies {total_elems} — \
                 refusing to zero-pad / truncate",
                out.len(),
                self.shape
            )));
        }
        // Blocks are sequence-major (see `append`); reorder back to head-major
        // `[B, kv_h, S, D]`.
        let b = self.shape[0] as usize;
        let kv_h = self.shape[1] as usize;
        let s = self.shape[2] as usize;
        let out = super::seq_layout::transpose_seq_heads(&out, b, s, kv_h, head_dim);
        Ok(out)
    }
}

/// Reconcile a rotor-V store's CPU `blocks` with its GPU ring so the returned
/// slice covers the full accumulated `shape[2]`.
///
/// On the fused symmetric decode path the per-step CPU block download is skipped
/// — the GPU ring is the source of truth for the decode tail, and `blocks` trail
/// `shape[2]` (a **ring-only tail**). This rebuilds the missing prefix from the
/// ring on demand — the single point where a block consumer (`dequant`, or the
/// SSD spill via `try_deep_clone`) reconciles the two. When `blocks` already
/// cover `shape[2]` (the CPU append and SSD-hydrate paths, and every V-only
/// rotor cache, which never feeds the ring) the borrow is returned untouched, so
/// those paths pay no GPU readback.
///
/// **Invariant (enforced loudly, never zero-padded):** `blocks` track the ring
/// exactly, or the ring exists and supplies the tail. Any state where the CPU
/// blocks fall short of `shape[2]` and the ring cannot make up the difference is
/// an `Error` — the caller must not fabricate a zeroed gap.
///
/// Shared by [`QuantRotorV3`] and [`super::QuantRotorV4`] (the block payload and
/// ring layout are identical modulo codes bit-width). Mirror of the K-side
/// `synced_rotor_k_blocks`.
///
/// # Errors
///
/// Returns [`rmlx_core::error::Error::Quant`] on a malformed shape, when the
/// blocks over-run `shape[2]`, or when a ring-only tail exists but the ring is
/// absent / too short to cover it.
pub(crate) fn synced_rotor_v_blocks<'a>(
    blocks: &'a [RotorBlocks],
    shape: &[i32],
    gpu: &QuantKGpuRing,
    device: Device,
) -> Result<std::borrow::Cow<'a, [RotorBlocks]>> {
    if shape.len() != 4 {
        return Err(Error::Quant(format!(
            "synced_rotor_v_blocks: malformed shape {shape:?}"
        )));
    }
    let b = shape.first().copied().unwrap_or(0).max(0) as usize;
    let kv_h = shape.get(1).copied().unwrap_or(0).max(0) as usize;
    let full_seq = shape.get(2).copied().unwrap_or(0).max(0) as usize;
    let head_dim = shape.get(3).copied().unwrap_or(0).max(0) as usize;
    let full_tokens = b * kv_h * full_seq;
    let blocks_tokens: usize = blocks.iter().map(|blk| blk.n_tokens).sum();

    if blocks_tokens == full_tokens {
        return Ok(std::borrow::Cow::Borrowed(blocks));
    }
    if blocks_tokens > full_tokens {
        return Err(Error::Quant(format!(
            "rotor V store: CPU blocks hold {blocks_tokens} tokens but shape[2] implies \
             {full_tokens} — blocks over-run the accumulated shape (internal invariant)"
        )));
    }

    // Ring-only tail: the GPU ring must supply the whole prefix. It is
    // sequence-major and stores per-token norms already, so the readback is one
    // block covering `[0, full_seq)`. Refuse to fabricate a zeroed gap.
    let seq_i32 = i32::try_from(full_seq).map_err(|_| {
        Error::Quant(format!(
            "rotor V store: shape[2]={full_seq} exceeds i32::MAX"
        ))
    })?;
    let Some((codes, scales, norms)) = gpu.packed_view_cpu(seq_i32, device)? else {
        return Err(Error::Quant(format!(
            "rotor V store: CPU blocks cover {blocks_tokens} tokens but shape[2] needs \
             {full_tokens} and the GPU ring is absent — refusing to zero-pad the decode tail"
        )));
    };
    let n_groups = n_groups_for(head_dim);
    let want_codes = full_tokens * n_groups;
    if codes.len() != want_codes || scales.len() != want_codes || norms.len() != full_tokens {
        return Err(Error::Quant(format!(
            "rotor V store: ring readback size mismatch (codes {} scales {} norms {}, \
             want codes/scales {want_codes} norms {full_tokens}) — cannot rebuild blocks",
            codes.len(),
            scales.len(),
            norms.len(),
        )));
    }
    Ok(std::borrow::Cow::Owned(vec![RotorBlocks {
        codes,
        scales,
        norms,
        n_tokens: full_tokens,
    }]))
}

#[cfg(test)]
#[path = "quant_rotor_v3_tests.rs"]
mod quant_rotor_v3_tests;
