// Rotor3 K-side storage (Cl(3,0) Clifford rotor sandwich,
// 3-bit Lloyd-Max codebook, optional 1-bit QJL residual).
//
// Mirror of `quant_rotor_v3.rs` (`QuantRotorV3`) on the K axis. The codec
// itself is axis-agnostic — the same per-(layer, head) static rotor table and
// per-token (codes, scales, norms) tuple format are reused. The K-side fork
// adds the optional QJL sideband (`qjl_codes` packed 1-bit signs + per-token
// `qjl_norms`) when `crate::rotor_qjl::rotor_qjl_enabled()` is true at first
// append.
#![allow(
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::doc_lazy_continuation
)]
//! Quantized K buffer: `QuantRotorK3` (rotor3 K codec).

use std::borrow::Cow;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::clifford::make_rotor_table;
use crate::rotorquant::{
    make_qjl_projection, n_groups_for, rotor3_k_decode, rotor3_k_encode, RotorQuantError,
    ROTOR3_BITS, ROTOR3_GROUP_SIZE,
};

use super::QuantKGpuRing;

/// Bit-width of the rotor3 K codec.
pub const ROTOR3_K_BITS: u8 = ROTOR3_BITS;

/// Multivector group size (identical to the V-side rotor3 codec).
pub const ROTOR3_K_GROUP_SIZE: usize = ROTOR3_GROUP_SIZE;

/// One token-batch's rotor3-K payload: codes + per-group scales + per-token
/// L2 norm + optional packed QJL signs + optional per-token residual L2 norm.
///
/// Same shape conventions as `RotorBlocks` plus two QJL fields. When QJL is
/// disabled, `qjl_codes` and `qjl_norms` are empty `Vec`s.
#[derive(Debug, Clone)]
pub struct RotorKBlocks {
    /// Packed 3-bit codes; pack convention = 10 vals/u32 (planar3 / iso3).
    pub codes: Vec<u32>,
    /// Per-group scale: `n_tokens * n_groups` f32 entries (flat).
    pub scales: Vec<f32>,
    /// Per-token L2 norm of the (pre-rotation) input vector.
    pub norms: Vec<f32>,
    /// Packed 1-bit QJL signs per token (LSB = element 0). Empty when QJL
    /// is disabled. Length per token = `ceil(head_dim / 8)`.
    pub qjl_codes: Vec<u8>,
    /// Per-token residual L2 norm (post-rotor-MSE-recon). Empty when QJL is
    /// disabled. Length = `n_tokens`.
    pub qjl_norms: Vec<f32>,
    /// Number of tokens this block represents.
    pub n_tokens: usize,
}

impl RotorKBlocks {
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
            qjl_codes,
            qjl_norms,
            // Inline metadata (no heap payload).
            n_tokens: _,
        } = self;
        crate::bytes::vec_bytes(codes)
            + crate::bytes::vec_bytes(scales)
            + crate::bytes::vec_bytes(norms)
            + crate::bytes::vec_bytes(qjl_codes)
            + crate::bytes::vec_bytes(qjl_norms)
    }
}

impl super::BlockRows for RotorKBlocks {
    fn rows(&self) -> usize {
        self.n_tokens
    }

    /// The exhaustive destructure is the drift guard: a new payload field
    /// cannot be added without this failing to compile, which is what stops a
    /// sideband from surviving a mid-block truncation at its full length.
    fn retain_rows(&mut self, rows: usize) -> bool {
        let Self {
            codes,
            scales,
            norms,
            qjl_codes,
            qjl_norms,
            n_tokens,
        } = self;
        let lengths = [
            codes.len(),
            scales.len(),
            norms.len(),
            qjl_codes.len(),
            qjl_norms.len(),
        ];
        if !super::rows_split_ok(&lengths, *n_tokens, rows) {
            return false;
        }
        super::retain_rows_in(codes, *n_tokens, rows);
        super::retain_rows_in(scales, *n_tokens, rows);
        super::retain_rows_in(norms, *n_tokens, rows);
        super::retain_rows_in(qjl_codes, *n_tokens, rows);
        super::retain_rows_in(qjl_norms, *n_tokens, rows);
        *n_tokens = rows;
        true
    }
}

/// Accumulated rotor3 K cache.
///
/// Holds:
///   * `rotors` — static `[n_groups, 4]` rotor table generated once on first
///     `append` (or supplied by SSD hydrate).
///   * `qjl_s_matrix` — static `[head_dim, head_dim]` JL projection matrix
///     generated once on first `append` (or hydrated from SSD). `None` when
///     QJL is disabled.
///   * `blocks` — per-append payload (`RotorKBlocks`).
///   * `shape` — accumulated `[B, kv_h, S_total, D]`.
///
/// Carries both forms of the payload:
///
/// * `blocks` — CPU `RotorKBlocks`, the source of truth for `dequant()`, the
///   SSD spill/hydrate round-trip, and the QJL path.
/// * `gpu` — the optional GPU-resident packed ring ([`QuantKGpuRing`]), populated by
///   `gpu_append` on the QJL-off GPU encode path. When present it lets the
///   rotor flash-decode kernel read the quant store directly instead of paying a
///   full-prefix CPU `dequant()` per decode step.
pub struct QuantRotorK3 {
    /// Static rotor table for this layer/head, flat `[n_groups * 4]` f32.
    pub rotors: Vec<f32>,
    /// GPU-resident packed ring. Empty until the first `gpu_append`.
    pub gpu: QuantKGpuRing,
    /// Static QJL projection matrix, flat `[qjl_dim * head_dim]` f32. `None`
    /// when QJL is disabled — chosen at first `append` time from the global
    /// [`crate::rotor_qjl::rotor_qjl_enabled`] toggle.
    pub qjl_s_matrix: Option<Vec<f32>>,
    /// Accumulated per-token blocks (one entry per append call).
    pub blocks: Vec<RotorKBlocks>,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Layer index (0-based) — used to seed the rotor table.
    pub layer_idx: u32,
    /// Head index (0-based). Currently always `0` (one rotor table per layer).
    pub head_idx: u32,
    /// Bit-width tag (always [`ROTOR3_K_BITS`]).
    pub bits: u8,
}

impl std::fmt::Debug for QuantRotorK3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantRotorK3")
            .field("n_rotors", &(self.rotors.len() / 4))
            .field("use_qjl", &self.qjl_s_matrix.is_some())
            .field("gpu_resident", &self.gpu.is_allocated())
            .field("n_blocks", &self.blocks.len())
            .field("shape", &self.shape)
            .field("layer_idx", &self.layer_idx)
            .field("head_idx", &self.head_idx)
            .field("bits", &self.bits)
            .finish()
    }
}

impl QuantRotorK3 {
    /// Construct an empty `QuantRotorK3` for `init_shape = [B, kv_h, 0, D]`.
    ///
    /// Both `rotors` and `qjl_s_matrix` are left empty — they are populated
    /// lazily on the first `append` call once `head_dim` is known.
    ///
    /// The provisioned window is deliberately **not** stored here: it lives on
    /// the `KvStorage` variant, grows as the sequence does, and is passed to
    /// [`Self::gpu_append`] per call. A copy cached on the store would be a
    /// snapshot that silently goes stale the moment the window grows.
    #[must_use]
    pub fn new(init_shape: Vec<i32>, layer_idx: u32) -> Self {
        Self {
            rotors: Vec::new(),
            gpu: QuantKGpuRing::default(),
            qjl_s_matrix: None,
            blocks: Vec::new(),
            shape: init_shape,
            layer_idx,
            head_idx: 0,
            bits: ROTOR3_K_BITS,
        }
    }

    /// Build a `QuantRotorK3` from pre-computed CPU blocks (SSD hydrate path).
    ///
    /// The provisioned window is not taken here — see [`Self::new`]. A hydrated
    /// store therefore cannot disagree with the window the cache is currently
    /// sized to, however long it sat on disk.
    ///
    /// When `qjl_s_matrix` is `Some(_)` the QJL sideband was active at write
    /// time; the reader hydrates accordingly.
    #[must_use]
    pub fn from_cpu_blocks(
        rotors: Vec<f32>,
        qjl_s_matrix: Option<Vec<f32>>,
        blocks: Vec<RotorKBlocks>,
        shape: Vec<i32>,
        layer_idx: u32,
    ) -> Self {
        debug_assert!(
            shape.len() == 4,
            "QuantRotorK3::from_cpu_blocks expects a 4-element [B, kv_h, S, D] shape, got {shape:?}"
        );
        Self {
            rotors,
            // Hydrated caches start CPU-only; the ring is rebuilt lazily from
            // the next GPU append.
            gpu: QuantKGpuRing::default(),
            qjl_s_matrix,
            blocks,
            shape,
            layer_idx,
            head_idx: 0,
            bits: ROTOR3_K_BITS,
        }
    }

    /// Append one K slice (CPU path).
    ///
    /// On the first call:
    ///   * The rotor table is generated via [`make_rotor_table`].
    ///   * The QJL projection matrix is generated via [`make_qjl_projection`]
    ///     **iff** [`crate::rotor_qjl::rotor_qjl_enabled`] returns `true`.
    ///
    /// Both decisions are sticky for the lifetime of the cache: a second call
    /// with a flipped global toggle does **not** add/remove the QJL sideband
    /// mid-stream.
    ///
    /// # Errors
    /// Forwards any [`RotorQuantError`] from [`rotor3_k_encode`].
    #[allow(
        clippy::indexing_slicing,
        reason = "shape rank verified above (new_shape.len() != 4 guard) and by append caller contract [B, H, S, D]"
    )]
    pub fn append(&mut self, f32_data: &[f32], new_shape: &[i32]) -> Result<()> {
        if new_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantRotorK3::append: expected 4D new_shape, got {new_shape:?}"
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
            if crate::rotor_qjl::rotor_qjl_enabled() {
                self.qjl_s_matrix = Some(make_qjl_projection(head_dim));
            }
        }

        // Store each chunk sequence-major (`[B, new_seq, kv_h, D]`) so the
        // per-append blocks share one layout; a head-major store transposes
        // heads across multi-append GQA caches (kv_h>1). The static rotor table
        // and QJL projection are group/projection-keyed (not token), so the
        // reorder leaves them correct; the per-token QJL sideband (qjl_codes /
        // qjl_norms) reorders with the token rows. See
        // [`super::QuantIsoV3::append`].
        let seq_major =
            super::seq_layout::transpose_heads_seq(f32_data, b, kv_h, new_seq, head_dim);

        let (codes, scales, norms, qjl_codes, qjl_norms) = rotor3_k_encode(
            &seq_major,
            &self.rotors,
            head_dim,
            self.qjl_s_matrix.as_deref(),
        )
        .map_err(|e: RotorQuantError| Error::Mlx(format!("rotor3_k encode: {e}")))?;

        self.blocks.push(RotorKBlocks {
            codes,
            scales,
            norms,
            qjl_codes,
            qjl_norms,
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

    /// Concatenate the accumulated CPU blocks into flat sequence-major
    /// `(codes, scales, norms)` — the form [`QuantKGpuRing::seed_from_cpu`] wants.
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
    /// rather than a field so it cannot go stale as the window grows during
    /// decode — the same contract `QuantK::append` / `QuantPlanarK::append` use.
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
                "QuantRotorK3::gpu_append: n_groups for head_dim={head_dim} exceeds i32::MAX"
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

    /// Reset the accumulated sequence to zero. Buffers are kept for reuse.
    /// Does **not** touch the rotor table or QJL projection.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape.len() >= 4 checked immediately before indexing shape[2]"
    )]
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.gpu.clear();
        if self.shape.len() >= 4 {
            self.shape[2] = 0;
        }
    }

    /// Truncate the accumulated sequence to `n` positions.
    ///
    /// A cut that lands inside a block **splits** that block (see
    /// [`super::truncate_plan`]) rather than dropping it. A speculative-decode
    /// partial accept always cuts mid-block — the verifier appends its whole
    /// multi-token chunk as one block and then keeps only the accepted prefix —
    /// so dropping the block would discard the accepted tokens along with the
    /// rejected ones and leave `blocks` short of `shape[2]`. On the CPU append
    /// path (a QJL-carrying store, or a CPU-device run) no GPU ring exists to
    /// rebuild that gap, and the next `dequant` / `try_deep_clone` aborts the
    /// request rather than fabricate a zeroed prefix.
    ///
    /// The GPU ring is **kept**, not cleared: this mirrors how the flat GPU-buffer
    /// codecs truncate (`QuantK` / K8V4 etc. just lower `shape[2]` and overwrite
    /// on the next append). Lowering `shape[2]` to `n` makes the ring's logical
    /// fill `n`; the stale `[n, prev)` capacity is overwritten by the next append
    /// and never read (`packed_view` always slices to `shape[2]`).
    #[allow(
        clippy::indexing_slicing,
        reason = "shape.len() >= 4 checked immediately before indexing shape[2]"
    )]
    pub fn truncate_to(&mut self, n: i32) {
        let n = n.max(0);
        let plan = super::truncate_plan(
            self.blocks.iter().map(super::BlockRows::rows),
            &self.shape,
            n,
        );
        super::apply_truncate_plan(&mut self.blocks, &plan);
        // NB: no `self.gpu.clear()` — the ring is the source of truth for a
        // ring-only decode tail; see the doc comment above.
        if self.shape.len() >= 4 {
            self.shape[2] = n;
        }
    }

    /// Deep-clone (CPU path is plain `Vec` clones).
    ///
    /// # Errors
    /// Currently infallible on the CPU path; returns `Result` for parity.
    pub fn try_deep_clone(&self) -> Result<Self> {
        // Materialise any ring-only tail into complete CPU blocks first: the
        // clone starts CPU-only (the ring is not cloned), and both the
        // prompt-cache snapshot and the SSD spill clone route through here, so
        // this is the single point where a store leaving the live decode loop
        // reconciles its blocks with the ring. A short-blocks clone with no
        // ring would silently truncate the store — refused loudly instead.
        let blocks =
            synced_rotor_k_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?.into_owned();
        Ok(Self {
            rotors: self.rotors.clone(),
            // The clone starts CPU-only: `blocks` carries the full payload, so
            // the ring re-seeds from them on the clone's first GPU append.
            // Sharing the source's Arrays would alias one ring across two
            // independent caches.
            gpu: QuantKGpuRing::default(),
            qjl_s_matrix: self.qjl_s_matrix.clone(),
            blocks,
            shape: self.shape.clone(),
            layer_idx: self.layer_idx,
            head_idx: self.head_idx,
            bits: self.bits,
        })
    }

    /// Resident bytes held by this store: CPU blocks, the static rotor table
    /// and QJL projection, plus the GPU ring.
    ///
    /// The ring is real resident memory and is counted at its full allocation.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            rotors,
            gpu,
            qjl_s_matrix,
            blocks,
            // Geometry / tags, not allocations.
            shape: _,
            layer_idx: _,
            head_idx: _,
            bits: _,
        } = self;
        crate::bytes::vec_bytes(rotors)
            + crate::bytes::opt_vec_bytes(qjl_s_matrix.as_ref())
            + blocks.iter().map(RotorKBlocks::byte_size).sum::<u64>()
            + gpu.byte_size()
    }

    /// True when the QJL sideband is active on this cache.
    #[must_use]
    pub fn use_qjl(&self) -> bool {
        self.qjl_s_matrix.is_some()
    }

    /// Dequantize all accumulated K slices into one flat f32 vector of length
    /// `prod(shape)`.
    ///
    /// When the QJL sideband is present, the per-token correction is applied
    /// in-line by [`rotor3_k_decode`].
    ///
    /// # Errors
    /// Returns an `Error::Mlx` if [`rotor3_k_decode`] fails for any block.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape.len() != 4 early-return guard above ensures shape[3] is in-bounds"
    )]
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantRotorK3::dequant: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        let total_elems: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out: Vec<f32> = Vec::with_capacity(total_elems);

        // Reconcile the CPU blocks with the GPU ring: on the fused decode path
        // the decode tail lives only in the ring (`blocks` trail `shape[2]`),
        // and this rebuilds it on demand rather than decoding a short prefix and
        // zero-padding the gap. Loud on any unrecoverable disagreement.
        let blocks = synced_rotor_k_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?;

        if blocks.is_empty() {
            out.resize(total_elems, 0.0);
            return Ok(out);
        }

        if self.rotors.is_empty() {
            return Err(Error::Mlx(
                "QuantRotorK3::dequant: rotor table is empty but blocks were appended".into(),
            ));
        }

        for blk in blocks.iter() {
            let dec = rotor3_k_decode(
                &blk.codes,
                &blk.scales,
                &blk.norms,
                &self.rotors,
                head_dim,
                &blk.qjl_codes,
                &blk.qjl_norms,
                self.qjl_s_matrix.as_deref(),
            )
            .map_err(|e: RotorQuantError| Error::Mlx(format!("rotor3_k decode: {e}")))?;
            out.extend_from_slice(&dec);
        }
        // `synced_rotor_k_blocks` guarantees the blocks cover `shape[2]`, so a
        // length mismatch here is an internal invariant break — surface it
        // loudly rather than zero-padding or truncating a decoded prefix.
        if out.len() != total_elems {
            return Err(Error::Mlx(format!(
                "QuantRotorK3::dequant: decoded {} elems but shape {:?} implies {total_elems} — \
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

/// Reconcile a rotor-K store's CPU `blocks` with its GPU ring so the returned
/// slice covers the full accumulated `shape[2]`.
///
/// On the fused decode path the per-step CPU block download is skipped — the
/// GPU ring is the source of truth for the decode tail, and `blocks` trail
/// `shape[2]` (a **ring-only tail**). This rebuilds the missing prefix from the
/// ring on demand — the single point where a block consumer (`dequant`, or the
/// SSD spill via `try_deep_clone`) reconciles the two. When `blocks` already
/// cover `shape[2]` (the CPU append, QJL, and SSD-hydrate paths) the borrow is
/// returned untouched, so those paths pay no GPU readback.
///
/// **Invariant (enforced loudly, never zero-padded):** `blocks` track the ring
/// exactly, or the ring exists and supplies the tail. Any state where the CPU
/// blocks fall short of `shape[2]` and the ring cannot make up the difference is
/// an `Error` — the caller must not fabricate a zeroed gap.
///
/// Shared by `QuantRotorK3` and `QuantRotorK4` (the block payload and ring
/// layout are identical modulo codes bit-width).
///
/// # Errors
///
/// Returns [`Error::Quant`] on a malformed shape, when the blocks over-run
/// `shape[2]`, or when a ring-only tail exists but the ring is absent /
/// too short to cover it.
pub(crate) fn synced_rotor_k_blocks<'a>(
    blocks: &'a [RotorKBlocks],
    shape: &[i32],
    gpu: &QuantKGpuRing,
    device: Device,
) -> Result<Cow<'a, [RotorKBlocks]>> {
    if shape.len() != 4 {
        return Err(Error::Quant(format!(
            "synced_rotor_k_blocks: malformed shape {shape:?}"
        )));
    }
    let b = shape.first().copied().unwrap_or(0).max(0) as usize;
    let kv_h = shape.get(1).copied().unwrap_or(0).max(0) as usize;
    let full_seq = shape.get(2).copied().unwrap_or(0).max(0) as usize;
    let head_dim = shape.get(3).copied().unwrap_or(0).max(0) as usize;
    let full_tokens = b * kv_h * full_seq;
    let blocks_tokens: usize = blocks.iter().map(|blk| blk.n_tokens).sum();

    if blocks_tokens == full_tokens {
        return Ok(Cow::Borrowed(blocks));
    }
    if blocks_tokens > full_tokens {
        return Err(Error::Quant(format!(
            "rotor K store: CPU blocks hold {blocks_tokens} tokens but shape[2] implies \
             {full_tokens} — blocks over-run the accumulated shape (internal invariant)"
        )));
    }

    // Ring-only tail: the GPU ring must supply the whole prefix. It is
    // sequence-major and stores per-token norms already, so the readback is one
    // block covering `[0, full_seq)`. Refuse to fabricate a zeroed gap.
    let seq_i32 = i32::try_from(full_seq).map_err(|_| {
        Error::Quant(format!(
            "rotor K store: shape[2]={full_seq} exceeds i32::MAX"
        ))
    })?;
    let Some((codes, scales, norms)) = gpu.packed_view_cpu(seq_i32, device)? else {
        return Err(Error::Quant(format!(
            "rotor K store: CPU blocks cover {blocks_tokens} tokens but shape[2] needs \
             {full_tokens} and the GPU ring is absent — refusing to zero-pad the decode tail"
        )));
    };
    let n_groups = n_groups_for(head_dim);
    let want_codes = full_tokens * n_groups;
    if codes.len() != want_codes || scales.len() != want_codes || norms.len() != full_tokens {
        return Err(Error::Quant(format!(
            "rotor K store: ring readback size mismatch (codes {} scales {} norms {}, \
             want codes/scales {want_codes} norms {full_tokens}) — cannot rebuild blocks",
            codes.len(),
            scales.len(),
            norms.len(),
        )));
    }
    Ok(Cow::Owned(vec![RotorKBlocks {
        codes,
        scales,
        norms,
        // The GPU ring path is QJL-off by dispatcher contract, so the rebuilt
        // block carries no residual sideband.
        qjl_codes: Vec::new(),
        qjl_norms: Vec::new(),
        n_tokens: full_tokens,
    }]))
}

#[cfg(test)]
#[path = "quant_rotor_k3_tests.rs"]
mod quant_rotor_k3_tests;
