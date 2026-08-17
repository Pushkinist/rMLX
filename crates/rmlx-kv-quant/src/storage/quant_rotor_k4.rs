// Rotor4 K-side storage (Cl(3,0) Clifford rotor sandwich,
// 4-bit Lloyd-Max codebook, optional 1-bit QJL residual).
//
// Mirror of `quant_rotor_k3.rs` with `bits=4` and the dense 8-vals-per-u32
// pack from `rotor4_k_encode` / `rotor4_k_decode`. Identical storage layout
// modulo the codes bit-width — `RotorKBlocks` is shared from `quant_rotor_k3`.
#![allow(
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::doc_lazy_continuation
)]
//! Quantized K buffer: `QuantRotorK4` (rotor4 K codec).

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::clifford::make_rotor_table;
use crate::rotorquant::{
    make_qjl_projection, n_groups_for, rotor4_k_decode, rotor4_k_encode, RotorQuantError,
    ROTOR4_BITS, ROTOR4_GROUP_SIZE,
};
use crate::storage::quant_rotor_k3::{synced_rotor_k_blocks, RotorKBlocks};

use super::QuantKGpuRing;

/// Bit-width of the rotor4 K codec.
pub const ROTOR4_K_BITS: u8 = ROTOR4_BITS;

/// Multivector group size (identical to rotor3 / rotor4 V-side codecs).
pub const ROTOR4_K_GROUP_SIZE: usize = ROTOR4_GROUP_SIZE;

/// Accumulated rotor4 K cache. See [`QuantRotorK3`](super::QuantRotorK3) for the
/// field semantics — same structure, 4-bit codes via the rotor4 codec.
pub struct QuantRotorK4 {
    /// Static rotor table for this layer/head.
    pub rotors: Vec<f32>,
    /// GPU-resident packed ring. Empty until the first `gpu_append`.
    pub gpu: QuantKGpuRing,
    /// Static QJL projection matrix (None when QJL is disabled at first append).
    pub qjl_s_matrix: Option<Vec<f32>>,
    /// Accumulated per-token blocks.
    pub blocks: Vec<RotorKBlocks>,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Layer index used to seed the rotor table.
    pub layer_idx: u32,
    /// Head index (currently always 0).
    pub head_idx: u32,
    /// Bit-width tag (always [`ROTOR4_K_BITS`]).
    pub bits: u8,
}

impl std::fmt::Debug for QuantRotorK4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantRotorK4")
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

impl QuantRotorK4 {
    /// Construct an empty `QuantRotorK4`.
    ///
    /// The provisioned window is not stored here — see [`QuantRotorK3::new`].
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
            bits: ROTOR4_K_BITS,
        }
    }

    /// Build a `QuantRotorK4` from pre-computed CPU blocks (SSD hydrate path).
    /// The provisioned window is not taken here — see
    /// [`QuantRotorK3::from_cpu_blocks`].
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
            "QuantRotorK4::from_cpu_blocks expects a 4-element [B, kv_h, S, D] shape, got {shape:?}"
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
            bits: ROTOR4_K_BITS,
        }
    }

    /// Append one K slice. See [`QuantRotorK3::append`] for the rotor-table /
    /// QJL-projection lazy-init semantics.
    ///
    /// # Errors
    /// Forwards any [`RotorQuantError`] from [`rotor4_k_encode`].
    #[allow(
        clippy::indexing_slicing,
        reason = "shape rank verified above (new_shape.len() != 4 guard) and by append caller contract [B, H, S, D]"
    )]
    pub fn append(&mut self, f32_data: &[f32], new_shape: &[i32]) -> Result<()> {
        if new_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantRotorK4::append: expected 4D new_shape, got {new_shape:?}"
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

        // Store each chunk sequence-major so the per-append blocks share one
        // layout; static rotor/QJL projection are group/projection-keyed and
        // the per-token QJL sideband reorders with the token rows. See
        // [`super::QuantIsoV3::append`].
        let seq_major =
            super::seq_layout::transpose_heads_seq(f32_data, b, kv_h, new_seq, head_dim);

        let (codes, scales, norms, qjl_codes, qjl_norms) = rotor4_k_encode(
            &seq_major,
            &self.rotors,
            head_dim,
            self.qjl_s_matrix.as_deref(),
        )
        .map_err(|e: RotorQuantError| Error::Mlx(format!("rotor4_k encode: {e}")))?;

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
    /// `(codes, scales, norms)`. See [`QuantRotorK3::flatten_blocks`].
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

    /// Push one GPU-encoded chunk into the GPU ring. Mirror of
    /// [`QuantRotorK3::gpu_append`].
    ///
    /// # Errors
    ///
    /// Forwards [`QuantKGpuRing::seed_from_cpu`] / [`QuantKGpuRing::append_encoded`]
    /// errors.
    ///
    /// `max_seq` is a parameter, not a field — see [`QuantRotorK3::gpu_append`].
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
                "QuantRotorK4::gpu_append: n_groups for head_dim={head_dim} exceeds i32::MAX"
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

    /// GPU packed view of the first `kv_seq` positions. Mirror of
    /// [`QuantRotorK3::gpu_packed_view`].
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

    /// Reset the accumulated sequence to zero.
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
    /// See [`QuantRotorK3::truncate_to`] — a mid-block cut splits the trailing
    /// block, and the GPU ring is kept (not cleared) so a ring-only decode tail
    /// up to `n` survives, matching the flat GPU-buffer codecs' truncate
    /// semantics.
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
        // NB: no `self.gpu.clear()` — the ring holds the ring-only decode tail.
        if self.shape.len() >= 4 {
            self.shape[2] = n;
        }
    }

    /// Deep-clone.
    ///
    /// # Errors
    /// Infallible on the CPU path; returns `Result` for parity.
    pub fn try_deep_clone(&self) -> Result<Self> {
        // Materialise any ring-only tail into complete CPU blocks first — see
        // [`QuantRotorK3::try_deep_clone`] for the full rationale (this is the
        // single reconcile point for the prompt-cache and SSD spill clones).
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

    /// True when the QJL sideband is active.
    #[must_use]
    pub fn use_qjl(&self) -> bool {
        self.qjl_s_matrix.is_some()
    }

    /// Dequantize all accumulated K slices.
    ///
    /// # Errors
    /// Returns an `Error::Mlx` if [`rotor4_k_decode`] fails for any block.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape.len() != 4 early-return guard above ensures shape[3] is in-bounds"
    )]
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantRotorK4::dequant: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        let total_elems: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out: Vec<f32> = Vec::with_capacity(total_elems);

        // Reconcile CPU blocks with the GPU ring (ring-only decode tail rebuild;
        // loud on an unrecoverable gap). See [`QuantRotorK3::dequant`].
        let blocks = synced_rotor_k_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?;

        if blocks.is_empty() {
            out.resize(total_elems, 0.0);
            return Ok(out);
        }

        if self.rotors.is_empty() {
            return Err(Error::Mlx(
                "QuantRotorK4::dequant: rotor table is empty but blocks were appended".into(),
            ));
        }

        for blk in blocks.iter() {
            let dec = rotor4_k_decode(
                &blk.codes,
                &blk.scales,
                &blk.norms,
                &self.rotors,
                head_dim,
                &blk.qjl_codes,
                &blk.qjl_norms,
                self.qjl_s_matrix.as_deref(),
            )
            .map_err(|e: RotorQuantError| Error::Mlx(format!("rotor4_k decode: {e}")))?;
            out.extend_from_slice(&dec);
        }
        // `synced_rotor_k_blocks` guarantees full coverage — a mismatch is an
        // internal invariant break, surfaced loudly rather than zero-padded.
        if out.len() != total_elems {
            return Err(Error::Mlx(format!(
                "QuantRotorK4::dequant: decoded {} elems but shape {:?} implies {total_elems} — \
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

#[cfg(test)]
#[path = "quant_rotor_k4_tests.rs"]
mod quant_rotor_k4_tests;
