// Rotor4 (Cl(3,0) Clifford rotor sandwich) 4-bit V storage.
//
// Mirrors `quant_rotor_v3.rs` (`QuantRotorV3`) one-for-one with `bits=4` and
// the dense 8-vals-per-u32 pack from the parameterized `rotor4_encode` /
// `rotor4_decode` codec.
//
// Step decision (parameterize vs fork): the encode/decode functions in
// `crate::rotorquant` are parameterized over bits ∈ {3,4}; the storage layer
// is forked because `QuantRotorV3` is name-stable across crates (SSD writer/
// reader, dispatch). Renaming to a generic `QuantRotorV` would cause large
// cross-crate churn for no benefit — bits is fixed per storage variant.
// Mirrors the iso3→iso4 fork (`quant_iso_v.rs` → `quant_iso_v4.rs`).
#![allow(
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::indexing_slicing,
    clippy::doc_lazy_continuation
)]
//! Quantized V buffer: `QuantRotorV4` (rotor4 V codec).

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device};

use crate::clifford::make_rotor_table;
use crate::rotorquant::{
    n_groups_for, rotor4_decode, rotor4_encode, RotorQuantError, ROTOR4_BITS, ROTOR4_GROUP_SIZE,
};
use crate::storage::quant_rotor_v3::RotorBlocks;

use super::RotorGpuK;

/// Bit-width of the rotor4 V codec (fixed at 4-bit — see
/// [`crate::rotorquant::rotor4_encode`]).
pub const ROTOR4_V_BITS: u8 = ROTOR4_BITS;

/// Multivector group size (3 grade-1 elements per group; one rotor) —
/// identical to rotor3.
pub const ROTOR4_V_GROUP_SIZE: usize = ROTOR4_GROUP_SIZE;

/// Accumulated rotor4 V cache.
///
/// Holds:
///   * `rotors` — static `[n_groups, 4]` table generated once at first append
///     (or supplied by SSD hydrate). Reseeded via
///     [`crate::clifford::make_rotor_table`] using (`layer_idx`, `head_idx`).
///   * `blocks` — per-append payload ([`RotorBlocks`]).
///   * `shape` — accumulated `[B, kv_h, S_total, D]`.
///
/// Carries both forms of the payload — CPU `blocks` (source of truth for
/// `dequant()` and the SSD round-trip) plus the optional GPU-resident packed
/// ring `gpu`. Mirror of [`super::QuantRotorV3`]; see its docs.
pub struct QuantRotorV4 {
    /// Static rotor table for this layer/head, flat `[n_groups * 4]` f32 in
    /// `[s, b12, b13, b23]` per-rotor order. Initialised lazily on first
    /// `append`; never replaced.
    pub rotors: Vec<f32>,
    /// GPU-resident packed ring. Empty until the first `gpu_append`.
    pub gpu: RotorGpuK,
    /// Accumulated per-token blocks (one entry per append call; `dequant`
    /// flattens them). Reuses [`RotorBlocks`] — the struct is bits-agnostic.
    pub blocks: Vec<RotorBlocks>,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Maximum sequence length the storage was provisioned for.
    pub max_seq: i32,
    /// Layer index (0-based) — used to seed the rotor table.
    pub layer_idx: u32,
    /// Head index (0-based) — used to seed the rotor table.
    pub head_idx: u32,
    /// Bit-width tag (always [`ROTOR4_V_BITS`] for this codec).
    pub bits: u8,
}

impl std::fmt::Debug for QuantRotorV4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantRotorV4")
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

impl QuantRotorV4 {
    /// Construct an empty `QuantRotorV4` for `init_shape = [B, kv_h, 0, D]`.
    ///
    /// `rotors` is left empty — it is generated lazily on the first `append`
    /// call once `head_dim = init_shape[3]` is known. Use
    /// [`Self::with_rotors`] (or [`Self::from_cpu_blocks`]) if a pre-computed
    /// rotor table is on hand (SSD hydrate path).
    #[must_use]
    pub fn new(init_shape: Vec<i32>, max_seq: i32, layer_idx: u32) -> Self {
        Self {
            rotors: Vec::new(),
            gpu: RotorGpuK::default(),
            blocks: Vec::new(),
            shape: init_shape,
            max_seq,
            layer_idx,
            head_idx: 0,
            bits: ROTOR4_V_BITS,
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
            gpu: RotorGpuK::default(),
            blocks: Vec::new(),
            shape: init_shape,
            max_seq,
            layer_idx,
            head_idx: 0,
            bits: ROTOR4_V_BITS,
        }
    }

    /// Build a `QuantRotorV4` from pre-computed CPU blocks (SSD hydrate path).
    #[must_use]
    pub fn from_cpu_blocks(
        rotors: Vec<f32>,
        blocks: Vec<RotorBlocks>,
        shape: Vec<i32>,
        layer_idx: u32,
    ) -> Self {
        debug_assert!(
            shape.len() == 4,
            "QuantRotorV4::from_cpu_blocks expects a 4-element [B, kv_h, S, D] shape, got {shape:?}"
        );
        let max_seq = if shape.len() >= 3 { shape[2] } else { 0 };
        Self {
            rotors,
            // Hydrated caches start CPU-only; the ring is rebuilt lazily from
            // the next GPU append.
            gpu: RotorGpuK::default(),
            blocks,
            shape,
            max_seq,
            layer_idx,
            head_idx: 0,
            bits: ROTOR4_V_BITS,
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
    /// Forwards any [`RotorQuantError`] from [`rotor4_encode`].
    pub fn append(&mut self, f32_data: &[f32], new_shape: &[i32]) -> Result<()> {
        if new_shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantRotorV4::append: expected 4D new_shape, got {new_shape:?}"
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

        // Store each chunk sequence-major so the per-append blocks share one
        // layout; the static rotor table is group-position-keyed (not token),
        // so the reorder leaves it correct. See [`super::QuantIsoV3::append`].
        let seq_major =
            super::seq_layout::transpose_heads_seq(f32_data, b, kv_h, new_seq, head_dim);

        let (codes, scales, norms) =
            rotor4_encode(&seq_major, &self.rotors, head_dim).map_err(|e: RotorQuantError| {
                rmlx_core::error::Error::Mlx(format!("rotor4 encode: {e}"))
            })?;

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
    /// Drops trailing blocks until the cumulative `n_tokens` count is `<= n`
    /// and lowers `shape[2]` to `n`. Does **not** touch the rotor table.
    pub fn truncate_to(&mut self, n: i32) {
        let n_usize = n.max(0) as usize;
        let mut acc: usize = 0;
        let mut keep = 0usize;
        for (i, blk) in self.blocks.iter().enumerate() {
            if acc + blk.n_tokens <= n_usize {
                acc += blk.n_tokens;
                keep = i + 1;
            } else {
                break;
            }
        }
        self.blocks.truncate(keep);
        // The ring's filled prefix no longer matches `blocks`; drop it rather
        // than leave a longer-than-truncated prefix live.
        self.gpu.clear();
        if self.shape.len() >= 4 {
            self.shape[2] = n;
        }
    }

    /// Deep-clone (CPU path is plain `Vec` clones).
    ///
    /// # Errors
    ///
    /// Currently infallible on the CPU path; returns `Result` for parity with
    /// the other `Quant*` structs.
    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(Self {
            rotors: self.rotors.clone(),
            // The clone starts CPU-only: `blocks` carries the full payload, so
            // the ring re-seeds from them on the clone's first GPU append.
            // Sharing the source's Arrays would alias one ring across two
            // independent caches.
            gpu: RotorGpuK::default(),
            blocks: self.blocks.clone(),
            shape: self.shape.clone(),
            max_seq: self.max_seq,
            layer_idx: self.layer_idx,
            head_idx: self.head_idx,
            bits: self.bits,
        })
    }

    /// Concatenate the accumulated CPU blocks into flat sequence-major
    /// `(codes, scales, norms)` — the form [`RotorGpuK::seed_from_cpu`] wants.
    fn flatten_blocks(&self) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
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

    /// Push one GPU-encoded chunk into the GPU ring, seeding it from the
    /// accumulated CPU blocks first when it is not yet live. Mirror of
    /// [`super::QuantRotorV3::gpu_append`].
    ///
    /// # Errors
    ///
    /// Forwards [`RotorGpuK::seed_from_cpu`] / [`RotorGpuK::append_encoded`]
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
        device: Device,
    ) -> Result<()> {
        let max_seq = self.max_seq;
        if !self.gpu.is_allocated() && prev_seq > 0 {
            let (c, s, n) = self.flatten_blocks();
            self.gpu
                .seed_from_cpu(&c, &s, &n, kv_h, head_dim, prev_seq, max_seq, device)?;
        }
        self.gpu.append_encoded(
            codes, scales, norms, kv_h, head_dim, prev_seq, new_seq, max_seq, device,
        )
    }

    /// GPU packed view of the first `kv_seq` positions, or `None` when the ring
    /// is not live (CPU path — caller falls back to `dequant`).
    ///
    /// # Errors
    ///
    /// Forwards [`RotorGpuK::packed_view`] errors.
    pub fn gpu_packed_view(
        &self,
        kv_seq: i32,
        device: Device,
    ) -> Result<Option<(Array, Array, Array)>> {
        self.gpu.packed_view(kv_seq, device)
    }

    /// Approximate byte footprint of the accumulated payload.
    ///
    /// Counts the rotor table **once** (it is layer-static) plus all
    /// accumulated per-token block buffers and the GPU ring when live.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        let mut total = self.rotors.len() * size_of::<f32>();
        for blk in &self.blocks {
            total += blk.codes.len() * size_of::<u32>();
            total += blk.scales.len() * size_of::<f32>();
            total += blk.norms.len() * size_of::<f32>();
        }
        total + self.gpu.byte_size()
    }

    /// Dequantize all accumulated V slices into one flat f32 vector of length
    /// `prod(shape)`.
    ///
    /// # Errors
    ///
    /// Returns an `Error::Mlx` if the underlying [`rotor4_decode`] fails for
    /// any block, or if `rotors` is empty (no append happened yet).
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantRotorV4::dequant: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        let total_elems: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out: Vec<f32> = Vec::with_capacity(total_elems);

        if self.blocks.is_empty() {
            out.resize(total_elems, 0.0);
            return Ok(out);
        }

        if self.rotors.is_empty() {
            return Err(rmlx_core::error::Error::Mlx(
                "QuantRotorV4::dequant: rotor table is empty but blocks were appended".into(),
            ));
        }

        for blk in &self.blocks {
            let dec = rotor4_decode(&blk.codes, &blk.scales, &blk.norms, &self.rotors, head_dim)
                .map_err(|e: RotorQuantError| {
                    rmlx_core::error::Error::Mlx(format!("rotor4 decode: {e}"))
                })?;
            out.extend_from_slice(&dec);
        }
        if out.len() < total_elems {
            out.resize(total_elems, 0.0);
        } else if out.len() > total_elems {
            out.truncate(total_elems);
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
#[path = "quant_rotor_v4_tests.rs"]
mod quant_rotor_v4_tests;
