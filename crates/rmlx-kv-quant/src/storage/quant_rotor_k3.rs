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

use rmlx_core::error::Result;

use crate::clifford::make_rotor_table;
use crate::rotorquant::{
    make_qjl_projection, n_groups_for, rotor3_k_decode, rotor3_k_encode, RotorQuantError,
    ROTOR3_BITS, ROTOR3_GROUP_SIZE,
};

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
/// CPU-only. SDPA falls through the dequant-then-SDPA legacy path — no MSL
/// kernel yet for the K-side rotor codec.
pub struct QuantRotorK3 {
    /// Static rotor table for this layer/head, flat `[n_groups * 4]` f32.
    pub rotors: Vec<f32>,
    /// Static QJL projection matrix, flat `[qjl_dim * head_dim]` f32. `None`
    /// when QJL is disabled — chosen at first `append` time from the global
    /// [`crate::rotor_qjl::rotor_qjl_enabled`] toggle.
    pub qjl_s_matrix: Option<Vec<f32>>,
    /// Accumulated per-token blocks (one entry per append call).
    pub blocks: Vec<RotorKBlocks>,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Maximum sequence length the storage was provisioned for.
    pub max_seq: i32,
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
            .field("n_blocks", &self.blocks.len())
            .field("shape", &self.shape)
            .field("max_seq", &self.max_seq)
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
    #[must_use]
    pub fn new(init_shape: Vec<i32>, max_seq: i32, layer_idx: u32) -> Self {
        Self {
            rotors: Vec::new(),
            qjl_s_matrix: None,
            blocks: Vec::new(),
            shape: init_shape,
            max_seq,
            layer_idx,
            head_idx: 0,
            bits: ROTOR3_K_BITS,
        }
    }

    /// Build a `QuantRotorK3` from pre-computed CPU blocks (SSD hydrate path).
    ///
    /// `max_seq` is the provisioned model window for this layer (NOT the
    /// accumulated sequence length `shape[2]`). Deriving `max_seq` from
    /// `shape[2]` here was identified as a silent regression on next-append;
    /// we take it explicitly from the start.
    ///
    /// When `qjl_s_matrix` is `Some(_)` the QJL sideband was active at write
    /// time; the reader hydrates accordingly.
    #[must_use]
    pub fn from_cpu_blocks(
        rotors: Vec<f32>,
        qjl_s_matrix: Option<Vec<f32>>,
        blocks: Vec<RotorKBlocks>,
        shape: Vec<i32>,
        max_seq: i32,
        layer_idx: u32,
    ) -> Self {
        debug_assert!(
            shape.len() == 4,
            "QuantRotorK3::from_cpu_blocks expects a 4-element [B, kv_h, S, D] shape, got {shape:?}"
        );
        Self {
            rotors,
            qjl_s_matrix,
            blocks,
            shape,
            max_seq,
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
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantRotorK3::append: expected 4D new_shape, got {new_shape:?}"
            )));
        }
        let head_dim = new_shape[3] as usize;
        let n_tokens_total =
            (new_shape[0] as usize) * (new_shape[1] as usize) * (new_shape[2] as usize);

        if self.rotors.is_empty() {
            let n_groups = n_groups_for(head_dim);
            self.rotors = make_rotor_table(self.layer_idx, self.head_idx, n_groups);
            if crate::rotor_qjl::rotor_qjl_enabled() {
                self.qjl_s_matrix = Some(make_qjl_projection(head_dim));
            }
        }

        let (codes, scales, norms, qjl_codes, qjl_norms) = rotor3_k_encode(
            f32_data,
            &self.rotors,
            head_dim,
            self.qjl_s_matrix.as_deref(),
        )
        .map_err(|e: RotorQuantError| {
            rmlx_core::error::Error::Mlx(format!("rotor3_k encode: {e}"))
        })?;

        self.blocks.push(RotorKBlocks {
            codes,
            scales,
            norms,
            qjl_codes,
            qjl_norms,
            n_tokens: n_tokens_total,
        });

        if self.shape.len() != 4 || self.shape[0] == 0 {
            self.shape = new_shape.to_vec();
        } else {
            self.shape[2] += new_shape[2];
        }
        Ok(())
    }

    /// Reset the accumulated sequence to zero. Buffers are kept for reuse.
    /// Does **not** touch the rotor table or QJL projection.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape.len() >= 4 checked immediately before indexing shape[2]"
    )]
    pub fn reset(&mut self) {
        self.blocks.clear();
        if self.shape.len() >= 4 {
            self.shape[2] = 0;
        }
    }

    /// Truncate the accumulated sequence to `n` tokens.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape.len() >= 4 checked immediately before indexing shape[2]"
    )]
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
        if self.shape.len() >= 4 {
            self.shape[2] = n;
        }
    }

    /// Deep-clone (CPU path is plain `Vec` clones).
    ///
    /// # Errors
    /// Currently infallible on the CPU path; returns `Result` for parity.
    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(Self {
            rotors: self.rotors.clone(),
            qjl_s_matrix: self.qjl_s_matrix.clone(),
            blocks: self.blocks.clone(),
            shape: self.shape.clone(),
            max_seq: self.max_seq,
            layer_idx: self.layer_idx,
            head_idx: self.head_idx,
            bits: self.bits,
        })
    }

    /// Approximate byte footprint of the accumulated payload.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        let mut total = self.rotors.len() * size_of::<f32>();
        if let Some(s) = &self.qjl_s_matrix {
            total += s.len() * size_of::<f32>();
        }
        for blk in &self.blocks {
            total += blk.codes.len() * size_of::<u32>();
            total += blk.scales.len() * size_of::<f32>();
            total += blk.norms.len() * size_of::<f32>();
            total += blk.qjl_codes.len();
            total += blk.qjl_norms.len() * size_of::<f32>();
        }
        total
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
            return Err(rmlx_core::error::Error::Mlx(format!(
                "QuantRotorK3::dequant: malformed shape {:?}",
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
                "QuantRotorK3::dequant: rotor table is empty but blocks were appended".into(),
            ));
        }

        for blk in &self.blocks {
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
            .map_err(|e: RotorQuantError| {
                rmlx_core::error::Error::Mlx(format!("rotor3_k decode: {e}"))
            })?;
            out.extend_from_slice(&dec);
        }
        if out.len() < total_elems {
            out.resize(total_elems, 0.0);
        } else if out.len() > total_elems {
            out.truncate(total_elems);
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "quant_rotor_k3_tests.rs"]
mod quant_rotor_k3_tests;
