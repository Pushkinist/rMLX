// IsoQuant 3-bit K storage struct.
//
// Mirror of `QuantIsoV3` (see `quant_iso_v.rs`) for the K axis. The IsoQuant
// codec (`crate::isoquant::iso_encode_fast` / `iso_decode_fast`) is axis-
// agnostic — it consumes a flat `[B, kv_h, S, D]` f32 row buffer and a
// per-row `head_dim`. We fork the storage struct (not the codec) for the
// same reason `QuantIsoV4` was forked from `QuantIsoV3`: the name is stable
// across crates (SSD writer/reader, helpers, dispatch), and renaming to a
// generic `QuantIso` would cause large cross-crate churn for no benefit —
// bits is fixed per storage variant.
#![allow(
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::indexing_slicing,
    clippy::doc_lazy_continuation
)]
//! Quantized K buffer: `QuantIsoK3` (IsoQuant 3-bit K codec).

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};

use crate::isoquant::{iso_decode_fast, iso_encode_fast, IsoQuantError};
use crate::storage::quant_iso_v::{synced_iso_v_blocks, IsoBlocks};

use super::QuantKGpuRing;

/// Bit-width of the iso3 K codec (fixed at 3-bit — identical codebook to the
/// V-side [`crate::storage::QuantIsoV3`]).
pub const ISO_K3_BITS: u8 = 3;

/// Elements per iso group: the component count of a quaternion.
///
/// **Codec-wide, and not a tunable.** The iso codec's group *is* one quaternion
/// block (`iso_encode_fast` rotates `[w, x, y, z]` at a time), so this is fixed
/// by the algebra rather than chosen per bit width — 3-bit and 4-bit iso differ
/// in codebook and pack stride, never in this. The per-width aliases
/// [`ISO_K3_GROUP_SIZE`] / [`crate::storage::ISO_K4_GROUP_SIZE`] are defined
/// from it so they cannot drift apart: the iso4 GPU paths derive their ring
/// stride through [`iso_n_groups_for`] while `QuantIsoK4::append` uses the K4
/// alias, and a divergence would silently mismatch the iso4 ring against its
/// own CPU blocks.
pub const ISO_QUAT_BLOCK_SIZE: usize = 4;

/// Quaternion-block size for the iso3 K codec. Alias of
/// [`ISO_QUAT_BLOCK_SIZE`] — see it for why this is not independently tunable.
pub const ISO_K3_GROUP_SIZE: usize = ISO_QUAT_BLOCK_SIZE;

/// Per-token group count for an iso K row of `head_dim` elements — **both** bit
/// widths.
///
/// One quaternion block per group, so `head_dim / 4`. This is iso's group rule
/// and it lives here, in iso's own store — [`QuantKGpuRing`] is told the
/// resulting integer and knows nothing about quaternions.
#[must_use]
pub fn iso_n_groups_for(head_dim: usize) -> usize {
    head_dim / ISO_QUAT_BLOCK_SIZE
}

/// [`iso_n_groups_for`] with the group-size invariant enforced, as `i32`.
///
/// A `head_dim` that is not a multiple of the quaternion block size would
/// silently truncate the last partial group, so it is an error rather than a
/// rounded-down group count.
///
/// # Errors
///
/// Returns [`Error::Quant`] when `head_dim` is not a positive multiple of
/// [`ISO_QUAT_BLOCK_SIZE`].
pub(crate) fn iso_n_groups_i32(head_dim: i32, what: &str) -> Result<i32> {
    let hd = usize::try_from(head_dim)
        .map_err(|_| Error::Quant(format!("{what}: head_dim={head_dim} must be positive")))?;
    if hd == 0 || !hd.is_multiple_of(ISO_QUAT_BLOCK_SIZE) {
        return Err(Error::Quant(format!(
            "{what}: head_dim={head_dim} must be a positive multiple of the quaternion \
             block size {ISO_QUAT_BLOCK_SIZE}"
        )));
    }
    i32::try_from(iso_n_groups_for(hd)).map_err(|_| {
        Error::Quant(format!(
            "{what}: n_groups for head_dim={head_dim} exceeds i32::MAX"
        ))
    })
}

/// Accumulated IsoQuant K cache (3-bit, quaternion SO(4) fast mode).
///
/// Carries both forms of the payload:
///
/// * `blocks` — CPU [`IsoBlocks`], the source of truth for `dequant()` and the
///   SSD spill/hydrate round-trip.
/// * `gpu` — the optional GPU-resident packed ring ([`QuantKGpuRing`]),
///   populated by [`Self::gpu_append`]. When present it lets the iso
///   flash-decode kernel read the quant store directly instead of paying a
///   full-prefix CPU `dequant()` per decode step.
///
/// Storage payload layout is identical to [`crate::storage::QuantIsoV3`] — the
/// only distinction at the storage level is the role on the SDPA path.
pub struct QuantIsoK3 {
    /// Accumulated per-token blocks (one entry per append call; `dequant`
    /// flattens them).
    pub blocks: Vec<IsoBlocks>,
    /// GPU-resident packed ring. Empty until the first [`Self::gpu_append`].
    pub gpu: QuantKGpuRing,
    /// Accumulated shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Maximum sequence length the storage was provisioned for.
    ///
    /// Inert: nothing sizes a buffer from it. The provisioned window that the
    /// GPU ring grows against is a **per-call** `max_seq` argument to
    /// [`Self::gpu_append`], read from the live `KvStorage` variant by the
    /// caller. A snapshot cached here would go stale the moment the window
    /// grows during decode, which is exactly how the rotor ring came to reject
    /// valid appends mid-stream — do not wire the ring to this field.
    pub max_seq: i32,
    /// Bit-width tag (always [`ISO_K3_BITS`]).
    pub bits: u8,
}

impl std::fmt::Debug for QuantIsoK3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantIsoK3")
            .field("n_blocks", &self.blocks.len())
            .field("gpu_resident", &self.gpu.is_allocated())
            .field("shape", &self.shape)
            .field("max_seq", &self.max_seq)
            .field("bits", &self.bits)
            .finish()
    }
}

impl QuantIsoK3 {
    /// Construct an empty `QuantIsoK3` for `init_shape = [B, kv_h, 0, D]`.
    #[must_use]
    pub fn new(init_shape: Vec<i32>, max_seq: i32) -> Self {
        Self {
            blocks: Vec::new(),
            gpu: QuantKGpuRing::default(),
            shape: init_shape,
            max_seq,
            bits: ISO_K3_BITS,
        }
    }

    /// Append one K slice (CPU path). Same contract as `QuantIsoV3::append`.
    ///
    /// # Errors
    /// Forwards any [`IsoQuantError`] from [`iso_encode_fast`].
    pub fn append(&mut self, f32_data: &[f32], new_shape: &[i32]) -> Result<()> {
        if new_shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantIsoK3::append: expected 4D new_shape, got {new_shape:?}"
            )));
        }
        let b = new_shape[0] as usize;
        let kv_h = new_shape[1] as usize;
        let new_seq = new_shape[2] as usize;
        let head_dim = new_shape[3] as usize;
        let n_tokens_total = b * kv_h * new_seq;

        // Store each chunk sequence-major (`[B, new_seq, kv_h, D]`) so the
        // per-append blocks share one layout; a head-major store transposes
        // heads across multi-append GQA caches (kv_h>1) when `dequant`
        // reshapes head-major over the full sequence. See
        // [`super::QuantIsoV3::append`].
        let seq_major =
            super::seq_layout::transpose_heads_seq(f32_data, b, kv_h, new_seq, head_dim);

        let (codes, scales, quaternions, norms) =
            iso_encode_fast(&seq_major, head_dim, ISO_K3_GROUP_SIZE, ISO_K3_BITS)
                .map_err(|e: IsoQuantError| Error::Mlx(format!("iso_k3 encode: {e}")))?;

        self.blocks.push(IsoBlocks {
            codes,
            scales,
            quaternions,
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

    /// Construct from pre-computed CPU blocks (SSD hydrate path).
    ///
    /// `max_seq` must be the provisioned model window for this layer, **not**
    /// the accumulated sequence length at spill time (`shape[2]`). Passing
    /// `shape[2]` here would set a stale ceiling equal to the spilled length,
    /// causing the next append after hydration to reject tokens that would
    /// fit within the true model window.
    #[must_use]
    pub fn from_cpu_blocks(blocks: Vec<IsoBlocks>, shape: Vec<i32>, max_seq: i32) -> Self {
        debug_assert!(
            shape.len() == 4,
            "QuantIsoK3::from_cpu_blocks expects a 4-element [B, kv_h, S, D] shape, got {shape:?}"
        );
        Self {
            blocks,
            // Hydrated caches start CPU-only; the ring is rebuilt lazily from
            // the next GPU append.
            gpu: QuantKGpuRing::default(),
            shape,
            max_seq,
            bits: ISO_K3_BITS,
        }
    }

    /// Reset the accumulated sequence length to zero.
    pub fn reset(&mut self) {
        self.blocks.clear();
        // The ring would otherwise keep a prefix the CPU blocks no longer
        // claim. Dropped here; the next `gpu_append` re-seeds from `blocks`.
        self.gpu.clear();
        if self.shape.len() >= 4 {
            self.shape[2] = 0;
        }
    }

    /// Truncate the accumulated sequence to `n` tokens.
    ///
    /// The GPU ring is **kept**, not cleared — mirror of the rotor K store's
    /// `truncate_to`. Lowering `shape[2]` to `n` makes the ring's logical fill
    /// `n`; the stale `[n, prev)` capacity is overwritten by the next append and
    /// never read (`packed_view` slices to `shape[2]`). This preserves any
    /// ring-only decode tail up to `n`, so `dequant` / an SSD spill can still
    /// rebuild it via [`synced_iso_v_blocks`]. Clearing the ring here would
    /// discard the tail (the only copy of `[frozen_prefix, n)`), leaving `blocks`
    /// short of `shape[2]` with no ring — the divergent state `dequant` rejects
    /// loudly, which would abort generation on the speculative-decode rollback
    /// path.
    pub fn truncate_to(&mut self, n: i32) {
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
    /// Materialises any ring-only decode tail into complete CPU blocks first —
    /// mirror of [`super::QuantIsoV3::try_deep_clone`]. A short-blocks clone with
    /// no ring would silently truncate the store; [`synced_iso_v_blocks`] rejects
    /// that loudly instead.
    ///
    /// # Errors
    /// Forwards a [`synced_iso_v_blocks`] reconciliation error.
    pub fn try_deep_clone(&self) -> Result<Self> {
        let blocks =
            synced_iso_v_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?.into_owned();
        Ok(Self {
            // The clone starts CPU-only: `blocks` carries the full payload, so
            // the ring re-seeds from them on the clone's first GPU append.
            // Sharing the source's Arrays would alias one ring across two
            // independent caches.
            gpu: QuantKGpuRing::default(),
            blocks,
            shape: self.shape.clone(),
            max_seq: self.max_seq,
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
    /// `codes` / `scales` / `norms` are the iso encode kernel's GPU outputs for
    /// a **sequence-major** chunk; `norms` must already be per-token (the iso
    /// quantize kernel emits a per-group norm slot — the caller deduplicates).
    /// `prev_seq` is the accumulated sequence length **before** this chunk.
    ///
    /// This only maintains the ring — the caller still pushes the matching CPU
    /// block, which stays the source of truth for `dequant()` and SSD spill.
    ///
    /// `max_seq` is the window the cache is provisioned for **right now**, read
    /// from the active `KvStorage` variant by the caller. It is a parameter
    /// rather than a field so it cannot go stale as the window grows during
    /// decode — see the note on [`Self::max_seq`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Quant`] when `head_dim` violates the group-size
    /// invariant, and forwards [`QuantKGpuRing::seed_from_cpu`] /
    /// [`QuantKGpuRing::append_encoded`] errors.
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
        let n_groups = iso_n_groups_i32(head_dim, "QuantIsoK3::gpu_append")?;
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

    /// Resident bytes held by this store: CPU blocks plus the GPU ring.
    ///
    /// The ring is real resident memory and is counted at its full allocation
    /// — omitting it under-reports this cache's KV bytes, which is the number
    /// the codec is judged on.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            blocks,
            gpu,
            // Geometry / tags, not allocations.
            shape: _,
            max_seq: _,
            bits: _,
        } = self;
        blocks.iter().map(IsoBlocks::byte_size).sum::<u64>() + gpu.byte_size()
    }

    /// Dequantize all accumulated K slices into one flat f32 vector of length
    /// `prod(shape)`.
    ///
    /// # Errors
    /// Returns an `Error::Mlx` if the underlying [`iso_decode_fast`] fails.
    pub fn dequant(&self) -> Result<Vec<f32>> {
        if self.shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantIsoK3::dequant: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        let total_elems: usize = self.shape.iter().map(|&d| d as usize).product();
        let mut out: Vec<f32> = Vec::with_capacity(total_elems);

        // Reconcile the CPU blocks with the GPU ring: on the fused K-only /
        // symmetric decode path the decode tail lives only in the ring
        // (`blocks` trail `shape[2]`), and this rebuilds it on demand rather than
        // decoding a short prefix and zero-padding the gap. Loud on any
        // unrecoverable disagreement.
        let blocks = synced_iso_v_blocks(&self.blocks, &self.shape, &self.gpu, Device::Gpu)?;

        if blocks.is_empty() {
            // Loud on a lost decode tail — see [`super::QuantIsoV3::dequant`].
            // (Zeros are layout-invariant, so the genuine no-token case needs no
            // reorder.)
            if total_elems != 0 {
                return Err(Error::Mlx(format!(
                    "QuantIsoK3::dequant: no blocks but shape {:?} implies {total_elems} elems — \
                     refusing to zero-pad a lost decode tail",
                    self.shape
                )));
            }
            return Ok(out);
        }

        for blk in blocks.iter() {
            let dec = iso_decode_fast(
                &blk.codes,
                &blk.scales,
                &blk.quaternions,
                &blk.norms,
                head_dim,
                ISO_K3_GROUP_SIZE,
                ISO_K3_BITS,
            )
            .map_err(|e: IsoQuantError| Error::Mlx(format!("iso_k3 decode: {e}")))?;
            out.extend_from_slice(&dec);
        }
        // `synced_iso_v_blocks` guarantees the blocks cover `shape[2]`, so a
        // length mismatch here is an internal invariant break — surface it loudly
        // rather than zero-padding or truncating a decoded prefix.
        if out.len() != total_elems {
            return Err(Error::Mlx(format!(
                "QuantIsoK3::dequant: decoded {} elems but shape {:?} implies {total_elems} — \
                 refusing to zero-pad / truncate",
                out.len(),
                self.shape
            )));
        }
        // Blocks are sequence-major (see `append`); reorder back to the logical
        // head-major `[B, kv_h, S, D]`.
        let b = self.shape[0] as usize;
        let kv_h = self.shape[1] as usize;
        let s = self.shape[2] as usize;
        let out = super::seq_layout::transpose_seq_heads(&out, b, s, kv_h, head_dim);
        Ok(out)
    }

    /// GPU dequant via on-demand `Array::from_bytes` upload.
    ///
    /// See [`crate::storage::QuantIsoV3::dequant_gpu`] for the algorithm and
    /// rationale. K-side mirror: identical pack layout, same MSL kernel
    /// (axis-agnostic).
    ///
    /// # Errors
    ///
    /// - `Error::Mlx` if `Array::from_bytes` / kernel dispatch fails.
    /// - `Error::Quant` if `shape` is malformed or `head_dim` violates the
    ///   `ISO_K3_GROUP_SIZE` multiple constraint.
    pub fn dequant_gpu(&self, device: Device) -> Result<Array> {
        if self.shape.len() != 4 {
            return Err(Error::Mlx(format!(
                "QuantIsoK3::dequant_gpu: malformed shape {:?}",
                self.shape
            )));
        }
        let head_dim = self.shape[3] as usize;
        if head_dim == 0 || !head_dim.is_multiple_of(ISO_K3_GROUP_SIZE) {
            return Err(Error::Quant(format!(
                "QuantIsoK3::dequant_gpu: head_dim={head_dim} must be a positive multiple of \
                 ISO_K3_GROUP_SIZE={ISO_K3_GROUP_SIZE}"
            )));
        }
        let n_groups = head_dim / ISO_K3_GROUP_SIZE;

        let mut codes_bytes: Vec<u8> = Vec::new();
        let mut scales_bytes: Vec<u8> = Vec::new();
        let mut quats_bytes: Vec<u8> = Vec::new();
        let mut norms_bytes: Vec<u8> = Vec::new();
        let mut total_groups: usize = 0;

        for blk in &self.blocks {
            for &c in &blk.codes {
                codes_bytes.extend_from_slice(&c.to_le_bytes());
            }
            for &s in &blk.scales {
                scales_bytes.extend_from_slice(&s.to_le_bytes());
            }
            for &q in &blk.quaternions {
                quats_bytes.extend_from_slice(&q.to_le_bytes());
            }
            for &n in &blk.norms {
                let n_bytes = n.to_le_bytes();
                for _ in 0..n_groups {
                    norms_bytes.extend_from_slice(&n_bytes);
                }
            }
            // Checked arithmetic — see V-side mirror.
            let blk_groups = blk.n_tokens.checked_mul(n_groups).ok_or_else(|| {
                Error::Quant("dequant_gpu: blk.n_tokens * n_groups overflow".to_owned())
            })?;
            total_groups = total_groups
                .checked_add(blk_groups)
                .ok_or_else(|| Error::Quant("dequant_gpu: total_groups overflow".to_owned()))?;
        }

        // Guard against silent shape divergence — see V-side mirror.
        let declared_total: usize = self.shape.iter().map(|&d| d as usize).product();
        let actual_total: usize = total_groups.checked_mul(ISO_K3_GROUP_SIZE).ok_or_else(|| {
            Error::Quant("dequant_gpu: total_groups * ISO_K3_GROUP_SIZE overflow".to_owned())
        })?;
        if actual_total != declared_total {
            return Err(Error::Quant(format!(
                "dequant_gpu: actual_total={actual_total} (blocks×groups×group_size) != \
                 declared_total={declared_total} (prod(shape)={:?}); refusing to silently \
                 truncate/pad",
                self.shape
            )));
        }

        if total_groups == 0 {
            return Array::from_bytes(&[][..], &self.shape, Dtype::F32);
        }

        let codes_arr = Array::from_bytes(&codes_bytes, &[total_groups as i32], Dtype::U32)?;
        let scales_arr = Array::from_bytes(&scales_bytes, &[total_groups as i32], Dtype::F32)?;
        let quats_arr = Array::from_bytes(&quats_bytes, &[(total_groups * 4) as i32], Dtype::F32)?;
        let norms_arr = Array::from_bytes(&norms_bytes, &[total_groups as i32], Dtype::F32)?;

        let flat = crate::isoquant_msl::iso_dequantize_v3_gpu(
            &codes_arr,
            &scales_arr,
            &quats_arr,
            &norms_arr,
            head_dim,
            Dtype::F32,
            device,
        )?;

        // CPU blocks are sequence-major (see `append`): reshape to
        // `[B, S, kv_h, D]`, then reorder heads↔seq back to `[B, kv_h, S, D]`.
        let seq_major_shape = [self.shape[0], self.shape[2], self.shape[1], self.shape[3]];
        let out = flat.reshape(&seq_major_shape, device)?;
        out.transpose(&[0, 2, 1, 3], device)?.contiguous(device)
    }
}

#[cfg(test)]
#[path = "quant_iso_k_tests.rs"]
mod quant_iso_k_tests;
