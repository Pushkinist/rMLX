// Shared GPU-resident ring for the packed K codecs that carry a
// `(codes, per-group scales, per-token norm)` payload — rotor3 / rotor4 and
// iso3 / iso4 today.
//
// Those stores differ in bit width, codebook, and rotation algebra, but their
// ring bookkeeping is identical, so they embed this helper instead of each
// carrying a copy. The ring is deliberately codec-agnostic: it is told
// `n_groups` rather than deriving it, so no codec's group-size rule leaks in
// here.
#![allow(unreachable_pub, clippy::exhaustive_structs)]
//! GPU packed ring for a quantized K store: `QuantKGpuRing`.
//!
//! # Why this exists
//!
//! These K codecs' CPU form (`RotorKBlocks` / `IsoBlocks`) is a `Vec<u32>` per
//! append. Decoding attention from it means `dequant()` — a full-prefix CPU
//! decode plus re-upload on **every decode step**, which is O(seq) host work
//! per token with the GPU idle.
//!
//! This ring keeps the same payload (codes / per-group scales / per-token L2
//! norms) resident on the GPU, appended in place, so a flash-decode kernel can
//! read the quant store directly. It is the K-side analogue of the GPU buffers
//! on [`super::QuantPlanarK`].
//!
//! # Layout
//!
//! Buffers are flat and **sequence-major** — the canonical flat-KV layout in
//! this crate. For a token at sequence position `s` and KV head `h`:
//!
//! * `codes[(s * kv_h + h) * n_groups + g]`  (1 u32 per group; what a group
//!   spans is the codec's business — 8 Cl(3,0) multivector components for
//!   rotor, one 4-element quaternion block for iso)
//! * `scales[(s * kv_h + h) * n_groups + g]` (1 f32 per group)
//! * `norms[s * kv_h + h]`                   (1 f32 per token)
//!
//! One sequence position therefore occupies `kv_h * n_groups` code words and
//! `kv_h` norms, which is what lets an append land at a fixed per-step stride.
//!
//! # Batch
//!
//! Batch is **not** interleaved into the stride: the ring is written for
//! `b == 1`, matching one `KvCache` per request. The dispatcher gates on
//! `b == 1` and falls back to the CPU path otherwise, rather than silently
//! writing a layout the kernel would misread.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{zeros, Array, Device, Dtype};

use super::KV_PAGE_SIZE;

/// GPU-resident packed K payload plus its growth bookkeeping.
///
/// `None` buffers mean "not yet allocated" — the ring is created lazily on the
/// first GPU append, so a CPU-only cache never pays for it.
#[derive(Debug, Default)]
pub struct QuantKGpuRing {
    /// Packed codes, flat `u32 [capacity * kv_h * n_groups]`.
    pub codes: Option<Array>,
    /// Per-group scales, flat `f32 [capacity * kv_h * n_groups]`.
    pub scales: Option<Array>,
    /// Per-token L2 norms, flat `f32 [capacity * kv_h]`.
    pub norms: Option<Array>,
    /// Code words / scales per sequence position (`kv_h * n_groups`).
    pub codes_per_step: i32,
    /// Norms per sequence position (`kv_h`).
    pub norms_per_step: i32,
    /// Sequence positions the buffers are currently sized for.
    pub capacity: i32,
}

impl QuantKGpuRing {
    /// True once the ring has been allocated.
    #[must_use]
    pub fn is_allocated(&self) -> bool {
        self.codes.is_some() && self.scales.is_some() && self.norms.is_some()
    }

    /// Drop the ring. Used by `reset` / `truncate_to`, where the CPU blocks
    /// remain the source of truth and the ring would otherwise hold a stale
    /// prefix. Re-allocated (and re-seeded) on the next GPU append.
    pub fn clear(&mut self) {
        self.codes = None;
        self.scales = None;
        self.norms = None;
        self.capacity = 0;
    }

    /// Approximate resident byte footprint.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        if !self.is_allocated() {
            return 0;
        }
        let cap = self.capacity.max(0) as usize;
        let cps = self.codes_per_step.max(0) as usize;
        let nps = self.norms_per_step.max(0) as usize;
        // codes (u32) + scales (f32) + norms (f32).
        cap * cps * 4 + cap * cps * 4 + cap * nps * 4
    }

    /// Append one already-encoded chunk into the ring at `prev_seq`.
    ///
    /// `codes` / `scales` / `norms` are the GPU outputs of the codec's encode
    /// kernel for a **sequence-major** chunk of `new_seq` positions.
    ///
    /// `n_groups` is the codec's per-token group count — passed in rather than
    /// derived, so the ring stays codec-agnostic (rotor's `ceil(head_dim / 3)`
    /// and iso's `head_dim / 4` are both just an integer here).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Quant`] on a shape-contract violation and [`Error::Mlx`]
    /// on an MLX allocation / slice failure.
    pub fn append_encoded(
        &mut self,
        codes: &Array,
        scales: &Array,
        norms: &Array,
        kv_h: i32,
        n_groups: i32,
        prev_seq: i32,
        new_seq: i32,
        max_seq: i32,
        device: Device,
    ) -> Result<()> {
        if kv_h <= 0 || n_groups <= 0 || new_seq <= 0 || prev_seq < 0 {
            return Err(Error::Quant(format!(
                "QuantKGpuRing::append_encoded: bad shape kv_h={kv_h}, n_groups={n_groups}, \
                 prev_seq={prev_seq}, new_seq={new_seq}"
            )));
        }
        let codes_per_step = kv_h * n_groups;
        let norms_per_step = kv_h;

        if self.is_allocated()
            && (self.codes_per_step != codes_per_step || self.norms_per_step != norms_per_step)
        {
            return Err(Error::Quant(format!(
                "QuantKGpuRing::append_encoded: stride changed under a live ring \
                 (codes {} -> {codes_per_step}, norms {} -> {norms_per_step})",
                self.codes_per_step, self.norms_per_step
            )));
        }
        self.codes_per_step = codes_per_step;
        self.norms_per_step = norms_per_step;

        let needed = prev_seq
            .checked_add(new_seq)
            .ok_or_else(|| Error::Quant("QuantKGpuRing::append_encoded: seq overflow".into()))?;
        if needed > max_seq {
            return Err(Error::Quant(format!(
                "QuantKGpuRing::append_encoded: needed={needed} exceeds max_seq={max_seq}"
            )));
        }

        if !self.is_allocated() {
            // Allocating here and writing only `[prev_seq, needed)` would leave
            // `[0, prev_seq)` zeroed — attention over a zeroed prefix, silently
            // wrong rather than an error. A prefix that already exists must come
            // in through `seed_from_cpu` first. Enforced here, not just in the
            // `QuantRotorK::gpu_append` caller, because this type is publicly
            // re-exported.
            if prev_seq > 0 {
                return Err(Error::Quant(format!(
                    "QuantKGpuRing::append_encoded: ring is unallocated but prev_seq={prev_seq} \
                     tokens already exist — seed_from_cpu() must upload the prefix first, \
                     or [0, {prev_seq}) would silently read as zeros"
                )));
            }
            let cap = page_round(needed, max_seq);
            self.alloc(cap, device)?;
        } else if needed > self.capacity {
            let cap = page_round(needed, max_seq);
            self.grow(cap, prev_seq, device)?;
        }

        let (cps, nps) = (self.codes_per_step, self.norms_per_step);
        write_range(&mut self.codes, codes, prev_seq * cps, needed * cps, device)?;
        write_range(
            &mut self.scales,
            scales,
            prev_seq * cps,
            needed * cps,
            device,
        )?;
        write_range(&mut self.norms, norms, prev_seq * nps, needed * nps, device)?;
        Ok(())
    }

    /// Allocate the ring and upload an already-accumulated CPU prefix into it.
    ///
    /// Needed because the prefill path bulk-quantizes K on the CPU: at the
    /// first decode step the ring is empty while `filled_seq` positions already
    /// exist in `RotorKBlocks`. Appending straight at `filled_seq` would leave
    /// `[0, filled_seq)` as zeros — silently wrong attention, not a slow path.
    /// The same applies after an SSD hydrate or a deep clone.
    ///
    /// `codes` / `scales` / `norms` are the concatenated CPU blocks in
    /// sequence-major order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Quant`] when the supplied lengths disagree with
    /// `filled_seq` and the derived per-step strides, and [`Error::Mlx`] on an
    /// MLX allocation / upload failure.
    #[allow(clippy::too_many_arguments)]
    pub fn seed_from_cpu(
        &mut self,
        codes: &[u32],
        scales: &[f32],
        norms: &[f32],
        kv_h: i32,
        n_groups: i32,
        filled_seq: i32,
        max_seq: i32,
        device: Device,
    ) -> Result<()> {
        if self.is_allocated() {
            return Ok(());
        }
        if kv_h <= 0 || n_groups <= 0 || filled_seq <= 0 {
            return Err(Error::Quant(format!(
                "QuantKGpuRing::seed_from_cpu: bad shape kv_h={kv_h}, n_groups={n_groups}, \
                 filled_seq={filled_seq}"
            )));
        }
        self.codes_per_step = kv_h * n_groups;
        self.norms_per_step = kv_h;

        let want_codes = i64::from(filled_seq) * i64::from(self.codes_per_step);
        let want_norms = i64::from(filled_seq) * i64::from(self.norms_per_step);
        if codes.len() as i64 != want_codes
            || scales.len() as i64 != want_codes
            || norms.len() as i64 != want_norms
        {
            return Err(Error::Quant(format!(
                "QuantKGpuRing::seed_from_cpu: CPU prefix length mismatch at filled_seq={filled_seq} \
                 (codes {} want {want_codes}, scales {} want {want_codes}, norms {} want \
                 {want_norms})",
                codes.len(),
                scales.len(),
                norms.len(),
            )));
        }
        if filled_seq > max_seq {
            return Err(Error::Quant(format!(
                "QuantKGpuRing::seed_from_cpu: filled_seq={filled_seq} exceeds max_seq={max_seq}"
            )));
        }

        let cap = page_round(filled_seq, max_seq);
        self.alloc(cap, device)?;

        let codes_arr = u32_slice_to_array(codes)?;
        let scales_arr = f32_slice_to_array(scales)?;
        let norms_arr = f32_slice_to_array(norms)?;

        let (cps, nps) = (self.codes_per_step, self.norms_per_step);
        write_range(&mut self.codes, &codes_arr, 0, filled_seq * cps, device)?;
        write_range(&mut self.scales, &scales_arr, 0, filled_seq * cps, device)?;
        write_range(&mut self.norms, &norms_arr, 0, filled_seq * nps, device)?;
        tracing::debug!(
            filled_seq,
            cap,
            "QuantKGpuRing: seeded ring from accumulated CPU blocks"
        );
        Ok(())
    }

    /// Slice the ring down to the `kv_seq` filled positions.
    ///
    /// Returns `None` when the ring is not allocated (CPU path, or before the
    /// first GPU append) so the caller can fall through to the CPU dequant.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Quant`] when `kv_seq` exceeds the allocated capacity and
    /// [`Error::Mlx`] on an MLX slice failure.
    pub fn packed_view(
        &self,
        kv_seq: i32,
        device: Device,
    ) -> Result<Option<(Array, Array, Array)>> {
        if !self.is_allocated() {
            return Ok(None);
        }
        if kv_seq <= 0 {
            return Ok(None);
        }
        if kv_seq > self.capacity {
            return Err(Error::Quant(format!(
                "QuantKGpuRing::packed_view: kv_seq={kv_seq} exceeds capacity={}",
                self.capacity
            )));
        }
        let cps = self.codes_per_step;
        let nps = self.norms_per_step;
        let codes = slice_prefix(self.codes.as_ref(), kv_seq * cps, "codes", device)?;
        let scales = slice_prefix(self.scales.as_ref(), kv_seq * cps, "scales", device)?;
        let norms = slice_prefix(self.norms.as_ref(), kv_seq * nps, "norms", device)?;
        Ok(Some((codes, scales, norms)))
    }

    fn alloc(&mut self, cap: i32, device: Device) -> Result<()> {
        let cps = self.codes_per_step;
        let nps = self.norms_per_step;
        self.codes = Some(zeros(&[cap * cps], Dtype::U32, device)?);
        self.scales = Some(zeros(&[cap * cps], Dtype::F32, device)?);
        self.norms = Some(zeros(&[cap * nps], Dtype::F32, device)?);
        self.capacity = cap;
        tracing::debug!(cap, cps, nps, "QuantKGpuRing: ring allocated");
        Ok(())
    }

    /// Reallocate to `cap` positions, copying the `prev_seq` filled prefix.
    fn grow(&mut self, cap: i32, prev_seq: i32, device: Device) -> Result<()> {
        let cps = self.codes_per_step;
        let nps = self.norms_per_step;
        let old_capacity = self.capacity;
        self.codes = Some(regrow(
            self.codes.take(),
            cap * cps,
            prev_seq * cps,
            Dtype::U32,
            "codes",
            device,
        )?);
        self.scales = Some(regrow(
            self.scales.take(),
            cap * cps,
            prev_seq * cps,
            Dtype::F32,
            "scales",
            device,
        )?);
        self.norms = Some(regrow(
            self.norms.take(),
            cap * nps,
            prev_seq * nps,
            Dtype::F32,
            "norms",
            device,
        )?);
        self.capacity = cap;
        tracing::debug!(
            old_capacity,
            new_capacity = cap,
            prev_seq,
            "QuantKGpuRing: ring grown"
        );
        Ok(())
    }
}

/// Round `needed` up to a whole number of `KV_PAGE_SIZE` pages, capped at
/// `max_seq`. Mirrors the paged-growth policy the other GPU KV buffers use.
fn page_round(needed: i32, max_seq: i32) -> i32 {
    let pages = needed.div_euclid(KV_PAGE_SIZE) + i32::from(needed.rem_euclid(KV_PAGE_SIZE) != 0);
    (pages * KV_PAGE_SIZE).min(max_seq).max(needed)
}

/// Allocate `new_len` and copy the `filled` prefix out of `old`.
fn regrow(
    old: Option<Array>,
    new_len: i32,
    filled: i32,
    dtype: Dtype,
    what: &str,
    device: Device,
) -> Result<Array> {
    let fresh = zeros(&[new_len], dtype, device)?;
    if filled <= 0 {
        return Ok(fresh);
    }
    let old = old.ok_or_else(|| {
        Error::Mlx(format!(
            "QuantKGpuRing::grow: {what} buffer vanished mid-grow (internal invariant)"
        ))
    })?;
    let prefix = old.slice(&[0], &[filled], &[1], device)?;
    fresh.slice_update(&prefix, &[0], &[filled], &[1], device)
}

/// `slice_update` `src` into `buf[start..stop]`.
fn write_range(
    buf: &mut Option<Array>,
    src: &Array,
    start: i32,
    stop: i32,
    device: Device,
) -> Result<()> {
    let target = buf.take().ok_or_else(|| {
        Error::Mlx("QuantKGpuRing::append_encoded: buffer absent after alloc".into())
    })?;
    let span = stop - start;
    let src_len: i32 = src.shape().iter().product();
    if src_len != span {
        return Err(Error::Quant(format!(
            "QuantKGpuRing::append_encoded: encoded chunk length {src_len} != target span {span}"
        )));
    }
    let flat = src.reshape(&[span], device)?;
    *buf = Some(target.slice_update(&flat, &[start], &[stop], &[1], device)?);
    Ok(())
}

/// Upload a `u32` slice as a flat 1-D GPU array.
fn u32_slice_to_array(vals: &[u32]) -> Result<Array> {
    // SAFETY: `u32` is `Copy` with a fixed 4-byte layout and alignment ≥ `u8`,
    // so reinterpreting the slice as `&[u8]` of 4× the length is sound.
    // `Array::from_bytes` copies before the borrow ends.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), vals.len() * 4) };
    let len = i32::try_from(vals.len())
        .map_err(|_| Error::Quant("QuantKGpuRing: u32 prefix exceeds i32::MAX".into()))?;
    Array::from_bytes(bytes, &[len], Dtype::U32)
}

/// Upload an `f32` slice as a flat 1-D GPU array.
fn f32_slice_to_array(vals: &[f32]) -> Result<Array> {
    // SAFETY: same reasoning as `u32_slice_to_array` — `f32` is `Copy`, 4-byte,
    // alignment ≥ `u8`; `Array::from_bytes` copies before the borrow ends.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), vals.len() * 4) };
    let len = i32::try_from(vals.len())
        .map_err(|_| Error::Quant("QuantKGpuRing: f32 prefix exceeds i32::MAX".into()))?;
    Array::from_bytes(bytes, &[len], Dtype::F32)
}

fn slice_prefix(buf: Option<&Array>, len: i32, what: &str, device: Device) -> Result<Array> {
    let b = buf.ok_or_else(|| {
        Error::Mlx(format!(
            "QuantKGpuRing::packed_view: {what} buffer absent (internal invariant)"
        ))
    })?;
    b.slice(&[0], &[len], &[1], device)
}

#[cfg(test)]
#[path = "quant_k_gpu_ring_tests.rs"]
mod quant_k_gpu_ring_tests;
