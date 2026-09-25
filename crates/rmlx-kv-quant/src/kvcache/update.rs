// LOC-exempt: the dispatch itself — the `KvStorage` and `KvQuant` matches, the
// prefill and decode capacity bookkeeping, the bf16 decode mirror, the
// GPU-state and residency walks, and the helpers more than one codec family
// calls. The per-family bodies live in the sibling `update_*.rs` modules;
// docs/KV_UPDATE_PATH.md describes the layout.
//! Update paths: `update`, prefill, GPU state management, and storage-specific appenders.

use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::storage::KvStorage;
use crate::KvQuant;

use super::helpers::{slice_v_prefix, storage_variant_name};
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
pub(super) fn layer_idx_u32(layer_idx: usize) -> u32 {
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
pub(super) fn cast_store_bf16(arr: &Array, device: Device) -> Result<Option<Array>> {
    if arr.dtype() == Dtype::Bf16 {
        Ok(None)
    } else {
        Ok(Some(arr.astype(Dtype::Bf16, device)?))
    }
}

// ── Rotor3 / rotor4 MSL helpers ───────────────────────────────────────────────

/// The GPU-resident `(codes, scales, norms)` triple a packed-K encode produced
/// (rotor or iso), retained so a caller can push it straight into a GPU ring
/// instead of re-uploading the downloaded CPU copy.
///
/// `norms` is always the **per-token** form the decode kernels index
/// (`norms[tok_idx]`), not the per-group form the encode kernels emit.
pub(super) struct PackedKEncodedGpu {
    pub(super) codes: Array,
    pub(super) scales: Array,
    pub(super) norms: Array,
}

/// Collapse a packed-K encode kernel's per-group norms (`[n_tokens, n_groups]`,
/// each row a repeat of that token's L2) to the per-token `[n_tokens]` form.
///
/// Shared by the rotor and iso encode paths — both kernels emit the per-group
/// form and both rings store per-token. GPU-side equivalent of the
/// `norms_per_group[tok * n_groups]` pick in
/// [`crate::rotorquant_msl::rotor_gpu_outputs_to_cpu`] /
/// [`crate::isoquant_msl::iso3_gpu_outputs_to_cpu`] — column 0 of each row.
pub(super) fn collapse_group_norms_to_token(
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

/// Advance a rotor K store's accumulated `shape` by one appended chunk,
/// matching the `QuantRotorK{3,4}::append` bookkeeping. The iso K and V
/// appenders keep the same bookkeeping, so the name says what the helper does
/// and not which family first needed it. Shared by the block-pushing and
/// ring-only append paths so `shape[2]` advances identically whether or not a
/// CPU block is materialised.
#[allow(
    clippy::indexing_slicing,
    reason = "shape rank-4 guard above each indexing site; new_shape rank validated by upstream encoder helper"
)]
pub(super) fn bump_ring_k_shape(shape: &mut Vec<i32>, new_shape: &[i32]) {
    if shape.len() != 4 || shape[0] == 0 {
        *shape = new_shape.to_vec();
    } else {
        shape[2] += new_shape[2];
    }
}

/// Recover `head_dim` from a 4-D new_shape without silently swallowing
/// malformed shapes (previous `.get(3).unwrap_or(0)` pattern). Returns
/// `Error::Mlx` on rank mismatch.
pub(super) fn head_dim_from_shape(new_shape: &[i32], ctx: &str) -> Result<usize> {
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

/// Whether a rotor K GPU append should also maintain the store's GPU ring.
///
/// Only the K-only codecs (`RotorKOnly3` / `RotorKOnly4`) have a kernel that
/// reads the ring — `Rotor{3,4}Sym` and `RotorK{3,4}Asym` quantize V as well and
/// the flash kernel takes bf16 V only, so a ring built for them is never read.
/// It is not free: one `u32` code word plus one
/// [`crate::storage::KV_SIDEBAND_DTYPE`] scale per group and one such norm per
/// token, so
/// `capacity * kv_h * (n_groups * (4 + KV_SIDEBAND_DTYPE.itemsize()) + KV_SIDEBAND_DTYPE.itemsize())`
/// bytes per layer, growing with context — at 4k over a 36-layer model, on the
/// order of a few hundred MB of pure waste. Written as the expression rather
/// than as bytes so a sideband-width change moves it.
///
/// Passed down from the caller rather than inferred here, so eligibility lives
/// with the dispatcher that knows it.
///
/// Both K-only paths maintain — the prefill-time `update_rotor_k_only` as well
/// as the fused decode entry. They only reach a GPU append when QJL is off,
/// which is exactly when the flash kernel is eligible, so the ring they build is
/// the one decode reads. Letting prefill fill it incrementally also avoids
/// making the first decode step re-seed the whole prefix from the CPU blocks
/// (`seed_from_cpu` still covers the paths that genuinely start cold: SSD
/// hydrate, deep clone, and a mid-run fall back to the CPU append).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RingFeed {
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

/// Accumulated sequence length held by a rotor K storage `shape`
/// (`[B, kv_h, S, D]`), or 0 for a not-yet-shaped buffer.
pub(super) fn accumulated_seq(shape: &[i32]) -> i32 {
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
pub(super) fn is_ring_only_append(feed: RingFeed, new_shape: &[i32]) -> bool {
    feed == RingFeed::MaintainRingOnly && matches!(b_kv_h_new_seq(new_shape), Ok((1, _, _)))
}

/// `(b, kv_h, new_seq)` from a rank-4 `[B, kv_h, S, D]` shape.
pub(super) fn b_kv_h_new_seq(new_shape: &[i32]) -> Result<(i32, i32, i32)> {
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
pub(super) fn packed_k_chunk_seq_major(
    new_k: &Array,
    new_shape: &[i32],
    device: Device,
) -> Result<Array> {
    let (_b, _kv_h, new_seq) = b_kv_h_new_seq(new_shape)?;
    if new_seq == 1 {
        // Identity permutation — skip the copy on the decode hot path.
        return new_k.try_clone();
    }
    new_k.transpose(&[0, 2, 1, 3], device)?.contiguous(device)
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
            KvStorage::Mixed { .. } => Err(Error::Mlx(
                "Contract violation: KvCache::update called on a Mixed cache. \
                     These caches MUST be driven through KvCache::update_and_sdpa (universal \
                     wrapper). Direct update() bypasses the quantized SDPA and leaves the cache \
                     in an inconsistent state."
                    .into(),
            )),
            // Affine-K / turbo-V decode update at the variant's V width and
            // TCQ flag, one entry over all four spellings.
            KvStorage::K8VTurbo3 { .. }
            | KvStorage::K8VTurbo2 { .. }
            | KvStorage::K8VTurbo3Tcq { .. }
            | KvStorage::K8VTurbo2Tcq { .. } => self.update_k8_turbo_v(new_k, new_v, device),
            // TurboSym3 / TurboSym4 decode update — symmetric Lloyd-Max K + V
            // at the variant's code width, one entry over both.
            KvStorage::TurboSym3 { .. } | KvStorage::TurboSym4 { .. } => {
                self.update_tsym(new_k, new_v, device)
            }
            // PlanarK decode update — K is PlanarQuant 4-bit, V bf16.
            KvStorage::PlanarK { .. } => self.update_planar_k(new_k, new_v, device),
            // IsoV3 / IsoV4 decode update — K = affine q8_0, V = IsoQuant at
            // the variant's code width.
            KvStorage::IsoV3 { .. } | KvStorage::IsoV4 { .. } => {
                self.update_iso_v(new_k, new_v, device)
            }
            // RotorV3 / RotorV4 decode update — K = affine q8_0, V = rotor at
            // the variant's code width (CPU).
            KvStorage::RotorV3 { .. } | KvStorage::RotorV4 { .. } => {
                self.update_rotor_v(new_k, new_v, device)
            }
            // Iso symmetric / K-only decode updates, one entry per family over
            // both code widths.
            KvStorage::IsoSym3 { .. } | KvStorage::IsoSym4 { .. } => {
                self.update_iso_sym(new_k, new_v, device)
            }
            KvStorage::IsoKOnly3 { .. } | KvStorage::IsoKOnly4 { .. } => {
                self.update_iso_k_only(new_k, new_v, device)
            }
            // Symmetric / K-only rotor variants, one entry per family over
            // both code widths.
            KvStorage::RotorSym3 { .. } | KvStorage::RotorSym4 { .. } => {
                self.update_rotor_sym(new_k, new_v, device)
            }
            KvStorage::RotorKOnly3 { .. } | KvStorage::RotorKOnly4 { .. } => {
                self.update_rotor_k_only(new_k, new_v, device)
            }
            // Asymmetric rotor K + affine V variants.
            KvStorage::RotorKAsym3 { .. } | KvStorage::RotorKAsym4 { .. } => {
                self.update_rotor_k_asym(new_k, new_v, device)
            }
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
    /// `QuantIsoV`, the bf16 mirrors), which is every store this function can
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
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    /// Finalize prefill: quantize the accumulated raw K/V into the storage buffers.
    pub fn exit_prefill(&mut self, device: Device) -> Result<()> {
        // Snapshot before the storage borrows below; one arm forwards it to
        // `MixedKvState::bulk_init_from_fp16`.
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
        // first decode step expands the mirror. And a raw-linear MSL kernel —
        // which is what the flash-decode paths feed this seed to — reads the
        // parent's leading bytes under the slice's shape, i.e. head 0's whole
        // row window in place of every head. Materialising row-major here, on
        // the inference thread that owns the Metal stream, is what makes the
        // seed mean what its shape says for every later reader.
        let is_bf16_storage = matches!(self.storage, KvStorage::None { .. });
        let need_k_seed = is_bf16_storage || self.quant.feeds_bf16_k_at_decode(self.shares_kv);
        let need_v_seed = is_bf16_storage || self.quant.feeds_bf16_v_at_decode(self.shares_kv);
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
        // `IsoKOnly3/4`, `RotorKOnly3/4`, `Iso3Sym`, `Iso4Sym`,
        // `Rotor3Sym`, `Rotor4Sym` — eight of the arms below. The rest are the
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
            // The eight listed there are the ones that still run.
            KvQuant::K8V8 => self.exit_prefill_k8v8(&k_full, &v_full, device)?,
            KvQuant::K8V4 => self.exit_prefill_k8v4(&k_full, &v_full, device)?,
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
                self.exit_prefill_planar(&k_full, &v_full, device)?;
            }
            KvQuant::Mixed { .. } | KvQuant::RotK { .. } => {
                self.exit_prefill_mixed(&k_full, &v_full, device, total_seq, policy)?;
            }
            KvQuant::K8VTurbo3
            | KvQuant::K8VTurbo2
            | KvQuant::K8VTurbo3Tcq
            | KvQuant::K8VTurbo2Tcq => {
                self.exit_prefill_k8_turbo_v(&k_full, &v_full, device, total_seq)?;
            }
            KvQuant::TurboSym3 | KvQuant::TurboSym4 => {
                self.exit_prefill_turbo_sym(&k_full, &v_full, device, total_seq)?;
            }
            KvQuant::PlanarK => self.exit_prefill_planar_k(&k_full, device, total_seq)?,
            KvQuant::Iso3 | KvQuant::Iso4 => {
                self.exit_prefill_iso_v(&k_full, &v_full, device, total_seq)?;
            }
            KvQuant::Rotor3 | KvQuant::Rotor4 => {
                self.exit_prefill_rotor_v(&k_full, &v_full, device, total_seq)?;
            }
            KvQuant::Iso3Sym | KvQuant::Iso4Sym => {
                self.exit_prefill_iso_sym(&k_full, &v_full, device, total_seq)?;
            }
            KvQuant::IsoKOnly3 | KvQuant::IsoKOnly4 => {
                self.exit_prefill_iso_k_only(&k_full, device, total_seq)?;
            }
            KvQuant::Rotor3Sym | KvQuant::Rotor4Sym => {
                self.exit_prefill_rotor_sym(&k_full, &v_full, device, total_seq)?;
            }
            KvQuant::RotorKOnly3 | KvQuant::RotorKOnly4 => {
                self.exit_prefill_rotor_k_only(&k_full, device, total_seq)?;
            }
            KvQuant::RotorK3Asym { .. } | KvQuant::RotorK4Asym { .. } => {
                self.exit_prefill_rotor_k_asym(&k_full, &v_full, device, total_seq)?;
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
            // Model topology, not an allocation. It decides *whether* the two
            // decode mirrors above exist; the bytes are then counted off the
            // buffers themselves, so it must not be added a second time here.
            shares_kv: _,
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
    ///
    /// Fails when the cache cannot reach `n` without losing a position it
    /// would still have to serve — an SWA ring left in rotated order past its
    /// wrap. [`Self::can_truncate_to`] is the predicate, and a caller that has
    /// another way to reach the state (re-prefill) should ask it first.
    pub fn truncate_to(&mut self, n: i32) -> Result<()> {
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
        if let Some(ref mut rot) = self.rotating {
            let delta = self.offset - n;
            if !rot.roll_back(delta)? {
                return Err(Error::Mlx(format!(
                    "KvCache::truncate_to: layer {} is a sliding-window ring that has \
                     wrapped and is not in temporal order, so rolling it back {delta} \
                     positions from {} would drop keys it still has to serve",
                    self.layer_idx, self.offset,
                )));
            }
            self.offset = rot.offset;
            return Ok(());
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
        Ok(())
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

    /// V-only bf16 update helper for IsoKOnly variants, returning the whole
    /// mirror allocation and the number of valid positions in it.
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
    ///
    /// The mirror is `[b, kv_h, max_seq, head_dim]` and the returned length is
    /// its valid prefix on axis 2. Callers that dispatch a flash-decode kernel
    /// pass the allocation on whole, because cutting the prefix out of a
    /// head-major mirror yields a view that is row-contiguous only when
    /// `b * kv_h == 1` — flattening it anywhere else copies the prefix, once
    /// per layer per decode step. Callers that need an exactly-sized tensor use
    /// [`Self::update_decode_fp16_v_only`].
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "decode_fp16_v is Some by construction: the needs_expand branch above always sets it, and this path is only reached after that guard"
    )]
    pub(super) fn update_decode_fp16_v_slab(
        &mut self,
        new_v: &Array,
        max_seq: i32,
        device: Device,
    ) -> Result<(Array, i32)> {
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

        Ok((v_buf.try_clone()?, new_offset))
    }

    /// [`Self::update_decode_fp16_v_slab`] cut to its valid prefix, for callers
    /// that need an exactly-sized `[b, kv_h, offset, head_dim]` V tensor.
    pub(super) fn update_decode_fp16_v_only(
        &mut self,
        new_v: &Array,
        max_seq: i32,
        device: Device,
    ) -> Result<Array> {
        let (v_slab, v_seq) = self.update_decode_fp16_v_slab(new_v, max_seq, device)?;
        slice_v_prefix(&v_slab, v_seq, device)
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
            // rather than re-learning it on its first append. The same is true
            // of the sharing topology: a branch of a producer layer is still a
            // producer layer, and a clone that forgot it would drop the mirror
            // its consumers read at the branch's first `exit_prefill`.
            stream_dtype: self.stream_dtype,
            shares_kv: self.shares_kv,
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
}

// ── Iso decode-update bodies, one per family over both code widths ──────────
//
// The three `KvCache::update_iso_*` entries above resolve their storage variant
// to a code width and hand the stores here; a body below is the one arithmetic
// both widths of a family share, with the width as the store's `BITS`.

// ── Rotor decode-update bodies, one per family over both code widths ─────────
//
// The four `KvCache::update_rotor_*` entries above resolve their storage
// variant to a code width and hand the stores here; a body below is the one
// arithmetic both widths of a family share, with the width as the store's
// `BITS`.

/// The `storage mismatch` error a cache path returns when the dispatch hands
/// it a variant outside the family it serves. `expected` names every variant
/// that family accepts, since one entry now serves the whole shape.
///
/// [`Error::KvStorageMismatch`] rather than [`Error::Mlx`] because the
/// condition is a construction-time defect: the cache was built for one
/// `KvQuant` and is being driven as another. `Error::Mlx` is the transient
/// class the server's retry envelope replays, and replaying this one is
/// futile — the second attempt builds the same wrong cache.
pub(super) fn storage_mismatch(expected: &'static str, storage: &KvStorage) -> Error {
    Error::KvStorageMismatch {
        expected,
        got: storage_variant_name(storage),
    }
}

/// Warn when the `--kv-quant` spelling names one code width and the storage
/// the cache was built with carries another.
///
/// The prefill entries resolve the width from `KvStorage`, which is what the
/// decode path has always done. A disagreement therefore encodes at the
/// storage's width instead of failing, where a per-spelling entry used to
/// return a mismatch. It is a construction-time defect either way; this is
/// what makes it visible. Deliberately not an assert: the storage is the
/// authority both halves of the cache already follow.
pub(super) fn warn_if_width_disagrees(quant: KvQuant, quant_bits: u32, storage_bits: u8) {
    if quant_bits != u32::from(storage_bits) {
        tracing::warn!(
            quant = %quant,
            quant_bits,
            storage_bits,
            "exit_prefill: the quant spelling and the storage disagree on the \
             code width; encoding at the storage's width"
        );
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
