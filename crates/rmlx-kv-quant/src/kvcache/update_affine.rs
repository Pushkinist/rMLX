//! Affine (q8 K, affine V) KV update path.
//!
//! Holds every update-side body that only the affine storage types use: the
//! `update_k8v4` / `update_k8v8` entries, the TurboFlash head-major K8V4
//! buffers and their allocate, grow and append helpers. The `KvStorage`
//! dispatch and the helpers with more than one family caller stay in
//! [`super::update`].
#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::too_many_lines
)]

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{zeros, Array, Device, Dtype};

use crate::storage::{KvStorage, QuantK, QuantV};
use crate::turbo_flash_msl::{turbo_flash_sdpa, turbo_flash_should_run};
use crate::KvQuant;

use super::helpers::{arrays_to_f32, f32_vec_to_array};
use super::KvCache;

impl KvCache {
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
    pub(super) fn update_k8v4(
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
    pub(super) fn update_and_sdpa_k8v4_flash_inner(
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
    pub(super) fn alloc_flash_buffers(
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
    pub(super) fn grow_flash_buffers(
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
    pub(super) fn append_flash_buffers_from_fp16(
        &mut self,
        start: i32,
        n: i32,
        device: Device,
    ) -> Result<()> {
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
    pub(super) fn append_flash_buffers_from_new(
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
    pub(super) fn update_k8v8(
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
}
