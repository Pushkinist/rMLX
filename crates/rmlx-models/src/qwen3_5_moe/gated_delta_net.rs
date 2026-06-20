// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! GatedDeltaNet block — real per-step recurrence (matches mlx-lm gated_delta_ops).
//!
//! Algorithm (per timestep t, state shape [B, Hv, Dv, Dk]):
//! beta = sigmoid(b) # [B, T, Hv]
//! g = exp(-exp(A_log) * softplus(a + dt_bias)) # [B, T, Hv]
//! for t in 0..T:
//! decay = g[:, t][..., None, None] # [B, Hv, 1, 1]
//! state = state * decay
//! kv_mem = (state * k_t[..., None, :]).sum(-1) # [B, Hv, Dv]
//! delta = (v_t - kv_mem) * beta_t[..., None] # [B, Hv, Dv]
//! state = state + k_t[..., None, :] * delta[..., None]
//! y_t = (state * q_t[..., None, :]).sum(-1) # [B, Hv, Dv]
//! y = stack(ys, axis=1) # [B, T, Hv, Dv]
//!
//! q/k/v come from conv1d(in_proj_qkv(x)) split + per-head RMSNorm + scale.
//! Final output: out_proj(silu(z) * rms_norm(y, norm_weight)).
//!
//! Reference: mlx-lm 0.31 mlx_lm/models/gated_delta.py::gated_delta_ops.

use rmlx_core::error::Result;
use rmlx_mlx::{
    add, concatenate, conv1d, exp, log1p, multiply, rms_norm, sigmoid, silu, Array, Device, Dtype,
};

use rmlx_kv_quant::LinearAttnCache;

use super::layers::Linear;

#[allow(missing_debug_implementations)]
pub(super) struct GatedDeltaNet {
    pub(super) in_proj_qkv: Linear,
    pub(super) in_proj_z: Linear,
    pub(super) in_proj_b: Linear,
    pub(super) in_proj_a: Linear,
    pub(super) conv1d_weight: Array, // [conv_dim, kernel_size, 1]
    pub(super) norm_weight: Array,   // RMSNormGated weight [head_v_dim]
    /// Pre-computed `exp(A_log.astype(f32))` reshaped to `[1, 1, num_v_heads]`.
    /// `A_log` is invariant across calls; pre-computing the inner exp moves
    /// one f32 elementwise op out of the per-step hot path (saves one kernel
    /// launch per GatedDeltaNet layer per decode step — there are 30).
    pub(super) exp_a_log_f32: Array, // [1, 1, num_v_heads] f32
    pub(super) dt_bias_3d: Array,    // [1, 1, num_v_heads] in input dtype
    /// Pre-computed `inv_scale^2 = head_k_dim^-1` scalar in input dtype.
    /// Used to scale q after rms_norm. Saves a per-step scalar_f32+astype.
    pub(super) inv_scale_sq_arr: Array,
    /// Pre-computed `inv_scale = head_k_dim^-0.5` scalar in input dtype.
    pub(super) inv_scale_arr: Array,
    pub(super) out_proj: Linear,
    pub(super) num_k_heads: usize,
    pub(super) num_v_heads: usize,
    pub(super) head_k_dim: usize,
    pub(super) head_v_dim: usize,
    pub(super) key_dim: usize,   // num_k_heads * head_k_dim
    pub(super) value_dim: usize, // num_v_heads * head_v_dim
    pub(super) eps: f32,
}

/// Numerically stable softplus: log1p(exp(x)) is OK for moderate x; for large |x|
/// we use the standard threshold so we don't overflow exp.
fn softplus(x: &Array, device: Device) -> Result<Array> {
    // softplus(x) = log1p(exp(x))
    // Acceptable here because a + dt_bias values are bounded (~|10|).
    let e = exp(x, device)?;
    log1p(&e, device)
}

// `negative` op wrapper — `negative` is a name collision with std::ops in some contexts;
// alias it locally.
fn negative_arr(x: &Array, device: Device) -> Result<Array> {
    rmlx_mlx::negative(x, device)
}

impl GatedDeltaNet {
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(
        &self,
        x: &Array,
        cache: Option<&mut LinearAttnCache>,
        device: Device,
    ) -> Result<Array> {
        let s = x.shape();
        let batch = s[0];
        let seq = s[1];
        let bs = batch as usize;
        let ts = seq as usize;
        let hv = self.num_v_heads;
        let hk = self.num_k_heads;
        let dk = self.head_k_dim;
        let dv = self.head_v_dim;
        let qk_repeat = (hv / hk) as i32; // 32/16 = 2

        // ── Projections ───────────────────────────────────────────────────
        // qkv: [B, S, key_dim*2 + value_dim]
        let qkv = self.in_proj_qkv.forward(x, device)?;
        // z: [B, S, value_dim]
        let z = self.in_proj_z.forward(x, device)?;
        // b, a: [B, S, num_v_heads]
        let b_proj = self.in_proj_b.forward(x, device)?;
        let a_proj = self.in_proj_a.forward(x, device)?;

        // ── Depthwise conv1d on qkv ───────────────────────────────────────
        // conv1d_weight shape: [conv_dim, kernel, 1] (MLX depthwise layout).
        let w_shape = self.conv1d_weight.shape();
        let kernel = w_shape[1] as usize;
        let conv_dim = self.key_dim * 2 + self.value_dim;

        // Conv tail prefix: either the cached `(kernel-1)` last tokens of the
        // previous call's conv input, or zeros for a fresh prefill.
        // mlx-lm reference: `conv_input = mx.concatenate([conv_state, qkv], axis=1)`.
        let pad = kernel - 1;
        let prev_conv_state: Option<Array> = cache.as_ref().and_then(|c| {
            c.conv_state
                .as_ref()
                .map(|a| a.try_clone().expect("clone conv_state"))
        });
        let conv_prefix = if let Some(arr) = prev_conv_state {
            arr
        } else {
            let pad_size = bs * pad * conv_dim;
            let pad_data = vec![0.0_f32; pad_size];
            let pad_bytes = unsafe {
                std::slice::from_raw_parts(pad_data.as_ptr().cast::<u8>(), pad_data.len() * 4)
            };
            let pad_arr =
                Array::from_bytes(pad_bytes, &[batch, pad as i32, conv_dim as i32], Dtype::F32)?;
            pad_arr.astype(qkv.dtype(), device)?
        };
        let qkv_padded = concatenate(&[&conv_prefix, &qkv], 1, device)?;

        // Native MLX depthwise conv1d — replaces the kernel-step manual loop
        // (was kernel * 3 ops per call: slice + mul + add). groups=conv_dim
        // makes this depthwise; weight layout [conv_dim, kernel, 1] is the
        // MLX-native layout (the loader doesn't transpose it).
        let conv_raw = conv1d(
            &qkv_padded,
            &self.conv1d_weight,
            1,
            0,
            1,
            conv_dim as i32,
            device,
        )?;
        let conv_out = silu(&conv_raw, device)?; // [B, S, conv_dim]

        // ── Split into q, k, v ────────────────────────────────────────────
        let kd_total = self.key_dim as i32; // num_k_heads * head_k_dim
        let vd_total = self.value_dim as i32;
        let q_flat = conv_out.slice(&[0, 0, 0], &[batch, seq, kd_total], &[1, 1, 1], device)?;
        let k_flat = conv_out.slice(
            &[0, 0, kd_total],
            &[batch, seq, kd_total * 2],
            &[1, 1, 1],
            device,
        )?;
        let v_flat = conv_out.slice(
            &[0, 0, kd_total * 2],
            &[batch, seq, kd_total * 2 + vd_total],
            &[1, 1, 1],
            device,
        )?;

        let q4 = q_flat.reshape(&[batch, seq, hk as i32, dk as i32], device)?;
        let k4 = k_flat.reshape(&[batch, seq, hk as i32, dk as i32], device)?;
        let v4 = v_flat.reshape(&[batch, seq, hv as i32, dv as i32], device)?;

        // ── q/k RMSNorm + per-head scaling ────────────────────────────────
        // Per mlx-lm gated_delta_net:
        // inv_scale = Dk ** -0.5
        // q = (inv_scale ** 2) * rms_norm(q, None, 1e-6)
        // k = inv_scale * rms_norm(k, None, 1e-6)
        let q4 = rms_norm(&q4, None, 1e-6, device)?;
        let k4 = rms_norm(&k4, None, 1e-6, device)?;

        // Use pre-computed scalar arrays (built at load time) — saves a
        // scalar_f32 + astype dance every call * 30 GDN layers.
        let q4_scaled_full = multiply(&q4, &self.inv_scale_sq_arr, device)?;
        let k4_scaled_full = multiply(&k4, &self.inv_scale_arr, device)?;
        // q4_scaled_full / k4_scaled_full shape: [B, S, num_k_heads, head_k_dim]
        // (UN-repeated; the GPU kernel
        // does the GQA repeat internally
        // via hk_idx = hv_idx / (Hv/Hk)).
        // v4 shape: [B, S, num_v_heads, head_v_dim]

        // ── Compute g and beta ────────────────────────────────────────────
        // beta = sigmoid(b_proj) # [B, S, Hv] (model dtype)
        // g = exp(-exp(A_log.astype(f32)) * softplus(a_proj + dt_bias)) # [B, S, Hv] f32
        //
        // Match mlx-lm `compute_g` exactly: A_log is cast to f32 BEFORE the
        // outer exp, so the exp/multiply/exp pipeline runs in f32. Doing the
        // exp in bf16 (as the previous ops-loop did) loses precision and
        // contributed to drift versus mlx-lm's kernel path.
        let beta_full = sigmoid(&b_proj, device)?;

        // a_proj: [B, S, Hv]; dt_bias_3d: pre-shaped [1, 1, Hv].
        // exp_a_log_f32 is pre-computed at load time so per-step skips:
        // reshape A_log + astype f32 + exp. 3 fewer kernel launches per
        // GDN layer per decode step.
        let a_plus = add(&a_proj, &self.dt_bias_3d, device)?;
        let sp = softplus(&a_plus, device)?; // [B, S, Hv]
        let neg_prod = {
            let prod = multiply(&self.exp_a_log_f32, &sp, device)?; // promotes to f32
            negative_arr(&prod, device)?
        };
        let g_full = exp(&neg_prod, device)?; // [B, S, Hv] f32

        // ── Recurrent state (carries across decode steps via the cache) ───
        // state shape: [B, Hv, Dv, Dk] f32 (matches mlx-lm dtype).
        let state_in = match cache.as_ref().and_then(|c| c.delta_state.as_ref()) {
            Some(prev) => prev.try_clone()?,
            None => rmlx_mlx::zeros(
                &[batch, hv as i32, dv as i32, dk as i32],
                Dtype::F32,
                device,
            )?,
        };

        // ── Recurrence dispatch ───────────────────────────────────────────
        // Always the sequential MSL kernel `gated_delta_step_gpu`, for both
        // decode (T=1) and prefill (T=chunk). It is a byte-for-byte port of
        // mlx-lm 0.31's `gated_delta_kernel`, which mlx-lm itself uses for the
        // whole prompt (`use_kernel=True` default): the `for t in 0..T` loop
        // runs inside one Metal dispatch with the recurrent state in registers
        // while the `B*Hv*Dv*32` threads supply the parallelism. The
        // per-timestep loop is sequential WITHIN a thread, but the GPU hides
        // that latency across heads/dims — so a single dispatch over a large T
        // beats both small chunks (more dispatches) and the ops-graph path
        // (`gated_delta_prefill_ops`), whose Rust per-step graph build explodes
        // to ~184K–1.47M lazy nodes at T=256..2048.
        //
        // Chaining the kernel across prefill chunks is numerically exact: the
        // f32 `state_in`→`state_out` carries between calls, so chunked prefill
        // equals a single full-prompt call, and the prefill→decode handoff now
        // shares one reduction order (decode already used this kernel).
        let g_f32 = if g_full.dtype() == Dtype::F32 {
            g_full
        } else {
            g_full.astype(Dtype::F32, device)?
        };

        // Unify q/k/v/beta to a single dtype before the kernel — MLX affine
        // INT4 matmul returns the input dtype but scalar Bf16 multiplications
        // can silently promote to F32. Use v4's dtype as ground truth.
        let target_dtype = v4.dtype();
        let q4_scaled_full = if q4_scaled_full.dtype() == target_dtype {
            q4_scaled_full
        } else {
            q4_scaled_full.astype(target_dtype, device)?
        };
        let k4_scaled_full = if k4_scaled_full.dtype() == target_dtype {
            k4_scaled_full
        } else {
            k4_scaled_full.astype(target_dtype, device)?
        };
        let beta_full = if beta_full.dtype() == target_dtype {
            beta_full
        } else {
            beta_full.astype(target_dtype, device)?
        };
        let (y_bf16, state_out) = crate::gated_delta_msl::gated_delta_step_gpu(
            &q4_scaled_full,
            &k4_scaled_full,
            &v4,
            &g_f32,
            &beta_full,
            &state_in,
            device,
        )?;

        // ── Persist recurrent state + conv tail into the cache ────────────
        // mlx-lm reference: `cache[1] = state` and
        // `cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])`.
        if let Some(c) = cache {
            c.delta_state = Some(state_out);

            // Save the last (kernel-1) tokens of qkv_padded as the new
            // conv_state for the next call. mlx-lm uses `mx.contiguous(...)`
            // — slice() in mlx-c also produces a contiguous result here.
            let total_len = (pad + ts) as i32;
            let tail = qkv_padded.slice(
                &[0, total_len - pad as i32, 0],
                &[batch, total_len, conv_dim as i32],
                &[1, 1, 1],
                device,
            )?;
            c.conv_state = Some(tail);
        }
        // Suppress unused warnings when ts/qk_repeat aren't otherwise referenced
        // after the kernel call subsumes the per-step loop.
        let _ = (ts, qk_repeat);

        // ── RMSNormGated + out_proj ───────────────────────────────────────
        // norm: y -> rms_norm(y, norm_weight, eps), then * silu(z).
        let z_heads = z.reshape(&[batch, seq, hv as i32, dv as i32], device)?;
        let y_normed = rms_norm(&y_bf16, Some(&self.norm_weight), self.eps, device)?;
        let z_silu = silu(&z_heads, device)?;
        let gated = multiply(&y_normed, &z_silu, device)?;
        let gated_flat = gated.reshape(&[batch, seq, vd_total], device)?;

        self.out_proj.forward(&gated_flat, device)
    }
}
