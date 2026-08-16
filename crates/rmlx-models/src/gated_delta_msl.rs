// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel input data for MSL dispatch
#![allow(unsafe_code)]

//! GatedDeltaNet MSL kernel — port of mlx-lm 0.31's `gated_delta_kernel`.
//!
//! # What this is
//!
//! GPU port of the recurrent linear-attention update used by Qwen3.5-MoE
//! GatedDeltaNet layers. The reference lives in mlx-lm at
//! `mlx_lm/models/gated_delta.py::_make_gated_delta_kernel` (non-vectorized,
//! no-mask flavor — the only one Qwen3.6 uses).
//!
//! # Why this matters
//!
//! mlx-lm has TWO implementations: an ops fallback (`gated_delta_ops`) and the
//! Metal kernel (`gated_delta_kernel`). Their reduction orders differ — the
//! kernel uses a `simd_sum` across 32 lanes per `dv` slice while the ops path
//! does an `mx.sum(axis=-1)` over 128 elements with a different tree. After
//! ~16 decode tokens the bf16 trajectories diverge. rMLX previously matched
//! the ops path; this kernel closes the gap to the kernel path so we are
//! token-for-token identical to mlx-lm 0.31 default (kernel) generation.
//!
//! # Algorithm (matches mlx-lm gated_delta_kernel non-vectorized + no mask)
//!
//! Per (batch, hv head) pair, with `n_per_t = Dk / 32` state elements per
//! thread along the key axis:
//!
//! ```text
//! state[n_per_t] = state_in[..., dv_idx, dk_lane*n_per_t + i] (loaded as f32)
//! for t in 0..T:
//! state[i] *= g[b, t, hv] # decay
//! kv_mem = simd_sum(state[i] * k[b, t, hk, dk_idx])
//! delta = (v[b, t, hv, dv] - kv_mem) * beta[b, t, hv]
//! state[i] += k[b, t, hk, dk_idx] * delta # rank-1 update
//! y = simd_sum(state[i] * q[b, t, hk, dk_idx])
//! if simd_lane == 0: y_out[b, t, hv, dv] = (InT)y
//! state_out[..., dv_idx, dk_lane*n_per_t + i] = (StT)state[i]
//! ```
//!
//! `hk_idx = hv_idx / (Hv / Hk)` does the GQA repeat inside the kernel — q/k
//! are passed UN-repeated (shape `[B, T, Hk, Dk]`).
//!
//! # Layout
//!
//! - Grid: `(32, Dv, B * Hv)` total threads.
//! - Threadgroup: `(32, 4, 1)` = 128 threads = 4 simdgroups.
//! - `simd_sum` reduces the 32 dk lanes within one simdgroup.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching.

#![allow(clippy::float_cmp)]
use rmlx_core::error::{Error, Result};
use rmlx_mlx::compile::{compile, Closure};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{
    add, expand_dims, multiply, repeat_axis, stack_axis, subtract, sum_axis, Array, Device, Dtype,
};
use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex, OnceLock};

// ── MSL kernel source ────────────────────────────────────────────────────────
//
// The body lives in a `.metal` file so `make check-metal-compiles` and
// `make check-metal-format` see it; `include_str!` embeds it at compile time,
// so the binary still carries no runtime data files.
//
// MLX's `mlx_fast_metal_kernel` wraps this body with a Metal `[[kernel]]`
// signature. Inputs become `device const T* name` (or `constant T* name` for
// arrays under 8 elements; scalar 0-D arrays become `constant T& name`).
// Outputs become `device T* name`. Templated names (`InT`, `StT`, `Dk`, `Dv`,
// `Hk`, `Hv`) must be supplied via `set_template_int` / `set_template_dtype`
// at dispatch time.
const KERNEL_SOURCE: &str = include_str!("metal/gated_delta_step.metal");

// ── Kernel singleton ─────────────────────────────────────────────────────────

static GATED_DELTA_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn gated_delta_kernel_handle() -> Result<&'static MetalKernel> {
    GATED_DELTA_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "gated_delta_step",
                "", // no header — kernel is self-contained
                KERNEL_SOURCE,
                &["q", "k", "v", "g", "beta", "state_in", "T"],
                &["y", "state_out"],
            )
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("gated_delta kernel init: {e}")))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// GPU GatedDeltaNet step kernel.
///
/// Mirrors mlx-lm 0.31's `gated_delta_kernel` (non-vectorized, no mask).
///
/// # Inputs
///
/// - `q`: `[B, T, Hk, Dk]` model dtype (bf16/f16/f32). UN-repeated along Hk.
/// - `k`: `[B, T, Hk, Dk]` same dtype as q.
/// - `v`: `[B, T, Hv, Dv]` same dtype as q.
/// - `g`: `[B, T, Hv]` f32 — gating decay (one scalar per Hv head).
/// - `beta`: `[B, T, Hv]` same dtype as q — delta rule coefficient.
/// - `state_in`: `[B, Hv, Dv, Dk]` f32 — recurrent state at t=0.
///
/// # Outputs
///
/// - `y`: `[B, T, Hv, Dv]` same dtype as q.
/// - `state_out`: `[B, Hv, Dv, Dk]` f32.
///
/// # Constraints
///
/// - `Dk % 32 == 0` (each simdgroup of 32 threads covers Dk via `n_per_t`
///   state elements per thread).
/// - `Hv % Hk == 0` (GQA repeat factor must be exact).
/// - `B`, `T`, `Hk`, `Hv`, `Dv` ≥ 1.
///
/// # Errors
///
/// Returns `Error::Mlx` on kernel compile/dispatch failure or `Error::Quant`
/// on shape mismatch / unsupported dimensions.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn gated_delta_step_gpu(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state_in: &Array,
    device: Device,
) -> Result<(Array, Array)> {
    // ── Shape validation ─────────────────────────────────────────────────────
    let qs = q.shape();
    let ks = k.shape();
    let vs = v.shape();
    let gs = g.shape();
    let bs = beta.shape();
    let ss = state_in.shape();

    if qs.len() != 4 || ks.len() != 4 || vs.len() != 4 {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: q/k/v must be 4-D, got q={qs:?} k={ks:?} v={vs:?}"
        )));
    }
    if gs.len() != 3 || bs.len() != 3 {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: g/beta must be 3-D [B,T,Hv], got g={gs:?} beta={bs:?}"
        )));
    }
    if ss.len() != 4 {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: state_in must be 4-D [B,Hv,Dv,Dk], got {ss:?}"
        )));
    }

    let batch = qs[0];
    let seq = qs[1];
    let hk = qs[2];
    let dk = qs[3];
    let hv = vs[2];
    let dv = vs[3];

    if ks[0] != batch || ks[1] != seq || ks[2] != hk || ks[3] != dk {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: k shape {ks:?} must match q shape {qs:?}"
        )));
    }
    if vs[0] != batch || vs[1] != seq {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: v[0..2] {:?} must match q[0..2] {:?}",
            &vs[..2],
            &qs[..2]
        )));
    }
    if gs[0] != batch || gs[1] != seq || gs[2] != hv {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: g shape {gs:?} must be [B={batch},T={seq},Hv={hv}]"
        )));
    }
    if bs[0] != batch || bs[1] != seq || bs[2] != hv {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: beta shape {bs:?} must be [B={batch},T={seq},Hv={hv}]"
        )));
    }
    if ss[0] != batch || ss[1] != hv || ss[2] != dv || ss[3] != dk {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: state_in shape {ss:?} must be [B={batch},Hv={hv},Dv={dv},Dk={dk}]"
        )));
    }
    if dk % 32 != 0 {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: Dk={dk} must be a multiple of 32"
        )));
    }
    if hv % hk != 0 {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: Hv={hv} must be a multiple of Hk={hk}"
        )));
    }

    // ── Dtype validation ─────────────────────────────────────────────────────
    let in_dtype = q.dtype();
    if k.dtype() != in_dtype || v.dtype() != in_dtype || beta.dtype() != in_dtype {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: q/k/v/beta dtype must match (got q={:?} k={:?} v={:?} beta={:?})",
            in_dtype,
            k.dtype(),
            v.dtype(),
            beta.dtype()
        )));
    }
    if g.dtype() != Dtype::F32 {
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: g must be f32, got {:?}",
            g.dtype()
        )));
    }
    let state_dtype = state_in.dtype();
    if state_dtype != Dtype::F32 {
        // The kernel supports any state dtype via StT, but we only ever store
        // f32 state in `LinearAttnCache`. Reject other dtypes loudly — if a
        // future caller wants different state precision they should add a
        // matching dtype to `LinearAttnCache` first.
        return Err(Error::Quant(format!(
            "gated_delta_step_gpu: state_in must be f32 (only currently used dtype), got {state_dtype:?}"
        )));
    }

    // T as a 0-D i32 input (mlx-lm passes T as a Python int which `to_array`
    // wraps into a scalar i32 array; 0-D arrays appear in the kernel as
    // `const constant int& T` — usable as a numeric literal).
    let t_bytes = seq.to_le_bytes();
    let t_input = Array::from_bytes(&t_bytes, &[], Dtype::I32)?;

    // ── Dispatch ─────────────────────────────────────────────────────────────
    let kernel = gated_delta_kernel_handle()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(q)?;
    invoke.add_input(k)?;
    invoke.add_input(v)?;
    invoke.add_input(g)?;
    invoke.add_input(beta)?;
    invoke.add_input(state_in)?;
    invoke.add_input(&t_input)?;

    invoke.add_output_shape(&[batch, seq, hv, dv], in_dtype)?;
    invoke.add_output_shape(&[batch, hv, dv, dk], state_dtype)?;

    invoke.set_template_dtype("InT", in_dtype)?;
    invoke.set_template_dtype("StT", state_dtype)?;
    invoke.set_template_int("Dk", dk)?;
    invoke.set_template_int("Dv", dv)?;
    invoke.set_template_int("Hk", hk)?;
    invoke.set_template_int("Hv", hv)?;

    // Grid (32, Dv, B*Hv) total threads; threadgroup (32, 4, 1).
    invoke.set_grid(32, dv, batch * hv)?;
    invoke.set_thread_group(32, 4, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    tracing::trace!(batch, seq, hk, hv, dk, dv, "gated_delta step dispatched");
    if outputs.len() < 2 {
        return Err(Error::Mlx(format!(
            "gated_delta_step_gpu: expected 2 outputs, got {}",
            outputs.len()
        )));
    }
    let state_out = outputs.remove(1);
    let y = outputs.remove(0);
    Ok((y, state_out))
}

// ── Ops-based prefill (reference / equivalence oracle) ────────────────────────
//
// NOT on the production path. Production prefill and decode both run
// `gated_delta_step_gpu` (the MSL kernel) at every T — see
// `qwen3_5_moe::gated_delta_net`. This function builds the same recurrence as a
// lazy MLX ops graph (one element-wise node per timestep) and mirrors mlx-lm's
// `gated_delta_ops` (the `use_kernel=False` fallback). It is retained as the
// equivalence oracle that `gated_delta_msl_tests` checks `gated_delta_step_gpu`
// against. It is NOT a faster prefill path: the per-step graph build explodes to
// ~184K nodes at T=256 and ~1.47M at T=2048 across the 30 GDN layers, which is
// exactly why the kernel — one dispatch with the T-loop in registers — wins.
//
// # Algorithm
//
// Mirrors mlx-lm `_gated_delta_step_ops` unrolled over T steps:
//
// ```text
// # Inputs (already GQA-repeated to Hv):
// # q, k: [B, T, Hv, Dk] (bf16/f16/f32)
// # v: [B, T, Hv, Dv]
// # g: [B, T, Hv] f32 — decay gate
// # beta: [B, T, Hv] (same dtype as q)
// # state:[B, Hv, Dv, Dk] f32
// #
// for t in 0..T:
// g_t = g[:, t].reshape([B, Hv, 1, 1]) # broadcast decay
// state = state * g_t # element-wise
// kv_mem = sum(state * k_t[..., None, :], -1) # [B, Hv, Dv]
// delta = (v_t - kv_mem) * beta_t[..., None] # [B, Hv, Dv]
// state = state + k_t[..., None, :] * delta[..., None] # rank-1 update
// y_t = sum(state * q_t[..., None, :], -1) # [B, Hv, Dv]
// y = stack(ys, axis=1) # [B, T, Hv, Dv]
// ```
//
// # GQA handling
//
// `q` and `k` arrive with shape `[B, T, Hk, Dk]` (UN-repeated). This
// function repeats them to `[B, T, Hv, Dk]` once before the loop —
// matches the implicit GQA repeat done by `gated_delta_step_gpu` via
// `hk_idx = hv_idx / (Hv / Hk)` inside the Metal kernel.

/// Ops-based GatedDeltaNet prefill: builds a lazy MLX graph over T timesteps.
///
/// Reference / test-only equivalence oracle — production no longer dispatches
/// this. Both prefill and decode run [`gated_delta_step_gpu`] (the MSL kernel)
/// at every T; this exists so `gated_delta_msl_tests` can assert the kernel
/// matches the ops-graph numerics within fp16 tolerance.
///
/// # Inputs
///
/// - `q`: `[B, T, Hk, Dk]` model dtype. UN-repeated; GQA handled here.
/// - `k`: `[B, T, Hk, Dk]` same dtype as q.
/// - `v`: `[B, T, Hv, Dv]` same dtype as q.
/// - `g`: `[B, T, Hv]` f32 — decay gate.
/// - `beta`: `[B, T, Hv]` same dtype as q.
/// - `state_in`: `[B, Hv, Dv, Dk]` f32 — recurrent state at t = 0.
///
/// # Outputs
///
/// - `y`: `[B, T, Hv, Dv]` same dtype as q.
/// - `state_out`: `[B, Hv, Dv, Dk]` f32.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn gated_delta_prefill_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state_in: &Array,
    device: Device,
) -> Result<(Array, Array)> {
    // ── Shape validation ─────────────────────────────────────────────────────
    let qs = q.shape();
    let ks = k.shape();
    let vs = v.shape();
    let gs = g.shape();
    let bs_shape = beta.shape();
    let ss = state_in.shape();

    if qs.len() != 4 || ks.len() != 4 || vs.len() != 4 {
        return Err(Error::Quant(format!(
            "gated_delta_prefill_ops: q/k/v must be 4-D, got q={qs:?} k={ks:?} v={vs:?}"
        )));
    }
    if gs.len() != 3 || bs_shape.len() != 3 {
        return Err(Error::Quant(format!(
            "gated_delta_prefill_ops: g/beta must be 3-D [B,T,Hv], got g={gs:?} beta={bs_shape:?}"
        )));
    }
    if ss.len() != 4 {
        return Err(Error::Quant(format!(
            "gated_delta_prefill_ops: state_in must be 4-D [B,Hv,Dv,Dk], got {ss:?}"
        )));
    }

    let batch = qs[0];
    let seq = qs[1];
    let hk = qs[2];
    let dk = qs[3];
    let hv = vs[2];
    let dv = vs[3];

    if hv % hk != 0 {
        return Err(Error::Quant(format!(
            "gated_delta_prefill_ops: Hv={hv} must be a multiple of Hk={hk}"
        )));
    }

    if ss[0] != batch || ss[1] != hv || ss[2] != dv || ss[3] != dk {
        return Err(Error::Quant(format!(
            "gated_delta_prefill_ops: state_in {ss:?} must be [B={batch},Hv={hv},Dv={dv},Dk={dk}]"
        )));
    }
    if g.dtype() != Dtype::F32 {
        return Err(Error::Quant(format!(
            "gated_delta_prefill_ops: g must be f32, got {:?}",
            g.dtype()
        )));
    }
    if state_in.dtype() != Dtype::F32 {
        return Err(Error::Quant(format!(
            "gated_delta_prefill_ops: state_in must be f32, got {:?}",
            state_in.dtype()
        )));
    }

    let in_dtype = q.dtype();

    // ── Compile cache lookup ─────────────────────────────────────────────────
    // Shape-aware `compile` (NOT `compile_shapeless`) caches the traced graph
    // for the specific (batch, T, Hk, Hv, Dk, Dv, in_dtype) tuple. First call
    // for a given shape pays the trace+compile cost (one-shot); subsequent
    // calls with the same shape replay the compiled Metal program directly,
    // skipping the Rust-side per-step graph build.
    let key = ShapeKey {
        batch,
        seq,
        hk,
        hv,
        dk,
        dv,
        in_dtype_tag: dtype_tag(in_dtype),
    };
    let cls = get_or_compile(&key, in_dtype, device)?;

    // Apply the compiled closure. Inputs are passed as raw `&Array`; outputs
    // are returned as a Vec<Array>.
    let outputs = cls
        .apply(&[q, k, v, g, beta, state_in])
        .map_err(|e| Error::Mlx(format!("gated_delta_prefill_ops: closure apply: {e}")))?;
    if outputs.len() != 2 {
        return Err(Error::Mlx(format!(
            "gated_delta_prefill_ops: expected 2 outputs from closure, got {}",
            outputs.len()
        )));
    }
    let mut iter = outputs.into_iter();
    let y_out = iter.next().expect("y output");
    let state_out = iter.next().expect("state output");
    Ok((y_out, state_out))
}

// ── Compile cache for the GDN prefill ops graph ──────────────────────────────
//
// The GDN prefill loop body is a sequential per-timestep graph build. Without
// caching, every prefill chunk traces ~T × 24 lazy ops nodes from Rust, then
// MLX compiles them to Metal at evaluation time. Caching the compiled Metal
// program by input shape eliminates the Rust-side trace cost for repeated
// chunks of the same T. This is the rMLX equivalent of mlx-lm's `@mx.compile`
// decorator on `_gated_delta_step_ops`.
//
// Use `compile` (shape-aware), NOT `compile_shapeless`: the inner loop emits
// `Slice` ops with concrete bounds (`0..ti..ti+1..`) that the symbolic
// shapeless tracer rejects (`Slice cannot infer output shapes`). Shape-aware
// compile retraces per shape; the cache key carries the shape so we get one
// compiled closure per (B, T, Hk, Hv, Dk, Dv, in_dtype) tuple.

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ShapeKey {
    batch: i32,
    seq: i32,
    hk: i32,
    hv: i32,
    dk: i32,
    dv: i32,
    /// Hash-friendly tag for `in_dtype` (`Dtype` doesn't impl `Hash`).
    in_dtype_tag: u8,
}

fn dtype_tag(d: Dtype) -> u8 {
    match d {
        Dtype::Bf16 => 0,
        Dtype::F16 => 1,
        Dtype::F32 => 2,
        Dtype::U8 => 3,
        Dtype::U32 => 4,
        Dtype::I32 => 5,
    }
}

static COMPILE_CACHE: OnceLock<Mutex<FxHashMap<ShapeKey, Arc<Closure>>>> = OnceLock::new();

fn compile_cache() -> &'static Mutex<FxHashMap<ShapeKey, Arc<Closure>>> {
    COMPILE_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn get_or_compile(key: &ShapeKey, in_dtype: Dtype, device: Device) -> Result<Arc<Closure>> {
    {
        let cache = compile_cache().lock().expect("compile cache poisoned");
        if let Some(cls) = cache.get(key) {
            return Ok(Arc::clone(cls));
        }
    }
    // Build & compile outside the lock to avoid blocking other shapes.
    let cls = build_compiled_closure(*key, in_dtype, device)?;
    let arc = Arc::new(cls);
    let mut cache = compile_cache().lock().expect("compile cache poisoned");
    // Another thread may have inserted in the meantime — that's fine, prefer
    // the existing entry to avoid divergent closures floating around.
    Ok(Arc::clone(cache.entry(*key).or_insert(arc)))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn build_compiled_closure(key: ShapeKey, in_dtype: Dtype, device: Device) -> Result<Closure> {
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 6 {
            return Err(Error::Mlx(format!(
                "gated_delta_prefill_ops closure: expected 6 inputs (q,k,v,g,beta,state), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let q = iter.next().expect("q");
        let k = iter.next().expect("k");
        let v = iter.next().expect("v");
        let g = iter.next().expect("g");
        let beta = iter.next().expect("beta");
        let state_in = iter.next().expect("state_in");
        gated_delta_prefill_ops_body(&q, &k, &v, &g, &beta, &state_in, key, in_dtype, device)
    });
    compile(raw).map_err(|e| Error::Mlx(format!("gated_delta_prefill_ops compile: {e}")))
}

/// Pure-ops body of the GDN prefill graph. Identical math to the previous
/// in-line implementation; extracted so the same code path runs both inside
/// the compiled closure and (theoretically) standalone. Inputs are NOT
/// shape-validated here — the public entry already validated.
fn gated_delta_prefill_ops_body(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state_in: &Array,
    key: ShapeKey,
    in_dtype: Dtype,
    device: Device,
) -> Result<Vec<Array>> {
    let ShapeKey {
        batch,
        seq,
        hk,
        hv,
        dk,
        dv,
        in_dtype_tag: _,
    } = key;
    let qk_repeat = hv / hk;

    // ── GQA repeat: [B, T, Hk, Dk] -> [B, T, Hv, Dk] ───────────────────────
    let (q_rep, k_rep) = if qk_repeat > 1 {
        let qr = repeat_axis(q, qk_repeat, 2, device)?;
        let kr = repeat_axis(k, qk_repeat, 2, device)?;
        (qr, kr)
    } else {
        (q.try_clone()?, k.try_clone()?)
    };

    // Cast q/k/v/beta to f32 (state is f32).
    let q_f32 = if q_rep.dtype() == Dtype::F32 {
        q_rep
    } else {
        q_rep.astype(Dtype::F32, device)?
    };
    let k_f32 = if k_rep.dtype() == Dtype::F32 {
        k_rep
    } else {
        k_rep.astype(Dtype::F32, device)?
    };
    let v_f32 = if v.dtype() == Dtype::F32 {
        v.try_clone()?
    } else {
        v.astype(Dtype::F32, device)?
    };
    let beta_f32 = if beta.dtype() == Dtype::F32 {
        beta.try_clone()?
    } else {
        beta.astype(Dtype::F32, device)?
    };

    // ── Per-timestep ops loop ────────────────────────────────────────────────
    let mut state = state_in.try_clone()?;
    let mut ys: Vec<Array> = Vec::with_capacity(seq as usize);

    for t in 0..(seq as usize) {
        let ti = t as i32;

        let k_t = k_f32
            .slice(
                &[0, ti, 0, 0],
                &[batch, ti + 1, hv, dk],
                &[1, 1, 1, 1],
                device,
            )?
            .reshape(&[batch, hv, dk], device)?;

        let v_t = v_f32
            .slice(
                &[0, ti, 0, 0],
                &[batch, ti + 1, hv, dv],
                &[1, 1, 1, 1],
                device,
            )?
            .reshape(&[batch, hv, dv], device)?;

        let q_t = q_f32
            .slice(
                &[0, ti, 0, 0],
                &[batch, ti + 1, hv, dk],
                &[1, 1, 1, 1],
                device,
            )?
            .reshape(&[batch, hv, dk], device)?;

        let g_t_flat = g
            .slice(&[0, ti, 0], &[batch, ti + 1, hv], &[1, 1, 1], device)?
            .reshape(&[batch, hv], device)?;
        let g_t = expand_dims(&g_t_flat, -1, device)?;
        let g_t = expand_dims(&g_t, -1, device)?;

        let beta_t_flat = beta_f32
            .slice(&[0, ti, 0], &[batch, ti + 1, hv], &[1, 1, 1], device)?
            .reshape(&[batch, hv], device)?;
        let beta_t = expand_dims(&beta_t_flat, -1, device)?;

        let k_t_exp = expand_dims(&k_t, -2, device)?;
        let q_t_exp = expand_dims(&q_t, -2, device)?;

        state = multiply(&state, &g_t, device)?;

        let kv_prod = multiply(&state, &k_t_exp, device)?;
        let kv_mem = sum_axis(&kv_prod, -1, device)?;

        let diff = subtract(&v_t, &kv_mem, device)?;
        let delta = multiply(&diff, &beta_t, device)?;

        let delta_exp = expand_dims(&delta, -1, device)?;

        let rank1 = multiply(&k_t_exp, &delta_exp, device)?;
        state = add(&state, &rank1, device)?;

        let yq = multiply(&state, &q_t_exp, device)?;
        let y_t = sum_axis(&yq, -1, device)?;

        ys.push(y_t);
    }

    let ys_refs: Vec<&Array> = ys.iter().collect();
    let y_stacked = stack_axis(&ys_refs, 1, device)?;

    let y_out = if y_stacked.dtype() == in_dtype {
        y_stacked
    } else {
        y_stacked.astype(in_dtype, device)?
    };

    Ok(vec![y_out, state])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "gated_delta_msl_tests.rs"]
mod gated_delta_msl_tests;
