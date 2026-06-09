//! Permutation-equivalence tests for the Qwen3.5-MoE `SwitchMlp`.
//!
//! Same contract as the Gemma4 MoE tests: the sorted-dispatch prefill path
//! (`forward_sorted`) must produce outputs identical to the broadcast path
//! (`forward_broadcast`) for *any* routing. The sorted path permutes the
//! flattened `[n*tk]` expert indices, gathers/scatters x rows, and runs the
//! gathered quantized matmuls with `sorted_indices=true`. A single greedy A/B
//! is weak, so these tests assert equivalence directly on the two internal
//! paths with duplicate experts, skewed distributions, and token counts on
//! both sides of the `SORT_DISPATCH_THRESHOLD` (64).
//!
//! Exercises the real production path (`Linear::Quantized` → `gather_qmm`),
//! which is Metal-only — hence `#[ignore]`. Run with
//! `cargo test -p rmlx-models --lib qwen3_5_moe::moe -- --ignored`.

use super::{Linear, SwitchMlp};
use rmlx_mlx::{quantize, Array, Device, Dtype};

const DEV: Device = Device::Gpu;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }
    fn next_u(&mut self, modulo: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 33) as u32) % modulo
    }
}

#[allow(
    clippy::unwrap_used,
    reason = "test helper: from_bytes on a fixed-size in-memory buffer cannot fail"
)]
fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).unwrap()
}

#[allow(
    clippy::unwrap_used,
    reason = "test helper: from_bytes on a fixed-size in-memory buffer cannot fail"
)]
fn i32_arr(data: &[i32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::I32).unwrap()
}

#[allow(
    clippy::unwrap_used,
    reason = "test helper: CPU eval + to_bytes on a materialized array cannot fail"
)]
fn to_vec(a: &Array) -> Vec<f32> {
    let f = a.astype(Dtype::F32, Device::Cpu).unwrap();
    f.eval().unwrap();
    f.to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[allow(
    clippy::unwrap_used,
    reason = "test helper: quantize on a valid bf16 GPU array cannot fail"
)]
fn quant_linear(data: &[f32], shape: &[i32]) -> Linear {
    let w = f32_arr(data, shape).astype(Dtype::Bf16, DEV).unwrap();
    let (weight, scales, biases) = quantize(&w, GROUP_SIZE, BITS, DEV).unwrap();
    Linear::Quantized {
        weight,
        scales,
        biases: Some(biases),
        group_size: GROUP_SIZE,
        bits: BITS,
        mode: "affine".to_string(),
    }
}

/// gate/up: [ne, inter, hidden]; down: [ne, hidden, inter].
fn make_switch(rng: &mut Lcg, ne: i32, hidden: i32, inter: i32) -> SwitchMlp {
    let gate: Vec<f32> = (0..ne * inter * hidden)
        .map(|_| rng.next_f32() * 0.1)
        .collect();
    let up: Vec<f32> = (0..ne * inter * hidden)
        .map(|_| rng.next_f32() * 0.1)
        .collect();
    let down: Vec<f32> = (0..ne * hidden * inter)
        .map(|_| rng.next_f32() * 0.1)
        .collect();
    SwitchMlp {
        gate_proj: quant_linear(&gate, &[ne, inter, hidden]),
        up_proj: quant_linear(&up, &[ne, inter, hidden]),
        down_proj: quant_linear(&down, &[ne, hidden, inter]),
    }
}

fn make_routing(rng: &mut Lcg, n: i32, tk: i32, ne: i32, mode: u8) -> (Array, Array) {
    let mut idx = Vec::with_capacity((n * tk) as usize);
    for _ in 0..n {
        for _ in 0..tk {
            let e = if mode == 1 {
                if rng.next_u(10) < 8 {
                    (rng.next_u(2)) as i32
                } else {
                    rng.next_u(ne as u32) as i32
                }
            } else {
                rng.next_u(ne as u32) as i32
            };
            idx.push(e);
        }
    }
    let w: Vec<f32> = (0..n * tk).map(|_| 0.5 + rng.next_f32() * 0.25).collect();
    (i32_arr(&idx, &[n, tk]), f32_arr(&w, &[n, tk]))
}

#[allow(
    clippy::unwrap_used,
    reason = "test: forward paths on valid GPU arrays cannot fail under matching shapes"
)]
fn assert_paths_match(n: i32, tk: i32, ne: i32, hidden: i32, inter: i32, mode: u8, seed: u64) {
    const ATOL: f32 = 0.06;
    const RTOL: f32 = 0.03;
    let mut rng = Lcg(seed);
    let switch = make_switch(&mut rng, ne, hidden, inter);
    let xdata: Vec<f32> = (0..n * hidden).map(|_| rng.next_f32()).collect();
    let x = f32_arr(&xdata, &[n, hidden])
        .astype(Dtype::Bf16, DEV)
        .unwrap();
    let (idx, w) = make_routing(&mut rng, n, tk, ne, mode);
    let w = w.astype(Dtype::Bf16, DEV).unwrap();

    let bc = switch.forward_broadcast(&x, &idx, &w, DEV).unwrap();
    let sorted = switch.forward_sorted(&x, &idx, &w, DEV).unwrap();

    let bc_v = to_vec(&bc);
    let so_v = to_vec(&sorted);
    assert_eq!(bc_v.len(), so_v.len(), "shape mismatch");
    // Both paths run the identical gather_qmm/silu/multiply sequence; only the
    // row order into the expert weights differs. Equivalence is exact in real
    // arithmetic and exact-up-to-accumulation-reorder under bf16 int4. Assert
    // numpy-`allclose`-style (atol + rtol*|b|): atol absorbs the bf16 reorder
    // floor on near-zero elements, rtol the proportional drift on large ones.
    // A real index/scatter bug produces order-1 divergence on many elements,
    // far above this envelope.
    let mut worst = 0.0f32;
    for (a, b) in bc_v.iter().zip(&so_v) {
        let allowed = ATOL + RTOL * b.abs();
        worst = worst.max((a - b).abs() - allowed);
    }
    assert!(
        worst <= 0.0,
        "sorted vs broadcast diverge beyond atol+rtol*|b| by {worst} (n={n} tk={tk} ne={ne} mode={mode})"
    );
}

#[test]
#[ignore = "needs Metal device + single-MLX claim (gather_qmm is GPU-only)"]
fn sorted_eq_broadcast_below_threshold() {
    assert_paths_match(8, 4, 8, 256, 512, 0, 0xC0FFEE);
}

#[test]
#[ignore = "needs Metal device + single-MLX claim (gather_qmm is GPU-only)"]
fn sorted_eq_broadcast_above_threshold() {
    assert_paths_match(64, 4, 8, 256, 512, 0, 0xBADBEEF);
}

#[test]
#[ignore = "needs Metal device + single-MLX claim (gather_qmm is GPU-only)"]
fn sorted_eq_broadcast_skewed() {
    assert_paths_match(64, 4, 8, 256, 512, 1, 0x5EED);
}

#[test]
#[ignore = "needs Metal device + single-MLX claim (gather_qmm is GPU-only)"]
fn sorted_eq_broadcast_many_dup() {
    assert_paths_match(32, 8, 4, 256, 512, 1, 0xABCD1234);
}
