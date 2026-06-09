//! Behaviour pins for the fused-quant shared-KV decode share
//! (`update_and_sdpa_returning_shared_kv`).
//!
//! These tests run on a Metal GPU (`#[ignore]`, `--test-threads=1`) because the
//! Mixed path's `quantized_matmul` / `mx.quantize` require Metal. They pin the
//! variant the producer surfaces to a Gemma4 shared-KV consumer:
//!
//! * PREFILL → [`SharedKvOut::Bf16`] (quant store not built until
//!   `exit_prefill`; sharing bf16 during prefill is correct and cheap).
//! * DECODE  → [`SharedKvOut::MixedQuant`] carrying the live quant 3-tuples and
//!   the codec's quant params, so the consumer attends the quant store directly
//!   (no bf16 re-inflation of the O(ctx) global KV).
//!
//! The non-shared Mixed decode hot path (Bonsai / Qwen3) is unaffected — it does
//! not call this method.

use super::core::KvCache;
use crate::kvcache::SharedKvOut;
use crate::KvQuant;
use rmlx_mlx::{Array, Device, Dtype};

#[allow(
    clippy::unwrap_used,
    reason = "test helper: from_bytes only fails on a programmer-error shape mismatch which would abort the test anyway"
)]
fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: Apple-Silicon-only build (CLAUDE.md Hard rule 1); f32 is 4-byte
    // LE on this target. `data` is borrowed read-only and copied into MLX
    // before the borrow ends.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).unwrap()
}

const KV_H: i32 = 8;
const Q_H: i32 = 16;
const HEAD_DIM: i32 = 128;
const MAX_SEQ: i32 = 512;
const PREFILL_SEQ: i32 = 64;

/// Producer surfaces `Bf16` at prefill and `MixedQuant` at decode, with the
/// quant 3-tuples spanning the full accumulated prefix+1 and carrying the
/// codec's quant params.
#[test]
#[ignore = "requires Metal GPU; run with --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts via unwrap/expect on the happy path; a failure aborts the test which is the intent"
)]
fn shared_kv_quant_surfaces_mixedquant_at_decode() {
    let device = Device::Gpu;
    // K8V4 Mixed: K 8-bit, V 4-bit, group size 64.
    let quant = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    let mut cache = KvCache::with_quant_max_seq(quant, MAX_SEQ);

    cache.enter_prefill();
    let pref_kv = [1_i32, KV_H, PREFILL_SEQ, HEAD_DIM];
    let n_kv: usize = pref_kv.iter().map(|&d| d as usize).product();
    let k_pref = f32_arr(&vec![0.05_f32; n_kv], &pref_kv);
    let v_pref = f32_arr(&vec![0.07_f32; n_kv], &pref_kv);
    let pref_q = [1_i32, Q_H, PREFILL_SEQ, HEAD_DIM];
    let n_q: usize = pref_q.iter().map(|&d| d as usize).product();
    let q_pref = f32_arr(&vec![0.03_f32; n_q], &pref_q);

    let (_out, shared_pref) = cache
        .update_and_sdpa_returning_shared_kv(&q_pref, &k_pref, &v_pref, 1.0, "causal", None, device)
        .expect("prefill shared-kv");
    assert!(
        matches!(shared_pref, SharedKvOut::Bf16(_, _)),
        "prefill must surface bf16 (quant store not built until exit_prefill)"
    );
    cache.exit_prefill(device).expect("exit_prefill");

    // Decode step 1.
    let step_kv = [1_i32, KV_H, 1, HEAD_DIM];
    let n_s: usize = step_kv.iter().map(|&d| d as usize).product();
    let k_step = f32_arr(&vec![0.09_f32; n_s], &step_kv);
    let v_step = f32_arr(&vec![0.11_f32; n_s], &step_kv);
    let step_q = [1_i32, Q_H, 1, HEAD_DIM];
    let n_qs: usize = step_q.iter().map(|&d| d as usize).product();
    let q_step = f32_arr(&vec![0.02_f32; n_qs], &step_q);

    let (_out2, shared_dec) = cache
        .update_and_sdpa_returning_shared_kv(&q_step, &k_step, &v_step, 1.0, "", None, device)
        .expect("decode shared-kv");

    match shared_dec {
        SharedKvOut::MixedQuant {
            k,
            v,
            k_bits,
            v_bits,
            ..
        } => {
            // Quant K codes span the full accumulated length (prefix + 1).
            assert_eq!(
                k.codes.shape()[2],
                PREFILL_SEQ + 1,
                "quant K codes must span [0:offset] over the full prefix+1"
            );
            assert_eq!(
                v.codes.shape()[2],
                PREFILL_SEQ + 1,
                "quant V codes must span [0:offset] over the full prefix+1"
            );
            assert_eq!(k_bits, 8, "K8V4: K is 8-bit");
            assert_eq!(v_bits, 4, "K8V4: V is 4-bit");
        }
        SharedKvOut::Bf16(_, _) => {
            panic!("decode must surface MixedQuant (fused-quant share), not bf16")
        }
    }
}
