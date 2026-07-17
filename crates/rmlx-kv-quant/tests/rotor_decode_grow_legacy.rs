// Decode-side `max_seq` growth on the rotor codec's **legacy** (non-fused) path.
//
// `KvCache::update_and_sdpa` sends a rotor K-only decode step to the fused
// flash-decode helper only when `rotor_flash_shape_ok` accepts the shape, which
// requires a power-of-two `head_dim` (the kernel tree-reduces over `head_dim`
// threads). Any other `head_dim` — 96 and 192 are ordinary attention shapes —
// falls through to the legacy `update_rotor_k_only_{3,4}` path instead.
//
// That fall-through still feeds the same GPU ring, so it is bound by the same
// provisioned `max_seq` and must grow with it. A fix applied only to the fused
// helper would be shape-dependent: correct at `head_dim=128`, broken at 96.
// CLAUDE.md hard rule 10 requires codec paths to key off codec + shape, never to
// be correct only for some shapes.
//
// # Why an integration binary
//
// The legacy path gates its GPU encode on the process-global rotor-QJL toggle
// (default **on**, which routes to the CPU append and never touches the ring).
// A unit test that flipped it would leak into every sibling in the same address
// space. Integration tests get their own process, so `install_rotor_qjl(false)`
// is isolated here — the same reasoning `tests/prefill_hard_cap.rs` documents
// for the hard-cap `OnceLock`. Every test in this binary wants QJL off, so the
// first-wins `OnceLock` is deterministic.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::wildcard_enum_match_arm,
    unsafe_code,
    missing_docs
)]

use rmlx_kv_quant::rotor_qjl::install_rotor_qjl;
use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};

const KV_H: i32 = 2;
const N_Q_HEADS: i32 = 8;
/// Not a power of two → `rotor_flash_shape_ok` rejects → legacy path.
const NON_POW2_HEAD_DIM: i32 = 96;
/// Small enough to saturate cheaply.
const MAX_SEQ: i32 = 128;

fn skip_gpu() -> bool {
    std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1")
}

/// Deterministic filler; mirrors `test_utils::lcg_data` closely enough for a
/// shape/plumbing test (integration binaries cannot reach the crate-private
/// helper).
fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("f32_array")
}

/// Prefill `n` tokens, then run `steps` single-token decode steps.
fn prefill_then_decode(quant: KvQuant, head_dim: i32, n: i32, steps: u64) -> Result<i32, String> {
    let device = Device::Gpu;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut cache = KvCache::with_quant_max_seq(quant, MAX_SEQ);

    let pf = (n * KV_H * head_dim) as usize;
    let k = f32_array(&lcg(pf, 1), &[1, KV_H, n, head_dim]);
    let v = f32_array(&lcg(pf, 2), &[1, KV_H, n, head_dim]);
    let q = f32_array(
        &lcg((n * N_Q_HEADS * head_dim) as usize, 3),
        &[1, N_Q_HEADS, n, head_dim],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .map_err(|e| format!("prefill: {e}"))?;
    cache
        .exit_prefill(device)
        .map_err(|e| format!("exit: {e}"))?;

    for step in 0..steps {
        let one = (KV_H * head_dim) as usize;
        let k1 = f32_array(&lcg(one, 10 + step), &[1, KV_H, 1, head_dim]);
        let v1 = f32_array(&lcg(one, 20 + step), &[1, KV_H, 1, head_dim]);
        let q1 = f32_array(
            &lcg((N_Q_HEADS * head_dim) as usize, 30 + step),
            &[1, N_Q_HEADS, 1, head_dim],
        );
        let out = cache
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .map_err(|e| format!("decode step {step}: {e}"))?;
        out.eval().map_err(|e| format!("eval step {step}: {e}"))?;
    }
    Ok(cache.offset())
}

/// A saturated prompt must decode on the legacy path too, not just the fused one.
///
/// `head_dim=96` is the whole point: it is the shape the flash kernel cannot
/// take, so the step routes through `update_rotor_k_only_3`. Both paths feed the
/// same ring and are bound by the same `max_seq`.
fn legacy_path_grows(quant: KvQuant, label: &str) {
    if skip_gpu() {
        return;
    }
    install_rotor_qjl(false);

    let offset = prefill_then_decode(quant, NON_POW2_HEAD_DIM, MAX_SEQ, 32)
        .unwrap_or_else(|e| panic!("{label}: legacy-path decode past a saturated max_seq: {e}"));
    assert_eq!(
        offset,
        MAX_SEQ + 32,
        "{label}: every legacy-path decode step must land"
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_decode_grow_legacy -- --ignored --test-threads=1"]
fn rotor3_legacy_path_decode_grows_past_a_saturated_max_seq() {
    legacy_path_grows(KvQuant::RotorKOnly3, "rotor3");
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_decode_grow_legacy -- --ignored --test-threads=1"]
fn rotor4_legacy_path_decode_grows_past_a_saturated_max_seq() {
    legacy_path_grows(KvQuant::RotorKOnly4, "rotor4");
}

/// The fused path at a power-of-two `head_dim` must stay green alongside it —
/// pins that the growth is shape-independent rather than traded from one shape
/// to another.
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_decode_grow_legacy -- --ignored --test-threads=1"]
fn rotor3_fused_path_still_grows_at_power_of_two_head_dim() {
    if skip_gpu() {
        return;
    }
    install_rotor_qjl(false);

    let offset = prefill_then_decode(KvQuant::RotorKOnly3, 128, MAX_SEQ, 32)
        .unwrap_or_else(|e| panic!("fused-path decode past a saturated max_seq: {e}"));
    assert_eq!(
        offset,
        MAX_SEQ + 32,
        "every fused-path decode step must land"
    );
}
