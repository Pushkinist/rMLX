// `RMLX_KV_MAX_SEQ_HARD_CAP` end-to-end smoke.
//
// The hard-cap branch of `ensure_prefill_capacity` is read once per process
// via a `OnceLock`. The unit test module in
// `crates/rmlx-kv-quant/src/kvcache/prefill_grow_tests.rs` deliberately
// leaves the env var unset because `OnceLock` is process-global and parallel
// unit tests share the same address space; setting it from one test would
// leak into siblings. Integration tests live in their own binary, so the
// `OnceLock` is isolated — no `--test-threads=1` needed and no `#[ignore]`.
//
// This test sets `RMLX_KV_MAX_SEQ_HARD_CAP=128` and drives a `KvQuant::None`
// cache (bf16, no MSL kernel) at `Device::Cpu` with a single 200-token
// prefill chunk. The hard cap fires before any allocation; the error returns
// the typed `Error::KvHardCapExceeded { requested: 200, cap: 128 }`.
//
// CLAUDE.md hard rule 1 (Apple-Silicon-only build) — CPU dispatch is still
// valid for compute paths that have no GPU-only kernel.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::wildcard_enum_match_arm,
    unsafe_code,
    missing_docs
)]

use rmlx_core::error::Error;
use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};

const TEST_KV_H: i32 = 2;
const TEST_HEAD_DIM: i32 = 64;
const TEST_INITIAL_MAX_SEQ: i32 = 32;
const HARD_CAP: i32 = 128;
const PREFILL_LEN: i32 = 200;

fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: Apple-Silicon-only build (CLAUDE.md hard rule 1); f32 is 4-byte
    // LE on this target. `data` is borrowed read-only and the bytes are
    // copied into MLX before the borrow ends.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("Array::from_bytes")
}

#[test]
fn hard_cap_rejects_prefill_above_cap() {
    // SAFETY: integration-test binary; `OnceLock` is per-process so this
    // assignment only affects this test executable. No concurrent writer.
    unsafe {
        std::env::set_var("RMLX_KV_MAX_SEQ_HARD_CAP", HARD_CAP.to_string());
    }

    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, TEST_INITIAL_MAX_SEQ);
    cache.enter_prefill();

    let chunk_shape = [1_i32, TEST_KV_H, PREFILL_LEN, TEST_HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&vec![0.1_f32; n_chunk], &chunk_shape);
    let v = f32_arr(&vec![0.2_f32; n_chunk], &chunk_shape);

    let err = cache
        .update(&k, &v, device)
        .expect_err("hard cap: prefill must fail above RMLX_KV_MAX_SEQ_HARD_CAP");

    match err {
        Error::KvHardCapExceeded { requested, cap } => {
            assert_eq!(requested, PREFILL_LEN, "requested seq length should match");
            assert_eq!(cap, HARD_CAP, "cap should match RMLX_KV_MAX_SEQ_HARD_CAP");
        }
        other => panic!(
            "hard cap: expected Error::KvHardCapExceeded {{ requested: {PREFILL_LEN}, cap: {HARD_CAP} }}, got: {other:?}",
        ),
    }
}
