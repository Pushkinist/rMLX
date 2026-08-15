// Rotor decode routing contract: which MSL kernel each rotor codec reaches
// through the public `KvCache::update_and_sdpa`, at the production decode
// shape (b = 1, head_dim in the fused-QK supported set).
//
// The rotor family does NOT share one kernel. `update_and_sdpa` tries the
// dedicated flash-decode-over-quant arms first, and only the codecs no such
// arm covers fall through to the head-major fused-QK shadow path:
//
//   Rotor3Sym  / Rotor4Sym      -> rotor_flash_decode_symv  (quant K + quant V)
//   RotorKOnly3 / RotorKOnly4   -> rotor_flash_decode       (quant K + bf16 V)
//   RotorK3Asym / RotorK4Asym   -> rotor fused-QK           (no flash arm exists)
//
// That ordering is deliberate: the flash arms read the packed rotor rings
// directly and keep no bf16 K mirror, while the fused-QK shadow path requires
// one. A regression that let a Sym / K-only codec fall through to fused-QK
// would silently double its KV footprint; one that stopped the asym codecs
// reaching fused-QK would strand them with no GPU decode kernel at all. Each
// case asserts the expected family fired AND that the other two did not, so
// both directions are caught.
//
// Every test here sets `RMLX_FUSED_QK=1`, because `fused_qk_enabled()` latches
// a process-global `OnceLock` on first read: one binary cannot observe both
// gate states. The gate-off behaviour (asym codecs reach no rotor kernel and
// serve from the warm bf16 mirror) is the shipped default and is documented in
// `docs/KV_QUANT.md`, not asserted here.
//
// `#[ignore]`-gated because they need the GPU; run via:
//   cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- \
//       --ignored --test-threads=1
//
// CLAUDE.md hard rule 8 (single MLX process): the test claims no port — the
// integration runner naturally serialises tests within one process. Run
// after `pkill ... && rm -f /tmp/rmlx.<port>.claim`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unusual_byte_groupings,
    clippy::indexing_slicing,
    unsafe_code,
    missing_docs
)]
//! Rotor decode routing-contract integration test.

use rmlx_kv_quant::rotor_flash_decode_msl::rotor_flash_decode_dispatch_count;
use rmlx_kv_quant::rotor_flash_decode_symv_msl::rotor_symv_flash_decode_dispatch_count;
use rmlx_kv_quant::rotor_fused_qk_msl::rotor_fused_qk_dispatch_count;
use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};

/// Which MSL kernel family a rotor codec's decode step is expected to reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kernel {
    /// `rotor_fused_qk` — head-major shadow, reached from `update_and_sdpa`
    /// only when no flash-decode arm claimed the codec first.
    FusedQk,
    /// `rotor_flash_decode` — quant K over a bf16 V mirror.
    FlashKOnly,
    /// `rotor_flash_decode_symv` — quant K over quant V.
    FlashSymV,
    /// No rotor kernel at all: the legacy bf16 SDPA fallback.
    None,
}

/// Snapshot of all three rotor decode counters.
#[derive(Debug, Clone, Copy)]
struct Counts {
    fused_qk: u64,
    flash_k_only: u64,
    flash_sym_v: u64,
}

fn counts() -> Counts {
    Counts {
        fused_qk: rotor_fused_qk_dispatch_count(),
        flash_k_only: rotor_flash_decode_dispatch_count(),
        flash_sym_v: rotor_symv_flash_decode_dispatch_count(),
    }
}

fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("from_bytes")
}

fn skip_if_no_gpu() -> bool {
    std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1")
}

fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let frac = (state >> 32) as u32 as f32 / u32::MAX as f32;
            frac * 2.0 - 1.0
        })
        .collect()
}

/// Drive prefill + one decode step on `codec` and assert the decode landed on
/// `expect` — and on nothing else.
fn assert_routes_to(codec: KvQuant, name: &str, expect: Kernel) {
    let device = Device::Gpu;
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads: i32 = kv_h * heads_per_kv;
    let head_dim: i32 = 128;
    let prefill_seq: i32 = 64;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let mut cache = KvCache::with_quant_max_seq(codec, 4096);
    cache.enter_prefill();
    let prefill_k_shape = [b, kv_h, prefill_seq, head_dim];
    let n_k: usize = prefill_k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(n_k, 0xB1B2_C3D4_E5F6_0020);
    let v_data = lcg_data(n_k, 0xB1B2_C3D4_E5F6_0021);
    let prefill_k = make_f32_array(&k_data, &prefill_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 k");
    let prefill_v = make_f32_array(&v_data, &prefill_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 v");
    let _full = cache
        .update(&prefill_k, &prefill_v, device)
        .expect("update prefill");
    cache.exit_prefill(device).expect("exit_prefill");
    assert_eq!(cache.offset(), prefill_seq, "{name}: offset after prefill");

    let decode_q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = decode_q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xB1B2_C3D4_E5F6_0022);
    let q_arr = make_f32_array(&q_data, &decode_q_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q");

    let new_k_shape = [b, kv_h, 1, head_dim];
    let new_n: usize = new_k_shape.iter().map(|&d| d as usize).product();
    let new_k = make_f32_array(&lcg_data(new_n, 0xB1B2_C3D4_E5F6_0023), &new_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_k");
    let new_v = make_f32_array(&lcg_data(new_n, 0xB1B2_C3D4_E5F6_0024), &new_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_v");

    let before = counts();
    let out_res = cache.update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device);
    let after = counts();
    let out = out_res.unwrap_or_else(|e| panic!("{name}: update_and_sdpa failed: {e}"));

    let d_fused = after.fused_qk - before.fused_qk;
    let d_k_only = after.flash_k_only - before.flash_k_only;
    let d_sym_v = after.flash_sym_v - before.flash_sym_v;
    eprintln!(
        "{name}: expect={expect:?} deltas fused_qk={d_fused} flash_k_only={d_k_only} \
         flash_sym_v={d_sym_v}"
    );

    let (want_fused, want_k_only, want_sym_v) = match expect {
        Kernel::FusedQk => (true, false, false),
        Kernel::FlashKOnly => (false, true, false),
        Kernel::FlashSymV => (false, false, true),
        Kernel::None => (false, false, false),
    };
    assert_kernel(name, "rotor_fused_qk", d_fused, want_fused);
    assert_kernel(name, "rotor_flash_decode", d_k_only, want_k_only);
    assert_kernel(name, "rotor_flash_decode_symv", d_sym_v, want_sym_v);

    out.eval().expect("eval rotor SDPA output");
    let _ = out.to_bytes().expect("rotor SDPA output materialised");
}

fn assert_kernel(name: &str, kernel: &str, delta: u64, wanted: bool) {
    if wanted {
        assert!(
            delta >= 1,
            "{name}: expected the {kernel} kernel to dispatch, but its counter did not \
             move (delta={delta}) — the decode step took a different arm of \
             `update_and_sdpa`"
        );
    } else {
        assert_eq!(
            delta, 0,
            "{name}: the {kernel} kernel dispatched (delta={delta}) but this codec is \
             routed elsewhere — the `update_and_sdpa` arm order changed"
        );
    }
}

fn set_fused_qk_on() {
    // SAFETY: process-global env var, single-threaded test enforced.
    unsafe {
        std::env::set_var("RMLX_FUSED_QK", "1");
    }
    unsafe {
        std::env::set_var("RMLX_FUSED_QK_MIN", "8");
    }
    // Ensure rotor QJL is OFF — the fused-QK kernel does not consume the QJL
    // residual, and a QJL-carrying store also keeps the flash arms out. QJL is
    // off by default, but set `0` explicitly so a stray `RMLX_ROTOR_QJL=1` in
    // the environment cannot move every codec onto the legacy path.
    unsafe {
        std::env::set_var("RMLX_ROTOR_QJL", "0");
    }
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor3_sym_routes_to_flash_decode_symv() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    assert_routes_to(KvQuant::Rotor3Sym, "Rotor3Sym", Kernel::FlashSymV);
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor4_sym_routes_to_flash_decode_symv() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    assert_routes_to(KvQuant::Rotor4Sym, "Rotor4Sym", Kernel::FlashSymV);
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_3_routes_to_flash_decode() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    assert_routes_to(KvQuant::RotorKOnly3, "RotorKOnly3", Kernel::FlashKOnly);
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_4_routes_to_flash_decode() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    assert_routes_to(KvQuant::RotorKOnly4, "RotorKOnly4", Kernel::FlashKOnly);
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor_k_asym_3_routes_to_fused_qk() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    assert_routes_to(
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        "RotorK3Asym(v=q4_g64)",
        Kernel::FusedQk,
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor_k_asym_4_routes_to_fused_qk() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    assert_routes_to(
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        "RotorK4Asym(v=q4_g64)",
        Kernel::FusedQk,
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor_qjl_on_falls_back_to_legacy_sdpa() {
    // With `RMLX_ROTOR_QJL=1` no rotor kernel may run: the flash arms refuse a
    // QJL-carrying store (`rotor_sym_store_uses_qjl`) and `try_fused_qk_dispatch`
    // short-circuits, because neither kernel reproduces the 1-bit K-side
    // residual. The legacy bf16 SDPA path takes over.
    if skip_if_no_gpu() {
        return;
    }
    // SAFETY: process-global env var, single-threaded test enforced.
    unsafe {
        std::env::set_var("RMLX_FUSED_QK", "1");
    }
    unsafe {
        std::env::set_var("RMLX_FUSED_QK_MIN", "8");
    }
    unsafe {
        std::env::set_var("RMLX_ROTOR_QJL", "1");
    }
    assert_routes_to(KvQuant::Rotor3Sym, "Rotor3Sym + QJL", Kernel::None);
    // Clean up env for subsequent tests.
    unsafe {
        std::env::remove_var("RMLX_ROTOR_QJL");
    }
}
