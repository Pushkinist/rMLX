//! Tests for load-time MSL precompile + codec classification.
//!
//! These are CPU-only / classification tests — they assert the precompile
//! entrypoint is a no-op for `none` and for CPU-hot-path codecs, and that the
//! `carries_msl` / `cpu_hot_path_reason` classifiers are internally consistent.
//! The actual Metal warm dispatch is exercised by the on-device smoke test
//! (no Metal context in unit tests).

use super::*;
use rmlx_mlx::Device;

#[test]
fn precompile_is_noop_on_cpu_device() {
    // Device::Cpu → always a no-op, regardless of codec.
    for kq in [
        KvQuant::None,
        KvQuant::K8V4,
        KvQuant::Rotor3,
        KvQuant::Iso3,
        KvQuant::RotKTq4V,
    ] {
        assert!(
            precompile_kv_codec_msl(kq, 256, 1, Device::Cpu).is_ok(),
            "precompile on CPU device must be a no-op Ok for {kq}"
        );
    }
}

#[test]
fn none_carries_no_msl() {
    assert!(!KvQuant::None.carries_msl(), "bf16 KV carries no shader");
    assert!(
        KvQuant::None.cpu_hot_path_reason().is_none(),
        "None is not a CPU-fallback codec — it is a no-op bf16 path"
    );
}

#[test]
fn metal_codecs_carry_msl_and_are_not_cpu_fallback() {
    // Codecs whose hot path is genuinely Metal: carry MSL, not CPU-fallback.
    for kq in [
        KvQuant::K8V4,
        KvQuant::K8V8,
        KvQuant::Planar,
        KvQuant::RotKTq4V,
        KvQuant::TurboSym4,
    ] {
        assert!(kq.carries_msl(), "{kq} must carry MSL");
        assert!(
            kq.cpu_hot_path_reason().is_none(),
            "{kq} runs on Metal — must NOT be flagged CPU-fallback"
        );
    }
}

#[test]
fn v_only_iso_rotor_families_are_cpu_fallback_with_reason() {
    // V-only iso / rotor codecs and the rotor-K-asym variants: `update_iso3*` /
    // `update_rotor3*` early-return to the bf16 decode seed, so the iso/rotor
    // encode that runs (at prefill) is CPU. They carry MSL (q8 K-side) but must
    // report a CPU-hot-path reason. This is grounded in the dispatcher, not by
    // fiat — flip the verdict and this test must flip with it.
    // Note: the **symmetric** codecs (`Iso{3,4}Sym`, `Rotor{3,4}Sym`) are NOT in
    // this list — their decode is the quant-V flash kernel over both packed
    // rings, so their hot path is Metal. Those verdicts are checked in
    // `iso_symmetric_families_are_metal` /
    // `rotor_symmetric_families_are_metal_when_qjl_off`.
    for kq in [
        KvQuant::Iso3,
        KvQuant::Iso4,
        KvQuant::Rotor3,
        KvQuant::Rotor4,
        KvQuant::RotorK3Asym {
            v_bits: 8,
            v_group_size: 128,
        },
        KvQuant::RotorK4Asym {
            v_bits: 8,
            v_group_size: 128,
        },
    ] {
        assert!(
            kq.carries_msl(),
            "{kq} K-side is q8_0 MSL — carries_msl must be true"
        );
        assert!(
            kq.cpu_hot_path_reason().is_some(),
            "{kq} (V-only iso/rotor or rotor-K-asym) must be flagged as a CPU-hot-path codec"
        );
    }
}

#[test]
#[allow(unsafe_code)]
fn rotor_symmetric_families_are_metal_when_qjl_off() {
    // `Rotor3Sym` / `Rotor4Sym` have NO bf16 decode-seed early-return: with QJL
    // off (the default) decode is the quant-V flash kernel over both packed
    // rings, so the verdict must be `None` (Metal) — mirror of the K-only iso
    // test. A QJL-carrying store still keeps the CPU dequant path; that branch
    // is gated on the global toggle and not exercised here (QJL is off by
    // default). Grounded in the dispatcher, not by fiat.
    //
    // The whole body is a *reader* of the process-global QJL gate — the explicit
    // check below and every `cpu_hot_path_reason()` call in the loop — so it must
    // serialize against the tests that drive `RMLX_ROTOR_QJL`, or it samples one
    // of their in-flight mutations and fails intermittently.
    //
    // The lock serializes access; it does not reset state. So establish the
    // QJL-off precondition rather than asserting one this test never set — an
    // inherited `RMLX_ROTOR_QJL=1` (a developer's export, `RMLX_ROTOR_QJL=1 make
    // test`) would otherwise fail here with a message blaming the test. The
    // guard restores the prior value on drop.
    let _guard = crate::test_utils::env_lock();
    if crate::rotor_qjl::rotor_qjl_cli_is_set() {
        return;
    }
    // SAFETY: env lock held — no concurrent env reader/writer.
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
    assert!(
        !crate::rotor_qjl::rotor_qjl_enabled(),
        "QJL must be off once the env override is cleared and no CLI override is installed"
    );
    for kq in [KvQuant::Rotor3Sym, KvQuant::Rotor4Sym] {
        assert!(
            kq.carries_msl(),
            "{kq} carries MSL (rotor K + quant-V flash kernels)"
        );
        assert!(
            kq.cpu_hot_path_reason().is_none(),
            "{kq} decodes through the quant-V flash kernel with QJL off — must be Metal (None)"
        );
    }
}

#[test]
fn iso_symmetric_families_are_metal() {
    // `Iso3Sym` / `Iso4Sym` have NO bf16 decode-seed early-return: decode is the
    // quant-V flash kernel over both packed iso rings, so the verdict must be
    // `None` (Metal). Iso carries no QJL sideband, so there is no CPU-fallback
    // gate — unlike the rotor sym mirror. Grounded in the dispatcher, not by fiat.
    for kq in [KvQuant::Iso3Sym, KvQuant::Iso4Sym] {
        assert!(
            kq.carries_msl(),
            "{kq} carries MSL (iso K + quant-V flash kernels)"
        );
        assert!(
            kq.cpu_hot_path_reason().is_none(),
            "{kq} decodes through the quant-V flash kernel — must be Metal (None)"
        );
    }
}

#[test]
fn k_only_iso_families_are_metal_on_gpu() {
    // `IsoKOnly3/4` have NO bf16 decode-seed early-return: `update_iso_k_only_*`
    // dispatches the iso{3,4} MSL kernel every decode step on GPU, so the verdict
    // must be `None` (Metal) — not classified CPU-hot-path. They are still skipped
    // by the q8 precompile because their K kernel is iso MSL, not q8
    // (`is_k_only_iso_rotor`).
    for kq in [KvQuant::IsoKOnly3, KvQuant::IsoKOnly4] {
        assert!(
            kq.cpu_hot_path_reason().is_none(),
            "{kq} dispatches the iso K MSL kernel every decode step — must be Metal (None)"
        );
        assert!(
            kq.is_k_only_iso_rotor(),
            "{kq} is a K-only iso codec — precompile must skip the q8 warm for it"
        );
    }
}

#[test]
#[allow(unsafe_code)]
fn k_only_rotor_families_are_qjl_aware() {
    // `RotorKOnly3/4` have NO bf16 early-return; the GPU K encode is gated on
    // `!rotor_qjl_enabled()`. The verdict tracks the live QJL gate: QJL on (the
    // default) → CPU (`Some`); QJL off → Metal (`None`). Drive both states via
    // the same env the dispatcher reads, so the test is tied to the dispatcher.
    let _guard = crate::test_utils::env_lock();
    // The codec identity (K-only rotor) is QJL-independent — assert it always.
    for kq in [KvQuant::RotorKOnly3, KvQuant::RotorKOnly4] {
        assert!(
            kq.is_k_only_iso_rotor(),
            "{kq} is a K-only rotor codec — precompile must skip the q8 warm for it"
        );
    }
    // `rotor_qjl_enabled()` consults the CLI OnceLock before the env var; if a
    // prior test in this process installed the CLI override the env toggle below
    // is shadowed, so skip the QJL-state assertions (same pattern as
    // `rotor_qjl_default_is_on`).
    if crate::rotor_qjl::rotor_qjl_cli_is_set() {
        return;
    }
    for kq in [KvQuant::RotorKOnly3, KvQuant::RotorKOnly4] {
        // SAFETY: env lock held — no concurrent env reader/writer.
        unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };
        assert!(
            kq.cpu_hot_path_reason().is_some(),
            "{kq} with QJL on must be a CPU-hot-path codec (QJL forces K append onto CPU)"
        );
        unsafe { std::env::set_var("RMLX_ROTOR_QJL", "off") };
        assert!(
            kq.cpu_hot_path_reason().is_none(),
            "{kq} with QJL off dispatches the rotor K MSL kernel — must be Metal (None)"
        );
    }
    // No manual restore: `_guard` puts `RMLX_ROTOR_QJL` back on drop, including
    // while unwinding from either assertion above. Restoring here would only
    // cover the path that does not need it.
}

#[test]
fn precompile_skips_cpu_hot_path_codecs() {
    // Even on a GPU device the precompile is a documented no-op for the V-only
    // iso/rotor CPU-fallback codecs (nothing to warm). We can only assert the
    // classifier off-device, which the function consults to take the skip branch.
    for kq in [KvQuant::Rotor3, KvQuant::Iso3, KvQuant::Iso4] {
        assert!(
            kq.cpu_hot_path_reason().is_some(),
            "{kq} drives the precompile cpu_hot_path skip branch"
        );
    }
    // K-only iso/rotor codecs are Metal (None from cpu_hot_path_reason) but the
    // q8 precompile still skips them via `is_k_only_iso_rotor`.
    for kq in [KvQuant::IsoKOnly3, KvQuant::IsoKOnly4] {
        assert!(
            kq.is_k_only_iso_rotor(),
            "{kq} drives the precompile is_k_only_iso_rotor skip branch"
        );
    }
}

/// Faithful regression for the K8V8 `exit_prefill` stream-safety fix: run the
/// exact q8_0 quantize + `Array::eval()` (the op K8V8 `exit_prefill` runs, and
/// the one whose lazy eval faulted with "There is no Stream(cpu, N) in current
/// thread") on a **freshly-spawned non-main worker thread**, after the worker
/// registers its default streams via `ensure_cpu_default_stream` /
/// `ensure_gpu_default_stream` — exactly as the `arch::generate_greedy` entry
/// point does before every generation.
///
/// The graph is built AND evaluated on the same worker thread (the supported
/// path); it must succeed. Requires a Metal GPU, so it is ignored by default and
/// run with `-- --ignored` on a GPU host.
#[test]
#[ignore = "requires Metal GPU; run with `-- --ignored` in a GPU-capable environment"]
fn k8v8_q8_quantize_eval_on_worker_thread() {
    use rmlx_mlx::{Array, Dtype};

    let ok = std::thread::spawn(|| {
        // Register the worker's default CPU + GPU streams, as the generate
        // entry point does. Idempotent.
        rmlx_mlx::ensure_cpu_default_stream();
        rmlx_mlx::ensure_gpu_default_stream();

        // Shape mirrors a small K8V8 prefill K/V slice: [B=1, kv_h=1, S, D=256].
        // 256 elements/token → group=128 aligned.
        let shape = [1i32, 1, 8, 256];
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let warm = Array::from_bytes(&vec![0u8; n * 2], &shape, Dtype::Bf16).unwrap();

        // The exact exit_prefill K8V8 op: GPU q8 quantize, then eval.
        let (codes, scales) = crate::q8_msl::q8_quantize_gpu(&warm, Device::Gpu).unwrap();
        codes.eval().unwrap();
        scales.eval().unwrap();
        true
    })
    .join()
    .expect("worker thread panicked");
    assert!(
        ok,
        "K8V8 q8 quantize + eval must succeed on a spawned worker thread"
    );
}
