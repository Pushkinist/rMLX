use super::{
    resolve_dispatch_policy, FusedQkMode, PlanarFlashDecodeMode, RotKFusedMode, SparseAttnMode,
    TurboFlashMode,
};
use rmlx_core::DispatchPolicy;

// ── SSD recorder wiring smoke ─────────────────────────────
//
// Full bootstrap verification (i.e. confirming `ssd_event_recorder()` returns
// `Some` after `run_serve`) is structurally hard: `run_serve` starts a
// blocking tokio runtime that loops forever, so we cannot call it in a unit
// test and then inspect state. Instead we test the two smaller units:
//
// 1. `set_ssd_event_recorder` is callable and does not panic.
// The OnceLock first-writer wins — subsequent calls in the same process
// are silent no-ops, which is also the desired production behavior.
// 2. `register_ssd_prom_hooks` is callable and does not panic.
//
// The end-to-end proof that events actually fire comes from Part 3: the live
// run on gemma-4-e2b which queries the `events` table and confirms
// at least one `SsdSpill` row and one `SsdHydrate` row were recorded.

#[test]
fn set_ssd_event_recorder_is_callable() {
    use std::sync::Arc;
    // Use a tempdir DB so this test does not collide with the workspace DB.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let db = dir.path().join("test.db");
    let rec = rmlx_metrics::events::EventRecorder::open_at(&db, "test-ssd-wiring")
        .expect("open EventRecorder");
    // set_ssd_event_recorder is a OnceLock: first call succeeds, subsequent
    // calls in the same process are silent no-ops. Both cases are non-panicking.
    rmlx_kv_ssd::set_ssd_event_recorder(Arc::new(rec));
    // No assertion needed beyond "did not panic".
}

#[test]
fn register_ssd_prom_hooks_is_callable() {
    // register_ssd_prom_hooks installs two OnceLock closures; subsequent
    // calls in the same process are silent no-ops.
    rmlx_server::register_ssd_prom_hooks();
    // No assertion needed beyond "did not panic".
}

// ── Kernel-gate precedence ───────────────────────────────────────────────────
//
// Every gate resolves from exactly two inputs: the flag and the
// environment-derived fallback. `resolve_dispatch_policy` takes both as
// arguments, so these tests assert on the resolved `DispatchPolicy` — the
// value the KV caches actually capture — instead of on the process
// environment, and never mutate global state.
//
// The matrix under test, identical for all five tri-state gates:
//
// | flag   | env unset | env `=1` |
// |--------|-----------|----------|
// | `on`   | ON        | ON       |
// | `off`  | OFF       | OFF (hard override) |
// | `auto` | OFF       | ON (back-compat opt-in) |

/// All-off environment: no gate variable set.
fn env_clear() -> DispatchPolicy {
    DispatchPolicy::default()
}

/// Every gate variable exported as `=1`.
fn env_all_set() -> DispatchPolicy {
    DispatchPolicy {
        fused_qk: true,
        sparse_attn: true,
        turbo_flash: true,
        turbo_flash_lock: true,
        planar_flash_decode: true,
        rot_k_fused: true,
        ..DispatchPolicy::default()
    }
}

/// Resolve with every flag at its clap default (`auto`, lock off).
fn resolve_defaults(env: DispatchPolicy) -> DispatchPolicy {
    resolve_dispatch_policy(
        FusedQkMode::Auto,
        SparseAttnMode::Auto,
        TurboFlashMode::Auto,
        false,
        PlanarFlashDecodeMode::Auto,
        RotKFusedMode::Auto,
        env,
    )
}

/// Resolve with every flag forced `on` (lock on).
fn resolve_all_on(env: DispatchPolicy) -> DispatchPolicy {
    resolve_dispatch_policy(
        FusedQkMode::On,
        SparseAttnMode::On,
        TurboFlashMode::On,
        true,
        PlanarFlashDecodeMode::On,
        RotKFusedMode::On,
        env,
    )
}

/// Resolve with every flag forced `off` (lock off).
fn resolve_all_off(env: DispatchPolicy) -> DispatchPolicy {
    resolve_dispatch_policy(
        FusedQkMode::Off,
        SparseAttnMode::Off,
        TurboFlashMode::Off,
        false,
        PlanarFlashDecodeMode::Off,
        RotKFusedMode::Off,
        env,
    )
}

/// Every gate as a `(name, extractor)` pair, so a test covers all of them and
/// names the one that failed.
const GATES: &[(&str, fn(&DispatchPolicy) -> bool)] = &[
    ("fused_qk", |p| p.fused_qk),
    ("sparse_attn", |p| p.sparse_attn),
    ("turbo_flash", |p| p.turbo_flash),
    ("turbo_flash_lock", |p| p.turbo_flash_lock),
    ("planar_flash_decode", |p| p.planar_flash_decode),
    ("rot_k_fused", |p| p.rot_k_fused),
];

#[test]
fn auto_with_a_clear_environment_resolves_every_gate_off() {
    let p = resolve_defaults(env_clear());
    for (name, get) in GATES {
        assert!(!get(&p), "{name}: auto + clear env must resolve OFF");
    }
}

#[test]
fn auto_honours_an_environment_opt_in() {
    // Back-compat: an exported `RMLX_<GATE>=1` still turns the kernel on when
    // the flag is left at `auto`.
    let p = resolve_defaults(env_all_set());
    for (name, get) in GATES {
        assert!(get(&p), "{name}: auto + env opt-in must resolve ON");
    }
}

#[test]
fn on_forces_every_gate_on_regardless_of_environment() {
    for env in [env_clear(), env_all_set()] {
        let p = resolve_all_on(env);
        for (name, get) in GATES {
            assert!(get(&p), "{name}: flag=on must resolve ON");
        }
    }
}

#[test]
fn off_is_a_hard_override_of_an_environment_opt_in() {
    // The reason `off` exists: a stale `=1` in a shell, a CI job or a profile
    // must not survive an explicit `off`.
    let p = resolve_all_off(env_all_set());
    for (name, get) in GATES {
        if *name == "turbo_flash_lock" {
            // The lock has no `off` arm — it is a plain bool flag, and its
            // environment opt-in is only cleared by unsetting the variable.
            continue;
        }
        assert!(!get(&p), "{name}: flag=off must override the env opt-in");
    }
    assert!(
        p.turbo_flash_lock,
        "turbo_flash_lock has no off arm; the env opt-in must survive"
    );
}

#[test]
fn off_with_a_clear_environment_resolves_every_gate_off() {
    let p = resolve_all_off(env_clear());
    for (name, get) in GATES {
        assert!(!get(&p), "{name}: flag=off + clear env must resolve OFF");
    }
}

#[test]
fn thresholds_pass_through_from_the_environment_untouched() {
    // Neither threshold has a flag, so resolution must not rewrite them.
    let env = DispatchPolicy {
        fused_qk_min_kv_seq: 7,
        turbo_flash_min_kv_seq: 11,
        ..DispatchPolicy::default()
    };
    for resolved in [
        resolve_defaults(env),
        resolve_all_on(env),
        resolve_all_off(env),
    ] {
        assert_eq!(resolved.fused_qk_min_kv_seq, 7);
        assert_eq!(resolved.turbo_flash_min_kv_seq, 11);
    }
}

#[test]
fn each_gate_is_wired_to_its_own_flag() {
    // Cross-talk guard: flipping one flag on must not move any other field.
    // A copy-paste in `resolve_dispatch_policy` that read the wrong flag, or
    // wrote the wrong field, fails here.
    let base = resolve_defaults(env_clear());

    let only_fused_qk = resolve_dispatch_policy(
        FusedQkMode::On,
        SparseAttnMode::Auto,
        TurboFlashMode::Auto,
        false,
        PlanarFlashDecodeMode::Auto,
        RotKFusedMode::Auto,
        env_clear(),
    );
    assert_eq!(
        only_fused_qk,
        DispatchPolicy {
            fused_qk: true,
            ..base
        }
    );

    let only_sparse_attn = resolve_dispatch_policy(
        FusedQkMode::Auto,
        SparseAttnMode::On,
        TurboFlashMode::Auto,
        false,
        PlanarFlashDecodeMode::Auto,
        RotKFusedMode::Auto,
        env_clear(),
    );
    assert_eq!(
        only_sparse_attn,
        DispatchPolicy {
            sparse_attn: true,
            ..base
        }
    );

    let only_turbo_flash = resolve_dispatch_policy(
        FusedQkMode::Auto,
        SparseAttnMode::Auto,
        TurboFlashMode::On,
        false,
        PlanarFlashDecodeMode::Auto,
        RotKFusedMode::Auto,
        env_clear(),
    );
    assert_eq!(
        only_turbo_flash,
        DispatchPolicy {
            turbo_flash: true,
            ..base
        }
    );

    let only_lock = resolve_dispatch_policy(
        FusedQkMode::Auto,
        SparseAttnMode::Auto,
        TurboFlashMode::Auto,
        true,
        PlanarFlashDecodeMode::Auto,
        RotKFusedMode::Auto,
        env_clear(),
    );
    assert_eq!(
        only_lock,
        DispatchPolicy {
            turbo_flash_lock: true,
            ..base
        }
    );

    let only_planar = resolve_dispatch_policy(
        FusedQkMode::Auto,
        SparseAttnMode::Auto,
        TurboFlashMode::Auto,
        false,
        PlanarFlashDecodeMode::On,
        RotKFusedMode::Auto,
        env_clear(),
    );
    assert_eq!(
        only_planar,
        DispatchPolicy {
            planar_flash_decode: true,
            ..base
        }
    );

    let only_rot_k = resolve_dispatch_policy(
        FusedQkMode::Auto,
        SparseAttnMode::Auto,
        TurboFlashMode::Auto,
        false,
        PlanarFlashDecodeMode::Auto,
        RotKFusedMode::On,
        env_clear(),
    );
    assert_eq!(
        only_rot_k,
        DispatchPolicy {
            rot_k_fused: true,
            ..base
        }
    );
}
