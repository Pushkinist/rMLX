use super::{
    apply_fused_qk_flags, apply_planar_flash_decode_flags, apply_sparse_attn_flags,
    apply_turbo_flags, FusedQkMode, PlanarFlashDecodeMode, SparseAttnMode, TurboFlashMode,
};
use std::sync::Mutex;

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

// Env mutations are process-global. Serialize these tests with a static
// mutex so parallel test threads do not bleed into each other.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const TF: &str = "RMLX_TURBO_FLASH";
const TFL: &str = "RMLX_TURBO_FLASH_LOCK";

fn get(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

fn remove(key: &str) {
    std::env::remove_var(key);
}

// Each apply_turbo_flags test asserts on std::env::var(TF),
// NOT on `rmlx_kv_quant::turbo_flash_msl::turbo_flash_enabled()`. The latter
// is a OnceLock that may already be latched by another test in the same
// binary (test order is non-deterministic) — checking it here would couple
// us to global state we cannot reset. These tests only validate the
// env-setter logic in `apply_turbo_flags`; the OnceLock ↔ env contract is
// covered by the dispatch-counter assertion in the NIAH harness.

#[test]
fn turbo_flash_on_sets_env() {
    // Asserts on std::env::var(TF), not on turbo_flash_enabled(): that gate is
    // a OnceLock latched on first read, so reading it here would couple the
    // test to whichever test in this binary ran first.
    let _guard = ENV_LOCK.lock().unwrap();
    remove(TF);
    remove(TFL);

    apply_turbo_flags(TurboFlashMode::On, false);

    assert_eq!(
        get(TF).as_deref(),
        Some("1"),
        "mode=On must force RMLX_TURBO_FLASH=1"
    );
    assert_eq!(
        get(TFL),
        None,
        "RMLX_TURBO_FLASH_LOCK must be untouched when lock flag is false"
    );

    remove(TF);
}

#[test]
fn turbo_flash_off_hard_override_removes_env() {
    // Asserts on std::env::var(TF), not on turbo_flash_enabled(): that gate is
    // a OnceLock latched on first read and cannot be reset between tests.
    //
    // Explicit `--turbo-flash off` is a HARD override. If the shell sets
    // RMLX_TURBO_FLASH=1 and the user then passes `--turbo-flash off`, the
    // off must win — otherwise the OnceLock in `turbo_flash_msl` would latch
    // true on first read and the `off` flag becomes silently a no-op.
    //
    // RMLX_TURBO_FLASH_LOCK is independent and is left alone by this arm.
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var(TF, "1");
    std::env::set_var(TFL, "1");

    apply_turbo_flags(TurboFlashMode::Off, false);

    assert_eq!(
        get(TF),
        None,
        "mode=Off must REMOVE pre-existing RMLX_TURBO_FLASH (hard override)"
    );
    assert_eq!(
        get(TFL).as_deref(),
        Some("1"),
        "mode=Off must not clear pre-existing RMLX_TURBO_FLASH_LOCK \
         (the lock flag is a separate axis)"
    );

    remove(TF);
    remove(TFL);
}

#[test]
fn turbo_flash_off_unset_env_stays_unset() {
    // Asserts on std::env::var(TF), not on turbo_flash_enabled(): that gate is
    // a OnceLock latched on first read and cannot be reset between tests.
    let _guard = ENV_LOCK.lock().unwrap();
    remove(TF);
    remove(TFL);

    apply_turbo_flags(TurboFlashMode::Off, false);

    assert_eq!(
        get(TF),
        None,
        "mode=Off + env unset → RMLX_TURBO_FLASH must remain unset"
    );
    assert_eq!(
        get(TFL),
        None,
        "mode=Off + env unset → RMLX_TURBO_FLASH_LOCK must remain unset"
    );
}

// ── --planar-flash-decode env-setter tests ────────────────────────────
//
// Mirror the turbo_flash_* tests above (same OnceLock first-read latch
// concern). Each test asserts on `std::env::var(PFD)` directly — NOT on
// `rmlx_kv_quant::planar_flash_decode_msl::planar_flash_decode_enabled()`,
// because that gate is a OnceLock latched on first read and we can't reset
// it between tests in the same binary.

const PFD: &str = "RMLX_PLANAR_FLASH_DECODE";

#[test]
fn planar_flash_decode_on_sets_env() {
    let _guard = ENV_LOCK.lock().unwrap();
    remove(PFD);

    apply_planar_flash_decode_flags(PlanarFlashDecodeMode::On);

    assert_eq!(
        get(PFD).as_deref(),
        Some("1"),
        "mode=On must force RMLX_PLANAR_FLASH_DECODE=1"
    );
    remove(PFD);
}

#[test]
fn planar_flash_decode_off_hard_override_removes_env() {
    // Same hard-override semantics as turbo_flash off: a stale shell
    // RMLX_PLANAR_FLASH_DECODE=1 must NOT be allowed to latch the OnceLock to
    // true when the user explicitly passes `--planar-flash-decode off`.
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var(PFD, "1");

    apply_planar_flash_decode_flags(PlanarFlashDecodeMode::Off);

    assert_eq!(
        get(PFD),
        None,
        "mode=Off must REMOVE pre-existing RMLX_PLANAR_FLASH_DECODE (hard override)"
    );
    remove(PFD);
}

#[test]
fn planar_flash_decode_auto_currently_off_on_every_host() {
    // Validation complete (HOLD: perf -0.19%, NIAH blocked by pre-existing
    // PlanarK chunked-prefill bug). Test name + contract remain correct.
    // Update only when the HOLD is lifted.
    let _guard = ENV_LOCK.lock().unwrap();
    remove(PFD);

    apply_planar_flash_decode_flags(PlanarFlashDecodeMode::Auto);

    assert_eq!(
        get(PFD),
        None,
        "Auto must leave RMLX_PLANAR_FLASH_DECODE unset (HOLD, see planar-flash-decode validation report) (was: {:?})",
        get(PFD)
    );
    remove(PFD);
}

#[test]
fn planar_flash_decode_off_unset_env_stays_unset() {
    // OnceLock scope: the gate inside `planar_flash_decode_msl` caches its
    // env read on first call. `apply_planar_flash_decode_flags(Off)` is a
    // pure env-setter (idempotent vs. an already-unset var); this test only
    // checks the env-setter does NOT add the var when the user asks for OFF
    // and the var was not set to begin with.
    let _guard = ENV_LOCK.lock().unwrap();
    remove(PFD);

    apply_planar_flash_decode_flags(PlanarFlashDecodeMode::Off);

    assert_eq!(
        get(PFD),
        None,
        "mode=Off + env unset → RMLX_PLANAR_FLASH_DECODE must remain unset"
    );
}

#[test]
fn turbo_flash_auto_holds_off_on_this_host() {
    // Asserts on std::env::var(TF), not on turbo_flash_enabled(): that gate is
    // a OnceLock latched on first read and cannot be reset between tests.
    //
    // Auto is a HOLD on every host, recognised family or not: the kernel is a
    // measured 2.0-4.25x decode loss on K8V4 at kv_seq > 4096 and it perturbs
    // the output. Whichever family this host reports, the env must stay unset.
    let _guard = ENV_LOCK.lock().unwrap();
    remove(TF);
    remove(TFL);

    apply_turbo_flags(TurboFlashMode::Auto, false);

    let family = rmlx_core::apple_gpu::apple_silicon_generation();
    assert_eq!(
        get(TF),
        None,
        "Auto must leave RMLX_TURBO_FLASH unset (HOLD) on family {family:?}; was: {:?}",
        get(TF)
    );
    remove(TF);
}

#[test]
fn turbo_flash_auto_does_not_clear_an_opt_in_env() {
    // Auto is a HOLD, not a hard override: an operator who exported
    // RMLX_TURBO_FLASH=1 still gets the kernel (and a warn! saying so). Only
    // `--turbo-flash off` removes the var. Family-independent, so this holds
    // on whichever host runs it.
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var(TF, "1");

    apply_turbo_flags(TurboFlashMode::Auto, false);

    assert_eq!(
        get(TF).as_deref(),
        Some("1"),
        "Auto must leave a pre-existing RMLX_TURBO_FLASH=1 alone (back-compat)"
    );
    remove(TF);
}

// ── --fused-qk env-setter tests ──────────────────────────────────────────────
//
// Mirror the planar_flash_decode_* tests above (same OnceLock first-read latch
// concern). Each test asserts on `std::env::var(FQK)` directly — NOT on
// `rmlx_kv_quant::fused_qk_enabled()`, because that gate is a OnceLock
// latched on first read and we can't reset it between tests in the same binary.

const FQK: &str = "RMLX_FUSED_QK";

#[test]
fn fused_qk_on_sets_env() {
    // mode=On must force-set RMLX_FUSED_QK=1.
    let _guard = ENV_LOCK.lock().unwrap();
    remove(FQK);

    apply_fused_qk_flags(FusedQkMode::On);

    assert_eq!(
        get(FQK).as_deref(),
        Some("1"),
        "mode=On must force RMLX_FUSED_QK=1"
    );
    remove(FQK);
}

#[test]
fn fused_qk_off_hard_override_removes_env() {
    // Explicit `--fused-qk off` is a HARD override. A stale RMLX_FUSED_QK=1
    // in the shell must be removed so the OnceLock cannot latch true on
    // first read.
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var(FQK, "1");

    apply_fused_qk_flags(FusedQkMode::Off);

    assert_eq!(
        get(FQK),
        None,
        "mode=Off must REMOVE pre-existing RMLX_FUSED_QK (hard override)"
    );
    remove(FQK);
}

#[test]
fn fused_qk_auto_currently_off_on_every_host() {
    // HOLD: Auto stays OFF on every host.
    // Kernel stubs are present but not dispatching (codec implementations pending).
    // Update only when the HOLD is lifted for all five codecs.
    //
    // OnceLock scope: `fused_qk_enabled()` in `rmlx_kv_quant` caches its
    // env read on first call. This test only validates the env-setter side —
    // do NOT call `fused_qk_enabled()` here (would latch the OnceLock for the
    // rest of the binary's test run).
    let _guard = ENV_LOCK.lock().unwrap();
    remove(FQK);

    apply_fused_qk_flags(FusedQkMode::Auto);

    assert_eq!(
        get(FQK),
        None,
        "Auto must leave RMLX_FUSED_QK unset (HOLD: kernel stubs pending) (was: {:?})",
        get(FQK)
    );
    remove(FQK);
}

#[test]
fn fused_qk_off_unset_env_stays_unset() {
    // OnceLock scope: `fused_qk_enabled()` caches its env read on first call.
    // `apply_fused_qk_flags(Off)` is a pure env-setter (idempotent vs. an
    // already-unset var); this test only checks the env-setter does NOT add
    // the var when the user asks for OFF and the var was not set to begin with.
    let _guard = ENV_LOCK.lock().unwrap();
    remove(FQK);

    apply_fused_qk_flags(FusedQkMode::Off);

    assert_eq!(
        get(FQK),
        None,
        "mode=Off + env unset → RMLX_FUSED_QK must remain unset"
    );
}

// ── --sparse-attn env-setter tests ───────────────────────────────────────────
//
// Mirror the fused_qk_* tests above (same OnceLock first-read latch concern).
// Each test asserts on `std::env::var(SA)` directly — NOT on
// `rmlx_kv_quant::sparse_attn_enabled()`, because that gate is a OnceLock
// latched on first read and we can't reset it between tests in the same
// binary.

const SA: &str = "RMLX_SPARSE_ATTN";

#[test]
fn sparse_attn_on_sets_env() {
    // mode=On must force-set RMLX_SPARSE_ATTN=1.
    let _guard = ENV_LOCK.lock().unwrap();
    remove(SA);

    apply_sparse_attn_flags(SparseAttnMode::On);

    assert_eq!(
        get(SA).as_deref(),
        Some("1"),
        "mode=On must force RMLX_SPARSE_ATTN=1"
    );
    remove(SA);
}

#[test]
fn sparse_attn_off_hard_override_removes_env() {
    // Explicit `--sparse-attn off` is a HARD override. A stale RMLX_SPARSE_ATTN=1
    // in the shell must be removed so the OnceLock cannot latch true on first read.
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var(SA, "1");

    apply_sparse_attn_flags(SparseAttnMode::Off);

    assert_eq!(
        get(SA),
        None,
        "mode=Off must REMOVE pre-existing RMLX_SPARSE_ATTN (hard override)"
    );
    remove(SA);
}

#[test]
fn sparse_attn_auto_currently_off_on_every_host() {
    // HOLD: Auto stays OFF on every host. Phase-1/phase-2 MSL kernels
    // not yet dispatching in production flows. Update only when the HOLD is
    // lifted.
    //
    // OnceLock scope: `sparse_attn_enabled()` in `rmlx_kv_quant` caches its
    // env read on first call. This test only validates the env-setter side —
    // do NOT call `sparse_attn_enabled()` here (would latch the OnceLock for
    // the rest of the binary's test run).
    let _guard = ENV_LOCK.lock().unwrap();
    remove(SA);

    apply_sparse_attn_flags(SparseAttnMode::Auto);

    assert_eq!(
        get(SA),
        None,
        "Auto must leave RMLX_SPARSE_ATTN unset (HOLD: kernels dormant on normal generate flow) (was: {:?})",
        get(SA)
    );
    remove(SA);
}

#[test]
fn sparse_attn_off_unset_env_stays_unset() {
    // OnceLock scope: `sparse_attn_enabled()` caches its env read on first
    // call. `apply_sparse_attn_flags(Off)` is a pure env-setter (idempotent
    // vs. an already-unset var); this test only checks the env-setter does
    // NOT add the var when the user asks for OFF and the var was not set to
    // begin with.
    let _guard = ENV_LOCK.lock().unwrap();
    remove(SA);

    apply_sparse_attn_flags(SparseAttnMode::Off);

    assert_eq!(
        get(SA),
        None,
        "mode=Off + env unset → RMLX_SPARSE_ATTN must remain unset"
    );
}
