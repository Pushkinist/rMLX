// Two dispatch policies, live at the same time, in one process.
//
// This is the property the `OnceLock` kernel gates could not provide: their
// first read froze the value for the process lifetime, so a driver that wanted
// to compare two kernel paths had to fork a second process — and pay a second
// model load and a second thermal state for it. `DispatchPolicy` is a value
// each `KvCache` captures at construction, so two caches built under different
// policies stay independent and can be alternated freely.
//
// The GPU test below interleaves the two arms ABBA over one pair of caches and
// asserts, per step, that the TurboFlash dispatch counter moves for the ON
// cache and not for the OFF one. Blocked runs (all of A, then all of B) would
// pass even if the second arm had silently inherited the first arm's policy;
// interleaving does not.
//
// `#[ignore]` on the GPU test: a shared Metal context driven from parallel
// `cargo test` threads aborts the whole binary. Run via:
//   cargo test -p rmlx-kv-quant --test dispatch_policy_two_arms -- \
//       --ignored --test-threads=1 --nocapture
//
// CLAUDE.md hard rule 8 (single MLX process): the test claims no port — the
// integration runner serialises tests within one process. Preflight with
//   pkill -f "rmlx serve"; rm -f /tmp/rmlx.*.claim
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unusual_byte_groupings,
    clippy::indexing_slicing,
    missing_docs
)]
//! Two dispatch policies live in one process.

use rmlx_core::{dispatch_policy, set_dispatch_policy, DispatchPolicy};
use rmlx_kv_quant::turbo_flash_msl::turbo_flash_dispatch_count;
use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};

const B: i32 = 1;
const KV_H: i32 = 2;
const HEADS_PER_KV: i32 = 4;
const N_Q_HEADS: i32 = KV_H * HEADS_PER_KV;
const HEAD_DIM: i32 = 128;
const MAX_SEQ: i32 = 4096;
const PREFILL_SEQ: i32 = 64;
/// ABBA over two arms: four slots, each arm first and last once, so a
/// position-dependent effect cancels instead of loading onto one arm.
const SLOTS: [usize; 4] = [0, 1, 1, 0];

/// TurboFlash on, threshold dropped so the kernel fires on this short cache.
fn arm_on() -> DispatchPolicy {
    DispatchPolicy {
        turbo_flash: true,
        turbo_flash_min_kv_seq: 0,
        ..DispatchPolicy::default()
    }
}

/// The control arm: identical apart from the kernel gate, so a dispatch-count
/// difference can only come from the gate.
fn arm_off() -> DispatchPolicy {
    DispatchPolicy {
        turbo_flash: false,
        ..arm_on()
    }
}

/// Serialises the two tests that swap the process default. They are the only
/// writers in this binary; the GPU test never touches it.
static PROCESS_DEFAULT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn skip_if_no_gpu() -> bool {
    std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1")
}

fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("from_bytes")
}

fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let frac = (state >> 32) as u32 as f32 / u32::MAX as f32;
            frac.mul_add(2.0, -1.0)
        })
        .collect()
}

fn bf16(data: &[f32], shape: &[i32], device: Device) -> Array {
    make_f32_array(data, shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 cast")
}

/// Build a K8V4 cache under `policy` and prefill it with `PREFILL_SEQ` tokens.
fn prefilled(policy: DispatchPolicy, seed: u64, device: Device) -> KvCache {
    let mut cache =
        KvCache::with_quant_max_seq(KvQuant::K8V4, MAX_SEQ).with_dispatch_policy(policy);
    cache.enter_prefill();
    let shape = [B, KV_H, PREFILL_SEQ, HEAD_DIM];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let k = bf16(&lcg_data(n, seed), &shape, device);
    let v = bf16(&lcg_data(n, seed ^ 0xFFFF), &shape, device);
    let _ = cache.update(&k, &v, device).expect("prefill update");
    cache.exit_prefill(device).expect("exit_prefill");
    cache
}

/// One decode step, returning the TurboFlash dispatch-count delta it caused.
fn decode_step(cache: &mut KvCache, seed: u64, device: Device) -> u64 {
    let kv_shape = [B, KV_H, 1, HEAD_DIM];
    let q_shape = [B, N_Q_HEADS, 1, HEAD_DIM];
    let n_kv: usize = kv_shape.iter().map(|&d| d as usize).product();
    let n_q: usize = q_shape.iter().map(|&d| d as usize).product();
    let new_k = bf16(&lcg_data(n_kv, seed), &kv_shape, device);
    let new_v = bf16(&lcg_data(n_kv, seed ^ 0xAAAA), &kv_shape, device);
    let q = bf16(&lcg_data(n_q, seed ^ 0x5555), &q_shape, device);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let before = turbo_flash_dispatch_count();
    let _ = cache
        .update_and_sdpa(&q, &new_k, &new_v, scale, "", None, device)
        .expect("decode step");
    turbo_flash_dispatch_count() - before
}

/// A cache captures the process default at construction and keeps it, so
/// replacing the default afterwards cannot reach back into a live cache.
#[test]
fn caches_built_under_different_process_defaults_keep_their_own_policy() {
    let _guard = PROCESS_DEFAULT_LOCK.lock().unwrap();
    let restore = dispatch_policy();

    set_dispatch_policy(arm_on());
    let a = KvCache::with_quant_max_seq(KvQuant::K8V4, MAX_SEQ);
    set_dispatch_policy(arm_off());
    let b = KvCache::with_quant_max_seq(KvQuant::K8V4, MAX_SEQ);

    assert_eq!(a.dispatch_policy(), arm_on(), "cache A kept the ON policy");
    assert_eq!(
        b.dispatch_policy(),
        arm_off(),
        "cache B took the OFF policy"
    );
    assert!(
        a.dispatch_policy() != b.dispatch_policy(),
        "two policies must be able to be live at once"
    );

    set_dispatch_policy(restore);
}

/// `with_dispatch_policy` overrides the captured default, so a caller can set
/// an arm without touching process state at all.
#[test]
fn with_dispatch_policy_overrides_the_process_default() {
    let _guard = PROCESS_DEFAULT_LOCK.lock().unwrap();
    let restore = dispatch_policy();

    set_dispatch_policy(arm_off());
    let c = KvCache::with_quant_max_seq(KvQuant::K8V4, MAX_SEQ).with_dispatch_policy(arm_on());
    assert_eq!(c.dispatch_policy(), arm_on());
    assert_eq!(
        dispatch_policy(),
        arm_off(),
        "the override must not write back to the process default"
    );

    set_dispatch_policy(restore);
}

/// GPU: alternate the two arms ABBA over one pair of caches and check the
/// TurboFlash dispatch counter per step. The ON cache must dispatch on every
/// one of its steps and the OFF cache on none of them — with the arms
/// interleaved, not blocked.
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test dispatch_policy_two_arms -- --ignored --test-threads=1"]
fn two_policies_alternate_within_one_process() {
    if skip_if_no_gpu() {
        return;
    }
    let device = Device::Gpu;

    // Both caches are built up front, so both policies are live for the whole
    // run rather than one replacing the other.
    let mut caches = [
        prefilled(arm_on(), 0x0D15_0A11_0000_0001, device),
        prefilled(arm_off(), 0x0D15_0A11_0000_0002, device),
    ];
    assert_eq!(caches[0].dispatch_policy(), arm_on());
    assert_eq!(caches[1].dispatch_policy(), arm_off());

    let mut on_steps = 0_u32;
    let mut on_dispatches = 0_u64;
    let mut off_steps = 0_u32;
    let mut off_dispatches = 0_u64;

    for (i, &arm) in SLOTS.iter().enumerate() {
        let seed = 0x0D15_0A11_1000_0000 ^ (i as u64);
        let delta = decode_step(&mut caches[arm], seed, device);
        eprintln!("slot {i}: arm={arm} turbo_flash dispatch delta={delta}");
        if arm == 0 {
            on_steps += 1;
            on_dispatches += delta;
            assert!(
                delta > 0,
                "slot {i}: the ON arm must dispatch TurboFlash on every step \
                 (delta={delta}); a policy that had leaked from the OFF arm \
                 would read 0 here"
            );
        } else {
            off_steps += 1;
            off_dispatches += delta;
            assert_eq!(
                delta, 0,
                "slot {i}: the OFF arm must not dispatch TurboFlash \
                 (delta={delta}); a policy that had leaked from the ON arm \
                 would read > 0 here"
            );
        }
    }

    eprintln!(
        "ABBA over {} slots: on={on_dispatches} dispatches in {on_steps} steps, \
         off={off_dispatches} in {off_steps} steps",
        SLOTS.len()
    );
    assert!(on_steps > 0 && off_steps > 0, "both arms must have run");
    assert!(
        on_dispatches > 0,
        "the ON arm never dispatched — the kernel gate is unreachable from a \
         cache policy and this test proves nothing"
    );
    assert_eq!(off_dispatches, 0, "the OFF arm must stay dormant");
}
