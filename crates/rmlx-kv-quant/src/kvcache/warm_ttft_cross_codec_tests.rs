//! Warm-TTFT bf16-K cross-codec audit (diagnostic regression lock).
//!
//! The warm-TTFT contract was originally pinned for PlanarK only. This file
//! extends the empirical proof across three behavioural classes, so the audit
//! table in `docs/KV_CACHE.md` §9.6 is backed by executable assertions rather
//! than code-reading:
//!
//! 1. **Shortcut codecs** (`decode_fp16_k.is_some()` early-return present):
//!    K8V4, K8V8 (asserted in this file). TurboSym3 is covered by code-reading
//!    (same `update_decode_fp16` dispatch path); PlanarK is asserted in the
//!    sibling `warm_ttft_tests.rs`. After `exit_prefill` the bf16 K+V mirror is
//!    live; every decode `update()` routes through `update_decode_fp16` and the
//!    quant codec stays **frozen** (its `shape[2]` does not advance past the
//!    prefill length). Decode-phase K AND V are bf16, not re-quantised.
//!
//!    `Iso3Sym` used to be in this class but is now a **fused symmetric** codec:
//!    it keeps no bf16 mirror on either axis (decode reads both packed iso
//!    rings), so it is asserted alongside the K-only family below rather than
//!    here — see `iso_sym3_fused_no_seed_at_decode`.
//!
//! 2. **K-only codecs** (no `decode_fp16_k.is_some()` shortcut in the
//!    `update_<arch>` body; V via the V-only helper): IsoKOnly3,
//!    RotorKOnly3. These quantise K at **every** decode step — the K codec
//!    `shape[2]` advances by 1 per step, and V rides the bf16
//!    `decode_fp16_v` mirror.
//!
//! An earlier audit found that `exit_prefill` unconditionally populated
//! `decode_fp16_k` for **every** quant arm, INCLUDING the K-only codecs
//! whose `update_<arch>` never reads it. For IsoKOnly3 / RotorKOnly3 the bf16 K
//! seed was allocated at exit_prefill and held, unused, for the whole decode
//! window — dead memory.
//!
//! The fix gates the K-seed materialisation on
//! `KvQuant::feeds_bf16_k_at_decode`, which is `false` for the K-only family.
//! So the K-only tests below assert `decode_fp16_k` is **absent** after
//! exit_prefill+decode (the V seed is still present, and the K codec still
//! advances every decode step — correctness is byte-unchanged, only the dead
//! K buffer is reclaimed). The shortcut-codec assertions are unchanged.
//!
//! The asymmetry is the whole point of the audit: the "warm-TTFT" bf16
//! decode shortcut fires for the *Sym / V-quant family (codec frozen at
//! decode) but is structurally absent from the K-only family's decode body
//! (codec runs at decode). See the audit table + per-codec rationale in
//! `docs/KV_CACHE.md` §9.6.

use super::core::KvCache;
use crate::storage::KvStorage;
use crate::KvQuant;
use rmlx_mlx::{Array, Device, Dtype};

const TEST_KV_H: i32 = 8;
const TEST_HEAD_DIM: i32 = 128;
const TEST_MAX_SEQ: i32 = 512;
const TEST_PREFILL_SEQ: i32 = 256;

fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: Apple-Silicon-only build (CLAUDE.md Hard rule 1); f32 is 4-byte
    // LE on this target. `data` is borrowed read-only and copied into MLX
    // before the borrow ends.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("Array::from_bytes")
}

/// Outcome of driving a cache through prefill → exit_prefill → one decode step.
struct DecodeOutcome {
    /// Whether the bf16 K seed (`decode_fp16_k`) is live after the decode step.
    k_seed_live: bool,
    /// Whether the bf16 V seed (`decode_fp16_v`) is live after the decode step.
    v_seed_live: bool,
    /// Cache offset after one decode step.
    offset: i32,
    /// `resident_bytes()` residency total after one decode step.
    resident_bytes: u64,
}

/// Bytes of the *filled* prefix of a `[B, kv_h, seq, D]` bf16 mirror, computed
/// straight from the array's own shape and dtype.
///
/// The independent restatement is deliberate: it lets the residency asserts
/// below anchor to the buffers the cache actually holds rather than to a
/// per-codec bit-width formula, which is precisely the thing that reported a
/// confident number for memory it was not measuring.
///
/// Scope: this pins **which buffers are counted** — that the dead K seed is
/// not, and the live V seed is. The K store's own total is taken from
/// `KvStorage::resident_bytes`, i.e. from the accounting, so these do not
/// independently validate the store's magnitude. `resident_ring_tests`
/// (`ring_bytes_match_independent_geometry`) is where that is checked.
#[allow(
    clippy::indexing_slicing,
    reason = "test helper: decode mirrors are always the 4-D [B, kv_h, seq, D] shape asserted below"
)]
fn filled_mirror_bytes(a: &Array, filled: i32) -> u64 {
    let s = a.shape();
    assert_eq!(s.len(), 4, "decode mirror must be 4-D [B, kv_h, seq, D]");
    let per_pos = s[0] as u64 * s[1] as u64 * s[3] as u64 * a.dtype().itemsize() as u64;
    per_pos * (filled.max(0) as u64).min(s[2] as u64)
}

/// Drive a cache through prefill (one chunk) → exit_prefill → one decode step.
fn drive_one_decode(cache: &mut KvCache, device: Device) -> DecodeOutcome {
    cache.enter_prefill();

    let prefill_shape = [1i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_pref = f32_arr(&vec![0.123f32; n_pref], &prefill_shape);
    let v_pref = f32_arr(&vec![0.456f32; n_pref], &prefill_shape);
    cache
        .update(&k_pref, &v_pref, device)
        .expect("prefill chunk");
    cache.exit_prefill(device).expect("exit_prefill");

    let step_shape = [1i32, TEST_KV_H, 1, TEST_HEAD_DIM];
    let n_step: usize = step_shape.iter().map(|&d| d as usize).product();
    let k_step = f32_arr(&vec![0.789f32; n_step], &step_shape);
    let v_step = f32_arr(&vec![0.321f32; n_step], &step_shape);
    cache
        .update(&k_step, &v_step, device)
        .expect("decode step 1");

    DecodeOutcome {
        k_seed_live: cache.decode_fp16_k_for_test().is_some(),
        v_seed_live: cache.decode_fp16_v_for_test().is_some(),
        offset: cache.offset(),
        resident_bytes: cache.resident_bytes(),
    }
}

/// Shortcut-codec contract: after one decode step the bf16 K mirror is live
/// (proves `update_decode_fp16` ran) and the packed store was never built.
///
/// The store is what the contract is really about. A codec that mirrors both
/// axes has no decode path over its packed buffer, so `exit_prefill` building
/// one would put a second full copy of the context next to a mirror that is
/// already bf16-sized — and hold it, unread, until the request ends. The three
/// assertions below are one statement each of that: the store is empty before
/// decode, still empty after (nothing re-arms it mid-window), and the cache's
/// whole residency is the two mirrors' filled prefixes and nothing else.
fn assert_shortcut_codec(quant: KvQuant, codec_seq_after: impl Fn(&KvCache) -> i32) {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
    cache.enter_prefill();
    let prefill_shape = [1i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_pref = f32_arr(&vec![0.123f32; n_pref], &prefill_shape);
    let v_pref = f32_arr(&vec![0.456f32; n_pref], &prefill_shape);
    cache
        .update(&k_pref, &v_pref, device)
        .expect("prefill chunk");
    cache.exit_prefill(device).expect("exit_prefill");

    let codec_seq_pre_decode = codec_seq_after(&cache);
    assert_eq!(
        codec_seq_pre_decode, 0,
        "{quant:?}: exit_prefill must not bulk-encode a packed store that no decode \
         path reads — got a store of {codec_seq_pre_decode} positions alongside the \
         bf16 mirror"
    );
    assert_eq!(
        cache.storage().resident_bytes(),
        0,
        "{quant:?}: the packed store must hold no bytes at all after exit_prefill"
    );

    let step_shape = [1i32, TEST_KV_H, 1, TEST_HEAD_DIM];
    let n_step: usize = step_shape.iter().map(|&d| d as usize).product();
    let k_step = f32_arr(&vec![0.789f32; n_step], &step_shape);
    let v_step = f32_arr(&vec![0.321f32; n_step], &step_shape);
    cache
        .update(&k_step, &v_step, device)
        .expect("decode step 1");

    assert!(
        cache.decode_fp16_k_for_test().is_some(),
        "{quant:?}: warm-TTFT contract — decode_fp16_k MUST be live after exit_prefill + decode"
    );
    let codec_seq_post_decode = codec_seq_after(&cache);
    assert_eq!(
        codec_seq_post_decode, 0,
        "{quant:?}: warm-TTFT shortcut violation — the quant K codec allocated and \
         advanced to {codec_seq_post_decode} on a decode step while decode_fp16_k was \
         live (the codec must stay quiescent; decode reads bf16)"
    );
    assert_eq!(
        cache.offset(),
        TEST_PREFILL_SEQ + 1,
        "{quant:?}: offset after one decode step"
    );

    // Residency is the sharpest form of the claim: the cache holds the two
    // mirrors' filled prefixes and nothing else. Fails if a store is built, if
    // a mirror is dropped, or if either is double-counted.
    let k_seed = cache
        .decode_fp16_k_for_test()
        .expect("K mirror is live (asserted above)");
    let v_seed = cache
        .decode_fp16_v_for_test()
        .expect("V mirror is live: this family reads bf16 on both axes");
    let expected =
        filled_mirror_bytes(k_seed, cache.offset()) + filled_mirror_bytes(v_seed, cache.offset());
    assert_eq!(
        cache.resident_bytes(),
        expected,
        "{quant:?}: resident_bytes must equal the two bf16 mirrors' filled prefixes \
         and nothing else"
    );
}

#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts a construction-time invariant: storage matches the KvQuant the cache was built with; any other variant is a construction bug an explicit panic catches sooner."
)]
fn k8_codec_seq(cache: &KvCache) -> i32 {
    match cache.storage() {
        KvStorage::K8V4 { k, .. } | KvStorage::K8V8 { k, .. } => k
            .as_ref()
            .and_then(|q| q.shape.get(2).copied())
            .unwrap_or(0),
        _ => panic!("expected K8V4/K8V8 storage"),
    }
}

#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "construction-time invariant — see k8_codec_seq."
)]
fn iso_sym3_k_codec_seq(cache: &KvCache) -> i32 {
    match cache.storage() {
        KvStorage::IsoSym3 { k, .. } => k
            .as_ref()
            .map_or(0, |q| q.shape.get(2).copied().unwrap_or(0)),
        _ => panic!("expected IsoSym3 storage"),
    }
}

#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "construction-time invariant — see k8_codec_seq."
)]
fn iso_k_only3_k_codec_seq(cache: &KvCache) -> i32 {
    match cache.storage() {
        KvStorage::IsoKOnly3 { k, .. } => k
            .as_ref()
            .map_or(0, |q| q.shape.get(2).copied().unwrap_or(0)),
        _ => panic!("expected IsoKOnly3 storage"),
    }
}

#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "construction-time invariant — see k8_codec_seq."
)]
fn rotor_k_only3_k_codec_seq(cache: &KvCache) -> i32 {
    match cache.storage() {
        KvStorage::RotorKOnly3 { k, .. } => k
            .as_ref()
            .map_or(0, |q| q.shape.get(2).copied().unwrap_or(0)),
        _ => panic!("expected RotorKOnly3 storage"),
    }
}

/// Every codec, swept: `exit_prefill` builds a packed store if and only if
/// `materialises_packed_store()` says so, and the cache's residency matches.
///
/// The three hand-written cases below observe `K8V4`, `K8V8` and `PlanarK` in
/// detail — the codec-specific accessors make them worth keeping. But
/// `exit_prefill`'s behaviour changed for **18** codecs, and a missed reader in
/// any of the other 15 is silent by construction: nothing errors, the decode
/// still works off the mirror, and only the residency moves. So the property is
/// stated once over `ALL_KV_QUANTS`, with the expectation **derived from the
/// predicate** rather than listed, which is what makes it exhaustive as new
/// codecs land.
///
/// The residency oracle is a same-shape `KvQuant::None` cache, which shares no
/// arithmetic with the code under test: a store-free codec must report exactly
/// what plain bf16 reports.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn exit_prefill_builds_a_store_exactly_when_the_predicate_says_so() {
    let device = Device::Cpu;

    let prefill = |quant: KvQuant| -> KvCache {
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
        cache.enter_prefill();
        let shape = [1i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
        let n: usize = shape.iter().map(|&d| d as usize).product();
        cache
            .update(
                &f32_arr(&vec![0.123f32; n], &shape),
                &f32_arr(&vec![0.456f32; n], &shape),
                device,
            )
            .expect("prefill chunk");
        cache.exit_prefill(device).expect("exit_prefill");
        cache
    };

    // One decode step on top, because that is where a *missed reader* shows up:
    // every codec body lazily allocates an empty store and appends to it when
    // it is reached (`if k.is_none() { *k = Some(..) }`), so a decode path that
    // still routes into the codec leaves bytes behind even though the prefill
    // gate built nothing.
    let decode_once = |cache: &mut KvCache| {
        let step = [1i32, TEST_KV_H, 1, TEST_HEAD_DIM];
        let n: usize = step.iter().map(|&d| d as usize).product();
        cache
            .update(
                &f32_arr(&vec![0.789f32; n], &step),
                &f32_arr(&vec![0.321f32; n], &step),
                device,
            )
            .expect("decode step");
    };

    let mut bf16_cache = prefill(KvQuant::None);
    decode_once(&mut bf16_cache);
    let bf16_bytes = bf16_cache.resident_bytes();

    for &quant in crate::ALL_KV_QUANTS {
        // `KvQuant::None` takes the else branch and belongs there: it builds no
        // store and its residency IS the bf16 baseline. Paged storage is a
        // separate lifecycle selected by a CLI flag, not by the codec, and is
        // off here.
        let mut cache = prefill(quant);
        let store_after_prefill = cache.storage().resident_bytes();

        if quant.materialises_packed_store() {
            // No decode step here: `Mixed` / `RotK` refuse
            // `update()` by contract (they must go through `update_and_sdpa`),
            // and the property under test on this branch is only that the store
            // was built.
            assert!(
                store_after_prefill > 0,
                "{quant:?} must build its packed store at exit_prefill — decode \
                 reads it, or one of its axes has no mirror to read"
            );
        } else {
            decode_once(&mut cache);
            let store_bytes = cache.storage().resident_bytes();
            assert_eq!(
                store_after_prefill, 0,
                "{quant:?} reads no packed store at decode, so exit_prefill must \
                 build none — it built {store_after_prefill} bytes"
            );
            assert_eq!(
                store_bytes, 0,
                "{quant:?} allocated a packed store during decode — some decode \
                 path still routes into the codec body, so the store it reads is \
                 the one exit_prefill was told not to build"
            );
            assert_eq!(
                cache.resident_bytes(),
                bf16_bytes,
                "{quant:?} holds only its two bf16 mirrors, so its residency must \
                 equal plain bf16 at the same shape"
            );
        }
    }
}

#[test]
fn k8v4_warm_ttft_freezes_codec() {
    assert_shortcut_codec(KvQuant::K8V4, k8_codec_seq);
}

#[test]
fn k8v8_warm_ttft_freezes_codec() {
    assert_shortcut_codec(KvQuant::K8V8, k8_codec_seq);
}

/// Fused-symmetric contract: `Iso3Sym` is NOT a warm-TTFT shortcut codec. Its
/// decode is the quant-V flash kernel over both packed iso rings (on GPU) / a
/// re-quantise of both axes (on CPU), so NEITHER bf16 seed is materialised
/// (`feeds_bf16_k_at_decode` and `feeds_bf16_v_at_decode` are both `false`) and
/// the K codec advances one position per decode step rather than freezing.
#[test]
fn iso_sym3_fused_no_seed_at_decode() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::Iso3Sym, TEST_MAX_SEQ);
    let out = drive_one_decode(&mut cache, device);

    // Fused symmetric: neither bf16 seed is materialised — the codec exists to
    // delete that mirror.
    assert!(
        !out.k_seed_live,
        "Iso3Sym: exit_prefill must NOT populate decode_fp16_k (fused, no mirror)"
    );
    assert!(
        !out.v_seed_live,
        "Iso3Sym: exit_prefill must NOT populate decode_fp16_v (fused, no mirror)"
    );
    // The K codec advances at decode — no warm-TTFT shortcut.
    assert_eq!(
        iso_sym3_k_codec_seq(&cache),
        TEST_PREFILL_SEQ + 1,
        "Iso3Sym: K codec MUST advance by 1 per decode step (no bf16 seed to freeze on)"
    );
    assert_eq!(
        out.offset,
        TEST_PREFILL_SEQ + 1,
        "Iso3Sym offset after one decode step"
    );
}

/// K-only contract: IsoKOnly3 quantises K at every decode step and
/// never reads the bf16 K seed. The K-seed materialisation is gated on
/// `KvQuant::feeds_bf16_k_at_decode`, which is `false` for the K-only family:
/// the K seed is ABSENT while the V seed stays live, the K codec still
/// advances one position per decode step, and the reported residency drops by the
/// bf16 K-seed cost. Output is byte-unchanged.
#[test]
fn iso_k_only3_quant_at_decode() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::IsoKOnly3, TEST_MAX_SEQ);
    let out = drive_one_decode(&mut cache, device);

    // The bf16 K seed is NOT allocated for the K-only family —
    // the decode body never reads it, so exit_prefill skips it.
    assert!(
        !out.k_seed_live,
        "IsoKOnly3: exit_prefill must NOT populate decode_fp16_k (K-only never reads it)"
    );
    // The bf16 V seed IS still live — K-only decode reads it via
    // update_decode_fp16_v_only.
    assert!(
        out.v_seed_live,
        "IsoKOnly3: decode_fp16_v MUST stay live (V seed feeds the K-only decode path)"
    );
    // The K-only codec still runs at decode — proof the warm-TTFT shortcut does
    // NOT short-circuit this path (no decode_fp16_k.is_some() gate in the body).
    assert_eq!(
        iso_k_only3_k_codec_seq(&cache),
        TEST_PREFILL_SEQ + 1,
        "IsoKOnly3: K codec MUST advance by 1 per decode step (quant-at-decode), \
         not freeze at the prefill length — the bf16 K seed is NOT consulted here"
    );
    assert_eq!(
        out.offset,
        TEST_PREFILL_SEQ + 1,
        "IsoKOnly3 offset after one decode step"
    );
    // Residency: the total must be exactly the two things this cache really
    // holds — whatever the iso3 K store allocated, plus the filled prefix of
    // the surviving bf16 V seed. Anchored to the live buffers, not to a
    // nominal bit-width: iso3 blocks also carry per-group quaternions, scales
    // and norms, so a "3 bits per element" figure would not be this cache's
    // memory. Fails if the absent K seed is counted anyway, if the V seed is
    // missed, or if either is double-counted.
    let v_seed = cache
        .decode_fp16_v_for_test()
        .expect("V seed is live (asserted above)");
    let expected = cache.storage().resident_bytes() + filled_mirror_bytes(v_seed, out.offset);
    assert_eq!(
        out.resident_bytes, expected,
        "IsoKOnly3: resident_bytes must equal the iso3 K store plus the bf16 V seed's \
         filled prefix and nothing else",
    );
}

/// K-only contract for the rotor family — same as IsoKOnly3 but rotor3 K.
/// Pins the truth that RotorKOnly3 does NOT consult `decode_fp16_k`. The
/// dead K seed was dropped; the K rotor codec still runs every decode step.
#[test]
fn rotor_k_only3_quant_at_decode() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::RotorKOnly3, TEST_MAX_SEQ);
    let out = drive_one_decode(&mut cache, device);

    assert!(
        !out.k_seed_live,
        "RotorKOnly3: exit_prefill must NOT populate decode_fp16_k (K-only never reads it)"
    );
    assert!(
        out.v_seed_live,
        "RotorKOnly3: decode_fp16_v MUST stay live (V seed feeds the K-only decode path)"
    );
    assert_eq!(
        rotor_k_only3_k_codec_seq(&cache),
        TEST_PREFILL_SEQ + 1,
        "RotorKOnly3: K rotor codec MUST advance by 1 per decode step (quant-at-decode); \
         the asym-variant rustdoc claim that RotorKOnly3 bypasses K once the seed is live \
         is FALSE — the body has no seed gate"
    );
    assert_eq!(
        out.offset,
        TEST_PREFILL_SEQ + 1,
        "RotorKOnly3 offset after one decode step"
    );
    // Residency: the total must be exactly the two things this cache really
    // holds — whatever the rotor3 K store allocated (blocks plus the static
    // rotor table, and the GPU ring once one is live), plus the filled prefix
    // of the surviving bf16 V seed. Anchored to the live buffers, not to a
    // nominal bit-width. Fails if the absent K seed is counted anyway, if the
    // V seed is missed, or if either is double-counted.
    let v_seed = cache
        .decode_fp16_v_for_test()
        .expect("V seed is live (asserted above)");
    let expected = cache.storage().resident_bytes() + filled_mirror_bytes(v_seed, out.offset);
    assert_eq!(
        out.resident_bytes, expected,
        "RotorKOnly3: resident_bytes must equal the rotor3 K store plus the bf16 V seed's \
         filled prefix and nothing else",
    );
}

/// The `Mixed` / `RotK` bf16 mirror exists for a cross-layer-KV consumer and
/// for nothing else, so `exit_prefill` builds it exactly when the cache says
/// its architecture shares K/V.
///
/// The two arms are the same codec, the same geometry and the same prefill —
/// only `with_shares_kv` differs. The dense arm must hold its packed store and
/// nothing more; the shared arm must hold that store plus both mirrors, and the
/// difference must be exactly the mirror pair. Anchoring the delta to the live
/// buffers (rather than to a nominal bit-width) is what makes this fail if a
/// mirror is half-elided, counted twice, or replaced by a shorter one.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn mixed_family_mirror_is_built_exactly_when_the_arch_shares_kv() {
    let device = Device::Cpu;

    let prefill = |quant: KvQuant, shares_kv: bool| -> KvCache {
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ).with_shares_kv(shares_kv);
        cache.enter_prefill();
        let shape = [1i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
        let n: usize = shape.iter().map(|&d| d as usize).product();
        cache
            .update(
                &f32_arr(&vec![0.123f32; n], &shape),
                &f32_arr(&vec![0.456f32; n], &shape),
                device,
            )
            .expect("prefill chunk");
        cache.exit_prefill(device).expect("exit_prefill");
        cache
    };

    // Both members of the Mixed machinery, at the codec shapes `--kv-bits 4`
    // and `--kv-quant rot_k_v4g64` resolve to.
    let codecs = [
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        },
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 64,
        },
    ];

    for quant in codecs {
        let dense = prefill(quant, false);
        assert!(
            dense.decode_fp16_k_for_test().is_none() && dense.decode_fp16_v_for_test().is_none(),
            "{quant:?}: with no cross-layer KV sharing nothing reads the bf16 mirror, so \
             exit_prefill must build neither axis"
        );
        let dense_store = dense.storage().resident_bytes();
        assert!(
            dense_store > 0,
            "{quant:?}: the packed store is what decode reads and must still be built"
        );
        assert_eq!(
            dense.resident_bytes(),
            dense_store,
            "{quant:?}: a dense cache holds its packed store and nothing else"
        );

        let shared = prefill(quant, true);
        let k = shared
            .decode_fp16_k_for_test()
            .expect("shared-KV K mirror is the share handed to consumer layers");
        let v = shared
            .decode_fp16_v_for_test()
            .expect("shared-KV V mirror is the share handed to consumer layers");
        let mirror_pair =
            filled_mirror_bytes(k, TEST_PREFILL_SEQ) + filled_mirror_bytes(v, TEST_PREFILL_SEQ);
        assert_eq!(
            shared.storage().resident_bytes(),
            dense_store,
            "{quant:?}: the packed store is identical under both topologies — the \
             topology decides the mirror, never the store"
        );
        assert_eq!(
            shared.resident_bytes(),
            dense_store + mirror_pair,
            "{quant:?}: a shared-KV cache holds the same store plus both mirrors"
        );
        assert_eq!(
            shared.resident_bytes() - dense.resident_bytes(),
            mirror_pair,
            "{quant:?}: the whole difference between the two topologies is the mirror pair"
        );
    }
}

/// `shares_kv` moves the `Mixed` machinery and **nothing else**.
///
/// Every other codec's mirror disposition is a property of the codec alone, so
/// building the same cache under both topologies must reproduce byte-identical
/// residency for all of them. Without this, widening `shares_kv`'s reach — to a
/// codec whose decode genuinely needs its mirror on every arch — would silently
/// strand that codec's decode on zeros, and only a served model would say so.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn shares_kv_moves_only_the_mixed_machinery() {
    let device = Device::Cpu;

    let prefill = |quant: KvQuant, shares_kv: bool| -> KvCache {
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ).with_shares_kv(shares_kv);
        cache.enter_prefill();
        let shape = [1i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
        let n: usize = shape.iter().map(|&d| d as usize).product();
        cache
            .update(
                &f32_arr(&vec![0.123f32; n], &shape),
                &f32_arr(&vec![0.456f32; n], &shape),
                device,
            )
            .expect("prefill chunk");
        cache.exit_prefill(device).expect("exit_prefill");
        cache
    };

    for &quant in crate::ALL_KV_QUANTS {
        let dense = prefill(quant, false);
        let shared = prefill(quant, true);
        let same_k =
            dense.decode_fp16_k_for_test().is_some() == shared.decode_fp16_k_for_test().is_some();
        let same_v =
            dense.decode_fp16_v_for_test().is_some() == shared.decode_fp16_v_for_test().is_some();
        if quant.uses_mixed_path() {
            assert!(
                !same_k && !same_v,
                "{quant:?} is the Mixed machinery: its mirror must follow the topology"
            );
            assert!(
                shared.resident_bytes() > dense.resident_bytes(),
                "{quant:?}: the shared-KV arm holds strictly more (the mirror pair)"
            );
        } else {
            assert!(
                same_k && same_v,
                "{quant:?} mirrors on its own codec properties — `shares_kv` must not \
                 move either axis"
            );
            assert_eq!(
                dense.resident_bytes(),
                shared.resident_bytes(),
                "{quant:?}: residency must be byte-identical under both topologies"
            );
        }
    }
}

/// A branch clone of a shared-KV producer is still a shared-KV producer.
///
/// `try_deep_clone` is how a request branches (prompt-cache reuse, speculative
/// rollback). A clone that forgot the topology would build no mirror at its
/// next `exit_prefill`, and the branch's first shared-source decode step would
/// then be refused — a live request failing on a field copy. Asserted in both
/// directions so a clone that hard-codes either value fails.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn deep_clone_carries_the_sharing_topology() {
    for shares in [false, true] {
        let cache = KvCache::with_quant_max_seq(
            KvQuant::Mixed {
                k_bits: 8,
                v_bits: 4,
                k_group_size: 64,
                v_group_size: 64,
            },
            TEST_MAX_SEQ,
        )
        .with_shares_kv(shares);
        let clone = cache.try_deep_clone().expect("deep clone");
        assert_eq!(
            clone.shares_kv(),
            shares,
            "a branch clone must carry the producer's topology, not re-derive it"
        );
    }
}

/// The `with_quant*` constructors default `shares_kv` to `false`, and on every
/// dense architecture in the tree that default is the only thing that decides.
///
/// No dense arch builder calls `with_shares_kv` at all — Gemma4's loop and the
/// speculative verifier stacks are the whole caller set — so the constructor
/// default alone keeps the `Mixed` / `RotK` bf16 mirror off Bonsai, Qwen3,
/// Qwen3.5-MoE and everything else. Flipping it to `true` restores two full
/// bf16 buffers per layer across the tree, which is the entire residency this
/// codec family was carrying, and every other test in this file sets the flag
/// explicitly, so none of them would notice. Pin the constructors, then pin the
/// consequence the default exists for.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn with_quant_constructors_default_to_no_cross_layer_sharing() {
    let device = Device::Cpu;
    let mixed = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };

    // Every public constructor that takes no topology argument. All three
    // funnel through `with_quant_max_seq`, but a future one need not, and the
    // windowed form is the one Gemma4 itself calls.
    let defaults = [
        ("with_quant", KvCache::with_quant(mixed)),
        (
            "with_quant_max_seq",
            KvCache::with_quant_max_seq(mixed, TEST_MAX_SEQ),
        ),
        (
            "with_quant_max_seq_window(None)",
            KvCache::with_quant_max_seq_window(mixed, TEST_MAX_SEQ, None),
        ),
        (
            "with_quant_max_seq_window(Some)",
            KvCache::with_quant_max_seq_window(mixed, TEST_MAX_SEQ, Some(64)),
        ),
    ];
    for (name, cache) in defaults {
        assert!(
            !cache.shares_kv(),
            "{name}: cross-layer KV sharing is a per-arch declaration, so an \
             undeclared cache must not claim it — this default is what every \
             dense architecture runs under"
        );
    }

    // The consequence, on both members of the Mixed machinery: a producer built
    // through the default holds its packed store and nothing else.
    for quant in [
        mixed,
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 64,
        },
    ] {
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
        cache.enter_prefill();
        let shape = [1i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
        let n: usize = shape.iter().map(|&d| d as usize).product();
        cache
            .update(
                &f32_arr(&vec![0.123f32; n], &shape),
                &f32_arr(&vec![0.456f32; n], &shape),
                device,
            )
            .expect("prefill chunk");
        cache.exit_prefill(device).expect("exit_prefill");
        assert!(
            cache.decode_fp16_k_for_test().is_none() && cache.decode_fp16_v_for_test().is_none(),
            "{quant:?}: a cache built through the default declares no sharing, so nothing \
             reads the bf16 mirror and exit_prefill must build neither axis"
        );
        assert_eq!(
            cache.resident_bytes(),
            cache.storage().resident_bytes(),
            "{quant:?}: a default-built producer holds its packed store and nothing else"
        );
    }
}
