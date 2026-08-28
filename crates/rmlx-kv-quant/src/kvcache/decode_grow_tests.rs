//! Decode-side growth of the provisioned `max_seq`.
//!
//! `max_seq` is provisioned lazily: it starts at the small default and the
//! prefill path grows it as the prompt fills. Decode then keeps appending, so
//! it has to grow it too — otherwise the cap freezes at whatever the prompt
//! happened to need and every store that bounds its capacity by `max_seq`
//! stops accepting appends the moment generation crosses it.
//!
//! The headroom a prompt leaves behind is incidental (`next_pow2(prompt) -
//! prompt`), which is why this went unnoticed: a prompt that lands well below
//! the provisioned bound generates for thousands of tokens before it bites,
//! while a prompt that *saturates* the bound dies on the very first generated
//! token. These tests pin the saturated case, the just-under-the-boundary
//! case, and the two ways the cap is allowed to say no.
//!
//! # Why the rotor K-only codec
//!
//! It is the codec whose ring reports the overflow **loudly** (`QuantKGpuRing::
//! append_encoded` errors), so a regression here is unambiguous rather than a
//! silently short attention prefix. The growth itself is model-agnostic and
//! lives on the shared storage `max_seq`, not in this codec.
//!
//! # Coverage boundary
//!
//! `HEAD_DIM` here is a power of two, so every step takes the **fused**
//! flash-decode path. The legacy (non-fused) path — which a non-power-of-two
//! `head_dim` selects, and which feeds the same ring — is covered by
//! `tests/rotor_decode_grow_legacy.rs`. It lives in its own binary because it
//! needs the process-global rotor-QJL toggle off. Both are needed: a fix to one
//! path is not a fix to the other.
//!
//! # No env dependence
//!
//! These never touch `RMLX_ROTOR_QJL`: pre-seeding the store with a rotor table
//! pins its QJL decision (the codec only revisits it while `rotors.is_empty()`),
//! so the ring-backed path is reached deterministically.

use super::KvCache;
use crate::clifford::make_rotor_table;
use crate::quant::KvQuant;
use crate::rotorquant::n_groups_for;
use crate::storage::{bf16_round, KvStorage, QuantRotorK3, QuantRotorK4};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_core::error::Error;
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{Array, Device, Dtype};

const KV_H: i32 = 2;
const N_Q_HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;
/// Small enough to saturate cheaply in a unit test.
const MAX_SEQ: i32 = 128;

/// LCG seeds for the prefill chunk and for decode step `i`.
const PREFILL_SEED: u64 = 1;
const DECODE_SEED_BASE: u64 = 10;

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("f32_array")
}

/// Read a 1-D f32 `Array` back to the host.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn read_f32(arr: &Array) -> Vec<f32> {
    arr.eval().expect("eval");
    let bytes = arr.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Rotor K-only cache with QJL pinned off and an explicit `max_seq`.
fn seeded_cache(quant: KvQuant, max_seq: i32, ceiling: Option<i32>) -> KvCache {
    let n_groups = n_groups_for(HEAD_DIM as usize);
    let rotors = make_rotor_table(0, 0, n_groups);
    let shape = vec![1, KV_H, 0, HEAD_DIM];

    let storage = if quant == KvQuant::RotorKOnly4 {
        KvStorage::RotorKOnly4 {
            k: Some(QuantRotorK4::from_cpu_blocks(
                rotors,
                None,
                Vec::new(),
                shape,
                0,
            )),
            max_seq,
        }
    } else {
        KvStorage::RotorKOnly3 {
            k: Some(QuantRotorK3::from_cpu_blocks(
                rotors,
                None,
                Vec::new(),
                shape,
                0,
            )),
            max_seq,
        }
    };
    let cache = KvCache::from_storage(storage, quant, 0, 0, DispatchPolicy::default(), false);
    match ceiling {
        Some(c) => cache.with_max_seq_ceiling(c),
        None => cache,
    }
}

/// The K row for `(seq, head)` of the synthetic input, mirroring what
/// [`prefill`] / [`decode_steps`] feed the cache.
fn k_row(seq: i32, head: i32, prefill_len: i32) -> Vec<f32> {
    let hd = HEAD_DIM as usize;
    let h = head as usize;
    if seq < prefill_len {
        let data = lcg_data((prefill_len * KV_H * HEAD_DIM) as usize, PREFILL_SEED);
        let base = h * prefill_len as usize * hd + seq as usize * hd;
        data[base..base + hd].to_vec()
    } else {
        let step = (seq - prefill_len) as u64;
        let data = lcg_data((KV_H * HEAD_DIM) as usize, DECODE_SEED_BASE + step);
        let base = h * hd;
        data[base..base + hd].to_vec()
    }
}

/// The codec's per-token L2 norm, recomputed on the CPU.
///
/// Mirrors `rotorquant`'s encode: `sqrt(sum(x^2))`, floored at `1e-8`.
fn cpu_norm(row: &[f32]) -> f32 {
    let sq: f32 = row.iter().map(|&x| x * x).sum();
    sq.sqrt().max(1e-8)
}

/// One prefill chunk of `n` tokens through the production entry point.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn prefill(cache: &mut KvCache, n: i32, device: Device) {
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let sz = (n * KV_H * HEAD_DIM) as usize;
    let k = f32_array(&lcg_data(sz, PREFILL_SEED), &[1, KV_H, n, HEAD_DIM]);
    let v = f32_array(&lcg_data(sz, 2), &[1, KV_H, n, HEAD_DIM]);
    let q = f32_array(
        &lcg_data((n * N_Q_HEADS * HEAD_DIM) as usize, 3),
        &[1, N_Q_HEADS, n, HEAD_DIM],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");
}

/// `n` single-token decode steps. Returns the first error, if any.
fn decode_steps(cache: &mut KvCache, n: u64, device: Device) -> Result<(), Error> {
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    for step in 0..n {
        let one = (KV_H * HEAD_DIM) as usize;
        let k1 = f32_array(
            &lcg_data(one, DECODE_SEED_BASE + step),
            &[1, KV_H, 1, HEAD_DIM],
        );
        let v1 = f32_array(&lcg_data(one, 20 + step), &[1, KV_H, 1, HEAD_DIM]);
        let q1 = f32_array(
            &lcg_data((N_Q_HEADS * HEAD_DIM) as usize, 30 + step),
            &[1, N_Q_HEADS, 1, HEAD_DIM],
        );
        let out = cache.update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)?;
        out.eval()?;
    }
    Ok(())
}

/// The rotor store's GPU ring, if live.
fn ring_norms(cache: &KvCache, kv_seq: i32, device: Device) -> Option<Vec<f32>> {
    let view = if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.packed_view(kv_seq, device)
    } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.packed_view(kv_seq, device)
    } else {
        return None;
    };
    if let Ok(Some((_, _, norms))) = view {
        Some(crate::test_utils::read_sideband_plane(&norms))
    } else {
        None
    }
}

/// The bf16 V mirror's per-token slice at `seq`, flattened over heads.
///
/// The mirror is `[B, kv_h, max_seq, head_dim]`; a slot that was never written
/// still reads as its `zeros()` init value.
fn v_mirror_slot(cache: &KvCache, seq: i32, device: Device) -> Option<(i32, Vec<f32>)> {
    let v = cache.decode_fp16_v.as_ref()?;
    let shape = v.shape();
    let (kv_h, cap, head_dim) = (shape[1], shape[2], shape[3]);
    let row = v
        .slice(
            &[0, 0, seq, 0],
            &[1, kv_h, seq + 1, head_dim],
            &[1, 1, 1, 1],
            device,
        )
        .ok()?
        .astype(Dtype::F32, device)
        .ok()?;
    Some((cap, read_f32(&row)))
}

/// A prompt that exactly saturates the provisioned `max_seq` must still decode.
///
/// This is the zero-headroom case: `prefill == max_seq`, so the very first
/// generated token crosses the bound. Pre-fix the ring rejected it with
/// `needed=129 exceeds max_seq=128` and decode died on step 1.
///
/// Also pins the V side. The K ring is only half the story: the bf16 V mirror is
/// allocated `[B, kv_h, max_seq, head_dim]` and is bound by the *same* scalar, so
/// growing the ring alone would let K extend while every V append landed
/// out of bounds — a `slice_update` no-op, silent. Asserting `offset` alone
/// cannot see that: `offset` advances either way.
fn saturated_prompt_decodes(quant: KvQuant, label: &str) {
    let device = Device::Gpu;
    let mut cache = seeded_cache(quant, MAX_SEQ, None);
    prefill(&mut cache, MAX_SEQ, device);
    assert_eq!(cache.offset(), MAX_SEQ, "{label}: prompt saturates max_seq");

    decode_steps(&mut cache, 64, device)
        .unwrap_or_else(|e| panic!("{label}: decode past a saturated max_seq must not error: {e}"));

    let total = MAX_SEQ + 64;
    assert_eq!(
        cache.offset(),
        total,
        "{label}: every decode step must land"
    );

    // V mirror grew in lockstep with the ring …
    let (cap, last) = v_mirror_slot(&cache, total - 1, device)
        .unwrap_or_else(|| panic!("{label}: V mirror absent — the test would prove nothing"));
    assert!(
        cap >= total,
        "{label}: V mirror capacity {cap} < sequence {total} — V appends past the \
         old bound were silently dropped while K grew"
    );
    // … and the last decoded token actually landed in it (a dropped append
    // leaves the slot at its zero init).
    assert!(
        last.iter().any(|&x| x != 0.0),
        "{label}: V mirror slot at seq={} is all zeros — the append was silently \
         dropped and attention would read a zeroed value vector",
        total - 1
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test decode_grow -- --ignored --test-threads=1"]
fn rotor3_decode_grows_past_a_saturated_max_seq() {
    if skip_if_no_gpu_env() {
        return;
    }
    saturated_prompt_decodes(KvQuant::RotorKOnly3, "rotor3");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test decode_grow -- --ignored --test-threads=1"]
fn rotor4_decode_grows_past_a_saturated_max_seq() {
    if skip_if_no_gpu_env() {
        return;
    }
    saturated_prompt_decodes(KvQuant::RotorKOnly4, "rotor4");
}

/// A prompt just under the bound must decode straight through it.
///
/// The incidental headroom (here 6 tokens) is what made this look healthy on a
/// short generation: the bound only bites once generation outruns it.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test decode_grow -- --ignored --test-threads=1"]
fn rotor3_decode_grows_across_the_boundary_from_just_under() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let prefill_len = MAX_SEQ - 6;
    let mut cache = seeded_cache(KvQuant::RotorKOnly3, MAX_SEQ, None);
    prefill(&mut cache, prefill_len, device);

    decode_steps(&mut cache, 32, device)
        .unwrap_or_else(|e| panic!("decode across the boundary must not error: {e}"));

    assert_eq!(cache.offset(), prefill_len + 32, "every decode step lands");
}

/// Growth stops at the `--max-ctx` ceiling — loudly, and with the typed error.
///
/// The backstop matters as much as the growth: a request that genuinely cannot
/// fit the configured context must be rejected, never quietly truncated.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test decode_grow -- --ignored --test-threads=1"]
fn decode_past_the_ceiling_errors_loudly() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    // Ceiling == prompt: there is no room to generate even one token.
    let mut cache = seeded_cache(KvQuant::RotorKOnly3, MAX_SEQ, Some(MAX_SEQ));
    prefill(&mut cache, MAX_SEQ, device);

    let err = decode_steps(&mut cache, 1, device)
        .expect_err("a decode step past the ceiling must be rejected, not served");
    let seen = format!("{err:?}");
    let Error::KvCeilingExceeded { requested, ceiling } = err else {
        panic!("expected Error::KvCeilingExceeded, got: {seen}");
    };
    assert_eq!(requested, MAX_SEQ + 1, "reports the sequence it needed");
    assert_eq!(ceiling, MAX_SEQ, "reports the configured ceiling");
}

/// No append is silently dropped: every ring slot matches a CPU re-encode.
///
/// The failure this guards against is not an error but a **gap** — an append
/// whose `slice_update` lands out of bounds is a no-op, leaving the slot at its
/// zero-init value while `offset` marches on. Attention then reads a zeroed
/// token as if it were real.
///
/// The ring's per-token L2 norm is the cheapest field to re-derive exactly
/// (`sqrt(sum(x^2))`, un-quantised), so it is the one checked here: a dropped
/// append reads back as `0.0` against a re-encode that never is.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test decode_grow -- --ignored --test-threads=1"]
fn decode_past_max_seq_lands_every_append_in_the_ring() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let prefill_len = MAX_SEQ;
    let n_decode = 64_i32;
    let mut cache = seeded_cache(KvQuant::RotorKOnly3, MAX_SEQ, None);
    prefill(&mut cache, prefill_len, device);
    decode_steps(&mut cache, n_decode as u64, device).unwrap_or_else(|e| panic!("decode: {e}"));

    let total = prefill_len + n_decode;
    let norms = ring_norms(&cache, total, device)
        .unwrap_or_else(|| panic!("ring must be live — otherwise this test proves nothing"));
    assert_eq!(
        norms.len() as i32,
        total * KV_H,
        "ring must carry one norm per (token, kv head) over the whole sequence"
    );

    for seq in 0..total {
        for head in 0..KV_H {
            #[allow(
                clippy::indexing_slicing,
                reason = "length asserted equal to total * KV_H above"
            )]
            let got = norms[(seq * KV_H + head) as usize];
            // The re-encode is rounded to the width the ring stores, because
            // that is what a decode reconstructs with — comparing against an
            // unrounded f32 would be measuring the narrowing, not the append.
            //
            // The allowance is exactly one bf16 step at the reference, and no
            // more. The two sides are independent float reductions — an MSL
            // kernel with fma contraction on against a sequential Rust sum — so
            // they can disagree in the last f32 bit, and a disagreement that
            // straddles a bf16 rounding boundary lands them on adjacent stored
            // values. Pinning bit-equality would make that a red test on a
            // toolchain bump rather than on a defect. It stays a *stored-width*
            // bound rather than a relative epsilon: the failure this exists to
            // catch is a dropped append, which reads back as 0.0 and is caught
            // by the assertion above whatever the tolerance, and a mis-strided
            // read, which would have to coincide with the neighbouring bf16
            // value to survive.
            let want = bf16_round(cpu_norm(&k_row(seq, head, prefill_len)));
            // The next representable bf16 above `want`: a bf16 value has zero
            // low 16 bits as an f32, so the successor is one increment of the
            // stored half.
            let one_bf16_ulp = f32::from_bits(want.abs().to_bits() + (1 << 16)) - want.abs();
            assert!(
                got > 0.0,
                "ring slot (seq={seq}, head={head}) reads back as {got} — the append was \
                 dropped and attention would read a zeroed token"
            );
            assert!(
                (got - want).abs() <= one_bf16_ulp,
                "ring slot (seq={seq}, head={head}) = {got} is more than one stored step \
                 from the CPU re-encode rounded to the sideband width ({want}, step \
                 {one_bf16_ulp})"
            );
        }
    }
}
