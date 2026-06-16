//! GPU layout round-trip for the Paged K storage handoff.
//!
//! `KvCache::update_paged` quantizes the head-major `new_k` (`[B, kv_h, S, D]`)
//! and feeds the codes to `PagedKStorage::append`, which lays them out
//! token-major in fixed-size page slabs (`words_per_token` per token slot).
//! Quantizing head-major and indexing token-major scrambled per-head K on any
//! chunk spanning more than one token-per-head (multi-page prefill or
//! multi-append decode) when `kv_h > 1`. The fix reorders `new_k` to
//! sequence-major before quantizing and transposes the dequant output back.
//!
//! This test reproduces the exact `update_paged` handoff (transpose → quantize
//! → append → gather → dequant → transpose-back) and asserts the round-trip
//! reconstructs the true head-major K within q8 noise across a multi-page,
//! multi-append, multi-head schedule. `#[ignore]` — Metal context; run with
//! `--ignored --test-threads=1`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    unsafe_code,
    missing_docs
)]

use rmlx_kv_quant::paged::PagedKStorage;
use rmlx_kv_quant::q8_msl::{q8_dequantize_gpu, q8_quantize_gpu};
use rmlx_mlx::{Array, Device, Dtype};

fn arr(vals: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), vals.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("arr")
}

fn to_vec(a: &Array) -> Vec<f32> {
    a.eval().expect("eval");
    a.to_bytes()
        .expect("bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk")))
        .collect()
}

/// Distinct, small per-(head,token,dim) value so a head transposition (≥ ~0.1)
/// is unmistakable against q8 noise.
fn expected(h: i32, t: i32, dd: i32) -> f32 {
    (h * 100 + t * 5 + dd % 7) as f32 * 0.001
}

/// Head-major `[1, kv_h, seq, d]` chunk for tokens `[base_t, base_t+seq)`.
fn head_major(kv_h: i32, seq: i32, d: i32, base_t: i32) -> Vec<f32> {
    let mut v = Vec::with_capacity((kv_h * seq * d) as usize);
    for h in 0..kv_h {
        for t in 0..seq {
            for dd in 0..d {
                v.push(expected(h, base_t + t, dd));
            }
        }
    }
    v
}

/// Mirror of `update_paged`'s K handoff with the sequence-major fix.
fn append_chunk(pk: &mut PagedKStorage, kv_h: i32, seq: i32, d: i32, base_t: i32) {
    let shape = [1, kv_h, seq, d];
    let k_hm = arr(&head_major(kv_h, seq, d, base_t), &shape);
    let k_sm = k_hm
        .transpose(&[0, 2, 1, 3], Device::Gpu)
        .expect("transpose")
        .contiguous(Device::Gpu)
        .expect("contiguous");
    let (codes, scales) = q8_quantize_gpu(&k_sm, Device::Gpu).expect("quantize");
    pk.append(&shape, codes, scales, Device::Gpu)
        .expect("append");
}

fn dequant_head_major(pk: &PagedKStorage, kv_h: i32, s_total: i32, d: i32) -> Vec<f32> {
    let (codes, scales) = pk.gather(Device::Gpu).expect("gather");
    let sm_shape = [1, s_total, kv_h, d];
    let out = q8_dequantize_gpu(&codes, &scales, &sm_shape, Dtype::F32, Device::Gpu)
        .expect("dequant")
        .transpose(&[0, 2, 1, 3], Device::Gpu)
        .expect("transpose")
        .contiguous(Device::Gpu)
        .expect("contiguous");
    to_vec(&out)
}

fn max_err(out: &[f32], kv_h: i32, s_total: i32, d: i32) -> f32 {
    let mut m = 0.0f32;
    let mut i = 0usize;
    for h in 0..kv_h {
        for t in 0..s_total {
            for dd in 0..d {
                m = m.max((out[i] - expected(h, t, dd)).abs());
                i += 1;
            }
        }
    }
    m
}

#[test]
#[ignore = "GPU Metal context — run with --ignored --test-threads=1"]
fn paged_k_multipage_multiappend_multi_head_roundtrip() {
    if std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1") {
        return;
    }
    // page_tokens=32 (default). A 70-token prefill spans 3 pages, then 5
    // single-token decode appends. kv_h=8, head_dim=128 (Bonsai-ish).
    let (kv_h, d) = (8i32, 128i32);
    let page_tokens = 32i32;
    let max_seq = 256i32;
    let n_pages = ((max_seq + page_tokens - 1) / page_tokens) as usize;
    let mut pk = PagedKStorage::new(max_seq, page_tokens, n_pages);

    append_chunk(&mut pk, kv_h, 70, d, 0);
    for t in 70..75 {
        append_chunk(&mut pk, kv_h, 1, d, t);
    }
    let s_total = 75;
    let out = dequant_head_major(&pk, kv_h, s_total, d);
    let m = max_err(&out, kv_h, s_total, d);
    assert!(
        m < 0.05,
        "paged K multipage/multiappend max abs error {m} — expected q8 noise, not head scramble"
    );
}

/// Pre-fix reproduction: the head-major handoff (quantize head-major, append
/// token-major, dequant head-major — no reorder) scrambles per-head K on a
/// multi-page, multi-head chunk. Confirms the fix's reorder is load-bearing.
#[test]
#[ignore = "GPU Metal context — run with --ignored --test-threads=1"]
fn paged_k_head_major_handoff_scrambles() {
    if std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1") {
        return;
    }
    let (kv_h, d) = (8i32, 128i32);
    let page_tokens = 32i32;
    let max_seq = 256i32;
    let n_pages = ((max_seq + page_tokens - 1) / page_tokens) as usize;
    let mut pk = PagedKStorage::new(max_seq, page_tokens, n_pages);

    // Pre-fix handoff: quantize head-major directly (no transpose), append a
    // prefill chunk then per-token decode steps, dequant head-major (no
    // transpose-back). The second-and-later chunks land at a token offset where
    // the head-major store and the head-major reshape disagree for kv_h > 1.
    let append_hm = |pk: &mut PagedKStorage, seq: i32, base_t: i32| {
        let shape = [1, kv_h, seq, d];
        let k_hm = arr(&head_major(kv_h, seq, d, base_t), &shape);
        let (codes, scales) = q8_quantize_gpu(&k_hm, Device::Gpu).expect("quantize");
        pk.append(&shape, codes, scales, Device::Gpu)
            .expect("append");
    };
    append_hm(&mut pk, 40, 0);
    for t in 40..45 {
        append_hm(&mut pk, 1, t);
    }
    let s_total = 45i32;
    let full_shape = [1, kv_h, s_total, d];
    let (g_codes, g_scales) = pk.gather(Device::Gpu).expect("gather");
    let out = q8_dequantize_gpu(&g_codes, &g_scales, &full_shape, Dtype::F32, Device::Gpu)
        .expect("dequant");
    let out = to_vec(&out);
    let m = max_err(&out, kv_h, s_total, d);
    assert!(
        m > 0.1,
        "expected head-major handoff to scramble a multi-page GQA chunk, got max err {m}"
    );
}

#[test]
#[ignore = "GPU Metal context — run with --ignored --test-threads=1"]
fn paged_k_single_head_control() {
    if std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1") {
        return;
    }
    let (kv_h, d) = (1i32, 128i32);
    let page_tokens = 32i32;
    let max_seq = 256i32;
    let n_pages = ((max_seq + page_tokens - 1) / page_tokens) as usize;
    let mut pk = PagedKStorage::new(max_seq, page_tokens, n_pages);
    append_chunk(&mut pk, kv_h, 70, d, 0);
    append_chunk(&mut pk, kv_h, 1, d, 70);
    let out = dequant_head_major(&pk, kv_h, 71, d);
    let m = max_err(&out, kv_h, 71, d);
    assert!(m < 0.05, "paged K kv_h=1 control max abs error {m}");
}
