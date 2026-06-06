//! Qwen3-VL-MoE text-decoder token-parity gate.
//!
//! Greedy-decodes a fixed TEXT-ONLY chat prompt at temp=0 through the rMLX
//! `Qwen3VlMoeText` decoder and asserts the first decoded token ids match the
//! mlx-vlm reference (captured with the same snapshot + prompt). This is the
//! decode-correctness gate for the 3D interleaved M-RoPE + plain-GQA + MoE path.
//!
//! Reference (mlx-vlm 0.5.0, greedy, temp=0):
//! prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one
//! word.<|im_end|>\n<|im_start|>assistant\n"
//! first ids = [59604 ("Paris"), 151645 (<|im_end|>), 198 ("\n"), ...]
//!
//! Gated behind `RMLX_VL_TEST_MODEL` + `#[ignore]` (large model load):
//! RMLX_VL_TEST_MODEL=/path/to/Qwen3-VL-30B-A3B-Instruct-4bit \
//! cargo test -p rmlx-models --test qwen3_vl_moe_text_parity -- --ignored --nocapture

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::float_cmp,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value
)]

use std::path::PathBuf;

use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};
use rmlx_models::qwen3_vl_moe;

/// The chat-templated text-only prompt token ids (verified via the snapshot
/// tokenizer / mlx-vlm `apply_chat_template`).
const PROMPT_IDS: &[u32] = &[
    151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 21806, 304, 825, 3409, 13, 151645, 198,
    151644, 77091, 198,
];

/// mlx-vlm reference greedy decode (first 3 meaningful tokens).
const REFERENCE_FIRST: &[u32] = &[59604, 151645, 198];

fn argmax_logits(logits: &Array, device: Device) -> u32 {
    let f = logits.astype(Dtype::F32, device).expect("astype");
    Array::eval(&f).expect("materialise");
    let bytes = f.to_bytes().expect("to_bytes");
    let row: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &x) in row.iter().enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    best.0 as u32
}

#[ignore]
#[test]
fn qwen3_vl_moe_text_decode_matches_mlx_vlm() {
    let Ok(p) = std::env::var("RMLX_VL_TEST_MODEL") else {
        eprintln!("RMLX_VL_TEST_MODEL not set — skipping qwen3_vl_moe text parity test");
        return;
    };
    let model_path = PathBuf::from(p);

    let device = Device::Gpu;
    let model = qwen3_vl_moe::load_text_from_path(&model_path).expect("load qwen3_vl_moe text");
    let n_layers = model.cfg.num_hidden_layers;

    let mut kv: Vec<KvCache> = (0..n_layers)
        .map(|_| KvCache::with_quant(KvQuant::None))
        .collect();

    // Prefill the prompt.
    let prefill_logits = model
        .forward_seq_with_cache(PROMPT_IDS, Some(&mut kv), device)
        .expect("prefill");
    let mut next = argmax_logits(&prefill_logits, device);

    let mut decoded: Vec<u32> = vec![next];
    // Decode a few more steps to validate the cached M-RoPE decode path.
    for _ in 0..4 {
        let step_logits = model
            .forward_seq_with_cache(&[next], Some(&mut kv), device)
            .expect("decode step");
        next = argmax_logits(&step_logits, device);
        decoded.push(next);
    }

    eprintln!("rMLX qwen3_vl_moe decoded ids: {decoded:?}");
    eprintln!("mlx-vlm reference first ids:   {REFERENCE_FIRST:?}");

    for (i, &r) in REFERENCE_FIRST.iter().enumerate() {
        assert_eq!(
            decoded[i], r,
            "token mismatch at position {i}: rMLX={} mlx-vlm={r}\n  full rMLX = {decoded:?}",
            decoded[i]
        );
    }
}
