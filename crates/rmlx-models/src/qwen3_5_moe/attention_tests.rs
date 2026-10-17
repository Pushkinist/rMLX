//! The shared prefill mask of the full-attention layers: its rows, its dtype,
//! and the steps that need none.

use rmlx_mlx::{zeros, Device, Dtype};

use super::{prefill_attention_mask, FullAttention};
use crate::layers::build_chunked_prefill_mask;
use crate::qwen3_5_moe::layers::{Linear, RmsNorm};

const HEAD_DIM: usize = 256;

/// A step with no array mask (the first chunk is causal, a decode step has one
/// row) builds none.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "every unwrap is on an MLX op over a fixture of known shape; a failure is a broken fixture"
)]
fn a_step_without_an_array_mask_builds_none() {
    for (offset, seq) in [(0, 1766), (4096, 1)] {
        let mask = prefill_attention_mask(HEAD_DIM, offset, seq, Dtype::Bf16, Device::Cpu).unwrap();
        assert!(mask.is_none(), "offset {offset}, {seq} rows");
    }
}

/// Off the GPU the node is not padded, so the mask has the chunk's rows, in
/// the dtype of the queries.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "every unwrap is on an MLX op over a fixture of known shape; a failure is a broken fixture"
)]
fn the_mask_has_the_chunks_rows_where_the_node_is_not_padded() {
    let mask = prefill_attention_mask(HEAD_DIM, 4096, 1766, Dtype::F16, Device::Cpu)
        .unwrap()
        .unwrap();
    assert_eq!(mask.shape(), vec![1, 1, 1766, 4096 + 1766]);
    assert_eq!(mask.dtype(), Dtype::F16);
}

/// On the GPU the mask has the rows of the padded attention node, so the layer
/// copies none.
#[test]
#[ignore = "builds the mask on the Metal GPU stream"]
#[allow(
    clippy::unwrap_used,
    reason = "every unwrap is on an MLX op over a fixture of known shape; a failure is a broken fixture"
)]
fn the_mask_has_the_rows_of_the_padded_node_on_the_gpu() {
    let mask = prefill_attention_mask(HEAD_DIM, 4096, 1766, Dtype::F16, Device::Gpu)
        .unwrap()
        .unwrap();
    assert_eq!(mask.shape(), vec![1, 1, 1792, 4096 + 1766]);
    assert_eq!(mask.dtype(), Dtype::F16);
    let aligned = prefill_attention_mask(HEAD_DIM, 4096, 2048, Dtype::Bf16, Device::Gpu)
        .unwrap()
        .unwrap();
    assert_eq!(aligned.shape(), vec![1, 1, 2048, 4096 + 2048]);
}

/// A layer whose queries are F16 takes the shared mask, built in bf16, and
/// casts it: MLX refuses a mask that does not promote to the output type.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "every unwrap is on an MLX op over a fixture of known shape; a failure is a broken fixture"
)]
fn a_layer_with_f16_queries_casts_a_bf16_mask() {
    let device = Device::Cpu;
    let (hidden, heads, head_dim, seq, offset) = (16i32, 2i32, 8i32, 3i32, 2i32);
    let plain = |rows: i32, cols: i32| Linear::Plain {
        weight: zeros(&[rows, cols], Dtype::F16, device).unwrap(),
    };
    let norm = || RmsNorm {
        weight: zeros(&[head_dim], Dtype::F16, device).unwrap(),
        eps: 1e-6,
    };
    let layer = FullAttention {
        q_proj: plain(heads * head_dim * 2, hidden),
        k_proj: plain(head_dim, hidden),
        v_proj: plain(head_dim, hidden),
        o_proj: plain(hidden, heads * head_dim),
        q_norm: norm(),
        k_norm: norm(),
        n_heads: heads as usize,
        n_kv_heads: 1,
        head_dim: head_dim as usize,
        scale: 0.35,
        rope_theta: 10_000.0,
        rope_dims: head_dim as usize,
    };
    let x = zeros(&[1, seq, hidden], Dtype::F16, device).unwrap();
    // No cache: the keys are the chunk's own, so the mask covers `seq` keys.
    let mask = build_chunked_prefill_mask(0, seq, device).unwrap();
    assert_eq!(mask.dtype(), Dtype::Bf16);
    let out = layer
        .forward(&x, offset, None, Some(&mask), device)
        .unwrap();
    out.eval().unwrap();
    assert_eq!(out.shape(), vec![1, seq, hidden]);
}
