//! The Hadamard transform of an input that several linear layers read is
//! computed once per forward pass, and each layer still gets its own output.
//!
//! The groups are `q_proj` / `k_proj` / `v_proj` of the full-attention layer,
//! `gate_proj` / `up_proj` of the dense MLP and `in_proj_qkv` / `in_proj_z` of
//! the gated-delta-net layer, read from the synthetic pack, whose fold gives
//! every module of one input width the same transform. The count is
//! [`FORWARD_CALLS`], the transforms a forward pass builds on this thread.
//!
//! The output checks compose each module from the forwards of its own linear
//! layers, so a forward that hands one layer's output to another's consumer
//! misses them. They run on the CPU, where the gated-delta-net recurrence (a
//! Metal kernel) cannot, so that layer has a count and no output check.

use std::cell::Cell;
use std::sync::Arc;

use rmlx_mlx::{concatenate, multiply, sigmoid, silu, Array, Device, Dtype};

use super::super::super::decoder_layer::AttnBlock;
use super::super::super::layers::SharedInput;
use super::super::super::model::Qwen3_5MoeText;
use super::*;
use crate::hadamard::hadamard_tests::{max_abs_diff, read_f32, signs, uniform, widen};
use crate::hadamard::{SignedBlockHadamard, FORWARD_CALLS};

/// `f`'s result and the transforms it built.
fn transforms_built<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = FORWARD_CALLS.with(Cell::get);
    let out = f();
    (out, FORWARD_CALLS.with(Cell::get) - before)
}

/// One token of the hidden width, in the bf16 the model's activations have.
fn one_token(seed: u64) -> Result<Array> {
    Array::from_f32_slice(&uniform(seed, TINY_WIDTH), &[1, 1, TINY_WIDTH as i32])?
        .astype(Dtype::Bf16, Device::Cpu)
}

fn values(a: &Array) -> Result<Vec<f64>> {
    Ok(widen(&read_f32(&a.astype(Dtype::F32, Device::Cpu)?)?))
}

fn layer_mlp(model: &Qwen3_5MoeText, layer: usize) -> &super::super::super::moe::DenseMlp {
    match model.layers.get(layer).map(|l| &l.mlp) {
        Some(MlpBlock::Dense(m)) => m,
        _ => panic!("layer {layer} of the synthetic pack has a dense MLP"),
    }
}

/// `q_proj`, `k_proj` and `v_proj` share one transform; `o_proj` reads
/// another input and builds its own.
#[test]
fn the_attention_projections_share_one_transform() -> Result<()> {
    let model = ATTENTION_ONLY.load()?;
    let Some(AttnBlock::Full(attn)) = model.layers.first().map(|l| &l.attn) else {
        panic!("layer 0 of the attention-only pack is full attention");
    };
    let x = one_token(1)?;
    let (out, built) = transforms_built(|| attn.forward(&x, 0, None, None, Device::Cpu));
    out?.eval()?;
    assert_eq!(built, 2, "one for q/k/v, one for o_proj");
    Ok(())
}

/// `gate_proj` and `up_proj` share one transform; `down_proj` builds its own.
#[test]
fn the_mlp_gate_and_up_projections_share_one_transform() -> Result<()> {
    let model = ATTENTION_ONLY.load()?;
    let mlp = layer_mlp(&model, 0);
    let x = one_token(2)?;
    let (out, built) = transforms_built(|| mlp.forward(&x, Device::Cpu));
    out?.eval()?;
    assert_eq!(built, 2, "one for gate/up, one for down_proj");
    Ok(())
}

/// `in_proj_qkv` and `in_proj_z` share one transform. `in_proj_b` and
/// `in_proj_a` are plain F32 in the pack and build none. On the CPU the
/// forward stops at the recurrence kernel, after the projections and before
/// `out_proj`.
#[test]
fn the_gated_delta_net_projections_share_one_transform() -> Result<()> {
    let model = PACK.load()?;
    let Some(AttnBlock::Linear(gdn)) = model.layers.first().map(|l| &l.attn) else {
        panic!("layer 0 of the pack is a gated-delta-net layer");
    };
    let x = one_token(3)?;
    let (out, built) = transforms_built(|| gdn.forward(&x, None, Device::Cpu));
    let refused = out.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(
        refused.contains("Only supports the GPU"),
        "the CPU forward stops at the Metal recurrence kernel, got {refused:?}"
    );
    assert_eq!(built, 1, "one for in_proj_qkv/in_proj_z");
    Ok(())
}

/// The dense MLP is `down(silu(gate(x)) * up(x))` of its own linear layers'
/// forwards. Swapping `gate_proj` and `up_proj` misses it.
#[test]
fn the_mlp_is_its_own_layers_composed() -> Result<()> {
    let model = ATTENTION_ONLY.load()?;
    let mlp = layer_mlp(&model, 0);
    let d = Device::Cpu;
    let x = one_token(4)?;
    let got = values(&mlp.forward(&x, d)?)?;
    let compose = |gate: &Linear, up: &Linear| -> Result<Vec<f64>> {
        let gated = multiply(&silu(&gate.forward(&x, d)?, d)?, &up.forward(&x, d)?, d)?;
        values(&mlp.down_proj.forward(&gated, d)?)
    };
    let want = compose(&mlp.gate_proj, &mlp.up_proj)?;
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff == 0.0,
        "the MLP is not its layers composed: {diff:.3e}"
    );
    let swapped = max_abs_diff(&got, &compose(&mlp.up_proj, &mlp.gate_proj)?);
    assert!(swapped > 0.0, "gate and up swapped is not told apart");
    Ok(())
}

/// At one token the attention weights are 1, so each head's output is the
/// value row of its KV head and the layer is
/// `o_proj(v(x) per head * sigmoid(gate half of q(x)))` of its own linear
/// layers' forwards. `k_proj` does not reach this output; swapping it with
/// `q_proj` fails on their widths, and swapping `v_proj` with it misses the
/// value rows.
#[test]
fn the_one_token_attention_is_its_own_layers_composed() -> Result<()> {
    let model = ATTENTION_ONLY.load()?;
    let Some(AttnBlock::Full(attn)) = model.layers.first().map(|l| &l.attn) else {
        panic!("layer 0 of the attention-only pack is full attention");
    };
    let d = Device::Cpu;
    let x = one_token(5)?;
    let got = values(&attn.forward(&x, 0, None, None, d)?)?;
    let (heads, head_dim) = (TINY_HEADS as i32, TINY_HEAD_DIM as i32);
    let compose = |value: &Linear| -> Result<Vec<f64>> {
        let q = attn
            .q_proj
            .forward(&x, d)?
            .reshape(&[1, 1, heads, 2 * head_dim], d)?;
        let gate = q
            .slice(
                &[0, 0, 0, head_dim],
                &[1, 1, heads, 2 * head_dim],
                &[1, 1, 1, 1],
                d,
            )?
            .reshape(&[1, 1, heads * head_dim], d)?;
        let v = value.forward(&x, d)?;
        let per_head: Vec<&Array> = (0..heads).map(|_| &v).collect();
        let rows = concatenate(&per_head, 2, d)?;
        values(
            &attn
                .o_proj
                .forward(&multiply(&rows, &sigmoid(&gate, d)?, d)?, d)?,
        )
    };
    let want = compose(&attn.v_proj)?;
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff == 0.0,
        "the layer is not its layers composed: {diff:.3e}"
    );
    let swapped = max_abs_diff(&got, &compose(&attn.k_proj)?);
    assert!(swapped > 0.0, "v and k swapped is not told apart");
    Ok(())
}

/// `k_proj` reaches the attention output where the one-token check cannot:
/// over two tokens, other key weights move the second token's output and
/// leave the first's, which attends to itself alone. With the one-token
/// check this pins each of `q_proj`, `k_proj` and `v_proj` to its consumer.
#[test]
fn the_key_projection_reaches_the_second_token_only() -> Result<()> {
    let mut model = ATTENTION_ONLY.load()?;
    let Some(AttnBlock::Full(attn)) = model.layers.first_mut().map(|l| &mut l.attn) else {
        panic!("layer 0 of the attention-only pack is full attention");
    };
    let d = Device::Cpu;
    let x = Array::from_f32_slice(&uniform(7, 2 * TINY_WIDTH), &[1, 2, TINY_WIDTH as i32])?
        .astype(Dtype::Bf16, d)?;
    let rows = |out: &Array| -> Result<Vec<Vec<f64>>> {
        Ok(values(out)?
            .chunks_exact(TINY_WIDTH)
            .map(<[f64]>::to_vec)
            .collect())
    };
    let before = rows(&attn.forward(&x, 0, None, None, d)?)?;
    let Linear::Hadamard { transform, .. } = &attn.k_proj else {
        panic!("the pack folds k_proj");
    };
    let Linear::Quantized {
        weight,
        scales,
        biases,
        group_size,
        bits,
        mode,
    } = Folded::synthetic(7100, TINY_HEAD_DIM, TINY_WIDTH).linear()?
    else {
        panic!("a synthetic folded table is quantized");
    };
    // The loader keeps scales and biases in bf16; f16 ones would make the
    // keys, and so every token's output, f32.
    let other_keys = Linear::Quantized {
        weight,
        scales: scales.astype(Dtype::Bf16, d)?,
        biases: biases.map(|b| b.astype(Dtype::Bf16, d)).transpose()?,
        group_size,
        bits,
        mode,
    };
    attn.k_proj = Linear::hadamard(Arc::clone(transform), other_keys)?;
    let after = rows(&attn.forward(&x, 0, None, None, d)?)?;
    let moved: Vec<f64> = before
        .iter()
        .zip(&after)
        .map(|(b, a)| max_abs_diff(b, a))
        .collect();
    assert!(
        matches!(moved.as_slice(), [first, second] if *first == 0.0 && *second > 0.0),
        "per-token moves {moved:?}: the first token must keep its output, the second must not"
    );
    Ok(())
}

/// Each layer reading a [`SharedInput`] gets its own forward of the input:
/// two layers holding one transform, one holding a transform of the same
/// width with other signs, and a plain layer. One transform is built per
/// `Arc`.
#[test]
fn a_shared_input_gives_each_layer_its_own_forward() -> Result<()> {
    let shared = Arc::new(SignedBlockHadamard::new(TINY_BLOCK, &signs(1, TINY_WIDTH))?);
    let other = Arc::new(SignedBlockHadamard::new(TINY_BLOCK, &signs(2, TINY_WIDTH))?);
    let folded = |seed| Folded::synthetic(seed, 64, TINY_WIDTH).linear();
    let layers = [
        Linear::hadamard(Arc::clone(&shared), folded(11)?)?,
        Linear::hadamard(other, folded(12)?)?,
        folded(13)?,
        Linear::hadamard(shared, folded(14)?)?,
    ];
    let d = Device::Cpu;
    let x = one_token(6)?;
    let mut input = SharedInput::new(&x);
    let (got, built) = transforms_built(|| {
        layers
            .iter()
            .map(|l| values(&input.forward(l, d)?))
            .collect::<Result<Vec<_>>>()
    });
    let got = got?;
    assert_eq!(built, 2, "one transform per Arc");
    for (i, (layer, y)) in layers.iter().zip(&got).enumerate() {
        let diff = max_abs_diff(y, &values(&layer.forward(&x, d)?)?);
        assert!(diff == 0.0, "layer {i} is {diff:.3e} from its own forward");
    }
    for (i, a) in got.iter().enumerate() {
        for (j, b) in got.iter().enumerate().skip(i + 1) {
            assert!(max_abs_diff(a, b) > 0.0, "layers {i} and {j} agree");
        }
    }
    Ok(())
}
