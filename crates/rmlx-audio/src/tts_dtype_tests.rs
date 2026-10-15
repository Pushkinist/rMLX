//! The talker and the code predictor keep the dtype of their weights. With
//! bf16 weights, plain or affine 8-bit as in the checkpoint, the hidden stream
//! after every layer, the key/value cache of every layer, the code predictor's
//! input and the logits are bf16, in the prefill and in a decode step. A query
//! or key in f32 would make the attention output f32, and the residual add
//! would then make the hidden stream f32. CPU only, synthetic weights.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use super::tts_kv_tests::{attention, plain, Values, HEAD_DIM, PREFILL_ROWS};
use super::{
    lookup_embed, CodePredictor, Embedding, KvSlot, Linear, Quant, RmsNormLayer, TalkerLayer,
    TalkerMlp, PREDICTOR_KV_BLOCK_ROWS, TALKER_KV_BLOCK_ROWS,
};
use rmlx_mlx::{concatenate, quantize, Array, Device, Dtype};

const D: Device = Device::Cpu;
const LAYERS: usize = 2;
const VOCAB: i32 = 16;
/// The checkpoint's affine quantization.
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 8;

/// An affine 8-bit linear layer of weight `w`, with no bias: packed u32
/// weights, and scales and biases in the dtype of `w`.
fn quantized(w: &Array) -> Linear {
    let (w, scales, biases) = quantize(w, GROUP_SIZE, BITS, D).unwrap();
    assert_eq!(w.dtype(), Dtype::U32);
    Linear::Quant(Quant {
        w,
        scales,
        biases,
        bias: None,
        group_size: GROUP_SIZE,
        bits: BITS,
    })
}

/// How the layers of one fixture are built. Every input width of a quantized
/// linear is a multiple of the group size.
struct Form {
    name: &'static str,
    hidden: i32,
    mlp: i32,
    linear: fn(&Array) -> Linear,
}

const FORMS: [Form; 2] = [
    Form {
        name: "plain",
        hidden: 32,
        mlp: 64,
        linear: plain,
    },
    Form {
        name: "affine 8-bit",
        hidden: 64,
        mlp: 128,
        linear: quantized,
    },
];

fn bf16(values: &mut Values, shape: &[i32]) -> Array {
    values.array(shape).astype(Dtype::Bf16, D).unwrap()
}

impl Form {
    fn linear(&self, values: &mut Values, out: i32, inp: i32) -> Linear {
        (self.linear)(&bf16(values, &[out, inp]))
    }

    fn norm(&self, values: &mut Values) -> RmsNormLayer {
        RmsNormLayer {
            w: bf16(values, &[self.hidden]),
            eps: 1e-6,
        }
    }

    fn layer(&self, values: &mut Values) -> TalkerLayer {
        TalkerLayer {
            attn: attention(
                self.hidden,
                HEAD_DIM,
                |shape| bf16(values, shape),
                self.linear,
            ),
            mlp: TalkerMlp {
                gate: self.linear(values, self.mlp, self.hidden),
                up: self.linear(values, self.mlp, self.hidden),
                down: self.linear(values, self.hidden, self.mlp),
            },
            input_ln: self.norm(values),
            post_attn_ln: self.norm(values),
        }
    }
}

fn assert_bf16(what: &str, array: &Array) {
    assert_eq!(array.dtype(), Dtype::Bf16, "{what}");
}

fn assert_slots_bf16(stack: &str, slots: &[KvSlot], call: &str) {
    for (index, slot) in slots.iter().enumerate() {
        assert_bf16(
            &format!("{stack} K cache of layer {index}, {call}"),
            slot.k.as_ref().unwrap(),
        );
        assert_bf16(
            &format!("{stack} V cache of layer {index}, {call}"),
            slot.v.as_ref().unwrap(),
        );
    }
}

#[test]
fn the_talker_layers_keep_the_weight_dtype() {
    for form in &FORMS {
        let mut values = Values(0xbf16);
        let layers: Vec<TalkerLayer> = (0..LAYERS).map(|_| form.layer(&mut values)).collect();
        let mut slots: Vec<KvSlot> = (0..LAYERS)
            .map(|_| KvSlot::new(TALKER_KV_BLOCK_ROWS))
            .collect();
        let mut offset = 0;
        for rows in [PREFILL_ROWS, 1] {
            let call = format!("{} layers, {rows} rows at offset {offset}", form.name);
            let mut h = bf16(&mut values, &[1, rows, form.hidden]);
            for (index, (layer, slot)) in layers.iter().zip(slots.iter_mut()).enumerate() {
                h = layer.forward(&h, slot, offset, D).unwrap();
                assert_bf16(&format!("hidden after layer {index}, {call}"), &h);
            }
            assert_slots_bf16("talker", &slots, &call);
            offset += rows;
        }
    }
}

/// The first input of a frame is the talker's hidden and the group-0 code's
/// embedding, each through `small_to_mtp`, side by side, as in `synthesize`;
/// every later input is one embedding through it.
#[test]
fn the_code_predictor_keeps_the_weight_dtype() {
    for form in &FORMS {
        let mut values = Values(0xc0de);
        let hidden = form.hidden;
        let predictor = CodePredictor {
            codec_embeddings: vec![Embedding {
                w: bf16(&mut values, &[VOCAB, hidden]),
            }],
            small_to_mtp: form.linear(&mut values, hidden, hidden),
            layers: (0..LAYERS).map(|_| form.layer(&mut values)).collect(),
            norm: form.norm(&mut values),
            lm_heads: vec![form.linear(&mut values, VOCAB, hidden)],
        };
        let talker_hidden = bf16(&mut values, &[1, 1, hidden]);
        let embedding = &predictor.codec_embeddings[0];
        let project = |x: &Array| predictor.small_to_mtp.forward(x, D).unwrap();
        let first = concatenate(
            &[
                &project(&talker_hidden),
                &project(&lookup_embed(embedding, 3, D).unwrap()),
            ],
            1,
            D,
        )
        .unwrap();
        assert_bf16(&format!("{} code predictor input", form.name), &first);
        let next = project(&lookup_embed(embedding, 5, D).unwrap());

        let mut slots: Vec<KvSlot> = (0..LAYERS)
            .map(|_| KvSlot::new(PREDICTOR_KV_BLOCK_ROWS))
            .collect();
        let mut offset = 0;
        for input in [first, next] {
            let rows = input.shape()[1];
            let call = format!("{} layers, {rows} rows at offset {offset}", form.name);
            let logits = predictor.forward(&input, &mut slots, offset, 0, D).unwrap();
            assert_bf16(&format!("code predictor logits, {call}"), &logits);
            assert_slots_bf16("code predictor", &slots, &call);
            offset += rows;
        }
    }
}
