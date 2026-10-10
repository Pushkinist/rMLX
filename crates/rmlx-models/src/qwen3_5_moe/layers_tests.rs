//! The Hadamard `Linear` and `Embedding` variants against a host f64 oracle,
//! on `Device::Cpu`.
//!
//! The oracle decodes the packed codes on the host (`code * scale + bias`,
//! sixteen 2-bit codes per U32 word, lowest bits first) and applies the
//! Sylvester transform of the `crate::hadamard` tests. A linear layer is
//! `dequant(W') . H (D x)`; an embedding row is `D H dequant(E'[id])`, the
//! arithmetic of the vendor `Packed` module in the pack's
//! `runtime/runtime.py`. A one-off run of that module outside the repository
//! (MLX CPU device, real rows) agreed with this oracle to within f16 rounding.
//! Each value check also runs two controls: the layer with no transform, and
//! with the other direction. Both must miss the bound by more than ten
//! times.

use std::collections::BTreeMap;
use std::sync::Arc;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};
use rmlx_quant::awq::{f16_bits_to_f32, f32_to_f16_bits};

use super::{Embedding, HadamardFold, Linear};
use crate::hadamard::hadamard_tests::{
    max_abs_diff, oracle_forward, oracle_inverse, read_f32, round_to, signs, spacing, uniform,
    widen,
};
use crate::hadamard::SignedBlockHadamard;

/// Two blocks per synthetic row, so a transform over the full width, or at
/// the pack's block of 1024, moves the values.
const BLOCK: usize = 512;
const WIDTH: usize = 1024;
const GROUP: usize = 128;
const BITS: i32 = 2;
const CODES_PER_WORD: usize = 16;
/// Linear bound, relative to the largest output. The transform computes in
/// f32 and rounds once to the input dtype (the oracle rounds a bf16 input the
/// same way); the matmul accumulates in f32 with f16 scales. Measured at most
/// 2.5e-6 on every synthetic and real case, for f32 and bf16 inputs.
const REL_BOUND: f64 = 1e-5;
const CONTROL_MARGIN: f64 = 10.0;

/// A quantized folded table: affine 2-bit codes with group 128, and the
/// decoded values the oracle reads.
pub(crate) struct Folded {
    pub(crate) rows: usize,
    pub(crate) width: usize,
    pub(crate) words: Vec<u32>,
    pub(crate) scales: Vec<u16>,
    pub(crate) biases: Vec<u16>,
}

impl Folded {
    /// Random codes 0..=3; scales and biases are multiples of 2^-8 with at
    /// most 8 significant bits, so f16 and bf16 hold them exactly.
    pub(crate) fn synthetic(seed: u64, rows: usize, width: usize) -> Self {
        let codes: Vec<u32> = uniform(seed, rows * width)
            .iter()
            .map(|u| (((u + 1.0) * 2.0) as u32).min(3))
            .collect();
        let groups = rows * width / GROUP;
        let exact = |v: f32| f32_to_f16_bits(v);
        let scales = uniform(seed + 1, groups)
            .iter()
            .map(|u| exact((3.0 + (u.abs() * 37.0).floor()) / 256.0))
            .collect();
        let biases = uniform(seed + 2, groups)
            .iter()
            .map(|u| exact((u * 60.0).floor() / 256.0))
            .collect();
        let words = codes
            .chunks_exact(CODES_PER_WORD)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |w, (lane, code)| w | (code << (2 * lane)))
            })
            .collect();
        Self {
            rows,
            width,
            words,
            scales,
            biases,
        }
    }

    /// `[rows, width]` decoded in f64.
    pub(in crate::qwen3_5_moe) fn dequant(&self) -> Vec<f64> {
        let half =
            |v: &[u16]| -> Vec<f64> { v.iter().map(|b| f64::from(f16_bits_to_f32(*b))).collect() };
        self.dequant_with(&half(&self.scales), &half(&self.biases))
    }

    /// `[rows, width]` decoded in f64 with the given per-group scales and
    /// biases in place of the stored ones.
    fn dequant_with(&self, scales: &[f64], biases: &[f64]) -> Vec<f64> {
        let codes = self.words.iter().flat_map(|word| {
            (0..CODES_PER_WORD).map(move |lane| f64::from((word >> (2 * lane)) & 3))
        });
        let groups = scales
            .iter()
            .zip(biases)
            .flat_map(|(s, b)| std::iter::repeat_n((*s, *b), GROUP));
        codes
            .zip(groups)
            .map(|(code, (scale, bias))| code * scale + bias)
            .collect()
    }

    /// Weight `[rows, width / 16]` U32, scales and biases `[rows, width / 128]`
    /// F16, as the pack stores them.
    fn arrays(&self) -> Result<(Array, Array, Array)> {
        let rows = self.rows as i32;
        let words: Vec<u8> = self.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let half = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|b| b.to_le_bytes()).collect() };
        let groups = (self.width / GROUP) as i32;
        Ok((
            Array::from_bytes(
                &words,
                &[rows, (self.width / CODES_PER_WORD) as i32],
                Dtype::U32,
            )?,
            Array::from_bytes(&half(&self.scales), &[rows, groups], Dtype::F16)?,
            Array::from_bytes(&half(&self.biases), &[rows, groups], Dtype::F16)?,
        ))
    }

    pub(in crate::qwen3_5_moe) fn linear(&self) -> Result<Linear> {
        let (weight, scales, biases) = self.arrays()?;
        Ok(Linear::Quantized {
            weight,
            scales,
            biases: Some(biases),
            group_size: GROUP as i32,
            bits: BITS,
            mode: "affine".to_owned(),
        })
    }

    pub(in crate::qwen3_5_moe) fn embedding(&self) -> Result<Embedding> {
        let (weight, scales, biases) = self.arrays()?;
        Ok(Embedding::Quantized {
            weight,
            scales,
            biases: Some(biases),
            group_size: GROUP as i32,
            bits: BITS,
            mode: "affine".to_owned(),
        })
    }
}

/// `x [n, width] . W^T` in f64, `w` row-major `[rows, width]`.
fn host_matmul(x: &[f64], w: &[f64], width: usize) -> Vec<f64> {
    x.chunks_exact(width)
        .flat_map(|xr| {
            w.chunks_exact(width)
                .map(move |wr| xr.iter().zip(wr).map(|(a, b)| a * b).sum::<f64>())
        })
        .collect()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| f64::max(m, x.abs()))
}

fn transform(block: usize, s: &[f32]) -> Result<Arc<SignedBlockHadamard>> {
    Ok(Arc::new(SignedBlockHadamard::new(block, s)?))
}

/// `layer`, a Hadamard linear over `folded`, against the oracle of signs `s`
/// and `block`, for an input of each dtype in `dtypes` and leading shape
/// `lead`, with both controls.
#[allow(
    clippy::too_many_arguments,
    reason = "test helper: each argument is one fact of the case under test"
)]
pub(in crate::qwen3_5_moe) fn check_linear(
    what: &str,
    layer: &Linear,
    folded: &Folded,
    s: &[f32],
    block: usize,
    lead: &[i32],
    seed: u64,
    dtypes: &[Dtype],
) -> Result<()> {
    let t = transform(block, s)?;
    let plain = folded.linear()?;
    let w = folded.dequant();
    let width = folded.width;
    let n: usize = lead.iter().map(|d| *d as usize).product();
    let mut shape = lead.to_vec();
    shape.push(width as i32);
    let mut out_shape = lead.to_vec();
    out_shape.push(folded.rows as i32);

    for &dtype in dtypes {
        let x =
            Array::from_f32_slice(&uniform(seed, n * width), &shape)?.astype(dtype, Device::Cpu)?;
        let xs = widen(&read_f32(&x)?);
        let mut tx = oracle_forward(&xs, s, block);
        if dtype == Dtype::Bf16 {
            for v in &mut tx {
                *v = round_to(Dtype::Bf16, *v);
            }
        }
        let want = host_matmul(&tx, &w, width);
        let bound = REL_BOUND * max_abs(&want);

        let y = layer.forward(&x, Device::Cpu)?;
        assert_eq!(y.shape(), out_shape, "{what} {dtype:?}: output shape");
        let err = max_abs_diff(&widen(&read_f32(&y)?), &want);
        assert!(
            err <= bound,
            "{what} {dtype:?}: max |y - oracle| {err:.3e} > bound {bound:.3e}"
        );

        let no_transform = plain.forward(&x, Device::Cpu)?;
        let inverse = plain.forward(&t.inverse(&x, Device::Cpu)?, Device::Cpu)?;
        for (control, y) in [("no transform", no_transform), ("inverse", inverse)] {
            let miss = max_abs_diff(&widen(&read_f32(&y)?), &want);
            assert!(
                miss > CONTROL_MARGIN * bound,
                "{what} {dtype:?}: control `{control}` misses by {miss:.3e}, \
                 not more than {CONTROL_MARGIN} x bound {bound:.3e}"
            );
        }
    }
    Ok(())
}

/// `layer`, a Hadamard embedding over `folded`, against the oracle of signs
/// `s` and `block` at `ids`, with both controls. The output is bf16, as every
/// embedding of this model.
pub(in crate::qwen3_5_moe) fn check_embedding(
    what: &str,
    layer: &Embedding,
    folded: &Folded,
    s: &[f32],
    block: usize,
    ids: &[i32],
) -> Result<()> {
    let t = transform(block, s)?;
    let plain = folded.embedding()?;
    let table = folded.dequant();
    let width = folded.width;
    let rows: Vec<f64> = ids
        .iter()
        .filter_map(|id| table.chunks_exact(width).nth(*id as usize))
        .flatten()
        .copied()
        .collect();
    assert_eq!(
        rows.len(),
        ids.len() * width,
        "{what}: an id past the table"
    );
    let want = oracle_inverse(&rows, s, block);
    // One bf16 step at the largest value. Measured at most 0.4 of it: the
    // bf16 output rounding, plus the bf16 cast of an f16 row before the
    // transform (exact for the synthetic table).
    let bound = spacing(Dtype::Bf16, max_abs(&want));
    let id_array = Array::from_i32_slice(ids, &[2, (ids.len() / 2) as i32])?;

    let e = layer.forward(&id_array, Device::Cpu)?;
    assert_eq!(e.dtype(), Dtype::Bf16, "{what}: output dtype");
    assert_eq!(
        e.shape(),
        vec![2, (ids.len() / 2) as i32, width as i32],
        "{what}: output shape"
    );
    let err = max_abs_diff(&widen(&read_f32(&e)?), &want);
    assert!(
        err <= bound,
        "{what}: max |e - oracle| {err:.3e} > bound {bound:.3e}"
    );

    let no_transform = plain.forward(&id_array, Device::Cpu)?;
    let forward = t.forward(&plain.forward(&id_array, Device::Cpu)?, Device::Cpu)?;
    for (control, e) in [("no transform", no_transform), ("forward", forward)] {
        let miss = max_abs_diff(&widen(&read_f32(&e)?), &want);
        assert!(
            miss > CONTROL_MARGIN * bound,
            "{what}: control `{control}` misses by {miss:.3e}, not more than \
             {CONTROL_MARGIN} x bound {bound:.3e}"
        );
    }
    Ok(())
}

#[test]
fn a_hadamard_linear_is_the_folded_weight_times_the_forward_transform() -> Result<()> {
    let folded = Folded::synthetic(4100, 8, WIDTH);
    let s = signs(4110, WIDTH);
    let layer = Linear::hadamard(transform(BLOCK, &s)?, folded.linear()?)?;
    check_linear(
        "synthetic",
        &layer,
        &folded,
        &s,
        BLOCK,
        &[2, 3],
        4120,
        &[Dtype::F32, Dtype::Bf16],
    )
}

#[test]
fn a_hadamard_embedding_row_is_the_inverse_transform_of_the_folded_row() -> Result<()> {
    let folded = Folded::synthetic(4200, 6, WIDTH);
    let s = signs(4210, WIDTH);
    let layer = Embedding::hadamard(transform(BLOCK, &s)?, folded.embedding()?)?;
    check_embedding("synthetic", &layer, &folded, &s, BLOCK, &[3, 0, 5, 3])
}

fn is_model_error(r: &Result<impl Sized>, needles: &[&str]) -> bool {
    matches!(r, Err(Error::Model(m)) if needles.iter().all(|n| m.contains(n)))
}

#[test]
fn the_constructors_refuse_a_layer_the_pack_does_not_fold() -> Result<()> {
    let s = signs(4300, WIDTH);
    let wide = Folded::synthetic(4310, 4, 2 * WIDTH);
    let plain = || -> Result<Array> {
        Array::from_f32_slice(&uniform(4320, 4 * WIDTH), &[4, WIDTH as i32])
    };

    let r = Linear::hadamard(transform(BLOCK, &s)?, Linear::Plain { weight: plain()? });
    assert!(is_model_error(&r, &["not quantized"]), "plain linear");
    let r = Linear::hadamard(transform(BLOCK, &s)?, wide.linear()?);
    assert!(
        is_model_error(&r, &["2048", "1024"]),
        "linear of another width"
    );
    let nested = Linear::hadamard(
        transform(BLOCK, &s)?,
        Folded::synthetic(4330, 4, WIDTH).linear()?,
    )?;
    let r = Linear::hadamard(transform(BLOCK, &s)?, nested);
    assert!(
        is_model_error(&r, &["not quantized"]),
        "Hadamard linear in a Hadamard linear"
    );

    let r = Embedding::hadamard(transform(BLOCK, &s)?, Embedding::Plain { weight: plain()? });
    assert!(is_model_error(&r, &["not quantized"]), "plain embedding");
    let r = Embedding::hadamard(transform(BLOCK, &s)?, wide.embedding()?);
    assert!(
        is_model_error(&r, &["2048", "1024"]),
        "embedding of another width"
    );
    Ok(())
}

/// A 3-D expert stack `[experts, rows, width * bits / 32]`: its axis 1 is the
/// row count, not the packed input width, so it must not be read as one.
#[test]
fn a_folded_expert_stack_is_refused() -> Result<()> {
    let s = signs(4350, WIDTH);
    // Axis 1 holds 64 rows: read as packed columns, 64 * 32 / 2 = 1024 = WIDTH.
    let (experts, rows, packed) = (2, 64, WIDTH / 16);
    let groups = WIDTH / GROUP;
    let stack = Linear::Quantized {
        weight: Array::from_bytes(
            &vec![0x5a_u8; experts * rows * packed * 4],
            &[experts as i32, rows as i32, packed as i32],
            Dtype::U32,
        )?,
        scales: Array::from_f32_slice(
            &vec![0.25; experts * rows * groups],
            &[experts as i32, rows as i32, groups as i32],
        )?,
        biases: None,
        group_size: GROUP as i32,
        bits: BITS,
        mode: "affine".to_owned(),
    };
    let r = Linear::hadamard(transform(BLOCK, &s)?, stack);
    assert!(is_model_error(&r, &["expert stack"]), "a 3-D expert stack");
    Ok(())
}

#[test]
fn a_hadamard_layer_is_no_tied_head_and_no_expert() -> Result<()> {
    let s = signs(4400, WIDTH);
    let folded = Folded::synthetic(4410, 4, WIDTH);
    let x = Array::from_f32_slice(&uniform(4420, WIDTH), &[1, WIDTH as i32])?;

    let emb = Embedding::hadamard(transform(BLOCK, &s)?, folded.embedding()?)?;
    assert!(
        is_model_error(&emb.as_linear(&x, Device::Cpu), &["Hadamard"]),
        "a Hadamard embedding used as a tied head"
    );
    assert!(
        folded.embedding()?.as_linear(&x, Device::Cpu).is_ok(),
        "control: the plain table"
    );

    let lin = Linear::hadamard(transform(BLOCK, &s)?, folded.linear()?)?;
    let xe = x.reshape(&[1, 1, 1, WIDTH as i32], Device::Cpu)?;
    let experts = Array::from_i32_slice(&[0], &[1, 1])?;
    assert!(
        is_model_error(
            &lin.gather_forward(&xe, &experts, false, Device::Cpu),
            &["Hadamard"]
        ),
        "a Hadamard linear in an expert dispatch"
    );
    Ok(())
}

const BONSAI_2_SLUG: &str = "prism-ml__Ternary-Bonsai-2-27B-mlx-2bit";
/// The pack's own rows, one module of each input width, the head and the
/// embedding.
const PACK_LINEARS: [&str; 4] = [
    "language_model.model.layers.3.self_attn.q_proj",
    "language_model.model.layers.3.self_attn.o_proj",
    "language_model.model.layers.0.mlp.down_proj",
    "language_model.lm_head",
];
const PACK_EMBEDDING: &str = "language_model.model.embed_tokens";
const PACK_ROWS: usize = 16;
/// The pack's Hadamard block.
const BLOCK_PACK: usize = 1024;
/// Token ids of the embedding rows: low ids, `<|endoftext|>`, `<|im_end|>`,
/// and the last row.
const PACK_IDS: [usize; 6] = [0, 13, 151_643, 248_044, 248_046, 248_319];

fn loader_error(e: impl std::fmt::Display) -> Error {
    Error::Loader(e.to_string())
}

/// The bytes of `rows` of tensor `name`.
fn tensor_rows(st: &safetensors::SafeTensors<'_>, name: &str, rows: &[usize]) -> Result<Vec<u8>> {
    let view = st.tensor(name).map_err(loader_error)?;
    let data = view.data();
    let row_bytes = data.len() / view.shape().first().copied().unwrap_or(1);
    let bytes: Vec<u8> = rows
        .iter()
        .filter_map(|r| data.chunks_exact(row_bytes).nth(*r))
        .flatten()
        .copied()
        .collect();
    assert_eq!(
        bytes.len(),
        rows.len() * row_bytes,
        "{name}: a row past the tensor"
    );
    Ok(bytes)
}

fn pack_folded(st: &safetensors::SafeTensors<'_>, module: &str, rows: &[usize]) -> Result<Folded> {
    let words: Vec<u32> = tensor_rows(st, &format!("{module}.weight"), rows)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let half = |part: &str| -> Result<Vec<u16>> {
        Ok(tensor_rows(st, &format!("{module}.{part}"), rows)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect())
    };
    let width = words.len() * CODES_PER_WORD / rows.len();
    Ok(Folded {
        rows: rows.len(),
        width,
        words,
        scales: half("scales")?,
        biases: half("biases")?,
    })
}

/// The module's own `.signs` tensor.
fn pack_signs(st: &safetensors::SafeTensors<'_>, module: &str) -> Result<Vec<f32>> {
    let view = st
        .tensor(&format!("{module}.signs"))
        .map_err(loader_error)?;
    Ok(view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// The real pack's rows through the Hadamard variants, against the oracle
/// built from the stored bytes and the module's own `.signs` tensor, which
/// must be the parsed contract's vector of the module's width.
#[test]
fn the_ternary_bonsai_2_pack_rows_decode_through_the_hadamard_variants() -> Result<()> {
    let test = "the_ternary_bonsai_2_pack_rows_decode_through_the_hadamard_variants";
    let Some(dir) =
        crate::test_snapshot::snapshot_by_slug(test, BONSAI_2_SLUG, &["prism_hadamard_qwen35"])
    else {
        return Ok(());
    };
    let contract = rmlx_loader::HadamardContract::read(&dir)
        .map_err(loader_error)?
        .ok_or_else(|| Error::Loader("the pack names no contract".to_owned()))?;
    let shard = rmlx_loader::ShardHandle::open(&dir, "model.safetensors")?;
    let st = shard.safetensors()?;
    let contract_signs = |module: &str, list: &BTreeMap<String, usize>| {
        list.get(module)
            .and_then(|width| contract.signs.get(width))
            .cloned()
    };

    let rows: Vec<usize> = (0..PACK_ROWS).collect();
    for (k, module) in PACK_LINEARS.iter().enumerate() {
        let folded = pack_folded(&st, module, &rows)?;
        let s = pack_signs(&st, module)?;
        assert_eq!(
            contract_signs(module, &contract.forward),
            Some(s.clone()),
            "{module}: contract signs"
        );
        let layer = Linear::hadamard(transform(contract.block, &s)?, folded.linear()?)?;
        check_linear(
            module,
            &layer,
            &folded,
            &s,
            contract.block,
            &[2],
            4500 + k as u64,
            &[Dtype::F32, Dtype::Bf16],
        )?;
    }

    let folded = pack_folded(&st, PACK_EMBEDDING, &PACK_IDS)?;
    let s = pack_signs(&st, PACK_EMBEDDING)?;
    assert_eq!(
        contract_signs(PACK_EMBEDDING, &contract.inverse),
        Some(s.clone()),
        "embedding: contract signs"
    );
    let ids: Vec<i32> = (0..PACK_IDS.len() as i32).collect();
    let layer = Embedding::hadamard(transform(contract.block, &s)?, folded.embedding()?)?;
    check_embedding(PACK_EMBEDDING, &layer, &folded, &s, contract.block, &ids)
}

/// A fold over two widths: forward modules `m.a`, `m.b` (1024) and `m.w`
/// (2048), the inverse module `m.e` (1024).
fn fold() -> Result<HadamardFold> {
    let width = |n: &str, w: usize| (n.to_owned(), w);
    HadamardFold::new(
        BLOCK,
        &BTreeMap::from([
            (WIDTH, signs(4700, WIDTH)),
            (2 * WIDTH, signs(4701, 2 * WIDTH)),
        ]),
        BTreeMap::from([
            width("m.a", WIDTH),
            width("m.b", WIDTH),
            width("m.w", 2 * WIDTH),
        ]),
        BTreeMap::from([width("m.e", WIDTH)]),
    )
}

#[test]
fn the_fold_folds_each_contract_module_and_refuses_one_left_unbuilt() -> Result<()> {
    let narrow = Folded::synthetic(4710, 4, WIDTH);
    let wide = Folded::synthetic(4720, 4, 2 * WIDTH);

    let f = fold()?;
    assert!(
        matches!(f.linear("m.a", narrow.linear()?)?, Linear::Hadamard { .. }),
        "a forward module is folded"
    );
    assert!(
        matches!(f.linear("m.w", wide.linear()?)?, Linear::Hadamard { .. }),
        "a forward module of the second width takes its own transform"
    );
    assert!(
        matches!(
            f.embedding("m.e", narrow.embedding()?)?,
            Embedding::Hadamard { .. }
        ),
        "the inverse module is folded"
    );
    assert!(
        matches!(f.linear("m.x", narrow.linear()?)?, Linear::Quantized { .. }),
        "a linear the contract does not name is kept"
    );
    assert!(
        matches!(
            f.embedding("m.x", narrow.embedding()?)?,
            Embedding::Quantized { .. }
        ),
        "an embedding the contract does not name is kept"
    );
    assert!(
        is_model_error(&f.linear("m.e", narrow.linear()?), &["m.e"]),
        "the inverse module built as a linear layer"
    );
    assert!(
        is_model_error(&f.embedding("m.b", narrow.embedding()?), &["m.b"]),
        "a forward module built as an embedding"
    );
    assert!(
        is_model_error(&f.finish(), &["m.b"]),
        "a forward module with no layer built"
    );

    let f = fold()?;
    f.linear("m.a", narrow.linear()?)?;
    f.linear("m.b", narrow.linear()?)?;
    f.linear("m.w", wide.linear()?)?;
    assert!(
        is_model_error(&f.finish(), &["m.e"]),
        "the inverse module with no layer built"
    );

    // The transforms the fold hands out carry the contract's block and the
    // sign vector of each width.
    let f = fold()?;
    let a = f.linear("m.a", narrow.linear()?)?;
    f.linear("m.b", narrow.linear()?)?;
    let w = f.linear("m.w", wide.linear()?)?;
    let e = f.embedding("m.e", narrow.embedding()?)?;
    f.finish()?;
    let all = [Dtype::F32, Dtype::Bf16];
    check_linear(
        "m.a",
        &a,
        &narrow,
        &signs(4700, WIDTH),
        BLOCK,
        &[2],
        4730,
        &all,
    )?;
    check_linear(
        "m.w",
        &w,
        &wide,
        &signs(4701, 2 * WIDTH),
        BLOCK,
        &[2],
        4740,
        &all,
    )?;
    check_embedding(
        "m.e",
        &e,
        &narrow,
        &signs(4700, WIDTH),
        BLOCK,
        &[3, 0, 2, 1],
    )
}

/// Rows per module and inputs of the precision measurement.
const PRECISION_ROWS: usize = 32;
const PRECISION_INPUTS: usize = 4;
/// bf16 activations with bf16-cast scales may have at most this many times
/// the rms error of bf16 activations alone.
const SCALE_CAST_BOUND: f64 = 1.4;

fn rms_error(a: &[f64], reference: &[f64]) -> f64 {
    let err: f64 = a
        .iter()
        .zip(reference)
        .map(|(x, r)| (x - r) * (x - r))
        .sum();
    let norm: f64 = reference.iter().map(|r| r * r).sum();
    (err / norm).sqrt()
}

/// rMLX computes a pack's linear layers with bf16 activations and bf16-cast
/// scales (`load_util::bf16_scales`); the pack's contract names f16
/// (`dtype` `float16`), and the vendor runtime keeps f16 scales and f16
/// activations. On real rows of the pack, in host f64 with exact
/// accumulation, the vendor's f16 arithmetic is the reference. The floor is
/// the error of bf16 activations with the f16 scales kept. The scale cast is
/// an error of the same order, so the two together are bounded by
/// `SCALE_CAST_BOUND` times the floor. Control: scales cut to 4 significant
/// bits must miss that bound.
#[test]
fn the_bf16_scale_cast_is_bounded_by_the_bf16_activation_floor() -> Result<()> {
    let test = "the_bf16_scale_cast_is_bounded_by_the_bf16_activation_floor";
    let Some(dir) =
        crate::test_snapshot::snapshot_by_slug(test, BONSAI_2_SLUG, &["prism_hadamard_qwen35"])
    else {
        return Ok(());
    };
    let shard = rmlx_loader::ShardHandle::open(&dir, "model.safetensors")?;
    let st = shard.safetensors()?;
    let rows: Vec<usize> = (0..PRECISION_ROWS).collect();
    for (k, module) in PACK_LINEARS.iter().enumerate() {
        let folded = pack_folded(&st, module, &rows)?;
        let s = pack_signs(&st, module)?;
        let half =
            |v: &[u16]| -> Vec<f64> { v.iter().map(|b| f64::from(f16_bits_to_f32(*b))).collect() };
        let cast = |v: &[u16]| -> Result<Vec<f64>> {
            let bytes: Vec<u8> = v.iter().flat_map(|b| b.to_le_bytes()).collect();
            let a = Array::from_bytes(&bytes, &[v.len() as i32], Dtype::F16)?;
            Ok(widen(&read_f32(&crate::load_util::bf16_scales(a)?)?))
        };
        let coarse = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .map(|x| {
                    let step = 2f64.powf(x.abs().log2().floor() - 3.0);
                    (x / step).round() * step
                })
                .collect()
        };
        let (scales, biases) = (half(&folded.scales), half(&folded.biases));
        let w_f16 = folded.dequant_with(&scales, &biases);
        let w_cast = folded.dequant_with(&cast(&folded.scales)?, &cast(&folded.biases)?);
        let w_coarse = folded.dequant_with(&coarse(&scales), &coarse(&biases));

        let x = widen(&uniform(4800 + k as u64, PRECISION_INPUTS * folded.width));
        let rounded = |dtype: Dtype| -> Vec<f64> {
            let xs: Vec<f64> = x.iter().map(|v| round_to(dtype, *v)).collect();
            oracle_forward(&xs, &s, BLOCK_PACK)
                .iter()
                .map(|v| round_to(dtype, *v))
                .collect()
        };
        let (t_f16, t_bf16) = (rounded(Dtype::F16), rounded(Dtype::Bf16));
        let reference = host_matmul(&t_f16, &w_f16, folded.width);
        let floor = rms_error(&host_matmul(&t_bf16, &w_f16, folded.width), &reference);
        let with_cast = rms_error(&host_matmul(&t_bf16, &w_cast, folded.width), &reference);
        let with_coarse = rms_error(&host_matmul(&t_bf16, &w_coarse, folded.width), &reference);
        let cast_alone = rms_error(&host_matmul(&t_f16, &w_cast, folded.width), &reference);
        println!(
            "{module}: rms error vs f16: bf16 activations {floor:.3e}, plus the bf16 scale \
             cast {with_cast:.3e}, the scale cast alone {cast_alone:.3e}, bf16 activations \
             and 4-bit scales {with_coarse:.3e}"
        );
        let bound = floor * SCALE_CAST_BOUND;
        assert!(
            with_cast <= bound,
            "{module}: the scale cast {with_cast:.3e} > {bound:.3e}"
        );
        assert!(
            with_coarse > bound,
            "{module}: control, 4-bit scales {with_coarse:.3e} not above {bound:.3e}"
        );
    }
    Ok(())
}

/// The pack's gated-delta-net layers read: the three before its first
/// full-attention layer, and the one after it.
const PACK_GDN_LAYERS: [usize; 4] = [0, 1, 2, 4];
/// Inputs per layer of the gate measurement.
const GATE_INPUTS: usize = 64;
/// The gate path rMLX takes may have at most this many times the rms error
/// of bf16 activations alone, on the delta-net log-gate.
const GATE_BOUND: f64 = 1.1;

/// All values of the F32 tensor `name`, widened.
fn f32_tensor(st: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<f64>> {
    let view = st.tensor(name).map_err(loader_error)?;
    Ok(view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f64::from(f32::from_le_bytes(*c)))
        .collect())
}

fn softplus(v: f64) -> f64 {
    v.max(0.0) + (-v.abs()).exp().ln_1p()
}

/// The delta-net log-gate `-exp(A_log) * softplus(x . W_a + dt_bias)` and
/// the update weight `sigmoid(x . W_b)`, one value per input and head, with
/// `round` applied to each projection output.
fn delta_net_gate(
    x: &[f64],
    w_a: &[f64],
    w_b: &[f64],
    a_log: &[f64],
    dt_bias: &[f64],
    round: impl Fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    let width = w_a.len() / a_log.len();
    let a = host_matmul(x, w_a, width);
    let b = host_matmul(x, w_b, width);
    let heads = a_log.iter().zip(dt_bias).cycle();
    let log_g = a
        .iter()
        .zip(heads)
        .map(|(v, (a_log, dt_bias))| {
            let a_plus = f64::from((round(*v) + dt_bias) as f32);
            -f64::from(*a_log as f32).exp() * softplus(a_plus)
        })
        .collect();
    let beta = b.iter().map(|v| 1.0 / (1.0 + (-round(*v)).exp())).collect();
    (log_g, beta)
}

/// A design measurement, not a check of the loader: why the loader keeps the
/// pack's dense `in_proj_a` / `in_proj_b` F32. On the real rows of four
/// gated-delta-net layers, in host f64, f16 activations with the F32 weights
/// are the reference and bf16 activations alone are the floor. Kept F32 (bf16
/// activations, F32 weights, f32 outputs) is the floor plus one f32 rounding
/// of each projection output, so it stays within `GATE_BOUND` times the floor
/// by construction. The weights cast by `bf16_param` with bf16 outputs must
/// miss that bound: that is the measured cost the F32 choice avoids. The
/// structure test of the synthetic pack checks that the loader keeps F32.
#[test]
fn a_bf16_cast_of_the_gate_projections_raises_the_delta_net_gate_error_above_the_floor(
) -> Result<()> {
    let test =
        "a_bf16_cast_of_the_gate_projections_raises_the_delta_net_gate_error_above_the_floor";
    let Some(dir) =
        crate::test_snapshot::snapshot_by_slug(test, BONSAI_2_SLUG, &["prism_hadamard_qwen35"])
    else {
        return Ok(());
    };
    let shard = rmlx_loader::ShardHandle::open(&dir, "model.safetensors")?;
    let st = shard.safetensors()?;
    let cast = |w: &[f64]| -> Result<Vec<f64>> {
        let v: Vec<f32> = w.iter().map(|x| *x as f32).collect();
        let a = Array::from_f32_slice(&v, &[v.len() as i32])?;
        Ok(widen(&read_f32(&crate::load_util::bf16_param(a)?)?))
    };
    for layer in PACK_GDN_LAYERS {
        let p = format!("language_model.model.layers.{layer}");
        let get = |name: &str| f32_tensor(&st, &format!("{p}.{name}"));
        let (w_a, w_b) = (
            get("linear_attn.in_proj_a.weight")?,
            get("linear_attn.in_proj_b.weight")?,
        );
        let (a_log, dt_bias) = (get("linear_attn.A_log")?, get("linear_attn.dt_bias")?);
        let norm = get("input_layernorm.weight")?;
        let x: Vec<f64> = widen(&uniform(4900 + layer as u64, GATE_INPUTS * norm.len()))
            .iter()
            .zip(norm.iter().cycle())
            .map(|(u, n)| u * n)
            .collect();
        let at = |dtype: Dtype| -> Vec<f64> { x.iter().map(|v| round_to(dtype, *v)).collect() };
        let (x_f16, x_bf16) = (at(Dtype::F16), at(Dtype::Bf16));
        let exact = |v: f64| v;
        let to_f32 = |v: f64| f64::from(v as f32);
        let to_bf16 = |v: f64| round_to(Dtype::Bf16, v);

        let reference = delta_net_gate(&x_f16, &w_a, &w_b, &a_log, &dt_bias, exact);
        let floor = delta_net_gate(&x_bf16, &w_a, &w_b, &a_log, &dt_bias, exact);
        let kept_f32 = delta_net_gate(&x_bf16, &w_a, &w_b, &a_log, &dt_bias, to_f32);
        let cast = delta_net_gate(
            &x_bf16,
            &cast(&w_a)?,
            &cast(&w_b)?,
            &a_log,
            &dt_bias,
            to_bf16,
        );
        let err = |path: &(Vec<f64>, Vec<f64>)| {
            (
                rms_error(&path.0, &reference.0),
                rms_error(&path.1, &reference.1),
            )
        };
        let ((f_g, f_b), (r_g, r_b), (c_g, c_b)) = (err(&floor), err(&kept_f32), err(&cast));
        println!(
            "layer {layer}: rms error vs f16 of log g / beta: bf16 activations \
             {f_g:.3e} / {f_b:.3e}, F32 weights kept {r_g:.3e} / {r_b:.3e}, bf16 cast \
             {c_g:.3e} / {c_b:.3e}"
        );
        let bound = f_g * GATE_BOUND;
        assert!(
            r_g <= bound,
            "layer {layer}: the kept F32 gate path {r_g:.3e} > {bound:.3e}"
        );
        assert!(
            c_g > bound,
            "layer {layer}: control, the bf16 cast {c_g:.3e} not above {bound:.3e}"
        );
    }
    Ok(())
}
