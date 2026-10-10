//! The real Ternary-Bonsai-2-27B pack against the logits of the runtime the
//! pack ships.
//!
//! `tests/fixtures/hadamard_pack_vendor/logits.safetensors` holds that
//! runtime's logits for three sequences of 64 token ids, reduced to what the
//! gate reads: at each position the 20 largest logits of the runtime as
//! shipped, their ids and the log-sum-exp over the vocabulary (the
//! reference), and the same runtime after mlx-vlm's bf16 cast at those ids
//! (the floor: what bf16 activations alone move). `generate.py` beside the
//! file writes it.
//!
//! A candidate is compared with the reference by four statistics (see
//! [`Stats`]) and passes when each stays within [`GATE`] times the floor's
//! ([`gate`]).

use std::path::Path;
use std::sync::Arc;

use rmlx_core::error::{Error, Result};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_mlx::Device;

use super::decoder_layer::{AttnBlock, MlpBlock};
use super::layers::{Embedding, Linear};
use super::model::Qwen3_5MoeText;
use crate::hadamard::hadamard_tests::read_f32;
use crate::hadamard::SignedBlockHadamard;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/hadamard_pack_vendor/logits.safetensors"
);
const PACK: &str = "prism-ml__Ternary-Bonsai-2-27B-mlx-2bit";
const PACK_IDENTITY: &str = "prism_hadamard_qwen35";
/// Modules the pack's contract folds: 401 linear layers and the embedding.
const PACK_MODULES: usize = 402;
/// A candidate passes when each statistic is at most this many times the
/// floor's. rMLX runs the graph in bf16 like the floor run; its error per
/// projection was measured at 1.0 to 1.3 times the bf16 activation floor.
const GATE: f64 = 2.0;
/// The pack with its transforms removed must exceed every statistic of the
/// floor by more than this factor.
const CONTROL: f64 = 10.0;
/// The largest difference is gated over the ranks whose reference
/// probability is at least this, and rank 0. A logit error `d` moves the
/// probability of its token by at most `p (e^d - 1)`; below 1e-4 a sampler
/// draws the token less than once in 10 000 tokens, and the KL statistic
/// already weighs that mass.
const REACHABLE: f64 = 1e-4;
/// Ids in each sequence of the fixture.
const LENGTH: usize = 64;
/// Ids the decode arm prefills; it decodes the rest of each sequence one id
/// at a time.
const DECODE_FROM: usize = 48;

/// The reference at one position.
#[derive(Debug, Clone)]
struct Position {
    /// Ids of the reference's largest logits, largest first.
    top_ids: Vec<usize>,
    /// The reference's logits at `top_ids`.
    top: Vec<f32>,
    /// Log-sum-exp of the reference's logits over the vocabulary.
    lse: f32,
}

/// A candidate's logits at one position, as the gate reads them.
#[derive(Debug, Clone)]
struct Seen {
    /// The candidate's logits at the reference's `top_ids`.
    top: Vec<f32>,
    lse: f32,
    argmax: usize,
}

impl Seen {
    /// `row`, the candidate's logits over the vocabulary, read at the ids of
    /// `at`.
    fn of(at: &Position, row: &[f32]) -> Result<Self> {
        let top = at
            .top_ids
            .iter()
            .map(|&id| {
                row.get(id).copied().ok_or_else(|| {
                    Error::Model(format!("id {id} is outside a row of {}", row.len()))
                })
            })
            .collect::<Result<Vec<f32>>>()?;
        let argmax = row
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
                if v > best.1 {
                    (i, v)
                } else {
                    best
                }
            })
            .0;
        Ok(Self {
            top,
            lse: logsumexp(row),
            argmax,
        })
    }
}

fn logsumexp(row: &[f32]) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&v| f64::from(v - max).exp()).sum();
    (f64::from(max) + sum.ln()) as f32
}

/// The fixture: the token ids, and per position (sequence-major) the
/// reference and the floor.
#[derive(Debug)]
struct VendorReference {
    ids: Vec<Vec<u32>>,
    positions: Vec<Position>,
    floor: Vec<Seen>,
}

/// The values of tensor `name` in `st`, checked against `dtype` and `shape`.
fn tensor_bytes<'a>(
    st: &'a safetensors::SafeTensors<'a>,
    name: &str,
    dtype: safetensors::Dtype,
    shape: &[usize],
) -> Result<&'a [u8]> {
    let t = st
        .tensor(name)
        .map_err(|e| Error::Model(format!("vendor reference: {name}: {e}")))?;
    if t.dtype() != dtype || t.shape() != shape {
        return Err(Error::Model(format!(
            "vendor reference: {name} is {:?} {:?}, expected {dtype:?} {shape:?}",
            t.dtype(),
            t.shape()
        )));
    }
    Ok(t.data())
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn ids_of(values: Vec<i32>) -> Result<Vec<usize>> {
    values
        .into_iter()
        .map(|v| usize::try_from(v).map_err(|_| Error::Model(format!("negative id {v}"))))
        .collect()
}

impl VendorReference {
    /// # Errors
    /// `Error::Model` when the file does not parse, misses a tensor, or a
    /// tensor's dtype or shape does not fit `ids` and `reference_top_ids`.
    fn read(path: &Path) -> Result<Self> {
        use safetensors::Dtype::{F32, I32};
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Model(format!("vendor reference {}: {e}", path.display())))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| Error::Model(format!("vendor reference {}: {e}", path.display())))?;
        let ids = st
            .tensor("ids")
            .map_err(|e| Error::Model(format!("vendor reference: ids: {e}")))?;
        let top = st
            .tensor("reference_top_ids")
            .map_err(|e| Error::Model(format!("vendor reference: reference_top_ids: {e}")))?;
        let (&[seqs, len], &[_, _, k]) = (ids.shape(), top.shape()) else {
            return Err(Error::Model(format!(
                "vendor reference: ids {:?} and reference_top_ids {:?} are not [seqs, len] \
                 and [seqs, len, k]",
                ids.shape(),
                top.shape()
            )));
        };
        let grid = [seqs, len];
        let tops = [seqs, len, k];
        let token_ids = ids_of(i32s(tensor_bytes(&st, "ids", I32, &grid)?))?;
        let top_ids = ids_of(i32s(tensor_bytes(&st, "reference_top_ids", I32, &tops)?))?;
        let top = f32s(tensor_bytes(&st, "reference_top_logits", F32, &tops)?);
        let lse = f32s(tensor_bytes(&st, "reference_logsumexp", F32, &grid)?);
        let floor_top = f32s(tensor_bytes(&st, "floor_top_logits", F32, &tops)?);
        let floor_lse = f32s(tensor_bytes(&st, "floor_logsumexp", F32, &grid)?);
        let floor_argmax = ids_of(i32s(tensor_bytes(&st, "floor_argmax", I32, &grid)?))?;
        if k < 2 {
            return Err(Error::Model(format!(
                "vendor reference: {k} top logits per position; the margin needs two"
            )));
        }
        let positions = top_ids
            .chunks_exact(k)
            .zip(top.chunks_exact(k))
            .zip(&lse)
            .map(|((ids, top), &lse)| Position {
                top_ids: ids.to_vec(),
                top: top.to_vec(),
                lse,
            })
            .collect();
        let floor = floor_top
            .chunks_exact(k)
            .zip(&floor_lse)
            .zip(&floor_argmax)
            .map(|((top, &lse), &argmax)| Seen {
                top: top.to_vec(),
                lse,
                argmax,
            })
            .collect();
        Ok(Self {
            ids: token_ids
                .chunks_exact(len)
                .map(|row| row.iter().map(|&id| id as u32).collect())
                .collect(),
            positions,
            floor,
        })
    }
}

/// A candidate against the reference.
#[derive(Debug, Clone)]
struct Stats {
    /// Largest `|candidate - reference|` over the reference's top ids of every
    /// position that a sampler reaches ([`REACHABLE`]).
    reach_max: f64,
    /// Largest `|candidate - reference|` over all the reference's top ids.
    /// Reported, not gated.
    top_max: f64,
    /// Root mean square of the differences over all the reference's top ids.
    top_rms: f64,
    /// Mean over positions of `KL(reference || candidate)` on the buckets
    /// "each top id" and "the rest of the vocabulary": a lower bound of the
    /// full divergence.
    kl: f64,
    /// Positions whose argmax over the vocabulary is not the reference's.
    disagree: Vec<usize>,
}

/// `KL(reference || candidate)` at one position, on the top ids and the rest.
fn bucket_kl(at: &Position, seen: &Seen) -> f64 {
    let log_p: Vec<f64> = at
        .top
        .iter()
        .map(|&v| f64::from(v) - f64::from(at.lse))
        .collect();
    let log_q: Vec<f64> = seen
        .top
        .iter()
        .map(|&v| f64::from(v) - f64::from(seen.lse))
        .collect();
    let rest =
        |logs: &[f64]| (1.0 - logs.iter().map(|l| l.exp()).sum::<f64>()).max(f64::MIN_POSITIVE);
    let (p_rest, q_rest) = (rest(&log_p), rest(&log_q));
    log_p
        .iter()
        .zip(&log_q)
        .map(|(p, q)| p.exp() * (p - q))
        .sum::<f64>()
        + p_rest * (p_rest.ln() - q_rest.ln())
}

fn stats(reference: &[Position], seen: &[Seen]) -> Result<Stats> {
    if reference.len() != seen.len() || reference.is_empty() {
        return Err(Error::Model(format!(
            "{} candidate positions against {} reference positions",
            seen.len(),
            reference.len()
        )));
    }
    let mut reach_max = 0.0f64;
    let mut top_max = 0.0f64;
    let mut square_sum = 0.0f64;
    let mut count = 0usize;
    let mut kl = 0.0f64;
    let mut disagree = Vec::new();
    for (p, (at, s)) in reference.iter().zip(seen).enumerate() {
        if at.top.len() != s.top.len() {
            return Err(Error::Model(format!(
                "position {p}: {} candidate logits against {} reference logits",
                s.top.len(),
                at.top.len()
            )));
        }
        for (rank, (r, c)) in at.top.iter().zip(&s.top).enumerate() {
            let d = f64::from(*c) - f64::from(*r);
            if rank == 0 || (f64::from(*r) - f64::from(at.lse)).exp() >= REACHABLE {
                reach_max = reach_max.max(d.abs());
            }
            top_max = top_max.max(d.abs());
            square_sum += d * d;
            count += 1;
        }
        kl += bucket_kl(at, s);
        if at.top_ids.first() != Some(&s.argmax) {
            disagree.push(p);
        }
    }
    Ok(Stats {
        reach_max,
        top_max,
        top_rms: (square_sum / count as f64).sqrt(),
        kl: kl / reference.len() as f64,
        disagree,
    })
}

/// The rules `candidate` breaks, empty when it passes:
/// - its argmax is the reference's at every decided position, one whose
///   reference top-2 margin is above `floor.reach_max`;
/// - it disagrees at no more than `GATE x max(floor disagreements, 1)`
///   positions in all;
/// - `reach_max`, `top_rms` and `kl` are each at most `GATE` times the floor's.
fn gate(reference: &[Position], floor: &Stats, candidate: &Stats) -> Vec<String> {
    let reach = floor.reach_max;
    let mut broken: Vec<String> = candidate
        .disagree
        .iter()
        .filter_map(|&p| {
            let margin = reference.get(p).and_then(|at| match at.top.as_slice() {
                [first, second, ..] => Some(f64::from(*first) - f64::from(*second)),
                _ => None,
            })?;
            (margin > reach).then(|| {
                format!("argmax differs at decided position {p}: margin {margin:.4} > {reach:.4}")
            })
        })
        .collect();
    let budget = GATE * floor.disagree.len().max(1) as f64;
    if candidate.disagree.len() as f64 > budget {
        broken.push(format!(
            "argmax differs at {} positions, budget {budget}",
            candidate.disagree.len()
        ));
    }
    for (name, c, f) in [
        ("reach_max", candidate.reach_max, floor.reach_max),
        ("top_rms", candidate.top_rms, floor.top_rms),
        ("kl", candidate.kl, floor.kl),
    ] {
        if c > GATE * f {
            broken.push(format!("{name} {c:.4e} > {GATE} x floor {f:.4e}"));
        }
    }
    broken
}

/// The statistics `control` does not push past `CONTROL` times the floor's.
fn control_short(floor: &Stats, control: &Stats) -> Vec<String> {
    [
        ("reach_max", control.reach_max, floor.reach_max),
        ("top_rms", control.top_rms, floor.top_rms),
        ("kl", control.kl, floor.kl),
    ]
    .into_iter()
    .filter(|&(_, c, f)| c <= CONTROL * f)
    .map(|(name, c, f)| format!("control {name} {c:.4e} <= {CONTROL} x floor {f:.4e}"))
    .collect()
}

/// One line of figures for `what`, with each statistic as a multiple of the
/// floor's.
fn report(what: &str, s: &Stats, floor: &Stats) -> String {
    format!(
        "{what}: reach_max {:.4e} ({:.2} x floor), top_rms {:.4e} ({:.2} x), kl {:.4e} ({:.2} x), \
         argmax differs at {} positions {:?}; top_max over 20 ranks {:.4e} ({:.2} x)",
        s.reach_max,
        s.reach_max / floor.reach_max,
        s.top_rms,
        s.top_rms / floor.top_rms,
        s.kl,
        s.kl / floor.kl,
        s.disagree.len(),
        s.disagree,
        s.top_max,
        s.top_max / floor.top_max
    )
}

/// `model`'s logits over each sequence of `reference`, with fresh caches:
/// one prefill forward over the first `prefill` ids, then one decode forward
/// per remaining id. With `prefill` the sequence length this is one prefill
/// forward. Returns the positions read, the floor at them and the
/// candidate's logits at the reference's ids.
fn seen_by(
    model: &Qwen3_5MoeText,
    reference: &VendorReference,
    prefill: usize,
) -> Result<(Vec<Position>, Vec<Seen>, Vec<Seen>)> {
    let layers = model.cfg.num_hidden_layers;
    let mut read = (Vec::new(), Vec::new(), Vec::new());
    let mut first = 0;
    for ids in &reference.ids {
        let (head, tail) = ids
            .split_at_checked(prefill)
            .filter(|(head, _)| !head.is_empty())
            .ok_or_else(|| Error::Model(format!("prefill {prefill} of {} ids", ids.len())))?;
        let mut kv: Vec<KvCache> = (0..layers)
            .map(|_| KvCache::with_quant(KvQuant::None))
            .collect();
        let mut lin: Vec<LinearAttnCache> = (0..layers).map(|_| LinearAttnCache::new()).collect();
        let mut rows = vec![read_f32(&model.forward_seq_last_k_with_cache(
            head,
            if tail.is_empty() { head.len() } else { 1 },
            &mut kv,
            Some(&mut lin),
            Device::Gpu,
        )?)?];
        for id in tail {
            rows.push(read_f32(&model.forward_seq_last_k_with_cache(
                std::slice::from_ref(id),
                1,
                &mut kv,
                Some(&mut lin),
                Device::Gpu,
            )?)?);
        }
        let skipped = if tail.is_empty() { 0 } else { prefill - 1 };
        let vocab = rows.first().map_or(0, Vec::len) / if tail.is_empty() { ids.len() } else { 1 };
        for (offset, row) in rows.iter().flat_map(|r| r.chunks_exact(vocab)).enumerate() {
            let p = first + skipped + offset;
            let (Some(at), Some(floor)) = (reference.positions.get(p), reference.floor.get(p))
            else {
                return Err(Error::Model(format!("no reference position {p}")));
            };
            read.2.push(Seen::of(at, row)?);
            read.0.push(at.clone());
            read.1.push(floor.clone());
        }
        first += ids.len();
    }
    Ok(read)
}

/// `layer` replaced by the folded layer it wraps when it is a Hadamard layer;
/// the transform it took off.
fn take_transform(layer: &mut Linear) -> Option<Arc<SignedBlockHadamard>> {
    let taken = std::mem::replace(
        layer,
        Linear::Plain {
            weight: rmlx_mlx::scalar_f32(0.0),
        },
    );
    if let Linear::Hadamard { transform, inner } = taken {
        *layer = *inner;
        Some(transform)
    } else {
        *layer = taken;
        None
    }
}

fn unwrap_embedding(table: &mut Embedding) -> bool {
    let taken = std::mem::replace(
        table,
        Embedding::Plain {
            weight: rmlx_mlx::scalar_f32(0.0),
        },
    );
    if let Embedding::Hadamard { inner, .. } = taken {
        *table = *inner;
        true
    } else {
        *table = taken;
        false
    }
}

/// Replace every Hadamard layer of `model` by the folded layer it wraps, so
/// the model computes with the pack's weights and no transform. Returns the
/// number of layers replaced.
fn drop_transforms(model: &mut Qwen3_5MoeText) -> usize {
    let mut linears: Vec<&mut Linear> = model.lm_head.iter_mut().collect();
    for layer in &mut model.layers {
        match &mut layer.attn {
            AttnBlock::Linear(g) => linears.extend([
                &mut g.in_proj_qkv,
                &mut g.in_proj_z,
                &mut g.in_proj_b,
                &mut g.in_proj_a,
                &mut g.out_proj,
            ]),
            AttnBlock::Full(a) => {
                linears.extend([&mut a.q_proj, &mut a.k_proj, &mut a.v_proj, &mut a.o_proj]);
            }
        }
        match &mut layer.mlp {
            MlpBlock::Dense(m) => {
                linears.extend([&mut m.gate_proj, &mut m.up_proj, &mut m.down_proj]);
            }
            // The loader refuses a sparse MoE layer in a pack.
            MlpBlock::Moe(_) => {}
        }
    }
    let linear = linears.into_iter().filter_map(take_transform).count();
    linear + usize::from(unwrap_embedding(&mut model.embed_tokens))
}

/// The `down_proj` of decoder layer `layer`, the module the one-module
/// controls take the transform off.
fn down_proj(model: &mut Qwen3_5MoeText, layer: usize) -> Result<&mut Linear> {
    match model.layers.get_mut(layer).map(|l| &mut l.mlp) {
        Some(MlpBlock::Dense(m)) => Ok(&mut m.down_proj),
        _ => Err(Error::Model(format!("layer {layer} has no dense MLP"))),
    }
}

/// The gate's verdict on `model` with the transform of layer `layer`'s
/// `down_proj` taken off, which is then put back.
fn one_module_control(
    model: &mut Qwen3_5MoeText,
    reference: &VendorReference,
    floor: &Stats,
    layer: usize,
) -> Result<(Stats, Vec<String>)> {
    let transform = take_transform(down_proj(model, layer)?)
        .ok_or_else(|| Error::Model(format!("layer {layer} down_proj is not folded")))?;
    let control = stats(&reference.positions, &seen_by(model, reference, LENGTH)?.2);
    let slot = down_proj(model, layer)?;
    let inner = std::mem::replace(
        slot,
        Linear::Plain {
            weight: rmlx_mlx::scalar_f32(0.0),
        },
    );
    *slot = Linear::hadamard(transform, inner)?;
    let control = control?;
    let broken = gate(&reference.positions, floor, &control);
    Ok((control, broken))
}

/// The real pack's logits meet the gate against the vendor runtime, in one
/// prefill and in decode steps; the same model with its transforms removed
/// misses the floor by more than [`CONTROL`] times on every statistic, and
/// with the transform of one module removed it breaks the gate.
#[test]
#[ignore = "GPU Metal: loads the real pack and runs it on Device::Gpu"]
fn the_real_pack_meets_the_vendor_logits_and_misses_them_without_its_transforms_on_gpu(
) -> Result<()> {
    let test =
        "the_real_pack_meets_the_vendor_logits_and_misses_them_without_its_transforms_on_gpu";
    let Some(dir) = crate::test_snapshot::snapshot_by_slug(test, PACK, &[PACK_IDENTITY]) else {
        return Ok(());
    };
    let reference = VendorReference::read(Path::new(FIXTURE))?;
    let floor = stats(&reference.positions, &reference.floor)?;
    let crate::arch::Architecture::Qwen3_5Moe(mut model) =
        crate::arch::load_model(&dir, Device::Gpu, &crate::arch::LoadOpts::default())?
    else {
        return Err(Error::Model(format!(
            "{PACK} loads as another architecture"
        )));
    };
    let pack = stats(
        &reference.positions,
        &seen_by(&model, &reference, LENGTH)?.2,
    )?;
    let (decode_at, decode_floor, decoded) = seen_by(&model, &reference, DECODE_FROM)?;
    let decode_floor = stats(&decode_at, &decode_floor)?;
    let decode = stats(&decode_at, &decoded)?;
    println!("{}", report("floor (vendor bf16 cast)", &floor, &floor));
    println!("{}", report("rMLX", &pack, &floor));
    println!("{}", report("decode floor", &decode_floor, &decode_floor));
    println!("{}", report("rMLX decode", &decode, &decode_floor));
    let mut broken = gate(&reference.positions, &floor, &pack);
    broken.extend(gate(&decode_at, &decode_floor, &decode));
    for layer in [0, 63] {
        let (one, one_broken) = one_module_control(&mut model, &reference, &floor, layer)?;
        let what = format!("rMLX, layer {layer} down_proj transform removed");
        println!(
            "{}; gate breaks {one_broken:?}",
            report(&what, &one, &floor)
        );
        if one_broken.is_empty() {
            broken.push(format!("{what} passes the gate"));
        }
    }
    let dropped = drop_transforms(&mut model);
    let control = stats(
        &reference.positions,
        &seen_by(&model, &reference, LENGTH)?.2,
    )?;
    println!("{}", report("rMLX, transforms removed", &control, &floor));
    broken.extend(control_short(&floor, &control));
    if dropped != PACK_MODULES {
        broken.push(format!(
            "the control removed {dropped} transforms, the pack folds {PACK_MODULES}"
        ));
    }
    if broken.is_empty() {
        Ok(())
    } else {
        Err(Error::Model(broken.join("; ")))
    }
}

/// The committed reference: three sequences of 64 ids, the reference's top
/// logits in descending order under its log-sum-exp, and a floor that is not
/// zero.
#[test]
fn the_committed_vendor_reference_is_well_formed_and_has_a_floor() -> Result<()> {
    let reference = VendorReference::read(Path::new(FIXTURE))?;
    assert_eq!(
        reference.ids.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![LENGTH; 3]
    );
    assert_eq!(reference.positions.len(), 192);
    for (p, at) in reference.positions.iter().enumerate() {
        assert_eq!(at.top.len(), 20, "position {p}");
        assert!(
            at.top.windows(2).all(|w| matches!(w, [a, b] if a >= b)),
            "position {p}: the top logits are not in descending order"
        );
        assert!(
            at.top.first().is_some_and(|&t| t <= at.lse),
            "position {p}: the largest logit exceeds the log-sum-exp"
        );
    }
    let floor = stats(&reference.positions, &reference.floor)?;
    println!("{}", report("floor (vendor bf16 cast)", &floor, &floor));
    let near = |got: f64, pinned: f64| ((got - pinned) / pinned).abs() < 1e-4;
    assert!(
        near(floor.reach_max, 0.482_04)
            && near(floor.top_max, 0.617_07)
            && near(floor.top_rms, 0.043_979)
            && near(floor.kl, 1.748_9e-4),
        "the committed floor moved: {floor:?}"
    );
    assert_eq!(floor.disagree, vec![13]);
    Ok(())
}

#[cfg(test)]
#[path = "pack_gate_tests.rs"]
mod pack_gate_tests;
