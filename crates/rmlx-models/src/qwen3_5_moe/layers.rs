//! Local layer primitives: Linear (plain/quantized/PARO/Hadamard), Embedding,
//! RmsNorm, helpers.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

use rmlx_core::error::{Error, Result};
use rmlx_mlx::compile::{compile_shapeless, Closure};
use rmlx_mlx::{broadcast_to, expand_dims, rms_norm, Array, Device, Dtype};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock};

use crate::hadamard::SignedBlockHadamard;

// ---------------------------------------------------------------------------
// Local layer primitives
// ---------------------------------------------------------------------------

/// Pre-computed rotation parameters for a single PARO linear layer.
///
/// Built once at load time from the raw tensor bytes in the checkpoint.
/// All arrays are GPU-resident F16/I32 arrays ready for `paro_rotate_gpu`.
#[allow(missing_debug_implementations)]
pub(super) struct ParoRotation {
    /// `[krot, hidden/2]` I32 packed pair indices.
    pub(super) packed_pairs: Array,
    /// `[krot, hidden/2]` F16 cosine values.
    pub(super) cos_theta: Array,
    /// `[krot, hidden/2]` F16 sine values.
    pub(super) sin_theta: Array,
    /// `[1, hidden]` F16 per-channel scales.
    pub(super) channel_scales: Array,
    /// Actual krot for this layer.
    pub(super) krot: usize,
    /// Group size used by both the rotation kernel and the INT4 matmul.
    pub(super) group_size: usize,
}

#[allow(missing_debug_implementations)]
pub(super) enum Linear {
    Plain {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
    /// ParoQuant INT4: pre-rotation of input activations + standard INT4 affine matmul.
    ///
    /// Forward:
    /// 1. Reshape x to [batch, hidden].
    /// 2. paro_rotate_gpu(x, rotation params).
    /// 3. quantized_matmul(rotated_x, weight, scales, biases, 128, 4, "affine", true).
    Paro {
        rotation: ParoRotation,
        weight: Array,
        scales: Array,
        biases: Array, // always present in PARO (derived from qzeros)
    },
    /// A linear layer of a Prism Hadamard pack: `inner` is the quantized
    /// folded weight `W' = W D H`, so `y = W' . transform.forward(x)`. Build
    /// it with [`Linear::hadamard`].
    Hadamard {
        transform: Arc<SignedBlockHadamard>,
        inner: Box<Linear>,
    },
}

/// The input width of a quantized weight `[rows, width * bits / 32]` (U32).
fn quantized_width(weight: &Array, bits: i32) -> Option<usize> {
    let packed = usize::try_from(*weight.shape().last()?).ok()?;
    let bits = usize::try_from(bits).ok().filter(|b| *b > 0)?;
    (packed * 32)
        .is_multiple_of(bits)
        .then(|| packed * 32 / bits)
}

/// Refuse a folded weight that is not quantized (`None`) or whose input width
/// is not the sign width of `transform`.
fn check_folded(
    what: &str,
    transform: &SignedBlockHadamard,
    weight: Option<(&Array, i32)>,
) -> Result<()> {
    let Some((weight, bits)) = weight else {
        return Err(Error::Model(format!(
            "Hadamard {what}: the folded weight is not quantized"
        )));
    };
    if weight.ndim() != 2 {
        return Err(Error::Model(format!(
            "Hadamard {what}: the folded weight has {} axes; an expert stack \
             [experts, rows, columns] has no fold",
            weight.ndim()
        )));
    }
    match quantized_width(weight, bits) {
        Some(width) if width == transform.width() => Ok(()),
        width => Err(Error::Model(format!(
            "Hadamard {what}: weight input width {} is not the sign width {}",
            width.map_or_else(|| "unknown".to_owned(), |w| w.to_string()),
            transform.width()
        ))),
    }
}

impl Linear {
    /// A Hadamard linear layer over the quantized folded weight `inner`.
    ///
    /// # Errors
    /// `Error::Model` when `inner` is not quantized or its input width is not
    /// the sign width of `transform`.
    pub(super) fn hadamard(transform: Arc<SignedBlockHadamard>, inner: Linear) -> Result<Linear> {
        let weight = match &inner {
            Linear::Quantized { weight, bits, .. } => Some((weight, *bits)),
            Linear::Plain { .. } | Linear::Paro { .. } | Linear::Hadamard { .. } => None,
        };
        check_folded("linear", &transform, weight)?;
        Ok(Linear::Hadamard {
            transform,
            inner: Box::new(inner),
        })
    }

    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Linear::Plain { weight } => {
                rmlx_mlx::matmul(x, &weight.transpose(&[1, 0], device)?, device)
            }
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => rmlx_mlx::quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                true,
                device,
            ),
            Linear::Paro {
                rotation,
                weight,
                scales,
                biases,
            } => {
                // Flatten to [batch_flat, hidden] for the rotation kernel.
                let shape = x.shape();
                let hidden = *shape.last().unwrap_or(&0) as usize;
                let batch_flat: i32 = shape.iter().product::<i32>() / hidden as i32;
                let x_2d = x.reshape(&[batch_flat, hidden as i32], device)?;

                let rotated = crate::paroquant_msl::paro_rotate_gpu(
                    &x_2d,
                    &rotation.packed_pairs,
                    &rotation.cos_theta,
                    &rotation.sin_theta,
                    &rotation.channel_scales,
                    rotation.krot,
                    rotation.group_size,
                    device,
                )?;

                // Restore original shape before matmul.
                let rotated = if shape.len() > 2 {
                    rotated.reshape(&shape, device)?
                } else {
                    rotated
                };

                rmlx_mlx::quantized_matmul(
                    &rotated,
                    weight,
                    scales,
                    Some(biases),
                    rotation.group_size as i32,
                    4,
                    "affine",
                    true,
                    device,
                )
            }
            Linear::Hadamard { transform, inner } => {
                inner.forward(&transform.forward(x, device)?, device)
            }
        }
    }

    /// Batched expert dispatch via gather_qmm.
    /// `x`: `[n_tokens, 1, 1, hidden]` (caller expands before calling).
    /// `rhs_indices`: `[n_tokens, top_k]`.
    /// Returns `[n_tokens, top_k, 1, out_dim]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn gather_forward(
        &self,
        x: &Array,
        rhs_indices: &Array,
        sorted_indices: bool,
        device: Device,
    ) -> Result<Array> {
        match self {
            Linear::Hadamard { .. } => Err(Error::Model(
                "a Hadamard linear has no expert dispatch: Hadamard packs are dense".to_owned(),
            )),
            // Paro layers do not appear as MoE expert projections (PARO 27B is dense).
            // Fall back to the standard quantized matmul after rotation.
            Linear::Paro { .. } => {
                let s = x.shape();
                let n_tokens = s[0];
                let tk = rhs_indices.shape()[1];
                let hidden_in = *s.last().unwrap_or(&0);
                let x_2d = x.reshape(&[n_tokens * tk, hidden_in], device)?;
                // Reshape rhs_indices to [n_tokens*tk] for sequential expert dispatch.
                // This is a slow fallback — PARO models are dense, so this path is not hit.
                let logit = self.forward(&x_2d, device)?;
                let out_dim = logit.shape()[1];
                logit.reshape(&[n_tokens, tk, 1, out_dim], device)
            }
            Linear::Plain { weight } => {
                // Fallback: slow but correct. Rarely hit (all tensors quantized in practice).
                let s = x.shape();
                let nd = s.len();
                let n_tokens = s[0];
                let tk = rhs_indices.shape()[1];
                let hidden_in = s[nd - 1];

                let x_flat = x.reshape(&[n_tokens, hidden_in], device)?;
                let rhs_flat = rhs_indices.reshape(&[n_tokens * tk], device)?;
                let w_sel = weight.take(&rhs_flat, 0, device)?;
                let mut tok_data = vec![0i32; (n_tokens * tk) as usize];
                for i in 0..(n_tokens as usize) {
                    for j in 0..(tk as usize) {
                        tok_data[i * tk as usize + j] = i as i32;
                    }
                }
                let tok_bytes = unsafe {
                    std::slice::from_raw_parts(tok_data.as_ptr().cast::<u8>(), tok_data.len() * 4)
                };
                let tok_idx = Array::from_bytes(tok_bytes, &[n_tokens * tk], Dtype::I32)?;
                let x_sel = x_flat.take(&tok_idx, 0, device)?;
                let out = rmlx_mlx::matmul(&x_sel, &w_sel.transpose(&[0, 2, 1], device)?, device)?;
                let out_dim = out.shape()[1];
                out.reshape(&[n_tokens, tk, 1, out_dim], device)
            }
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => rmlx_mlx::gather_qmm(
                x,
                weight,
                scales,
                biases.as_ref(),
                None,
                rhs_indices,
                *group_size,
                *bits,
                mode,
                sorted_indices,
                device,
            ),
        }
    }
}

#[allow(missing_debug_implementations)]
pub(super) enum Embedding {
    Plain {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
    /// The embedding of a Prism Hadamard pack: `inner` is the quantized folded
    /// table `E' = E D H`, so a row is `transform.inverse(E'[id])`. Build it
    /// with [`Embedding::hadamard`].
    Hadamard {
        transform: Arc<SignedBlockHadamard>,
        inner: Box<Embedding>,
    },
}

impl Embedding {
    /// A Hadamard embedding over the quantized folded table `inner`.
    ///
    /// # Errors
    /// `Error::Model` when `inner` is not quantized or its row width is not
    /// the sign width of `transform`.
    pub(super) fn hadamard(
        transform: Arc<SignedBlockHadamard>,
        inner: Embedding,
    ) -> Result<Embedding> {
        let weight = match &inner {
            Embedding::Quantized { weight, bits, .. } => Some((weight, *bits)),
            Embedding::Plain { .. } | Embedding::Hadamard { .. } => None,
        };
        check_folded("embedding", &transform, weight)?;
        Ok(Embedding::Hadamard {
            transform,
            inner: Box::new(inner),
        })
    }

    pub(super) fn forward(&self, ids: &Array, device: Device) -> Result<Array> {
        match self {
            Embedding::Hadamard { transform, inner } => {
                transform.inverse(&inner.forward(ids, device)?, device)
            }
            Embedding::Plain { weight } => weight.take(ids, 0, device),
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => embed_lookup(
                ids,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                device,
            ),
        }
    }

    pub(super) fn as_linear(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            // The pack contract refuses a tied head; it defines no forward use
            // of the inverse-folded table.
            Embedding::Hadamard { .. } => Err(Error::Model(
                "a Hadamard embedding is not a tied output head".to_owned(),
            )),
            Embedding::Plain { weight } => {
                rmlx_mlx::matmul(x, &weight.transpose(&[1, 0], device)?, device)
            }
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => rmlx_mlx::quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                true,
                device,
            ),
        }
    }
}

/// The Hadamard folding of one pack: one transform per input width, shared
/// by every module of that width, and the modules the contract folds forward
/// (linear layers) or inverse (the embedding), keyed by tensor name without
/// `.weight`.
///
/// The loader passes each layer it builds through [`HadamardFold::linear`]
/// or [`HadamardFold::embedding`], then calls [`HadamardFold::finish`],
/// which refuses a contract module the loader built no layer for: that
/// module's weights would otherwise stay unread or load unfolded.
pub(super) struct HadamardFold {
    transforms: BTreeMap<usize, Arc<SignedBlockHadamard>>,
    forward: BTreeMap<String, usize>,
    inverse: BTreeMap<String, usize>,
    built: RefCell<BTreeSet<String>>,
}

impl HadamardFold {
    /// # Errors
    /// `Error::Model` when a sign vector builds no transform.
    pub(super) fn new(
        block: usize,
        signs: &BTreeMap<usize, Vec<f32>>,
        forward: BTreeMap<String, usize>,
        inverse: BTreeMap<String, usize>,
    ) -> Result<Self> {
        let transforms = signs
            .iter()
            .map(|(width, s)| Ok((*width, Arc::new(SignedBlockHadamard::new(block, s)?))))
            .collect::<Result<_>>()?;
        Ok(Self {
            transforms,
            forward,
            inverse,
            built: RefCell::new(BTreeSet::new()),
        })
    }

    /// The transform of contract module `name`, which this marks as built.
    fn transform(&self, name: &str, width: usize) -> Result<Arc<SignedBlockHadamard>> {
        self.built.borrow_mut().insert(name.to_owned());
        self.transforms.get(&width).cloned().ok_or_else(|| {
            Error::Model(format!(
                "Hadamard module {name}: no sign vector of width {width}"
            ))
        })
    }

    /// `inner` folded when the contract names `name` a forward module, else
    /// `inner` unchanged.
    ///
    /// # Errors
    /// `Error::Model` when `name` is the contract's inverse module, or when
    /// [`Linear::hadamard`] refuses `inner`.
    pub(super) fn linear(&self, name: &str, inner: Linear) -> Result<Linear> {
        if self.inverse.contains_key(name) {
            return Err(Error::Model(format!(
                "Hadamard module {name} is folded inverse; it is not a linear layer"
            )));
        }
        match self.forward.get(name) {
            Some(&width) => Linear::hadamard(self.transform(name, width)?, inner),
            None => Ok(inner),
        }
    }

    /// `inner` folded when the contract names `name` the inverse module, else
    /// `inner` unchanged.
    ///
    /// # Errors
    /// `Error::Model` when `name` is a forward module of the contract, or when
    /// [`Embedding::hadamard`] refuses `inner`.
    pub(super) fn embedding(&self, name: &str, inner: Embedding) -> Result<Embedding> {
        if self.forward.contains_key(name) {
            return Err(Error::Model(format!(
                "Hadamard module {name} is folded forward; it is not an embedding"
            )));
        }
        match self.inverse.get(name) {
            Some(&width) => Embedding::hadamard(self.transform(name, width)?, inner),
            None => Ok(inner),
        }
    }

    /// # Errors
    /// `Error::Model` naming the first contract module no layer was built for.
    pub(super) fn finish(self) -> Result<()> {
        let built = self.built.into_inner();
        match self
            .forward
            .keys()
            .chain(self.inverse.keys())
            .find(|name| !built.contains(*name))
        {
            Some(name) => Err(Error::Model(format!(
                "Hadamard pack: the loader built no layer for contract module {name}"
            ))),
            None => Ok(()),
        }
    }
}

/// Quantized embedding lookup: gather the selected rows on-device and
/// `dequantize` them. Mirrors mlx-lm's `QuantizedEmbedding.__call__`
/// (`dequantize(weight[ids], scales[ids], biases[ids], …)`).
///
/// Earlier versions ran this through `Device::Cpu` with an `eye(seq) @ w`
/// quantized-matmul trick. That is `O(seq²)` in the identity matrix and forces
/// a GPU↔CPU round-trip: for short text prompts it is merely wasteful, but for
/// longer sequences the `seq²` CPU matmul grows quickly. The on-device
/// `take + dequantize` path is `O(seq)` and keeps everything on `device`,
/// letting MLX fuse the lookup with the following layers (identical to
/// [`crate::qwen3::qwen_embedding_lookup`]).
pub(super) fn embed_lookup(
    ids: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: &str,
    device: Device,
) -> Result<Array> {
    let weight_rows = weight.take(ids, 0, device)?;
    let scales_rows = scales.take(ids, 0, device)?;
    let biases_rows = biases.map(|b| b.take(ids, 0, device)).transpose()?;
    let dq = rmlx_mlx::dequantize(
        &weight_rows,
        &scales_rows,
        biases_rows.as_ref(),
        group_size,
        bits,
        mode,
        device,
    )?;
    // Downstream layers (RmsNorm, attention masks) expect BF16 activations.
    // `dequantize` returns the scales' dtype; force BF16 so downstream
    // mask promotion stays consistent.
    if dq.dtype() == Dtype::Bf16 {
        Ok(dq)
    } else {
        dq.astype(Dtype::Bf16, device)
    }
}

pub(super) struct RmsNorm {
    pub(super) weight: Array,
    pub(super) eps: f32,
}

impl RmsNorm {
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        rms_norm(x, Some(&self.weight), self.eps, device)
    }
}

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn repeat_kv(x: &Array, repeat: usize, device: Device) -> Result<Array> {
    let s = x.shape();
    if repeat == 1 {
        return x.reshape(&s, device);
    }
    let (b, kv_h, seq, d) = (s[0], s[1], s[2], s[3]);
    let x5 = expand_dims(x, 2, device)?;
    let bc = broadcast_to(&x5, &[b, kv_h, repeat as i32, seq, d], device)?;
    bc.reshape(&[b, kv_h * repeat as i32, seq, d], device)
}

// ---------------------------------------------------------------------------
// qk_norm_fused — compile_shapeless fusion of (q rms_norm, k rms_norm)
// ---------------------------------------------------------------------------
//
// Collapse the two RMSNorm dispatches that bracket Q and K into one
// compiled Metal program per layer per step. Pattern lifted from
// `qwen3.rs::qk_norm_fused`. Same shape-agnostic dtype/device/eps cache.

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct QkNormKey {
    in_dtype_tag: u8,
    device_tag: u8,
    eps_bits: u32,
}

fn qk_norm_dtype_tag(d: Dtype) -> u8 {
    match d {
        Dtype::Bf16 => 0,
        Dtype::F16 => 1,
        Dtype::F32 => 2,
        Dtype::U8 => 3,
        Dtype::U32 => 4,
        Dtype::I32 => 5,
    }
}

fn qk_norm_device_tag(d: Device) -> u8 {
    match d {
        Device::Gpu => 0,
        Device::Cpu => 1,
    }
}

static QK_NORM_COMPILE_CACHE: OnceLock<Mutex<FxHashMap<QkNormKey, Arc<Closure>>>> = OnceLock::new();

fn qk_norm_compile_cache() -> &'static Mutex<FxHashMap<QkNormKey, Arc<Closure>>> {
    QK_NORM_COMPILE_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn qk_norm_get_or_compile(key: QkNormKey, eps: f32, device: Device) -> Result<Arc<Closure>> {
    {
        let cache = qk_norm_compile_cache()
            .lock()
            .expect("qk_norm cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(Arc::clone(cls));
        }
    }
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 4 {
            return Err(Error::Mlx(format!(
                "qk_norm_fused closure: expected 4 inputs (q, k, q_w, k_w), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let q = iter.next().expect("q");
        let k = iter.next().expect("k");
        let q_w = iter.next().expect("q_w");
        let k_w = iter.next().expect("k_w");
        let qn = rms_norm(&q, Some(&q_w), eps, device)?;
        let kn = rms_norm(&k, Some(&k_w), eps, device)?;
        Ok(vec![qn, kn])
    });
    let compiled = compile_shapeless(raw)
        .map_err(|e| Error::Mlx(format!("qk_norm compile_shapeless: {e}")))?;
    let arc = Arc::new(compiled);
    let mut cache = qk_norm_compile_cache()
        .lock()
        .expect("qk_norm cache poisoned");
    Ok(Arc::clone(
        cache.entry(key).or_insert_with(|| Arc::clone(&arc)),
    ))
}

/// Fused per-head Q/K RMSNorm via one compiled closure.
///
/// Math identical to two separate `rms_norm` calls. Used by Qwen3.5-MoE
/// FullAttention to drop the two RMSNorm dispatches per layer per step
/// to one compiled Metal program.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
pub(super) fn qk_norm_fused(
    q: &Array,
    k: &Array,
    q_w: &Array,
    k_w: &Array,
    eps: f32,
    device: Device,
) -> Result<(Array, Array)> {
    let key = QkNormKey {
        in_dtype_tag: qk_norm_dtype_tag(q.dtype()),
        device_tag: qk_norm_device_tag(device),
        eps_bits: eps.to_bits(),
    };
    let compiled = qk_norm_get_or_compile(key, eps, device)?;
    let mut outs = compiled.apply(&[q, k, q_w, k_w])?;
    if outs.len() != 2 {
        return Err(Error::Mlx(format!(
            "qk_norm_fused: expected 2 outputs, got {}",
            outs.len()
        )));
    }
    let kn = outs.pop().expect("kn");
    let qn = outs.pop().expect("qn");
    Ok((qn, kn))
}

#[cfg(test)]
#[path = "layers_tests.rs"]
pub(super) mod layers_tests;
