//! KV weight-norm calibration algorithm.
//!
//! CPU-only pass: loads K/V projection weights from safetensors shards,
//! computes per-head L2 norms, and returns top-K high-precision index lists.
//!
//! Ported from `multi_turboquant/calibration/generate_metadata.py`.
//!
//! # Public API
//!
//! - [`calibrate_model`] — run the full calibration pass for a model directory.
//! - [`detect_kv_weight_pattern`] — detect the safetensors weight-name pattern.
//! - [`top_k_norm_indices`] — per-tensor top-K norm computation.
//! - [`bytes_to_f32`] — dtype-aware byte-to-float conversion (F32 / BF16 / F16).

use std::collections::BTreeMap;
use std::path::Path;

use tracing::{debug, info};

use rmlx_core::{Error, Result};

use crate::calibration_writer::{outlier_count_for, CalibrationMeta, KvCalibration, LayerCalib};
use crate::{load_config, load_shard_index, QuantConfig, ShardSet};

// ── public API ────────────────────────────────────────────────────────────────

/// Run the KV calibration pass for a model directory.
///
/// Walks all safetensors shards and computes per-head L2 norms for
/// `k_proj.weight` and `v_proj.weight`. Returns a populated [`KvCalibration`]
/// ready for serialization.
///
/// # Arguments
///
/// - `model_dir` — path to the MLX model snapshot.
/// - `recipe` — user-facing recipe name (e.g. `"turbo3"`).
/// - `internal_recipe` — internal recipe name (e.g. `"turboquant35"`).
#[allow(
    clippy::cognitive_complexity,
    reason = "linear setup → detect → loop; each step is straightforward; splitting would add indirection without clarity"
)]
pub fn calibrate_model(
    model_dir: &Path,
    recipe: &str,
    internal_recipe: &str,
) -> Result<KvCalibration> {
    let cfg = load_config(model_dir)?;

    let num_layers = cfg_num_layers(&cfg)
        .ok_or_else(|| Error::Config("num_hidden_layers missing from config.json".to_string()))?;
    let num_kv_heads = cfg_num_kv_heads(&cfg).ok_or_else(|| {
        Error::Config(
            "num_key_value_heads (or num_attention_heads) missing from config.json".to_string(),
        )
    })?;
    let head_dim = cfg.head_dim().ok_or_else(|| {
        Error::Config("head_dim cannot be determined from config.json".to_string())
    })?;
    let dtype_str = cfg
        .extras
        .get("torch_dtype")
        .and_then(|v| v.as_str())
        // mtq defaults to "float16" when config omits torch_dtype (generate_metadata.py:188).
        .unwrap_or("float16")
        .to_string();
    let model_name = model_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let outlier_k = outlier_count_for(head_dim as u32, internal_recipe)?;

    info!(
        model = %model_name,
        num_layers,
        num_kv_heads,
        head_dim,
        recipe,
        internal_recipe,
        outlier_k,
        "kv-calibrate: starting weight-norm pass"
    );

    // Open shard index + shard set.
    let idx = load_shard_index(model_dir)?;
    let shard_set = ShardSet::open(model_dir, &idx)?;
    info!(shards = shard_set.len(), "kv-calibrate: shards opened");

    // Detect weight naming patterns.
    let k_pattern = detect_kv_weight_pattern(&idx.weight_map, "k_proj").ok_or_else(|| {
        Error::Config(
            "cannot detect k_proj weight naming convention; tried model.layers.{i}.self_attn, \
             transformer.h.{i}.attn, model.layers.{i}.attention, and fallback scan"
                .to_string(),
        )
    })?;
    let v_pattern = detect_kv_weight_pattern(&idx.weight_map, "v_proj")
        .unwrap_or_else(|| k_pattern.replace("k_proj", "v_proj"));

    info!(
        k_pattern = %k_pattern,
        v_pattern = %v_pattern,
        "kv-calibrate: weight patterns detected"
    );

    // Calibrate per layer.
    let mut layers: BTreeMap<String, LayerCalib> = BTreeMap::new();

    for layer_idx in 0..num_layers {
        let k_name = k_pattern.replace("{}", &layer_idx.to_string());
        let v_name = v_pattern.replace("{}", &layer_idx.to_string());

        // Sparse-attention models (e.g. Qwen3.5-MoE) place K/V projections only
        // on a subset of layers. Skip layers that have no k_proj weight — there
        // is nothing to calibrate, and the per-layer loop must not hard-error.
        if !idx.weight_map.contains_key(&k_name) {
            debug!(
                layer_idx,
                k_name = %k_name,
                "kv-calibrate: no k_proj weight for this layer — skipping"
            );
            continue;
        }

        debug!(
            layer_idx,
            k_name = %k_name,
            v_name = %v_name,
            "kv-calibrate: processing layer"
        );

        let k_indices = top_k_norm_indices(
            &shard_set,
            &idx.weight_map,
            &k_name,
            num_kv_heads,
            head_dim,
            outlier_k as usize,
            cfg.quantization.as_ref(),
        )
        .map_err(|e| Error::Loader(format!("kv-calibrate layer {layer_idx} k_proj: {e}")))?;

        let v_indices = top_k_norm_indices(
            &shard_set,
            &idx.weight_map,
            &v_name,
            num_kv_heads,
            head_dim,
            outlier_k as usize,
            cfg.quantization.as_ref(),
        )
        .map_err(|e| Error::Loader(format!("kv-calibrate layer {layer_idx} v_proj: {e}")))?;

        let layer_key = layer_key_from_pattern(&k_pattern, layer_idx);
        layers.insert(
            layer_key,
            LayerCalib {
                key_high_precision_indices: k_indices,
                value_high_precision_indices: v_indices,
                codebook: None,
            },
        );
    }

    info!(
        layer_count = layers.len(),
        "kv-calibrate: weight-norm pass complete"
    );

    Ok(KvCalibration {
        version: 1,
        recipe: internal_recipe.to_string(),
        head_size: head_dim as u32,
        model_name,
        transform_version: "structured_hadamard_v1".to_string(),
        codebook_version: "lloyd_beta_v1".to_string(),
        layers,
        calibration: CalibrationMeta {
            method: "weight_norm".to_string(),
            objective: "l2_norm".to_string(),
            num_prompts: 0,
            max_seq_len: 0,
            batch_size: 0,
            num_observed_tokens: 0,
            dtype: dtype_str,
            device: "cpu".to_string(),
            prompts_sha256: String::new(),
        },
        head_budgets: None,
    })
}

// ── weight pattern detection ──────────────────────────────────────────────────

/// Detect the safetensors weight-name pattern for `projection` (e.g. `"k_proj"`).
///
/// Ported from `multi_turboquant/calibration/generate_metadata.py::_detect_weight_pattern`.
///
/// Probes known prefixes at layer index 0, then falls back to a scan of
/// all tensor names looking for `"{projection}"` + `".0."` in the name.
///
/// The `language_model.model.layers.*` prefix is probed for multimodal
/// checkpoints (Gemma4 / Qwen3.5-MoE), whose text decoder is nested under
/// `language_model`. The fallback scan requires the candidate to end in
/// `<projection>.weight` and skips vision/audio-tower tensors so it cannot
/// latch onto a same-named `<projection>.input_max` scalar in an encoder
/// (which is not a quantizable K/V projection weight).
pub fn detect_kv_weight_pattern(
    weight_map: &BTreeMap<String, String>,
    projection: &str,
) -> Option<String> {
    let patterns = [
        format!("model.layers.{{}}.self_attn.{projection}.weight"),
        format!("language_model.model.layers.{{}}.self_attn.{projection}.weight"),
        format!("transformer.h.{{}}.attn.{projection}.weight"),
        format!("model.layers.{{}}.attention.{projection}.weight"),
    ];

    for pattern in &patterns {
        let test_name = pattern.replace("{}", "0");
        if weight_map.contains_key(&test_name) {
            return Some(pattern.clone());
        }
    }

    // Fallback: scan for a real projection *weight* (must end in
    // `<projection>.weight`), skipping vision/audio encoder towers. The layer
    // placeholder is recovered from the `.layers.<N>.` (or `.h.<N>.`) segment
    // of the candidate — NOT a literal `.0.`, because sparse-attention models
    // (e.g. Qwen3.5-MoE) place their first K/V projection on a layer > 0.
    let weight_suffix = format!("{projection}.weight");
    for name in weight_map.keys() {
        let is_tower = name.starts_with("vision_tower")
            || name.starts_with("audio_tower")
            || name.contains(".vision_tower.")
            || name.contains(".audio_tower.");
        if is_tower || !name.ends_with(&weight_suffix) {
            continue;
        }
        if let Some(pat) = templatize_layer_index(name) {
            return Some(pat);
        }
    }

    None
}

/// Replace the numeric layer index in a `*.layers.<N>.*` (or `*.h.<N>.*`)
/// tensor name with the `{}` placeholder, returning the template.
///
/// Returns `None` when no `.layers.<N>.` / `.h.<N>.` segment is found.
fn templatize_layer_index(name: &str) -> Option<String> {
    for marker in [".layers.", ".h."] {
        if let Some(start) = name.find(marker) {
            let after = start + marker.len();
            let rest = name.get(after..)?;
            let digits_end = rest.find(|c: char| !c.is_ascii_digit())?;
            if digits_end == 0 {
                continue; // marker not followed by a number
            }
            let prefix = name.get(..after)?;
            let suffix = rest.get(digits_end..)?;
            return Some(format!("{prefix}{{}}{suffix}"));
        }
    }
    None
}

// ── tensor loading + norm ─────────────────────────────────────────────────────

/// Load a weight tensor, compute per-head L2 norms, and return top-K indices.
///
/// The weight is reshaped to `[num_kv_heads, head_dim, in_dim]`.
/// L2 norm is computed across the input dimension (dim 2 of the reshape).
/// Top-K indices per head are sorted ascending.
///
/// # Float vs quantized snapshots
///
/// - **Unquantized** (F32 / BF16 / F16 weight, no sibling `.scales`): the weight
///   bytes are decoded directly via [`bytes_to_f32`]; `total_elems` is the
///   product of the packed `.weight` shape (which equals the logical shape).
/// - **Quantized** (U32-packed `.weight` + sibling `.scales`): the weight is
///   dequantized to logical `rows * cols` f32 via the shared `rmlx_quant`
///   codecs (affine or mxfp), keyed off `quant.mode_or_default()`. The L2-norm
///   pass then runs on the LOGICAL element count — the packed `.weight` shape
///   is NOT used for `total_elems` on this path. See [`dequant_kv_weight`].
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established: row_base + in_dim = head_base + dim_idx*in_dim + in_dim \
              <= num_kv_heads_usize*head_dim*in_dim = f32_flat.len() by the divisibility check above; \
              norms[..k] is bounded by k <= head_dim = norms.len() after the outlier_count_for guard"
)]
pub fn top_k_norm_indices(
    shard_set: &ShardSet,
    weight_map: &BTreeMap<String, String>,
    tensor_name: &str,
    num_kv_heads: u32,
    head_dim: usize,
    k: usize,
    quant: Option<&QuantConfig>,
) -> Result<Vec<Vec<u32>>> {
    // Quantized iff the sibling `<base>.scales` tensor exists for this `.weight`.
    let scales_name = scales_sibling_name(tensor_name);
    let is_quantized = scales_name
        .as_deref()
        .is_some_and(|s| weight_map.contains_key(s));

    let (f32_flat, total_elems) = if is_quantized {
        let quant = quant.ok_or_else(|| {
            Error::Loader(format!(
                "'{tensor_name}' is quantized (sibling .scales present) but config.json has no \
                 `quantization` block — cannot dequantize for kv-calibrate"
            ))
        })?;
        let flat = dequant_kv_weight(shard_set, weight_map, tensor_name, quant)?;
        let n = flat.len();
        (flat, n)
    } else {
        let shard_filename = weight_map.get(tensor_name).ok_or_else(|| {
            Error::Loader(format!("tensor '{tensor_name}' not found in shard index"))
        })?;

        let handle = shard_set.get(shard_filename).ok_or_else(|| {
            Error::Loader(format!(
                "shard '{shard_filename}' not open (needed for '{tensor_name}')"
            ))
        })?;

        let st = handle.safetensors()?;
        let tensor_view = st.tensor(tensor_name).map_err(|e| {
            Error::Loader(format!(
                "cannot get tensor '{tensor_name}' from shard '{shard_filename}': {e}"
            ))
        })?;

        let shape = tensor_view.shape();
        let dtype = tensor_view.dtype();
        let data_bytes = tensor_view.data();

        let flat = bytes_to_f32(data_bytes, dtype).map_err(|e| {
            Error::Loader(format!(
                "dtype conversion for '{tensor_name}' ({dtype:?}): {e}"
            ))
        })?;

        let total_elems: usize = shape.iter().product();
        if flat.len() != total_elems {
            return Err(Error::Loader(format!(
                "'{tensor_name}': expected {total_elems} f32 elements after dtype conversion, \
                 got {}",
                flat.len()
            )));
        }
        (flat, total_elems)
    };

    let num_kv_heads_usize = num_kv_heads as usize;
    if !total_elems.is_multiple_of(num_kv_heads_usize * head_dim) {
        return Err(Error::Loader(format!(
            "'{tensor_name}': total elements {total_elems} not divisible by \
             num_kv_heads({num_kv_heads_usize}) * head_dim({head_dim})"
        )));
    }
    let in_dim = total_elems / (num_kv_heads_usize * head_dim);

    // bounds proof: k <= head_dim (guaranteed by outlier_count_for returning k < head_dim)
    // and norms.len() == head_dim, so norms[..k] is safe.
    debug_assert!(k < head_dim, "k={k} must be < head_dim={head_dim}");

    let mut result: Vec<Vec<u32>> = Vec::with_capacity(num_kv_heads_usize);

    for head in 0..num_kv_heads_usize {
        let head_base = head * head_dim * in_dim;
        let mut norms: Vec<(usize, f32)> = (0..head_dim)
            .map(|dim_idx| {
                let row_base = head_base + dim_idx * in_dim;
                // SAFETY: row_base + in_dim <= head_base + head_dim * in_dim
                //   = (head + 1) * head_dim * in_dim <= num_kv_heads * head_dim * in_dim
                //   = total_elems = f32_flat.len()
                let norm_sq: f32 = f32_flat[row_base..row_base + in_dim]
                    .iter()
                    .map(|&x| x * x)
                    .sum();
                (dim_idx, norm_sq.sqrt())
            })
            .collect();

        // Sort descending by norm; break ties by index ascending.
        norms.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        // SAFETY: k < head_dim = norms.len() by debug_assert above.
        let mut top_k: Vec<u32> = norms[..k].iter().map(|&(idx, _)| idx as u32).collect();
        top_k.sort_unstable();
        result.push(top_k);
    }

    Ok(result)
}

// ── quantized-weight dequant ───────────────────────────────────────────────────

/// Map a `<base>.weight` tensor name to its sibling `<base>.scales` name.
///
/// Returns `None` when `tensor_name` does not end in `.weight` (no quantized
/// sibling convention applies — treat as plain).
fn scales_sibling_name(tensor_name: &str) -> Option<String> {
    tensor_name
        .strip_suffix(".weight")
        .map(|base| format!("{base}.scales"))
}

/// Read a tensor's raw bytes + shape from the shard set (zero-copy borrow of the
/// mmap is not possible across the safetensors handle here, so bytes are copied).
fn read_tensor_bytes(
    shard_set: &ShardSet,
    weight_map: &BTreeMap<String, String>,
    name: &str,
) -> Result<(Vec<u8>, Vec<usize>, safetensors::Dtype)> {
    let shard_filename = weight_map
        .get(name)
        .ok_or_else(|| Error::Loader(format!("tensor '{name}' not found in shard index")))?;
    let handle = shard_set.get(shard_filename).ok_or_else(|| {
        Error::Loader(format!(
            "shard '{shard_filename}' not open (needed for '{name}')"
        ))
    })?;
    let st = handle.safetensors()?;
    let t = st.tensor(name).map_err(|e| {
        Error::Loader(format!(
            "cannot get tensor '{name}' from shard '{shard_filename}': {e}"
        ))
    })?;
    Ok((t.data().to_vec(), t.shape().to_vec(), t.dtype()))
}

/// Resolve the effective `QuantConfig` for `weight_name`, honouring per-tensor
/// `tensor_overrides` (longest-prefix-match wins). Most snapshots have none, so
/// this returns the top-level config; but a snapshot that overrides a K/V
/// projection's bits/group_size/mode would otherwise be dequantized with the
/// wrong params and silently mis-rank.
fn effective_quant<'a>(quant: &'a QuantConfig, weight_name: &str) -> &'a QuantConfig {
    let Some(overrides) = quant.tensor_overrides.as_ref() else {
        return quant;
    };
    overrides
        .iter()
        .filter(|(prefix, _)| weight_name.starts_with(prefix.as_str()))
        .max_by_key(|(prefix, _)| prefix.len())
        .map_or(quant, |(_, cfg)| cfg)
}

/// Truncate an f32 to bf16 and return its 2 little-endian bytes.
///
/// BF16 is the upper 16 bits of an IEEE-754 f32. Round-to-nearest-even on the
/// dropped low 16 bits keeps the per-group scale faithful enough for the L2-norm
/// ranking. Used to bridge F16-scale affine snapshots (e.g. Ternary-Bonsai) into
/// the BF16-scale `rmlx_quant::affine` codec, which reads scales/biases as bf16.
///
/// Assumes finite input (per-group scales/biases are finite); a NaN/Inf could
/// flip across the RNE carry, but no quant scale buffer carries those.
fn f32_to_bf16_le(x: f32) -> [u8; 2] {
    let bits = x.to_bits();
    // round-to-nearest-even
    let rounding_bias = 0x0000_7FFF + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding_bias);
    ((rounded >> 16) as u16).to_le_bytes()
}

/// Re-encode a scale/bias byte buffer of arbitrary float dtype (F16/BF16/F32)
/// into a flat BF16-LE byte buffer the affine codec accepts.
///
/// BF16 input is returned as-is (no transcode). F16/F32 are decoded via
/// [`bytes_to_f32`] and re-truncated to bf16.
fn scales_to_bf16_le(bytes: &[u8], dtype: safetensors::Dtype) -> Result<Vec<u8>> {
    if dtype == safetensors::Dtype::BF16 {
        return Ok(bytes.to_vec());
    }
    let f32s = bytes_to_f32(bytes, dtype)
        .map_err(|e| Error::Loader(format!("scale/bias dtype conversion ({dtype:?}): {e}")))?;
    let mut out = Vec::with_capacity(f32s.len() * 2);
    for v in f32s {
        out.extend_from_slice(&f32_to_bf16_le(v));
    }
    Ok(out)
}

/// Dequantize a quantized K/V projection `.weight` to a flat logical-row-major
/// `Vec<f32>` of length `rows * cols`, reusing the `rmlx_quant` codecs.
///
/// Dispatch on `quant.mode_or_default()`:
/// - `"affine"` → [`rmlx_quant::affine::dequant_vec`] (weight + scales + biases).
///   Logical cols derived from the `.scales` shape (`scales_cols * group_size`).
/// - `"mxfp8"` / `"mxfp4"` / `"nvfp4"` → [`rmlx_quant::mxfp::dequant_vec`]
///   (weight + scales). For these the packed `.weight` byte length already equals
///   `rows * cols` logical elements; logical cols = `scales_cols * group_size`.
///
/// The returned vector is the LOGICAL `[out_features, in_features]` weight in
/// row-major order — directly consumable by the per-head reshape/L2-norm pass.
fn dequant_kv_weight(
    shard_set: &ShardSet,
    weight_map: &BTreeMap<String, String>,
    weight_name: &str,
    quant: &QuantConfig,
) -> Result<Vec<f32>> {
    // Honour a per-tensor override if the snapshot has one for this K/V weight.
    let quant = effective_quant(quant, weight_name);

    let base = weight_name.strip_suffix(".weight").ok_or_else(|| {
        Error::Loader(format!(
            "'{weight_name}' does not end in .weight — cannot resolve quant siblings"
        ))
    })?;
    let scales_name = format!("{base}.scales");

    let (weight_bytes, weight_shape, _wdtype) =
        read_tensor_bytes(shard_set, weight_map, weight_name)?;
    let (scales_bytes, scales_shape, scales_dtype) =
        read_tensor_bytes(shard_set, weight_map, &scales_name)?;

    // weight rows == scales rows == out_features; scales cols × group_size == in_features.
    let (&rows, &scales_cols) = match (weight_shape.first(), scales_shape.get(1)) {
        (Some(r), Some(sc)) if weight_shape.len() == 2 && scales_shape.len() == 2 => (r, sc),
        _ => {
            return Err(Error::Loader(format!(
                "'{weight_name}': expected rank-2 weight + scales, got weight {weight_shape:?}, \
                 scales {scales_shape:?}"
            )));
        }
    };
    let group_size = quant.group_size as usize;
    // Logical cols (= in_features) derived from the scales shape: one scale per
    // group along the input dim. This is independent of the packed weight layout.
    let cols = scales_cols * group_size;
    if rows == 0 || cols == 0 {
        return Err(Error::Loader(format!(
            "'{weight_name}': degenerate dequant shape rows={rows} cols={cols}"
        )));
    }

    let mode = quant.mode_or_default();
    match mode {
        "affine" => {
            let biases_name = format!("{base}.biases");
            let (biases_bytes, _bshape, biases_dtype) =
                read_tensor_bytes(shard_set, weight_map, &biases_name)?;

            let scales_bf16 = scales_to_bf16_le(&scales_bytes, scales_dtype)?;
            let biases_bf16 = scales_to_bf16_le(&biases_bytes, biases_dtype)?;

            let params = rmlx_quant::affine::AffineParams {
                bits: quant.bits,
                group_size: quant.group_size,
                storage: rmlx_quant::affine::CodeStorage::U32Le,
                rows,
                cols,
            };
            rmlx_quant::affine::dequant_vec(&params, &weight_bytes, &scales_bf16, &biases_bf16)
        }
        "mxfp8" | "mxfp4" | "nvfp4" => {
            // `cols` above used the config `group_size`, which for a valid MX
            // snapshot equals the family's scale granularity (mxfp8/4 → 32,
            // nvfp4 → 16). `mxfp::dequant_vec` length-validates packed/scales
            // against `rows*cols`, so a config/​family mismatch hard-errors here
            // rather than silently mis-ranking.
            let family = match mode {
                "mxfp8" => rmlx_quant::MxFamily::Mxfp8,
                "mxfp4" => rmlx_quant::MxFamily::Mxfp4,
                // nvfp4 snapshots produced by MLX carry the signed-scale bug; match it.
                _ => rmlx_quant::MxFamily::Nvfp4 {
                    compat_mlx_signed_scale: true,
                },
            };
            let params = rmlx_quant::MxParams { family, rows, cols };
            rmlx_quant::mxfp::dequant_vec(&params, &weight_bytes, &scales_bytes)
        }
        other => Err(Error::Loader(format!(
            "kv-calibrate: unsupported quantization mode '{other}' for '{weight_name}' — \
             supported: affine, mxfp8, mxfp4, nvfp4. Dequantize the snapshot to a float \
             format first, or extend dequant_kv_weight()."
        ))),
    }
}

// ── dtype conversion ──────────────────────────────────────────────────────────

/// Convert raw safetensors bytes to a flat `Vec<f32>`.
///
/// Supports F32, BF16, F16. Other dtypes return `Err`.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "safetensors::Dtype has many variants; all non-F32/BF16/F16 forms are unsupported; \
              exhaustively listing all of them would be fragile against upstream Dtype additions"
)]
pub fn bytes_to_f32(
    bytes: &[u8],
    dtype: safetensors::Dtype,
) -> std::result::Result<Vec<f32>, String> {
    use safetensors::Dtype as D;
    match dtype {
        D::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(format!("F32 byte count {} not divisible by 4", bytes.len()));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| {
                    // chunks_exact(4) guarantees c.len() == 4; indexing is safe.
                    #[allow(clippy::indexing_slicing)]
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                })
                .collect())
        }
        D::BF16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(format!(
                    "BF16 byte count {} not divisible by 2",
                    bytes.len()
                ));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| {
                    // chunks_exact(2) guarantees c.len() == 2; indexing is safe.
                    #[allow(clippy::indexing_slicing)]
                    // BF16 = high 16 bits of IEEE 754 f32.
                    let bits = u32::from(u16::from_le_bytes([c[0], c[1]])) << 16;
                    f32::from_bits(bits)
                })
                .collect())
        }
        D::F16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(format!("F16 byte count {} not divisible by 2", bytes.len()));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| {
                    // chunks_exact(2) guarantees c.len() == 2; indexing is safe.
                    #[allow(clippy::indexing_slicing)]
                    f16_to_f32(u16::from_le_bytes([c[0], c[1]]))
                })
                .collect())
        }
        _ => Err(format!(
            "unsupported dtype {dtype:?} for kv-calibrate (expected F32, BF16, or F16)"
        )),
    }
}

/// Convert a raw IEEE 754 half-precision float16 bit pattern to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = u32::from(bits & 0x3FF);

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal.
        let mut m = mant;
        let mut e: i32 = 1 - 15 + 127;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        #[allow(clippy::cast_sign_loss)]
        let f32_bits = sign | ((e as u32) << 23) | (m << 13);
        return f32::from_bits(f32_bits);
    }
    if exp == 0x1F {
        let f32_bits = sign | (0xFF << 23) | (mant << 13);
        return f32::from_bits(f32_bits);
    }
    let f32_bits = sign | ((u32::from(exp) + (127 - 15)) << 23) | (mant << 13);
    f32::from_bits(f32_bits)
}

// ── layer key extraction ──────────────────────────────────────────────────────

/// Derive the layer-level attention prefix from the full weight pattern at a given index.
///
/// e.g. `"model.layers.{}.self_attn.k_proj.weight"` at index 3
///   → `"model.layers.3.self_attn"`
pub fn layer_key_from_pattern(k_pattern: &str, layer_idx: u32) -> String {
    let filled = k_pattern.replace("{}", &layer_idx.to_string());
    for suffix in &[".self_attn.", ".attn.", ".attention."] {
        if let Some(pos) = filled.rfind(suffix) {
            // Keep everything up to and excluding the trailing dot.
            let attn_prefix = &filled[..pos + suffix.len() - 1];
            return attn_prefix.to_string();
        }
    }
    // Fallback: strip last two dot-components (projection name + "weight").
    let mut parts: Vec<&str> = filled.split('.').collect();
    if parts.len() >= 2 {
        parts.truncate(parts.len() - 2);
    }
    parts.join(".")
}

// ── config helpers ────────────────────────────────────────────────────────────

fn cfg_num_layers(cfg: &crate::ModelConfig) -> Option<u32> {
    cfg.text_config
        .as_ref()
        .and_then(|tc| tc.num_hidden_layers)
        .or_else(|| {
            cfg.extras
                .get("num_hidden_layers")
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
        })
}

fn cfg_num_kv_heads(cfg: &crate::ModelConfig) -> Option<u32> {
    cfg.text_config
        .as_ref()
        .and_then(|tc| tc.num_key_value_heads)
        .or_else(|| {
            cfg.extras
                .get("num_key_value_heads")
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
        })
        .or_else(|| {
            cfg.text_config
                .as_ref()
                .and_then(|tc| tc.num_attention_heads)
                .or_else(|| {
                    cfg.extras
                        .get("num_attention_heads")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|n| u32::try_from(n).ok())
                })
        })
}

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod calibration_tests;
