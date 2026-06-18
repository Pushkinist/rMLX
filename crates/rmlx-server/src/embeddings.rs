//! `POST /v1/embeddings` — jina-embeddings-v4 text embeddings.
//!
//! Single-vector (default) and multi-vector (`return_multivector: true`) text
//! embeddings with runtime LoRA task selection (`retrieval` | `text-matching`
//! | `code`) and matryoshka truncation (`dimensions`).
//!
//! The request/response shape mirrors oMLX's `/v1/embeddings`
//! (`omlx/api/embedding_models.py`) and the OpenAI embeddings API, plus jina's
//! de-facto `return_multivector` extension.
//!
//! ## Model placement
//!
//! `JinaEmbeddingsV4Model` is an encoder, NOT a causal LM — it has no
//! `Architecture` enum variant and no `Generator` impl. It is therefore kept
//! out of `AppState::slots` (every slot is a `dyn Generator`). Instead a
//! single lazily-loaded [`JinaEmbedModel`] lives in `AppState::embed_slot`
//! (Metal is single-process, so one resident embedding model is enough).
//!
//! ## Tokenization (jina convention)
//!
//! Per `modeling_jina_embeddings_v4.py:91-119,373-399,442`:
//! - prepend the task prefix `f"{Query|Passage}: {text}"` — `prompt_name`
//!   `query`→`Query`, `passage`→`Passage`; **`text-matching` always uses
//!   `Query`** regardless of `prompt_name` (ref lines 387-389).
//! - NO BOS, NO chat template (`tokenizer_config.json` has
//!   `add_bos_token=false`, `bos_token=null`); `tokenizer_io::encode`
//!   (`add_special_tokens=false`) is exactly correct.
//!
//! ## GPU serialisation
//!
//! Embedding compute runs in `spawn_blocking` while holding the process-wide
//! `gpu_gate` mutex — the same single-Metal-context defense the generator
//! path uses. Per-request `apply_task` runs inside that critical section so
//! the live LoRA matches the request's task.

#![allow(
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::too_many_lines
)]
use std::path::Path;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use parking_lot::RwLock as PLRwLock;
use rmlx_models::jina_v4::{self, JinaV4, JinaV4Task};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::logged_json::LoggedJson;
use crate::openai::{ApiErrorCategory, AppState};

// ── Request ───────────────────────────────────────────────────────────────────

/// `POST /v1/embeddings` request body (OpenAI + oMLX + jina shape).
#[allow(
    clippy::exhaustive_structs,
    reason = "wire DTO — struct-literal construction required for deserialization; adding a field requires a serde default so existing callers continue to work"
)]
#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    /// Registry model id (must resolve to a `JinaEmbeddingsV4Model` entry).
    pub model: String,
    /// One string or a list of strings to embed.
    pub input: EmbeddingInput,
    /// `"float"` (default) or `"base64"`.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Matryoshka truncation dim — must be one of the model's
    /// `matryoshka_dims` ({128,256,512,1024,2048} for jina-v4).
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// LoRA task: `retrieval` (default) | `text-matching` | `code`.
    #[serde(default)]
    pub task: Option<String>,
    /// jina text prefix selector: `query` (default) | `passage`.
    #[serde(default)]
    pub prompt_name: Option<String>,
    /// When true, return per-token multi-vector embeddings (`[[f32;128];seq]`)
    /// instead of a single pooled vector.
    #[serde(default)]
    pub return_multivector: bool,
}

/// `input` accepts a single string, a list of strings (OpenAI text shape),
/// a single image object `{"image": "<data-URI|base64|path>"}`, or a list of
/// such image objects (oMLX image shape). Untagged: text variants are tried
/// first so the text wire contract is byte-identical (an image
/// object never deserializes as a string).
#[allow(
    clippy::exhaustive_enums,
    reason = "wire DTO — four input shapes (Single/Many/OneImage/ManyImages) are the complete embedding-input wire contract; adding a shape requires a protocol version bump"
)]
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// A single text string to embed.
    Single(String),
    /// Multiple text strings to embed in one batch.
    Many(Vec<String>),
    /// A single image item to embed (oMLX image shape).
    OneImage(ImageItem),
    /// Multiple image items to embed in one batch.
    ManyImages(Vec<ImageItem>),
}

/// A single image input item: `{"image": "<data-URI | raw base64 | file
/// path>"}` (oMLX convention; the `items:[{image:...}]` form maps to
/// [`EmbeddingInput::ManyImages`]).
#[allow(
    clippy::exhaustive_structs,
    reason = "wire DTO — single-field image item; the 'image' key is the complete oMLX image-input wire contract"
)]
#[derive(Debug, Deserialize)]
pub struct ImageItem {
    /// Image source: data URI, raw base64 string, or file path.
    pub image: String,
}

/// Normalised request payload: either a list of text strings or a list of
/// raw image source strings (kept disjoint — jina embeds text or image, not
/// a mix, mirroring `encode_text` / `encode_image`).
enum NormInput {
    Texts(Vec<String>),
    Images(Vec<String>),
}

impl EmbeddingInput {
    fn normalize(self) -> NormInput {
        match self {
            EmbeddingInput::Single(s) => NormInput::Texts(vec![s]),
            EmbeddingInput::Many(v) => NormInput::Texts(v),
            EmbeddingInput::OneImage(i) => NormInput::Images(vec![i.image]),
            EmbeddingInput::ManyImages(v) => {
                NormInput::Images(v.into_iter().map(|i| i.image).collect())
            }
        }
    }
}

// ── Response ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct EmbeddingsResponse {
    object: &'static str, // "list"
    data: Vec<EmbeddingData>,
    model: String,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct EmbeddingData {
    object: &'static str, // "embedding"
    index: usize,
    /// `[f32]` (single-vector float), `[[f32]]` (multi-vector float), or a
    /// base64 string (single-vector, `encoding_format=base64`).
    embedding: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

// ── Lazily-loaded jina embedding model ────────────────────────────────────────

/// A resident jina-embeddings-v4 model plus the registry id it was loaded
/// for. Kept in `AppState::embed_slot` (not `slots`).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed model-slot struct — two fields are the complete jina embed model contract; adding a field requires updating all JinaEmbedModel construction sites in the serve path"
)]
pub struct JinaEmbedModel {
    /// Registry model id this instance was loaded for.
    pub id: String,
    /// The loaded encoder (mutable: `apply_task` swaps the live LoRA).
    pub model: JinaV4,
}

impl std::fmt::Debug for JinaEmbedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JinaEmbedModel")
            .field("id", &self.id)
            .field("active_task", &self.model.active_task())
            .finish()
    }
}

/// Architecture string that routes a registry entry to the embedding path.
pub const JINA_V4_ARCH: &str = "JinaEmbeddingsV4Model";

// ── Error helper ──────────────────────────────────────────────────────────────

fn err(
    state: &AppState,
    cat: ApiErrorCategory,
    status: StatusCode,
    ty: &str,
    msg: &str,
) -> Response {
    state.error_counts.increment(cat);
    let body = json!({ "error": { "message": msg, "type": ty } });
    (status, Json(body)).into_response()
}

// ── Handler ───────────────────────────────────────────────────────────────────

/// `POST /v1/embeddings` handler.
pub(crate) async fn embeddings(
    State(state): State<AppState>,
    LoggedJson(req): LoggedJson<EmbeddingsRequest>,
) -> Response {
    // ── Validate request fields (400) ────────────────────────────────────────
    let encoding_format = req.encoding_format.as_deref().unwrap_or("float");
    if encoding_format != "float" && encoding_format != "base64" {
        return err(
            &state,
            ApiErrorCategory::BadRequest,
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "encoding_format must be 'float' or 'base64'",
        );
    }
    if encoding_format == "base64" && req.return_multivector {
        return err(
            &state,
            ApiErrorCategory::BadRequest,
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "encoding_format 'base64' is not supported with return_multivector=true",
        );
    }

    let task = match req.task.as_deref() {
        None => JinaV4Task::DEFAULT,
        Some(t) => match JinaV4Task::from_name(t) {
            Ok(t) => t,
            Err(_) => {
                return err(
                    &state,
                    ApiErrorCategory::BadRequest,
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "task must be one of 'retrieval', 'text-matching', 'code'",
                );
            }
        },
    };

    // jina prefix: prompt_name query|passage; text-matching forces "Query".
    let prompt_name = req.prompt_name.as_deref().unwrap_or("query");
    let prefix_word = match prompt_name {
        "query" => "Query",
        "passage" => "Passage",
        _ => {
            return err(
                &state,
                ApiErrorCategory::BadRequest,
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "prompt_name must be 'query' or 'passage'",
            );
        }
    };
    let prefix_word = if task == JinaV4Task::TextMatching {
        "Query"
    } else {
        prefix_word
    };

    let norm_input = req.input.normalize();
    let is_empty = match &norm_input {
        NormInput::Texts(v) => v.is_empty(),
        NormInput::Images(v) => v.is_empty(),
    };
    if is_empty {
        return err(
            &state,
            ApiErrorCategory::BadRequest,
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "input must contain at least one text string or image",
        );
    }

    // ── Resolve + load the embedding model ───────────────────────────────────
    let entry = match state.registry.get(&req.model) {
        None => {
            return err(
                &state,
                ApiErrorCategory::NotFound,
                StatusCode::NOT_FOUND,
                "not_found_error",
                &format!("model '{}' not found in registry", req.model),
            );
        }
        Some(e) => e,
    };
    if entry.arch != JINA_V4_ARCH {
        return err(
            &state,
            ApiErrorCategory::BadRequest,
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!(
                "model '{}' (arch '{}') is not an embedding model; \
                 /v1/embeddings requires a {JINA_V4_ARCH} snapshot",
                req.model, entry.arch
            ),
        );
    }
    let abs_path = entry.abs_path.clone();
    let model_id = req.model.clone();

    // dimensions must validate against the model's matryoshka set; defer the
    // exact check to pooling::validate_truncate_dim (run under the gate).
    let truncate_dim = req.dimensions;
    let return_multivector = req.return_multivector;

    // ── Tokenize (jina: prefix, no BOS, no chat template) ────────────────────
    let tokenizer = match &entry.tokenizer {
        Some(tk) => Arc::clone(tk),
        None => {
            return err(
                &state,
                ApiErrorCategory::Upstream,
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                &format!("model '{}' has no tokenizer.json", req.model),
            );
        }
    };

    // For the image path, tokenize the fixed jina image prompt once (no
    // prefix, no BOS, no chat template — exactly the text-path encoder); the
    // single `<|image_pad|>` placeholder is expanded per image inside the
    // model. For text, prefix + tokenize each input as before (byte-identical
    let want_base64 = encoding_format == "base64";
    let embed_slot = Arc::clone(&state.embed_slot);
    let gpu_gate = Arc::clone(&state.gpu_gate);
    // Clone the shared multimodal cache into the blocking compute task.
    let mm_cache = Arc::clone(&state.mm_cache);

    let (compute, prompt_tokens): (ComputeKind, usize) = match norm_input {
        NormInput::Texts(inputs) => {
            let prefixed: Vec<String> = inputs
                .iter()
                .map(|t| format!("{prefix_word}: {t}"))
                .collect();
            let mut token_ids: Vec<Vec<i64>> = Vec::with_capacity(prefixed.len());
            for text in &prefixed {
                match crate::tokenizer_io::encode(&tokenizer, text) {
                    Ok(ids) => token_ids.push(ids.into_iter().map(i64::from).collect()),
                    Err(e) => {
                        return err(
                            &state,
                            ApiErrorCategory::Upstream,
                            StatusCode::SERVICE_UNAVAILABLE,
                            "service_unavailable",
                            &format!("tokenizer encode failed: {e}"),
                        );
                    }
                }
            }
            let n = token_ids.iter().map(Vec::len).sum();
            (ComputeKind::Text { token_ids }, n)
        }
        NormInput::Images(sources) => {
            // Decode each image source (data-URI | raw base64 | file path)
            // and preprocess to pixel_values (image front-end).
            let pcfg = match jina_v4::ImagePreprocessConfig::from_model_dir(&abs_path) {
                Ok(c) => c,
                Err(e) => {
                    return err(
                        &state,
                        ApiErrorCategory::Internal,
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "internal_error",
                        &format!("preprocessor_config load failed: {e}"),
                    );
                }
            };
            let mut pvs: Vec<jina_v4::PixelValues> = Vec::with_capacity(sources.len());
            for src in &sources {
                let bytes =
                    match crate::image_io::load_image(src, crate::image_io::DEFAULT_HTTP_TIMEOUT) {
                        Ok(b) => b,
                        Err(m) => {
                            return err(
                                &state,
                                ApiErrorCategory::BadRequest,
                                StatusCode::BAD_REQUEST,
                                "invalid_request_error",
                                &m,
                            );
                        }
                    };
                match jina_v4::preprocess_image_bytes(&bytes, &pcfg) {
                    Ok(pv) => pvs.push(pv),
                    Err(e) => {
                        return err(
                            &state,
                            ApiErrorCategory::BadRequest,
                            StatusCode::BAD_REQUEST,
                            "invalid_request_error",
                            &format!("image preprocess failed: {e}"),
                        );
                    }
                }
            }
            // Tokenize the fixed image prompt once (single image_pad).
            let prompt_ids: Vec<i64> =
                match crate::tokenizer_io::encode(&tokenizer, jina_v4::image_prompt()) {
                    Ok(ids) => ids.into_iter().map(i64::from).collect(),
                    Err(e) => {
                        return err(
                            &state,
                            ApiErrorCategory::Upstream,
                            StatusCode::SERVICE_UNAVAILABLE,
                            "service_unavailable",
                            &format!("tokenizer encode failed: {e}"),
                        );
                    }
                };
            // Usage proxy: prompt token count per image (placeholder expands
            // to grid tokens at compute time; report the prompt length as a
            // stable, cheap usage metric, same spirit as the text path).
            let n = prompt_ids.len() * pvs.len();
            (
                ComputeKind::Image {
                    prompt_ids,
                    pixel_values: pvs,
                },
                n,
            )
        }
    };

    // ── Compute under the GPU gate (single Metal context) ────────────────────
    let result = tokio::task::spawn_blocking(move || {
        let cache_ref: Option<&rmlx_models::multimodal_cache::MultimodalCache> =
            if mm_cache.is_disabled() {
                None
            } else {
                Some(mm_cache.as_ref())
            };
        compute_embeddings(
            &embed_slot,
            &gpu_gate,
            &model_id,
            &abs_path,
            task,
            compute,
            truncate_dim,
            return_multivector,
            want_base64,
            cache_ref,
        )
    })
    .await;

    let data = match result {
        Err(join_err) => {
            tracing::error!(error = %join_err, "embeddings: compute task panicked");
            return err(
                &state,
                ApiErrorCategory::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "embedding compute task panicked",
            );
        }
        Ok(Err(EmbedError::BadDimensions(m))) => {
            return err(
                &state,
                ApiErrorCategory::BadRequest,
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &m,
            );
        }
        Ok(Err(EmbedError::Load(m))) => {
            return err(
                &state,
                ApiErrorCategory::Upstream,
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                &m,
            );
        }
        Ok(Err(EmbedError::Compute(m))) => {
            return err(
                &state,
                ApiErrorCategory::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &m,
            );
        }
        Ok(Ok(d)) => d,
    };

    let resp = EmbeddingsResponse {
        object: "list",
        data,
        model: req.model,
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    };
    (StatusCode::OK, Json(resp)).into_response()
}

// ── Blocking compute ──────────────────────────────────────────────────────────

enum EmbedError {
    /// 400 — invalid `dimensions` (not in matryoshka set).
    BadDimensions(String),
    /// 503 — model load failure.
    Load(String),
    /// 500 — forward/pooling failure.
    Compute(String),
}

/// What `compute_embeddings` runs: a list of pre-tokenized text sequences
/// (text path, unchanged) or a list of preprocessed images plus the
/// single shared image-prompt token ids (image path).
enum ComputeKind {
    Text {
        token_ids: Vec<Vec<i64>>,
    },
    Image {
        prompt_ids: Vec<i64>,
        pixel_values: Vec<jina_v4::PixelValues>,
    },
}

#[allow(clippy::too_many_arguments)]
fn compute_embeddings(
    embed_slot: &PLRwLock<Option<JinaEmbedModel>>,
    gpu_gate: &parking_lot::Mutex<()>,
    model_id: &str,
    abs_path: &Path,
    task: JinaV4Task,
    compute: ComputeKind,
    truncate_dim: Option<usize>,
    return_multivector: bool,
    want_base64: bool,
    mm_cache: Option<&rmlx_models::multimodal_cache::MultimodalCache>,
) -> Result<Vec<EmbeddingData>, EmbedError> {
    // Single Metal context: serialise the whole compute (load + forward).
    let _gpu = gpu_gate.lock();

    // Lazily load (or reuse) the resident embedding model.
    {
        let needs_load = {
            let slot = embed_slot.read();
            slot.as_ref().is_none_or(|m| m.id != model_id)
        };
        if needs_load {
            tracing::info!(model_id, path = %abs_path.display(), "embeddings: loading jina-v4");
            let model = jina_v4::load_from_path(abs_path)
                .map_err(|e| EmbedError::Load(format!("failed to load '{model_id}': {e}")))?;
            *embed_slot.write() = Some(JinaEmbedModel {
                id: model_id.to_owned(),
                model,
            });
        }
    }

    let mut slot = embed_slot.write();
    let Some(holder) = slot.as_mut() else {
        // Structural invariant: embed_slot was populated in the block above
        // under the same gpu_gate. Reaching here means an unexpected state.
        tracing::error!(
            model_id,
            lock_name = "embed_slot",
            "embeddings: embed_slot is None after populate block — internal error"
        );
        return Err(EmbedError::Load(
            "internal error: embed_slot missing after load".to_owned(),
        ));
    };

    // Switch the live LoRA to the requested task (clean replace; idempotent).
    if holder.model.active_task() != task {
        holder
            .model
            .apply_task(task)
            .map_err(|e| EmbedError::Compute(format!("apply_task failed: {e}")))?;
    }

    let device = rmlx_mlx::Device::Gpu;

    // Registering a thread-local GPU stream + CommandEncoder once per thread entry point.
    // tokio blocking-pool threads start with no GPU stream context; MLX's array
    // materialisation then fails with "There is no Stream(gpu, 0) in current thread".
    // Mirrors the pattern used at the text and image generate entry points.
    if device == rmlx_mlx::Device::Gpu {
        rmlx_mlx::ensure_gpu_default_stream();
    }

    // Build (single_vec | multi_vec) per item with a uniform serializer so
    // the response shape is identical for text and image.
    let mut data = Vec::new();
    let push_single = |data: &mut Vec<EmbeddingData>, index: usize, v: Vec<f32>| {
        let embedding = if want_base64 {
            serde_json::Value::String(f32_vec_to_base64(&v))
        } else {
            json!(v)
        };
        data.push(EmbeddingData {
            object: "embedding",
            index,
            embedding,
        });
    };
    let push_multi = |data: &mut Vec<EmbeddingData>, index: usize, mv: Vec<Vec<f32>>| {
        let rows: Vec<serde_json::Value> = mv.into_iter().map(|row| json!(row)).collect();
        data.push(EmbeddingData {
            object: "embedding",
            index,
            embedding: serde_json::Value::Array(rows),
        });
    };

    match compute {
        ComputeKind::Text { token_ids } => {
            data.reserve(token_ids.len());
            for (index, ids) in token_ids.iter().enumerate() {
                if return_multivector {
                    let mv = holder
                        .model
                        .embed_multi(ids, device)
                        .map_err(|e| EmbedError::Compute(format!("embed_multi failed: {e}")))?;
                    push_multi(&mut data, index, mv);
                } else {
                    let v = holder
                        .model
                        .embed_single(ids, device, truncate_dim)
                        .map_err(classify_single_err)?;
                    push_single(&mut data, index, v);
                }
            }
        }
        ComputeKind::Image {
            prompt_ids,
            pixel_values,
        } => {
            // Scope every mm-cache entry to this loaded model so a shared
            // (multi-model) cache never serves another model's vision output
            // for the same image.
            let model_sig = rmlx_models::multimodal_cache::model_sig(model_id);
            data.reserve(pixel_values.len());
            for (index, pv) in pixel_values.iter().enumerate() {
                if return_multivector {
                    let mv = holder
                        .model
                        .embed_image_multi(&prompt_ids, pv, device, mm_cache, model_sig)
                        .map_err(|e| {
                            EmbedError::Compute(format!("embed_image_multi failed: {e}"))
                        })?;
                    push_multi(&mut data, index, mv);
                } else {
                    let v = holder
                        .model
                        .embed_image_single(
                            &prompt_ids,
                            pv,
                            device,
                            truncate_dim,
                            mm_cache,
                            model_sig,
                        )
                        .map_err(classify_single_err)?;
                    push_single(&mut data, index, v);
                }
            }
        }
    }
    Ok(data)
}

/// `embed_single` rejects an out-of-set `truncate_dim` with a `Config` error
/// containing `"invalid truncate_dim"` — surface that as 400, everything else
/// as 500.
fn classify_single_err(e: rmlx_core::error::Error) -> EmbedError {
    let m = e.to_string();
    if m.contains("invalid truncate_dim") || m.contains("exceeds embedding dim") {
        EmbedError::BadDimensions(m)
    } else {
        EmbedError::Compute(format!("embed_single failed: {m}"))
    }
}

/// Base64-encode a little-endian `f32` vector (OpenAI `encoding_format`
/// `base64` convention: raw IEEE-754 LE bytes, standard base64).
fn f32_vec_to_base64(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    base64_encode(&bytes)
}

/// Minimal standard-alphabet base64 (no padding-free, RFC 4648). Avoids
/// pulling a new crate for a single small encode.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
fn base64_encode(input: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(*chunk.get(1).unwrap_or(&0));
        let b2 = u32::from(*chunk.get(2).unwrap_or(&0));
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(T[(n >> 6) as usize & 63] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(T[n as usize & 63] as char);
        } else {
            out.push('=');
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "embeddings_tests.rs"]
mod embeddings_tests;
