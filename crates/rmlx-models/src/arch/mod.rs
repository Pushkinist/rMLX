//! Architecture dispatch: load any supported model snapshot by reading
//! config.architectures[0] and routing to the right backend.
//!
//! Wired:
//! Gemma4ForConditionalGeneration -> Architecture::Gemma4
//! Gemma3ForConditionalGeneration -> Architecture::Gemma3
//! Qwen2ForCausalLM -> Architecture::Qwen2
//! Qwen3ForCausalLM -> Architecture::Qwen3
//! LagunaForCausalLM -> Architecture::Laguna
//! Qwen3_5MoeForConditionalGeneration -> Architecture::Qwen3_5Moe
//!
//! All other architectures return Error::Model with a "not yet supported" message.

#![allow(
    clippy::cognitive_complexity,
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::used_underscore_binding
)]

pub(crate) mod loader;
pub(crate) mod phases;
pub(crate) mod registry;

pub use loader::{load_model, run_smoke_probe, smoke_prompt_ids, LoadOpts, SMOKE_PROMPT};
pub use phases::{read_load_phases, LoadPhases};
pub use registry::{is_arch_supported, KNOWN_ARCHS};

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::bitnet::BitNetText;
use crate::decode_loop::SmokeVerdict;
use crate::gemma3::Gemma3Text;
use crate::gemma4::Gemma4Text;
use crate::laguna::LagunaText;
use crate::qwen2::Qwen2Text;
use crate::qwen3::Qwen3Text;
use crate::qwen3_5_moe::Qwen3_5MoeText;
use crate::qwen3_vl_moe::model::Qwen3VlMoe;

// ---------------------------------------------------------------------------
// Architecture enum
// ---------------------------------------------------------------------------

/// All architectures rMLX can dispatch.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum -- architecture registry; adding a variant requires updating load_model, all match arms, and KNOWN_ARCHS"
)]
pub enum Architecture {
    /// `Gemma4ForConditionalGeneration` -- mxfp8, optional MoE, vision + audio.
    Gemma4(Gemma4Text),
    /// `Gemma3ForConditionalGeneration` -- affine-int8, SigLIP vision tower.
    Gemma3(Gemma3Text),
    /// `Qwen2ForCausalLM` -- dense, affine-quantized.
    Qwen2(Qwen2Text),
    /// `Qwen3ForCausalLM` -- dense, affine-quantized, per-head q/k RMSNorm.
    Qwen3(Qwen3Text),
    /// `LagunaForCausalLM` -- sparse MoE, mxfp8, per-tensor quant overrides.
    Laguna(LagunaText),
    /// `Qwen3_5MoeForConditionalGeneration` -- hybrid GatedDeltaNet + full-attn, sparse MoE.
    Qwen3_5Moe(Qwen3_5MoeText),
    /// `Qwen3VLMoeForConditionalGeneration` -- vision-language sparse MoE.
    Qwen3VlMoe(Qwen3VlMoe),
    /// `BitNetForCausalLM` -- ternary-weight (b1.58), GQA, Relu2, sub-norms.
    BitNet(BitNetText),
}

impl std::fmt::Debug for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Architecture::Gemma4(m) => write!(
                f,
                "Architecture::Gemma4(layers={}, hidden={}, vocab={})",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size
            ),
            Architecture::Gemma3(m) => write!(
                f,
                "Architecture::Gemma3(layers={}, hidden={}, vocab={})",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size
            ),
            Architecture::Qwen2(m) => write!(
                f,
                "Architecture::Qwen2(layers={}, hidden={}, vocab={})",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size
            ),
            Architecture::Qwen3(m) => write!(
                f,
                "Architecture::Qwen3(layers={}, hidden={}, vocab={})",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size
            ),
            Architecture::Laguna(m) => write!(
                f,
                "Architecture::Laguna(layers={}, hidden={}, vocab={}, experts={})",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size, m.cfg.num_experts
            ),
            Architecture::Qwen3_5Moe(m) => write!(
                f,
                "Architecture::Qwen3_5Moe(layers={}, hidden={}, vocab={}, experts={})",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size, m.cfg.num_experts
            ),
            Architecture::Qwen3VlMoe(m) => write!(
                f,
                "Architecture::Qwen3VlMoe(layers={}, hidden={}, vocab={}, experts={})",
                m.text.cfg.num_hidden_layers,
                m.text.cfg.hidden_size,
                m.text.cfg.vocab_size,
                m.text.cfg.num_experts
            ),
            Architecture::BitNet(m) => write!(
                f,
                "Architecture::BitNet(layers={}, hidden={}, vocab={})",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size
            ),
        }
    }
}

impl Architecture {
    /// Run a full-sequence forward pass, return logits for the last position.
    ///
    /// Shape: `[1, 1, vocab_size]`.
    pub fn forward_seq(&self, ids: &[u32], device: Device) -> Result<Array> {
        match self {
            Architecture::Gemma4(m) => m.forward_seq(ids, device),
            Architecture::Gemma3(m) => m.forward_seq(ids, device),
            Architecture::Qwen2(m) => m.forward_seq(ids, device),
            Architecture::Qwen3(m) => m.forward_seq(ids, device),
            Architecture::Laguna(m) => m.forward_seq(ids, device),
            Architecture::Qwen3_5Moe(m) => m.forward_seq(ids, device),
            Architecture::Qwen3VlMoe(m) => m.text.forward_seq(ids, device),
            Architecture::BitNet(m) => m.forward_seq(ids, device),
        }
    }

    /// Run a full-sequence forward pass, return logits for the last `k` positions.
    ///
    /// Speculative-decoding scaffold. Returns shape
    /// `[1, k, vocab_size]`. Currently wires Gemma4 only; other architectures
    /// return `Error::Model` until they are wired.
    #[allow(
        clippy::unreachable,
        reason = "Architecture::Gemma4 arm is handled by the outer match; \
                  the inner arm is structurally unreachable -- required to exhaust the enum \
                  for the arch-name string table"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn forward_seq_last_k(&self, ids: &[u32], k: usize, device: Device) -> Result<Array> {
        match self {
            Architecture::Gemma4(m) => m.forward_seq_last_k(ids, k, device),
            other => {
                let arch = match other {
                    Architecture::Gemma4(_) => unreachable!(),
                    Architecture::Gemma3(_) => "Gemma3",
                    Architecture::Qwen2(_) => "Qwen2",
                    Architecture::Qwen3(_) => "Qwen3",
                    Architecture::Laguna(_) => "Laguna",
                    Architecture::Qwen3_5Moe(_) => "Qwen3_5Moe",
                    Architecture::Qwen3VlMoe(_) => "Qwen3VlMoe",
                    Architecture::BitNet(_) => "BitNet",
                };
                Err(Error::Model(format!(
                    "forward_seq_last_k not yet wired for {arch}"
                )))
            }
        }
    }

    /// Cache-using last-K forward.
    ///
    /// Like `forward_seq_last_k` but reads + writes the provided per-layer
    /// `kv_caches`. Used by speculative decoding to feed K+1 verifier tokens
    /// in one forward call against a persistent cache. Returns shape
    /// `[1, k, vocab_size]`.
    ///
    /// `lin_caches` carries the per-layer `LinearAttnCache` for architectures
    /// that have GatedDeltaNet (linear-attention) layers alongside FullAttention
    /// layers (e.g. Qwen3.5MoE). Pass `None` for Gemma4 / Gemma3 / Qwen2 /
    /// Qwen3 -- they have no GDN layers and ignore this parameter.
    ///
    /// Gemma4 (FullAttention-only) and Qwen3.5MoE (FullAttention + GDN
    /// hybrid, ) are both wired. Qwen3.5MoE advances its recurrent
    /// `lin_caches` alongside `kv_caches` in this single forward. Other
    /// architectures return `Error::Model`.
    #[allow(
        clippy::unreachable,
        reason = "Architecture::Gemma4 and Qwen3_5Moe arms are handled by the outer match; \
                  the inner arms are structurally unreachable -- required to exhaust the enum \
                  for the arch-name string table"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn forward_seq_last_k_with_cache(
        &self,
        ids: &[u32],
        k: usize,
        kv_caches: &mut [rmlx_kv_quant::KvCache],
        lin_caches: Option<&mut [rmlx_kv_quant::LinearAttnCache]>,
        device: Device,
    ) -> Result<Array> {
        match self {
            Architecture::Gemma4(m) => {
                // Gemma4 has no GDN layers; lin_caches is always None here.
                let _ = lin_caches;
                m.forward_seq_last_k_with_cache(ids, k, kv_caches, device)
            }
            Architecture::Qwen3_5Moe(m) => {
                // Qwen3.5MoE is a hybrid FullAttention + GatedDeltaNet stack.
                // The lin_caches bundle (GDN recurrent state) is advanced
                // alongside kv_caches (FA state) in this single forward.
                m.forward_seq_last_k_with_cache(ids, k, kv_caches, lin_caches, device)
            }
            other => {
                let arch = match other {
                    Architecture::Gemma4(_) | Architecture::Qwen3_5Moe(_) => unreachable!(),
                    Architecture::Gemma3(_) => "Gemma3",
                    Architecture::Qwen2(_) => "Qwen2",
                    Architecture::Qwen3(_) => "Qwen3",
                    Architecture::Laguna(_) => "Laguna",
                    Architecture::Qwen3VlMoe(_) => "Qwen3VlMoe",
                    Architecture::BitNet(_) => "BitNet",
                };
                Err(Error::Model(format!(
                    "forward_seq_last_k_with_cache not yet wired for {arch}"
                )))
            }
        }
    }

    /// Cache-using forward returning the **pre-final-norm** hidden states at
    /// the last `k` positions (-- MTP conditioning signal).
    ///
    /// MTP drafters condition on the verifier's penultimate hidden state -- the
    /// decoder-trunk output *before* the final RMSNorm and LM head. This routes
    /// to `Gemma4Text::forward_hidden_states`; all other architectures return
    /// `Error::Model` (only Gemma4 is wired for ). Returns `[1, k, hidden]`.
    ///
    /// `lin_caches` is reserved for hybrid (GDN) archs; Gemma4 ignores it.
    #[allow(
        clippy::unreachable,
        reason = "Architecture::Gemma4 arm is handled by the outer match; \
                  the inner arm is structurally unreachable -- required to exhaust the enum \
                  for the arch-name string table"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn forward_hidden_states(
        &self,
        ids: &[u32],
        k: usize,
        kv_caches: &mut [rmlx_kv_quant::KvCache],
        lin_caches: Option<&mut [rmlx_kv_quant::LinearAttnCache]>,
        device: Device,
    ) -> Result<Array> {
        match self {
            Architecture::Gemma4(m) => {
                let _ = lin_caches;
                m.forward_hidden_states(ids, k, Some(kv_caches), device)
            }
            other => {
                let arch = match other {
                    Architecture::Gemma4(_) => unreachable!(),
                    Architecture::Gemma3(_) => "Gemma3",
                    Architecture::Qwen2(_) => "Qwen2",
                    Architecture::Qwen3(_) => "Qwen3",
                    Architecture::Laguna(_) => "Laguna",
                    Architecture::Qwen3_5Moe(_) => "Qwen3_5Moe",
                    Architecture::Qwen3VlMoe(_) => "Qwen3VlMoe",
                    Architecture::BitNet(_) => "BitNet",
                };
                Err(Error::Model(format!(
                    "forward_hidden_states not yet wired for {arch} (Gemma4 only)"
                )))
            }
        }
    }

    /// Like [`forward_hidden_states`] but also returns the verifier's
    /// per-layer-type shared K/V for the Gemma4-assistant MTP drafter.
    /// Routes to `Gemma4Text::forward_hidden_states_shared_kv`; Gemma4 only.
    /// Returns `(hidden[1,k,H], sliding_kv, full_kv, kv_offset)`.
    #[allow(clippy::type_complexity)]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn forward_hidden_states_shared_kv(
        &self,
        ids: &[u32],
        k: usize,
        kv_caches: &mut [rmlx_kv_quant::KvCache],
        device: Device,
    ) -> Result<(Array, (Array, Array), (Array, Array), i32)> {
        match self {
            Architecture::Gemma4(m) => m.forward_hidden_states_shared_kv(ids, k, kv_caches, device),
            _ => Err(Error::Model(
                "forward_hidden_states_shared_kv: Gemma4 only (assistant MTP)".into(),
            )),
        }
    }

    /// Target-scaled single-token input embedding. Gemma4
    /// only. Returns `[1, 1, hidden]` at the target's `embed_scale =
    /// sqrt(hidden)` (mlx-vlm `bind()` semantics -- see `Gemma4Text`).
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn embed_token_raw(&self, tok: u32, device: Device) -> Result<Array> {
        match self {
            Architecture::Gemma4(m) => m.embed_token_raw(tok, device),
            _ => Err(Error::Model(
                "embed_token_raw: Gemma4 only (assistant MTP)".into(),
            )),
        }
    }

    /// Multi-layer hidden capture for the DFlash drafter.
    ///
    /// Forwards `ids` through the verifier reading + writing `kv_caches` (and
    /// `lin_caches` for GDN hybrids), capturing the residual-stream output of
    /// each layer in `capture_layer_ids` and concatenating them along the
    /// feature axis. Returns `[1, k, len(capture_layer_ids) * hidden]` at the
    /// last `k` positions. Routes to the Qwen3.5MoE verifier; other archs
    /// return `Error::Model`.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn forward_hidden_states_multi(
        &self,
        ids: &[u32],
        k: usize,
        capture_layer_ids: &[usize],
        kv_caches: &mut [rmlx_kv_quant::KvCache],
        lin_caches: Option<&mut [rmlx_kv_quant::LinearAttnCache]>,
        device: Device,
    ) -> Result<Array> {
        match self {
            Architecture::Qwen3_5Moe(m) => m.forward_hidden_states_multi(
                ids,
                k,
                capture_layer_ids,
                kv_caches,
                lin_caches,
                device,
            ),
            _ => Err(Error::Model(
                "forward_hidden_states_multi: Qwen3.5MoE only (DFlash)".into(),
            )),
        }
    }

    /// Combined verify forward for the DFlash round-loop: one cached
    /// pass returning `(last-k logits, concatenated multi-layer hidden)`.
    /// Qwen3.5MoE only.
    #[allow(clippy::type_complexity)]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn forward_verify_capture(
        &self,
        ids: &[u32],
        k: usize,
        capture_layer_ids: &[usize],
        kv_caches: &mut [rmlx_kv_quant::KvCache],
        lin_caches: Option<&mut [rmlx_kv_quant::LinearAttnCache]>,
        device: Device,
    ) -> Result<(Array, Array)> {
        match self {
            Architecture::Qwen3_5Moe(m) => {
                m.forward_verify_capture(ids, k, capture_layer_ids, kv_caches, lin_caches, device)
            }
            _ => Err(Error::Model(
                "forward_verify_capture: Qwen3.5MoE only (DFlash)".into(),
            )),
        }
    }

    /// Combined verify forward returning final-normed hidden (hot-path).
    ///
    /// Same as [`forward_verify_capture`] but also returns
    /// `final_hidden[1,k,H]` -- the final-RMSNorm'd hidden at the last `k`
    /// positions. Used by the EAGLE-3 hot-path to perform a restricted-vocab
    /// matmul against the draft-vocab lm_head subset.
    ///
    /// Returns `(concat_hidden[1,k,n_aux*H], final_hidden[1,k,H])`. No
    /// full-vocab logits materialised -- caller computes them at a single
    /// correction position via `logits_from_hidden`.
    /// Qwen3.5MoE only.
    #[allow(clippy::type_complexity)]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn forward_verify_capture_hot(
        &self,
        ids: &[u32],
        k: usize,
        capture_layer_ids: &[usize],
        kv_caches: &mut [rmlx_kv_quant::KvCache],
        lin_caches: Option<&mut [rmlx_kv_quant::LinearAttnCache]>,
        device: Device,
    ) -> Result<(Array, Array)> {
        match self {
            Architecture::Qwen3_5Moe(m) => m.forward_verify_capture_hot(
                ids,
                k,
                capture_layer_ids,
                kv_caches,
                lin_caches,
                device,
            ),
            _ => Err(Error::Model(
                "forward_verify_capture_hot: Qwen3.5MoE only".into(),
            )),
        }
    }

    /// Chunked verifier prefill for long prompts.
    ///
    /// Processes `ids` in windows of at most `chunk_size` tokens, accumulating
    /// KV/GDN caches normally across chunks. Returns
    /// `(logits[1,1,vocab], concat_hidden[1,n,n_aux*hidden])` where logits
    /// covers only the single last prompt position (sufficient for the first
    /// bonus token) and `concat_hidden` covers all `n` positions (needed for
    /// the drafter KV prefill).
    ///
    /// Avoids materialising a `[1, n, vocab]` logit tensor in a single Metal
    /// command buffer, eliminating GPU timeouts for n > ~1k on Qwen3.6-MoE.
    /// Qwen3.5MoE only.
    #[allow(clippy::type_complexity)]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn forward_verify_capture_chunked(
        &self,
        ids: &[u32],
        capture_layer_ids: &[usize],
        kv_caches: &mut [rmlx_kv_quant::KvCache],
        lin_caches: Option<&mut [rmlx_kv_quant::LinearAttnCache]>,
        chunk_size: usize,
        device: Device,
    ) -> Result<(Array, Array)> {
        match self {
            Architecture::Qwen3_5Moe(m) => m.forward_verify_capture_chunked(
                ids,
                capture_layer_ids,
                kv_caches,
                lin_caches,
                chunk_size,
                device,
            ),
            _ => Err(Error::Model(
                "forward_verify_capture_chunked: Qwen3.5MoE only".into(),
            )),
        }
    }

    /// Restricted-vocab logits for the EAGLE-3 hot-path.
    ///
    /// Computes `hidden @ W_hot.T` where `W_hot` contains only the `hot_ids`
    /// rows of the LM head weight. See `Qwen3_5MoeText::hot_logits_from_final_hidden`
    /// for details. Qwen3.5MoE only.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn hot_logits_from_final_hidden(
        &self,
        hidden: &Array,
        hot_ids: &Array,
        device: Device,
    ) -> Result<Array> {
        match self {
            Architecture::Qwen3_5Moe(m) => m.hot_logits_from_final_hidden(hidden, hot_ids, device),
            _ => Err(Error::Model(
                "hot_logits_from_final_hidden: Qwen3.5MoE only".into(),
            )),
        }
    }

    /// Raw input-token embedding for the DFlash drafter.
    ///
    /// Returns `[1, n, hidden]` with NO scale (the Qwen3.5 verifier's
    /// `embed_tokens` is a bare `nn.Embedding`, so the DFlash `bind()` resolves
    /// `embed_scale = 1.0`). Qwen3.5MoE only; other archs return `Error::Model`.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn embed_tokens_raw(&self, ids: &[i32], device: Device) -> Result<Array> {
        match self {
            Architecture::Qwen3_5Moe(m) => m.embed_token_ids(ids, device),
            _ => Err(Error::Model(
                "embed_tokens_raw: Qwen3.5MoE only (DFlash)".into(),
            )),
        }
    }

    /// Apply the verifier's final RMSNorm to a pre-final-norm hidden (MTP
    /// drafter conditioning -- `speculative_draft_hidden`). Gemma4 only.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn apply_final_norm(&self, hidden: &Array, device: Device) -> Result<Array> {
        match self {
            Architecture::Gemma4(m) => m.apply_final_norm(hidden, device),
            _ => Err(Error::Model(
                "apply_final_norm: Gemma4 only (assistant MTP)".into(),
            )),
        }
    }

    /// Re-derive logits from a **pre-final-norm** hidden state: `final_norm`
    /// then the LM head.
    ///
    /// Inverse of the tail dropped by [`Architecture::forward_hidden_states`].
    /// Every architecture that implements it applies the norm — a caller
    /// holding an already-normed hidden wants
    /// [`Architecture::logits_from_final_hidden`] instead. The two contracts
    /// used to share this one name, and the arch that did not norm quietly
    /// reweighted the vocabulary by the norm's own weight vector for every
    /// caller that handed it a raw capture.
    /// `hidden`: `[1, n, hidden]` -> `[1, n, vocab]`.
    #[allow(
        clippy::unreachable,
        reason = "Architecture::Gemma4 and Qwen3_5Moe arms are handled by the outer match; \
                  the inner arms are structurally unreachable -- required to exhaust the enum \
                  for the arch-name string table"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn logits_from_hidden(&self, hidden: &Array, device: Device) -> Result<Array> {
        match self {
            Architecture::Gemma4(m) => m.logits_from_hidden(hidden, device),
            Architecture::Qwen3_5Moe(m) => m.logits_from_hidden(hidden, device),
            other => {
                let arch = match other {
                    Architecture::Gemma4(_) | Architecture::Qwen3_5Moe(_) => unreachable!(),
                    Architecture::Gemma3(_) => "Gemma3",
                    Architecture::Qwen2(_) => "Qwen2",
                    Architecture::Qwen3(_) => "Qwen3",
                    Architecture::Laguna(_) => "Laguna",
                    Architecture::Qwen3VlMoe(_) => "Qwen3VlMoe",
                    Architecture::BitNet(_) => "BitNet",
                };
                Err(Error::Model(format!(
                    "logits_from_hidden not yet wired for {arch} (Gemma4 and \
                     Qwen3.5-MoE only)"
                )))
            }
        }
    }

    /// Logits from a hidden state the caller has **already final-normed**.
    ///
    /// The counterpart of [`Architecture::logits_from_hidden`]: LM head (and
    /// the architecture's logit softcap) with no norm. A verify pass that
    /// captured after the final norm — or a drafter with a final norm of its
    /// own — holds this shape.
    #[allow(
        clippy::unreachable,
        reason = "Architecture::Gemma4 and Qwen3_5Moe arms are handled by the outer match; \
                  the inner arms are structurally unreachable -- required to exhaust the enum \
                  for the arch-name string table"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn logits_from_final_hidden(&self, hidden: &Array, device: Device) -> Result<Array> {
        match self {
            Architecture::Gemma4(m) => m.logits_from_final_hidden(hidden, device),
            Architecture::Qwen3_5Moe(m) => m.logits_from_final_hidden(hidden, device),
            other => {
                let arch = match other {
                    Architecture::Gemma4(_) | Architecture::Qwen3_5Moe(_) => unreachable!(),
                    Architecture::Gemma3(_) => "Gemma3",
                    Architecture::Qwen2(_) => "Qwen2",
                    Architecture::Qwen3(_) => "Qwen3",
                    Architecture::Laguna(_) => "Laguna",
                    Architecture::Qwen3VlMoe(_) => "Qwen3VlMoe",
                    Architecture::BitNet(_) => "BitNet",
                };
                Err(Error::Model(format!(
                    "logits_from_final_hidden not yet wired for {arch} (Gemma4 and \
                     Qwen3.5-MoE only)"
                )))
            }
        }
    }

    /// Cache-using single-token forward.
    ///
    /// Returns logits at the last position, shape `[1, 1, vocab_size]`,
    /// reading + writing the provided per-layer `kv_caches`. Used by the
    /// draft decode loop in speculative decoding.
    ///
    /// `lin_caches` carries the per-layer `LinearAttnCache` for architectures
    /// with GDN layers (e.g. Qwen3.5MoE). Pass `None` for Gemma4 / others.
    /// See `forward_seq_last_k_with_cache` for the full Phase-4 wiring note.
    #[allow(
        clippy::unreachable,
        reason = "Architecture::Gemma4 and Qwen3_5Moe arms are handled by the outer match; \
                  the inner arms are structurally unreachable -- required to exhaust the enum \
                  for the arch-name string table"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn forward_arr_with_cache(
        &self,
        ids_arr: &Array,
        seq: i32,
        kv_caches: &mut [rmlx_kv_quant::KvCache],
        lin_caches: Option<&mut [rmlx_kv_quant::LinearAttnCache]>,
        device: Device,
    ) -> Result<Array> {
        match self {
            Architecture::Gemma4(m) => {
                // Gemma4 has no GDN layers; lin_caches is always None here.
                let _ = lin_caches;
                m.forward_arr(ids_arr, seq, Some(kv_caches), device)
            }
            Architecture::Qwen3_5Moe(m) => {
                // Hybrid FA + GDN: advance both kv_caches and lin_caches.
                m.forward_arr(ids_arr, seq, Some(kv_caches), lin_caches, device)
            }
            other => {
                let arch = match other {
                    Architecture::Gemma4(_) | Architecture::Qwen3_5Moe(_) => unreachable!(),
                    Architecture::Gemma3(_) => "Gemma3",
                    Architecture::Qwen2(_) => "Qwen2",
                    Architecture::Qwen3(_) => "Qwen3",
                    Architecture::Laguna(_) => "Laguna",
                    Architecture::Qwen3VlMoe(_) => "Qwen3VlMoe",
                    Architecture::BitNet(_) => "BitNet",
                };
                Err(Error::Model(format!(
                    "forward_arr_with_cache not yet wired for {arch}"
                )))
            }
        }
    }

    /// Number of decoder layers (for sizing per-layer caches).
    pub fn num_hidden_layers(&self) -> usize {
        match self {
            Architecture::Gemma4(m) => m.cfg.num_hidden_layers,
            Architecture::Gemma3(m) => m.cfg.num_hidden_layers,
            Architecture::Qwen2(m) => m.cfg.num_hidden_layers,
            Architecture::Qwen3(m) => m.cfg.num_hidden_layers,
            Architecture::Laguna(m) => m.cfg.num_hidden_layers,
            Architecture::Qwen3_5Moe(m) => m.cfg.num_hidden_layers,
            Architecture::Qwen3VlMoe(m) => m.text.cfg.num_hidden_layers,
            Architecture::BitNet(m) => m.cfg.num_hidden_layers,
        }
    }

    /// Whether this architecture has GatedDeltaNet (linear-attention) layers
    /// that require a per-layer `LinearAttnCache` alongside the standard
    /// `KvCache`. Only the Qwen3.5MoE hybrid stack does today; all others
    /// are pure FullAttention and ignore `lin_caches` (pass `None`).
    /// BitNet has no GDN layers — always returns `false` for it.
    ///
    /// Used by the speculative decoder to decide whether to allocate +
    /// snapshot/restore the recurrent GDN state per round.
    pub fn needs_lin_caches(&self) -> bool {
        matches!(self, Architecture::Qwen3_5Moe(_))
    }

    /// Whether this architecture emits `<think>...</think>` reasoning tokens
    /// that the server should surface on a separate output channel (A3).
    ///
    /// `true` only for the Qwen3 family (Qwen3 dense + Qwen3.5 MoE). It says
    /// the architecture *can* produce `<think>...</think>` — nothing more.
    /// Whether a given checkpoint's chat template prefills an open `<think>`,
    /// prefills a closed `<think></think>` so the model answers directly, or
    /// prefills nothing varies **per checkpoint** inside this same family, so
    /// the server reads the initial reasoning channel off the rendered prompt
    /// rather than off this flag. All other architectures (including BitNet)
    /// emit plain assistant text — the state machine in the server's decode
    /// loop is skipped entirely when this returns `false`.
    pub fn supports_thinking(&self) -> bool {
        matches!(self, Architecture::Qwen3(_) | Architecture::Qwen3_5Moe(_))
    }

    /// Whether this architecture's decoder layers read **each other's** K/V —
    /// a cross-layer-KV (shared-KV) topology.
    ///
    /// `true` only for Gemma4, whose consumer layers project no K/V of their
    /// own and attend over a designated producer layer's cache via
    /// `KvCache::update_and_sdpa_shared_source`. Every other stack in the tree
    /// gives each layer its own K/V.
    ///
    /// Every caller that builds a `KvCache` for an unknown architecture — the
    /// speculative verifier stacks — must pass this to
    /// `KvCache::with_shares_kv`. It is what decides whether the `Mixed` /
    /// `RotK` codecs keep their bf16 K/V mirror: on a stack that shares, the
    /// mirror is the share; on one that does not, nothing reads it and it is
    /// two full bf16 buffers of dead memory per layer.
    ///
    /// The match is exhaustive so a new architecture is classified rather than
    /// inheriting a default. The `true` arm reads
    /// [`crate::gemma4::SHARES_KV_ACROSS_LAYERS`] rather than restating it: that
    /// const is also what Gemma4's own cache builder, its SSD hydrate and its
    /// KV-residency advisory pass, so this accessor cannot drift from them. A
    /// literal here could be flipped on its own, leaving Gemma4's generate path
    /// working while Gemma4 speculative decoding lost the mirror.
    ///
    /// The non-sharing arms read their own arch's const for the same reason,
    /// even though every one of them is `false` and `false` is also the
    /// `KvCache` constructor default (pinned by
    /// `with_quant_constructors_default_to_no_cross_layer_sharing`). They were
    /// literals while the value only decided whether a mirror nothing reads got
    /// allocated — bounded to residency. It now also selects the
    /// boundary-layer codec (`kv_cache::boundary_floor`), so it reaches decoded
    /// output, and it is read from three places per arch: the cache-building
    /// loop, the prompt-cache seed, and here. One const per arch is what keeps
    /// those three from disagreeing.
    pub fn shares_kv_across_layers(&self) -> bool {
        match self {
            Architecture::Gemma4(_) => crate::gemma4::SHARES_KV_ACROSS_LAYERS,
            Architecture::Gemma3(_) => crate::gemma3::SHARES_KV_ACROSS_LAYERS,
            Architecture::Qwen2(_) => crate::qwen2::SHARES_KV_ACROSS_LAYERS,
            Architecture::Qwen3(_) => crate::qwen3::SHARES_KV_ACROSS_LAYERS,
            Architecture::Laguna(_) => crate::laguna::SHARES_KV_ACROSS_LAYERS,
            Architecture::Qwen3_5Moe(_) => crate::qwen3_5_moe::SHARES_KV_ACROSS_LAYERS,
            Architecture::Qwen3VlMoe(_) => crate::qwen3_vl_moe::SHARES_KV_ACROSS_LAYERS,
            Architecture::BitNet(_) => crate::bitnet::SHARES_KV_ACROSS_LAYERS,
        }
    }

    /// SWA window size in tokens for layer `i`, or `None` if it is a
    /// full-attention layer. Used by the SWA logic to decide whether to
    /// instantiate a rotating ring-buffer KV cache for the layer.
    ///
    /// Non-SWA architectures (Qwen2/Qwen3/Laguna/Qwen3_5Moe) return `None` for
    /// every layer.
    pub fn layer_sliding_window(&self, i: usize) -> Option<i32> {
        match self {
            Architecture::Gemma4(m) => match m.cfg.layer_types.get(i)? {
                crate::gemma4::LayerType::SlidingAttention => Some(m.cfg.sliding_window as i32),
                crate::gemma4::LayerType::FullAttention => None,
            },
            Architecture::Gemma3(m) => match m.cfg.layer_types.get(i)? {
                crate::gemma3::LayerType::SlidingAttention => Some(m.cfg.sliding_window as i32),
                crate::gemma3::LayerType::FullAttention => None,
            },
            Architecture::Qwen2(_)
            | Architecture::Qwen3(_)
            | Architecture::Laguna(_)
            | Architecture::Qwen3_5Moe(_)
            | Architecture::Qwen3VlMoe(_)
            | Architecture::BitNet(_) => None,
        }
    }

    /// Max position embeddings (for KV cache `max_seq` derivation).
    ///
    /// Returns `0` for architectures whose config struct does not surface
    /// the field -- callers should treat 0 as "use `KV_MAX_SEQ_DEFAULT`".
    pub fn max_position_embeddings(&self) -> i32 {
        match self {
            Architecture::Gemma4(m) => m.cfg.max_position_embeddings as i32,
            Architecture::Qwen3_5Moe(m) => m.cfg.max_position_embeddings as i32,
            Architecture::Qwen3(m) => m.cfg.max_position_embeddings as i32,
            Architecture::Qwen3VlMoe(m) => m.max_position_embeddings as i32,
            // Configs below don't expose mpe directly; the per-arch generate
            // paths derive max_seq from KV_MAX_SEQ_DEFAULT in that case.
            Architecture::Gemma3(_) => 0,
            Architecture::Qwen2(_) => 0,
            Architecture::Laguna(_) => 0,
            Architecture::BitNet(m) => m.cfg.max_position_embeddings as i32,
        }
    }

    /// Positional capacity of the loaded checkpoint, RoPE scaling included.
    ///
    /// The input to [`crate::context::resolve_context`], the one producer of
    /// every context bound in the tree. Qwen3 is the only architecture that
    /// implements RoPE scaling, so it is the only one whose limits can carry
    /// a [`crate::context::ContextScaling`]; the rest report their trained
    /// window and say so.
    pub fn context_limits(&self) -> crate::context::ContextLimits {
        match self {
            Architecture::Qwen3(m) => m.cfg.context,
            Architecture::Gemma4(_)
            | Architecture::Gemma3(_)
            | Architecture::Qwen2(_)
            | Architecture::Laguna(_)
            | Architecture::Qwen3_5Moe(_)
            | Architecture::Qwen3VlMoe(_)
            | Architecture::BitNet(_) => {
                crate::context::ContextLimits::trained_only(self.max_position_embeddings())
            }
        }
    }

    /// Hidden (model) size -- width of the decoder-trunk hidden state.
    ///
    /// Used by to size the MTP drafter (its `fc` consumes `2 * hidden`)
    /// and to validate the shape returned by [`Architecture::forward_hidden_states`].
    pub fn hidden_size(&self) -> usize {
        match self {
            Architecture::Gemma4(m) => m.cfg.hidden_size,
            Architecture::Gemma3(m) => m.cfg.hidden_size,
            Architecture::Qwen2(m) => m.cfg.hidden_size,
            Architecture::Qwen3(m) => m.cfg.hidden_size,
            Architecture::Laguna(m) => m.cfg.hidden_size,
            Architecture::Qwen3_5Moe(m) => m.cfg.hidden_size,
            Architecture::Qwen3VlMoe(m) => m.text.cfg.hidden_size,
            Architecture::BitNet(m) => m.cfg.hidden_size,
        }
    }

    /// KV-head count for the standard self-attention layers (the value
    /// folded into the layout key + every per-layer KV-cache geometry). For
    /// hybrid GDN archs we return the *attention* KV head count; the linear-
    /// attention head count is layout-orthogonal.
    pub fn num_key_value_heads(&self) -> usize {
        match self {
            Architecture::Gemma4(m) => m.cfg.num_key_value_heads,
            Architecture::Gemma3(m) => m.cfg.num_key_value_heads,
            Architecture::Qwen2(m) => m.cfg.num_key_value_heads,
            Architecture::Qwen3(m) => m.cfg.num_key_value_heads,
            Architecture::Laguna(m) => m.cfg.num_key_value_heads,
            Architecture::Qwen3_5Moe(m) => m.cfg.num_key_value_heads,
            Architecture::Qwen3VlMoe(m) => m.text.cfg.num_key_value_heads,
            Architecture::BitNet(m) => m.cfg.num_key_value_heads,
        }
    }

    /// per-head dim for the standard self-attention layers (the value
    /// folded into the layout key + every per-layer KV-cache geometry).
    pub fn head_dim(&self) -> usize {
        match self {
            Architecture::Gemma4(m) => m.cfg.head_dim,
            Architecture::Gemma3(m) => m.cfg.head_dim,
            Architecture::Qwen2(m) => m.cfg.head_dim,
            Architecture::Qwen3(m) => m.cfg.head_dim,
            Architecture::Laguna(m) => m.cfg.head_dim,
            Architecture::Qwen3_5Moe(m) => m.cfg.head_dim,
            Architecture::Qwen3VlMoe(m) => m.text.cfg.head_dim,
            Architecture::BitNet(m) => m.cfg.head_dim,
        }
    }

    /// Vocabulary size (for logit slicing / argmax).
    pub fn vocab_size(&self) -> usize {
        match self {
            Architecture::Gemma4(m) => m.cfg.vocab_size,
            Architecture::Gemma3(m) => m.cfg.vocab_size,
            Architecture::Qwen2(m) => m.cfg.vocab_size,
            Architecture::Qwen3(m) => m.cfg.vocab_size,
            Architecture::Laguna(m) => m.cfg.vocab_size,
            Architecture::Qwen3_5Moe(m) => m.cfg.vocab_size,
            Architecture::Qwen3VlMoe(m) => m.text.cfg.vocab_size,
            Architecture::BitNet(m) => m.cfg.vocab_size,
        }
    }

    /// Architecture class name as a static string, matching the canonical
    /// `model_type` identifiers used throughout rMLX (tracing fields, metrics).
    ///
    /// Prefer this over the `architectures[0]` string a checkpoint declares:
    /// the two can disagree, and where they do this one is the truth. Safety
    /// predicates keyed on the architecture (the Qwen-MoE K-side codec guard in
    /// particular) must use this, because a declaration is model-side data that
    /// nothing validates.
    ///
    /// **Resolved for the Qwen3.5 family.** Both Qwen3.5 arch strings share a
    /// single loader and model struct, so the sparse-MoE vs dense-SwiGLU
    /// distinction is recovered from the built layers. Every other arm returns
    /// the variant's canonical name.
    ///
    /// Known exception: `Qwen3VlMoe` also picks dense-vs-MoE per layer
    /// (`mlp_only_layers` / `num_experts` / `decoder_sparse_step`), but there is
    /// no registered dense Qwen3-VL arch string to report, so an all-dense
    /// Qwen3-VL checkpoint is still labelled MoE and still refused K-side
    /// codecs. Closing that needs a second registered class and arms for it in
    /// every consumer, not just an accessor here.
    ///
    /// `Gemma4UnifiedForConditionalGeneration` is a deliberate alias: it
    /// resolves to `Gemma4ForConditionalGeneration`, and every consumer carries
    /// an explicit arm for the declared form (see `registry::is_declared_arch_alias`).
    pub fn arch_class(&self) -> &'static str {
        match self {
            Architecture::Gemma4(_) => "Gemma4ForConditionalGeneration",
            Architecture::Gemma3(_) => "Gemma3ForConditionalGeneration",
            Architecture::Qwen2(_) => "Qwen2ForCausalLM",
            Architecture::Qwen3(_) => "Qwen3ForCausalLM",
            Architecture::Laguna(_) => "LagunaForCausalLM",
            Architecture::Qwen3_5Moe(m) => m.arch_class(),
            Architecture::Qwen3VlMoe(_) => "Qwen3VLMoeForConditionalGeneration",
            Architecture::BitNet(_) => "BitNetForCausalLM",
        }
    }

    /// Re-check the arch invariants for `kv_quant` against the **resolved**
    /// architecture, immediately before it is used to build KV caches.
    ///
    /// The startup resolvers validate against the checkpoint's declared
    /// `architectures[0]`, which is model-side data: a snapshot whose
    /// declaration disagrees with what the loader built passes those checks
    /// while running the guarded code path. This is the enforcing copy — it
    /// asks the loaded model, so no declaration can route around it, and it
    /// also covers quants that enter after startup (a per-request override).
    ///
    /// Arch-agnostic: it delegates to the one shared invariant table, so a
    /// codec that is legal on this architecture costs a single enum match.
    pub fn validate_kv_quant(&self, kv_quant: rmlx_kv_quant::KvQuant) -> Result<()> {
        crate::kv_cache::validate_resolved_kv_quant(self.arch_class(), &kv_quant).map_err(|e| {
            Error::Config(format!(
                "KV codec rejected for the architecture this snapshot actually resolves to \
                 ({}): {e}",
                self.arch_class()
            ))
        })
    }

    /// One-line summary string for diagnostics and tracing.
    pub fn config_summary(&self) -> String {
        match self {
            Architecture::Gemma4(m) => format!(
                "Gemma4ForConditionalGeneration layers={} hidden={} vocab={}",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size
            ),
            Architecture::Gemma3(m) => format!(
                "Gemma3ForConditionalGeneration layers={} hidden={} vocab={}",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size
            ),
            Architecture::Qwen2(m) => format!(
                "Qwen2ForCausalLM layers={} hidden={} vocab={}",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size
            ),
            Architecture::Qwen3(m) => format!(
                "Qwen3ForCausalLM layers={} hidden={} vocab={}",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size
            ),
            Architecture::Laguna(m) => format!(
                "LagunaForCausalLM layers={} hidden={} vocab={} experts={}",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size, m.cfg.num_experts
            ),
            Architecture::Qwen3_5Moe(m) => format!(
                "Qwen3_5MoeForConditionalGeneration layers={} hidden={} vocab={} experts={}",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size, m.cfg.num_experts
            ),
            Architecture::Qwen3VlMoe(m) => format!(
                "Qwen3VLMoeForConditionalGeneration layers={} hidden={} vocab={} experts={}",
                m.text.cfg.num_hidden_layers,
                m.text.cfg.hidden_size,
                m.text.cfg.vocab_size,
                m.text.cfg.num_experts
            ),
            Architecture::BitNet(m) => format!(
                "BitNetForCausalLM layers={} hidden={} vocab={}",
                m.cfg.num_hidden_layers, m.cfg.hidden_size, m.cfg.vocab_size
            ),
        }
    }

    /// Greedy generation using KV-cache prefill + decode (all architectures).
    ///
    /// `kv_quant` selects the KV cache quantization mode. Pass `None` for the
    /// auto default ([`crate::kv_cache::DEFAULT_KV_QUANT`]); pass `Some(q)` to
    /// override (e.g. from the `--kv-quant` CLI flag).
    ///
    /// `max_ctx_override` sets the KV buffer size. Pass `None` to use the
    /// model's `max_position_embeddings` (capped at `KV_MAX_SEQ_DEFAULT`).
    /// Pass `Some(n)` to use `n` directly -- no further capping. Typically
    /// sourced from the `--max-ctx` CLI flag.
    ///
    /// `step_fn` is called once per generated token (immediately after the step
    /// is pushed to the result vector) so callers can stream tokens as produced.
    /// Pass `&mut |_| {}` to discard per-step notifications.
    ///
    /// Returns `Vec<ProbeStep>` -- one entry per generated token.
    #[tracing::instrument(skip_all, fields(n_tokens, prompt_len = prompt_ids.len()))]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy<'a>(
        &self,
        tokenizer: &'a tokenizers::Tokenizer,
        prompt_ids: &[u32],
        n_tokens: usize,
        device: Device,
        kv_quant_override: Option<rmlx_kv_quant::KvQuant>,
        max_ctx_override: Option<i32>,
        prompt_cache_slots: usize,
        eos_ids: &'a [u32],
        step_fn: &'a mut dyn FnMut(&crate::decode_loop::ProbeStep) -> Option<u32>,
        // A6.2: optional sampler constraint. `None` = unmasked argmax (the
        // hot path; identical to pre-A6.2 behaviour). `Some(_)` enables the
        // masked branch in each arch's `argmax` call sites; in A6.2 the only
        // impl is `NoOpConstraint` (all-allow), so it is plumbing-only and
        // produces byte-identical output at temp=0.
        //
        // The per-request borrows share `'a`: the shared decode loop stores them
        // together in one `DecodeCtx<'a>` for the duration of the call. `'a`
        // appears only in parameter position, so callers are not over-constrained
        // — they pass all of these from a single request frame.
        constraint: Option<&'a mut dyn crate::ConstraintEngine>,
        // A7.2: sampling config + per-request RNG. `sampler_cfg.temperature
        // <= 0.0` keeps the untouched greedy GPU argmax path byte-for-byte;
        // `> 0.0` routes to the host categorical sampler. The route handler
        // constructs both from `GenerationRequest.sampling`.
        sampler_cfg: &'a crate::sampler::SamplerConfig,
        rng: &'a mut crate::sampler::Pcg32,
        // A7.3: logit-penalty configuration. `penalty_cfg.penalties_active() ==
        // false` keeps the temp=0 pure-GPU argmax path byte-for-byte untouched.
        // `token_history` accumulates every emitted token id; the arch trims it
        // to the trailing-20 window before each `apply_penalties` call.
        penalty_cfg: &'a crate::sampler::PenaltyConfig,
        token_history: &'a mut Vec<u32>,
    ) -> Result<Vec<crate::decode_loop::ProbeStep>> {
        use crate::kv_cache::DEFAULT_KV_QUANT;
        use rmlx_kv_quant::KvQuant;
        // Resolve the effective quant: explicit override wins, else the auto
        // default. It is the same constant every other resolution path reads,
        // so this site needs no config signals of its own.
        let arch_name = self.arch_class();
        let kv_quant: KvQuant = kv_quant_override.unwrap_or(DEFAULT_KV_QUANT);

        // Enforce the arch invariants against the resolved architecture before
        // a single KV cache is built. The startup resolvers key off the
        // declared `architectures[0]`; this one asks the loaded model, so a
        // mismatched declaration cannot route around the guard, and a quant
        // supplied per-request is checked on the same terms as a launch flag.
        self.validate_kv_quant(kv_quant)?;

        // Register the thread-local CPU + GPU streams + CommandEncoders once per
        // thread entry point. tokio blocking-pool threads start with no MLX stream
        // context; MLX 0.31/0.32 made default streams and CPU command encoders
        // thread-local, so the first op scheduled on a stream this thread never
        // registered faults with "There is no Stream(<device>, N) in current thread".
        // The CPU stream is registered unconditionally: even on the GPU device path,
        // K8V8 `exit_prefill` schedules a reduction on the CPU stream. No-op if
        // already registered; zero ML-semantic effect.
        rmlx_mlx::ensure_cpu_default_stream();
        if device == Device::Gpu {
            rmlx_mlx::ensure_gpu_default_stream();
        }

        tracing::info!(
            arch = arch_name,
            ?kv_quant,
            ?max_ctx_override,
            n_eos = eos_ids.len(),
            "arch::generate_greedy: resolved KV cache quant (auto or override)"
        );

        match self {
            Architecture::Gemma4(m) => crate::gemma4::generate_greedy(
                m,
                tokenizer,
                prompt_ids,
                n_tokens,
                device,
                kv_quant,
                max_ctx_override,
                prompt_cache_slots,
                eos_ids,
                step_fn,
                constraint,
                sampler_cfg,
                rng,
                penalty_cfg,
                token_history,
                // text path -- no precomputed image embeds.
                None,
            ),
            Architecture::Gemma3(m) => crate::gemma3::generate_greedy(
                m,
                tokenizer,
                prompt_ids,
                n_tokens,
                device,
                kv_quant,
                max_ctx_override,
                prompt_cache_slots,
                eos_ids,
                step_fn,
                constraint,
                sampler_cfg,
                rng,
                penalty_cfg,
                token_history,
                // text path -- no precomputed image embeds.
                None,
            ),
            Architecture::Qwen2(m) => crate::qwen2::generate_greedy(
                m,
                tokenizer,
                prompt_ids,
                n_tokens,
                device,
                kv_quant,
                max_ctx_override,
                prompt_cache_slots,
                eos_ids,
                step_fn,
                constraint,
                sampler_cfg,
                rng,
                penalty_cfg,
                token_history,
            ),
            Architecture::Qwen3(m) => crate::qwen3::generate_greedy(
                m,
                tokenizer,
                prompt_ids,
                n_tokens,
                device,
                kv_quant,
                max_ctx_override,
                prompt_cache_slots,
                eos_ids,
                step_fn,
                constraint,
                sampler_cfg,
                rng,
                penalty_cfg,
                token_history,
            ),
            Architecture::Laguna(m) => crate::laguna::generate_greedy(
                m,
                tokenizer,
                prompt_ids,
                n_tokens,
                device,
                kv_quant,
                max_ctx_override,
                prompt_cache_slots,
                eos_ids,
                step_fn,
                constraint,
                sampler_cfg,
                rng,
                penalty_cfg,
                token_history,
            ),
            Architecture::Qwen3_5Moe(m) => crate::qwen3_5_moe::generate_greedy(
                m,
                tokenizer,
                prompt_ids,
                n_tokens,
                device,
                kv_quant,
                max_ctx_override,
                prompt_cache_slots,
                eos_ids,
                step_fn,
                constraint,
                sampler_cfg,
                rng,
                penalty_cfg,
                token_history,
            ),
            Architecture::Qwen3VlMoe(m) => crate::qwen3_vl_moe::generate_greedy(
                &m.text,
                tokenizer,
                prompt_ids,
                n_tokens,
                device,
                kv_quant,
                max_ctx_override,
                prompt_cache_slots,
                eos_ids,
                step_fn,
                constraint,
                sampler_cfg,
                rng,
                penalty_cfg,
                token_history,
            ),
            Architecture::BitNet(m) => crate::bitnet::generate_greedy(
                m,
                tokenizer,
                prompt_ids,
                n_tokens,
                device,
                kv_quant,
                max_ctx_override,
                prompt_cache_slots,
                eos_ids,
                step_fn,
                constraint,
                sampler_cfg,
                rng,
                penalty_cfg,
                token_history,
            ),
        }
    }

    /// greedy generation from precomputed multimodal `inputs_embeds`.
    ///
    /// Like [`generate_greedy`](Self::generate_greedy) but prefills from the
    /// scatter-merged `embeds` `[1, seq, hidden]` + `masked_ids` `[seq]`
    /// (built by `gemma4::build_inputs_embeds`) instead of plain token ids.
    /// The prompt cache is bypassed for image prompts. Decode after the first
    /// token is identical to the text path (plain token-id forwards).
    ///
    /// Only the Gemma4 architecture supports image input today; other archs
    /// return `Error::Model`.
    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn generate_image<'a>(
        &self,
        tokenizer: &'a tokenizers::Tokenizer,
        prompt_ids: &[u32],
        embeds: Array,
        masked_ids: Array,
        n_tokens: usize,
        device: Device,
        kv_quant_override: Option<rmlx_kv_quant::KvQuant>,
        max_ctx_override: Option<i32>,
        prompt_cache_slots: usize,
        eos_ids: &'a [u32],
        // The per-request borrows share `'a`: the shared decode loop stores
        // them together in one `DecodeCtx<'a>` for the duration of the call.
        step_fn: &'a mut dyn FnMut(&crate::decode_loop::ProbeStep) -> Option<u32>,
        constraint: Option<&'a mut dyn crate::ConstraintEngine>,
        sampler_cfg: &'a crate::sampler::SamplerConfig,
        rng: &'a mut crate::sampler::Pcg32,
        penalty_cfg: &'a crate::sampler::PenaltyConfig,
        token_history: &'a mut Vec<u32>,
    ) -> Result<Vec<crate::decode_loop::ProbeStep>> {
        use crate::kv_cache::DEFAULT_KV_QUANT;
        use rmlx_kv_quant::KvQuant;

        // Same enforcement as the text entry, and in the same position: reject
        // before touching MLX thread state, so a refused codec costs nothing.
        // The `None` fallback below is the auto default, unquantised bf16 —
        // accepted by every arch invariant by construction, so it needs no
        // separate check.
        if let Some(kq) = kv_quant_override {
            self.validate_kv_quant(kq)?;
        }

        // Register the thread-local CPU + GPU streams + CommandEncoders once per
        // thread entry point (mirrors the text `generate_greedy` entry). The CPU
        // stream is registered unconditionally because K8V8 `exit_prefill` schedules
        // a reduction on the CPU stream even on the GPU device path; MLX 0.31/0.32
        // made those streams thread-local. No-op if already registered; zero
        // ML-semantic effect.
        rmlx_mlx::ensure_cpu_default_stream();
        if device == Device::Gpu {
            rmlx_mlx::ensure_gpu_default_stream();
        }

        match self {
            Architecture::Gemma4(m) => {
                let kv_quant: KvQuant = kv_quant_override.unwrap_or(DEFAULT_KV_QUANT);
                crate::gemma4::generate_greedy(
                    m,
                    tokenizer,
                    prompt_ids,
                    n_tokens,
                    device,
                    kv_quant,
                    max_ctx_override,
                    prompt_cache_slots,
                    eos_ids,
                    step_fn,
                    constraint,
                    sampler_cfg,
                    rng,
                    penalty_cfg,
                    token_history,
                    Some((embeds, masked_ids)),
                )
            }
            Architecture::Gemma3(m) => {
                let kv_quant: KvQuant = kv_quant_override.unwrap_or(DEFAULT_KV_QUANT);
                // Gemma3 has no per-layer-input gating, so `masked_ids` is
                // unused (the scatter-merged embeds carry everything). The
                // prompt cache is bypassed internally for an image prompt
                // (`has_image` → consume returns Miss without touching the
                // cache, and the snapshot store-back is `!has_image`-gated), but
                // `prompt_cache_slots` is still threaded so the cache shell is
                // sized consistently with the text path.
                let _ = masked_ids;
                crate::gemma3::generate_greedy(
                    m,
                    tokenizer,
                    prompt_ids,
                    n_tokens,
                    device,
                    kv_quant,
                    max_ctx_override,
                    prompt_cache_slots,
                    eos_ids,
                    step_fn,
                    constraint,
                    sampler_cfg,
                    rng,
                    penalty_cfg,
                    token_history,
                    Some(embeds),
                )
            }
            _ => Err(Error::Model(
                "image input is only supported by the Gemma4 and Gemma3 architectures".into(),
            )),
        }
    }

    /// borrow the inner `Gemma3Text` when this is a Gemma3 model.
    ///
    /// `None` for any other architecture. Used by the server to build
    /// multimodal `inputs_embeds` (`gemma3::build_inputs_embeds`) before
    /// dispatching to [`generate_image`](Self::generate_image).
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn as_gemma3(&self) -> Option<&Gemma3Text> {
        match self {
            Architecture::Gemma3(m) => Some(m),
            _ => None,
        }
    }

    /// borrow the inner `Gemma4Text` when this is a Gemma4 model.
    ///
    /// `None` for any other architecture. Used by the server to build
    /// multimodal `inputs_embeds` (`gemma4::build_inputs_embeds`) before
    /// dispatching to [`generate_image`](Self::generate_image).
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn as_gemma4(&self) -> Option<&Gemma4Text> {
        match self {
            Architecture::Gemma4(m) => Some(m),
            _ => None,
        }
    }

    /// borrow the inner `Qwen3VlMoe` when this is a Qwen3-VL-MoE model.
    ///
    /// `None` for any other architecture. Used by the server to run the vision
    /// tower + scatter + 3D-MRoPE image branch.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    pub fn as_qwen3vl_moe(&self) -> Option<&Qwen3VlMoe> {
        match self {
            Architecture::Qwen3VlMoe(m) => Some(m),
            _ => None,
        }
    }

    /// Prompt-cache stats for the arch's shared PromptCache, when it has one.
    ///
    /// Reads the arch-specific global static that `generate_greedy` writes after
    /// each generation call.
    pub fn cache_stats(&self) -> Option<crate::CacheStats> {
        match self {
            Architecture::Gemma4(_) => crate::gemma4::gemma4_cache_stats(),
            Architecture::Gemma3(_) => crate::gemma3::gemma3_cache_stats(),
            Architecture::Qwen3_5Moe(_) => crate::qwen3_5_moe::qwen3_5_moe_cache_stats(),
            Architecture::Qwen3(_) => crate::qwen3::read_cache_stats(),
            Architecture::Qwen2(_) => crate::qwen2::qwen2_cache_stats(),
            Architecture::BitNet(_) => crate::bitnet::bitnet_cache_stats(),
            Architecture::Qwen3VlMoe(_) => crate::qwen3_vl_moe::qwen3_vl_moe_cache_stats(),
            Architecture::Laguna(_) => crate::laguna::laguna_cache_stats(),
        }
    }

    /// Drop every snapshot held by this arch's prompt cache and reset its
    /// hit/miss counters, so the next generation misses in RAM.
    ///
    /// The measurement commands use this to make repeated generations of the
    /// same prompt comparable while keeping the cache configured as production
    /// has it. Reconfiguring to zero slots also misses every time, but it
    /// measures a different cache than the one being served.
    ///
    /// **A RAM miss is not the same as a prefill.** This clears RAM slots only;
    /// an attached SSD KV tier keeps its source, and the next request can still
    /// be served by hydrating a `.kvb` — recorded as `ssd_hits`, never as
    /// `hits`. A caller that needs the next generation to *prefill* must check
    /// the outcome (`hits == 0 && ssd_hits == 0`) rather than trust this call.
    /// The SSD tier is attached only from the server today, so the measurement
    /// commands do not meet it in practice; they check anyway.
    pub fn clear_prompt_cache(&self) {
        match self {
            Architecture::Gemma4(_) => crate::gemma4::prompt_cache::PROMPT_CACHE.clear(),
            Architecture::Gemma3(_) => crate::gemma3::prompt_cache::PROMPT_CACHE.clear(),
            Architecture::Qwen3_5Moe(_) => crate::qwen3_5_moe::prompt_cache::PROMPT_CACHE.clear(),
            Architecture::Qwen3(_) => crate::qwen3::QWEN3_PROMPT_CACHE.clear(),
            Architecture::Qwen2(_) => crate::qwen2::prompt_cache::PROMPT_CACHE.clear(),
            Architecture::BitNet(_) => crate::bitnet::prompt_cache::PROMPT_CACHE.clear(),
            Architecture::Qwen3VlMoe(_) => crate::qwen3_vl_moe::prompt_cache::PROMPT_CACHE.clear(),
            Architecture::Laguna(_) => crate::laguna::prompt_cache::PROMPT_CACHE.clear(),
        }
    }

    /// Actual on-device KV-cache bytes used by the most recent `generate_greedy`
    /// call on **this model instance**, paired with the store sequence they were
    /// written at.
    ///
    /// Reads the [`crate::kv_bytes::KvBytesCounter`] the model owns, which its
    /// generate path writes at the end of every generate call. Per instance, not
    /// per arch: two models of the same architecture resident at once each keep
    /// their own count, so a reader can never be handed the other one's figure
    /// under this one's name.
    ///
    /// Callers that *record* the byte count as a measurement must sample this
    /// before and after the generation and require
    /// [`crate::kv_bytes::KvBytesSample::seq`] to have advanced. Without that
    /// check, a generation that returns before reaching the store (an early-out
    /// prefill path, or an arch that does not maintain the counter) yields the
    /// previous call's byte count — or the `0` initialiser — with nothing to
    /// distinguish it from a fresh reading.
    pub fn kv_cache_bytes_sample(&self) -> crate::kv_bytes::KvBytesSample {
        match self {
            Architecture::Gemma4(m) => m.kv_bytes.sample(),
            Architecture::Gemma3(m) => m.kv_bytes.sample(),
            Architecture::Qwen3_5Moe(m) => m.kv_bytes.sample(),
            Architecture::Qwen3(m) => m.kv_bytes.sample(),
            Architecture::Qwen2(m) => m.kv_bytes.sample(),
            Architecture::BitNet(m) => m.kv_bytes.sample(),
            Architecture::Qwen3VlMoe(m) => m.text.kv_bytes.sample(),
            Architecture::Laguna(m) => m.kv_bytes.sample(),
        }
    }

    /// Record the KV-cache byte total the generation that just finished on this
    /// model instance produced.
    ///
    /// The write counterpart of [`Self::kv_cache_bytes_sample`], reaching the
    /// same per-instance counter. Generation paths that own their caches
    /// directly — the speculative round loops, which never call
    /// `generate_greedy` and so never reach an arch's own store site — report
    /// through here, so a caller that samples around the call can attribute the
    /// figure to it. Without the store, every such caller reads
    /// [`crate::kv_bytes::KvBytesVerdict::Unreported`] forever: correct, but
    /// no measurement.
    ///
    /// A speculative pair reports on its **verifier**, which is the model whose
    /// KV the figure describes. When draft and verifier share an architecture
    /// that is still unambiguous, because each holds its own counter.
    ///
    /// `post` is the witness minted by the decode phase, which pins the sample
    /// to after decode — see [`crate::decode_loop::PostDecode`].
    pub(crate) fn store_kv_cache_bytes(&self, n: u64, post: crate::decode_loop::PostDecode) {
        match self {
            Architecture::Gemma4(m) => m.kv_bytes.store(n, post),
            Architecture::Gemma3(m) => m.kv_bytes.store(n, post),
            Architecture::Qwen3_5Moe(m) => m.kv_bytes.store(n, post),
            Architecture::Qwen3(m) => m.kv_bytes.store(n, post),
            Architecture::Qwen2(m) => m.kv_bytes.store(n, post),
            Architecture::BitNet(m) => m.kv_bytes.store(n, post),
            Architecture::Qwen3VlMoe(m) => m.text.kv_bytes.store(n, post),
            Architecture::Laguna(m) => m.kv_bytes.store(n, post),
        }
    }

    /// Bare byte count from [`Self::kv_cache_bytes_sample`], for the reporting
    /// surfaces (`/metrics/cache`, the server's per-request gauge) that display
    /// the last-known figure and have no generation boundary to check it
    /// against.
    pub fn kv_cache_bytes(&self) -> u64 {
        self.kv_cache_bytes_sample().bytes
    }

    /// Smoke probe verdict -- delegates to gemma4::classify_smoke for now.
    pub fn classify_smoke(steps: &[crate::decode_loop::ProbeStep]) -> SmokeVerdict {
        crate::gemma4::classify_smoke(steps)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
