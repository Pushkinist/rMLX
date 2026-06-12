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

    /// Re-derive logits from a pre-final-norm hidden state.
    ///
    /// Inverse of the tail dropped by [`Architecture::forward_hidden_states`].
    /// Routes to `Gemma4Text::logits_from_hidden`; other architectures return
    /// `Error::Model`. `hidden`: `[1, n, hidden]` -> `[1, n, vocab]`.
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
                    "logits_from_hidden not yet wired for {arch} (Gemma4 only)"
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
    /// `true` only for the Qwen3 family (Qwen3 dense + Qwen3.5 MoE). Their
    /// chat templates prefill the assistant turn with an open `<think>\n`
    /// block, then the model produces reasoning text and emits the literal
    /// `</think>` once it switches to its final answer. All other
    /// architectures (including BitNet) emit plain assistant text — the state
    /// machine in the server's decode loop is skipped entirely when this
    /// returns `false`.
    pub fn supports_thinking(&self) -> bool {
        matches!(self, Architecture::Qwen3(_) | Architecture::Qwen3_5Moe(_))
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
    pub fn arch_class(&self) -> &'static str {
        match self {
            Architecture::Gemma4(_) => "Gemma4ForConditionalGeneration",
            Architecture::Gemma3(_) => "Gemma3ForConditionalGeneration",
            Architecture::Qwen2(_) => "Qwen2ForCausalLM",
            Architecture::Qwen3(_) => "Qwen3ForCausalLM",
            Architecture::Laguna(_) => "LagunaForCausalLM",
            Architecture::Qwen3_5Moe(_) => "Qwen3_5MoeForConditionalGeneration",
            Architecture::Qwen3VlMoe(_) => "Qwen3VLMoeForConditionalGeneration",
            Architecture::BitNet(_) => "BitNetForCausalLM",
        }
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
    /// `kv_quant` selects the KV cache quantization mode. Pass `None` to use
    /// the arch default selected by `KvCacheBuilder::for_arch_default` (the
    /// "auto" behaviour mandated by CLAUDE.md for Qwen MoE). Pass
    /// `Some(q)` to override (e.g. from the `--kv-quant` CLI flag).
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
        use crate::kv_cache::KvCacheBuilder;
        use rmlx_kv_quant::KvQuant;
        // Resolve the effective quant: explicit override wins; otherwise ask the builder.
        let arch_name = match self {
            Architecture::Gemma4(_) => "Gemma4ForConditionalGeneration",
            Architecture::Gemma3(_) => "Gemma3ForConditionalGeneration",
            Architecture::Qwen2(_) => "Qwen2ForCausalLM",
            Architecture::Qwen3(_) => "Qwen3ForCausalLM",
            Architecture::Laguna(_) => "LagunaForCausalLM",
            Architecture::Qwen3_5Moe(_) => "Qwen3_5MoeForConditionalGeneration",
            Architecture::Qwen3VlMoe(_) => "Qwen3VLMoeForConditionalGeneration",
            Architecture::BitNet(_) => "BitNetForCausalLM",
        };
        // `for_arch_default` is deprecated; callers with a full ModelConfig
        // should use `resolve_default`. This arch-dispatch site has no ModelConfig
        // available (the arch-level generate_greedy gets signals separately); the
        // function is a no-op (always K8V8) so the behaviour is preserved.
        #[allow(
            deprecated,
            reason = "no ModelConfig at arch-dispatch site; for_arch_default is a no-op returning K8V8"
        )]
        let kv_quant: KvQuant =
            kv_quant_override.unwrap_or_else(|| KvCacheBuilder::for_arch_default(arch_name));

        // Ensure the GPU default stream is registered for the calling thread.
        // tokio blocking-pool threads start with no GPU stream context; MLX's
        // array materialisation then fails with "There is no Stream(gpu, 0) in
        // current thread". Registering the process-global GPU stream once per
        // thread entry point avoids this without changing any ML semantics.
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
        use crate::kv_cache::KvCacheBuilder;
        use rmlx_kv_quant::KvQuant;
        match self {
            Architecture::Gemma4(m) => {
                // `for_arch_default` deprecated; no ModelConfig at this call site → keep K8V8.
                #[allow(
                    deprecated,
                    reason = "no ModelConfig at generate_image Gemma4 arm; for_arch_default is a no-op returning K8V8"
                )]
                let kv_quant: KvQuant = kv_quant_override.unwrap_or_else(|| {
                    KvCacheBuilder::for_arch_default("Gemma4ForConditionalGeneration")
                });
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
                // `for_arch_default` deprecated; no ModelConfig at this call site → K8V8.
                #[allow(
                    deprecated,
                    reason = "no ModelConfig at generate_image Gemma3 arm; for_arch_default is a no-op returning K8V8"
                )]
                let kv_quant: KvQuant = kv_quant_override.unwrap_or_else(|| {
                    KvCacheBuilder::for_arch_default("Gemma3ForConditionalGeneration")
                });
                // Gemma3 has no per-layer-input gating, so `masked_ids` is
                // unused (the scatter-merged embeds carry everything). Gemma3's
                // generate_greedy also takes no prompt_cache_slots (image
                // prompts are one-shot -- the cache is bypassed internally).
                let _ = (masked_ids, prompt_cache_slots);
                crate::gemma3::generate_greedy(
                    m,
                    tokenizer,
                    prompt_ids,
                    n_tokens,
                    device,
                    kv_quant,
                    max_ctx_override,
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

    /// Actual on-device KV-cache bytes used by the most recent `generate_greedy`
    /// call on this architecture.
    ///
    /// Reads the arch-specific prompt-cache static that `generate_greedy` writes
    /// via `store_kv_cache_bytes` at the end of every generate call.  Returns 0
    /// for architectures that do not yet maintain that static (Qwen2, BitNet,
    /// Laguna, Qwen3VlMoe).
    pub fn kv_cache_bytes(&self) -> u64 {
        match self {
            Architecture::Gemma4(_) => crate::gemma4::gemma4_kv_cache_bytes(),
            // NOTE: Gemma3 has its own generate path (`crate::gemma3::generate_greedy`)
            // that does NOT call `store_kv_cache_bytes`, so `gemma4_kv_cache_bytes()`
            // always returns 0 (or the last Gemma4 value if both were loaded in the
            // same process). Gemma3 KV byte reporting is not yet wired.
            Architecture::Gemma3(_) => 0,
            Architecture::Qwen3_5Moe(_) => crate::qwen3_5_moe::qwen3_5_moe_kv_cache_bytes(),
            Architecture::Qwen3(_) => crate::qwen3::read_kv_cache_bytes(),
            Architecture::Qwen2(_)
            | Architecture::Laguna(_)
            | Architecture::Qwen3VlMoe(_)
            | Architecture::BitNet(_) => 0,
        }
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
