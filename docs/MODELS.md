# Model Architecture Reference

Per-architecture support matrix for rMLX. Documents the config schema, weight
and KV quantization support, modalities, context limits, special features, and
known limitations for every architecture the server can load.

> **Adding a model?** This page is the per-arch *reference*. For the
> integration surface — the shared seams a new text architecture wires into,
> the per-arch points it must still write, and the verification ritual — see
> [`docs/ADDING_A_MODEL.md`](ADDING_A_MODEL.md).

## Contents

- [Overview](#overview)
- [Architecture support table](#architecture-support-table)
- [Qwen2](#qwen2-qwen2forcausallm)
- [Qwen3](#qwen3-qwen3forcausallm)
- [Qwen3.5 MoE](#qwen35-moe-qwen3_5moeforconditionalgeneration)
- [Qwen3-VL MoE](#qwen3-vl-moe-qwen3vlmoeforconditionalgeneration)
- [Gemma3](#gemma3-gemma3forconditionalgeneration)
- [Gemma4](#gemma4-gemma4forconditionalgeneration)
- [Laguna](#laguna-lagunaforcausallm)
- [BitNet b1.58](#bitnet-b158-bitnetforcausallm)
- [Jina V4](#jina-v4-jinaembeddingsv4model)
- [Whisper (audio STT)](#whisper-audio-stt)
- [Silero VAD (long-audio chunking)](#silero-vad-long-audio-chunking)
- [Qwen3-TTS (text-to-speech)](#qwen3-tts-text-to-speechase-4b-pending)
- [Speculative drafters](#speculative-drafters)
- [KV layout matrix](#kv-layout-matrix)
- [Modality summary](#modality-summary)
- [See also](#see-also)

---

## Overview

All architectures are identified by the `architectures[0]` field in
`config.json`. The loader reads this field at startup (`arch.rs:KNOWN_ARCHS`)
and dispatches to the matching backend. Unknown values are rejected with an
error before any weight I/O occurs.

The generative architectures (`Qwen2`, `Qwen3`, `Qwen3_5Moe`, `Qwen3VlMoe`,
`Gemma3`, `Gemma4`, `Laguna`) implement `Architecture::generate_greedy` and are
served via `/v1/chat/completions` and `/v1/completions`. `JinaEmbeddingsV4Model`
is an encoder-only model served via `/v1/embeddings`; it never enters the
`Architecture` enum.

### Shared decode loop

The per-token decode loop is **shared**, not re-copied per architecture. The
pipelined loop (`choose_token` → `async_eval(next)` → drain the previous pending
token → feed), the sampling fork, and Fresh chunked prefill live in
`rmlx_models::decode_loop` (`pipelined_decode`, `choose_token`,
`chunked_prefill`). The pipelined family — `Qwen3`, `Qwen3_5Moe`, `Gemma3`,
`Gemma4` — funnels every cache-lookup outcome (Miss / Exact / Prefix / image /
HydratedTail) into `pipelined_decode`.

A converting or new architecture supplies only:

- a **`forward_step` closure** `impl FnMut(&Array) -> Result<Array>` — the one
  per-token hole, monomorphized with no `dyn` dispatch;
- its **prefill / prompt-cache policy** — Fresh prefill via `chunked_prefill`,
  plus any genuinely different per-arch flush (Gemma SWA prefix-append,
  Qwen3.5-MoE HydratedTail append) which stays arch-side;
- arch-side **`decode_profile` / KV-byte accounting** after the loop returns,
  fed by the returned `DecodeStats`.

`resolve_pieces` toggles per-token piece resolution (`id_to_token` + per-step
`debug!`) on (Gemma, Qwen3) or off (Qwen3.5-MoE, which pushes empty pieces).
`Qwen2` is not yet on the shared loop; the deliberately-sync-loop archs
(`Laguna`, `Qwen3VlMoe`, `BitNet`) keep their own loops.

---

## Architecture support table

| Architecture string | Enum variant | Modalities | Default KV quant | Max ctx | Smoke |
|---|---|---|---|---|---|
| `Qwen2ForCausalLM` | `Architecture::Qwen2` | text | `K8V8` | config | green |
| `Qwen3ForCausalLM` | `Architecture::Qwen3` | text | `K8V8` (affine-2b: `Mixed k8v4 g64`) | config | green |
| `Qwen3_5MoeForConditionalGeneration` | `Architecture::Qwen3_5Moe` | text | `K8V8` | config | green |
| `Qwen3_5ForConditionalGeneration` | `Architecture::Qwen3_5Moe` | text | `K8V8` (PARO: `K8V4`) | config | green |
| `Qwen3VLMoeForConditionalGeneration` | `Architecture::Qwen3VlMoe` | text + image | `bf16` | config | green |
| `Gemma3ForConditionalGeneration` | `Architecture::Gemma3` | text + image | `Planar` | `KV_MAX_SEQ_DEFAULT` | green |
| `Gemma4ForConditionalGeneration` | `Architecture::Gemma4` | text + image + audio | `K8V8` / `Planar` / `K8V4` | config | green |
| `Gemma4UnifiedForConditionalGeneration` | `Architecture::Gemma4` (alias) | text + image (12B; audio not yet) | `K8V8` | config | green |
| `LagunaForCausalLM` | `Architecture::Laguna` | text | `K8V8` | `KV_MAX_SEQ_DEFAULT` | green |
| `BitNetForCausalLM` | `Architecture::BitNet` | text | `K8V8` | 4 096 | green |
| `JinaEmbeddingsV4Model` | (encoder — no enum variant) | text + image | n/a | 128 000 | green |

`KV_MAX_SEQ_DEFAULT` = 32 768 tokens (fallback when the config does not declare
`max_position_embeddings`).

---

## Qwen2 (`Qwen2ForCausalLM`)

### Config schema

Top-level `config.json` (no `text_config` nesting).

| Field | Type | Notes |
|---|---|---|
| `num_hidden_layers` | int | — |
| `hidden_size` | int | — |
| `num_attention_heads` | int | — |
| `num_key_value_heads` | int | GQA grouping |
| `head_dim` | int | derived as `hidden_size / num_attention_heads` when absent |
| `intermediate_size` | int | SwiGLU FFN width |
| `vocab_size` | int | — |
| `rope_theta` | float | RoPE base frequency |
| `max_position_embeddings` | int | optional; falls back to `KV_MAX_SEQ_DEFAULT` |
| `quantization.group_size` | int | weight quant group size |
| `quantization.bits` | int | weight quant bit width |
| `quantization.mode` | string | `"affine"` or `"mxfp8"` |

### Key structural properties

- Plain RMSNorm (plain-gamma weight, no +1 shift).
- Additive bias on `q_proj`, `k_proj`, `v_proj` (`.bias` tensor alongside any
  quantization `.biases`). This is unique to Qwen2 within the Qwen family.
- No per-head q/k norms (those appear in Qwen3).
- Full RoPE over the entire `head_dim`.
- SwiGLU MLP (`gate_proj`, `up_proj`, `down_proj`).
- Weight tying: `lm_head` absent when `tie_word_embeddings=true`.

### Weight quantization

Accepts any combination loadable by the MLX safetensors loader: bf16, affine
(any group size and bit width), mxfp8. Quant parameters are read from the global
`quantization` block; no per-tensor overrides are present in known snapshots.

Reference snapshot: `jinaai/ReaderLM-v2` (28 layers, hidden=1536, g64 b4
affine).

### KV quantization

All `KvQuant` variants supported. Default: `K8V8`.

The Qwen2 KV cache is a standard full-attention layout (no SWA). All KV quant
modes (`K8V4`, `K8V8`, `Planar`, `Mixed`, `RotK`, `RotKTq4V`) are accepted via
`--kv-quant` or `--cache-type-k/v`.

### Modalities

Text only. No vision or audio tower.

### Maximum context

`max_position_embeddings` from config when present; otherwise `KV_MAX_SEQ_DEFAULT`
(32 768). No YaRN or rope scaling.

### Special features

None specific to Qwen2. Thinking mode (`<think>` tokens) is not active for this
arch (`supports_thinking()` returns `false`).

### Known limitations

- `forward_seq_last_k_with_cache` not yet wired. Speculative decoding falls
  through to the Phase-2 re-prefill path.
- No YARN / rope scaling support.

### Smoke-probe status

Green. Validated against `ReaderLM-v2` at temp=0.

---

## Qwen3 (`Qwen3ForCausalLM`)

### Config schema

Top-level `config.json` (same flat layout as Qwen2).

| Field | Type | Notes |
|---|---|---|
| `num_hidden_layers` | int | — |
| `hidden_size` | int | — |
| `num_attention_heads` | int | — |
| `num_key_value_heads` | int | GQA |
| `head_dim` | int | explicit in config (not derived) |
| `intermediate_size` | int | — |
| `vocab_size` | int | — |
| `rope_theta` | float | base frequency |
| `rope_scaling` | object | optional; `rope_type="yarn"` for Bonsai |
| `max_position_embeddings` | int | surfaced to `Architecture::max_position_embeddings` |
| `quantization.*` | — | same as Qwen2 |

### Key structural properties

- Per-head q/k RMSNorm before RoPE: after reshaping to `[B, S, n_heads,
  head_dim]` the `q_norm` and `k_norm` weights (`[head_dim]`) are applied, then
  the tensor is transposed and RoPE runs.
- No additive bias on projections (`attention_bias=false`).
- Thinking-mode aware: `supports_thinking()` returns `true`. The server's decode
  loop separates `<think>...</think>` tokens from the final answer and surfaces
  them on the `reasoning_content` field.
- Prompt cache (block-hashed, persistent across turns) is active.
- SSD spill/hydration for KV blocks is wired.

### Weight quantization

Full coverage: bf16, affine (any group/bits), mxfp8. Bonsai-8B uses `g128 b2`
(2-bit ternary affine — the `prism-ml/Ternary-Bonsai-8B-mlx-2bit` snapshot).

### KV quantization

All `KvQuant` variants. Default resolution:

- Weight bits = 2 (Bonsai): `Mixed { k_bits=8, v_bits=4, k_group_size=64,
  v_group_size=64 }` — the MLX affine mixed-quant path (feeds `quantized_matmul`
  directly inside SDPA, bypassing the per-decode-step full dequantize).
- Other Qwen3 dense: `K8V8`.

Qwen3 is a full-attention-only architecture; all KV quant modes are valid.

### Modalities

Text only.

### Maximum context

`max_position_embeddings` from config. YARN (arXiv 2309.00071) is wired
when `rope_scaling.rope_type == "yarn"` is present in config.json (Bonsai
ships `factor=4.0, original_max_position_embeddings=16384`, extending
the effective context to 65 536 tokens). Without `rope_scaling` the model
runs un-scaled RoPE.

Forward-looking lever: for Qwen3-family models that lack `rope_scaling`
but want context extension, set `RMLX_YARN_FACTOR=<f>` (and optionally
`RMLX_YARN_ORIGINAL_MAX=<u>`, defaulting to `max_position_embeddings`)
before `arch::load_model`. Synthesises a YARN config with the paper
defaults `beta_fast=32, beta_slow=1`.

### Special features

- **Thinking mode.** `<think>` prefix is prefilled; the model emits reasoning
  text until `</think>`, then continues with the final answer. The server splits
  these into `reasoning_content` and `content`.
- **Prompt cache.** Block-hashed KV reuse across turns. Slot count configurable
  via `--prompt-cache-slots`.
- **SSD spill.** KV blocks spill to SSD when RAM pressure is high and are
  hydrated back before they enter SDPA.

### Known limitations

- `forward_seq_last_k_with_cache` not wired; speculative decode uses Phase-2
  fallback.

### Smoke-probe status

Green. Validated against `Ternary-Bonsai-8B-mlx-2bit` and `DR-Venus-4B-RL-mlx-8Bit`
at temp=0.

---

## Qwen3.5 MoE (`Qwen3_5MoeForConditionalGeneration`)

Also handles `Qwen3_5ForConditionalGeneration` (PARO dense variant). Both route
to `Architecture::Qwen3_5Moe`.

### Config schema

Config fields live under `text_config` in `config.json`.

| Field | Type | Notes |
|---|---|---|
| `num_hidden_layers` | int | — |
| `hidden_size` | int | — |
| `num_attention_heads` | int | — |
| `num_key_value_heads` | int | GQA |
| `head_dim` | int | optional; derived from `hidden_size / num_attention_heads` if absent |
| `vocab_size` | int | — |
| `full_attention_interval` | int | FullAttention every N layers; GDN fills the rest (default 4) |
| `num_experts` | int | total expert count |
| `num_experts_per_tok` | int | experts activated per token |
| `moe_intermediate_size` | int | per-expert FFN width |
| `shared_expert_intermediate_size` | int | shared dense expert width |
| `norm_topk_prob` | bool | normalise top-k router probabilities |
| `linear_num_value_heads` | int | GDN value head count |
| `linear_num_key_heads` | int | GDN key head count |
| `linear_key_head_dim` | int | GDN key head dimension |
| `linear_value_head_dim` | int | GDN value head dimension |
| `linear_conv_kernel_dim` | int | GDN LightConv kernel width |
| `rope_parameters.rope_theta` | float | RoPE base (default 10 000 000) |
| `rope_parameters.partial_rotary_factor` | float | fraction of head_dim rotated (default 0.25) |
| `max_position_embeddings` | int | — |
| `quantization.*` | — | global quant block; per-tensor overrides in dict |

### Key structural properties

**Hybrid FullAttention + GatedDeltaNet (GDN) stack.** The decoder is not a
uniform transformer. Every `full_attention_interval`-th layer (default every 4th)
is a standard FullAttention + MoE layer; the remaining layers are GatedDeltaNet
(linear-attention recurrent) + MoE layers. This means:

- `needs_lin_caches()` returns `true`. The server allocates a parallel
  `Vec<LinearAttnCache>` alongside the standard `Vec<KvCache>`. The GDN
  recurrent state (conv buffer + delta state) has no sequence axis — it cannot
  be truncated via `KvCache::truncate_to`. Speculative-decode rollback requires
  snapshot/restore of this state followed by re-replay of the retained prefix.
- GDN layers carry separate weight tensors (`linear_attn.*`) not present in
  Qwen2/Qwen3.
- The model is sparse-MoE throughout; every layer dispatches through
  `num_experts_per_tok` of `num_experts` experts.

**Thinking mode.** `supports_thinking()` returns `true` (same behaviour as
Qwen3 dense).

**Prompt cache and SSD spill.** Both wired, same as Qwen3.

### Weight quantization

Same coverage as Qwen3. Known snapshots include 8-bit affine and mxfp8. The PARO
variant (`Qwen3_5ForConditionalGeneration`) uses a paroquant layout detected by
the `is_paroquant` signal in `KvCacheBuilder::resolve_default`.

### KV quantization

All `KvQuant` variants except K-bits < 8 (enforced by `validate_resolved` for
this arch class — Qwen MoE PPL degrades severely with under-8-bit K).

Default resolution:
- `Qwen3_5MoeForConditionalGeneration`: `K8V8`.
- `Qwen3_5ForConditionalGeneration` PARO: `K8V4`; non-PARO: `K8V8`.

The 25% FullAttention layer density means the Mixed MLX affine path (which
routes only FA layers through `quantized_matmul`) shows smaller gains than on
dense architectures. Perf testing shows `K8V8` is faster than `Mixed k8v8 g64`
on this arch.

### Modalities

Text only. No vision or audio tower.

### Maximum context

`max_position_embeddings` from `text_config`.

### Special features

- **GDN recurrent layers.** Unique to this architecture family. The recurrent
  state enables efficient autoregressive decoding without growing KV caches for
  those layers.
- **Speculative decoding fully wired.** `forward_seq_last_k_with_cache`,
  `forward_verify_capture`, `forward_verify_capture_hot`,
  `forward_verify_capture_chunked`, `logits_from_hidden`, `embed_tokens_raw`
  all implemented. Supports both DFlash and EAGLE-3 drafters.
- **Restricted-vocab hot-path for EAGLE-3.** `hot_logits_from_final_hidden`
  computes `hidden @ W_hot.T` against only the draft-vocab rows of the LM head,
  avoiding materialising the full-vocab logit tensor at every verification step.
- **Chunked verifier prefill.** `forward_verify_capture_chunked` processes long
  prompts in windows to avoid GPU command-buffer timeout on Qwen3.6-MoE at
  n > ~1 000 tokens.
- **Thinking mode and prompt cache.** Both active.
- **Sorted-index expert gather at prefill.** Same optimization as Gemma4-26b:
  when `n_tokens*top_k ≥ 64`, the routed expert indices are sorted for
  contiguous per-expert access, the three `gather_qmm` calls run with
  `sorted_indices=true`, and the outputs are scattered back. Decode keeps the
  broadcast path. Mirrors mlx-lm `SwitchGLU`.

### Known limitations

- No vision input.

### Smoke-probe status

Green. Primary test target: `mlx-community/Qwen3.6-35B-A3B-8bit`.

---

## Qwen3-VL MoE (`Qwen3VLMoeForConditionalGeneration`)

### Config schema

Top-level config nests two sub-configs: `text_config` and `vision_config`.

**`text_config`** (plain Qwen3-MoE GQA, no GDN layers):

| Field | Type | Notes |
|---|---|---|
| `num_hidden_layers` | int | — |
| `hidden_size` | int | — |
| `num_attention_heads` | int | — |
| `num_key_value_heads` | int | GQA |
| `head_dim` | int | optional; derived if absent |
| `vocab_size` | int | — |
| `num_experts` | int | — |
| `num_experts_per_tok` | int | — |
| `decoder_sparse_step` | int | MoE every N layers (default 1) |
| `moe_intermediate_size` | int | — |
| `mlp_only_layers` | int[] | layer indices using dense MLP instead of MoE |
| `rope_theta` | float | — |
| `rope_scaling.mrope_section` | int[] | 3D M-RoPE channel widths (T, H, W); sums to head_dim/2 |
| `rope_scaling.mrope_interleaved` | bool | interleaved M-RoPE layout (true for Qwen3-VL) |
| `max_position_embeddings` | int | — |

**`vision_config`** (Qwen3-VL ViT):

| Field | Type | Notes |
|---|---|---|
| `depth` | int | ViT transformer blocks |
| `hidden_size` | int | — |
| `intermediate_size` | int | — |
| `out_hidden_size` | int | output projection target (must equal text `hidden_size`) |
| `num_heads` | int | — |
| `patch_size` | int | spatial patch size |
| `spatial_merge_size` | int | merge window for patch merging |
| `temporal_patch_size` | int | temporal patch size (video) |
| `num_position_embeddings` | int | learned position embedding table size |
| `deepstack_visual_indexes` | int[] | ViT layer indices whose output is injected into the decoder |

**Top-level special token ids:**
`image_token_id`, `video_token_id`, `vision_start_token_id`,
`vision_end_token_id`.

### Key structural properties

- Text decoder is a **plain Qwen3-MoE GQA** stack (full RoPE, per-head q/k
  RMSNorm, MoE every layer) — distinct from `Qwen3_5Moe` which has GDN linear-
  attention layers. This arch has no GDN; `needs_lin_caches()` returns `false`.
- Vision encoder uses LayerNorm (not RMSNorm), GELU-tanh MLP, and learned
  position embedding interpolation.
- **Deepstack visual injection.** Selected ViT layer outputs are captured and
  additively injected into the matching decoder layers via a cross-attention-like
  merge mechanism.
- **3D M-RoPE.** Temporal + height + width positional dimensions are encoded via
  a three-section rotary embedding whose sections sum to `head_dim / 2`. The
  `mrope_section` config field (`[24, 20, 20]` for the 30B snapshot) determines
  the split. Image patches receive (H, W) position indices; text tokens receive
  1D text positions.

### Weight quantization

Full coverage. Known snapshot: `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`
(64-group 4-bit affine).

### KV quantization

Default: `bf16` (unquantized). K8V8 measurably degrades decode quality on this
architecture at 4-bit weight quant — incoherent text and image output. Bf16 KV
reproduces the mlx-vlm reference exactly.

All other `KvQuant` modes are mechanically available via CLI override, but are
not validated as producing correct output.

### Modalities

Text and image input. The vision tower processes image patches; the M-RoPE
assigns spatial positions; patch tokens are scattered into the text embedding
sequence. Video input is structurally supported by the config (temporal patch
size, video token id) but not yet exercised in rMLX.

### Maximum context

`max_position_embeddings` from `text_config`.

### Special features

- **3D M-RoPE** for spatial + temporal + text position encoding.
- **Deepstack visual injection** at configurable ViT layer indices.
- **Vision tower** with spatial merge (`spatial_merge_size=2`) reducing the
  token count before injection.

### Known limitations

- `forward_seq_last_k_with_cache` not wired. Speculative decode uses Phase-2
  fallback.
- Video input not validated in rMLX even though the config supports it.
- No thinking-mode activation (arch does not expose `<think>` token handling).

### Smoke-probe status

Green. Validated against `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`.

---

## Gemma3 (`Gemma3ForConditionalGeneration`)

### Config schema

Fields live under `text_config`; vision fields under `vision_config`.

**`text_config`**:

| Field | Type | Notes |
|---|---|---|
| `num_hidden_layers` | int | — |
| `hidden_size` | int | — |
| `num_attention_heads` | int | — |
| `num_key_value_heads` | int | GQA |
| `head_dim` | int | explicit field |
| `intermediate_size` | int | — |
| `vocab_size` | int | — |
| `sliding_window` | int | SWA window in tokens (default 1024) |
| `_sliding_window_pattern` / `sliding_window_pattern` | int | period: every N-th layer is FullAttention (default 6) |
| `query_pre_attn_scalar` | float | attention scale = `scalar^-0.5` |
| `final_logit_softcapping` | float or null | optional; null in medgemma |
| `layer_types` | string[] | explicit per-layer `"sliding_attention"` / `"full_attention"` |
| `rope_local_base_freq` | float | RoPE theta for SWA layers (default 10 000) |
| `rope_theta` | float | RoPE theta for full-attention layers (default 1 000 000) |
| `tie_word_embeddings` | bool | — |

**`vision_config`** (standard SigLIP):

| Field | Type | Notes |
|---|---|---|
| `hidden_size` | int | default 1152 |
| `intermediate_size` | int | default 4304 |
| `num_hidden_layers` | int | default 27 |
| `num_attention_heads` | int | default 16 |
| `patch_size` | int | default 14 |
| `image_size` | int | default 896 |
| `mm_tokens_per_image` | int | soft tokens per image after AvgPool2d (default 256) |

### Key structural properties

- **SWA + FullAttention alternation.** The per-layer `layer_types` array (or
  the `sliding_window_pattern` rule: every N-th layer is full-attention, rest
  are SWA) determines the attention type per layer. SWA layers use a rotating
  ring-buffer KV cache sized to `sliding_window`; full-attention layers use the
  full KV buffer.
- Standard SigLIP vision tower: learned 1D position embeddings, bidirectional
  MHA, no ClippableLinear.
- `query_pre_attn_scalar`: the attention scale is `scalar^-0.5` rather than the
  standard `1/sqrt(head_dim)`.
- `final_logit_softcapping` is optional (null in medgemma).

### Weight quantization

Affine (any group/bits) and mxfp8. Reference snapshot: medgemma (SigLIP-based
vision + text).

### KV quantization

Default: `Planar` (q8_g128 K + PlanarQuant 4-bit V with per-pair Hadamard
rotation). The rotating SWA ring-buffer is active for sliding-attention layers;
quantized modes fall back to the standard full-size cache for those layers
(`with_quant_max_seq_window` semantics).

All `KvQuant` variants accepted.

### Modalities

Text and image input. The vision tower (SigLIP) produces 256 soft tokens per
image via AvgPool2d; an einsum projection maps them to the text hidden dimension
before scatter-merge into the embedding sequence.

### Maximum context

Falls back to `KV_MAX_SEQ_DEFAULT` (32 768) — the text config does not surface
`max_position_embeddings` directly to the architecture-level accessor in the
current implementation.

### Special features

- **SWA + FullAttention alternation.** Layer-type is read from `layer_types`
  first, then derived from `sliding_window_pattern`.
- **Dual RoPE theta.** SWA layers use `rope_local_base_freq`; full-attention
  layers use `rope_theta`.
- **Cross-layer KV sharing.** Not present in Gemma3 (that is a Gemma4 feature).

### Known limitations

- `max_position_embeddings` is not surfaced to `Architecture::max_position_embeddings`;
  falls back to `KV_MAX_SEQ_DEFAULT`.
- `forward_seq_last_k_with_cache` not wired. Speculative decode uses Phase-2.

### Smoke-probe status

Green. Validated against medgemma (text + vision).

---

## Gemma4 (`Gemma4ForConditionalGeneration`)

### Config schema

All text fields live under `text_config`; vision fields under `vision_config`;
audio fields under `audio_config`.

**`text_config`**:

| Field | Type | Notes |
|---|---|---|
| `num_hidden_layers` | int | 42 for e4b/e2b, larger for 26B/31B |
| `hidden_size` | int | 2560 for e4b |
| `num_attention_heads` | int | 8 for e4b |
| `num_key_value_heads` | int | SWA-layer KV heads |
| `num_global_key_value_heads` | int | full-attention KV heads (26B/31B) |
| `head_dim` | int | 256 for SWA layers |
| `global_head_dim` | int | 512 for full-attention layers (31B) |
| `intermediate_size` | int | dense MLP width |
| `vocab_size` | int | 262 144 |
| `sliding_window` | int | SWA window (default 512) |
| `layer_types` | string[] | per-layer `"sliding_attention"` / `"full_attention"` |
| `num_kv_shared_layers` | int | cross-layer KV sharing count (18 for e4b) |
| `hidden_size_per_layer_input` | int | AltUp residual gate width |
| `final_logit_softcapping` | float | default 30.0 |
| `attention_k_eq_v` | bool | K=V sharing for full-attention (26B/31B) |
| `enable_moe_block` | bool | sparse MoE activated (26B) |
| `num_experts` | int | MoE expert count |
| `top_k_experts` | int | experts per token |
| `moe_intermediate_size` | int | per-expert width |
| `rope_parameters.sliding_attention.rope_theta` | float | SWA RoPE theta (default 10 000) |
| `rope_parameters.full_attention.rope_theta` | float | full-attn RoPE theta (default 1 000 000) |
| `rope_parameters.full_attention.partial_rotary_factor` | float | fraction of global_head_dim rotated |
| `max_position_embeddings` | int | — |

**`vision_config`** (Gemma4 SigLIP):

| Field | Notes |
|---|---|
| `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads` | ViT dimensions |
| `patch_size` | 16 |
| `position_embedding_size` | one-hot position table length (10 240) |
| `pooling_kernel_size` | spatial pooling kernel (3) |
| `default_output_length` | pooled token budget (280) |
| `rope_theta` (via `rope_parameters.rope_theta`) | M-RoPE base (100) |
| `use_clipped_linears` | ClippableLinear activations (true for e4b) |
| `standardize` | post-pooling standardization |

**`audio_config`** (Conformer encoder):

| Field | Notes |
|---|---|
| `hidden_size`, `num_hidden_layers`, `num_attention_heads` | Conformer dimensions |
| `subsampling_conv_channels` | SSCP per-stage channels (`[128, 32]`) |
| `conv_kernel_size` | LightConv1d depthwise kernel (5) |
| `attention_chunk_size` | chunked attention window |
| `output_proj_dims` | optional output projection to text hidden size |
| `audio_token_id` | `<audio_soft_token>` id (258 881) |

### Known snapshots

| Snapshot | Hidden | Layers | MoE | KV sharing | SWA |
|---|---|---|---|---|---|
| `gemma-4-e2b-it-mxfp8` | 1536 | 26 | no | yes | yes |
| `gemma-4-e4b-it-mxfp8` | 2560 | 42 | no | yes (18 layers) | yes |
| `gemma-4-26b-a4b-it-mxfp8` | ~5376 | large | yes | yes | yes |
| `gemma-4-31b-it-mxfp8` | ~5376 | large | no | yes | yes |
| `gemma-4-12B-it-*` (Unified) | 3840 | 48 | no | yes (`k_eq_v`) | yes |

The 12B snapshots declare `architectures[0] = "Gemma4UnifiedForConditionalGeneration"`
— an encoder-free multimodal variant whose **text** decoder is identical to
Gemma4 (dense, `attention_k_eq_v`, no per-layer-input, no MoE). rMLX aliases the
arch string to the Gemma4 text loader for the decoder, and routes **vision**
through a dedicated encoder-free embedder (no SigLIP tower; see *Unified
(encoder-free) vision* below). **Audio** is not yet wired for 12B — the unified
audio front-end (`embed_audio` early-fusion) is a follow-up; the existing
`audio_tower.*` Conformer loader does not match the unified `embed_audio.*`
weights, so audio input is disabled on 12B (text + image serve end-to-end).

Text serves correctly at **all weight quants**, including the mixed 4/8-bit QAT
snapshots (`gemma-4-12B-it-qat-4bit` affine, `gemma-4-12B-it-qat-mxfp4`): their
`quantization` block keeps the MLP `gate/up/down` projections at 8-bit while the
rest is 4-bit, which the per-tensor override resolver handles unchanged. These
snapshots emit a degenerate filler token (`'1'`) on a *bare* instruction prompt
with no turn markers — `mlx-lm` reproduces this identically, and the mxfp8 build
degenerates the same way to `.`/`_`. The `--probe-smoke` heuristic therefore
templates its seed (see `docs/CLI.md` `info`) so the verdict matches the served
behaviour rather than the bare-prompt artifact.

### Key structural properties

**SWA + FullAttention alternation.** Per-layer `layer_types` array determines
the attention type. The default pattern is 5 sliding + 1 full, repeated. SWA
layers use a rotating ring-buffer KV cache; full-attention layers use the full
buffer.

**Cross-layer KV sharing.** `num_kv_shared_layers` sets the number of trailing
SWA layers whose K/V is produced by the model but immediately overwritten with
the shared K/V from the preceding full-attention layer. Only the full-attention
layers write independent K/V into the KV cache; SWA layers read the shared
tensor supplied by `update_and_sdpa_returning_kv`. This halves the KV memory
footprint for those layers.

**K=V sharing for full-attention layers (26B/31B).** When `attention_k_eq_v=true`,
the snapshot stores only `k_proj` for full-attention layers; `v_proj` is
absent and the loader reuses `k_proj` weights as `v_proj`. The KV head count
for full-attention layers is taken from `num_global_key_value_heads` in this
case.

**Dual head dimensions.** SWA layers use `head_dim` (256); full-attention
layers use `global_head_dim` (512 on 31B). Partial rotary factor applies to
`global_head_dim` for the full-attention RoPE.

**AltUp residual.** `hidden_size_per_layer_input` enables an AltUp-style
residual gate when non-zero.

**Stream stays at the model dtype (bf16 on mxfp8).** The activation stream runs
at the model dtype end-to-end. This matters for `--kv-quant none`: the global
(full-attention) K/V are stored at the stream dtype, so on mxfp8 they are bf16,
not f32 — about half the resident KV they would take if the stream widened. The
constants that could promote the stream — the embed-scale, the per-layer-input
scales, and the fused GeGLU / PLI-GeGLU activations (whose `gelu_tanh` constants
are f32) — adopt / restore the operand dtype, mirroring mlx-lm's weak-typed
Python floats. A unit-level dtype-lock test guards this against regression; see
docs/KV_QUANT.md "Gemma4 global `--kv-quant none` KV is bf16".

**Conformer audio encoder + native audio input.** Present in e4b and above. The
SSCP subsampling (two conv stages reducing the time dimension by 4×),
Macaron-style FFW blocks with `residual_weight=0.5`, chunked local attention, and
optional output projection implement the Conformer-S architecture from the Gemma4
multimodal paper. Audio tokens are marked with `audio_token_id` (258881 on e4b)
and scatter-merged into the embedding sequence similarly to image tokens.

The serve path is wired end-to-end: `/v1/chat/completions` `input_audio` content
parts (base64 WAV/etc.) are decoded → 16 kHz mono → USM log-mel front-end →
Conformer `audio_tower` → `embed_audio` projection → scattered at the `<|audio|>`
positions. The server loads the audio tower once at startup (alongside the vision
tower) when the snapshot ships an `audio_config` + `audio_tower.*` weights;
text-only / vision-only models reject `input_audio` with a clear `503 no audio
tower` (never a silent drop). The number of audio soft tokens spliced into the
prompt (`<|audio>` + `T_sub`×`<|audio|>` + `<audio|>`) is derived from the
encoder's SSCP downsample so the scatter aligns by construction. See
`docs/SERVER.md` § "Multimodal content parts".

**Speculative decoding fully wired for Gemma4.** `forward_seq_last_k_with_cache`,
`forward_hidden_states`, `forward_hidden_states_shared_kv`, `apply_final_norm`,
`logits_from_hidden`, `embed_token_raw` all implemented. The Gemma4 spec path is
the **assistant drafter** (`Gemma4AssistantDrafter`), selected with
`--draft-kind mtp` and a dedicated `*-it-assistant-bf16` draft snapshot. A plain
`Gemma4ForConditionalGeneration` model is **not** a valid `--draft-kind mtp`
draft (no MTP sidecar head) and is rejected at load — see `docs/SPECULATIVE.md`
§"`--draft-kind mtp` dispatch" and issue #23.

### Weight quantization

mxfp8 (group=32) is the primary format for public snapshots. Also supported:
unquantized bf16 (plain weights, no `.scales`); affine at any group/bits —
including affine-int4 with per-group `.biases` zero-points (Google QAT
snapshots); and the microscaling formats mxfp4 / nvfp4. On this bandwidth-bound
arch, 4-bit packed weights decode faster than mxfp8 (runtime `quantized_matmul`,
weights stay packed). mxfp4 is the recommended 4-bit format: nvfp4 loads and runs
but the MLX nvfp4 kernel currently yields degraded output on these snapshots.
Per-tensor quant overrides (router weights, QAT MLP blocks) are read from the
inline `quantization` dict.

### KV quantization

Default resolution by signal:
- `enable_moe_block=true` (26B MoE): `K8V8`.
- `hidden_size <= 2560` (e2b/e4b), not paroquant: `K8V8`.
- `hidden_size <= 2560`, paroquant: `K8V4`.
- `hidden_size >= 5376` (31B dense): `Planar`.

Cross-layer KV sharing is compatible with all `KvQuant` modes. The
`update_and_sdpa_returning_kv` path dequantizes to bf16 before passing the
shared KV to consumer layers, so `Mixed` mode also works.

### Modalities

Text, image, and audio input. rMLX implements all three towers:
- Vision: SigLIP-style ViT + VisionPooler + soft-token scatter.
- Audio: Conformer encoder + output projection + scatter.

### Unified (encoder-free) vision — `Gemma4UnifiedForConditionalGeneration` (12B)

The unified 12B has **no SigLIP `vision_tower`**. Vision is early-fusion: raw
pixel patches are projected straight into the shared 48-layer LM hidden space
(`mm_embed_dim = 3840`) as `num_soft_tokens` soft tokens. rMLX dispatches on
`architectures[0]` (`is_unified_arch`) and loads
`crates/rmlx-models/src/gemma4/vision/unified.rs` instead of the tower loader;
the Gemma4 text decoder is reused unchanged.

Per-image pipeline (faithful port of HF `gemma4_unified`
`Gemma4UnifiedVisionEmbedder` + `Gemma4UnifiedImageProcessor`):

1. Shared Gemma4 preprocess: aspect-ratio resize (mult of `model_patch_size=48`)
   + rescale to `[0,1]` (`do_normalize=false`).
2. Host patchify into 16px teacher patches (`[ry, rx, ch]`), then
   `patches_merge`: each `3×3` (`pooling_kernel_size`) group becomes one 48×48
   model patch (`patch_dim = 48²·3 = 6912`), interior laid out `[ky, ry, kx, rx,
   ch]` so the model patch is a *contiguous* sub-image; model-patch position =
   `(min teacher_x // k, min teacher_y // k)`.
3. On-device: `patch_ln1` (LayerNorm 6912) → `patch_dense` (quantized Linear
   6912→3840, +bias) → `patch_ln2` (LayerNorm 3840).
4. Factorized 2D positional embedding: `pos_embedding[x, 0, :] +
   pos_embedding[y, 1, :]` (table `[mm_posemb_size=1120, 2, 3840]`), added then
   `pos_norm` (LayerNorm 3840).
5. `embed_vision`: `RMSNormNoScale → embedding_projection` (3840 → text hidden) —
   the same [`MultimodalEmbedder`] the tower path reuses.
6. Scatter the soft tokens at the image-token run in `inputs_embeds`
   (`build_unified_inputs_embeds`), then run the shared text decoder from embeds.

`patch_ln1/ln2/pos_norm` are true **LayerNorm** (mean-subtraction, weight+bias),
not RMSNorm — verified against the snapshot's `.weight`+`.bias` tensors and the
upstream class. Color, spatial layout (4-quadrant, left/right/top/bottom), and
object counting are exact on the real 12B; fine-grained OCR is weaker than the
e4b SigLIP tower — an architectural property of the encoder-free 35M projection
(it lacks the semantic richness of a full vision encoder), not a port defect.

### Maximum context

`max_position_embeddings` from `text_config`.

### Special features

- **SWA + FullAttention alternation** with dual RoPE theta.
- **Cross-layer KV sharing** (`num_kv_shared_layers`).
- **K=V weight sharing** for full-attention layers on 26B/31B.
- **Sparse MoE** on 26B (dense + sparse block per layer). The expert dispatch
  uses **sorted-index gather** at prefill: when the flattened `n_tokens*top_k`
  count is ≥ 64 (multi-token prefill), the routed expert indices are sorted so
  each expert's rows are contiguous, the three gathered quantized matmuls run
  with `sorted_indices=true`, and the outputs are scattered back to token order.
  Contiguous per-expert access lets the gathered-matmul kernel run each expert
  as one dense block instead of scattered per-token gathers — a ~4× prefill
  speedup on 26B at 4k/16k/32k. Decode (single token, count < 64) keeps the
  simple broadcast path. Mirrors mlx-lm `SwitchGLU`. The remaining ~1.6× gap to
  mlx-lm prefill is inherent: Gemma4-26b runs a **dense MLP and the sparse
  experts in parallel every layer** plus three extra MoE RMSNorms, so it does
  strictly more per-layer work than a pure-MoE model.
- **Conformer audio encoder** with SSCP subsampling.
- **Gemma4-assistant MTP drafter.** The sidecar MTP head reads the verifier's
  pre-final-norm hidden state via `forward_hidden_states_shared_kv`, which also
  returns the per-layer-type shared K/V tensors to accelerate the drafter's own
  attention.
- **Final logit softcapping** (default tanh with cap 30.0).

### Known limitations

- `forward_seq_last_k_with_cache` for the standard two-model speculative path is
  wired but the MTP sidecar path is the primary spec target.

### Smoke-probe status

Green. Primary test targets: `gemma-4-e4b-it-mxfp8` (e4b) and
`gemma-4-26b-a4b-it-mxfp8` (26B MoE).

---

## Laguna (`LagunaForCausalLM`)

### Config schema

Top-level `config.json` (flat layout).

| Field | Type | Notes |
|---|---|---|
| `num_hidden_layers` | int | 40 for Laguna-XS.2 |
| `hidden_size` | int | — |
| `num_attention_heads` | int | global default; per-layer override via `num_attention_heads_per_layer` |
| `num_key_value_heads` | int | GQA |
| `head_dim` | int | optional; derived if absent |
| `intermediate_size` | int | dense FFN width |
| `vocab_size` | int | — |
| `sliding_window` | int | SWA window (default 512) |
| `layer_types` | string[] | per-layer `"full_attention"` / `"sliding_attention"` |
| `mlp_layer_types` | string[] | per-layer `"dense"` / `"sparse"` |
| `num_attention_heads_per_layer` | int[] | optional per-layer head count override |
| `num_experts` | int | — |
| `num_experts_per_tok` | int | — |
| `moe_intermediate_size` | int | — |
| `shared_expert_intermediate_size` | int | — |
| `moe_routed_scaling_factor` | float | router output scale (default 1.0) |
| `rope_parameters.full_attention.rope_theta` | float | full-attn RoPE base (default 500 000) |
| `rope_parameters.sliding_attention.rope_theta` | float | SWA RoPE base (default 10 000) |
| `rope_parameters.full_attention.partial_rotary_factor` | float | fraction of head_dim rotated for full-attn |
| `quantization.*` | — | global + per-tensor overrides |

### Key structural properties

- Layer 0 is dense (MLP). Layers 1-39 are sparse MoE (on the reference
  snapshot). The `mlp_layer_types` array specifies this; defaults to `[dense,
  sparse×(N-1)]`.
- Per-layer attention head count from `num_attention_heads_per_layer` (if
  present), enabling non-uniform head distributions across layers.
- SWA + FullAttention alternation driven by `layer_types`; defaults to all
  full-attention when the field is absent.
- Partial RoPE for full-attention layers (`partial_rotary_factor * head_dim`
  dimensions rotated).
- Softplus-gated attention: `g_proj` output passes through softplus and is
  multiplied into the attention values.
- Shared dense expert alongside the routed sparse experts.

### Weight quantization

Full coverage. Reference snapshot `mlx-community/Laguna-XS.2-mxfp8`: global
mxfp8 g32 b8 with per-tensor override on router weights (g64 b8 affine). Per-
tensor overrides are parsed from the inline `quantization` dict.

### KV quantization

Default: `K8V8`. All `KvQuant` modes accepted.

Note: per CLAUDE.md, Laguna is **out of scope for benchmarks and optimization
work**. It is present for correctness coverage.

### Modalities

Text only.

### Maximum context

Falls back to `KV_MAX_SEQ_DEFAULT` (32 768) — `max_position_embeddings` is not
surfaced to the architecture-level accessor.

### Special features

- **Per-layer variable head count.** Enables non-uniform attention capacity.
- **Softplus gating** in attention.
- **Per-tensor quant overrides** for router and gate weights.
- **SWA + FullAttention alternation** with dual RoPE theta.
- **Partial rotary** (full-attention layers only).

### Known limitations

- Do not benchmark or optimize per CLAUDE.md (along with DR-Venus).
- `forward_seq_last_k_with_cache` not wired. Speculative decode uses Phase-2.
- `max_position_embeddings` not surfaced at the architecture level.

### Smoke-probe status

Green. Validated against `Laguna-XS.2-mxfp8`.

---

## BitNet b1.58 (`BitNetForCausalLM`)

Microsoft BitNet b1.58 — a 2B-parameter decoder with ternary weight values
`{-1, 0, +1}` stored as 4-trits-per-byte U8.

### Config schema

Top-level `config.json` (no `text_config` nesting — same pattern as Qwen2).

| Field | Type | Notes |
|---|---|---|
| `num_hidden_layers` | int | 30 |
| `hidden_size` | int | 2560 |
| `num_attention_heads` | int | 20 |
| `num_key_value_heads` | int | 5 (GQA) |
| `intermediate_size` | int | 6912 |
| `vocab_size` | int | 128256 |
| `rms_norm_eps` | float | 1e-5 |
| `rope_theta` | float | 500 000 |
| `tie_word_embeddings` | bool | always `true`; no `lm_head` tensor |

`head_dim` is not stored in config; derived as `hidden_size / num_attention_heads` = 128.

### Weight format — ternary (int2-packed U8)

Each linear weight tensor is stored as U8 with shape `[N//4, K]`:

```
bits [1:0] of byte → trit 0
bits [3:2] of byte → trit 1
bits [5:4] of byte → trit 2
bits [7:6] of byte → trit 3

Raw encoding: 0 → 0, 1 → +1, 2 or 3 → -1
```

A sibling `*.weight_scale` tensor (BF16 scalar `[1]`) gives the absolute
magnitude of the non-zero trits. rMLX multiplies the scale in at load time,
writing a BF16 `[N, K]` matrix. Inference uses plain BF16 matmul — no
custom quantized kernel required.

See [WEIGHT_QUANTS.md § Ternary / BitLinear](WEIGHT_QUANTS.md#ternary--bitlinear-bitnetforcausallm) for the full encoding spec.

### Architecture specifics

- **Sub-norms**: `attn_sub_norm` (RMSNorm `[2560]`) applied to the
  concatenated attention output **before** `o_proj`; `ffn_sub_norm`
  (RMSNorm `[6912]`) applied after `relu2(gate) * up` and **before**
  `down_proj`. These are unique to BitNet — all other architectures norm
  before the sub-block, not inside it.
- **Relu2 activation**: `max(x, 0)^2` instead of SiLU/GeluTanh.
- **Tied LM head**: `embed_tokens` weight (`BF16 [128256, 2560]`) is reused
  as the LM head via a transpose matmul. There is no `lm_head` tensor in the
  checkpoint.
- **GQA**: 20 query heads, 5 KV heads (`n_kv_heads = n_q_heads / 4`).
- **RoPE**: full-dim rotation, `theta = 500 000`, no YaRN scaling.

### KV cache

Default: `K8V8`. Effective max context: 4 096 tokens (from `max_position_embeddings`
in config; capped at 4 096 by the loader).

### Limitations

- No vision or audio tower.
- Base model only; no instruct/chat fine-tune exists in the current snapshot
  (`mlx-community__bitnet-b1.58-2B-4T`). Generation without a proper system
  prompt will produce repetitive output.

### Smoke-probe status

Green. Validated against `mlx-community__bitnet-b1.58-2B-4T` (2026-05-28):

- Load: 6 361 ms (all CPU dequant; ternary U8 → BF16, 30 layers)
- Decode TPS (K8V8, release binary): **31.6 ± 0.2 TPS** (4 runs, avg_decode_ns from log)
- Smoke output: degenerate `"adooadoo..."` from `/v1/chat/completions`. This is
  expected — the snapshot is a base model with no instruct fine-tune. The model
  generates tokens correctly; output quality requires a prompt-tuned variant.
- `/v1/completions` (legacy text-completion endpoint): not implemented server-side.
  Use `/v1/chat/completions` with an appropriate system prompt.

**Performance note**: decode TPS is ~0.25× of the 127 TPS bandwidth ceiling. The
gap is not bandwidth on LM-head/projections — it is Metal kernel dispatch overhead
(~211 kernel launches per decode step). A Metal GEMV kernel would not close this
gap; kernel fusion across per-layer projections would be required but is out of
scope.

---

## Jina V4 (`JinaEmbeddingsV4Model`)

Jina V4 is an **encoder-only** embedding model. It does not enter the
`Architecture` enum and has no `generate_greedy` path. It is served exclusively
via `/v1/embeddings`.

### Config schema

Top-level config with `text_config` and `vision_config` sub-objects.

**`text_config`** (Qwen2.5-VL-3B backbone, plain bf16):

| Field | Notes |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 (GQA 8:1) |
| `intermediate_size` | 11 008 |
| `rms_norm_eps` | 1e-6 |
| `rope_theta` | 1 000 000.0 |
| `head_dim` | inferred as `hidden_size / num_attention_heads = 128` |
| `vocab_size` | 151 936 |
| `sliding_window` / `use_sliding_window` | disabled in jina-v4 |
| `max_position_embeddings` | 128 000 |
| `rope_scaling.mrope_section` | `[16, 24, 24]` (3D M-RoPE; sums to head_dim/2 = 64) |

**`vision_config`** (32-layer ViT, window + full-attn pattern):

| Field | Notes |
|---|---|
| `depth` | 32 |
| `hidden_size` | 1280 |
| `intermediate_size` | 3420 |
| `num_heads` | 16 |
| `out_hidden_size` | 2048 (maps to text hidden) |
| `fullatt_block_indexes` | `[7, 15, 23, 31]` — every 8th block is full-attention |
| `window_size` | 112 |
| `patch_size` | 14 |
| `spatial_patch_size` | 14 |
| `temporal_patch_size` | 2 |
| `spatial_merge_size` | 2 |

**Top-level embedding metadata:**

| Field | Notes |
|---|---|
| `single_vector_pool_strategy` | `"mean"` |
| `multi_vector_projector_dim` | 128 |
| `matryoshka_dims` | `[128, 256, 512, 1024, 2048]` — valid truncation sizes |
| `task_names` | `["retrieval", "text-matching", "code"]` — task-specific LoRA adapters |
| `mrope_section` | from `text_config.rope_scaling.mrope_section`; used for the image path |

**Special token ids:**
`vision_start_token_id`, `vision_end_token_id`, `vision_token_id`,
`image_token_id`, `bos_token_id`, `eos_token_id`.

### Key structural properties

- **Pure bf16 weights.** No quantization. The config carries no `quantization`
  block; the loader reads raw bf16 safetensors.
- **Vision ViT with mixed windowed and full-attention blocks.** Blocks at
  `fullatt_block_indexes` use global attention; all others use window attention
  (`window_size=112` patches). Vision MLP and `attn.proj` carry `bias=True`
  (jina-specific; stock mlx_vlm uses `bias=False`).
- **LoRA task adapters.** Three task-specific LoRA adapters (`retrieval`,
  `text-matching`, `code`) are loaded and can be switched per request.
- **Matryoshka embedding.** Output can be truncated to any of the declared
  `matryoshka_dims` without retraining, yielding smaller embeddings at lower
  recall.
- **3D M-RoPE for image input.** The `mrope_section` split applies to image
  patches; text tokens use 1D positions.
- **Multi-vector and single-vector outputs.** Single-vector: mean pool → L2
  normalize. Multi-vector: per-token projection to 128 dims.

### Weight quantization

None. Model is bf16 only.

### KV quantization

Not applicable. The encoder runs a full bidirectional pass; there is no
autoregressive KV cache.

### Modalities

Text and image input. Both produce `float32` embedding vectors.

### Maximum context

128 000 tokens (`max_position_embeddings` from `text_config`).

### Special features

- **Task-specific LoRA adapters** switchable per request.
- **Matryoshka truncation** to `[128, 256, 512, 1024, 2048]` dims.
- **Windowed + full-attention ViT** (full-attn at every 8th block).
- **3D M-RoPE** for spatial image position encoding.

### Known limitations

- No autoregressive generation; cannot be used with `/v1/chat/completions`.
- No quantized weight support (bf16 only).

### Smoke-probe status

Green. Validated against `jinaai/jina-embeddings-v4`.

---

## Speculative drafters

rMLX implements three speculative drafter families. All share the same
`SpeculativeDispatcher` infrastructure (persistent verifier + draft KV caches,
`truncate_to`-based rollback, optional GDN snapshot/restore).

### MTP head (`mtp`)

Multi-Token Prediction sidecar drafter. Conditions on the **verifier's
pre-final-norm hidden state** (`Architecture::forward_hidden_states`). The head
consists of a `fc` linear (`2H → H`), two pre-norm RMSNorms (for embedding and
hidden), a small decoder layer, and a final RMSNorm. The verifier's LM head is
reused for token prediction.

Wiring status:
- `fc` projection, pre-fc norms, final norm: fully loaded and executed.
- Single MTP decoder layer compute: pending — wired behind `Error::Model` until
  a checkpoint with an MTP head is available in Open Models.
- Target verifier: Gemma4 (via `forward_hidden_states` + `apply_final_norm`).

For the Gemma4-assistant variant, `forward_hidden_states_shared_kv` supplies
both the hidden state and the per-layer-type shared K/V tensors in one forward,
reducing the speculative prefill overhead.

See `docs/SPECULATIVE.md` for the full design.

### DFlash (`dflash`)

Block-diffusion non-autoregressive drafter. Drafts a whole `block_size` of
tokens in one parallel pass by denoising a masked block conditioned on the
verifier's **concatenated multi-layer hidden states** (from
`Architecture::forward_verify_capture`).

Properties:
- **Multi-layer conditioning.** Reads residual stream at `target_layer_ids`
  (e.g. 5 layers for Qwen3.6-MoE), projects the concatenation `5H → H` through
  the drafter `fc`.
- **Adaptive block size.** Grows/shrinks based on acceptance history.
- **GDN-aware rollback.** Snapshot/restore of `LinearAttnCache` + prefix replay
  on partial acceptance.
- **YARN RoPE** in the drafter layers (required for numeric alignment).
- Target verifier: `Qwen3_5Moe` (`Architecture::forward_verify_capture`,
  `embed_tokens_raw`, `logits_from_hidden`).
- Status: fully wired. Live-validated against `z-lab/Qwen3.6-35B-A3B-DFlash` +
  `mlx-community/Qwen3.6-35B-A3B-8bit` verifier. Accept rate matches mlx-vlm
  reference (0.515 on test prompt).

See `docs/SPECULATIVE.md` for algorithm details.

### EAGLE-3 (`eagle3`)

Autoregressive drafter with multi-layer feature fusion and a reduced draft
vocabulary.

Properties:
- **Multi-layer feature fusion.** Reads the verifier residual stream at three
  auxiliary layers (`eagle_aux_hidden_state_layer_ids`), concatenates along
  feature axis (`3H`), projects `3H → H` through the drafter `fc`.
- **Embed + hidden fusion.** The drafter's single decoder layer attends over
  `concat(input_layernorm(embed), hidden_norm(fc_out))` — a `2H`-wide
  attention input.
- **Reduced draft vocabulary + d2t remap.** Drafter `lm_head` covers 32 000
  draft tokens; a `d2t` buffer maps draft id → target id via additive offset.
- **Restricted-vocab hot-path.** For the Qwen3.6-MoE verifier,
  `Architecture::forward_verify_capture_hot` returns both the multi-layer
  concatenated hidden and the final RMSNorm'd hidden in one cached pass;
  `hot_logits_from_final_hidden` then computes restricted-vocab logits against
  only the `hot_ids` rows of the LM head.
- **GDN rollback** reuses `DFlashRoundState` infrastructure.
- Target verifier: `Qwen3_5Moe`.
- Status: reference-alignment pass complete (three structural divergences from
  mlx-vlm patched). Live accept-rate measurement pending.

See `docs/SPECULATIVE.md` for algorithm details.

---

## KV layout matrix

### KvQuant variants

| Variant | K codec | K group | V codec | V group |
|---|---|---|---|---|
| `None` (`bf16`) | bf16 (unquantized) | — | bf16 | — |
| `K8V8` | rMLX MSL q8_0 affine | 128 | rMLX MSL q8_0 affine | 128 |
| `K8V4` | rMLX MSL q8_0 affine | 128 | TurboQuant 4-bit Lloyd-Max N(0,1) | 32 |
| `Planar` | rMLX MSL q8_0 affine | 128 | PlanarQuant 4-bit + per-pair Hadamard | 32 |
| `Mixed{k,v}` | MLX affine (k_bits) | k_group | MLX affine (v_bits) | v_group |
| `RotK{v}` | Rotated affine 8-bit (FWHT basis) | 64 | MLX affine (v_bits) | v_group |
| `RotKTq4V` | Rotated affine 8-bit (FWHT basis) | 64 | TurboFlash 4-bit | 32 |
| `K8VTurbo3` | rMLX MSL q8_0 affine | 128 | TurboQuant 3-bit | 32 |

The K-side `q8_0` codec (K8V4, K8V8, Planar) is the rMLX MSL symmetric affine
variant: scale = max(|x|)/127, no bias term, group=128. This differs from the
MLX affine codec used in `Mixed` even at 8 bits (`Mixed` carries a bias term
and defaults to group=64).

### Per-arch KV defaults

| Architecture | Default KvQuant | Condition |
|---|---|---|
| `Qwen2ForCausalLM` | `K8V8` | always |
| `Qwen3ForCausalLM` | `Mixed{k8,v4,g64}` | weight bits = 2 (Bonsai) |
| `Qwen3ForCausalLM` | `K8V8` | other weights |
| `Qwen3_5MoeForConditionalGeneration` | `K8V8` | always |
| `Qwen3_5ForConditionalGeneration` | `K8V4` | paroquant snapshot |
| `Qwen3_5ForConditionalGeneration` | `K8V8` | other |
| `Qwen3VLMoeForConditionalGeneration` | `None` (bf16) | always |
| `Gemma3ForConditionalGeneration` | `Planar` | always |
| `Gemma4ForConditionalGeneration` | `K8V8` | MoE (26B) |
| `Gemma4ForConditionalGeneration` | `K8V8` | small (e2b/e4b), not paroquant |
| `Gemma4ForConditionalGeneration` | `K8V4` | small, paroquant |
| `Gemma4ForConditionalGeneration` | `Planar` | dense large (31B) |
| `LagunaForCausalLM` | `K8V8` | always |

Override via `--kv-quant <preset>` or `--cache-type-k <tag> --cache-type-v <tag>`.
Run `rmlx info --list-cache-types` for all valid tags.

### Constraints

- K-side bits < 8 rejected for `Qwen3_5MoeForConditionalGeneration` (PPL disaster).
- 2-bit K rejected for all architectures (incoherent attention output).
- `tq4` / `planar4` are V-side only codecs.
- `rot_k` is a K-side only codec; requires power-of-two `head_dim`.
- `tq4` requires `head_dim ∈ {128, 256}`.
- MLX affine: `head_dim % group_size == 0` and `head_dim % (32/bits) == 0`.

---

## Modality summary

| Architecture | Text | Image | Audio | Embeddings |
|---|---|---|---|---|
| `Qwen2ForCausalLM` | yes | no | no | no |
| `Qwen3ForCausalLM` | yes | no | no | no |
| `Qwen3_5MoeForConditionalGeneration` | yes | no | no | no |
| `Qwen3VLMoeForConditionalGeneration` | yes | yes | no | no |
| `Gemma3ForConditionalGeneration` | yes | yes | no | no |
| `Gemma4ForConditionalGeneration` | yes | yes | yes | no |
| `LagunaForCausalLM` | yes | no | no | no |
| `JinaEmbeddingsV4Model` | yes | yes | no | yes (only) |

---

## Whisper (audio STT)

Whisper is an encoder-decoder speech-to-text model. Unlike the generative LLM
architectures above, Whisper is served via dedicated audio endpoints, not
`/v1/chat/completions`.

### Supported snapshot

| Snapshot path (relative to `$RMLX_O_MODELS_ROOT`) | Format | Smoke status |
|---|---|---|
| `mlx-community__whisper-large-v3-mlx` | `.npz` weights + `config.json` | green (full 48-min real-audio regression, normalized WER ≈ 0.08) |

The mlx-community snapshot ships **without** a tokenizer. Use the companion
`openai/whisper-large-v3` HuggingFace tokenizer directory (contains
`tokenizer.json`) and set `--whisper-tokenizer-path` /
`RMLX_WHISPER_TOKENIZER_PATH`, or pass `--tokenizer` to `rmlx transcribe`.

#### Special-token layout (large-v3)

large-v3 has **100** language slots (`<|en|>`=50259 … `<|yue|>`=50358), one more
than v1/v2. This shifts every special after the language block up by one:
`<|translate|>`=50359, `<|transcribe|>`=50360, `<|startoflm|>`=50361,
`<|startofprev|>`=50362, `<|nospeech|>`=50363, `<|notimestamps|>`=50364, and
timestamp tokens `<|0.00|>`=50365 … `<|30.00|>`=51865. Getting these off by one
(the v2 layout) makes the decoder emit the wrong task token and treat
`<|notimestamps|>` as the timestamp sentinel — the root cause of the empty /
garbage transcripts fixed in this release.

### Config schema

| Field | Value (large-v3) | Description |
|---|---|---|
| `n_mels` | 128 | Mel filterbank bins |
| `n_audio_ctx` | 1500 | Audio context length (frames) |
| `n_audio_state` | 1280 | Encoder hidden dimension |
| `n_audio_head` | 20 | Encoder attention heads |
| `n_audio_layer` | 32 | Encoder layers |
| `n_vocab` | 51 866 | Vocabulary size (50 257 GPT-2 + 1 609 added) |
| `n_text_ctx` | 448 | Decoder context length (tokens) |
| `n_text_state` | 1280 | Decoder hidden dimension |
| `n_text_head` | 20 | Decoder attention heads |
| `n_text_layer` | 32 | Decoder layers |

### Architecture

Encoder: Conv1d stem (k=3 stride=1 pad=1, k=3 stride=2 pad=1) → fixed
sinusoidal positional embedding → 32 attention blocks → post LayerNorm.

Decoder: token embedding (GPT-2 BPE) → learned positional embedding → 32
cross-attention blocks (SOT prefix prefill, then greedy decode) → weight-tied
output projection.

### Inference

Both the audio endpoint and `rmlx transcribe` run through one shared long-form
engine (`rmlx_audio::transcribe::Transcriber`). The path is:

1. Audio decode via Symphonia → mono f32 (any container, incl. AAC/`.m4a`);
   downmix stereo and resample to 16 kHz internally.
2. Walk the audio in 30 s windows. Per window: log-mel (`MelExtractor`, 128 bins),
   encoder forward (`encode_mel`), then greedy decode in **timestamp mode**.
3. Each decode step applies the openai-whisper logit-filter chain —
   `SuppressBlank` (first step) + `SuppressTokens` (tokenizer-derived non-speech /
   special set) + `ApplyTimestampRules` (pairing, monotonic, BOS, timestamp/text
   tie-break). The suppress set is derived from the loaded tokenizer, not a
   hardcoded id list.
4. Parse timestamp tokens into segments with real cumulative times; advance the
   window seek by the last timestamp; condition the next window on the previous
   window's text (`<|startofprev|>` prompt).
5. BPE decode via `WhisperTokenizer`; emit `txt` / `json` / `srt` / `vtt`.

Decoding is greedy at temperature 0 — deterministic across runs.

### Smoke probe / regression

Integration tests live in `crates/rmlx-audio/tests/transcribe.rs` and resolve the
Whisper snapshot + tokenizer from **`RMLX_O_MODELS_ROOT` auto-discovery** (the same
convention as `make model-check-full`) — no bespoke env var. They **skip
gracefully** when the model or fixtures are absent:

- `say_clip_deterministic` — synthesises a known sentence with macOS `say`+`ffmpeg`,
  asserts low WER + byte-identical output across runs.
- `long_form_regression` — scans the gitignored
  `crates/rmlx-audio/tests/fixtures/` dir for any `*.{m4a,wav,mp3,…}` with a
  sibling `*.transcript.vtt`, transcribes the FULL file, and asserts normalized
  WER ≤ 0.30. Drop your own audio + reference VTT into that dir to enable it.

---

## Silero VAD (long-audio chunking)

Silero VAD v4 is used internally to chunk audio longer than 30 seconds before
passing it to Whisper. It is not an exposed API model — it is embedded as a
vendored safetensors asset (`crates/rmlx-audio/assets/silero_vad_16k.safetensors`).

### Architecture (16 kHz path)

| Stage | Description |
|---|---|
| STFT | Learned conv1d basis: 258 filters, 256-sample window, 128-sample hop. |
| Magnitude | `sqrt(real² + imag²)` → 129-bin magnitude spectrogram. |
| Encoder | 4× Conv1d layers (3-sample kernel, padding=1) with ReLU. |
| Decoder | 1-layer LSTM (hidden=128) + 1×1 Conv + sigmoid → per-frame speech probability. |

Weights extracted from the 16 kHz `then_branch` of the Silero VAD ONNX graph
and stored as F32 safetensors (1.2 MB). Convert script: `scripts/convert_silero_vad.py`.

License: MIT — see `crates/rmlx-audio/assets/NOTICE`.

### Chunking strategy

| Condition | Strategy |
|---|---|
| Audio ≤ 30 s | Single Whisper pass, no chunking. |
| Audio > 30 s | VAD-guided: voiced segments → merged chunks (≤ 30 s, 1 s overlap). |
| VAD returns empty | Sliding window fallback (30 s window, no overlap). |

---

## Qwen3-TTS (text-to-speech)

`Qwen3TTSForConditionalGeneration` — Qwen3 transformer talker + neural codec
decoder. Served via `POST /v1/audio/speech`.

### Status

Fully implemented. Returns `audio/wav` mono 24 kHz PCM.

### Architecture

| Component | Description |
|---|---|
| Talker | 28-layer Qwen3 decoder (hidden=2048, 16 heads, 8 KV heads, head_dim=128). Affine-8bit weights (group_size=64). MRoPE (sequential positions for text-only). Per-head Q/K RMSNorm. Generates audio token sequences. |
| CodePredictor | 5-layer mini-Qwen3 (hidden=1024, 16 heads, head_dim=64). Receives talker hidden state and generates codec groups 1..15 per step. |
| Codec decoder | SplitRVQ: 16 VQ codebooks (1 semantic + 15 acoustic), each 2048×256. Projected to 512 → pre-conv (k=3) → 8-layer pre-transformer (hidden=512→1024) → 2× ConvNeXt upsample (stride=2) → 4× ResNet decoder group (strides 8,5,4,3) → SnakeBeta → final conv+tanh → 24 kHz mono f32. |

### Available voices

`serena`, `vivian`, `ryan`, `aiden`, `eric`, `dylan`, `ono_anna`, `sohee`, `uncle_fu`
(from `talker_config.spk_id` in `config.json`).

### Configuration flags

| Flag | Env var | Purpose |
|---|---|---|
| `--tts-model-path` | `RMLX_TTS_MODEL_PATH` | Qwen3-TTS talker model snapshot directory. |
| `--tts-tokenizer-path` | `RMLX_TTS_TOKENIZER_PATH` | Codec decoder snapshot directory. |

---

## See also

- `docs/KV_CACHE.md` — KV quantization internals, codec implementations,
  asymmetric K/V design rationale.
- `docs/WEIGHT_QUANTS.md` — weight quantization formats, group sizes, per-tensor
  override parsing.
- `docs/SPECULATIVE.md` — speculative decoding design, MTP / DFlash / EAGLE-3
  algorithm details, verifier seam API.
