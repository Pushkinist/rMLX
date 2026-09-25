# Model Architecture Reference

What rMLX loads, per architecture: the arch strings, the loader facts, the
special layers, the modalities, the speculative role and the refusals.

> **Adding a model?** This page is the per-arch *reference*. The integration
> surface a new text architecture wires into is in
> [`docs/ADDING_A_MODEL.md`](ADDING_A_MODEL.md).

## Contents

- [Overview](#overview)
- [Architecture support table](#architecture-support-table)
- [Qwen2](#qwen2)
- [Qwen3](#qwen3)
- [Qwen3.5](#qwen35)
- [Qwen3-VL MoE](#qwen3-vl-moe)
- [Gemma3](#gemma3)
- [Gemma4](#gemma4)
- [Laguna](#laguna)
- [BitNet b1.58](#bitnet-b158)
- [Jina V4](#jina-v4)
- [Speculative verifier seams](#speculative-verifier-seams)
- [Whisper (audio STT)](#whisper-audio-stt)
- [Silero VAD](#silero-vad)
- [Qwen3-TTS](#qwen3-tts)
- [See also](#see-also)

---

## Overview

**Dispatch.** `arch::load_model` keys on `architectures[0]` in `config.json`.
A string outside `KNOWN_ARCHS` (`crates/rmlx-models/src/arch/registry.rs`) is
refused before any tensor I/O. `serve` runs the same check at startup.

**Declared and resolved class.** One arch string can cover several checkpoint
shapes. `Architecture::arch_class()` reports the class the loader built.
Anything that must be correct about the model reads that resolved class: the
KV codec guards, the SSD `layout_key`, tracing and metrics identity.
`load_model` warns with `declared_arch` and `resolved_arch` when the two
differ. `Gemma4UnifiedForConditionalGeneration` is a known alias of
`Gemma4ForConditionalGeneration` and logs at `debug` instead
(`is_declared_arch_alias`).

**Weight-quant preflight.** Before tensor I/O, `preflight_weight_quant`
refuses an affine `bits` outside `{2,3,4,5,6,8}`, globally or in any
`tensor_overrides` entry. A 1-bit checkpoint is refused there with a hint that
it needs an unreleased MLX kernel. See `docs/WEIGHT_QUANTS.md` §4.4.

**Routes.** The generative architectures serve `/v1/chat/completions` and
`/v1/messages`. `JinaEmbeddingsV4Model` serves `/v1/embeddings` only.

**KV codec.** `auto` is unquantised bf16 on every architecture
(`DEFAULT_KV_QUANT`); see `docs/KV_QUANT.md` § "The auto default". The
per-codec invariants that refuse a codec on a given shape are in
`docs/KV_CACHE.md` §5.

**Context.** The default ceiling is `min(positional capacity, 4096)`. A
`--max-ctx` above the capacity is refused. See `docs/CLI.md`
§ "Context ceiling". An architecture whose `max_position_embeddings()`
returns `0` has no capacity to enforce.

**Decode loop.** Qwen3, Qwen3.5, Gemma3 and Gemma4 decode through the shared
`rmlx_models::decode_loop` (`pipelined_decode`, `chunked_prefill`,
`choose_token`). Qwen2, Laguna, Qwen3-VL MoE and BitNet keep their own loops.
The per-arch prefill chunk is in `docs/KV_CACHE.md` § "Chunked prefill".

---

## Architecture support table

| `architectures[0]` | Variant | `arch_class()` | Input | Positional capacity | Spec verifier |
|---|---|---|---|---|---|
| `Qwen2ForCausalLM` | `Qwen2` | same | text | unknown (`0`) | no |
| `Qwen3ForCausalLM` | `Qwen3` | same | text | mpe, YaRN-extended | no |
| `Qwen3_5MoeForConditionalGeneration` | `Qwen3_5Moe` | resolved | text | mpe | yes |
| `Qwen3_5ForConditionalGeneration` | `Qwen3_5Moe` | resolved | text | mpe | yes |
| `Qwen3VLMoeForConditionalGeneration` | `Qwen3VlMoe` | same | text, image | mpe | no |
| `Gemma3ForConditionalGeneration` | `Gemma3` | same | text, image | unknown (`0`) | no |
| `Gemma4ForConditionalGeneration` | `Gemma4` | same | text, image, audio | mpe | yes |
| `Gemma4UnifiedForConditionalGeneration` | `Gemma4` | `Gemma4ForConditionalGeneration` | text, image, audio | mpe | yes |
| `LagunaForCausalLM` | `Laguna` | same | text | unknown (`0`) | no |
| `BitNetForCausalLM` | `BitNet` | same | text | mpe | no |
| `JinaEmbeddingsV4Model` | none (encoder) | — | text, image | — | — |

"mpe" is `max_position_embeddings` from the config the loader reads.
"resolved" means the class is read off the built layers (see Qwen3.5).
Audio on Gemma4 depends on the snapshot; see Gemma4.

Only Gemma4 and Qwen3.5 implement the verifier seams. Speculative decoding on
any other architecture fails with `not yet wired`; there is no fallback.

---

## Qwen2

`Qwen2ForCausalLM`. Flat `config.json`, no `text_config`.

- `head_dim` is always `hidden_size / num_attention_heads`; the config field
  is not read.
- Additive `.bias` on `q_proj`, `k_proj` and `v_proj`, beside any quant
  `.biases`.
- Plain RMSNorm, no per-head q/k norm, full RoPE, SwiGLU MLP.
- `lm_head` is absent when `tie_word_embeddings` is true.
- Weights: bf16, affine or mxfp8 from the global `quantization` block.
- `max_position_embeddings()` returns `0`, so `--max-ctx` is unbounded here.
- `supports_thinking()` is `false`. No speculative seams.

Snapshot in use: `mlx-community__jinaai-ReaderLM-v2`.

---

## Qwen3

`Qwen3ForCausalLM`. Flat `config.json`; `head_dim` is read from it and falls
back to `hidden_size / num_attention_heads`.

- Per-head q/k RMSNorm (`[head_dim]` weights) before RoPE. No projection
  bias.
- Float parameters (norm weights, quant scales and biases) are cast to bf16 at
  load, so a checkpoint that ships fp16 params keeps a bf16 stream and a bf16
  `--kv-quant none` cache. See `docs/KV_LAYER_POLICY.md` "Qwen3 dense KV is
  bf16 at `--kv-quant none`".
- `supports_thinking()` is `true`: the server splits `<think>…</think>` into
  `reasoning_content`.
- Weights: bf16, affine, mxfp8. `prism-ml__Ternary-Bonsai-8B-mlx-2bit` is
  affine `g128 b2`.

**YaRN.** Qwen3 is the only architecture that implements RoPE scaling
(`context_limits()`). A config `rope_scaling` with `rope_type = "yarn"` is
applied at load. Ternary-Bonsai-8B declares factor 4 over 16384.
`--yarn-factor` / `--yarn-original-max` override the config's scaling; the
original window defaults to the config's `original_max_position_embeddings`,
then `max_position_embeddings`. A checkpoint's own `beta_fast` / `beta_slow`
are kept. The paper defaults (32, 1) apply only when the config declares no
`rope_scaling`.

No speculative seams. The `head_budget` and `softmax_mass` recipes of
`rmlx kv-calibrate` accept only this architecture.

---

## Qwen3.5

`Qwen3_5MoeForConditionalGeneration` and `Qwen3_5ForConditionalGeneration`
both build `Architecture::Qwen3_5Moe`. Config fields sit under `text_config`.

**Dispatch on checkpoint facts.** `ModelConfig::is_paroquant()`
(`quantization_config.quant_method == "paroquant"`) selects the PARO loader.
Every other checkpoint takes the standard loader, which:

- finds the tensor prefix (`language_model.model` or `model.language_model`)
  by probing shard headers for the embedding;
- builds each layer's MLP from the tensors present: `mlp.switch_mlp.*` is a
  sparse MoE, `mlp.{gate,up,down}_proj` a dense SwiGLU.

**Resolved class.** `arch_class()` asks `has_sparse_moe_layers()`. A dense
checkpoint reports `Qwen3_5ForConditionalGeneration` whatever it declares, and
a declared-dense checkpoint that ships `switch_mlp` reports the MoE class. The
K-codec guard keys on this; see `docs/KV_LAYER_POLICY.md` § "What the guard
keys off".

**Hybrid stack.** Every `full_attention_interval`-th layer (default 4) is full
attention. The rest are GatedDeltaNet (GDN) linear-attention layers with
`linear_attn.*` weights.

- `needs_lin_caches()` is `true`: a `LinearAttnCache` per GDN layer sits beside
  the `KvCache` stack. The recurrent state has no sequence axis, so
  speculative rollback refolds it from a round tape (`docs/SPECULATIVE.md`
  § "One rollback, and the tape it rebuilds from").
- Prefill and decode both run the `gated_delta_step_gpu` Metal kernel. The
  ops-graph reference (`gated_delta_prefill_ops`) is test-only.
- Full-attention RoPE is partial (`rope_parameters.partial_rotary_factor`,
  default 0.25; base default 10 000 000). No YaRN.
- MoE layers route `num_experts_per_tok` of `num_experts`, plus a shared
  expert. `num_experts` defaults to `0`, which marks a dense checkpoint.
- At prefill, when `n_tokens × top_k ≥ 64`, expert indices are sorted and the
  gathered matmuls run with `sorted_indices = true` (mlx-lm `SwitchGLU`).
  Decode keeps the broadcast path.

**Weights.** bf16, affine, mxfp8, mxfp4 and mixed checkpoints (global mxfp8
with affine overrides on the router gates). `load_util::bf16_param` casts
every float parameter to bf16 at load, GDN `conv1d_weight` and `norm_weight`
included. `load_util::bf16_scales` casts quant scales only when they are
float, so uint8 E8M0 microscaling scales stay verbatim. See
`docs/KV_LAYER_POLICY.md` "Qwen3.6 MoE KV is bf16 at `--kv-quant none`".

`prism-ml__Bonsai-27B-mlx-1bit` declares `quantization.bits = 1` and is
refused by the preflight. Its 2-bit sibling loads.

**KV.** The Qwen MoE guard (`validate_resolved`) refuses every codec with
K below 8 bits on the MoE class. The dense class keeps them. See
`docs/KV_CACHE.md` §5.4.

**Other.** `supports_thinking()` is `true`. Text only. It is the verifier for
every drafter kind; see
[Speculative verifier seams](#speculative-verifier-seams).

Snapshots in use: `mlx-community__Qwen3.6-35B-A3B-8bit` (MoE),
`mlx-community__Qwen3.8-27B-mxfp8` (dense), `z-lab__Qwen3.6-27B-PARO`.

---

## Qwen3-VL MoE

`Qwen3VLMoeForConditionalGeneration`. `text_config` and `vision_config`.

**Text decoder.** A plain Qwen3-MoE GQA stack: per-head q/k RMSNorm, MoE
every `decoder_sparse_step` layers, dense MLP on `mlp_only_layers`. No GDN;
`needs_lin_caches()` is `false`.

**Vision tower.** LayerNorm, GELU-tanh MLP, interpolated learned position
embeddings, spatial patch merging. `deepstack_visual_indexes` names ViT layers
whose outputs are added at the image positions after the matching decoder
layers.

**3D M-RoPE.** `rope_scaling.mrope_section` splits `head_dim / 2` into
temporal, height and width sections. Image patches get 2D positions; text
tokens get 1D positions.

**Long image prompts.** Native tiling produces thousands of soft tokens per
large image. Three passes keep each Metal command buffer under the watchdog:

- the quantized embedding lookup is `take` plus `dequantize`, `O(seq)`;
- the vision tower tiles the query dimension of its full attention, so each
  query still attends to every key;
- the image prefill is chunked at 512 tokens.

The KV ring is sized from `--max-ctx`, so serve a large image with a
`--max-ctx` at least the soft-token count plus the text.

**KV.** `auto` is bf16, as everywhere. The Qwen MoE K guard applies to this
class too. An all-dense Qwen3-VL checkpoint still reports this class, since no
dense Qwen3-VL class is registered.

**Other.** `supports_thinking()` is `false`. The config carries video fields;
no video path exists. No speculative seams.

Snapshot in use: `mlx-community__Qwen3-VL-30B-A3B-Instruct-4bit`.

---

## Gemma3

`Gemma3ForConditionalGeneration`. `text_config` and `vision_config`.

- **SWA and full attention.** `layer_types` sets each layer's attention type;
  without it, every `sliding_window_pattern`-th layer (the config also spells
  it `_sliding_window_pattern`) is full attention. SWA layers use
  `rope_local_base_freq`, full layers `rope_theta`.
- The attention scale is `query_pre_attn_scalar^-0.5`.
- `final_logit_softcapping` is optional.
- No cross-layer KV sharing.
- **Vision.** A SigLIP tower. `mm_tokens_per_image` soft tokens per image
  (256) after AvgPool2d, projected to the text width and scattered into the
  embeddings.
- `max_position_embeddings()` returns `0`, so `--max-ctx` is unbounded here.
  The config's `rope_scaling` is not read.
- SWA layers run the bf16 ring under every codec; only full-attention layers
  are quantized (`docs/KV_CACHE.md` §5.7).
- No speculative seams.

Snapshot in use: `mlx-community__medgemma-1.5-4b-it-8bit`.

---

## Gemma4

`Gemma4ForConditionalGeneration`, and the 12B
`Gemma4UnifiedForConditionalGeneration` alias. `text_config`,
`vision_config`, and on some sizes `audio_config`.

`ModelConfig::is_paroquant()` selects `gemma4::load_from_path_paro`; other
checkpoints take `gemma4::load_from_path`.

### Snapshots

Read from each snapshot's `config.json`:

| Snapshot | Layers | Hidden | Shared-KV layers | `attention_k_eq_v` | MoE | Window | Input |
|---|---|---|---|---|---|---|---|
| `gemma-4-e2b-it-mxfp8` | 35 | 1536 | 20 | no | no | 512 | text, image, audio |
| `gemma-4-e4b-it-mxfp8` | 42 | 2560 | 18 | no | no | 512 | text, image, audio |
| `gemma-4-26b-a4b-it-mxfp8` | 30 | 2816 | 0 | yes | yes (128) | 1024 | text, image |
| `gemma-4-31b-it-mxfp8` | 60 | 5376 | 0 | yes | no | 1024 | text, image |
| `gemma-4-12B-it-*` (Unified) | 48 | 3840 | 0 | yes | no | 1024 | text, image, audio |

`head_dim` is 256 and `global_head_dim` 512 on every size.

### Layers

**SWA and full attention.** `layer_types` is the only authority for a layer's
attention type. The period differs between sizes, so derive the full-attention
indices from the array. SWA layers use the bf16 rotating ring.

**Attention geometry splits by layer class — both axes.** Both the head width
and the KV head count switch with the layer class:

| axis | SWA layers | full-attention layers |
|---|---|---|
| head width | `head_dim` | `global_head_dim` |
| KV heads | `num_key_value_heads` | `num_global_key_value_heads` |

`gemma4/loader.rs` selects the width and `gemma4/generate/mod.rs` builds the
per-layer `KvLayerShape` from the same pair. Without
`num_global_key_value_heads` (e2b, e4b) the full-attention count falls back to
`num_key_value_heads`. Sizing a global layer from `head_dim` or
`num_key_value_heads` reads the SWA geometry. Partial rotary applies to
`global_head_dim`. No layer is 128 wide, so none reaches MLX's fused attention
kernel; see `docs/FFI.md` §`scaled_dot_product_attention`.

**Cross-layer KV sharing.** The last `num_kv_shared_layers` layers project no
K/V. Each attends over the last non-shared layer of its own attention type
(`build_previous_kvs`), through `KvCache::update_and_sdpa_shared_source`.
`SHARES_KV_ACROSS_LAYERS` is `true` for Gemma4 only. It decides whether
`Mixed` and `RotK` keep the bf16 mirror the consumer layers read.

**K equals V.** With `attention_k_eq_v`, full-attention layers ship `k_proj`
only and the loader reuses it as `v_proj`.

**Sparse MoE (26B).** `enable_moe_block` runs a dense MLP and the routed
experts in every layer. Prefill sorts expert indices when
`n_tokens × top_k ≥ 64`, as on Qwen3.5.

**Per-layer input.** A non-zero `hidden_size_per_layer_input` enables the
per-layer input gate.

**Stream dtype.** The activation stream stays at the model dtype. The
embed scale, the per-layer-input scales and the fused GeGLU constants adopt the
operand dtype, so `--kv-quant none` stores bf16 global K/V on mxfp8. See
`docs/KV_LAYER_POLICY.md` "Gemma4 global KV is bf16 at `--kv-quant none`".

**Final logit softcapping.** `final_logit_softcapping`, default 30.

### Weights

mxfp8 (group 32), bf16, affine at any supported width (the Google QAT
snapshots ship affine int4 with `.biases`), mxfp4 and nvfp4. Per-tensor
overrides (router weights, the 8-bit MLP blocks of the 12B QAT snapshots) come
from the inline `quantization` dict.

The 12B QAT snapshots emit a filler token on a bare prompt with no turn
markers; `mlx-lm` does the same. The `--probe-smoke` seed is therefore
templated (`docs/CLI.md` `info`).

### Image and audio input (SigLIP and Conformer)

**Vision.** A SigLIP-style ViT, a pooler and a soft-token scatter. The
per-image block (`<boi>`, the soft tokens, `<eoi>`) is spliced inside the last
user turn, right after the opener `<|turn>user\n` (`[105, 2364, 107]`), as the
HF processor does. Without that opener it falls back to after BOS.

**Audio.** e2b and e4b ship a Conformer `audio_tower`: SSCP subsampling,
Macaron FFW blocks, chunked local attention, an optional output projection.
`input_audio` parts are decoded to 16 kHz mono, run through the USM log-mel
front end and the tower, and scattered at the `<|audio|>` positions. The
soft-token count follows the SSCP downsample. A model without an audio tower
refuses `input_audio` with "no audio tower". See `docs/SERVER.md`
§ "Multimodal content parts".

#### e4b QAT checkpoints — complex-image vision quality

The e4b QAT snapshots (`*-qat-bf16`, `*-qat-mxfp4`, `*-qat-nvfp4`,
`*-qat-4bit`) share the vision tower and the clipped-linear bounds of
`e4b-it-mxfp8`. Only the language weights and the multimodal projection
differ. On simple images they transcribe correctly. On dense, high-patch-count
images every QAT variant hallucinates, `qat-bf16` included, while
`e4b-it-mxfp8` reads them. The `mlx_vlm` reference fails the same way. It is a
checkpoint property, not a codec defect. Use `e4b-it-mxfp8` for complex-image
OCR.

### Unified (encoder-free) vision

The 12B ships no `vision_tower`. `is_unified_arch` selects
`crates/rmlx-models/src/gemma4/vision/unified.rs`; the text decoder is shared.
Per image, ported from HF `gemma4_unified`:

1. Resize to a multiple of 48 and rescale to `[0, 1]`, no normalisation.
2. Patchify into 16-px patches, then merge each `3 × 3` group into one
   contiguous 48 × 48 model patch (`patch_dim = 6912`).
3. `patch_ln1` (LayerNorm), `patch_dense` (quantized Linear 6912 → 3840),
   `patch_ln2`.
4. Add the factorised 2D position embedding
   (`pos_embedding[x, 0] + pos_embedding[y, 1]`), then `pos_norm`.
5. `embed_vision`: `RMSNormNoScale`, then the projection to the text width.
6. Scatter at the image-token run (`build_unified_inputs_embeds`).

The three LayerNorms carry weight and bias and use `eps = 1e-5`.
`rms_norm_eps` governs only the `embed_vision` norm.

**Bidirectional image attention.** Every soft token of an image attends to
every other soft token of the same image; text stays causal.
`build_vision_bidi_overlay` opens the intra-image block and merges it into each
layer's mask. The encoder-free path needs it: its patches carry no
pre-integrated context. The overlay also applies on the SigLIP path.

Two limits are inherent to the encoder-free projection. Fine OCR is weaker
than the SigLIP tower. Achromatic inputs are indistinguishable, because
`patch_ln1` normalises away the level of a grey pixel.

### Unified (encoder-free) audio

The 12B ships no `audio_tower` either. `is_unified_arch` selects
`crates/rmlx-models/src/gemma4/audio/unified.rs` before the Conformer loader.
The snapshot carries only `embed_audio.embedding_projection` (640 → 3840).

1. Decode and resample to 16 kHz mono (`rmlx-audio`).
2. `extract_waveform_frames`: zero-pad to a multiple of
   `audio_samples_per_token` (640) and reshape to `[num_tokens, 640]`. No mel,
   no windowing. `num_soft_tokens = ceil(num_samples / 640)`.
3. `embed_audio`: `RMSNormNoScale`, then the projection.
4. Scatter at the `<|audio|>` run (`build_unified_audio_inputs_embeds`).

The loader checks `output_proj_dims` against the projection's input width.
A request carrying both an image and audio is refused.

### Speculative decoding

Gemma4 implements the verifier seams the assistant drafter reads,
`forward_hidden_states_shared_kv` among them. The drafter is a dedicated
`*-it-assistant-bf16` snapshot. A plain `Gemma4ForConditionalGeneration` has
no MTP head: it drafts as `two_model`, and `--draft-kind mtp` refuses it. See
`docs/SPECULATIVE.md` § "`mtp` dispatch".

---

## Laguna

`LagunaForCausalLM`. Flat `config.json` with per-tensor quant overrides.

- `mlp_layer_types` sets dense or sparse per layer; the default is one dense
  layer, then sparse. Sparse layers carry a shared expert and scale the router
  output by `moe_routed_scaling_factor`.
- `num_attention_heads_per_layer` may set a per-layer head count.
- `layer_types` sets SWA or full attention; without it, every layer is full.
- Per-head q/k norms. Partial RoPE on full-attention layers.
- Attention output is gated per head by `softplus(g_proj(x))`.
- `max_position_embeddings()` returns `0`, so `--max-ctx` is unbounded here.
- No speculative seams.

Laguna is out of scope for benchmarks and optimisation (`CLAUDE.md`).

Snapshot in use: `mlx-community__Laguna-XS.2-mxfp8`.

---

## BitNet b1.58

`BitNetForCausalLM`. Ternary weights `{-1, 0, +1}` packed four per byte.

- Each linear weight is U8 `[N/4, K]` with a BF16 `weight_scale`. The loader
  unpacks and scales it to a BF16 `[N, K]` matrix, so inference is a plain
  BF16 matmul. Encoding: `docs/WEIGHT_QUANTS.md` § "Ternary / BitLinear".
- `attn_sub_norm` sits between the attention output and `o_proj`;
  `ffn_sub_norm` sits between `relu2(gate) * up` and `down_proj`.
- `relu2` activation: `max(x, 0)^2`.
- `tie_word_embeddings` must be true: the embedding is the LM head.
- `head_dim` is `hidden_size / num_attention_heads`. Full RoPE, no scaling.
- `max_position_embeddings` is required and is the positional capacity.
- `scripts/perf_ceiling.py` cannot size this model: it reads the packed `u8`
  headers, while the loader holds BF16. See `docs/PERF_BASELINE.md`
  "What the census cannot size".

`mlx-community__bitnet-b1.58-2B-4T` is a base model with no instruct tune, so
chat output without a system prompt is repetitive.

---

## Jina V4

`JinaEmbeddingsV4Model` is an encoder. It has no `Architecture` variant;
`load_model` refuses it and points at `/v1/embeddings`.

- Text backbone: Qwen2.5-VL, bf16, no `quantization` block.
- Vision: a ViT with windowed attention and full attention at
  `fullatt_block_indexes`. Its MLP and `attn.proj` carry biases.
- 3D M-RoPE for image patches; text uses 1D positions.
- Three LoRA task adapters (`retrieval`, `text-matching`, `code`), swapped per
  request.
- Matryoshka truncation to the config's `matryoshka_dims`.
- Single-vector output is a mean pool plus L2 norm; multi-vector output is a
  per-token 128-wide projection.

The request surface is in `docs/SERVER.md` § "Embeddings".

---

## Speculative verifier seams

The drafters read these `Architecture` methods. Any other architecture returns
an error naming the method.

| Method | Gemma4 | Qwen3.5 |
|---|---|---|
| `forward_seq_last_k_with_cache`, `forward_arr_with_cache` | yes | yes |
| `apply_final_norm`, `logits_from_hidden`, `logits_from_final_hidden` | yes | yes |
| `forward_hidden_states` | yes | no |
| `forward_hidden_states_shared_kv`, `embed_token_raw` | yes | no |
| `forward_hidden_states_multi` | no | yes |
| `forward_verify_capture`, `forward_verify_capture_hot`, `forward_verify_capture_chunked` | no | yes |
| `embed_tokens_raw`, `hot_logits_from_final_hidden` | no | yes |

Drafter kinds per verifier:

| Verifier | Drafters |
|---|---|
| Gemma4 | assistant (`mtp`), `two_model` |
| Qwen3.5 | MTP sidecar (`mtp`), `dflash`, `dflash2`, `eagle3`, `two_model` |

The drafters themselves, their loaders and refusals are in
`docs/SPECULATIVE.md`. Accept rate and decode rate depend on the pair and the
prompt; `docs/SPECULATIVE.md` § "Reading a run" says how they are measured.

---

## Whisper (audio STT)

Whisper serves `/v1/audio/transcriptions`, `/v1/audio/translations` and
`rmlx transcribe`. The snapshot is `mlx-community__whisper-large-v3-mlx`
(`.npz` weights). It ships no tokenizer: pass a directory with the
`openai/whisper-large-v3` `tokenizer.json` through `--whisper-tokenizer-path`
(`RMLX_WHISPER_TOKENIZER_PATH`), or `--tokenizer` to `rmlx transcribe`.

**Token layout (large-v3).** 100 language slots, `<|en|>` = 50259 to
`<|yue|>` = 50358. The task and control tokens follow:
`<|translate|>` = 50359, `<|transcribe|>` = 50360, `<|startoflm|>` = 50361,
`<|startofprev|>` = 50362, `<|nospeech|>` = 50363,
`<|notimestamps|>` = 50364. Timestamps run 50365 to 51865 in 0.02 s steps
(`crates/rmlx-audio/src/tokenizer.rs`).

**Model.** Encoder: two Conv1d stems (the second at stride 2), sinusoidal
positions, attention blocks, a final LayerNorm. Decoder: token and learned
position embeddings, cross-attention blocks, a tied output projection. Sizes
come from `config.json`.

**Inference.** The server and the CLI share
`rmlx_audio::transcribe::Transcriber`:

1. Decode with Symphonia (any container, AAC included), downmix, resample to
   16 kHz.
2. Walk the audio in 30 s windows: log-mel (`MelExtractor`), `encode_mel`,
   then greedy decode in timestamp mode.
3. Each step applies the openai-whisper filter chain: `SuppressBlank`,
   `SuppressTokens` (derived from the loaded tokenizer) and
   `ApplyTimestampRules`.
4. Timestamps become segments. The next window seeks to the last timestamp
   and is conditioned on the previous text through `<|startofprev|>`.
5. Render `txt`, `json`, `srt` or `vtt`.

Decoding is greedy at temperature 0.

**Tests.** `crates/rmlx-audio/tests/transcribe.rs` resolves the snapshot and
tokenizer under `RMLX_O_MODELS_ROOT` and skips when they are absent.
`say_clip_deterministic` synthesises a sentence with `say` and `ffmpeg` and
checks WER and run-to-run identity. `long_form_regression` transcribes every
audio file with a sibling `*.transcript.vtt` in the git-ignored
`crates/rmlx-audio/tests/fixtures/` and requires normalised WER ≤ 0.30.

---

## Silero VAD

Silero VAD v4 (16 kHz branch) is vendored as
`crates/rmlx-audio/assets/silero_vad_16k.safetensors`, converted by
`scripts/convert_silero_vad.py` (licence: `crates/rmlx-audio/assets/NOTICE`).
`rmlx_audio::transcript` uses it to split long audio into voiced chunks. No
server route and no CLI command calls that module; both use the 30 s window
walk above.

---

## Qwen3-TTS

`Qwen3TTSForConditionalGeneration` serves `POST /v1/audio/speech` and returns
mono 24 kHz audio as `wav` (default) or `pcm`. It needs `--tts-model-path`
(the talker snapshot) and `--tts-tokenizer-path` (the speech-tokenizer
snapshot).

| Component | What it does |
|---|---|
| Talker | A Qwen3 decoder with MRoPE and per-head q/k RMSNorm; emits the first codec group per step. |
| Code predictor | A small Qwen3 stack; predicts codec groups 1–15 from the talker hidden. |
| Codec decoder | Split RVQ (1 semantic + 15 acoustic codebooks), a pre-transformer and a convolutional upsampler to 24 kHz. |

Sizes come from `talker_config` and its `code_predictor_config`. Voices are the
keys of `talker_config.spk_id`.

---

## See also

- `docs/ADDING_A_MODEL.md` — the integration surface for a new architecture.
- `docs/WEIGHT_QUANTS.md` — weight formats and per-tensor overrides.
- `docs/KV_QUANT.md`, `docs/KV_CACHE.md` — KV codecs and their invariants.
- `docs/SPECULATIVE.md` — the drafters and the round loop.
- `docs/SERVER.md` — the routes, multimodal parts and embeddings.
