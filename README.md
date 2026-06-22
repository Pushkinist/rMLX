# rMLX

**Rust-native, single-binary [MLX](https://github.com/ml-explore/mlx) inference + conversion backend for Apple Silicon.**

[![Release](https://img.shields.io/github/v/release/Pushkinist/rMLX?sort=semver&color=blue)](https://github.com/Pushkinist/rMLX/releases)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](#requirements)
[![Rust](https://img.shields.io/badge/rust-1.95%2B-orange?logo=rust)](#requirements)

A native, no-Python **local LLM server** for Apple Silicon — a drop-in
OpenAI- and Anthropic-compatible alternative to `mlx_lm.server`, and a
Metal-native counterpart to `llama.cpp` that runs MLX-format models directly.
One `cargo build --release` artifact — no Python runtime, no GGUF translation layer — and the widest
weight × KV-quantization matrix any MLX server ships, including rotation-based
KV families (TurboQuant, IsoQuant, PlanarQuant, RotorQuant, ParoQuant) that no
other MLX server offers. Works as a local backend for any OpenAI/Anthropic-compatible
coding agent (Claude Code, Cursor, Aider, OpenCode).

> Status: **feature-complete native MLX backend** — OpenAI- and
> Anthropic-compatible text, tool/function calling, streaming, image + audio
> input, embeddings, and a multi-model registry. Apple Silicon only (Metal).
> Latest release version: see the badge above. See [What works](#what-works).

## Why

| Pain | rMLX answer |
|---|---|
| `mlx_lm.server` — Python venv juggling, slow startup, no KV rotation | Rust + lifted Metal kernels, instant warm-start, zero Python at runtime |
| Multi-model Python servers — heavy deps, always-on | Single binary, load-on-demand / unload-on-idle lifecycle |
| Experimental quant forks (TurboQuant / PlanarQuant / ParoQuant) live in separate llama.cpp or Python trees | All first-class on one MLX path |
| `llama.cpp` on Mac — GGUF conversion + a translation layer, no MLX-native KV quant | Runs MLX-format weights directly on Metal; MLX → MLX re-quant, no GGUF round-trip |

## What works

- **Text generation** — OpenAI-compatible `/v1/chat/completions`, plus an
  Anthropic-compatible `/v1/messages` surface. Streaming (SSE),
  temperature, top-k/p, penalties, thinking-budget, constrained / schema-guided
  decoding.
- **Image input** — vision-capable models accept images via `image_url` content
  parts (data-URI, http, file path, or base64): Gemma 4 SigLIP tower (e4b /
  26b), the encoder-free Gemma 4 12B `gemma4_unified` any-to-any architecture,
  jina-v4, and Qwen3-VL-MoE deepstack.
- **Audio input** — audio-capable models accept audio (Gemma 4 unified Conformer
  tower) plus Whisper speech-to-text via the model-agnostic `rmlx transcribe`
  CLI (txt / vtt / srt / json, long-form chunking).
- **Embeddings** — `/v1/embeddings`, including multimodal (text + image) jina-v4.
- **Tool / function calling** — OpenAI `tool_calls` and Anthropic `tool_use`,
  multi-turn, multiple emit formats (Qwen XML, Hermes-JSON, Gemma).
- **Multi-model registry** — serve many models from one process with
  load-on-demand / unload-on-idle, a bounded resident-model cap, and a shared
  multimodal encoder-output cache (scoped per model).
- **Quantization** — affine 2–8 bit, mxfp4 / mxfp8, nvfp4, ParoQuant weights;
  KV-cache quant incl. fp8, TurboQuant, RotorQuant, PlanarQuant, IsoQuant,
  paged-KV, mixed / asymmetric K/V, and an SSD KV tier.
- **Speculative decoding** — MTP, DFlash, and Eagle3 drafters.
- **Prompt caching** — automatic prefix caching with block hashing.

Conversion (`rmlx convert`, MLX → MLX re-quantize / layout repack) is a roadmap
target and not yet shipped.

Continuously smoke-tested end-to-end. The first four families carry committed
golden-token decode gates (temp=0, exact token-id match); embeddings and the
speculative drafters are validated end-to-end via their serving endpoints.

| Family | Example snapshot(s) | Arch |
|---|---|---|
| Gemma 4 | `gemma-4-e2b/e4b-it-mxfp8`, `gemma-4-26b-a4b-it-mxfp8` (MoE), `gemma-4-31b-it-mxfp8` (dense) | `Gemma4ForConditionalGeneration` |
| Qwen 3.6 | `Qwen3.6-35B-A3B-8bit` | `Qwen3_5MoeForConditionalGeneration` |
| Bonsai | `Ternary-Bonsai-8B-mlx-2bit` | `Qwen3ForCausalLM` |
| BitNet | `bitnet-b1.58-2B-4T` | `BitNetForCausalLM` |
| Embeddings | `jina-embeddings-v4` (text + image) | `JinaEmbeddingsV4Model` |

Google's Gemma 4 **QAT** low-bit checkpoints (`*-qat-4bit` / `-mxfp4` / `-nvfp4`
/ `-bf16`) load and serve text correctly on the same `Gemma4` arch — they need
QAT-specific weight handling (per-group zero-point `.biases`, router/MLP
overrides). One known limit: **e4b QAT complex-image vision is unreliable** — an
intrinsic limitation of the QAT *checkpoint* (the unquantized `qat-bf16` fails
the same way and the `mlx_vlm` reference reproduces it), not an rMLX codec
defect; use `e4b-it-mxfp8` for dense-image OCR. Details:
[`docs/MODELS.md`](docs/MODELS.md#e4b-qat-checkpoints--complex-image-vision-quality).

Speculative-decoding drafters are validated against their verifiers via
`--draft-kind mtp`: the Qwen 3.6 MTP sidecar (`Qwen3.6-35B-A3B-MTP-5bit`,
verifier `Qwen3.6-35B-A3B-8bit`) and the Gemma 4 assistant drafter
(`gemma-4-E2B-it-assistant-bf16`, verifier `gemma-4-e2b-it-mxfp8`).

## How rMLX compares

Other MLX servers are Python (`mlx-lm`, oMLX) or cover a narrower surface;
`llama.cpp` is native but reads GGUF, not MLX. rMLX is the one that is native
Rust **and** native MLX, with both API dialects and the full input-modality +
quantization surface in a single process.

| Capability | rMLX | `mlx-lm` | oMLX | mlxcel | `llama.cpp` |
|---|---|---|---|---|---|
| Language | Rust | Python | Python | Rust | C/C++ |
| Single binary, no Python runtime | ✅ | — | — | ✅ | ✅ |
| Native model format | MLX | MLX | MLX | MLX | GGUF |
| OpenAI API (`/v1/chat/completions`) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anthropic API (`/v1/messages`) | ✅ | — | ✅ | — | — |
| Image input (in-server) | ✅ | —¹ | ✅ | ✅ | ✅ |
| Audio input (in-server) | ✅ | — | — | — | ◐² |
| Embeddings (`/v1/embeddings`) | ✅ | — | ✅ | — | ✅ |
| KV-cache quantization | widest³ | 2 schemes | tiered, not quant⁴ | 1 (TurboQuant) | block types only⁵ |

<sub>¹ `mlx-lm` itself is text; vision lives in the separate `mlx-vlm` package.
² `llama.cpp` audio is in the `mtmd` CLI, not the HTTP server.
³ affine 2–8 bit, fp8, mxfp/nvfp4, **plus** five rotation-KV families
(TurboQuant, IsoQuant, PlanarQuant, RotorQuant, ParoQuant) no other MLX server
ships. ⁴ oMLX has a tiered RAM+SSD KV cache, not KV-bit quantization.
⁵ `llama.cpp` offers per-tensor block KV types (`q8_0`…`q5_1`) but no
rotation-KV families. Competitor cells verified against each project's README /
server docs (2026-06); capabilities evolve — corrections welcome.</sub>

## Performance

Decode throughput is competitive with `mlx-lm` across both lead families,
measured on an independent cross-backend harness (Apple M5 Max, batch=1,
temp=0; per-family grids under [`docs/models/`](docs/models)):

- **Qwen 3.6 35B-A3B** — rMLX leads decode at **every context (4k→128k)**,
  ≈ +12–15 % over `mlx-lm-turboquant` in our runs.
- **Gemma 4 (e2b / e4b / 26b)** — matches `mlx-lm` within run-to-run noise;
  decode there is weight-bandwidth-bound, so KV quant buys little. (31b dense
  trails slightly — bandwidth physics.)

**Prefill / time-to-first-token** is at parity with `mlx-lm`: a direct
`mlx-lm` run on the same 35B-A3B snapshot measures ≈ 2.7k–3.6k prompt tok/s,
versus rMLX's ≈ 3.0k — both bandwidth-bound at roughly the same level. (An
earlier draft cited a ~40–50× prefill deficit; that came from a non-physical
baseline and has been retracted after a direct measurement.)

## Requirements

- **Apple Silicon Mac** (M-series). Metal only — no CUDA / ROCm / x86.
- **Rust** stable (1.95+).
- **MLX + mlx-c** installed locally. rMLX links the stable `mlx-c` C ABI; it does
  not vendor or build MLX itself.

```sh
brew install mlx-c          # provides the MLX + mlx-c libraries
```

If your MLX install is not on the default homebrew cellar path, point the build
at it:

```sh
export MLX_C_PREFIX="$(brew --prefix mlx-c)"   # dir containing lib/libmlxc.dylib + include/
```

## Install

All paths build from source — rMLX links the system MLX/mlx-c libraries, so MLX
must be present (`brew install mlx-c`). The build targets the installing machine's
own chip, so a single method serves every Apple Silicon generation (M1–M5).

**Script** (ensures Rust + MLX, then builds):

```sh
curl -fsSL https://raw.githubusercontent.com/Pushkinist/rMLX/main/install.sh | bash
```

Prefer to inspect first (recommended for any `curl | bash`):

```sh
curl -fsSL https://raw.githubusercontent.com/Pushkinist/rMLX/main/install.sh -o install.sh
less install.sh && bash install.sh
```

**Homebrew** (via tap):

```sh
brew tap Pushkinist/rmlx
brew trust Pushkinist/rmlx   # one-time: Homebrew now requires explicitly trusting third-party taps
brew install rmlx
```

**Cargo**:

```sh
brew install mlx-c
MLX_C_PREFIX="$(brew --prefix mlx-c)" \
  cargo install --git https://github.com/Pushkinist/rMLX --bin rmlx rmlx-cli
```

## Build

For development / from a clone:

```sh
git clone https://github.com/Pushkinist/rMLX
cd rMLX
cp .env.example .env          # set RMLX_O_MODELS_ROOT to your models folder
cargo build --release        # → target/release/rmlx
```

Or use the Makefile wrapper (keeps the local gate identical to CI):

```sh
make build      # cargo build --workspace --release
make ci         # fmt-check + clippy + test + deny + audit (pre-merge gate)
```

## Run

Serve an MLX-format model directory (the `mlx-community` safetensors layout):

```sh
target/release/rmlx serve --model /path/to/mlx-community__gemma-4-e4b-it-mxfp8 --port 8080
```

Then call it like any OpenAI endpoint:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gemma-4-e4b",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "temperature": 0
  }'
```

Inspect a model's architecture + quantization without serving:

```sh
target/release/rmlx info --model /path/to/snapshot
```

See [`docs/CLI.md`](docs/CLI.md) for the full subcommand / flag reference.

## Documentation

| Doc | Topic |
|---|---|
| [`docs/CLI.md`](docs/CLI.md) | CLI subcommands, flags, env vars |
| [`docs/SERVER.md`](docs/SERVER.md) | HTTP server: OpenAI / Anthropic compat, routes, tool calling |
| [`docs/MODELS.md`](docs/MODELS.md) | Per-architecture model reference |
| [`docs/WEIGHT_QUANTS.md`](docs/WEIGHT_QUANTS.md) | Weight quantization formats |
| [`docs/KV_QUANT.md`](docs/KV_QUANT.md) | KV-cache quantization variants |
| [`docs/KV_CACHE.md`](docs/KV_CACHE.md) | KV cache architecture |
| [`docs/SPECULATIVE.md`](docs/SPECULATIVE.md) | Speculative decoding (MTP / DFlash / Eagle3) |
| [`docs/PROMPT_CACHE.md`](docs/PROMPT_CACHE.md) | Prompt + automatic prefix caching |
| [`docs/SAMPLING.md`](docs/SAMPLING.md) | Per-token sampling + constrained decoding |
| [`docs/FFI.md`](docs/FFI.md) | rmlx-mlx ↔ mlx-c FFI bridge |
| [`docs/METRICS_DB.md`](docs/METRICS_DB.md) | Metrics DB schema + `rmlx metrics` |

`CLAUDE.md` carries the architecture overview and the workspace crate graph.

## Non-goals

- Not a GGUF runtime (that is `llama.cpp`'s lane). MLX-format only; rMLX can
  re-quantize / convert MLX → MLX but never reads GGUF.
- No training / fine-tune / fuse / LoRA-merge. Quantization and format
  conversion are in scope; training is not.
- Multi-LoRA hot-swap per request is out of scope — fuse externally and load the
  merged snapshot.
- Apple Silicon only — no CUDA, ROCm, or x86 SIMD paths.

## Releasing

The version lives in exactly one place: `[workspace.package].version` in the
root `Cargo.toml`. Member crates inherit it via `version.workspace = true`, and
internal path deps omit a version (`deny.toml` sets `allow-wildcard-paths`).

1. Bump `version` in `Cargo.toml` `[workspace.package]`.
2. `make ci` green.
3. `make tag` — derives `v<version>` from `Cargo.toml`, creates the annotated
   tag.
4. `git push origin v<version>`, then cut the GitHub release from the tag.

This README is **not** version-bumped per release: the badge above tracks
GitHub releases automatically, and the Status line carries no version number.
Edit `README.md` only when capabilities materially change (a new modality,
architecture family, or endpoint). The full release flow — changelog, signing,
Homebrew bottle, tap — lives in [`docs/RELEASING.md`](docs/RELEASING.md).

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Sibling projects

rMLX stands on a lot of other people's work — the MLX ecosystem, the rotation-KV
quantization research it ports, and the servers it learned its API shape from.
Many thanks to:

**MLX foundation**

- [`ml-explore/mlx`](https://github.com/ml-explore/mlx) — the MLX array framework.
- [`ml-explore/mlx-c`](https://github.com/ml-explore/mlx-c) — the stable C ABI rMLX links against.
- [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm) — reference loader + numerics.
- [`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs) — community Rust binding over `mlx-c`.
- [`huggingface/safetensors`](https://github.com/huggingface/safetensors) — the weight format + Rust crate.

**KV / weight quantization research**

- [`aivrar/multi-turboquant`](https://github.com/aivrar/multi-turboquant) — TurboQuant KV toolkit.
- [`scrya-com/rotorquant`](https://github.com/scrya-com/rotorquant) — RotorQuant.
- [`ParaMind2025/isoquant`](https://github.com/ParaMind2025/isoquant) — IsoQuant / PlanarQuant.
- [`z-lab/paroquant`](https://github.com/z-lab/paroquant) — ParoQuant weight rotation.
- [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) — TurboQuant KV Metal kernels (llama.cpp).
- [`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus) — TurboQuant+ KV cache + multi-axis fidelity scoring.

**Servers & multimodal**

- [`Blaizzy/mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) — vision-language reference.
- [`jundot/omlx`](https://github.com/jundot/omlx) — multi-model MLX server (API-shape reference).
- [`EricLBuehler/mistral.rs`](https://github.com/EricLBuehler/mistral.rs) — fast, flexible Rust LLM inference engine.
- [`ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo) — NVIDIA datacenter-scale distributed inference framework.
