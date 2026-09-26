# rMLX

**Rust-native, single-binary [MLX](https://github.com/ml-explore/mlx) inference backend for Apple Silicon.**

[![Release](https://img.shields.io/github/v/release/Pushkinist/rMLX?sort=semver&color=blue)](https://github.com/Pushkinist/rMLX/releases)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](#requirements)
[![Rust](https://img.shields.io/badge/rust-1.95%2B-orange?logo=rust)](#requirements)

A native, no-Python local LLM server for Apple Silicon. It serves
MLX-format models over OpenAI- and Anthropic-compatible HTTP APIs, so any
client of those APIs can use it as a local backend. One
`cargo build --release` artifact; no Python runtime and no GGUF.

## What works

- **Text generation**: OpenAI `/v1/chat/completions` and Anthropic
  `/v1/messages`, streaming (SSE), temperature, top-k/p, penalties, a
  thinking budget, and schema-constrained decoding.
- **Image input** through `image_url` parts (HTTP(S) URL, data URI, file
  path or raw base64): Gemma 4, the encoder-free Gemma 4 unified checkpoint,
  Gemma 3 and Qwen3-VL-MoE.
- **Audio input**: Gemma 4 audio towers take `input_audio` parts. Whisper
  serves `/v1/audio/transcriptions` and `/v1/audio/translations`, and the
  `rmlx transcribe` CLI writes txt, vtt, srt or json. Long audio is walked
  in 30 s windows.
- **Speech output**: `/v1/audio/speech` runs Qwen3-TTS.
- **Embeddings**: `/v1/embeddings`, text and image, with jina-embeddings-v4.
- **Tool calling**: OpenAI `tool_calls` and Anthropic `tool_use`, multi-turn.
  The Qwen XML, Qwen JSON and Gemma call formats are parsed.
- **Multi-model registry**: load on demand, unload on idle, a cap on
  resident models, and a multimodal encoder-output cache keyed per model.
- **Weight formats**: bf16, affine 2, 3, 4, 5, 6 and 8 bits, mxfp4, mxfp8,
  nvfp4, ternary and ParoQuant.
- **KV cache**: 8-bit affine, mixed-width affine, and the rotation families
  TurboQuant, IsoQuant, PlanarQuant and RotorQuant; paged KV; asymmetric
  K/V; an SSD tier. Most codecs decode from a bf16 mirror, so they do not
  shrink the cache;
  [`docs/KV_QUANT.md`](docs/KV_QUANT.md#memory-and-bit-rate-summary) says
  which do.
- **Speculative decoding** (`--draft-kind`): an MTP head (Qwen 3.5 family,
  Gemma 4 assistant), DFlash, DFlash 2, EAGLE-3, or a smaller full model of
  the verifier's family.
- **Prompt caching**: automatic prefix caching with block hashing.

There is no `rmlx convert` command: MLX-to-MLX conversion is not
implemented.

Golden-token tests (temperature 0, exact token ids; `make model-check-full`)
cover these snapshots:

| Family | Snapshot | Arch |
|---|---|---|
| Gemma 4 | `gemma-4-e4b-it-mxfp8` | `Gemma4ForConditionalGeneration` |
| Qwen 3.6 | `Qwen3.6-35B-A3B-8bit` | `Qwen3_5MoeForConditionalGeneration` |
| Bonsai | `Ternary-Bonsai-8B-mlx-2bit` | `Qwen3ForCausalLM` |
| BitNet | `bitnet-b1.58-2B-4T` | `BitNetForCausalLM` |
| MedGemma | `medgemma-1.5-4b-it-8bit` | `Gemma3ForConditionalGeneration` |

[`docs/MODELS.md`](docs/MODELS.md) covers every architecture, including the
Gemma 4 QAT checkpoints and
[their complex-image limit](docs/MODELS.md#e4b-qat-checkpoints--complex-image-vision-quality).

## Performance

The decode anchors of the three test-target models, with the build and the
command that measured them, are in
[`docs/PERF_BASELINE.md`](docs/PERF_BASELINE.md). The same doc names the
scripts that measure a build against them and that run one cell against
`mlx-lm`, oMLX, ParoQuant or llama.cpp.

## Requirements

- **Apple Silicon Mac** (M-series). Metal only — no CUDA / ROCm / x86.
- **Rust** stable (1.95+).
- **MLX + mlx-c** installed locally. rMLX links the stable `mlx-c` C ABI; it
  does not vendor or build MLX itself.

```sh
brew install mlx-c          # provides the MLX + mlx-c libraries
```

The build finds MLX via `brew --prefix` (or `/opt/homebrew/opt/…`) on its own.
Point it elsewhere only if your install is somewhere else:

```sh
export MLX_C_PREFIX="$(brew --prefix mlx-c)"   # dir containing lib/libmlxc.dylib + include/
export MLX_PREFIX="$(brew --prefix mlx)"
```

rMLX is validated against one MLX / mlx-c pair, declared in
`crates/rmlx-mlx/mlx-pin.txt`. `rmlx healthcheck` and `make mlx-preflight`
check it. On M5 and later, `rmlx bench` and `rmlx baseline` refuse to
measure off the pair; serving is not blocked. At startup rMLX warns when the
MLX it loaded differs from the one it was built against. See
[`docs/FFI.md`](docs/FFI.md#pinned-mlx--mlx-c-pair).

### On M5 and later: the Neural Accelerator kernels

Some Homebrew MLX bottles omit the Neural Accelerator (NAX) GEMM kernels.
On M5 and later that slows prefill; decode and output are unaffected. M1 to
M4 have no Neural Accelerator, so there is nothing to check.

At startup, rMLX scans the `mlx.metallib` of the MLX it loaded. It warns
only when the host has a Neural Accelerator and the kernels are missing. To
check by hand:

```sh
strings "$(brew --prefix mlx)/lib/mlx.metallib" | grep -c steel_gemm_fused_nax
# M5+: a non-zero count. 0 means the bottle has no NAX kernels.
```

`make mlx-preflight` runs this and the other MLX checks.

### Extra tooling for kernel work (contributors only)

Not needed to build or run rMLX — only to compile-check or profile the Metal
kernels:

- **Full Xcode** (not just the Command Line Tools), selected with
  `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.
- **The Metal Toolchain component**, which Xcode does not install by default:

```sh
xcodebuild -downloadComponent MetalToolchain
```

With both present, `make check-metal-compiles` compiles every `.metal`
kernel natively, and `make profile-gputrace` captures a GPU trace for Xcode
([`docs/PROFILING.md`](docs/PROFILING.md)). Without them the compile gate
reports `SKIP`; hosted CI runs it strictly.

## Install

Each path builds from source and links the Homebrew MLX and mlx-c, so MLX
must be present (`brew install mlx-c`).

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
brew trust Pushkinist/rmlx   # one-time: Homebrew loads third-party taps only once trusted
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

Or use the Makefile wrapper:

```sh
make build      # cargo build --workspace --release
make ci         # the pre-merge gate
```

## Run

Serve an MLX-format model directory (the `mlx-community` safetensors layout).
The model id is the directory name:

```sh
target/release/rmlx serve --model /path/to/mlx-community__gemma-4-e4b-it-mxfp8 --port 8080
```

Then call it like any OpenAI endpoint:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community__gemma-4-e4b-it-mxfp8",
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
| [`docs/KV_QUANT.md`](docs/KV_QUANT.md) | KV-cache quantization contract; links the per-topic KV docs |
| [`docs/KV_CACHE.md`](docs/KV_CACHE.md) | KV cache architecture |
| [`docs/SPECULATIVE.md`](docs/SPECULATIVE.md) | Speculative decoding (MTP / DFlash / Eagle3) |
| [`docs/PROMPT_CACHE.md`](docs/PROMPT_CACHE.md) | Prompt + automatic prefix caching |
| [`docs/SAMPLING.md`](docs/SAMPLING.md) | Per-token sampling + constrained decoding |
| [`docs/FFI.md`](docs/FFI.md) | rmlx-mlx ↔ mlx-c FFI bridge |
| [`docs/METRICS_DB.md`](docs/METRICS_DB.md) | Metrics DB: ingest, identity + `rmlx metrics` |
| [`docs/METRICS_SCHEMA.md`](docs/METRICS_SCHEMA.md) | Metrics DB schema + metric registry |

`CLAUDE.md` carries the architecture overview and the workspace crate graph.

## Non-goals

- Not a GGUF runtime. rMLX reads MLX-format snapshots only.
- No training, fine-tuning, fusing or LoRA merging.
- No per-request LoRA hot-swap: fuse externally and load the merged snapshot.
- Apple Silicon only — no CUDA, ROCm, or x86 SIMD paths.

## Releasing

The version lives in `[workspace.package].version` in the root `Cargo.toml`.
The release flow is in [`docs/RELEASING.md`](docs/RELEASING.md).

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Sibling projects

rMLX stands on a lot of other people's work: the MLX ecosystem, the
rotation-KV quantization research it ports, and the servers it learned its API
shape from.
Many thanks to:

**MLX foundation**

- [`ml-explore/mlx`](https://github.com/ml-explore/mlx) — the MLX array framework.
- [`ml-explore/mlx-c`](https://github.com/ml-explore/mlx-c) — the stable C ABI rMLX links against.
- [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm) — reference loader + numerics.
- [`oxiglade/mlx-rs`](https://github.com/oxiglade/mlx-rs) — community Rust binding over `mlx-c`.
- [`safetensors/safetensors`](https://github.com/safetensors/safetensors) — the weight format + Rust crate.

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
