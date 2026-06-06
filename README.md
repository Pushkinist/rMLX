# rMLX

**Rust-native, single-binary [MLX](https://github.com/ml-explore/mlx) inference + conversion backend for Apple Silicon.**

No Python at runtime. One `cargo build --release` artifact. The widest weight ×
KV-quantization matrix any MLX server ships, including rotation-based KV families
(TurboQuant, IsoQuant, PlanarQuant, RotorQuant, ParoQuant) that no other MLX
server offers.

> Status: **0.1.0** — first feature-complete native MLX backend. Apple Silicon
> only (Metal). See [What works](#what-works).

## Why

| Pain | rMLX answer |
|---|---|
| `mlx_lm.server` — Python venv juggling, slow startup, no KV rotation | Rust + lifted Metal kernels, instant warm-start, zero Python at runtime |
| Multi-model Python servers — heavy deps, always-on | Single binary, load-on-demand / unload-on-idle lifecycle |
| Experimental quant forks (TurboQuant / PlanarQuant / ParoQuant) live in separate llama.cpp or Python trees | All first-class on one MLX path |

## What works

- **Text generation** — OpenAI-compatible `/v1/chat/completions` and
  `/v1/completions`, plus an Anthropic-compatible surface. Temperature, top-k/p,
  penalties, thinking-budget, constrained / schema-guided decoding.
- **Image input** — vision-capable models (Gemma 4 SigLIP tower, Qwen3-VL-MoE
  deepstack) accept images via `image_url` content parts (data-URI, http, file
  path, or base64).
- **Audio input** — audio transcription / translation endpoints for
  audio-capable models.
- **Embeddings** — `/v1/embeddings`, including multimodal (text + image) jina-v4.
- **Tool / function calling** — OpenAI `tool_calls` and Anthropic `tool_use`,
  multi-turn, multiple emit formats (Qwen XML, Hermes-JSON, Gemma).
- **Quantization** — affine 2–8 bit, mxfp4 / mxfp8, nvfp4, ParoQuant weights;
  KV-cache quant incl. fp8, TurboQuant, RotorQuant, PlanarQuant, IsoQuant,
  paged-KV, mixed / asymmetric K/V, and an SSD KV tier.
- **Speculative decoding** — MTP, DFlash, and Eagle3 drafters.
- **Prompt caching** — automatic prefix caching with block hashing.
- **Conversion** — `rmlx convert` re-quantizes / repacks MLX → MLX.

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

Speculative-decoding drafters are validated against their verifiers via
`--draft-kind mtp`: the Qwen 3.6 MTP sidecar (`Qwen3.6-35B-A3B-MTP-5bit`,
verifier `Qwen3.6-35B-A3B-8bit`) and the Gemma 4 assistant drafter
(`gemma-4-E2B-it-assistant-bf16`, verifier `gemma-4-e2b-it-mxfp8`).

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

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
