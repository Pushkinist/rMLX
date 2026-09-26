# rMLX E2E Feature-Proof Test Harness

The harness drives the real `rmlx` binary, case by case, and asserts on its
real output. It is a correctness gate, not a performance gate.

It writes a PASS/FAIL grid (feature × sub-feature) to
`report.{json,md}` under `<temp_dir>/rmlx_e2e_<pid>/e2e/`, a per-run home
outside the checkout, and prints the path at the end of the run.

## Form

- `crates/rmlx-cli/tests/e2e/manifest.toml` lists every case as a `[[case]]`
  row: `id`, `feature`, `subfeature`, `model`, and either `cli` or a serve
  request (`request`, `serve_flags`, `kv_quant`, or `ctk`/`ctv`), plus
  `assert` and `tags`. Speculative rows add `draft_model` and `draft_kind`.
- `crates/rmlx-cli/tests/e2e_harness.rs` is the one `#[ignore]` test. It runs
  `crates/rmlx-cli/tests/e2e/runner.rs` over the manifest and fails when any
  case fails.
- A **CLI** case runs `CARGO_BIN_EXE_rmlx` with the row's arguments (`$MODEL`
  is the resolved snapshot) and asserts on the exit code or stdout.
- A **serve** case spawns `rmlx serve`, waits for `/health`, sends a named
  request fixture over HTTP, and asserts on the response, the metrics or the
  run log. NIAH cases serve with `--max-ctx 16384`. `ctk`/`ctv` rows pass
  `--cache-type-k`/`--cache-type-v` instead of `--kv-quant`.
- `crates/rmlx-cli/tests/e2e/report.rs` writes the grid.
- `crates/rmlx-cli/tests/e2e/golden/<case_id>.json` holds the recorded token
  bytes of each `golden` case.

## Assertion kinds

| kind | passes when | used for |
|---|---|---|
| `golden` | The per-token chosen bytes (OpenAI non-stream logprobs `bytes`) at temperature 0 equal the recorded golden. A missing golden, or `RMLX_E2E_REGEN_GOLDEN=1`, records one and passes. | text core, OpenAI non-stream |
| `contains_coherent` | The output is coherent and contains `expect`. | streaming, Anthropic `/v1/messages` |
| `coherent` | The output is coherent: at least 3 distinct alphanumeric characters, at least two words of mean length above one, no word over 60% of three or more words, not NaN; it contains `expect` when one is given. | K-only and symmetric codecs, short context |
| `niah_retrieval` | The needle is recovered from a haystack of about 8k tokens at temperature 0. | quantized KV at long context |
| `cosine_vs_bf16` | The mean cosine of the per-position top-k logprob distributions against the model's own `none` run is at least `min_cosine` (default 0.99), over positions where the chosen ids agree. A chosen-id divergence in the first 8 tokens fails. | quantized KV fidelity |
| `thinking` | `reasoning_content` is non-empty and the answer (68) appears in the content or the reasoning. | thinking mode |
| `stop_halts` | The same prompt with a `stop` string returns a strictly shorter completion. | stop sequences |
| `exit_code` | The process exits with `expect`. | CLI surface |
| `metric_present` | The process exits 0 and stdout contains `expect` (required, non-empty). | `info --list-cache-types`, `metrics query` |
| `serve_refused` | `rmlx serve` with a KV codec the arch rejects exits with `expect` (default 78, `EX_CONFIG`) within 30 s, before `/health` binds. | per-arch KV refusal |
| `dispatch_fired` | Served with `--log verbose`, the run log holds at least one `update_and_sdpa` span with a resolved `path`, and the output is coherent. | attention dispatch |
| `model_lifecycle` | With `--registry` and `--max-loaded-models`: two models resident at cap 2; at cap 1 the eager preload evicts the first; an unload reports `loaded:false` and a second unload 404; a second `rmlx serve` on the held port exits 11. Without a second model only the single-model legs run. | multi-model lifecycle, claim |
| `byte_identical_restart` | With `--kv-ssd-cache-gb 1 --prompt-cache-slots 1` and one `RMLX_HOME`: a long prompt, an evicting prompt, a restart, the long prompt again. The two completions are byte-identical and `/metrics/cache` reports `ssd_hits ≥ 1`. | SSD KV tier |
| `cache_hit_equivalence` | A multi-block prompt sent twice returns the same `content`, and `/metrics/cache` `block_hits` rises between the two. | prompt cache |
| `image` | The bundled solid-red PNG (`tests/e2e/fixtures/vtest_red.png`) with a colour question gets an answer naming `expect` (default "red"). | image input |
| `tool_call` | The `tool_weather` fixture returns `finish_reason == "tool_calls"` and a call named `expect` (default `get_weather`). | tool calling |
| `spec_decode` | Served with `--draft-model` (and `--draft-kind` when the row names one) under `--log verbose`, one generation logs a `*_generate: done` summary with `accept_rate > 0`, and `expect` appears in `content` or `reasoning_content`. An absent drafter skips. The two-model loop logs `spec_generate_greedy_cached: done`, which this scrape does not match, so the two-model rows fail whenever both models resolve. | speculative decoding |

The dispatch counters of the specialised attention kernels are
process-internal atomics with no HTTP surface. The in-crate dispatch tests
cover them (`crates/rmlx-models/tests/sparse_attn_dispatch.rs`,
`crates/rmlx-kv-quant/tests/rotor_fused_qk_dispatch.rs`); `dispatch_fired`
checks only what the binary exposes.

### Verdicts

`PASS`, `FAIL`, `SKIP` (the model does not resolve), `PENDING` and `XFAIL`.

- A row tagged `phase2` is not run and records `PENDING`.
- A row tagged `xfail` whose failure matches its documented failure mode
  records `XFAIL` and does not fail the suite. Only `stop_halts` has one
  ("stop did not shorten output"). An HTTP error, a non-200 status, a spawn
  failure or a parse error stays `FAIL`. No row is tagged `xfail` now.

## Coverage

The manifest's `feature` column groups the rows:

| feature | what it runs |
|---|---|
| `cli_surface` | `info --list-cache-types`, `info --probe-smoke`, `baseline --record`, `metrics query`, `healthcheck --full` |
| `text_core` | chat non-stream and seed determinism (`golden`), chat stream, Anthropic, multi-turn, `stop` |
| `quant_matrix` | Bonsai at every `kv_quant` listed there, plus `rot_k` / `q4_g64` through `ctk`/`ctv`: `coherent` and `niah_retrieval` on each, `cosine_vs_bf16` on some |
| `quant_matrix_konly` | the K-only and symmetric codecs on Bonsai (tag `konly`): short-context `coherent` only |
| `thinking` | Bonsai `thinking` |
| `broader_models` | Gemma4-e4b, Qwen3.6-35B-A3B, gemma-4-26b-a4b and gemma-4-31b: `golden`, stream and Anthropic coherence, `niah_retrieval` and `cosine_vs_bf16` on a subset of legal codecs, a sliding-window-crossing NIAH on Gemma4, the Qwen3.6 KV refusals (`q4_g128` K, `tsym3`) and Qwen3.6 thinking |
| `ssd_kv_tier`, `prompt_cache` | `byte_identical_restart`, `cache_hit_equivalence` on Bonsai |
| `multi_model`, `attention` | `model_lifecycle`, `dispatch_fired` (k8v4) on Bonsai |
| `speculative` | Qwen3.6 with its DFlash and MTP drafters, Gemma4-e4b drafted by Gemma4-e2b, Qwen3.8-27B drafted by ornith-1.0-9b, all `spec_decode` (the two two-model rows fail, see the kind); `p2_speculative_decode` is a `phase2` row and records PENDING |
| `modalities`, `agent` | Gemma4-e4b `image`; Qwen3.6 and Bonsai `tool_call` |

The K-only and symmetric codecs re-quantize K at every decode step. On
Bonsai they lose coherence at long context, so their rows assert short-context
coherence only.

## How to run

```bash
make e2e
# or directly:
cargo test -p rmlx-cli --test e2e_harness -- --ignored --test-threads=1 --nocapture
```

`--test-threads=1` is required: one MLX process per Mac.
`RMLX_E2E_ONLY=id1,id2` runs only the named case ids.

### Model resolution (data-driven — adding a model needs no code edit)

A row's `model` is a spec, resolved by `resolve_model()` (`runner.rs`). First
hit wins:

1. `RMLX_E2E_MODEL_<SPEC>`, then `RMLX_TEST_MODEL_<SPEC>` (the spec
   upper-cased, non-alphanumerics as `_`), when the path exists. This works
   for a model with no manifest row.
2. The spec itself, when it contains `/` and exists.
3. `RMLX_O_MODELS_ROOT` (default `./models`) joined with the spec. An alias
   maps to its slug first: `BONSAI`, `GEMMA4_E2B`, `GEMMA4_E4B`, `QWEN36`.
   The alias table is closed; a new model comes in as a slug or a path.

A new model therefore needs only a manifest row carrying its slug, or, for one
run, `RMLX_E2E_MODEL_<SLUG>=<path>`. When a model does not resolve, its cases
record `SKIP` and the suite stays green.

### Preflight

Every model-touching case runs a preflight first and tears down the `rmlx
serve` it spawned afterwards. The preflight runs `pkill -f` on `rmlx serve`,
`mlx_lm`, `paroquant` and `omlx`, whoever started them, and removes every
`/tmp/rmlx.*.claim` file.
