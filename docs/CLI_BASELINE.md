# `rmlx baseline`

The `baseline` subcommand of the rmlx CLI, split out of [`CLI.md`](CLI.md) for
size: its flags, its summary line, its refusals, its prompt files, its
prompt-length cap and its GPU capture flags.

One generation per process, with EOS stop off. Prints one summary line.
`--record` also writes one `runs.db` row.

```bash
rmlx baseline --model /path/to/snapshot
rmlx baseline --model /path/to/snapshot --prompt-tokens 4096 --max-tokens 128 \
  --record --run-class gate
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | path | required | Model snapshot directory. |
| `--prompt` | path | `crates/rmlx-cli/tests/fixtures/baseline_prompt.txt` | Prompt file; see "Prompt files". Conflicts with `--prompt-tokens`. |
| `--prompt-tokens` | u32 | — | Uses `longctx_<N/1024>k.json` from the prompts directory. |
| `--prompts-dir` | path | the nearest `prompts/` holding `longctx_4k.json`, walking up from the working directory | Prompts directory. Env: `RMLX_PROMPTS_DIR`. |
| `--prompt-label` | string | the file name | Value of the record's `prompt` column. |
| `--device` | `cpu \| gpu` | `gpu` | Device. |
| `--max-tokens` / `--gen-tokens` | u32 | `32` | Tokens to generate. |
| `--max-ctx` / `--ctx-max` | u32 | `min(capacity, 4096)` | Context ceiling; see [`CLI.md`](CLI.md) § "Context ceiling". |
| `--yarn-factor`, `--yarn-original-max` | | — | As on `serve` ([`CLI.md`](CLI.md) § "`serve`"). |
| `--max-prompt-tokens` | usize | the context ceiling | Prompt cap; see "Prompt-length cap". Must be ≥ 1. |
| `--allow-truncate` | bool flag | off | Truncate an over-cap prompt on the GPU instead of refusing it. |
| `--kv-boundary-layers` | `HEAD,TAIL` | `2,8` | See [`CLI.md`](CLI.md) § "KV codec flags". |
| `--label` | string | — | Written to the record's `notes` column. |
| `--record` | bool flag | off | Writes a [`METRICS_DB.md`](METRICS_DB.md) §8.5 record to the buffer and ingests it into `runs.db`. Needs `--run-class`. A record that does not land exits non-zero after the summary line, naming the buffer file left for `rmlx metrics record --replay-pending`. A snapshot whose `config.json` names no weight format (no `quantization`, no ParoQuant `quantization_config`, no bfloat16 or float16 `dtype` / `torch_dtype`) is refused before the run ([`METRICS_DB.md`](METRICS_DB.md) §5.2). |
| `--run-class` | `gate \| ab_arm \| diagnostic` | — | Why the run is recorded. Required with `--record`, refused without it. Only `gate` rows reach `bests` and the gates ([`METRICS_DB.md`](METRICS_DB.md) §8.5). |
| `--git-sha` | string | — | Commit SHA for the record's `git_sha`; the binary never derives one. |
| `--emit-token-ids` | bool flag | off | Adds a `baseline: token_ids=<ids>` line, so an A/B harness can compare token streams. |

Plus the KV codec flags ([`CLI.md`](CLI.md) § "KV codec flags").

```text
baseline: model=<name>  load=<ms>  ttft_ms=<ms>  decode_tps=<n>  overall_tps=<n>
          prefill_tps=<n>  prompt_tokens=<n>  peak_rss=<n>MB
          metal_peak_mb=<n>  metal_gen_alloc_mb=<n>  kv_cache_bytes=<n>
```

- `peak_rss` is host RSS from `ps`.
- `metal_peak_mb` is the Metal allocator's peak during generation, weights
  included. `metal_gen_alloc_mb` is that peak minus what was live at the
  start. Only `metal_gen_alloc_mb` compares between two runs, and its
  start count is read unsettled ([`PROFILING.md`](PROFILING.md) § 9.2). Both
  read `0` with no Metal allocator.
- `kv_cache_bytes` is the filled prefix of the KV cache
  (`KvCache::resident_bytes`, [`METRICS_SCHEMA.md`](METRICS_SCHEMA.md) §4), not an
  allocator peak. When the reported count is zero it reads `n/a`, and the
  record omits the column; the timing row stands.
  `scripts/perf_ab.sh` parses it.

**Refusals.** If the model's KV byte counter did not advance, the generation
ended early without an error. `baseline` then prints no summary, writes no
record and exits non-zero.

## Prompt files

A file that parses as JSON with a non-empty `messages` array is a chat
fixture. Its messages go through the model's `chat_template.jinja` before
tokenizing, as on the HTTP route. So `--prompt-tokens N` measures N content
tokens. Any other file is tokenized as plain text. A chat fixture whose
messages are not `{"role": "<string>", "content": "<string>"}` is an error.

## Prompt-length cap

The cap defaults to the resolved context ceiling. A prompt over the cap:

- `--device cpu`: truncated, with a `warn!`.
- `--device gpu`, default cap, no `--allow-truncate`: an error. Raise
  `--max-ctx` to measure the whole prompt.
- `--device gpu` with `--allow-truncate` or an explicit
  `--max-prompt-tokens`: truncated, with a `warn!`.

An explicit `--max-prompt-tokens` above the ceiling is refused on either
device.

## GPU capture (`metal-capture` builds only)

These flags exist only in a binary built with
`--features rmlx-cli/metal-capture`. See [`PROFILING.md`](PROFILING.md) §5.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--gpu-capture` | path | — | Writes a `.gputrace` of a window of decode steps. Requires `--record --run-class diagnostic`: capture makes every timing false. |
| `--gpu-capture-skip` | u32 | `4` | Decode steps before the window. Requires `--gpu-capture`. |
| `--gpu-capture-steps` | u32 | `8` | Decode steps in the window. Requires `--gpu-capture`. |

The process needs `MTL_CAPTURE_ENABLED=1` at launch; `scripts/gpu_capture.sh`
(`make profile-gputrace`) sets it. These fail before the model loads: no
capture layer, an occupied destination, a missing parent directory, an empty
window. So does a `--max-tokens` below `skip + steps + 2`.
