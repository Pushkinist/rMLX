# scripts/ — index

Every script in this tree, one line each. **Add a row here when you add a
script.** An unindexed script is invisible: the next person writes a second one
that does the same thing, and neither is maintained.

Conventions:

* `make <target>` in the "via" column means the Makefile is the supported entry
  point — call it that way so the local gate and CI stay identical.
* "gate" scripts are hard-fail steps of `make ci`. They exist to be run by CI,
  not by hand, but each is runnable standalone for triage.
* Shared helpers live in `lib/` and are **sourced, not executed**.

---

## CI gates (`make ci`)

| Script | What it enforces |
|---|---|
| `check_eval_lock.sh` | Every MLX eval FFI call is made under the process-wide evaluation lock (25-symbol reach-set). |
| `check_eval_lock_fixtures.sh` | Recall test for the above — 26 synthetic scan roots, each asserting which rule fired. |
| `check_gpu_tests_ignored.sh` | A test that reaches `Device::Gpu` carries `#[ignore]`; an `#[ignore]` claiming Metal is reachable or declared. |
| `check_gpu_tests_ignored_fixtures.sh` | Recall test for the above, asserting each case's failure *reason*. |
| `check_kernel_dtype_contract.sh` | A custom-Metal-kernel dispatcher returns the caller's dtype, not the kernel's working precision. |
| `check_kernel_dtype_contract_fixtures.sh` | Recall test for the above. |
| `check_kv_layer_quants.sh` | The per-layer KV codec vector has one producer; every per-layer cache stack uses it or declares itself uniform. |
| `check_metal_compiles.sh` | Every `.metal` kernel compiles natively at `-std=metal3.0` and `-std=metal4.0`, and is named by its `probes/kernels.manifest`. |
| `check_metal_format.sh` | Every `.metal` file is formatted. |
| `check_no_decode_swallow.sh` | A failed decode step or failed sampler call cannot be swallowed into a silent success. |
| `check_no_inline_tests.sh` | No inline `#[cfg(test)] mod tests` outside `*_tests.rs` / `tests.rs`. |
| `check_no_kernel_input_eval.sh` | No blocking `Array::eval()` on a kernel input inside a dispatcher. |
| `check_no_kernel_input_eval_fixtures.sh` | Recall test for the above. |
| `check_no_scalar_f32_leak.sh` | No unguarded `scalar_f32(` in the metal-owning crates — the f32 decode-graph promotion class. |
| `metal_dirs.sh` | The list of directories holding gated `.metal` kernels. Sourced by the metal gates. |
| `file_size_report.sh` | Advisory (non-failing) LOC report for source files >1000 lines. |
| `target_size_report.sh` | Advisory (non-failing) `target/` size report. |

## Test execution

| Script | Via | What it does |
|---|---|---|
| `run_gpu_tests.sh` | `make gpu-test` | Runs the `#[ignore]` Metal tests per member crate, `--test-threads=1`. |
| `eval_lock_stress.sh` | `make eval-lock-stress` | Drives the evaluation-lock reproducer across N fresh processes. Deliberately out of `make ci`. |
| `schema_constraint_canary.sh` | — | Real-model proof for the `json_schema` constrained-decoding path. |
| `ssd_canary.sh` | — | End-to-end long-session SSD prompt-cache tier canary (see `docs/SSD_CANARY.md`). |
| `release_e2e/stage6_perf/codec_smoke_runner.sh` | `make smoke-codec-matrix` | KV-codec smoke + NIAH gate matrix. |
| `release_e2e/stage6_perf/niah_long_context.sh` | — | Needle-in-a-haystack long-context validation driver. |
| `parity/rmlx-vs-fork.sh` | — | Parity gate: does rMLX agree with the mlx-lm-turboquant fork. |
| `parity/jina_v4_parity.py` | — | jina-v4 embedding parity against the reference implementation. |

## Perf: A/B and regression

| Script | Via | What it does |
|---|---|---|
| `perf_ab.sh` | `perf_canary.sh --ab` | **ABBA-interleaved A/B of two `rmlx baseline` arms.** Host-quiescence gate, arm-distinguishability guard, token-id comparison, per-arm `metal_gen_alloc_mb` + resident `kv_cache_bytes`. Never writes `runs.db` — promote a result with `ingest/perf_ab_ingest.py`. |
| `perf_ab_selftest.sh` | `make canary-ab-selftest` | Mutation check for `perf_ab.sh` — every guard must fail when broken. |
| `perf_ab_ingest_selftest.sh` | `make canary-ab-ingest-selftest` | Mutation check for `ingest/perf_ab_ingest.py` — 15 cases over synthetic result files, one per refusal. Never writes `runs.db`. In `make ci`. |
| `bench_llama_ab_selftest.sh` | `make llama-ab-selftest` | Mutation check for `bench_llama_ab.sh` against a stub `llama-server` — 13 cases, one per guard. In `make ci`. |
| `bench_llama_ab.sh` | — | **ABBA-interleaved A/B of two `llama-server` arms** (fork vs upstream, codec vs codec). Same quiescence discipline as `perf_ab.sh`, reported over the server's own `timings` plus KV-buffer and peak-RSS columns. Never writes `runs.db`. |
| `perf_canary.sh` | `make perf-canary` | Fast decode-TPS canary over the three standard test-target models. |
| `regression_gate.sh` | — | Compare a committed baseline against the latest canary row. Exit 125 = `git bisect skip`, 1 = regression. |
| `perf-iter/bench_decode_tps.sh` | — | Per-iteration regression bench for a perf-fix campaign. |
| `perf-iter/diff_baseline.sh` | — | Compare two perf-iter JSONL files, emit per-cell deltas. |
| `perf_ceiling.py` | — | Static roofline calculator: bytes/step and the theoretical ceiling from a snapshot's `config.json` + safetensors index. |
| `sdpa_headdim_bench.py` | — | What MLX's SDPA dispatch costs as a function of `head_dim`. |
| `aggregate_decode_profile.py` | — | Aggregate per-model `decode_profile` lines from `profile_<MODEL>.txt`. |

## Cross-backend bench cells

| Script | What it does |
|---|---|
| `bench_cell.sh` | Per-cell driver for the cross-backend bench. Legacy mode drives the CBB Python harness over an HTTP backend; cache-type mode drives `rmlx baseline` directly and emits one §8.5 RunRecord. |
| `bench_codec_cell.sh` | Single-codec × single-model bench runner. |
| `bench_cache_types.sh` | Drive the cache-type combo matrix for one model. |
| `bench-records-sweep.sh` | 5-model × 4-KV-quant `BENCHMARK_CHAMPIONS` regression sweep. |
| `spec_bench.sh` | Bench a model in normal vs MTP speculative-decode mode. |
| `baseline/run_mlx-lm.sh` | Baseline measurement via Apple's stock `mlx-lm` loader. |
| `baseline/run_mlx-lm-turboquant.sh` | Baseline measurement via the `mlx-lm-turboquant` fork. |
| `baseline/run_oMLX.sh` | Baseline measurement via the oMLX Python server. |
| `baseline/group-A-baseline.sh` | Measure the rMLX baseline TPS for the Group-A regression gate. |
| `baseline/c1-gemma4-cold-equal.sh` | C1 acceptance: gemma4 partial-prefix reuse. |
| `baseline/d8-phase1-measure.sh` | Quantify the first-dispatch MSL-compile tax. |
| `autoresearch_run.sh` | Single autoresearch experiment run. |

## Metrics ingest

| Script | What it does |
|---|---|
| `ingest/llama_bench_ingest.py` | Convert `llama-bench -o json` rows into the §8.5 universal RunRecord and ingest them. |
| `ingest/llama_ab_ingest.py` | Promote one accepted `bench_llama_ab.sh` result into two §8.5 RunRecords (one per arm). Refuses a TAINTED run unless told otherwise. |
| `ingest/perf_ab_ingest.py` | Promote one accepted `perf_ab.sh` result into two §8.5 RunRecords (one per arm), carrying `decode_tps_warm` + `kv_cache_bytes`. Identity comes from the measured binary and is digest-checked against the run; refuses a TAINTED run, a weakened interference gate, or a cell key that disagrees with the measurement. |
| `lib/identity.sh` | Shared §8.5 run-identity (`rmlx metrics identity --json`) for bench scripts. **Source it.** |
| `lib/prefill_ms.py` | Read `decode_profile{prefill_ms}` back out of an rmlx run log. |

## Profiling / GPU capture

| Script | Via | What it does |
|---|---|---|
| `gpu_capture.sh` | `make gpu-capture` | Capture a Metal GPU trace of a bounded steady-state decode window. |
| `gputrace_preflight.sh` | `make gputrace-preflight` | Check host prerequisites for Apple's GPU tools against a Metal binary. |
| `gputrace_kernels.sh` | — | List the Metal functions a captured window referenced. |
| `gputrace_diff.sh` | — | Diff the function sets of two `.gputrace` bundles. |
| `gputrace_summary.sh` | — | Summarise a `.gputrace` bundle. |
| `mst_capture.sh` | `make mst-capture` | Record a Metal System Trace of a live `rmlx` run and export the GPU-interval table. |
| `traces_gc.sh` | `make traces-gc` | Bound how much disk `.rmlx/traces` holds. |
| `rmlx-capture.entitlements` | — | Entitlements plist for codesigning a capture-enabled binary. |

## Host / environment

| Script | Via | What it does |
|---|---|---|
| `mlx_preflight.sh` | `make mlx-preflight` | Verify the linked MLX stack is sane before benching. |
| `mlx_restore_pin.sh` | `make mlx-restore-pin` | Restore the pinned, nax-capable MLX pair. |
| `target_gc.sh` | `make target-gc` | Prune stale build profiles from `target/`. |
| `lib/env.sh` | — | Load repo `.env`, validate `RMLX_O_MODELS_ROOT`. **Source it.** |
| `lib/cpu_snapshot.sh` | — | Per-process cumulative CPU seconds, for interference gates. **Source it.** |
| `lib/busiest_between.awk` | — | Which process burned the most CPU between two `cpu_snapshot` files. |

## Release

| Script | Via | What it does |
|---|---|---|
| `release/package_binary.sh` | `make release-package` | Build + package the release binary for `aarch64-apple-darwin`. |
| `release/build_bottle.sh` | — | Build a Homebrew bottle from the installed keg. |
| `release/source_sha256.sh` | `make release-sha` | Compute the sha256 of a GitHub source tarball and patch the formula. |
| `release/sync_tap.sh` | `make tap-sync` | Sync `packaging/homebrew/rmlx.rb` into the Homebrew tap. |
| `release/sign_artifact.sh` | — | Keyless (sigstore) signature for the release tarball. |
| `release/changelog_section.sh` | — | Print one version's `CHANGELOG.md` section for the GitHub release body. |

## Quality / eval

| Script | What it does |
|---|---|
| `eval/wikitext2_ppl.py` | Wikitext-2 perplexity harness for rMLX. |
| `bench/run_upstream_ppl.py` | Run `mlx_lm.evaluate` wikitext-PPL across the in-scope model set. |
| `bench/build_ppl_drift_table.py` | Render the PPL × TPS × similarity-drift table. |
| `bench/extract_outputs.py` | Extract per-(model, backend, quant) greedy-output tuples from CBB `summary.csv`. |
| `bench/auto_resolution_smoke.py` | 5-model auto-resolution + regression smoke. Asserts every arch resolves `--kv-quant auto` to the one engine default, and that serving has not collapsed. |

## One-off assets

| Script | What it does |
|---|---|
| `gen_lloyd_codebook.py` | Generate Lloyd-Max optimal N(0,1) centroids for the TurboQuant codebook. |
| `convert_silero_vad.py` | Convert Silero VAD v4 ONNX weights to safetensors. Run once; output is committed. |

## `bench/` — campaign scripts

These are **historical, campaign-scoped** drivers kept for reproducibility of a
specific report. They hard-code models, contexts and flags for the campaign they
were written for. Read one before reusing it; prefer extending the general
drivers above.

`b1_turbo_flash_validate.sh`, `final_matrix_bench.sh`,
`fullctx_regression_bench.sh`, `gemma_matrix_bench.sh`, `p0b_prefill_bench.sh`,
`p0b_ttft_only.sh`, `p0b_vg2_niah.sh`, `p1a4_turbo_flash_lock_bench.sh`,
`p2a_turbo_flash_bench.sh`, `p2c1_remaining_cells.sh`,
`p2c1_spec_128k_bench.sh`, `t1_final_bench.sh`, `t2_final_bench.sh`,
`t3_final_bench.sh`, `turboquant_v3_bench.sh`, `vg2_niah_surrogate.sh`,
`vg2_turbo_flash_lock_qwen35b.sh`, `vg2_turbo_flash_qwen35b.sh`.
