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
| `check_kv_codec_disposition.sh` | Nine rules over the operator-facing KV help and every `docs/KV_QUANT.md` INERT banner, all derived from `ALL_KV_QUANTS` + the three decode predicates. Which help it reads is derived too — every `help =` / `long_help =` identifier clap renders, anywhere in the CLI crate, by bare name or by path — so there is no scope list and no exclusion list. Rejects any resident-KV ratio written into that help (one producer: `rmlx info --list-cache-types`) and fails when the listing the help points at has no call site. |
| `check_kv_codec_disposition_fixtures.sh` | Recall test for the above — 18 synthetic scan roots (one edit each), asserting which rule fired and exit 2 vs exit 1, incl. a help constant defined in a second module and referenced by path. |
| `check_kv_byte_model_parity.sh` | `scripts/perf_ceiling.py`'s KV byte model agrees with the engine's, per codec, per topology and per head dimension, over the sweep the engine emits. |
| `check_kv_byte_model_parity_fixtures.sh` | Recall test for the above — 9 synthetic manifests, each asserting which check fired and exit 2 vs exit 1, including head- and tail-layer edits that a sampled codec vector could not see. |
| `check_metal_compiles.sh` | Every `.metal` kernel compiles natively at `-std=metal3.1` and `-std=metal4.0`, and is named by its `probes/kernels.manifest`. |
| `check_metal_format.sh` | Every `.metal` file is formatted. |
| `check_doc_source_citations.sh` | Every `crates/...` source path cited in a tracked `docs/*.md` file resolves. The subsystem docs replace figures with citations; the trade only holds while the citations do. |
| `check_no_decode_swallow.sh` | A failed decode step or failed sampler call cannot be swallowed into a silent success. |
| `check_no_inline_tests.sh` | No inline `#[cfg(test)] mod tests` outside `*_tests.rs` / `tests.rs`. |
| `check_no_kernel_input_eval.sh` | No blocking `Array::eval()` on a kernel input inside a dispatcher. |
| `check_no_kernel_input_eval_fixtures.sh` | Recall test for the above. |
| `check_no_scalar_f32_leak.sh` | No unguarded `scalar_f32(` in the metal-owning crates — the f32 decode-graph promotion class. |
| `published_samples.py` | `verify`: the checked-in published-protocol sample sets (`prompts/published/`) re-derive from what the manifest records — every file's digest and byte length, the draw from its seed over the recorded pool, every sample's user message from its template and its verbatim upstream record, and every body digest from its messages. `--sources <dir>` adds the pinned upstream revision. `build` is the one producer of those files; both halves share the selector, the renderer and the digest, so a number's provenance cannot drift from the data it was measured on. |
| `check_published_samples_fixtures.sh` | Recall test for the above — 18 synthetic sample-set roots, one edit each: a flipped byte, a truncated file, a wrong seed, an undrawn sample, a duplicated and a dropped one, a stale body digest, an edited prompt with every digest re-blessed, an upstream file off its pinned revision, a reordered pool, a missing source, and each way the manifest itself can be empty, short, unreadable, absent or of a schema this script does not read. Each asserts the reason as well as exit 1 vs exit 2. |
| `metal_dirs.sh` | The list of directories holding gated `.metal` kernels. Sourced by the metal gates. |
| `file_size_report.sh` | Advisory (non-failing) LOC report for source files >1000 lines. |
| `target_size_report.sh` | Advisory (non-failing) `target/` size report. |

## Test execution

| Script | Via | What it does |
|---|---|---|
| `run_gpu_tests.sh` | `make gpu-test` | Runs the `#[ignore]` Metal tests per member crate, `--test-threads=1`, under Metal shader validation. |
| `gpu_validation_census.txt` | — | Data, not a script: the shader-validation hits `run_gpu_tests.sh` accepts, one entry per originating test — kernel, access kind, that test's count, crate, and the analysis it rests on. The expectation is the sum over the tests that ran; anything else fails naming the delta. |
| `run_gpu_tests_selftest.sh` | `make gpu-runner-selftest` | Recall test for the runner's reporting: a shader-validation hit and a crate failure in the same run are both reported, the access mix is the one observed, every census-pin verdict fails (or passes) with its own reason, and the tracked pin parses against the real classifier's population. Stubbed crates, no GPU. |
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
| `perf_ab.sh` | `perf_canary.sh --ab` | **ABBA-interleaved A/B of two `rmlx baseline` arms.** Host-quiescence gate, Metal-exclusivity gate, arm-distinguishability guard, token-id comparison, per-arm `metal_gen_alloc_mb` + resident `kv_cache_bytes`. `--synthetic-arms` declares the arms are stubs, so the run measures nothing and the machine is not consulted. Never writes `runs.db` — promote a result with `ingest/perf_ab_ingest.py`, which refuses a `--synthetic-arms` run outright. |
| `perf_ab_selftest.sh` | `make canary-ab-selftest` | Mutation check for `perf_ab.sh` — every guard must fail when broken. Runs under `--synthetic-arms`, so no case reads this machine; the host-gate cases supply `ps` and `pgrep` shims, and the count of cases that could reach this host is tallied and must be zero. In `make ci`. |
| `perf_ab_host_gate_fixtures.sh` | `make canary-ab-host-gate-fixtures` | Recall test for the measurement/logic boundary: the host gates still fire on a shimmed hostile host, `--synthetic-arms` makes the verdict identical on a hostile and a quiet one, and it waives no arm-reading guard. In `make ci`. |
| `perf_ab_ingest_selftest.sh` | `make canary-ab-ingest-selftest` | Mutation check for `ingest/perf_ab_ingest.py` — 17 cases over synthetic result files, one per refusal. Never writes `runs.db`. In `make ci`. |
| `bench_llama_ab_selftest.sh` | `make llama-ab-selftest` | Mutation check for `bench_llama_ab.sh` against a stub `llama-server` — 19 cases, one per guard. Every case declares `--synthetic-arms` and asserts a literal exit code; the taint-path and quiescence-gate cases supply a `ps` shim, and the count of cases that could reach this host is tallied and must be zero. In `make ci`. |
| `check_spec_metric_parity.sh` | `make check-spec-metric-parity` | CI gate: the speculative metrics `rmlx_metrics::registry::SPEC_METRICS` declares are the ones `spec_bench.sh` records. A Rust test already forces every declared figure to have an export column; nothing forced the bench to write a value for it, so a tenth metric would render `-` on every row for ever — the single-producer property holding inside Rust and leaking at the language boundary. The Rust list is the oracle; the reader key and cast the script maps each name to stay local to it. Fails in both directions, naming the delta. In `make ci`. |
| `check_spec_metric_parity_fixtures.sh` | `make check-spec-metric-parity-fixtures` | Recall test for the above: six synthetic scan roots, one edit each — a name declared and not recorded, recorded and not declared, deleted from the bench, a Python block indented off column 0, and the oracle renamed. Each asserts the literal exit code and greps the reason, because a gate that refuses for the wrong reason stops refusing when that reason moves. In `make ci`. |
| `spec_bench_selftest.sh` | `make spec-bench-selftest` | Mutation check for `spec_bench.sh` against a stub server — 44 cases over canned SSE responses, a canned ITL ring and canned round-loop `done` lines, asserting the value each arm ingests and the reason behind every refusal. Among them: the per-round split reaches the speculative row and no per-round figure reaches the no-drafter one, a `done` line whose derived field contradicts its own counters is refused — one case per derived field, not one for the field that happened to be tested — a counter a figure is derived from is required rather than defaulted to a zero nobody measured, a log that does not name its cell is refused rather than guessed at from the caller's flags, a `done` line whose seed count contradicts what it emitted is refused, the drift itself — a seed captured before the pre-round emission — is caught by the rounds' own emission budget, the recorded cell is the engine's rather than the block the script asked for, and a drafter kind that would escape the buffer directory is refused at parse. The stub streams on a fixed schedule so its reported rate is the rate the wire carried, and records the pid that bound the port so a foreign listener cannot stand in for it. No GPU, no model, and the stub answers `metrics record` without writing `runs.db`. In `make ci`. |
| `bench_llama_ab.sh` | — | **ABBA-interleaved A/B of two `llama-server` arms** (fork vs upstream, codec vs codec). Same quiescence discipline as `perf_ab.sh` and the same `--synthetic-arms` boundary, both from `lib/cpu_snapshot.sh`, reported over the server's own `timings` plus KV-buffer and peak-RSS columns. Never writes `runs.db`. |
| `perf_canary.sh` | `make perf-canary` | Fast decode-TPS canary over the three standard test-target models. Reads the resolved KV codec through `lib/server_kv_quant.py`, so the CSV carries the engine's own name for it. |
| `regression_gate.sh` | — | Compare a committed baseline against the latest canary row. Exit 125 = `git bisect skip`, 1 = regression. |
| `prefill_chunk_sweep.sh` | — | **Cyclic Latin-square sweep of one architecture's prefill chunk** over a set of levels and prompt lengths, which is what an `arch_default` row in `prefill_chunk.rs` is supposed to be backed by. Every level occupies every slot position exactly once, so this host's non-linear positional drift cancels where an ABBA block would leave it on one arm; the verdict is the paired within-row statistic, not a pooled median. Host-quiescence gate per slot and over the sweep (`lib/cpu_snapshot.sh`), Metal-exclusivity gate, and a token-digest comparison across levels — chunking that changes the output is a correctness failure, not a slower cell. Refuses to record anything when any gate trips; `--record` ingests one §8.5 cell per (model, prompt length, level) under `decode_config = 'prefill_chunk=<n>'`. |
| `perf-iter/bench_decode_tps.sh` | `make perf-iter` | Per-iteration regression bench for a perf-fix campaign. Takes each request's decode rate from the server via `lib/server_decode_tps.py`, identity from `lib/identity.sh`, and writes everything under `<RMLX_HOME>/metrics/`. |
| `perf-iter/diff_baseline.sh` | — | Compare two perf-iter JSONL files, emit per-cell deltas. |
| `perf_ceiling.py` | — | Static roofline calculator: bytes/step and the theoretical ceiling from a snapshot's `config.json` + safetensors index. Its KV byte model is a second copy of the engine's and is held to it by `check_kv_byte_model_parity.sh`; `--byte-model` is that gate's entry point. |
| `sdpa_headdim_bench.py` | — | What MLX's SDPA dispatch costs as a function of `head_dim`. |
| `aggregate_decode_profile.py` | — | Aggregate per-model `decode_profile` lines from `profile_<MODEL>.txt`. |

## Cross-backend bench cells

| Script | What it does |
|---|---|
| `bench_cell.sh` | Per-cell driver for the cross-backend bench. Legacy mode drives the CBB Python harness over an HTTP backend; cache-type mode drives `rmlx baseline` directly and emits one §8.5 RunRecord. |
| `bench_codec_cell.sh` | Single-codec × single-model bench runner. |
| `bench_cache_types.sh` | Drive the cache-type combo matrix for one model. |
| `bench-records-sweep.sh` | 5-model × 4-KV-quant `BENCHMARK_CHAMPIONS` regression sweep. |
| `bench/tri_engine_same_model.sh` | **llama.cpp vs rMLX vs stock mlx-lm on ONE checkpoint**, across each engine's KV options. Refuses to emit a llama.cpp row unless that binary's Metal tensor API probes live (an inert one reads ~3x low on prefill), and refuses any cell whose KV would push this host into swap. |
| `bench/tri_engine_summarize.py` | Ingest one raw artifact from `tri_engine_same_model.sh` into a normalized cell record, print the comparison table, or (`--geometry`) read the benched checkpoint's KV geometry out of its `config.json`. Owns the single definition of the cross-engine record shape, incl. the KV bits/value normalization that makes an allocated-for-n_ctx figure comparable to a filled-prefix one. |
| `spec_bench.sh` | Bench a model in normal vs speculative-decode mode, at any `--draft-kind` / `--draft-block-size` the engine ships. Both arms report the decode rate the engine measured over the first-emitted-token to last-emitted-token window — the speculative arm from the round loop's log line, the no-drafter arm from the server's ITL ring — and each is cross-checked against the same window timed client-side. `kv_quant` is the codec the run's log says it resolved, `prompt_tokens` is what the server counted, and the speculative row's `decode_config`, block size and every per-round figure — `tokens_per_round`, `accepted_per_step` and the draft / verify / loop `ms_per_round` split — come from the round loop's own `done` line rather than from this script's flags; a run that reports none of these is refused rather than recorded under a guess. |
| `baseline/run_mlx-lm.sh` | Baseline measurement via Apple's stock `mlx-lm` loader. |
| `baseline/run_mlx-lm-turboquant.sh` | Baseline measurement via the `mlx-lm-turboquant` fork. |
| `baseline/run_oMLX.sh` | Baseline measurement via the oMLX Python server. |
| `baseline/turbo_probe.py` | One identical decode loop run under either mlx-lm venv: decode TPS plus **true** KV residency (packed store *and* any dense dequant mirror) vs the cache's self-reported `nbytes`. `--seq` palindrome gives single-process ABBA. |
| `baseline/turbo_abba.sh` | Process-level ABBA (stock, fork, fork, stock) around `turbo_probe.py` for the cross-venv leg. Hashes each arm's `mlx_lm` source tree and **refuses (exit 6)** when the two match — a venv resolving `mlx_lm` from site-packages would otherwise produce a fork-vs-stock ratio of 1.000x that reads as a measured null. The digests go into the artifact. |
| `baseline/turbo_summarize.py` | Median / min / max / spread and per-mode ratios from `turbo_probe.py` jsonl. |
| `baseline/group-A-baseline.sh` | Measure the rMLX baseline TPS for the Group-A regression gate. |
| `baseline/c1-gemma4-cold-equal.sh` | C1 acceptance: gemma4 partial-prefix reuse. |
| `baseline/d8-phase1-measure.sh` | Quantify the first-dispatch MSL-compile tax. |
| `autoresearch_run.sh` | Single autoresearch experiment run. |

## Metrics ingest

| Script | What it does |
|---|---|
| `ingest/llama_bench_ingest.py` | Convert `llama-bench -o json` rows into the §8.5 universal RunRecord and ingest them. |
| `ingest/llama_ab_ingest.py` | Promote one accepted `bench_llama_ab.sh` result into two §8.5 RunRecords (one per arm). Refuses a TAINTED run unless told otherwise, and a `--synthetic-arms` run with no waiver at all. |
| `ingest/perf_ab_ingest.py` | Promote one accepted `perf_ab.sh` result into two §8.5 RunRecords (one per arm), carrying `decode_tps_warm` + `kv_cache_bytes`. Identity comes from the measured binary and is digest-checked against the run; refuses a TAINTED run, a weakened interference gate, or a cell key that disagrees with the measurement, and refuses a `--synthetic-arms` run with no waiver at all. `decode_config` is derived from each arm's own recorded `--kv-boundary-layers`, so a boundary sweep lands in its own cell. |
| `ingest/codec_inertness_ingest.py` | Promote `bench/codec_inertness_probe.sh` cells into §8.5 RunRecords. Records `kv_cache_bytes` only — the probe's unpaired throughput columns are not comparable and are deliberately dropped; the token-id digest travels in `notes`. `decode_config` is derived from the probe's `kv_boundary` column; a CSV whose rows are wider than its header is refused rather than filed under shifted column names. |
| `lib/identity.sh` | Shared §8.5 run-identity (`rmlx metrics identity --json`) for bench scripts. **Source it.** |
| `lib/prefill_ms.py` | Read `decode_profile{prefill_ms}` back out of an rmlx run log. |
| `lib/spec_round_log.py` | Read a speculative round loop's own `done` line — round counts, draft/accept totals, the per-round draft / verify / loop split, the block the engine ran, the cell it named, whether the run was `charged` — phases forced at every boundary, so a different and slower schedule — and the `decode_tps` it measured. The only reader of that line: `emitted / elapsed_ms` off it counts the prefill, and a `decode_tps` that is a bare number instead of `Some(x)` / `None` came from a binary that had not yet stopped reporting it that way, so it is refused rather than read. Every event's derived fields are checked against that event's own counters before aggregating, and events naming two cells are refused — an aggregate over them belongs to neither. |
| `lib/sse_decode_window.py` | Time a streamed chat-completions response over its decode window — first content token to last — plus the token count and a preview. Reports no rate at all for a response too short to have a window, and refuses a response whose tokens did not arrive one per chunk. |
| `lib/snapshot_identity.py` | Read a snapshot's `model_namespace` / `model` / `weight_quant` out of the snapshot: the `__` split of the directory name and the checkpoint's own `config.json`. Refuses a quantization it cannot name on the §5.2 whitelist. |
| `lib/run_log_for_pid.py` | Pick the run log a given pid wrote, out of the candidates on stdin — the `rmlx start` event states the pid. Refuses zero or more than one match, so a phase never reads another process's log as its own. |
| `lib/server_kv_quant.py` | Read the KV codec a run resolved out of its run log's `cache-type resolved` event. Refuses a Debug-rendered name — only the canonical lower-case spelling is one the flag accepts and the DB records. |
| `lib/server_decode_tps.py` | Read one request's decode rate off a running server's ITL ring (`GET /metrics/cache`), where `1000 / step_mean_ms` is the same first-token-to-last window the round loops report. Refuses unless exactly one new sample is attributable to the request. |

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
| `mlx_preflight.sh` | `make mlx-preflight` | Refuse to bench unless the MLX the built binary loaded is the pinned, nax-capable pair. Pre-filters on the `opt` symlinks, then asks the binary; `rmlx baseline` / `rmlx bench` refuse on their own regardless. |
| `mlx_restore_pin.sh` | `make mlx-restore-pin` | Restore the pair `crates/rmlx-mlx/mlx-pin.txt` names. |
| `target_gc.sh` | `make target-gc` | Prune stale build profiles from `target/`. |
| `lib/env.sh` | — | Load repo `.env`, validate `RMLX_O_MODELS_ROOT`. **Source it.** |
| `lib/mlx_pin.sh` | — | Parse `crates/rmlx-mlx/mlx-pin.txt` (same grammar as `parse_pin`, version shape allowlisted). **Source it.** |
| `lib/cpu_snapshot.sh` | — | Per-process cumulative CPU seconds, for interference gates, plus the `snapshot_ok` / `window_not_sampled` pair that both A/B harnesses share: it separates "nobody looked" (`--synthetic-arms`) from "the look failed" from a quiet host. **Source it.** |
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
| `bench/auto_resolution_smoke.py` | 5-model auto-resolution + regression smoke. Asserts every arch resolves **both** `--kv-quant auto` and `--kv-preset auto` to the one engine default, and that serving has not collapsed. |
| `bench/codec_inertness_probe.sh` | Per-codec `kv_cache_bytes` + greedy token-id digest + "packed store skipped" for one (model, prompt length). Separates a codec that changes something from one that is another spelling of `none`. `--kv-boundary-layers H,T` sweeps the boundary-floor counts and records them in a `kv_boundary` column. |

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

`b1_turbo_flash_validate.sh`, `p0b_prefill_bench.sh`, `p0b_ttft_only.sh`,
`p0b_vg2_niah.sh`, `p1a4_turbo_flash_lock_bench.sh`,
`p2a_turbo_flash_bench.sh`, `p2c1_remaining_cells.sh`,
`p2c1_spec_128k_bench.sh`, `turboquant_v3_bench.sh`, `vg2_niah_surrogate.sh`,
`vg2_turbo_flash_lock_qwen35b.sh`, `vg2_turbo_flash_qwen35b.sh`.

Six of these are gone rather than frozen: `t1_final_bench.sh`,
`t2_final_bench.sh`, `t3_final_bench.sh`, `fullctx_regression_bench.sh`,
`gemma_matrix_bench.sh` and `final_matrix_bench.sh` each wrote a permanent
`decode_tps_warm` row into `runs.db` from a whole-request stopwatch — a rate
that counts the prompt prefill, which is `overall_tps` under another metric's
name. None had a caller: no Makefile target, no doc, nothing but the row above.
A driver nobody invokes and that writes an uncorrectable wrong row when someone
does is not reproducibility, and the reports they produced are already written.
Git holds them. A campaign script that is meant to stay runnable takes its
decode rate from `lib/server_decode_tps.py` or `lib/spec_round_log.py` like the
live drivers do.
