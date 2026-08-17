# rMLX — common dev commands. Targets are thin wrappers over cargo / pre-commit.
#
# Usage examples:
#   make             # = make help
#   make ci          # full pre-merge gate
#   make serve       # serve primary test model on :8080
#   make info MODEL=/some/other/path

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Workspace root — the directory containing this Makefile.
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Local config: copy .env.example → .env and set your machine paths there.
# `-include` is silent when absent. Values become make variables (and the model
# root is exported below so bench scripts inherit it).
-include $(REPO_ROOT)/.env

# Model-snapshot root — the single directory holding your downloaded
# mlx-community__* / prism-ml__* / z-lab__* snapshots. Set it once:
#   - copy .env.example → .env and edit RMLX_O_MODELS_ROOT, or
#   - export RMLX_O_MODELS_ROOT in your shell.
# Dedicated model paths are built off it ($(O_MODELS_ROOT)/<snapshot>).
# Fallback is a repo-local ./models dir (gitignored) — drop snapshots there to
# run with zero config. No machine-specific path is baked in.
O_MODELS_ROOT ?= $(RMLX_O_MODELS_ROOT)
# Did an operator NAME a root — via .env, a shell export, or the command line —
# or are we about to invent one? Captured before the fallback overwrites it.
O_MODELS_ROOT_NAMED := $(strip $(O_MODELS_ROOT))
ifeq ($(O_MODELS_ROOT_NAMED),)
O_MODELS_ROOT := $(REPO_ROOT)/models
endif

# A named root is forwarded VERBATIM, wrong or not, and only the invented
# fallback is gated on existing.
#
# The distinction is load-bearing because `.env` is `-include`d: values from it
# are make variables, NOT environment variables, so they reach a child only
# through this `export`. Gating the export on the path existing would therefore
# suppress it precisely when the path is wrong — the child would see nothing
# set, report "no snapshot configured", and skip green at the one operator who
# did configure something. Forwarding it keeps a typo reaching the readers that
# can call it a typo.
#
# The fallback is the opposite case: nobody named it, so handing children a
# repo-local `models/` that need not exist manufactures a configuration no one
# chose, and readers cannot tell it from a deliberate one.
#
# Export the STRIPPED value. Make preserves trailing whitespace in a `.env`
# assignment, so `RMLX_O_MODELS_ROOT=/path ` forwards a path with an invisible
# trailing space: the child fails as Misconfigured with a message whose cause
# cannot be seen. ($(strip) also collapses runs of internal whitespace, so a
# root whose name contains two consecutive spaces is unsupported — a single
# internal space is untouched.)
ifneq ($(O_MODELS_ROOT_NAMED),)
export RMLX_O_MODELS_ROOT := $(O_MODELS_ROOT_NAMED)
else ifneq ($(wildcard $(O_MODELS_ROOT)/.),)
export RMLX_O_MODELS_ROOT := $(strip $(O_MODELS_ROOT))
endif

# Primary test model. Override at the CLI: make info MODEL=/path/to/other-snapshot
MODEL ?= $(O_MODELS_ROOT)/mlx-community__gemma-4-e4b-it-mxfp8
PORT  ?= 8080

# Decode-focused profiling shape (small prefill, long decode window) for
# `make profile-samply-debug`. Override at CLI: PROF_PROMPT=4096 PROF_GEN=100.
PROF_PROMPT ?= 1024
PROF_GEN    ?= 500

# Audit ignores: see deny.toml for rationale (paste + number_prefix transitive).
AUDIT_IGNORES := --ignore RUSTSEC-2024-0436 --ignore RUSTSEC-2025-0119

.PHONY: help build check test fmt fmt-check lint audit deny precommit hooks \
        ci ci-metrics tag release-package release-sha release-sign bottle tap-sync \
        clean serve chat info logs-tail metrics-summary \
        metrics-init metrics-doctor metrics-doctor-fix metrics-export \
        metrics-backup metrics-replay-pending metrics-prompts-sync \
        metrics-champions metrics-champions-rmlx \
        build-perf build-debug test-perf ci-perf gpu-test model-check model-check-full \
        profile-samply profile-samply-debug profile-instruments bench asm perf-iter \
        canary canary-gate canary-ab canary-ab-selftest \
        mlx-preflight mlx-restore-pin target-gc target-size-report profile-gputrace \
        profile-mst \
        build-capture test-capture gputrace-preflight traces-gc \
        ssd-canary ssd-canary-gate \
        bench-codec-cell \
        smoke-codec-matrix \
        e2e \
        file-size-report check-no-inline-tests check-no-scalar-f32-leak \
        check-no-decode-swallow check-gpu-tests-ignored \
        check-gpu-tests-ignored-fixtures \
        check-eval-lock check-eval-lock-fixtures eval-lock-stress \
        check-no-kernel-input-eval check-no-kernel-input-eval-fixtures \
        check-metal-compiles check-metal-format

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---- build / test ------------------------------------------------------
build:           ## cargo build --release
	cargo build --workspace --release

check:           ## cargo check (fast, no codegen)
	cargo check --workspace --all-targets

test:            ## cargo test --workspace
	cargo test --workspace

mlx-preflight:   ## verify the linked MLX stack (and nax kernels on M5+) before benching
	bash scripts/mlx_preflight.sh

target-gc:       ## report stale target/ profiles (APPLY=1 to prune; target/ has no size cap)
	bash scripts/target_gc.sh $(if $(APPLY),--apply,) $(if $(ALL),--all,)

target-size-report: ## advisory: print target/ size + hint when over threshold (non-failing; also runs at the end of `make ci`)
	@bash scripts/target_size_report.sh

# The re-sign is not optional decoration. Cargo emits an ad-hoc *linker-signed*
# binary with no entitlements at all; com.apple.security.get-task-allow is what
# marks the process attachable by Apple's developer tools, which is a
# prerequisite for the GPU debugger to work with what it captured.
# `codesign --force` is idempotent, so this also repairs a binary that a plain
# `cargo build` re-created without it.
build-capture: ## build + entitle the debug-only GPU-capture binary (release-debug + metal-capture, signed so Apple's GPU tools can attach)
	cargo build --profile release-debug --features rmlx-cli/metal-capture
	codesign --force --sign - --entitlements scripts/rmlx-capture.entitlements \
		target/release-debug/rmlx
	@bash scripts/gputrace_preflight.sh --binary target/release-debug/rmlx

gputrace-preflight: ## check the GPU-tools prerequisites (developer mode, get-task-allow, Xcode selected, Metal toolchain)
	@bash scripts/gputrace_preflight.sh $(if $(BIN),--binary $(BIN),)

traces-gc:       ## report .rmlx/traces against the retention caps (APPLY=1 to prune; MAX_COUNT=/MAX_TOTAL_GB=/MAX_AGE_DAYS= to retune)
	bash scripts/traces_gc.sh $(if $(APPLY),--apply,) $(if $(ALL),--all,) \
		$(if $(MAX_COUNT),--max-count $(MAX_COUNT),) \
		$(if $(MAX_TOTAL_GB),--max-total-gb $(MAX_TOTAL_GB),) \
		$(if $(MAX_AGE_DAYS),--max-age-days $(MAX_AGE_DAYS),)

# `make test` is `cargo test --workspace` without --all-features, so it compiles
# these out entirely. Run from `ci` as well, or an off-by-one in the window
# policy passes the gate green.
test-capture: ## run the feature-gated GPU-profiling unit tests (capture window + xctrace parser; plain `make test` skips them; also a `make ci` step)
	cargo test -p rmlx-mlx --features metal-capture --lib metal_capture
	cargo test -p rmlx-mlx --features metal-capture --lib xctrace
	cargo test -p rmlx-cli --features metal-capture --bin rmlx gpu_capture

profile-gputrace: ## capture a decode-window Metal GPU trace (CODEC= MODEL= required; enforces the .rmlx/traces cap unless KEEP_ALL=1)
	@if [ -z "$(CODEC)" ] || [ -z "$(MODEL)" ]; then \
		echo "Usage: make profile-gputrace CODEC=<codec> MODEL=<snapshot-abs-path> \
[PROMPT_TOKENS=4096] [SKIP=4] [STEPS=8] [GEN=N] [KEEP_ALL=1]"; \
		echo "Needs a binary from \`make build-capture\`."; \
		exit 2; \
	fi
	bash scripts/gpu_capture.sh --kv-quant $(CODEC) --model $(MODEL) \
		$(if $(PROMPT_TOKENS),--prompt-tokens $(PROMPT_TOKENS),) \
		$(if $(SKIP),--skip $(SKIP),) $(if $(STEPS),--steps $(STEPS),) \
		$(if $(GEN),--gen $(GEN),) $(if $(KEEP_ALL),--keep-all,)

# profile-mst: the timing half of GPU profiling. A .gputrace answers WHICH
# kernels ran; this answers HOW LONG they took and how long the GPU waited on
# the host (start-latency, the CPU->GPU gap) — the one signal a capture cannot
# give, since a replay has the replay's schedule. Records the live process,
# exports metal-gpu-intervals and parses it; the parser refuses a misaligned
# table rather than printing plausible wrong numbers.
#
# The recording starts at process launch (xctrace --attach is broken for this
# template). Weight load submits no GPU work so it leaves no rows, but prefill
# does: SKIP_MS= drops it from the summary, and DEFAULTS to the prefill_ms the
# run itself reported, so the decode boundary is measured rather than guessed.
# PROMPT_TOKENS must be a size with a checked-in prompt fixture.
# MODEL falls back to the workspace default like `make serve` / `make info`, so
# there is no "MODEL is required" guard to write here — it could never fire.
# scripts/mst_capture.sh validates the snapshot path and every numeric argument.
#
#   make profile-mst MODEL=<snapshot-abs-path> [CODEC=none] [TIME_LIMIT=18]
#     [SKIP_MS=<default: the run's measured prefill_ms>] [GEN=600] [KEEP=5]
#     [PROMPT_TOKENS=4096|8192|16384|32768|65536|131072]
profile-mst: ## record a Metal System Trace of a live rmlx run and summarise GPU time + CPU->GPU gap (CODEC= TIME_LIMIT= SKIP_MS= PROMPT_TOKENS= GEN= KEEP=)
	bash scripts/mst_capture.sh --model $(MODEL) \
		$(if $(CODEC),--kv-quant $(CODEC),) \
		$(if $(TIME_LIMIT),--time-limit $(TIME_LIMIT),) \
		$(if $(SKIP_MS),--skip-ms $(SKIP_MS),) \
		$(if $(PROMPT_TOKENS),--prompt-tokens $(PROMPT_TOKENS),) \
		$(if $(GEN),--max-tokens $(GEN),) \
		$(if $(KEEP),--keep $(KEEP),)

mlx-restore-pin: ## restore mlx 0.31.2 + mlx-c 0.6.0_2 (nax-capable pair) and relink
	bash scripts/mlx_restore_pin.sh

build-perf:      ## cargo build --profile release-perf (debug-assertions off, stripped)
	cargo build --workspace --profile release-perf

build-debug:     ## cargo build --profile release-debug (opt-level=3 + full DWARF, for samply)
	cargo build --workspace --profile release-debug

test-perf:       ## cargo test --profile release-perf (release-perf profile, panic=unwind forced by cargo for test harness)
	cargo test --workspace --profile release-perf

# ci-perf runs the GPU/Metal suite after `test-perf`, and it is the only shared
# gate that does. `make ci` cannot: the GPU tests need the Metal context to
# themselves (hard rule 8) and take minutes, which is the wrong price on every
# commit. This is a real new cost for `ci-perf` and not a free one — the target
# used to be a single `cargo test --workspace`, runnable next to a live
# `rmlx serve`, and now it refuses to start unless the GPU is idle. It is simply
# the cheapest place in the tree to pay it: `ci-perf` is already the long,
# pre-merge-only target, and the preflight line below makes the new precondition
# fail in milliseconds instead of after the release-perf half.
#
# Three lines, in this order, for two different reasons:
#
#   * `--preflight` FIRST because it is a precondition, not a test. It checks
#     only the things that cost nothing to check — RMLX_SKIP_GPU unset, no
#     competing MLX process, a non-empty classification — and those are the most
#     likely way this gate fails in daily use. Discovering a live `rmlx serve`
#     after `test-perf` has run throws away the ~16 min it took.
#   * `test-perf` before the tests themselves because it covers the whole
#     workspace, so a compile error anywhere shows up there, whereas the GPU run
#     visits five crates and holds the GPU while it does. Fail on the broad,
#     shareable step before spending exclusive-GPU minutes.
#
# The runner is invoked directly rather than through `$(MAKE) gpu-test`. Make
# propagates command-line variables to sub-makes, so `make ci-perf CRATE=…` or
# `VALIDATE=0` would silently reach the runner and narrow or disarm the gate:
# `--crate` shrinks the classified population in lockstep with the executed one,
# so the runner's own coverage check cannot tell that run from a complete one.
# The knobs stay on `gpu-test`, where a human asking for a subset means it.
#
# The two halves run under different profiles on purpose. The GPU run builds
# under `dev`, where debug assertions are live — 61 `debug_assert!` sites in
# rmlx-kv-quant alone — and those are correctness guards on correctness tests.
# `test-perf` is the one that must be release-perf, because that is the codegen a
# perf-sensitive change ships under. The consequence is in hard rule 9: no gate
# runs a GPU test with debug-assertions off, so a defect that only appears there
# has to be reproduced by hand.
#
# Cost: the GPU half is ~4.5 min warm (318 tests, serialized, under Metal shader
# validation). Whole target after a codec-layer edit, measured: ~21 min. The dev
# profile is not shared with `test-perf` and is what `make target-gc` prunes
# first, so a run after a GC pays a cold opt-level-0 build on top. See
# docs/TESTING.md.
ci-perf:         ## pre-push gate under release-perf + the serialized GPU/Metal suite (separate from make ci; run before merging perf-sensitive or codec-layer changes)
	@bash scripts/run_gpu_tests.sh --preflight
	$(MAKE) test-perf
	@bash scripts/run_gpu_tests.sh
	@echo "ci-perf ok"

# gpu-test: the execution step for the tests `check-gpu-tests-ignored` mandates.
# Every test reaching Device::Gpu must carry #[ignore] (a shared Metal context
# driven from parallel cargo-test threads aborts the whole binary), and until
# this target existed nothing ran them: `make test` passes no --ignored and the
# hosted CI has no Metal. GPU decode correctness for every KV codec sits in that
# category, and tests in it have gone red on main and stayed red undetected.
#
# It is NOT part of `make ci`: it needs exclusive access to the Metal context and
# is far too slow to block every commit. `ci-perf` runs the same suite as its
# third line — by calling the script directly, not this target, so that its
# population cannot be narrowed from the command line (see the note there). This
# target is the hand-driven entry point: narrow it with CRATE=/FILTER= while
# iterating on the codec layer, rather than paying for the whole gate each time.
#
# Metal shader validation is ON here. An out-of-bounds device store is dropped
# silently — command buffer completes, cb.error is nil, cargo exits 0, and the
# tests over the frozen buffer still pass — so without instrumentation this
# target cannot see the repo's documented silent-corruption class at all. It
# costs throughput, which is why it lives on this target and not on any cell
# whose numbers get recorded. VALIDATE=0 opts out.
gpu-test:        ## run the GPU/Metal #[ignore] tests serialized under Metal shader validation (CRATE= FILTER= to narrow, VALIDATE=0 to skip instrumentation); needs exclusive machine access
	@bash scripts/run_gpu_tests.sh $(if $(CRATE),--crate '$(CRATE)',) $(if $(FILTER),--filter '$(FILTER)',) \
		$(if $(filter 0,$(VALIDATE)),--no-shader-validation,)

# model-check: run only the model-logic crates (rmlx-models, rmlx-runtime,
# rmlx-quant) plus the KV-codec crate (rmlx-kv-quant). Excludes server, CLI, and
# metrics churn. The #[ignore] integration tests stay skipped here — this target
# runs without any model present.
#
# rmlx-kv-quant is included so the cache-level bf16 store-boundary floor (the
# model-agnostic f32-KV guard) is checked at every integration run: its
# bytes-per-element invariant tests (resident_bytes_tests) trip CI the moment a
# future arch leaks f32 into the unquantised KV store.
model-check:     ## cargo test -p rmlx-{models,runtime,quant,kv-quant} (no server/cli/metrics; no model needed)
	cargo test -p rmlx-models -p rmlx-runtime -p rmlx-quant -p rmlx-kv-quant

# model-check-full: run the model-logic unit tests (same as model-check) PLUS the
# per-arch golden-token integration tests, pinned to ONE model.
#
# MODEL is forwarded as RMLX_KV_TEST_MODEL, the golden harness's single-model
# override. It applies to the ONE golden whose architecture it serves: that golden
# runs the full 32-token assertion against <MODEL>. The others do not stand down —
# each falls through to its own slug under RMLX_O_MODELS_ROOT and runs if that
# snapshot is present, or skips if it is not. So this target covers at least the
# named model, and more on a machine with a populated models root.
#
# To run the goldens with no model pinned at all, use `make gpu-test`, which runs
# every #[ignore] GPU test including these.
#
# Avoid --include-ignored here: the lib's own #[ignore] tests (kv-cache equivalence)
# require a matching model + Metal context and segfault on arch mismatch. The five
# golden tests are integration test binaries (tests/*.rs) named explicitly below.
#
# --test-threads=1 is mandatory, not tidiness. The bonsai binary alone holds four
# #[ignore] GPU tests, and libtest runs a binary's tests on parallel threads: a
# shared Metal context driven from several of them aborts the whole binary
# ("Rust cannot catch foreign exceptions"), which is the hazard the #[ignore]
# rule exists for. Every other runner of these tests already serializes them.
#
# The guard checks the PATH, not the variable. MODEL has an unconditional default
# (see its definition above), so a `-n` test could never fire, and the default
# names a snapshot the machine need not have. That fabricated path is then
# forwarded as RMLX_KV_TEST_MODEL, which the golden harness reads as an operator
# naming a snapshot — a misconfiguration, failing every golden — when nobody named
# anything. Refusing up front, with the path in the message, is the honest form.
#
# Examples:
#   make model-check-full MODEL=/path/to/prism-ml__Ternary-Bonsai-8B-mlx-2bit
#   make model-check-full MODEL=/path/to/mlx-community__gemma-4-e4b-it-mxfp8
model-check-full: ## run model-logic crates + golden-token integration tests (MODEL= must name an existing snapshot; matching arch runs+passes, others fall through to their own slug or skip)
	@test -d "$(MODEL)" || { echo "model-check-full: MODEL must name an existing snapshot directory."; echo "  got: $(MODEL)"; echo "  usage: make model-check-full MODEL=/path/to/snapshot"; exit 1; }
	cargo test -p rmlx-models -p rmlx-runtime -p rmlx-quant
	RMLX_KV_TEST_MODEL="$(MODEL)" cargo test -p rmlx-models \
	  --test bonsai_golden_tokens \
	  --test gemma4_golden_tokens \
	  --test qwen3_golden_tokens \
	  --test bitnet_golden_tokens \
	  --test medgemma_golden_tokens \
	  -- --ignored --test-threads=1

# e2e: the feature-proof harness — drives the REAL rmlx binary per manifest case
# (CLI subprocess or `rmlx serve` + HTTP), asserts on real output, writes the
# PASS/FAIL grid to <RMLX_HOME>/e2e/report.{json,md}. Single-MLX discipline:
# --test-threads=1 is mandatory. Model-gated cases skip when no Bonsai snapshot
# resolves (RMLX_E2E_MODEL_BONSAI / RMLX_TEST_MODEL_BONSAI / RMLX_O_MODELS_ROOT).
# See docs/E2E_TEST_PLAN.md.
e2e:             ## run the E2E feature-proof harness (--ignored --test-threads=1) and print the grid path
	cargo test -p rmlx-cli --test e2e_harness -- --ignored --test-threads=1 --nocapture

# ---- format / lint -----------------------------------------------------
fmt:             ## cargo fmt --all (write)
	cargo fmt --all

fmt-check:       ## cargo fmt --check (CI)
	cargo fmt --all -- --check

lint:            ## cargo clippy -D warnings
	cargo clippy --workspace --all-targets --all-features -- -D warnings

# ---- security / supply chain ------------------------------------------
audit:           ## cargo audit (RustSec)
	cargo audit --deny warnings $(AUDIT_IGNORES)

deny:            ## cargo deny check (licenses, bans, sources, advisories)
	cargo deny --all-features check

# ---- pre-commit -------------------------------------------------------
precommit:       ## run all pre-commit hooks on the whole tree
	pre-commit run --all-files

hooks:           ## install the pre-commit git hook
	pre-commit install

# ---- advisory tooling --------------------------------------------------
file-size-report: ## advisory: print source files >1000 LOC (non-failing)
	@bash scripts/file_size_report.sh

check-no-inline-tests: ## CI gate: fail if any non-test.rs file has inline #[cfg(test)] mod tests { ... }
	@bash scripts/check_no_inline_tests.sh

check-no-scalar-f32-leak: ## CI gate: fail if arch-layer code has unguarded scalar_f32( not followed by .astype(
	@bash scripts/check_no_scalar_f32_leak.sh

check-no-decode-swallow: ## CI gate: fail if a decode-step failure breaks instead of propagating (would report as finish_reason="length")
	@bash scripts/check_no_decode_swallow.sh

check-eval-lock:  ## CI gate: fail if an MLX eval FFI call is made without the process-wide evaluation lock
	@bash scripts/check_eval_lock.sh

check-eval-lock-fixtures: ## CI gate: recall test for check-eval-lock (synthetic scan roots)
	@bash scripts/check_eval_lock_fixtures.sh

eval-lock-stress: ## run the evaluation-lock reproducer across RUNS fresh processes (default 60); not in `make ci`
	@bash scripts/eval_lock_stress.sh $(RUNS)

check-gpu-tests-ignored: ## CI gate: fail if a GPU-touching test in ANY workspace member lacks #[ignore] (would abort the whole test binary under parallel cargo test)
	@bash scripts/check_gpu_tests_ignored.sh

check-gpu-tests-ignored-fixtures: ## CI gate: the #[ignore] gate still fires on macro-generated and helper-reached GPU tests
	@bash scripts/check_gpu_tests_ignored_fixtures.sh

check-no-kernel-input-eval: ## CI gate: fail if a Metal-kernel dispatcher blocks on Array::eval() (serialises host vs GPU once per layer per decode step)
	@bash scripts/check_no_kernel_input_eval.sh

check-no-kernel-input-eval-fixtures: ## CI gate: the eval gate still fires on renamed/relocated/differently-spelled evals
	@bash scripts/check_no_kernel_input_eval_fixtures.sh

# METAL_STRICT=--strict turns a missing toolchain into a hard failure instead of
# a skip. CI sets it (the macOS runner ships the toolchain, so a skip there would
# mean the gate protected nothing); local runs leave it empty and skip.
METAL_STRICT ?=

check-metal-compiles: ## CI gate: every .metal kernel compiles natively at metal3.0 + metal4.0 and is named by its manifest (skips without the Xcode Metal toolchain; METAL_STRICT=--strict to require it)
	@bash scripts/check_metal_compiles.sh $(METAL_STRICT)

check-metal-format: ## CI gate: every .metal kernel is clang-format clean (skips without clang-format; METAL_STRICT=--strict to require it)
	@bash scripts/check_metal_format.sh $(METAL_STRICT)

# ---- one-shot CI gate -------------------------------------------------
ci: fmt-check lint test test-capture deny audit ci-metrics ## full pre-merge gate: fmt + clippy + test + feature-gated capture tests + deny + audit + metrics-sanity + inline-test + A/B-harness + MSL gates
	@bash scripts/check_no_inline_tests.sh
	@bash scripts/check_no_scalar_f32_leak.sh
	@bash scripts/check_no_decode_swallow.sh
	@bash scripts/check_eval_lock.sh
	@bash scripts/check_eval_lock_fixtures.sh
	@bash scripts/check_gpu_tests_ignored.sh
	@bash scripts/check_gpu_tests_ignored_fixtures.sh
	@bash scripts/check_no_kernel_input_eval.sh
	@bash scripts/check_no_kernel_input_eval_fixtures.sh
	@bash scripts/perf_ab_selftest.sh
	@bash scripts/check_metal_format.sh
	@bash scripts/check_metal_compiles.sh
	@bash scripts/file_size_report.sh || true
	@bash scripts/target_size_report.sh || true
	@echo "ci ok"

# tag: derive v<version> from the single source of truth
# ([workspace.package].version in Cargo.toml) and create an annotated git tag.
# No hand-typed version, no separate VERSION file.
tag:             ## create annotated git tag v<version> from Cargo.toml [workspace.package].version
	@v=$$(awk -F'"' '/^version = /{print $$2; exit}' Cargo.toml); \
	test -n "$$v" || { echo "tag: could not read [workspace.package].version from Cargo.toml"; exit 1; }; \
	git rev-parse "v$$v" >/dev/null 2>&1 && { echo "tag: v$$v already exists"; exit 1; } || true; \
	git tag -a "v$$v" -m "rMLX v$$v" && echo "tagged v$$v — push with: git push origin v$$v"

# ---- release (all local; hosted CI cannot build rMLX — no usable Metal) ----
release-package: ## build + bundle dist/rmlx-v<ver>-aarch64-apple-darwin.tar.gz (+ .sha256)
	bash scripts/release/package_binary.sh

release-sha:     ## print sha256 of the v<ver> GitHub source tarball (append --write to patch the formula)
	bash scripts/release/source_sha256.sh

release-sign:    ## keyless cosign-sign dist/rmlx-v<ver>-...tar.gz -> .cosign.bundle (needs cosign + browser OIDC)
	bash scripts/release/sign_artifact.sh

bottle:          ## build a Homebrew bottle from the installed rmlx keg (run after brew install --build-bottle)
	bash scripts/release/build_bottle.sh

tap-sync:        ## copy packaging/homebrew/rmlx.rb into the homebrew-rmlx tap and push
	bash scripts/release/sync_tap.sh

clean:           ## cargo clean
	cargo clean

# ---- run --------------------------------------------------------------
serve:           ## rmlx serve --model $(MODEL) --port $(PORT)
	cargo run --release --bin rmlx -- serve --model "$(MODEL)" --port $(PORT)

chat:            ## rmlx chat --model $(MODEL)
	cargo run --release --bin rmlx -- chat --model "$(MODEL)"

info:            ## rmlx info --model $(MODEL)
	cargo run --release --bin rmlx -- info --model "$(MODEL)"

# ---- logs + metrics (retention is append-only; see CLAUDE.md) ---------
logs-tail:       ## tail newest log file
	@ls -1t logs/*.jsonl 2>/dev/null | head -1 | xargs -I{} tail -f {}

metrics-summary: ## cat .rmlx/metrics/summary.csv (rolling)
	@RMLX_HOME="$${RMLX_HOME:-$$PWD/.rmlx}"; \
		test -f "$$RMLX_HOME/metrics/summary.csv" && cat "$$RMLX_HOME/metrics/summary.csv" || echo "no metrics yet"

metrics-init:    ## Initialize the metrics SQLite DB at metrics/runs.db (refuses if file exists)
	cargo run --release --bin rmlx -- metrics init

metrics-doctor:  ## Validate schema, FKs, whitelists, units/directions
	cargo run --release --bin rmlx -- metrics doctor

metrics-doctor-fix: ## Same as metrics-doctor with --fix for safe auto-repairs
	cargo run --release --bin rmlx -- metrics doctor --fix

metrics-export:  ## Regenerate BENCHMARK_CHAMPIONS.md (gitignored) from current export view
	RUST_LOG=error cargo run --release --bin rmlx -- metrics export --markdown > BENCHMARK_CHAMPIONS.md

metrics-backup:  ## VACUUM INTO snapshot of metrics/runs.db (default metrics/backups/runs-<ts>.db)
	cargo run --release --bin rmlx -- metrics backup --keep 30

metrics-replay-pending: ## Replay any orphaned metrics/buffer/pending/*.json
	cargo run --release --bin rmlx -- metrics record --replay-pending

metrics-prompts-sync: ## Sync rMLX/prompts/*.json into the prompts table
	cargo run --release --bin rmlx -- metrics prompts sync

metrics-champions: ## Print champion table (all backends, markdown)
	cargo run --release --bin rmlx -- metrics champions

metrics-champions-rmlx: ## Print champion table (rMLX only, markdown)
	cargo run --release --bin rmlx -- metrics champions --backend rmlx

ci-metrics:      ## Verify metrics DB sanity via doctor (no-op if DB absent)
	@RMLX_HOME="$${RMLX_HOME:-$$PWD/.rmlx}"; \
	if [ ! -f "$$RMLX_HOME/metrics/runs.db" ]; then \
		echo "$$RMLX_HOME/metrics/runs.db absent — skipping ci-metrics (run 'make metrics-init' once)"; \
		exit 0; \
	fi
	cargo run --release --bin rmlx -- metrics doctor

# ---- profiling (on-demand; NOT part of make ci) ---------------------------
# See docs/PROFILING.md for full runbook.
profile-samply:  ## samply record rmlx baseline (CPU sampling; opens Firefox Profiler)
	@command -v samply >/dev/null 2>&1 || { echo "install: cargo install samply"; exit 1; }
	samply record --rate 4000 -- \
	  ./target/release/rmlx baseline --model "$(MODEL)" --kv-quant k8v8

# release-debug = opt-level=3 + LTO + full DWARF, so inlined frames resolve in the
# flamegraph. Decode-focused defaults (small prefill, long decode) so steady-state
# decode dominates the samples, not prefill. Override: PROF_PROMPT, PROF_GEN, MODEL.
profile-samply-debug: build-debug ## samply flamegraph on release-debug (full DWARF, decode-focused)
	@command -v samply >/dev/null 2>&1 || { echo "install: cargo install samply && samply setup"; exit 1; }
	@pkill -f "rmlx serve" || true; rm -f /tmp/rmlx.*.claim
	samply record --rate 4000 -- \
	  ./target/release-debug/rmlx baseline --model "$(MODEL)" \
	  --prompt-tokens $(PROF_PROMPT) --max-tokens $(PROF_GEN) --max-ctx 8192

profile-instruments: ## xcrun xctrace Time Profiler on rmlx baseline (opens .trace)
	xcrun xctrace record \
	  --template 'Time Profiler' \
	  --launch -- ./target/release/rmlx baseline --model "$(MODEL)" --kv-quant k8v8

# ---- micro-benchmarks (on-demand; NOT part of make ci) --------------------
bench:           ## cargo bench -p rmlx-quant --bench dequant  (on-demand; not in CI)
	cargo bench -p rmlx-quant --bench dequant

# ---- perf-iter regression bench (on-demand; NOT part of make ci) ---------------
# Runs the 3-model regression bench in series (one process at a time — Apple
# Silicon Metal context rule).  Appends results to metrics/perf-iter/baseline.jsonl.
# Use diff_baseline.sh afterwards to compare against a saved snapshot.
#
# Env-var overrides (all optional):
#   MEASURE_RUNS=5   — more samples for stable per-finding numbers
#   WARMUP_RUNS=2    — extra warmup for cold GPU
#   METRICS_OUT=...  — redirect output (default: metrics/perf-iter/baseline.jsonl)
PERF_ITER_E2B  ?= $(O_MODELS_ROOT)/mlx-community__gemma-4-e2b-it-mxfp8
PERF_ITER_E4B  ?= $(O_MODELS_ROOT)/mlx-community__gemma-4-e4b-it-mxfp8
PERF_ITER_BNSI ?= $(O_MODELS_ROOT)/prism-ml__Ternary-Bonsai-8B-mlx-2bit

perf-iter:       ## run 3-model regression bench in series (appends to metrics/perf-iter/baseline.jsonl)
	@echo "=== perf-iter: gemma-4-e2b k8v8 ==="
	MODEL_PATH="$(PERF_ITER_E2B)"  KV_QUANT=k8v8 bash scripts/perf-iter/bench_decode_tps.sh
	@echo "=== perf-iter: gemma-4-e4b k8v8 ==="
	MODEL_PATH="$(PERF_ITER_E4B)"  KV_QUANT=k8v8 bash scripts/perf-iter/bench_decode_tps.sh
	@echo "=== perf-iter: Ternary-Bonsai-8B k8v4 ==="
	MODEL_PATH="$(PERF_ITER_BNSI)" KV_QUANT=k8v4 bash scripts/perf-iter/bench_decode_tps.sh
	@echo "=== perf-iter done. Compare with: bash scripts/perf-iter/diff_baseline.sh <prev.jsonl> metrics/perf-iter/baseline.jsonl ==="

# ---- canary TPS gate (DB-backed, release-perf binary) -----------------
#
# `make canary`      — runs perf_canary.sh (1 warmup + 3 measured per model),
#                      appends rows to both the LEGACY .rmlx/bench/perf_canary.csv
#                      AND (authoritative) runs.db via `rmlx baseline --record`.
#                      Requires the release-perf binary; build with `make build-perf`.
#
# `make canary-gate` — gates regressions by querying runs.db.
#                      Requires SHA=<last-green-sha> to compare against.
#                      Uses `rmlx metrics deltas --since-sha <SHA> --threshold-pct 3`.
#                      Exit codes: 0=clean, 1=regression, 125=no-baseline-skip.
#                      For the simulated-regression test, use CANARY_DB=/tmp/... to point at
#                      a temp DB so real runs.db is not polluted with fake rows.
#
# Protocol: --prompt-tokens 4096, --max-tokens 100, --max-ctx 8192, kv_quant=auto
# (arch resolver picks best-known quant: Bonsai→mixed_k8g64_v4g64, Gemma4-e4b→k8v8, Qwen3.6→k8v8)

CANARY_THRESHOLD_PCT ?= 3
# SHA to compare against for canary-gate; required — no default.
# Usage: make canary-gate SHA=3ba8aee

canary: mlx-preflight build-perf  ## run 3-model TPS canary (records into runs.db + legacy CSV); requires release-perf binary
	@pkill -f "rmlx serve" || true; pkill -f mlx_lm || true; sleep 1; rm -f /tmp/rmlx.*.claim
	bash scripts/perf_canary.sh

# The canary tracks ONE build over time. Comparing two builds (or two flag
# settings) needs the interleaved harness, or ordering and thermal drift land
# entirely on whichever arm ran second. See docs/PERF_BASELINE.md.
#
# Pass arms through ARGS, e.g.
#   make canary-ab ARGS='--binary-a target/release-perf/rmlx.main --binary-b target/release-perf/rmlx'
canary-ab: mlx-preflight build-perf  ## interleaved A/B of two arms (ARGS='--binary-a … --binary-b …'); see docs/PERF_BASELINE.md
	bash scripts/perf_canary.sh --ab $(ARGS)

canary-ab-selftest: ## mutation-check the A/B harness against stub binaries (no GPU, no model)
	bash scripts/perf_ab_selftest.sh

canary-gate:        ## gate TPS regressions via runs.db (SHA= required; e.g. make canary-gate SHA=3ba8aee)
	@test -n "$(SHA)" || { echo "ERROR: SHA= required. Usage: make canary-gate SHA=<last-green-sha>"; exit 125; }
	@RMLX_HOME="$${RMLX_HOME:-$$PWD/.rmlx}"; \
	DB_PATH="$${CANARY_DB:-$$RMLX_HOME/metrics/runs.db}"; \
	if [ ! -f "$$DB_PATH" ]; then \
		echo "skip: runs.db not found at $$DB_PATH (run 'make canary' first)"; \
		exit 125; \
	fi; \
	echo "==> canary-gate: comparing vs SHA=$(SHA) threshold=$(CANARY_THRESHOLD_PCT)%"; \
	RMLX_METRICS_DB="$$DB_PATH" cargo run --release --bin rmlx -- \
		metrics deltas \
		--since-sha "$(SHA)" \
		--threshold-pct $(CANARY_THRESHOLD_PCT) \
		--exit-code true

# ---- Spec-decode canary (MTP / DFlash / Eagle3 accept_rate gate) -------------
#
# `make spec-canary` — runs scripts/spec_bench.sh for gemma-4-e2b normal + MTP
#                      across the three canonical prompt classes (prose,
#                      structured, code). Ingests decode_tps_warm + spec-decode
#                      metrics (accept_rate, accepted_per_step, *_total counters)
#                      into runs.db.
#
# `make spec-canary-gate` — same SHA-based deltas gate as `make canary-gate`,
#                           but tagged so accept_rate and decode_tps_warm
#                           regressions on spec-decode cells get caught.
#
# Required env: VERIFIER_MODEL, DRAFTER_MODEL (resolve via LOCAL.md, gitignored).
# Threshold: shared with canary-gate (CANARY_THRESHOLD_PCT, default 3%).
#
# accept_rate is registered as `higher_better` in METRICS, so the existing
# `metrics deltas` direction-aware regression check fires automatically when
# accept_rate drops more than CANARY_THRESHOLD_PCT% vs the SHA baseline.

spec-canary: build-perf  ## run spec-decode canary (normal+MTP × 3 prompt classes); requires VERIFIER_MODEL + DRAFTER_MODEL env
	@test -n "$$VERIFIER_MODEL" || { echo "ERROR: VERIFIER_MODEL= required (resolve via LOCAL.md)"; exit 125; }
	@test -n "$$DRAFTER_MODEL"  || { echo "ERROR: DRAFTER_MODEL= required (resolve via LOCAL.md)";  exit 125; }
	@pkill -f "rmlx serve" || true; pkill -f mlx_lm || true; sleep 1; rm -f /tmp/rmlx.*.claim
	@echo "==> spec-canary: prose"
	BENCH_PROMPT_FILE=prompts/spec_bench/prose.json \
		bash scripts/spec_bench.sh --tag canary-prose
	@echo "==> spec-canary: structured (fibonacci)"
	BENCH_PROMPT_FILE=prompts/spec_bench/structured.json \
		bash scripts/spec_bench.sh --tag canary-structured
	@echo "==> spec-canary: code (fizzbuzz Rust)"
	BENCH_PROMPT_FILE=prompts/spec_bench/code.json \
		bash scripts/spec_bench.sh --tag canary-code

spec-canary-gate:   ## gate spec-decode regressions (decode_tps_warm + accept_rate); SHA= required
	@test -n "$(SHA)" || { echo "ERROR: SHA= required. Usage: make spec-canary-gate SHA=<last-green-sha>"; exit 125; }
	@RMLX_HOME="$${RMLX_HOME:-$$PWD/.rmlx}"; \
	DB_PATH="$${CANARY_DB:-$$RMLX_HOME/metrics/runs.db}"; \
	if [ ! -f "$$DB_PATH" ]; then \
		echo "skip: runs.db not found at $$DB_PATH (run 'make spec-canary' first)"; \
		exit 125; \
	fi; \
	echo "==> spec-canary-gate: comparing vs SHA=$(SHA) threshold=$(CANARY_THRESHOLD_PCT)%"; \
	RMLX_METRICS_DB="$$DB_PATH" cargo run --release --bin rmlx -- \
		metrics deltas \
		--since-sha "$(SHA)" \
		--threshold-pct $(CANARY_THRESHOLD_PCT) \
		--exit-code true

# ---- SSD-tier canary (POPULATE / REVISIT / EVICT) -------------------------
#
# `make ssd-canary` — runs scripts/ssd_canary.sh end-to-end against VERIFIER_MODEL.
#                     Spawns three server phases (POPULATE, REVISIT, EVICT), ingests
#                     per-phase observations tagged ssd-canary-{populate,revisit,evict}
#                     into runs.db, and writes CSVs + iteration_summary.json under
#                     .rmlx/proofs/step3-canary/.
#
# `make ssd-canary-gate SHA=<sha>` — queries runs.db via `rmlx metrics deltas`
#                     against the recorded baseline SHA. Direction-aware: higher-is-
#                     better metrics (ssd_spill_mb_per_s, ssd_hydrate_mb_per_s,
#                     prompt_cache_ssd_hits) flag on drop; lower-is-better metrics
#                     (ssd_spill_ms, ssd_hydrate_ms) flag on rise. Exits non-zero on
#                     regression beyond CANARY_THRESHOLD_PCT (default 3%).
#
# Required env: VERIFIER_MODEL (resolve via LOCAL.md, gitignored).
# Optional: SSD_GB (default 100), RMLX_HOME (default $PWD/.rmlx),
#           CANARY_DB (default $RMLX_HOME/metrics/runs.db),
#           CANARY_THRESHOLD_PCT (default 3).
# See docs/SSD_CANARY.md for the full env-var table and phase descriptions.

ssd-canary: build-perf  ## run SSD canary (POPULATE/REVISIT/EVICT) against VERIFIER_MODEL
	@test -n "$$VERIFIER_MODEL" || { echo "ERROR: VERIFIER_MODEL= required (resolve via LOCAL.md)"; exit 125; }
	@pkill -f "rmlx serve" || true; pkill -f mlx_lm || true; sleep 1; rm -f /tmp/rmlx.*.claim
	@echo "==> ssd-canary: populate + revisit + evict"
	bash scripts/ssd_canary.sh --tag ssd-canary --ssd-gb $${SSD_GB:-100}

ssd-canary-gate:   ## gate SSD-tier regressions; SHA= required, THRESHOLD_PCT=3 default
	@test -n "$(SHA)" || { echo "ERROR: SHA= required. Usage: make ssd-canary-gate SHA=<last-green-sha>"; exit 125; }
	@RMLX_HOME="$${RMLX_HOME:-$$PWD/.rmlx}"; \
	DB_PATH="$${CANARY_DB:-$$RMLX_HOME/metrics/runs.db}"; \
	if [ ! -f "$$DB_PATH" ]; then \
		echo "skip: runs.db not found at $$DB_PATH (run 'make ssd-canary' first)"; \
		exit 125; \
	fi; \
	echo "==> ssd-canary-gate: comparing vs SHA=$(SHA) threshold=$${CANARY_THRESHOLD_PCT:-3}%"; \
	RMLX_METRICS_DB="$$DB_PATH" cargo run --release --bin rmlx -- \
		metrics deltas \
		--since-sha "$(SHA)" \
		--threshold-pct $${CANARY_THRESHOLD_PCT:-3} \
		--exit-code true

# ---- Per-codec × per-model cell bench ------------------------------------
#
# Run a single-codec, single-model bench and append 3 rows to
# .rmlx/bench/codec_cells.csv.
#
# Usage:
#   make bench-codec-cell CODEC=<codec> MODEL=<snapshot-abs-path>
#
# Examples:
#   make bench-codec-cell CODEC=k8v4 MODEL=/path/to/mlx-community__gemma-4-e4b-it-mxfp8
#   make bench-codec-cell CODEC=iso3_sym MODEL=/path/to/prism-ml__Ternary-Bonsai-8B-mlx-2bit
#
# See scripts/bench_codec_cell.sh for full options (--max-tokens, --prompt-len).
# CSV schema documented in docs/PERF_BASELINE.md § "Per-codec × per-model cells".

bench-codec-cell: mlx-preflight  ## bench one codec × one model cell (CODEC= MODEL= required); appends to .rmlx/bench/codec_cells.csv
	@if [ -z "$(CODEC)" ] || [ -z "$(MODEL)" ]; then \
		echo "Usage: make bench-codec-cell CODEC=<codec> MODEL=<snapshot-abs-path>"; \
		exit 2; \
	fi
	bash scripts/bench_codec_cell.sh --kv-quant $(CODEC) --model $(MODEL)

# ---- Codec smoke + NIAH gate matrix -------------------------------------
# Drive scripts/release_e2e/stage6_perf/codec_smoke_runner.sh over
# kv_codec_matrix.toml. Honours single-MLX-process discipline (CLAUDE.md
# hard rule 8) via the runner's preflight. Self-hosted Metal required.
#
# Examples:
#   make smoke-codec-matrix                            # full matrix
#   make smoke-codec-matrix CODEC=k8v4                 # filter to one codec
#   make smoke-codec-matrix MATRIX_MODEL=bonsai-8b     # filter to one model
#   make smoke-codec-matrix RECORD=1                   # populate baselines (Exec B)
#
# CODEC / MATRIX_MODEL match the manifest's `codec_name` / `model` primary-key
# fields. The variable is `MATRIX_MODEL` (not `MODEL`) because the workspace
# `MODEL ?= …/gemma-4-e4b-it-mxfp8` default would otherwise leak into the
# filter for `make smoke-codec-matrix` with no explicit `MODEL=` override.

smoke-codec-matrix:  ## codec smoke + NIAH gate matrix (CODEC= MATRIX_MODEL= RECORD=1 optional)
	@args=""; \
		if [ -n "$(CODEC)" ]; then args="$$args --filter codec_name=$(CODEC)"; fi; \
		if [ -n "$(MATRIX_MODEL)" ]; then args="$$args --filter model=$(MATRIX_MODEL)"; fi; \
		if [ -n "$(RECORD)" ]; then args="$$args --record-baseline"; fi; \
		bash scripts/release_e2e/stage6_perf/codec_smoke_runner.sh $$args

# ---- asm inspection (on-demand; requires cargo-asm: cargo install cargo-asm) --
# Usage: make asm  (shows dequant_to_f32 in rmlx-quant by default)
# Override symbol: make asm ASM_SYM=rmlx_quant::mxfp::dequant_to_f32
ASM_SYM ?= rmlx_quant::affine::dequant_to_f32
asm:             ## cargo asm --release -p rmlx-quant $(ASM_SYM)  (codegen inspection)
	@command -v cargo-asm >/dev/null 2>&1 || { echo "install: cargo install cargo-asm"; exit 1; }
	cargo asm --rust --release -p rmlx-quant "$(ASM_SYM)"

# ---- J10: TheTom upstream kernel-fix watch (weekly; docs/research/J10-upstream-watch.md)
# After triaging the printed commits, bump LAST_REVIEWED_SHA below + in the doc.
UPSTREAM_REPO ?= ../llama-cpp-turboquant
UPSTREAM_BRANCH ?= feature/turboquant-kv-cache
LAST_REVIEWED_SHA ?= 2b61ea24e
upstream-check:  ## J10: print TheTom commits since LAST_REVIEWED_SHA on the watched branch
	@test -d "$(UPSTREAM_REPO)/.git" || { echo "upstream repo absent: $(UPSTREAM_REPO)"; exit 1; }
	@git -C "$(UPSTREAM_REPO)" fetch --quiet origin "$(UPSTREAM_BRANCH)" 2>/dev/null || git -C "$(UPSTREAM_REPO)" fetch --quiet 2>/dev/null || true
	@echo "new commits on $(UPSTREAM_BRANCH) since $(LAST_REVIEWED_SHA):"
	@git -C "$(UPSTREAM_REPO)" log --oneline "$(LAST_REVIEWED_SHA)..origin/$(UPSTREAM_BRANCH)" 2>/dev/null \
	  || git -C "$(UPSTREAM_REPO)" log --oneline "$(LAST_REVIEWED_SHA)..$(UPSTREAM_BRANCH)" 2>/dev/null \
	  || echo "(none, or SHA not in branch — verify LAST_REVIEWED_SHA)"
