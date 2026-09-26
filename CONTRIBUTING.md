# Contributing to rMLX

Thanks for your interest. rMLX is a Rust-native, single-binary MLX inference
backend for Apple Silicon. A few things up front:

- **Apple Silicon only.** Metal first. No CUDA, no ROCm, no x86 SIMD. You need a
  real Apple-Silicon Mac to build and test — GitHub-hosted macOS runners have no
  usable Metal device, so the full suite runs on hardware, not in hosted CI.
- **No Python at runtime.** One `cargo build --release` is the artifact.
- **MLX-format only.** GGUF is out of scope.
- **No training.** No fine-tuning, fusing or LoRA merging.

## Prerequisites

- macOS on Apple Silicon (M-series).
- Rust (MSRV **1.95**, edition 2021) — `rustup` recommended.
- MLX C bindings: `brew install mlx-c` (pulls `mlx`). The build links the
  `libmlxc` / `libmlx` dylibs from the Homebrew prefix.

## Build & test

```sh
make build          # cargo build --workspace --release
make check          # fast cargo check
make test           # workspace tests; skips the GPU tests
make ci             # the pre-PR gate: fmt, clippy, tests, deny, audit, CI gates
make ci-perf        # test-perf + the GPU/Metal suite; see below
```

`make ci` must be green before you open a PR. The per-commit hook runs the fast
checks; `cargo audit` / `cargo deny` are gated behind `make ci` (or
`pre-commit run --hook-stage manual`).

**`make ci` does not run the GPU tests.** Every test that reaches `Device::Gpu`
carries `#[ignore]` (a shared Metal context driven from parallel `cargo test`
threads aborts the whole binary), `make test` passes no `--ignored`, and the
hosted CI has no Metal at all. `make ci-perf` is the gate that runs them,
serialized and under Metal shader validation.

Run it as well as `make ci` if your change touches **`crates/rmlx-kv-quant`, any
`.metal` kernel, or a KV-cache / decode path**. It needs the GPU to itself —
stop any `rmlx serve` first. While iterating, `make gpu-test CRATE=… FILTER=…`
runs a narrowed subset.

Model-touching changes: see the regression-bench discipline in
[`CLAUDE.md`](CLAUDE.md). At minimum the three test-target families (Gemma4,
Qwen3.6, Bonsai) must still serve, each at its best-known KV quant, within ±1%
of the recorded decode TPS.

## Workflow

`main` holds released state only — tag it, or fast-forward it at release
time, but do not target it with a normal PR. Day-to-day work lands on the
current accumulation branch, `next/<name>` (ask a maintainer which one is
open, or check open PRs for the name).

1. Branch from `next/<name>`, not from `main` (`feat/…`, `fix/…`, `chore/…`).
   One issue gets one branch, one PR, and — once merged — one commit on
   `next/<name>`; commits inside the branch itself are unlimited.
2. Keep changes surgical — match existing style, no drive-by refactors.
3. Tests live in sibling `*_tests.rs` files (no inline `#[cfg(test)] mod`
   blocks — `make check-no-inline-tests` enforces this).
4. Keep the branch current with `next/<name>` by rebasing onto it — never
   merge `next/<name>` into the branch, and never open a second PR for the
   same issue.
5. `make ci` green locally on every chunk you push, plus `make ci-perf` when
   the change touches what §Build & test names. Every merge to `main` runs
   the whole `make ci-perf`.
6. Open a PR into `next/<name>`. The hosted checks
   (`.github/workflows/ci.yml`) run on every PR into `main` and into
   `next/<name>`, as the branch rulesets require. Fill in the PR template,
   including the `Removals` section — see below. A maintainer squash-merges it.

Both `main` and `next/<name>` are protected: changes land via PR with the
required checks green. Only a maintainer pushes to them directly, through the
ruleset bypass: the release fast-forward of `next/<name>` onto `main`, and the
rebase of `next/<name>` after a hotfix. A fix for a bug already released on
`main` branches from `main` directly as `hotfix/<issue>`. Both procedures are
in `docs/RELEASING.md`.

### No twins

Before adding a second copy of something, ask whether it is really a second
thing. Two types, functions, kernels, or files whose bodies differ only in a
compile-time constant (`bits`, `head_dim`, group size, a codebook) or in a
component's name are one item and a parameter — a const-generic or a
trait-bound blanket impl — not a file per variant. (This does not license a
generic with a single caller; that is still premature, see `CLAUDE.md`
§Simplicity rules.)

If your change touches speculative decoding's draft side, byte equality of
the greedy stream is not evidence — see the oracle rule in `CLAUDE.md` and
[`docs/SPEC_ANSWER_EQUIVALENCE.md`](docs/SPEC_ANSWER_EQUIVALENCE.md).

### Removals

Every PR names what it makes deletable — code, a gate, a flag, a doc section
— in the `Removals` section of the PR template. "Nothing" is a valid answer,
but write it down: deletion is a deliverable, not something that happens to
get scheduled later.

## Commit messages

Conventional-Commits style: `type(scope): summary`
(`feat`, `fix`, `perf`, `docs`, `test`, `chore`, `refactor`).

## Project layout

Workspace crates under `crates/rmlx-*`. Subsystem docs under `docs/` (see the
documentation map in [`CLAUDE.md`](CLAUDE.md)). Read the relevant doc before
touching a subsystem.

## License

By contributing you agree your work is dual-licensed under
**MIT OR Apache-2.0**, matching the project.
