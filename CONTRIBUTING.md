# Contributing to rMLX

Thanks for your interest. rMLX is a Rust-native, single-binary MLX inference +
conversion backend for Apple Silicon. A few things up front:

- **Apple Silicon only.** Metal first. No CUDA, no ROCm, no x86 SIMD. You need a
  real Apple-Silicon Mac to build and test — GitHub-hosted macOS runners have no
  usable Metal device, so the full suite runs on hardware, not in hosted CI.
- **No Python at runtime.** One `cargo build --release` is the artifact.
- **MLX-format only.** GGUF is out of scope.
- **No training.** Quant / format conversion is in scope; fine-tune / fuse /
  lora-merge is not.

## Prerequisites

- macOS on Apple Silicon (M-series).
- Rust (MSRV **1.95**, edition 2021) — `rustup` recommended.
- MLX C bindings: `brew install mlx-c` (pulls `mlx`). The build links the
  `libmlxc` / `libmlx` dylibs from the Homebrew prefix.

## Build & test

```sh
make build          # cargo build --workspace --release
make check          # fast cargo check
make test           # workspace tests (needs a Metal GPU)
make ci             # fmt-check + clippy + test + deny + audit  ← pre-PR gate
make ci-perf        # test-perf + the GPU/Metal suite  ← also required, see below
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
stop any `rmlx serve` first — and takes around 21 minutes. While iterating,
`make gpu-test CRATE=… FILTER=…` runs a narrowed subset in seconds.

Model-touching changes: see the regression-bench discipline in
[`CLAUDE.md`](CLAUDE.md). At minimum the three test-target families (Gemma4,
Qwen3.6, Bonsai) must still serve, each at its best-known KV quant, within ±1%
of the recorded decode TPS.

## Workflow

1. Branch from `main` (`feat/…`, `fix/…`, `chore/…`).
2. Keep changes surgical — match existing style, no drive-by refactors.
3. Tests live in sibling `*_tests.rs` files (no inline `#[cfg(test)] mod`
   blocks — `make check-no-inline-tests` enforces this).
4. `make ci` green locally — plus `make ci-perf` for codec-layer / `.metal`
   changes (see §Build & test).
5. Open a PR into `main`. Fill in the PR template.

`main` is protected: changes land via PR with CI green, not direct pushes.

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
