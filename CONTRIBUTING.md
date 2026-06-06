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
```

`make ci` must be green before you open a PR. The per-commit hook runs the fast
checks; `cargo audit` / `cargo deny` are gated behind `make ci` (or
`pre-commit run --hook-stage manual`).

Model-touching changes: see the regression-bench discipline in
[`CLAUDE.md`](CLAUDE.md). At minimum the three test-target families (Gemma4,
Qwen3.6, Bonsai) must still serve, each at its best-known KV quant, within ±1%
of the recorded decode TPS.

## Workflow

1. Branch from `main` (`feat/…`, `fix/…`, `chore/…`).
2. Keep changes surgical — match existing style, no drive-by refactors.
3. Tests live in sibling `*_tests.rs` files (no inline `#[cfg(test)] mod`
   blocks — `make check-no-inline-tests` enforces this).
4. `make ci` green locally.
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
