<!-- Keep it surgical. Match existing style. No drive-by refactors. -->

## What & why

<!-- One or two sentences. Link the issue if any (Closes #N). -->

## Changes

-

## Checklist

- [ ] `make ci` green locally (fmt + clippy + test + deny + audit)
- [ ] Tests added/updated in sibling `*_tests.rs` (no inline `#[cfg(test)] mod`)
- [ ] Docs/`CLAUDE.md` updated if behavior or layout changed
- [ ] **Model-touching change:** regression smoke run — Gemma4, Qwen3.6, Bonsai
      still serve at best-known KV quant, decode TPS within ±1% (N/A otherwise)
- [ ] No new `Cargo.toml` dependency without justification

## Notes for reviewer

<!-- Tradeoffs, follow-ups, anything non-obvious. -->
