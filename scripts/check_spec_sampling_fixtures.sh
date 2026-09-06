#!/usr/bin/env bash
# scripts/check_spec_sampling_fixtures.sh — recall test for
# scripts/check_spec_sampling.sh.
#
# A gate is only worth its runtime if it fires, and "the tree is clean" is no
# evidence of that: a scan whose regex stopped matching reports the same clean
# tree as a scan that works. Each case below is one edit to a synthetic source
# root, and each asserts the exit code AND the reason — a case that fails for
# the wrong reason has told us nothing about the rule it was meant to exercise.
#
# The clean root is built here rather than copied from the tree, so a case that
# passes against it is passing against a shape this file states rather than
# against whatever the repository happens to contain today.

set -uo pipefail

script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_spec_sampling.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

failures=0
cases=0

# Lay out a synthetic root: two drafter loops and a dispatch with two arms.
build_root() {
  local root="$1"
  rm -rf "$root"
  mkdir -p "$root/crates/rmlx-models/src/speculative/dflash2"
  mkdir -p "$root/crates/rmlx-server/src/engine"

  cat >"$root/crates/rmlx-models/src/speculative/mtp.rs" <<'RS'
pub fn mtp_generate(
    verifier: &Architecture,
    n_tokens: usize,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    let mut draw = super::VerifierDraw::new(sampler_cfg);
    if draw.sampling() {
        let _ = n_tokens;
    }
    Ok(vec![])
}

pub fn not_a_driver(verifier: &Architecture) -> usize {
    0
}
RS

  cat >"$root/crates/rmlx-models/src/speculative/dflash2/round.rs" <<'RS'
pub fn dflash2_generate(
    verifier: &Architecture,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    let mut draw = VerifierDraw::new(sampler_cfg);
    Ok(vec![])
}
RS

  # A sibling test file, which the scan must not read: the same shapes appear in
  # tests deliberately and are not production drivers.
  cat >"$root/crates/rmlx-models/src/speculative/mtp_tests.rs" <<'RS'
pub fn mtp_generate_harness(
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
) -> Result<Vec<ProbeStep>> {
    Ok(vec![])
}
RS

  cat >"$root/crates/rmlx-server/src/engine/speculative.rs" <<'RS'
fn drive() {
    let result = match &drafter {
        Drafter::MtpSidecar(d) => rmlx_models::speculative::mtp::mtp_generate(
            &dispatcher.verifier,
            n_tokens,
            &mut step_fn,
            &spec_sampler_cfg,
            dispatcher.device(),
        ),
        Drafter::DFlash2(d) => rmlx_models::speculative::dflash2::dflash2_generate(
            &dispatcher.verifier,
            &mut step_fn,
            &spec_sampler_cfg,
            dispatcher.device(),
        ),
    };
}
RS
}

# run <name> <expected-exit> <reason-substring>
run() {
  local name="$1" want_exit="$2" want_reason="$3"
  cases=$((cases + 1))
  local out rc
  out="$(SPEC_SAMPLING_ROOT="$work/root" bash "$script" 2>&1)"
  rc=$?
  if [ "$rc" != "$want_exit" ]; then
    printf 'FAIL %s: exit %s, expected %s\n%s\n' "$name" "$rc" "$want_exit" "$out"
    failures=$((failures + 1))
    return
  fi
  if [ -n "$want_reason" ] && ! printf '%s' "$out" | grep -qF -- "$want_reason"; then
    printf 'FAIL %s: exit %s was right but the reason was not.\n  wanted: %s\n  got:\n%s\n' \
      "$name" "$rc" "$want_reason" "$out"
    failures=$((failures + 1))
    return
  fi
  printf 'ok   %s (exit %s)\n' "$name" "$rc"
}

root="$work/root"

build_root "$root"
run "a clean root passes" 0 "take and read the request's sampler"

# 1. A loop that takes no sampler at all — the state every sidecar was in.
build_root "$root"
perl -0pi -e 's/    sampler_cfg: &crate::sampler::SamplerConfig,\n//' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
perl -0pi -e 's/super::VerifierDraw::new\(sampler_cfg\)/super::VerifierDraw::new(\&greedy())/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a driver with no sampler parameter is refused" 1 \
  "\`mtp_generate\` drives a generation but takes no"

# 2. A loop that takes the sampler and ignores it.
build_root "$root"
perl -0pi -e 's/VerifierDraw::new\(sampler_cfg\)/VerifierDraw::new(\&greedy())/' \
  "$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
run "a driver that never reads its sampler is refused" 1 \
  "\`dflash2_generate\` takes \`sampler_cfg\` and never reads it"

# 3. One dispatch arm dropping the configuration while the others keep it.
build_root "$root"
perl -0pi -e 's/            &mut step_fn,\n            &spec_sampler_cfg,\n            dispatcher.device\(\),\n        \),\n    \};/            &mut step_fn,\n            dispatcher.device(),\n        ),\n    };/' \
  "$root/crates/rmlx-server/src/engine/speculative.rs"
run "one dispatch arm dropping the sampler is refused" 1 \
  "the \`Drafter::DFlash2\` arm drives a"

# 4. A new driver added without a sampler — the sixth-loop case.
build_root "$root"
cat >>"$root/crates/rmlx-models/src/speculative/mtp.rs" <<'RS'

pub fn eagle9_generate(
    verifier: &Architecture,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    Ok(vec![])
}
RS
run "a newly added driver with no sampler is refused" 1 \
  "\`eagle9_generate\` drives a generation but takes no"

# 5. The scan finding nothing must not pass. This is the failure a renamed
#    argument would produce, and it is the one a gate reports as clean.
build_root "$root"
perl -0pi -e 's/step_fn: &mut dyn FnMut\(&ProbeStep\)/emit: \&mut dyn FnMut(\&Step)/g' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs" \
  "$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
run "a scan that matches no driver is a scan error, not a pass" 2 \
  "found no generation drivers"

# 6. A dispatch this gate cannot find is also a scan error.
build_root "$root"
perl -0pi -e 's/let result = match &drafter \{/let result = match drafter_kind {/' \
  "$root/crates/rmlx-server/src/engine/speculative.rs"
run "a dispatch the gate cannot read is a scan error, not a pass" 2 \
  "Rule 2 scanned nothing"

# 7. A missing tree is a scan error.
build_root "$root"
rm -rf "$root/crates/rmlx-models/src/speculative"
run "a missing speculative tree is a scan error" 2 \
  "no speculative source directory"

# 8. A missing dispatch is a scan error.
build_root "$root"
rm -f "$root/crates/rmlx-server/src/engine/speculative.rs"
run "a missing dispatch is a scan error" 2 "no speculative dispatch"

# 9. A sibling test file must not be scanned: its harness takes no sampler and
#    the gate must still pass.
build_root "$root"
run "a driver-shaped fn in a sibling test file is not scanned" 0 \
  "take and read the request's sampler"

echo
if [ "$failures" != "0" ]; then
  echo "check-spec-sampling-fixtures: $failures of $cases cases failed"
  exit 1
fi
echo "OK: $cases cases, every rule fired for its own reason."
