#!/usr/bin/env bash
# scripts/check_spec_charge_fixtures.sh — recall test for
# scripts/check_spec_charge.sh.
#
# A gate is only worth its runtime if it fires, and "the tree is clean" is no
# evidence of that: a scan whose regex stopped matching reports the same clean
# tree as a scan that works. Each case below is one edit to a synthetic source
# root, and each asserts the exit code AND the reason — a case that fails for
# the wrong reason has told us nothing about the rule it was meant to exercise.
#
# The clean root is built here rather than copied from the tree, so a case that
# passes against it is passing against a shape this file states rather than
# against whatever the repository happens to contain today. It carries the same
# census the tree does — three loops that charge their phases and four that do
# not — so the gate's own defaults are what the cases run against.

set -uo pipefail

script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_spec_charge.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

failures=0
cases=0

# One round loop: a rollback whose charge argument and a record whose `charged:`
# field name the same decision.
loop_src() {
  local name="$1" token="$2" bind="$3"
  if [ "$bind" = "bind" ]; then
    printf 'pub fn %s(\n    verifier: &Architecture,\n    device: Device,\n) -> Result<()> {\n    let charge_phases = super::phases_charged();\n' "$name"
  else
    printf 'pub fn %s(\n    verifier: &Architecture,\n    device: Device,\n) -> Result<()> {\n' "$name"
  fi
  cat <<RS
    while emitted.len() < n_tokens {
        if v_target < v_offset_before {
            super::rollback_round_caches(
                &mut v_caches,
                Some(&mut v_lin),
                &v_input,
                v_pre_round_offset,
                v_target,
                // This loop's one charge decision.
                $token,
                device,
            )?;
        }
        super::RoundPhases {
            round_ns: 0,
            charged: $token,
        }
        .log(SpecLoop::Kind, rounds, accept, num_draft, &[]);
    }
    Ok(())
}
RS
}

build_root() {
  local root="$1"
  rm -rf "$root"
  mkdir -p "$root/crates/rmlx-models/src/speculative/dflash2"

  {
    printf '//! Two two-model loops and a shared helper.\n\n'
    loop_src "spec_generate_greedy_cached" "false" "plain"
    printf '\n'
    loop_src "spec_generate_stochastic_cached" "false" "plain"
    printf '\nfn rollback_round_caches(\n    caches: &mut [KvCache],\n    lin: Option<&mut [LinearAttnCache]>,\n    fed: &[u32],\n    pre_round_offset: i32,\n    target: i32,\n    charge: bool,\n    device: Device,\n) -> Result<()> {\n    Ok(())\n}\n'
  } >"$root/crates/rmlx-models/src/speculative/mod.rs"

  loop_src "mtp_generate" "charge_phases" "bind" \
    >"$root/crates/rmlx-models/src/speculative/mtp.rs"
  loop_src "mtp_assistant_generate" "charge_phases" "bind" \
    >"$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
  loop_src "dflash2_generate" "charge_phases" "bind" \
    >"$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
  loop_src "dflash_generate" "false" "plain" \
    >"$root/crates/rmlx-models/src/speculative/dflash.rs"
  loop_src "eagle3_generate" "false" "plain" \
    >"$root/crates/rmlx-models/src/speculative/eagle3.rs"

  # A sibling test file, which the scan must not read: the same shapes appear in
  # tests deliberately and are not round loops.
  loop_src "a_charged_round_fixture" "true" "plain" \
    >"$root/crates/rmlx-models/src/speculative/round_stats_tests.rs"
}

# run <name> <expected-exit> <reason-substring>
run() {
  local name="$1" want_exit="$2" want_reason="$3"
  cases=$((cases + 1))
  local out rc
  out="$(SPEC_CHARGE_ROOT="$work/root" bash "$script" 2>&1)"
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
run "a clean root passes" 0 "7 speculative round loops, each naming one charge decision"

# 1. The rule the gate exists for: one loop whose rollback and record disagree.
#    The record says the round was not charged, the rollback charges it, and
#    every token and every count the loop reports is unchanged.
build_root "$root"
perl -0pi -e 's/            charged: charge_phases,/            charged: false,/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a loop whose rollback and record disagree is refused" 1 \
  "\`mtp_generate\` names more than one charge decision"

# 2. The same defect the other way round, on a loop that charges nothing.
build_root "$root"
perl -0pi -e 's/                false,\n                device,/                charge_phases,\n                device,/' \
  "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "a rollback that charges where the record does not is refused" 1 \
  "\`eagle3_generate\` names more than one charge decision"

# 3. A value wired in rather than decided.
build_root "$root"
perl -0pi -e 's/\bfalse\b/true/g' "$root/crates/rmlx-models/src/speculative/dflash.rs"
run "a hard-wired \`true\` is refused" 1 \
  "the charge census is \"charge_phases:3 false:3 true:1\""

# 4. A loop moved from one schedule to the other without the census moving.
build_root "$root"
perl -0pi -e 's/\bfalse\b/charge_phases/g' "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "a loop moved between the two schedules is refused" 1 \
  "the charge census is \"charge_phases:4 false:3\""

# 5. A loop the scan lost. Deleting one is the shape a rename produces, and a
#    loop with no gate on it reads as a pass.
build_root "$root"
rm -f "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "fewer round loops than the tree ships is a scan error" 2 \
  "scanned 6 round loops and the tree ships 7"

# 6. A call the scan cannot read the arguments of. One line is a legal Rust
#    shape and an unread call is not a checked one.
build_root "$root"
perl -0pi -e 's/            super::rollback_round_caches\(\n(?:.*\n)*?            \)\?;/            super::rollback_round_caches(\&mut v_caches, None, \&v_input, 0, 0, false, device)?;/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a call written on one line is a scan error" 2 \
  "\`mtp_generate\` makes a \`rollback_round_caches\` call this"

# 7. A call with an argument dropped is read back as the wrong position, so the
#    scan refuses it rather than reporting whatever landed there.
build_root "$root"
perl -0pi -e 's/                &v_input,\n//' \
  "$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
run "a call with an argument dropped is a scan error" 2 \
  "\`dflash2_generate\` makes a \`rollback_round_caches\` call this"

# 8. The scan finding nothing must not pass. This is what a rename of both
#    spellings produces, and it is the one a gate reports as clean.
build_root "$root"
perl -0pi -e 's/rollback_round_caches/rewind_round_caches/g; s/charged:/billed:/g' \
  "$root/crates/rmlx-models/src/speculative"/*.rs \
  "$root/crates/rmlx-models/src/speculative/dflash2"/*.rs
run "a scan that matches no charge site is a scan error, not a pass" 2 \
  "found no charge site"

# 9. A missing tree is a scan error.
build_root "$root"
rm -rf "$root/crates/rmlx-models/src/speculative"
run "a missing speculative tree is a scan error" 2 "no speculative source directory"

# 10. A sibling test file carries the same shapes deliberately — a `charged:
#     true` fixture among them — and must not be scanned.
build_root "$root"
run "a charge-shaped fn in a sibling test file is not scanned" 0 \
  "census charge_phases:3 false:4"

echo
if [ "$failures" != "0" ]; then
  echo "check-spec-charge-fixtures: $failures of $cases cases failed"
  exit 1
fi
echo "OK: $cases cases, every rule fired for its own reason."
