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
# not — plus the three shapes that must not be counted as loops: a fn with a
# driver's signature and no `RoundTotals` (the entry guard), a fn that builds a
# `RoundTotals` and takes no `step_fn` (a summary helper), and the shared seed
# emit, which has the signature and is handed a loop's totals rather than naming
# any of its own. Each pins one of the derivation's two conditions.

set -uo pipefail

script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_spec_charge.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

failures=0
cases=0

# One round loop: a driver's signature, a rollback whose charge argument and a
# record whose `charged:` field name the same decision, and the `RoundTotals` a
# round loop closes on and hands to the shared recorder.
loop_src() {
  local name="$1" token="$2" bind="$3"
  printf 'pub fn %s(\n    verifier: &Architecture,\n    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,\n    device: Device,\n) -> Result<()> {\n' "$name"
  if [ "$bind" = "bind" ]; then
    printf '    let charge_phases = super::phases_charged();\n'
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
                // This loop's one charge decision, first of two calls.
                $token,
                device,
            )?;
        }
        if d_target < d_offset_before {
            super::rollback_round_caches(
                &mut d_caches,
                Some(&mut d_lin),
                &d_fed,
                d_pre_round_offset,
                d_target,
                // The same decision, at the draft-side rollback.
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
    log_request_record(
        &RoundTotals {
            loop_kind: SpecLoop::Kind,
            rounds,
            charged: $token,
        },
        &emitted,
        seed_emitted,
        &window,
    );
    Ok(())
}
RS
}

# A fn that builds a `RoundTotals` and drives no generation. It is not a round
# loop, and condition (a) of the derivation is the only thing that says so.
stats_helper_src() {
  cat <<'RS'
pub fn summarise_rounds(rounds: usize, totals: &RoundTotals) -> RoundTotals {
    RoundTotals { rounds, ..*totals }
}
RS
}

# The shared recorder: the one fn that writes the record, from a destructured
# binding rather than a named value. RULE 5 is about a `charged:` field outside
# a round loop, and this is the fn it has to be exercised against — a rule that
# only ever saw helpers nobody would write would not be checked where it matters.
recorder_src() {
  cat <<'RS'
pub(crate) fn round_stats(totals: &RoundTotals, emitted: &[ProbeStep]) -> RoundStats {
    let &RoundTotals {
        loop_kind,
        rounds,
        charged,
    } = totals;
    RoundStats {
        loop_kind,
        rounds,
        emitted: emitted.len(),
        charged,
    }
}
RS
}

# The shared seed emit: a driver's signature, and it names no decision of its
# own — it is handed the loop's totals. Condition (b) is the only thing keeping
# it out of the population, and a scan that counted it would report eight loops
# and a census nobody wrote down.
seed_emit_src() {
  cat <<'RS'
pub(crate) fn emit_seed_token(
    tokenizer: &tokenizers::Tokenizer,
    seed: u32,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    emitted: &mut Vec<ProbeStep>,
    window: &mut DecodeWindow,
    eos_ids: &[u32],
    totals: &RoundTotals,
) -> bool {
    emit_step(tokenizer, seed, step_fn, emitted, window);
    if !eos_ids.contains(&seed) {
        return false;
    }
    log_request_record(totals, emitted, emitted.len(), window);
    true
}
RS
}

# The entry guard: a driver's signature, no round, no `RoundTotals`. It delegates.
dispatcher_src() {
  cat <<'RS'
pub fn spec_generate_greedy(
    verifier: &Architecture,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    device: Device,
) -> Result<()> {
    if drafts_per_round == 0 {
        return Err(Error::Model("no drafts".into()));
    }
    spec_generate_greedy_cached(verifier, step_fn, device)
}
RS
}

build_root() {
  local root="$1"
  rm -rf "$root"
  mkdir -p "$root/crates/rmlx-models/src/speculative/dflash2"

  {
    printf '//! Two two-model loops, the entry guard, and the shared helper.\n\n'
    dispatcher_src
    printf '\n'
    loop_src "spec_generate_greedy_cached" "false" "plain"
    printf '\n'
    loop_src "spec_generate_stochastic_cached" "false" "plain"
    printf '\n'
    stats_helper_src
    printf '\n'
    recorder_src
    printf '\n'
    seed_emit_src
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

  # A sibling test file, which the scan must not read: a `charged: true` fixture
  # is a legitimate thing for a test to build and is not a round loop.
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

# run_env <name> <expected-exit> <reason-substring> <VAR=value> ...
run_env() {
  local name="$1" want_exit="$2" want_reason="$3"
  shift 3
  cases=$((cases + 1))
  local out rc
  out="$(env "$@" SPEC_CHARGE_ROOT="$work/root" bash "$script" 2>&1)"
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
run "a clean root passes, and the entry guard is not a loop" 0 \
  "7 speculative round loops, each naming one charge decision"

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

# 3. A record's LAST field carries no comma. A matcher that required one read
#    this loop as having no record site at all and passed it.
build_root "$root"
perl -0pi -e 's/            charged: false,\n        \},/            charged: true\n        },/' \
  "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "a record whose last field drops its comma is still read" 1 \
  "\`eagle3_generate\` names more than one charge decision"

# 4. A value that is not a bare identifier is unreadable, not absent.
build_root "$root"
perl -0pi -e 's/            charged: false,/            charged: self.charged,/' \
  "$root/crates/rmlx-models/src/speculative/dflash.rs"
run "a record whose value the scan cannot read is a scan error" 2 \
  "\`dflash_generate\` has a charge site this gate could not read"

# 5. A value wired in rather than decided.
build_root "$root"
perl -0pi -e 's/\bfalse\b/true/g' "$root/crates/rmlx-models/src/speculative/dflash.rs"
run "a hard-wired \`true\` is refused" 1 \
  "the charge census is \"charge_phases:3 false:3 true:1\""

# 6. The census reads a spelling unless the binding is checked: a loop that
#    calls its local `charge_phases` and binds it to a literal charges every
#    request and would be counted among the three that ask.
build_root "$root"
perl -0pi -e 's/\bfalse\b/charge_phases/g' "$root/crates/rmlx-models/src/speculative/eagle3.rs"
perl -0pi -e 's/(\) -> Result<\(\)> \{\n)/$1    let charge_phases = true;\n/' \
  "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "a loop that binds \`charge_phases\` to a literal is refused" 1 \
  "\`eagle3_generate\` charges on \`charge_phases\` and does not bind it"

# 7. A loop genuinely moved from one schedule to the other, binding and all.
build_root "$root"
perl -0pi -e 's/\bfalse\b/charge_phases/g' "$root/crates/rmlx-models/src/speculative/eagle3.rs"
perl -0pi -e 's/(\) -> Result<\(\)> \{\n)/$1    let charge_phases = super::phases_charged();\n/' \
  "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "a loop moved between the two schedules is refused" 1 \
  "the charge census is \"charge_phases:4 false:3\""

# 8. A loop the scan lost. The population is derived, so this moves the census
#    rather than a number typed into the gate.
build_root "$root"
rm -f "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "a loop deleted moves the census" 1 \
  "the charge census is \"charge_phases:3 false:3\""

# 9. A driver whose rollback moved into a helper. The loop still runs rounds and
#    still records a decision; the gate can no longer follow the other half of
#    it, and a loop it cannot follow is not a loop it checks.
build_root "$root"
perl -0pi -e 's/            super::rollback_round_caches\(\n(?:.*\n)*?            \)\?;/            roll_it(\&mut v_caches, false, device)?;/g' \
  "$root/crates/rmlx-models/src/speculative/dflash.rs"
cat >>"$root/crates/rmlx-models/src/speculative/dflash.rs" <<'RS'

fn roll_it(caches: &mut [KvCache], charge: bool, device: Device) -> Result<()> {
    super::rollback_round_caches(
        caches,
        None,
        &[],
        0,
        0,
        charge,
        device,
    )
}
RS
run "a driver whose rollback is delegated to a helper is a scan error" 2 \
  "\`dflash_generate\` has 0 rollback and 2 record charge sites."

# 10. A call the scan cannot read the arguments of. One line is a legal Rust
#     shape and an unread call is not a checked one.
build_root "$root"
perl -0pi -e 's/            super::rollback_round_caches\(\n(?:.*\n)*?            \)\?;/            super::rollback_round_caches(\&mut v_caches, None, \&v_input, 0, 0, false, device)?;/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a call written on one line is a scan error" 2 \
  "\`mtp_generate\` has a charge site this gate could not read"

# 11. A call with an argument dropped is read back at the wrong position, so the
#     scan refuses it rather than reporting whatever landed there.
build_root "$root"
perl -0pi -e 's/                &v_input,\n//' \
  "$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
run "a call with an argument dropped is a scan error" 2 \
  "\`dflash2_generate\` has a charge site this gate could not read"

# 12. Both spellings renamed: the loops are still derived and now carry no
#     decision at all. A scan that found nothing must not pass.
build_root "$root"
perl -0pi -e 's/rollback_round_caches/rewind_round_caches/g; s/charged:/billed:/g' \
  "$root/crates/rmlx-models/src/speculative"/*.rs \
  "$root/crates/rmlx-models/src/speculative/dflash2"/*.rs
run "both charge spellings renamed away is a scan error" 2 \
  "has 0 rollback and 0 record charge sites."

# 13. The population itself renamed out from under the gate.
build_root "$root"
perl -0pi -e 's/RoundTotals \{/RoundLedger {/g' \
  "$root/crates/rmlx-models/src/speculative"/*.rs \
  "$root/crates/rmlx-models/src/speculative/dflash2"/*.rs
run "a scan that derives no round loop is a scan error, not a pass" 2 \
  "found no round loop"

# 14. A missing tree is a scan error.
build_root "$root"
rm -rf "$root/crates/rmlx-models/src/speculative"
run "a missing speculative tree is a scan error" 2 "no speculative source directory"

# 15. The filename rule, pinned in both directions: the same file carrying a
#     `charged: true` loop is skipped as `*_tests.rs` and scanned once renamed.
build_root "$root"
mv "$root/crates/rmlx-models/src/speculative/round_stats_tests.rs" \
  "$root/crates/rmlx-models/src/speculative/round_stats_fixtures.rs"
run "a charge-shaped loop is scanned once its file is no longer a test file" 1 \
  "the charge census is \"charge_phases:3 false:4 true:1\""

# 16. The token means the call and nothing else. ORing a second condition into
#     it is RULE 2's own stated defect wearing another spelling, and the
#     unanchored regex it replaced read this as a clean binding.
build_root "$root"
perl -0pi -e 's/let charge_phases = super::phases_charged\(\);/let charge_phases = super::phases_charged() || force_charge;/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a binding that ORs a second condition into the call is refused" 1 \
  "\`mtp_generate\` charges on \`charge_phases\` and does not bind it"

# 17. A binding shape the scan cannot read is a scan error, not a verdict about
#     the loop: refusing it as RULE 2 would name a defect that is not there.
build_root "$root"
perl -0pi -e 's/let charge_phases = super::phases_charged\(\);/let charge_phases: bool = super::phases_charged();/' \
  "$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
run "a binding this gate cannot read is a scan error, not a RULE 2 refusal" 2 \
  "\`dflash2_generate\` binds \`charge_phases\` in a shape this gate"

# 18. Every rollback, not the first one. A loop's second call carrying the other
#     decision is the same defect at a site a first-match scan never reaches.
build_root "$root"
perl -0pi -e 's/                \/\/ The same decision, at the draft-side rollback\.\n                charge_phases,/                \/\/ The same decision, at the draft-side rollback.\n                false,/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a second rollback call carrying the other decision is refused" 1 \
  "\`mtp_generate\` names more than one charge decision"

# 19. The census is a constant, not a default. A gate whose expectation can be
#     relaxed from the environment passes for whoever sets it.
build_root "$root"
run_env "the census cannot be waived from the environment" 0 \
  "census charge_phases:3 false:4" \
  WANT_CENSUS="charge_phases:9 false:9" SPEC_CHARGE_WANT_CENSUS="charge_phases:9 false:9"

# 20. A loop that hands the recorder totals built somewhere else names no
#     decision of its own and drops out of the derived population. On its own
#     that reads as a census that moved, which invites the census to be edited;
#     the rollback it still makes says otherwise.
build_root "$root"
perl -0pi -e 's/    log_request_record\(\n        &RoundTotals \{\n(?:.*\n)*?    \);/    log_request_record(&totals_for(rounds, false), &emitted, seed_emitted, &window);/' \
  "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "a loop that falls out of the population is not reported as a census move" 2 \
  "\`eagle3_generate\` rolls a round's caches back and is"

# 21. The same, from the other condition: a signature wrapped so the marker no
#     longer reads on one line. This is what rustfmt does to a longer parameter.
build_root "$root"
perl -0pi -e 's/    step_fn: &mut dyn FnMut\(&ProbeStep\) -> Option<u32>,/    step_fn: \&mut dyn FnMut(\n        \&ProbeStep,\n    ) -> Option<u32>,/' \
  "$root/crates/rmlx-models/src/speculative/dflash.rs"
run "a signature wrapped past the marker is not reported as a census move" 2 \
  "\`dflash_generate\` rolls a round's caches back and is"

# 22. A `where` clause sits between the closing parenthesis and the body. A scan
#     that opens the body at the parenthesis reads the clause as the body and
#     the body as nothing.
build_root "$root"
perl -0pi -e 's/\) -> Result<\(\)> \{/) -> Result<()>\nwhere\n    D: Drafter,\n{/' \
  "$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
run "a loop with a \`where\` clause is still read" 0 \
  "7 speculative round loops, each naming one charge decision"

# 23. RULE 5, at the shape that opened it: the recorder reading the decision off
#     the totals it was handed. The value is unreadable to this scan, and an
#     unreadable value was refused only inside a round loop — so a helper that
#     dropped the decision on the way through was a clean run.
build_root "$root"
perl -0pi -e 's/    log_request_record\(totals, emitted, emitted\.len\(\), window\);/    log_request_record(totals, emitted, emitted.len(), window);\n    let _ = Record { charged: totals.charged };/' \
  "$root/crates/rmlx-models/src/speculative/mod.rs"
run "a helper reading the decision off the totals it was handed is refused" 1 \
  "\`emit_seed_token\` writes 1 \`charged:\` field(s)"

# 24. The same rule with a value the scan can read. A decision named outside a
#     round loop has no rollback beside it to be held to, whether or not the
#     token is legible.
build_root "$root"
cat >>"$root/crates/rmlx-models/src/speculative/mtp.rs" <<'RS'

fn summarise_uncharged(rounds: usize) -> super::RoundStats {
    super::RoundStats {
        rounds,
        charged: false,
    }
}
RS
run "a decision named outside a round loop is refused" 1 \
  "\`summarise_uncharged\` writes 1 \`charged:\` field(s) — false —"

# 25. RULE 5 against the fn it is about. The recorder carries the decision
#     across as a destructured `charged,`; spelled as a field it becomes a value
#     named where no rollback can be checked against it, and a rule that
#     exempted the recorder by name would let exactly this through.
build_root "$root"
perl -0pi -e 's/        emitted: emitted\.len\(\),\n        charged,/        emitted: emitted.len(),\n        charged: false,/' \
  "$root/crates/rmlx-models/src/speculative/mod.rs"
run "the recorder naming the decision rather than carrying it across is refused" 1 \
  "\`round_stats\` writes 1 \`charged:\` field(s) — false —"

# 26. Condition (b) is what keeps the shared seed emit out of the population,
#     and nothing else does: it carries a driver's signature. Give it totals of
#     its own and it joins, with no rollback in it for the gate to read against.
build_root "$root"
perl -0pi -e 's/    log_request_record\(totals, emitted, emitted\.len\(\), window\);/    log_request_record(\n        \&RoundTotals \{\n            loop_kind: SpecLoop::Kind,\n            rounds: 0,\n            charged: false,\n        \},\n        emitted,\n        emitted.len(),\n        window,\n    );/' \
  "$root/crates/rmlx-models/src/speculative/mod.rs"
run "a helper that builds its own totals joins the population and is refused" 2 \
  "\`emit_seed_token\` has 0 rollback and 1 record charge sites."

# 27. The defect the gate exists for, at the seam the record moved to: the
#     totals handed over say the round was charged and the rollbacks say it was
#     not. Every token and every count the loop reports is unchanged.
build_root "$root"
perl -0pi -e 's/            charged: charge_phases,\n        \},/            charged: true,\n        },/' \
  "$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
run "a record call handing the recorder the other decision is refused" 1 \
  "\`mtp_assistant_generate\` names more than one charge decision"

echo
if [ "$failures" != "0" ]; then
  echo "check-spec-charge-fixtures: $failures of $cases cases failed"
  exit 1
fi
echo "OK: $cases cases, every rule fired for its own reason."
