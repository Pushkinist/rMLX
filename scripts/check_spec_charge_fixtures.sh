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
        super::rollback_round(
            &mut v_caches,
            Some(&mut v_lin),
            &v_input,
            v_pre_round_offset,
            v_target,
            // This loop's one charge decision, first of two calls.
            $token,
            device,
        )?;
        super::rollback_round(
            &mut d_caches,
            Some(&mut d_lin),
            &d_fed,
            d_pre_round_offset,
            d_target,
            // The same decision, at the draft-side rollback.
            $token,
            device,
        )?;
        super::log_round(
            &super::RoundReport {
                loop_kind: SpecLoop::Kind,
                round: rounds,
                charged: $token,
            },
            &[],
        );
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

# The shared refold-or-disarm: it is not a round loop and it calls the low-level
# rollback, which RULE 4 would once have read as a rollback moved out of a loop.
# It names no decision — it forwards the `charge` its caller named — and the
# needle is the call the loops make, so it is not a site.
shared_rollback_src() {
  cat <<'RS'
pub(crate) fn rollback_round(
    kv: &mut [KvCache],
    lin: Option<&mut [LinearAttnCache]>,
    round_tokens: &[u32],
    pre_round_offset: i32,
    target_offset: i32,
    charge: bool,
    device: Device,
) -> Result<()> {
    if target_offset < pre_round_offset + round_tokens.len() as i32 {
        return rollback_round_caches(
            kv,
            lin,
            round_tokens,
            pre_round_offset,
            target_offset,
            charge,
            device,
        );
    }
    Ok(())
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

write_round_common() {
  local root="$1"
  {
    printf '//! The shared refold-or-disarm, and the low-level pair beneath it.\n\n'
    shared_rollback_src
    printf '\nfn rollback_round_caches(\n    caches: &mut [KvCache],\n    lin: Option<&mut [LinearAttnCache]>,\n    fed: &[u32],\n    pre_round_offset: i32,\n    target: i32,\n    charge: bool,\n    device: Device,\n) -> Result<()> {\n    refold_lin_tapes(lin, fed.len(), 0, charge, device)\n}\n'
    printf '\nfn refold_lin_tapes(\n    lin: Option<&mut [LinearAttnCache]>,\n    round_len: usize,\n    kept: usize,\n    charge: bool,\n    device: Device,\n) -> Result<()> {\n    Ok(())\n}\n'
  } >"$root/crates/rmlx-models/src/speculative/round_common.rs"
}

# The configuration an entry builds and the shared loop is handed. The scanner
# resolves the forwarded parameter by this declared type, so the fixture states
# it once — at module level, where no rule of this gate reaches.
cfg_type_src() {
  cat <<'RS'
pub(crate) struct RoundCfg {
    pub loop_kind: SpecLoop,
    pub charged: bool,
}
RS
}

# The shared loop: it is handed its decision rather than making one, and every
# site names the field it was handed, through one readable binding.
forwarded_loop_src() {
  cat <<'RS'
pub(crate) fn round_loop_generate(
    verifier: &Architecture,
    cfg: &RoundCfg,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    device: Device,
) -> Result<()> {
    let charge = cfg.charged;
    while emitted.len() < n_tokens {
        super::rollback_round(
            &mut v_caches,
            Some(&mut v_lin),
            &v_input,
            v_pre_round_offset,
            v_target,
            // The decision the entry made, first of two calls.
            charge,
            device,
        )?;
        super::rollback_round(
            &mut d_caches,
            Some(&mut d_lin),
            &d_fed,
            d_pre_round_offset,
            d_target,
            // The same decision, at the draft-side rollback.
            charge,
            device,
        )?;
        super::log_round(
            &super::RoundReport {
                loop_kind: cfg.loop_kind,
                round: rounds,
                charged: charge,
            },
            &[],
        );
    }
    log_request_record(
        &RoundTotals {
            loop_kind: cfg.loop_kind,
            rounds,
            charged: charge,
        },
        &emitted,
        seed_emitted,
        &window,
    );
    Ok(())
}
RS
}

# One entry: the fn that starts a round loop, builds its configuration, and
# names the decision once on the way in. It runs no round, so it holds no
# totals and rolls nothing back.
entry_src() {
  local name="$1" token="$2" bind="$3"
  printf 'pub fn %s(\n    verifier: &Architecture,\n    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,\n    device: Device,\n) -> Result<()> {\n' "$name"
  if [ "$bind" = "bind" ]; then
    printf '    let charge_phases = super::phases_charged();\n'
  fi
  cat <<RS
    super::round_loop_generate(
        verifier,
        &RoundCfg {
            loop_kind: SpecLoop::Kind,
            charged: $token,
        },
        step_fn,
        device,
    )
}
RS
}

# A drafter rolling its own state back: outside every round loop, carrying the
# decision it was handed as a field of one of its own parameters.
drafter_rollback_src() {
  local charge="$1"
  cat <<RS

fn rollback(
    &mut self,
    ctx: &RoundCtx<'_>,
    v: &Verdict,
) -> Result<Option<CacheSpan>> {
    super::rollback_round(
        &mut self.caches,
        None,
        &self.fed,
        0,
        v.target,
        $charge,
        ctx.device,
    )?;
    Ok(None)
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
  } >"$root/crates/rmlx-models/src/speculative/mod.rs"

  write_round_common "$root"

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

# The tree mid-campaign: one drafter migrated. Its loop body is gone, the shared
# loop runs its rounds, and the decision it used to make inside the body is made
# at the entry that starts the loop. Six loops still carry their own body, and
# the census is unchanged — which is the property the re-key exists to keep.
build_mid_root() {
  local root="$1"
  rm -rf "$root"
  mkdir -p "$root/crates/rmlx-models/src/speculative/dflash2"

  {
    printf '//! Two two-model loops, the entry guard, the shared helpers, and the\n//! configuration a migrated drafter builds.\n\n'
    cfg_type_src
    printf '\n'
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
  } >"$root/crates/rmlx-models/src/speculative/mod.rs"

  write_round_common "$root"

  {
    printf '//! The one round loop.\n\n'
    forwarded_loop_src
  } >"$root/crates/rmlx-models/src/speculative/round_loop.rs"

  entry_src "mtp_generate" "charge_phases" "bind" \
    >"$root/crates/rmlx-models/src/speculative/mtp.rs"
  loop_src "mtp_assistant_generate" "charge_phases" "bind" \
    >"$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
  loop_src "dflash2_generate" "charge_phases" "bind" \
    >"$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
  loop_src "dflash_generate" "false" "plain" \
    >"$root/crates/rmlx-models/src/speculative/dflash.rs"
  loop_src "eagle3_generate" "false" "plain" \
    >"$root/crates/rmlx-models/src/speculative/eagle3.rs"
}

# The end state: one loop body, seven entries, and the same seven decisions.
build_end_root() {
  local root="$1"
  rm -rf "$root"
  mkdir -p "$root/crates/rmlx-models/src/speculative/dflash2"

  {
    printf '//! The entry guard, the two two-model entries, the shared helpers, and\n//! the configuration every entry builds.\n\n'
    cfg_type_src
    printf '\n'
    dispatcher_src
    printf '\n'
    entry_src "spec_generate_greedy_cached" "false" "plain"
    printf '\n'
    entry_src "spec_generate_stochastic_cached" "false" "plain"
    printf '\n'
    stats_helper_src
    printf '\n'
    recorder_src
    printf '\n'
    seed_emit_src
  } >"$root/crates/rmlx-models/src/speculative/mod.rs"

  write_round_common "$root"

  {
    printf '//! The one round loop.\n\n'
    forwarded_loop_src
  } >"$root/crates/rmlx-models/src/speculative/round_loop.rs"

  entry_src "mtp_generate" "charge_phases" "bind" \
    >"$root/crates/rmlx-models/src/speculative/mtp.rs"
  entry_src "mtp_assistant_generate" "charge_phases" "bind" \
    >"$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
  entry_src "dflash2_generate" "charge_phases" "bind" \
    >"$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
  entry_src "dflash_generate" "false" "plain" \
    >"$root/crates/rmlx-models/src/speculative/dflash.rs"
  entry_src "eagle3_generate" "false" "plain" \
    >"$root/crates/rmlx-models/src/speculative/eagle3.rs"
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
  "OK: 7 classic, 0 forwarded, 0 entries; census charge_phases:3 false:4 (7 sites)."

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
perl -0pi -e 's/            false,\n            device,/            charge_phases,\n            device,/' \
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

# 4. A value the token reader cannot read is unreadable, not absent. A field
#    access is a token; a call that resembles one is not, which is what the
#    delimiter anchor is for.
build_root "$root"
perl -0pi -e 's/            charged: false,/            charged: self.charged(),/' \
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
perl -0pi -e 's/        super::rollback_round\(\n(?:.*\n)*?        \)\?;/        roll_it(\&mut v_caches, false, device)?;/g' \
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
perl -0pi -e 's/        super::rollback_round\(\n(?:.*\n)*?        \)\?;/        super::rollback_round(\&mut v_caches, None, \&v_input, 0, 0, false, device)?;/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a call written on one line is a scan error" 2 \
  "\`mtp_generate\` has a charge site this gate could not read"

# 11. A call with an argument dropped is read back at the wrong position, so the
#     scan refuses it rather than reporting whatever landed there.
build_root "$root"
perl -0pi -e 's/            &v_input,\n//' \
  "$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
run "a call with an argument dropped is a scan error" 2 \
  "\`dflash2_generate\` has a charge site this gate could not read"

# 12. Both spellings renamed: the loops are still derived and now carry no
#     decision at all. A scan that found nothing must not pass.
build_root "$root"
perl -0pi -e 's/rollback_round\(/rewind_round(/g; s/charged:/billed:/g' \
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
perl -0pi -e 's/            \/\/ The same decision, at the draft-side rollback\.\n            charge_phases,/            \/\/ The same decision, at the draft-side rollback.\n            false,/' \
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
  "OK: 7 classic, 0 forwarded, 0 entries; census charge_phases:3 false:4 (7 sites)."

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

# 27. RULE 6: a loop that keeps its readable `rollback_round` call and reaches
#     past it to the low-level rollback underneath. Every token this gate reads
#     still agrees; the second decision is at a call it does not open. This is
#     the shape the needle move to `rollback_round(` would otherwise have let
#     through, and in the tree the same shape does not compile — the low-level
#     rollback is private to `round_common`.
build_root "$root"
perl -0pi -e 's/        super::log_round\(/        super::rollback_round_caches(\n            \&mut v_caches,\n            None,\n            \&v_input,\n            0,\n            0,\n            false,\n            device,\n        )?;\n        super::log_round(/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a loop reaching past the shared rollback to the low-level one is a scan error" 2 \
  "\`mtp_generate\` makes 1 call(s) to the low-level"

# 28. RULE 6 by file: a helper one hop from a loop, which a
#     fn-shaped rule read as a non-driver and let through. The loop's own
#     `rollback_round` call still reads clean and still carries `false`.
build_root "$root"
cat >>"$root/crates/rmlx-models/src/speculative/dflash.rs" <<'RS'

fn refold_it(lin: Option<&mut [LinearAttnCache]>, device: Device) -> Result<()> {
    refold_lin_tapes(lin, 3, 2, true, device)
}
RS
run "a helper one hop from a loop naming the low-level rollback is a scan error" 2 \
  "\`refold_it\` makes 1 call(s) to the low-level"

# 29. RULE 6 by fn: a round loop that lives in `round_common.rs`, which is where
#     the shared skeleton is going. The file's exemption is for the code beneath
#     `rollback_round`, not for a loop that happens to sit beside it, and its own
#     `rollback_round` call still reads clean and still carries `false`.
build_root "$root"
{
  printf '\n'
  loop_src "round_common_generate" "false" "plain"
} >>"$root/crates/rmlx-models/src/speculative/round_common.rs"
perl -0pi -e 's/(fn round_common_generate\((?:.*\n)*?)    while emitted/$1    refold_lin_tapes(None, 3, 2, true, device)?;\n    while emitted/' \
  "$root/crates/rmlx-models/src/speculative/round_common.rs"
run "a round loop in round_common.rs is read for the low-level rollback too" 2 \
  "\`round_common_generate\` makes 1 call(s) to the low-level"

# 30. The defect the gate exists for, at the seam the record moved to: the
#     totals handed over say the round was charged and the rollbacks say it was
#     not. Every token and every count the loop reports is unchanged.
build_root "$root"
perl -0pi -e 's/            charged: charge_phases,\n        \},/            charged: true,\n        },/' \
  "$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
run "a record call handing the recorder the other decision is refused" 1 \
  "\`mtp_assistant_generate\` names more than one charge decision"

# 31. RULE 7 by call: a loop that keeps every charge decision this gate reads
#     and writes its own round event instead of calling the one emitter. On its
#     own module target, which is the shape the collapse replaced.
build_root "$root"
perl -0pi -e 's/        super::log_round\(\n(?:.*\n)*?        \);\n/        tracing::debug!(\n            target: "rmlx_models::speculative::mtp",\n            loop_kind = ?SpecLoop::Kind,\n            round = rounds,\n            charged = charge_phases,\n            "speculative round"\n        );\n/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a loop that writes its own round event instead of calling the emit" 2 \
  "\`mtp_generate\` reaches the one round emit 0 time(s)."

# 32. RULE 7 by name, and the mutation the rule was written for: the same
#     bypass on the shared target. Every field in the same order, so the line it
#     writes is byte-identical to the one the emitter would have written and no
#     digest, pair or census can see it. Only the name can.
build_root "$root"
perl -0pi -e 's/        super::log_round\(\n(?:.*\n)*?        \);\n/        tracing::debug!(\n            target: super::PHASE_TARGET,\n            loop_kind = ?SpecLoop::Kind,\n            round = rounds,\n            charged = charge_phases,\n            "speculative round"\n        );\n/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a loop writing the shared target itself is refused by name" 2 \
  "src/speculative/mtp.rs names the per-round event's target."

# 33. RULE 7 by name, away from any loop: a helper that merely holds the string.
#     Every loop still reaches the emitter exactly once, so the call reading is
#     clean and the name reading is the only one that fires.
build_root "$root"
cat >>"$root/crates/rmlx-models/src/speculative/dflash.rs" <<'RS'

fn phase_target() -> &'static str {
    "rmlx::spec::phase"
}
RS
run "a helper naming the phase target outside its file is refused" 2 \
  "src/speculative/dflash.rs names the per-round event's target."

# 34. RULE 7 by call, the other way: one loop closing its round twice. Two
#     record shapes under one target is what the collapse removed, and a second
#     line per round doubles that pair's count in the pinned baseline.
build_root "$root"
perl -0pi -e 's/(        super::log_round\(\n(?:.*\n)*?        \);\n)/$1$1/' \
  "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "a loop closing its round twice is refused" 2 \
  "\`eagle3_generate\` reaches the one round emit 2 time(s)."

# 35. RULE 7 by name, by fn: a round loop that *moved* into the file that owns
#     the target — not an eighth loop, so the census is untouched and every
#     other reading is clean. It calls the emitter once, and the file reading
#     exempts the path, so the loop writes a second event on the target beside
#     the one it emitted and only the fn reading sees it. This is the direction
#     the shared skeleton is going, which is why the exemption cannot be by file
#     alone.
build_root "$root"
{
  printf '//! The shared record and the one round emit.\n\n'
  loop_src "eagle3_generate" "false" "plain"
} >"$root/crates/rmlx-models/src/speculative/round_stats.rs"
printf '//! The loop that used to live here moved into the shared record.\n' \
  >"$root/crates/rmlx-models/src/speculative/eagle3.rs"
perl -0pi -e 's/(        super::log_round\(\n(?:.*\n)*?        \);\n)/$1        tracing::debug!(target: PHASE_TARGET, round = rounds, "speculative round");\n/' \
  "$root/crates/rmlx-models/src/speculative/round_stats.rs"
run "a round loop inside the target's own file does not inherit its exemption" 2 \
  "\`eagle3_generate\` is a round loop and names the"

# 36. The needle is code, not prose. A trailing comment naming the emit on a
#     line that is not one would otherwise count as a call, and the loop that
#     dropped its emit would read as one that still makes it.
build_root "$root"
perl -0pi -e 's/        super::log_round\(\n(?:.*\n)*?        \);\n/        rounds += 1; \/\/ the round the shared log_round( would have closed\n/' \
  "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "a trailing comment naming the emit is not a call to it" 2 \
  "\`eagle3_generate\` reaches the one round emit 0 time(s)."

# 37. Every needle reads code. A doc comment in a non-loop file naming the
#     target is prose, not a second event on it — and the chunk that puts a
#     round emit behind a skeleton wrapper has to write exactly this sentence in
#     `round_common.rs`.
build_root "$root"
cat >>"$root/crates/rmlx-models/src/speculative/round_common.rs" <<'RS'

/// The per-round event goes out on `PHASE_TARGET`, and every loop reaches it
/// through the one `log_round(` call this gate counts.
fn seam_note() {}
RS
run "a doc comment naming the target and the emit is prose, not either" 0 \
  "OK: 7 classic, 0 forwarded, 0 entries; census charge_phases:3 false:4 (7 sites)."

# ---------------------------------------------------------------------------
# The collapse: the decision travels as a value.
#
# The cases above are all one tree shape — seven loops, each deciding for
# itself. Below are the two shapes the campaign passes through: mid-campaign,
# where one drafter has been migrated and its decision now lives at the entry
# that starts the shared loop, and the end state, where all seven do. The census
# is the same seven decisions in every one of them, which is the property that
# makes it worth reading at all.
# ---------------------------------------------------------------------------

# R1. The mid-campaign tree. The forwarded loop is named on the success line, so
#     a fixture asserting that a loop is forwarded has a line to assert against.
build_mid_root "$root"
run "the mid-campaign tree passes, and names what it passed as" 0 \
  "OK: 6 classic, 1 forwarded, 1 entries; census charge_phases:3 false:4 (7 sites). Forwarded: \`round_loop_generate\`."

# R2. The mutation the re-key exists for: the forwarded loop hard-wires the
#     decision it was handed. Both readings fire on the one tree — RULE 8 names
#     the line, the census names the count.
build_mid_root "$root"
perl -0pi -e 's/    let charge = cfg\.charged;/    let charge = false;/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "a forwarded loop that hard-wires its charge is named by RULE 8" 1 \
  "\`round_loop_generate\` is handed its charge in \`cfg\` and applies"
run "the same hard-wire puts an eighth decision in the census" 1 \
  "the charge census is \"charge:1 charge_phases:3 false:4\" (8 sites)"

# R3. The whole right-hand side, not a prefix of it: a second condition ORed
#     into the field charges on requests the entry did not ask for.
build_mid_root "$root"
perl -0pi -e 's/    let charge = cfg\.charged;/    let charge = cfg.charged || force_charge;/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "a forwarded loop ORing a condition into the field it was handed is refused" 1 \
  "\`round_loop_generate\` is handed its charge in \`cfg\` and applies"

# R4. The other legitimate spelling: no binding at all, the field named at every
#     site. A field access is a token, which is the reader change this needs.
build_mid_root "$root"
perl -0pi -e 's/    let charge = cfg\.charged;\n//; s/^(\s+)charge,$/$1cfg.charged,/gm; s/charged: charge,/charged: cfg.charged,/g' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "a forwarded loop naming the field at every site is read as one token" 0 \
  "OK: 6 classic, 1 forwarded, 1 entries; census charge_phases:3 false:4 (7 sites)."

# R5. An entry that drops its decision. Excluding the forwarded loop from the
#     census must not hide it: RULE 5's zero-arm names the entry, and the census
#     reads six sites where the tree records seven.
build_end_root "$root"
perl -0pi -e 's/            charged: charge_phases,\n//' \
  "$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
run "an entry that states no decision is named by RULE 5" 1 \
  "\`mtp_assistant_generate\` builds a round loop's configuration and states"
run "the same dropped decision is a census that lost a site" 1 \
  "the charge census is \"charge_phases:2 false:4\" (6 sites)"

# R6. RULE 2 over the entries: the token still has to mean the call.
build_end_root "$root"
perl -0pi -e 's/    let charge_phases = super::phases_charged\(\);/    let charge_phases = true;/' \
  "$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
run "an entry binding \`charge_phases\` to a literal is refused" 1 \
  "\`dflash2_generate\` charges on \`charge_phases\` and does not bind it"

# R7. An entry genuinely moved from one schedule to the other, binding and all.
build_end_root "$root"
perl -0pi -e 's/            charged: false,/            charged: charge_phases,/' \
  "$root/crates/rmlx-models/src/speculative/eagle3.rs"
perl -0pi -e 's/(\) -> Result<\(\)> \{\n)/$1    let charge_phases = super::phases_charged();\n/' \
  "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "an entry moved between the two schedules is refused" 1 \
  "the charge census is \"charge_phases:4 false:3\" (7 sites)"

# R8. RULE 5 unchanged: a fn in neither population naming a decision.
build_mid_root "$root"
cat >>"$root/crates/rmlx-models/src/speculative/mtp.rs" <<'RS'

fn summarise_uncharged(rounds: usize) -> super::RoundStats {
    super::RoundStats {
        rounds,
        charged: false,
    }
}
RS
run "a decision named outside both populations is refused" 1 \
  "\`summarise_uncharged\` writes 1 \`charged:\` field(s) — false —"

# R9. RULE 4's new arm: a drafter rolling its own state back carries the
#     decision it was handed, as a field of one of its own parameters.
build_mid_root "$root"
drafter_rollback_src "ctx.charged" \
  >>"$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a drafter rollback carrying the decision it was handed passes" 0 \
  "OK: 6 classic, 1 forwarded, 1 entries; census charge_phases:3 false:4 (7 sites)."

# R10. The same rollback deciding for itself. A literal there is a second
#      decision made where nothing can hold it to the loop that ordered it.
build_mid_root "$root"
drafter_rollback_src "false" \
  >>"$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a drafter rollback that decides for itself is refused" 1 \
  "\`rollback\` rolls a round back on \`false\`, which is not a"

# R11. An argument the scan cannot read back is unreadable, not absent — the
#      same disposition the record side has.
build_mid_root "$root"
drafter_rollback_src "ctx.charge_flag()" \
  >>"$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a drafter rollback on an unreadable charge is a scan error" 2 \
  "\`rollback\` rolls a round back on a \`charge\` this gate"

# R12. The end state: one loop body, seven entries, the same seven decisions.
build_end_root "$root"
run "the end state passes with every decision in an entry" 0 \
  "OK: 0 classic, 1 forwarded, 7 entries; census charge_phases:3 false:4 (7 sites). Forwarded: \`round_loop_generate\`."

# R13. The end state with the one loop deleted. A scan that finds nothing must
#      not pass, and at the end state there is one body left to find.
build_end_root "$root"
rm -f "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "the end state with its one loop deleted is a scan error" 2 \
  "found no round loop"

# R14. RULE 6 over the shared loop: the low-level rollback is round_common's
#      alone, and the loop beside it does not inherit that exemption.
build_mid_root "$root"
perl -0pi -e 's/        super::log_round\(/        super::rollback_round_caches(\n            \&mut v_caches,\n            None,\n            \&v_input,\n            0,\n            0,\n            false,\n            device,\n        )?;\n        super::log_round(/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "the shared loop reaching past the shared rollback is a scan error" 2 \
  "\`round_loop_generate\` makes 1 call(s) to the low-level"

# R15. RULE 7 over the shared loop: one round, one event.
build_mid_root "$root"
perl -0pi -e 's/(        super::log_round\(\n(?:.*\n)*?        \);\n)/$1$1/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "the shared loop closing its round twice is refused" 2 \
  "\`round_loop_generate\` reaches the one round emit 2 time(s)."

# R16. The shadow, on both arms. On the forwarded loop RULE 8 has something
#      exact to say about the second binding, so it is exit 1 and names it; on a
#      classic loop the same-token reading has become vacuous and it is exit 2.
build_mid_root "$root"
perl -0pi -e 's/(    let charge = cfg\.charged;\n)/$1    let charge = false;\n/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "a forwarded loop shadowing its binding is named by RULE 8" 1 \
  "\`round_loop_generate\` binds its charge token twice; the last binds it"

build_mid_root "$root"
perl -0pi -e 's/(    let charge_phases = super::phases_charged\(\);\n)/$1    let charge_phases = false;\n/' \
  "$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
run "a classic loop shadowing its binding is a scan error" 2 \
  "\`mtp_assistant_generate\` binds its charge token more than once"

# R17. RULE 5 over the entries: an entry names one decision, and two are two
#      decisions with nothing saying which one the loop applies.
build_end_root "$root"
perl -0pi -e 's/(            charged: false,\n)/$1            charged: true,\n/' \
  "$root/crates/rmlx-models/src/speculative/dflash.rs"
run "an entry stating two decisions is refused" 1 \
  "\`dflash_generate\` builds a round loop's configuration and states"

# R18. The entry arm of R16, and not a corner: at the end state all seven
#      decisions live in entries, so an unread shadow there is the whole census.
build_end_root "$root"
perl -0pi -e 's/(    let charge_phases = super::phases_charged\(\);\n)/$1    let charge_phases = false;\n/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "an entry shadowing its binding is a scan error" 2 \
  "\`mtp_generate\` binds its charge token more than once"

# R19. An entry that stops building the configuration leaves the population.
#      That must read as a lost decision rather than as a clean scan, which is
#      what the census does and no membership test could.
build_end_root "$root"
perl -0pi -e 's/    super::round_loop_generate\(\n(?:.*\n)*?    \)\n/    super::round_loop_legacy(verifier, charge_phases, step_fn, device)\n/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "an entry that stops building the configuration is a lost decision" 1 \
  "the charge census is \"charge_phases:2 false:4\" (6 sites)"

# R20. The configuration handed over is read-only inside the loop. A loop that
#      builds its own and forwards faithfully from that writes no site any
#      reading here can see.
build_mid_root "$root"
perl -0pi -e 's/    let charge = cfg\.charged;/    let cfg = RoundCfg::uncharged();\n    let charge = cfg.charged;/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "a forwarded loop rebinding the configuration's name is a scan error" 2 \
  "\`round_loop_generate\` rebinds or writes \`cfg\`, the configuration it"

# R21. The same, by assignment rather than by rebinding.
build_mid_root "$root"
perl -0pi -e 's/    cfg: &RoundCfg,/    cfg: \&mut RoundCfg,/; s/    let charge = cfg\.charged;/    cfg.charged = false;\n    let charge = cfg.charged;/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "a forwarded loop writing the configuration's charge field is a scan error" 2 \
  "\`round_loop_generate\` rebinds or writes \`cfg\`, the configuration it"

# R22. The delimiter anchor holds: a call is not the field it resembles, and a
#      reader without the anchor would count this as forwarded.
build_mid_root "$root"
perl -0pi -e 's/                charged: charge,/                charged: cfg.charged(),/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "a call that resembles the configuration field is a scan error" 2 \
  "\`round_loop_generate\` has a charge site this gate could not read"

# R23. Not a readable binding form. A reader that guesses is worse than one
#      that refuses, and this is the shape rustfmt cannot introduce.
build_mid_root "$root"
perl -0pi -e 's/    let charge = cfg\.charged;/    let mut charge = cfg.charged;/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "a forwarded loop binding through a \`let mut\` is a scan error" 2 \
  "\`round_loop_generate\` binds \`charge\` in a shape this gate cannot"

# R24. The other half of the per-binding reading: a token the sites name and no
#      binding anywhere. The scan never saw what it means, so the census would
#      be reading a name rather than a decision.
build_root "$root"
perl -0pi -e 's/    let charge_phases = super::phases_charged\(\);\n//' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a loop naming a token it never binds is refused" 1 \
  "\`mtp_generate\` names \`charge_phases\` as its decision and never binds"

echo
if [ "$failures" != "0" ]; then
  echo "check-spec-charge-fixtures: $failures of $cases cases failed"
  exit 1
fi
echo "OK: $cases cases, every rule fired for its own reason."
