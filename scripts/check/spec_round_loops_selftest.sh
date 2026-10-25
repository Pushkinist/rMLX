#!/usr/bin/env bash
# Recall test for the speculative round-loop gate: one edit per synthetic source root, rule and exit.
#
# Recall test for: check/spec_round_loops.sh charge sampling list-drivers
#
# A gate is only worth its runtime if it fires, and "the tree is clean" is no evidence of that:
# a scan whose regex stopped matching reports the same clean tree as a scan that works. Each
# case is one edit to a synthetic source root, and each asserts the exit code AND the reason: a
# case that fails for the wrong reason has told us nothing about the rule it was meant to
# exercise. Every planted edit must change a file; an edit that matches nothing ends the run
# with exit 2, because a case that ran on an unedited root proves nothing.
#
# The roots are built here rather than copied from the tree, so a case that passes against one
# is passing against a shape this file states rather than against whatever the repository
# holds today. Each carries the seven drafter paths of the tree, the charge census (three
# loops that charge their phases, four that do not) and the shapes that must not be counted as
# loops: the entry guard, a summary helper that builds a `RoundTotals` and drives nothing, the
# shared seed emit, which has a driver signature and is handed a loop's totals, and a loop-shaped
# fn in a sibling test file. Four shapes are built: the all-loops shape the campaign started
# from, the mid-campaign one where one drafter is an entry that hands its decision and its
# sampler to the shared loop, the one where the two-model greedy path is an entry beside a
# loop, and the end state the tree has: one shared loop and seven entries.
#
# The `charge` group also holds the cases of `list-drivers`, which reads the same population,
# and the argument cases of the tool.
#
# LOC-exempt: one planted tree per case; the builders are shared by both rule sets
# Usage:  bash scripts/check/spec_round_loops_selftest.sh [<charge|sampling>]
#         no group runs the cases of both
# Exit:   0 every case held; 1 a case did not, or none ran; 2 an unknown group, or a root or
#         an edit that could not be built

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="$REPO_ROOT/scripts/check/spec_round_loops.sh"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"

GROUP="${1:-both}"
case "$GROUP" in
charge | sampling) st_init "check-spec-$GROUP-fixtures" ;;
both) st_init spec-round-loops-selftest ;;
*)
    echo "usage: bash scripts/check/spec_round_loops_selftest.sh [<charge|sampling>]" >&2
    exit 2
    ;;
esac

ROOT="$ST_WORK/root"
SPEC=crates/rmlx-models/src/speculative

# edit <perl expression> <file>...: plant one edit; it must change at least one file.
edit() {
    local expr="$1" before after
    shift
    before=$(cat "$@" | cksum)
    perl -0pi -e "$expr" "$@" || st_die "perl failed on: $*"
    after=$(cat "$@" | cksum)
    [ "$before" != "$after" ] || st_die "the planted edit changed nothing: $expr"
}

# gate <mode> <name> <exit> [<reason>]: run the tool in <mode> over $ROOT.
gate() {
    local mode="$1" name="$2" want="$3" reason="${4:-}"
    if [ -n "$reason" ]; then
        st_case "$mode: $name" "$want" --has "$reason" -- bash "$TOOL" "$mode" --root "$ROOT"
    else
        st_case "$mode: $name" "$want" -- bash "$TOOL" "$mode" --root "$ROOT"
    fi
}

# ---- sources -----------------------------------------------------------------

# loop_src <name> <token> <bind|plain>: one round loop. A driver signature that takes the
# request's sampler, the draw built from it, a rollback whose charge argument and a record
# whose `charged:` field name the same decision, and the `RoundTotals` that makes it a loop.
loop_src() {
    local name="$1" token="$2" bind="$3"
    printf 'pub fn %s(\n    verifier: &Architecture,\n    n_tokens: usize,\n    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,\n    sampler_cfg: &crate::sampler::SamplerConfig,\n    device: Device,\n) -> Result<Vec<ProbeStep>> {\n' "$name"
    if [ "$bind" = "bind" ]; then
        printf '    let charge_phases = super::phases_charged();\n'
    fi
    cat <<RS
    let mut draw = super::VerifierDraw::new(sampler_cfg);
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
    Ok(vec![])
}
RS
}

# A fn that builds a `RoundTotals` and drives no generation: not a round loop.
stats_helper_src() {
    cat <<'RS'
pub fn summarise_rounds(rounds: usize, totals: &RoundTotals) -> RoundTotals {
    RoundTotals { rounds, ..*totals }
}
RS
}

# The shared recorder: the one fn that writes the record, from a destructured binding rather
# than a named value.
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

# The shared seed emit: a driver's signature, and it names no decision of its own; it is
# handed the loop's totals.
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

# The two-model entry guard: it routes a request to one of two paths by reading whether the
# sampler is active, and passes it to the one that takes it.
guard_src() {
    cat <<'RS'
pub fn spec_generate_greedy(
    verifier: &Architecture,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    if sampler_cfg.sampling_active() {
        spec_generate_stochastic_cached(verifier, step_fn, sampler_cfg, device)
    } else {
        spec_generate_greedy_cached(verifier, step_fn, sampler_cfg, device)
    }
}
RS
}

# The shared refold-or-disarm, and the low-level pair beneath it.
write_round_common() {
    {
        printf '//! The shared refold-or-disarm, and the low-level pair beneath it.\n\n'
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
        printf '\nfn rollback_round_caches(\n    caches: &mut [KvCache],\n    lin: Option<&mut [LinearAttnCache]>,\n    fed: &[u32],\n    pre_round_offset: i32,\n    target: i32,\n    charge: bool,\n    device: Device,\n) -> Result<()> {\n    refold_lin_tapes(lin, fed.len(), 0, charge, device)\n}\n'
        printf '\nfn refold_lin_tapes(\n    lin: Option<&mut [LinearAttnCache]>,\n    round_len: usize,\n    kept: usize,\n    charge: bool,\n    device: Device,\n) -> Result<()> {\n    Ok(())\n}\n'
    } >"$ROOT/$SPEC/round_common.rs"
}

# The configuration an entry builds and the shared loop is handed.
write_cfg_type() {
    cat >"$ROOT/$SPEC/round_cfg.rs" <<'RS'
pub(crate) struct RoundCfg {
    pub loop_kind: SpecLoop,
    pub charged: bool,
}
RS
}

# The shared loop: handed its decision and its sampler rather than making either, and every
# site names the field it was handed, through one readable binding.
forwarded_loop_src() {
    cat <<'RS'
pub(crate) fn round_loop_generate(
    verifier: &Architecture,
    cfg: &RoundCfg,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    let charge = cfg.charged;
    let mut draw = super::VerifierDraw::new(cfg.sampler_cfg);
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
    Ok(vec![])
}
RS
}

# entry_src <name> <token> <bind|plain>: the fn that starts the shared loop. It takes the
# request's sampler, builds the configuration carrying it, names the decision once, and runs
# no round, so it holds no totals and rolls nothing back.
entry_src() {
    local name="$1" token="$2" bind="$3"
    printf 'pub fn %s(\n    verifier: &Architecture,\n    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,\n    sampler_cfg: &crate::sampler::SamplerConfig,\n    device: Device,\n) -> Result<Vec<ProbeStep>> {\n' "$name"
    if [ "$bind" = "bind" ]; then
        printf '    let charge_phases = super::phases_charged();\n'
    fi
    cat <<RS
    super::round_loop_generate(
        verifier,
        &RoundCfg {
            loop_kind: SpecLoop::Kind,
            sampler_cfg,
            charged: $token,
        },
        step_fn,
        device,
    )
}
RS
}

# A drafter rolling its own state back: outside every round loop, carrying the decision it
# was handed as a field of one of its own parameters.
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

# A trait declaring what each drafter implements: bodiless declarations, one default body among
# them and a `where` clause on one. A `fn` item with no body is where a scanner that waits for
# the next `{` reads the following fn as this one's body and records neither.
drafter_trait_src() {
    cat <<'RS'
pub(crate) trait RoundDrafter {
    fn prefill(&mut self, ctx: &mut RoundCtx<'_>, prompt: &[u32]) -> Result<Prefilled>;

    fn block(&self, block_total: usize, remaining: usize) -> usize {
        round_block(block_total, remaining)
    }

    fn verify(&mut self, ctx: &mut RoundCtx<'_>, fed: &[u32]) -> Result<Verdict>;

    fn carry<T>(&self, v: &Verdict) -> T
    where
        T: From<u32>;

    fn condition(&mut self, ctx: &RoundCtx<'_>, v: &Verdict) -> Result<Option<Conditioning>>;
}

RS
}

# The server's dispatch: one arm per drafter kind, each passing the resolved sampler.
write_dispatch() {
    mkdir -p "$ROOT/crates/rmlx-server/src/engine"
    cat >"$ROOT/crates/rmlx-server/src/engine/speculative.rs" <<'RS'
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
            n_tokens,
            &mut step_fn,
            &spec_sampler_cfg,
            dispatcher.device(),
        ),
    };
}
RS
}

# Sibling test files, which the scan must not read: a `charged: true` fixture and a harness
# without a sampler are legitimate things for a test to build and are not production paths.
write_test_siblings() {
    cat >"$ROOT/$SPEC/mtp_tests.rs" <<'RS'
pub fn mtp_generate_harness(
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
) -> Result<Vec<ProbeStep>> {
    log_request_record(&RoundTotals { rounds, charged: false }, &emitted);
    Ok(vec![])
}
RS
    loop_src "a_charged_round_fixture" "true" "plain" >"$ROOT/$SPEC/round_stats_tests.rs"
}

# write_mod <greedy: loop|entry> <stochastic: loop|entry>: the module file with the guard, the
# two two-model paths in the given shape, and the helpers that are no loops.
write_mod() {
    {
        printf '//! The entry guard, the two two-model paths, and the shared helpers.\n\n'
        guard_src
        printf '\n'
        if [ "$1" = loop ]; then
            loop_src "spec_generate_greedy_cached" "false" "plain"
        else
            entry_src "spec_generate_greedy_cached" "false" "plain"
        fi
        printf '\n'
        if [ "$2" = loop ]; then
            loop_src "spec_generate_stochastic_cached" "false" "plain"
        else
            entry_src "spec_generate_stochastic_cached" "false" "plain"
        fi
        printf '\nfn summarise(rounds: usize) -> usize {\n    rounds\n}\n\n'
        stats_helper_src
        printf '\n'
        recorder_src
        printf '\n'
        seed_emit_src
    } >"$ROOT/$SPEC/mod.rs"
}

# The all-loops shape the campaign started from: seven paths, every one a body.
build_root() {
    rm -rf "$ROOT"
    mkdir -p "$ROOT/$SPEC/dflash2"
    write_mod loop loop
    write_round_common
    write_cfg_type
    loop_src "mtp_generate" "charge_phases" "bind" >"$ROOT/$SPEC/mtp.rs"
    loop_src "mtp_assistant_generate" "charge_phases" "bind" >"$ROOT/$SPEC/gemma4_assistant.rs"
    loop_src "dflash2_generate" "charge_phases" "bind" >"$ROOT/$SPEC/dflash2/round.rs"
    loop_src "dflash_generate" "false" "plain" >"$ROOT/$SPEC/dflash.rs"
    loop_src "eagle3_generate" "false" "plain" >"$ROOT/$SPEC/eagle3.rs"
    write_test_siblings
    write_dispatch
}

# Mid-campaign: one drafter migrated. Its loop body is gone, the shared loop runs its rounds,
# and the decision it made inside the body is made at the entry that starts the loop. Six
# loops still carry their own body, and the census is unchanged.
build_mid_root() {
    build_root
    entry_src "mtp_generate" "charge_phases" "bind" >"$ROOT/$SPEC/mtp.rs"
    {
        printf '//! The one round loop.\n\n'
        forwarded_loop_src
    } >"$ROOT/$SPEC/round_loop.rs"
}

# Two entries: the two-model greedy path beside a migrated sidecar, with the other four
# drafters still loop bodies. The smallest root in which the greedy path wears the shape it
# wears in the tree.
build_entry_root() {
    build_mid_root
    write_mod entry loop
}

# The end state, the tree's own census: one shared loop and seven entries, the stochastic
# acceptance rule among them.
build_end_root() {
    build_mid_root
    write_mod entry entry
    entry_src "mtp_assistant_generate" "charge_phases" "bind" >"$ROOT/$SPEC/gemma4_assistant.rs"
    entry_src "dflash2_generate" "charge_phases" "bind" >"$ROOT/$SPEC/dflash2/round.rs"
    entry_src "dflash_generate" "false" "plain" >"$ROOT/$SPEC/dflash.rs"
    entry_src "eagle3_generate" "false" "plain" >"$ROOT/$SPEC/eagle3.rs"
}

# ---- charge ------------------------------------------------------------------

cases_charge() {
    local loop_file

    build_root
    gate charge "a clean root passes, and the entry guard is not a loop" 0 \
        "OK: 7 classic, 0 forwarded, 0 entries; census charge_phases:3 false:4 (7 sites)."

    # The rule the gate exists for: one loop whose rollback and record disagree. The record
    # says the round was not charged, the rollback charges it, and every token and every count
    # the loop reports is unchanged.
    build_root
    edit 's/            charged: charge_phases,/            charged: false,/' "$ROOT/$SPEC/mtp.rs"
    gate charge "a loop whose rollback and record disagree is refused" 1 \
        "\`mtp_generate\` names more than one charge decision"

    # The same defect the other way round, on a loop that charges nothing.
    build_root
    edit 's/            false,\n            device,/            charge_phases,\n            device,/' "$ROOT/$SPEC/eagle3.rs"
    gate charge "a rollback that charges where the record does not is refused" 1 \
        "\`eagle3_generate\` names more than one charge decision"

    # The last field of a record carries no comma. A matcher that required one read this loop
    # as having no record site at all and passed it.
    build_root
    edit 's/            charged: false,\n        \},/            charged: true\n        },/' "$ROOT/$SPEC/eagle3.rs"
    gate charge "a record whose last field drops its comma is still read" 1 \
        "\`eagle3_generate\` names more than one charge decision"

    # A value the token reader cannot read is unreadable, not absent. A field access is a
    # token; a call that resembles one is not, which is what the delimiter anchor is for.
    build_root
    edit 's/            charged: false,/            charged: self.charged(),/' "$ROOT/$SPEC/dflash.rs"
    gate charge "a record whose value the scan cannot read is a scan error" 2 \
        "\`dflash_generate\` has a charge site this gate could not read"

    # A value wired in rather than decided.
    build_root
    edit 's/\bfalse\b/true/g' "$ROOT/$SPEC/dflash.rs"
    gate charge "a hard-wired \`true\` is refused" 1 \
        "the charge census is \"charge_phases:3 false:3 true:1\""

    # The census reads a spelling unless the binding is checked: a loop that calls its local
    # `charge_phases` and binds it to a literal charges every request and would be counted
    # among the three that ask.
    build_root
    edit 's/\bfalse\b/charge_phases/g' "$ROOT/$SPEC/eagle3.rs"
    edit 's/(\) -> Result<Vec<ProbeStep>> \{\n)/$1    let charge_phases = true;\n/' "$ROOT/$SPEC/eagle3.rs"
    gate charge "a loop that binds \`charge_phases\` to a literal is refused" 1 \
        "\`eagle3_generate\` charges on \`charge_phases\` and does not bind it"

    # A loop genuinely moved from one schedule to the other, binding and all.
    build_root
    edit 's/\bfalse\b/charge_phases/g' "$ROOT/$SPEC/eagle3.rs"
    edit 's/(\) -> Result<Vec<ProbeStep>> \{\n)/$1    let charge_phases = super::phases_charged();\n/' "$ROOT/$SPEC/eagle3.rs"
    gate charge "a loop moved between the two schedules is refused" 1 \
        "the charge census is \"charge_phases:4 false:3\""

    # A loop the scan lost. The population is derived, so this moves the census rather than a
    # number typed into the gate.
    build_root
    rm -f "$ROOT/$SPEC/eagle3.rs"
    gate charge "a loop deleted moves the census" 1 \
        "the charge census is \"charge_phases:3 false:3\""

    # A driver whose rollback moved into a helper. The loop still runs rounds and still
    # records a decision; the gate can no longer follow the other half of it.
    build_root
    edit 's/        super::rollback_round\(\n(?:.*\n)*?        \)\?;/        roll_it(\&mut v_caches, false, device)?;/g' "$ROOT/$SPEC/dflash.rs"
    cat >>"$ROOT/$SPEC/dflash.rs" <<'RS'

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
    gate charge "a driver whose rollback is delegated to a helper is a scan error" 2 \
        "\`dflash_generate\` has 0 rollback and 2 record charge sites."

    # A call the scan cannot read the arguments of. One line is a legal Rust shape and an
    # unread call is not a checked one.
    build_root
    edit 's/        super::rollback_round\(\n(?:.*\n)*?        \)\?;/        super::rollback_round(\&mut v_caches, None, \&v_input, 0, 0, false, device)?;/' "$ROOT/$SPEC/mtp.rs"
    gate charge "a call written on one line is a scan error" 2 \
        "\`mtp_generate\` has a charge site this gate could not read"

    # A call with an argument dropped is read back at the wrong position, so the scan refuses
    # it rather than reporting whatever landed there.
    build_root
    edit 's/            &v_input,\n//' "$ROOT/$SPEC/dflash2/round.rs"
    gate charge "a call with an argument dropped is a scan error" 2 \
        "\`dflash2_generate\` has a charge site this gate could not read"

    # Both spellings renamed: the loops are still derived and now carry no decision at all.
    build_root
    edit 's/rollback_round\(/rewind_round(/g; s/charged:/billed:/g' "$ROOT/$SPEC"/*.rs "$ROOT/$SPEC/dflash2"/*.rs
    gate charge "both charge spellings renamed away is a scan error" 2 \
        "has 0 rollback and 0 record charge sites."

    # The population itself renamed out from under the gate.
    build_root
    edit 's/RoundTotals \{/RoundLedger {/g' "$ROOT/$SPEC"/*.rs "$ROOT/$SPEC/dflash2"/*.rs
    gate charge "a scan that derives no round loop is a scan error, not a pass" 2 \
        "found no round loop"

    # A missing tree is a scan error.
    build_root
    rm -rf "${ROOT:?}/$SPEC"
    gate charge "a missing speculative tree is a scan error" 2 "no speculative source directory"

    # The filename rule, pinned in both directions: the same file carrying a `charged: true`
    # loop is skipped as `*_tests.rs` and scanned once renamed.
    build_root
    mv "$ROOT/$SPEC/round_stats_tests.rs" "$ROOT/$SPEC/round_stats_fixtures.rs"
    gate charge "a charge-shaped loop is scanned once its file is no longer a test file" 1 \
        "the charge census is \"charge_phases:3 false:4 true:1\""

    # The token means the call and nothing else. ORing a second condition into it is RULE 2's
    # own stated defect wearing another spelling.
    build_root
    edit 's/let charge_phases = super::phases_charged\(\);/let charge_phases = super::phases_charged() || force_charge;/' "$ROOT/$SPEC/mtp.rs"
    gate charge "a binding that ORs a second condition into the call is refused" 1 \
        "\`mtp_generate\` charges on \`charge_phases\` and does not bind it"

    # A binding shape the scan cannot read is a scan error, not a verdict about the loop:
    # refusing it as RULE 2 would name a defect that is not there.
    build_root
    edit 's/let charge_phases = super::phases_charged\(\);/let charge_phases: bool = super::phases_charged();/' "$ROOT/$SPEC/dflash2/round.rs"
    gate charge "a binding this gate cannot read is a scan error, not a RULE 2 refusal" 2 \
        "\`dflash2_generate\` binds \`charge_phases\` in a shape this gate"

    # Every rollback, not the first one. A second call carrying the other decision is the same
    # defect at a site a first-match scan never reaches.
    build_root
    edit 's/            \/\/ The same decision, at the draft-side rollback\.\n            charge_phases,/            \/\/ The same decision, at the draft-side rollback.\n            false,/' "$ROOT/$SPEC/mtp.rs"
    gate charge "a second rollback call carrying the other decision is refused" 1 \
        "\`mtp_generate\` names more than one charge decision"

    # The census is a constant, not a default. A gate whose expectation can be relaxed from
    # the environment passes for whoever sets it.
    build_root
    st_case "charge: the census cannot be waived from the environment" 0 \
        --has "census charge_phases:3 false:4" -- \
        env WANT_CENSUS="charge_phases:9 false:9" SPEC_CHARGE_WANT_CENSUS="charge_phases:9 false:9" \
        bash "$TOOL" charge --root "$ROOT"

    # A loop that hands the recorder totals built somewhere else names no decision of its own
    # and drops out of the derived population. On its own that reads as a census that moved,
    # which invites the census to be edited; the rollback it still makes says otherwise.
    build_root
    edit 's/    log_request_record\(\n        &RoundTotals \{\n(?:.*\n)*?    \);/    log_request_record(&totals_for(rounds, false), &emitted, seed_emitted, &window);/' "$ROOT/$SPEC/eagle3.rs"
    gate charge "a loop that falls out of the population is not reported as a census move" 2 \
        "\`eagle3_generate\` rolls a round's caches back and is"

    # The same, from the other condition: a signature wrapped so the marker no longer reads on
    # one line. This is what rustfmt does to a longer parameter.
    build_root
    edit 's/    step_fn: &mut dyn FnMut\(&ProbeStep\) -> Option<u32>,/    step_fn: \&mut dyn FnMut(\n        \&ProbeStep,\n    ) -> Option<u32>,/' "$ROOT/$SPEC/dflash.rs"
    gate charge "a signature wrapped past the marker is not reported as a census move" 2 \
        "\`dflash_generate\` rolls a round's caches back and is"

    # A `where` clause sits between the closing parenthesis and the body. A scan that opens the
    # body at the parenthesis reads the clause as the body and the body as nothing.
    build_root
    edit 's/\) -> Result<Vec<ProbeStep>> \{/) -> Result<Vec<ProbeStep>>\nwhere\n    D: Drafter,\n{/' "$ROOT/$SPEC/gemma4_assistant.rs"
    gate charge "a loop with a \`where\` clause is still read" 0 \
        "OK: 7 classic, 0 forwarded, 0 entries; census charge_phases:3 false:4 (7 sites)."

    # RULE 5, at the shape that opened it: the recorder reading the decision off the totals it
    # was handed. The value is unreadable to this scan, and an unreadable value was refused only
    # inside a round loop.
    build_root
    edit 's/    log_request_record\(totals, emitted, emitted\.len\(\), window\);/    log_request_record(totals, emitted, emitted.len(), window);\n    let _ = Record { charged: totals.charged };/' "$ROOT/$SPEC/mod.rs"
    gate charge "a helper reading the decision off the totals it was handed is refused" 1 \
        "\`emit_seed_token\` writes 1 \`charged:\` field(s)"

    # The same rule with a value the scan can read. A decision named outside a round loop has
    # no rollback beside it to be held to, whether or not the token is legible.
    build_root
    cat >>"$ROOT/$SPEC/mtp.rs" <<'RS'

fn summarise_uncharged(rounds: usize) -> super::RoundStats {
    super::RoundStats {
        rounds,
        charged: false,
    }
}
RS
    gate charge "a decision named outside a round loop is refused" 1 \
        "\`summarise_uncharged\` writes 1 \`charged:\` field(s) — false —"

    # RULE 5 against the fn it is about. The recorder carries the decision across as a
    # destructured `charged,`; spelled as a field it becomes a value named where no rollback
    # can be checked against it.
    build_root
    edit 's/        emitted: emitted\.len\(\),\n        charged,/        emitted: emitted.len(),\n        charged: false,/' "$ROOT/$SPEC/mod.rs"
    gate charge "the recorder naming the decision rather than carrying it across is refused" 1 \
        "\`round_stats\` writes 1 \`charged:\` field(s) — false —"

    # Condition (b) is what keeps the shared seed emit out of the population, and nothing else
    # does: it carries a driver's signature. Give it totals of its own and it joins, with no
    # rollback in it for the gate to read against.
    build_root
    edit 's/    log_request_record\(totals, emitted, emitted\.len\(\), window\);/    log_request_record(\n        \&RoundTotals \{\n            loop_kind: SpecLoop::Kind,\n            rounds: 0,\n            charged: false,\n        \},\n        emitted,\n        emitted.len(),\n        window,\n    );/' "$ROOT/$SPEC/mod.rs"
    gate charge "a helper that builds its own totals joins the population and is refused" 2 \
        "\`emit_seed_token\` has 0 rollback and 1 record charge sites."

    # RULE 6: a loop that keeps its readable `rollback_round` call and reaches past it to the
    # low-level rollback underneath. Every token this gate reads still agrees; the second
    # decision is at a call it does not open.
    build_root
    edit 's/        super::log_round\(/        super::rollback_round_caches(\n            \&mut v_caches,\n            None,\n            \&v_input,\n            0,\n            0,\n            false,\n            device,\n        )?;\n        super::log_round(/' "$ROOT/$SPEC/mtp.rs"
    gate charge "a loop reaching past the shared rollback to the low-level one is a scan error" 2 \
        "\`mtp_generate\` makes 1 call(s) to the low-level"

    # RULE 6 by file: a helper one hop from a loop, which a fn-shaped rule read as a non-driver
    # and let through.
    build_root
    cat >>"$ROOT/$SPEC/dflash.rs" <<'RS'

fn refold_it(lin: Option<&mut [LinearAttnCache]>, device: Device) -> Result<()> {
    refold_lin_tapes(lin, 3, 2, true, device)
}
RS
    gate charge "a helper one hop from a loop naming the low-level rollback is a scan error" 2 \
        "\`refold_it\` makes 1 call(s) to the low-level"

    # RULE 6 by fn: a round loop that lives in `round_common.rs`, where the shared skeleton is
    # going. The exemption is for the code beneath `rollback_round`, not for a loop beside it.
    build_root
    {
        printf '\n'
        loop_src "round_common_generate" "false" "plain"
    } >>"$ROOT/$SPEC/round_common.rs"
    edit 's/(fn round_common_generate\((?:.*\n)*?)    let mut draw/$1    refold_lin_tapes(None, 3, 2, true, device)?;\n    let mut draw/' "$ROOT/$SPEC/round_common.rs"
    gate charge "a round loop in round_common.rs is read for the low-level rollback too" 2 \
        "\`round_common_generate\` makes 1 call(s) to the low-level"

    # The defect the gate exists for, at the seam the record moved to: the totals handed over
    # say the round was charged and the rollbacks say it was not.
    build_root
    edit 's/            charged: charge_phases,\n        \},/            charged: true,\n        },/' "$ROOT/$SPEC/gemma4_assistant.rs"
    gate charge "a record call handing the recorder the other decision is refused" 1 \
        "\`mtp_assistant_generate\` names more than one charge decision"

    # RULE 7 by call: a loop that keeps every charge decision this gate reads and writes its own
    # round event instead of calling the one emitter, on its own module target.
    build_root
    edit 's/        super::log_round\(\n(?:.*\n)*?        \);\n/        tracing::debug!(\n            target: "rmlx_models::speculative::mtp",\n            loop_kind = ?SpecLoop::Kind,\n            round = rounds,\n            charged = charge_phases,\n            "speculative round"\n        );\n/' "$ROOT/$SPEC/mtp.rs"
    gate charge "a loop that writes its own round event instead of calling the emit" 2 \
        "\`mtp_generate\` reaches the one round emit 0 time(s)."

    # RULE 7 by name, the mutation the rule was written for: the same bypass on the shared
    # target. Every field in the same order, so the line it writes is byte-identical to the one
    # the emitter would have written and no digest, pair or census can see it.
    build_root
    edit 's/        super::log_round\(\n(?:.*\n)*?        \);\n/        tracing::debug!(\n            target: super::PHASE_TARGET,\n            loop_kind = ?SpecLoop::Kind,\n            round = rounds,\n            charged = charge_phases,\n            "speculative round"\n        );\n/' "$ROOT/$SPEC/mtp.rs"
    gate charge "a loop writing the shared target itself is refused by name" 2 \
        "src/speculative/mtp.rs names the per-round event's target."

    # RULE 7 by name, away from any loop: a helper that merely holds the string. Every loop
    # still reaches the emitter exactly once, so only the name reading fires.
    build_root
    cat >>"$ROOT/$SPEC/dflash.rs" <<'RS'

fn phase_target() -> &'static str {
    "rmlx::spec::phase"
}
RS
    gate charge "a helper naming the phase target outside its file is refused" 2 \
        "src/speculative/dflash.rs names the per-round event's target."

    # RULE 7 by call, the other way: one loop closing its round twice. Two record shapes under
    # one target is what the collapse removed.
    build_root
    edit 's/(        super::log_round\(\n(?:.*\n)*?        \);\n)/$1$1/' "$ROOT/$SPEC/eagle3.rs"
    gate charge "a loop closing its round twice is refused" 2 \
        "\`eagle3_generate\` reaches the one round emit 2 time(s)."

    # RULE 7 by name, by fn: a round loop that moved into the file that owns the target. It
    # calls the emitter once, and the file reading exempts the path, so only the fn reading sees
    # the second event it writes on the target.
    build_root
    {
        printf '//! The shared record and the one round emit.\n\n'
        loop_src "eagle3_generate" "false" "plain"
    } >"$ROOT/$SPEC/round_stats.rs"
    printf '//! The loop that used to live here moved into the shared record.\n' \
        >"$ROOT/$SPEC/eagle3.rs"
    edit 's/(        super::log_round\(\n(?:.*\n)*?        \);\n)/$1        tracing::debug!(target: PHASE_TARGET, round = rounds, "speculative round");\n/' "$ROOT/$SPEC/round_stats.rs"
    gate charge "a round loop inside the target's own file does not inherit its exemption" 2 \
        "\`eagle3_generate\` is a round loop and names the"

    # The needle is code, not prose. A trailing comment naming the emit on a line that is not
    # one would otherwise count as a call.
    build_root
    edit 's/        super::log_round\(\n(?:.*\n)*?        \);\n/        rounds += 1; \/\/ the round the shared log_round( would have closed\n/' "$ROOT/$SPEC/eagle3.rs"
    gate charge "a trailing comment naming the emit is not a call to it" 2 \
        "\`eagle3_generate\` reaches the one round emit 0 time(s)."

    # Every needle reads code. A doc comment in a non-loop file naming the target is prose, not a
    # second event on it.
    build_root
    cat >>"$ROOT/$SPEC/round_common.rs" <<'RS'

/// The per-round event goes out on `PHASE_TARGET`, and every loop reaches it
/// through the one `log_round(` call this gate counts.
fn seam_note() {}
RS
    gate charge "a doc comment naming the target and the emit is prose, not either" 0 \
        "OK: 7 classic, 0 forwarded, 0 entries; census charge_phases:3 false:4 (7 sites)."

    # The collapse: the decision travels as a value. The cases above are all one tree shape,
    # seven loops each deciding for itself. Below are the two shapes the campaign passes
    # through: mid-campaign and the end state. The census is the same seven decisions in every
    # one of them, which is the property that makes it worth reading at all.

    # The mid-campaign tree. The forwarded loop is named on the success line.
    build_mid_root
    gate charge "the mid-campaign tree passes, and names what it passed as" 0 \
        "OK: 6 classic, 1 forwarded, 1 entries; census charge_phases:3 false:4 (7 sites). Forwarded: \`round_loop_generate\`."

    # The mutation the re-key exists for: the forwarded loop hard-wires the decision it was
    # handed. Both readings fire on the one tree: RULE 8 names the line, the census the count.
    build_mid_root
    edit 's/    let charge = cfg\.charged;/    let charge = false;/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a forwarded loop that hard-wires its charge is named by RULE 8" 1 \
        "\`round_loop_generate\` is handed its charge in \`cfg\` and applies"
    gate charge "the same hard-wire puts an eighth decision in the census" 1 \
        "the charge census is \"charge:1 charge_phases:3 false:4\" (8 sites)"

    # The whole right-hand side, not a prefix of it.
    build_mid_root
    edit 's/    let charge = cfg\.charged;/    let charge = cfg.charged || force_charge;/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a forwarded loop ORing a condition into the field it was handed is refused" 1 \
        "\`round_loop_generate\` is handed its charge in \`cfg\` and applies"

    # The other legitimate spelling: no binding at all, the field named at every site.
    build_mid_root
    edit 's/    let charge = cfg\.charged;\n//; s/^(\s+)charge,$/$1cfg.charged,/gm; s/charged: charge,/charged: cfg.charged,/g' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a forwarded loop naming the field at every site is read as one token" 0 \
        "OK: 6 classic, 1 forwarded, 1 entries; census charge_phases:3 false:4 (7 sites)."

    # An entry that drops its decision. Excluding the forwarded loop from the census must not
    # hide it: RULE 5's zero-arm names the entry, and the census reads six sites.
    build_end_root
    edit 's/            charged: charge_phases,\n//' "$ROOT/$SPEC/gemma4_assistant.rs"
    gate charge "an entry that states no decision is named by RULE 5" 1 \
        "\`mtp_assistant_generate\` builds a round loop's configuration and states"
    gate charge "the same dropped decision is a census that lost a site" 1 \
        "the charge census is \"charge_phases:2 false:4\" (6 sites)"

    # RULE 2 over the entries: the token still has to mean the call.
    build_end_root
    edit 's/    let charge_phases = super::phases_charged\(\);/    let charge_phases = true;/' "$ROOT/$SPEC/dflash2/round.rs"
    gate charge "an entry binding \`charge_phases\` to a literal is refused" 1 \
        "\`dflash2_generate\` charges on \`charge_phases\` and does not bind it"

    # An entry genuinely moved from one schedule to the other, binding and all.
    build_end_root
    edit 's/            charged: false,/            charged: charge_phases,/' "$ROOT/$SPEC/eagle3.rs"
    edit 's/(\) -> Result<Vec<ProbeStep>> \{\n)/$1    let charge_phases = super::phases_charged();\n/' "$ROOT/$SPEC/eagle3.rs"
    gate charge "an entry moved between the two schedules is refused" 1 \
        "the charge census is \"charge_phases:4 false:3\" (7 sites)"

    # RULE 5 unchanged: a fn in neither population naming a decision.
    build_mid_root
    cat >>"$ROOT/$SPEC/mtp.rs" <<'RS'

fn summarise_uncharged(rounds: usize) -> super::RoundStats {
    super::RoundStats {
        rounds,
        charged: false,
    }
}
RS
    gate charge "a decision named outside both populations is refused" 1 \
        "\`summarise_uncharged\` writes 1 \`charged:\` field(s) — false —"

    # RULE 4's second arm: a drafter rolling its own state back carries the decision it was
    # handed, as a field of one of its own parameters.
    build_mid_root
    drafter_rollback_src "ctx.charged" >>"$ROOT/$SPEC/mtp.rs"
    gate charge "a drafter rollback carrying the decision it was handed passes" 0 \
        "OK: 6 classic, 1 forwarded, 1 entries; census charge_phases:3 false:4 (7 sites)."

    # The same rollback deciding for itself. A literal there is a second decision made where
    # nothing can hold it to the loop that ordered it.
    build_mid_root
    drafter_rollback_src "false" >>"$ROOT/$SPEC/mtp.rs"
    gate charge "a drafter rollback that decides for itself is refused" 1 \
        "\`rollback\` is handed the round's context in \`ctx\` and"

    # An argument the scan cannot read back is unreadable, not absent.
    build_mid_root
    drafter_rollback_src "ctx.charge_flag()" >>"$ROOT/$SPEC/mtp.rs"
    gate charge "a drafter rollback on an unreadable charge is a scan error" 2 \
        "\`rollback\` rolls a round back on a \`charge\` this gate"

    # The end state: one loop body, seven entries, the same seven decisions.
    build_end_root
    gate charge "the end state passes with every decision in an entry" 0 \
        "OK: 0 classic, 1 forwarded, 7 entries; census charge_phases:3 false:4 (7 sites). Forwarded: \`round_loop_generate\`."

    # The end state with the one loop deleted. A scan that finds nothing must not pass, and at
    # the end state there is one body left to find.
    build_end_root
    rm -f "$ROOT/$SPEC/round_loop.rs"
    gate charge "the end state with its one loop deleted is a scan error" 2 \
        "found no round loop"

    # RULE 6 over the shared loop: the low-level rollback is round_common's alone, and the loop
    # beside it does not inherit that exemption.
    build_mid_root
    edit 's/        super::log_round\(/        super::rollback_round_caches(\n            \&mut v_caches,\n            None,\n            \&v_input,\n            0,\n            0,\n            false,\n            device,\n        )?;\n        super::log_round(/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "the shared loop reaching past the shared rollback is a scan error" 2 \
        "\`round_loop_generate\` makes 1 call(s) to the low-level"

    # RULE 7 over the shared loop: one round, one event.
    build_mid_root
    edit 's/(        super::log_round\(\n(?:.*\n)*?        \);\n)/$1$1/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "the shared loop closing its round twice is refused" 2 \
        "\`round_loop_generate\` reaches the one round emit 2 time(s)."

    # The shadow, on both arms. On the forwarded loop RULE 8 has something exact to say about
    # the second binding, so it is exit 1 and names it; on a classic loop the same-token
    # reading has become vacuous and it is exit 2.
    build_mid_root
    edit 's/(    let charge = cfg\.charged;\n)/$1    let charge = false;\n/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a forwarded loop shadowing its binding is named by RULE 8" 1 \
        "\`round_loop_generate\` binds its charge token twice; the last binds it"

    build_mid_root
    edit 's/(    let charge_phases = super::phases_charged\(\);\n)/$1    let charge_phases = false;\n/' "$ROOT/$SPEC/gemma4_assistant.rs"
    gate charge "a classic loop shadowing its binding is a scan error" 2 \
        "\`mtp_assistant_generate\` binds its charge token more than once"

    # RULE 5 over the entries: an entry names one decision, and two are two decisions with
    # nothing saying which one the loop applies.
    build_end_root
    edit 's/(            charged: false,\n)/$1            charged: true,\n/' "$ROOT/$SPEC/dflash.rs"
    gate charge "an entry stating two decisions is refused" 1 \
        "\`dflash_generate\` builds a round loop's configuration and states"

    # The entry arm of the shadow, and not a corner: at the end state all seven decisions live
    # in entries, so an unread shadow there is the whole census.
    build_end_root
    edit 's/(    let charge_phases = super::phases_charged\(\);\n)/$1    let charge_phases = false;\n/' "$ROOT/$SPEC/mtp.rs"
    gate charge "an entry shadowing its binding is a scan error" 2 \
        "\`mtp_generate\` binds its charge token more than once"

    # An entry that stops building the configuration leaves the population. That must read as a
    # lost decision rather than as a clean scan, which is what the census does and no membership
    # test could.
    build_end_root
    edit 's/    super::round_loop_generate\(\n(?:.*\n)*?    \)\n/    super::round_loop_legacy(verifier, charge_phases, step_fn, device)\n/' "$ROOT/$SPEC/mtp.rs"
    gate charge "an entry that stops building the configuration is a lost decision" 1 \
        "the charge census is \"charge_phases:2 false:4\" (6 sites)"

    # The configuration handed over is read-only inside the loop. A loop that builds its own and
    # forwards faithfully from that writes no site any reading here can see.
    build_mid_root
    edit 's/    let charge = cfg\.charged;/    let cfg = RoundCfg::uncharged();\n    let charge = cfg.charged;/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a forwarded loop rebinding the configuration's name is a scan error" 2 \
        "\`round_loop_generate\` rebinds or writes \`cfg\`, the configuration it"

    # The same, by assignment rather than by rebinding.
    build_mid_root
    edit 's/    cfg: &RoundCfg,/    cfg: \&mut RoundCfg,/; s/    let charge = cfg\.charged;/    cfg.charged = false;\n    let charge = cfg.charged;/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a forwarded loop writing the configuration's charge field is a scan error" 2 \
        "\`round_loop_generate\` takes its \`RoundCfg\` by \`&mut\` (or by"

    # The delimiter anchor holds: a call is not the field it resembles, and a reader without the
    # anchor would count this as forwarded.
    build_mid_root
    edit 's/                charged: charge,/                charged: cfg.charged(),/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a call that resembles the configuration field is a scan error" 2 \
        "\`round_loop_generate\` has a charge site this gate could not read"

    # Not a readable binding form. A reader that guesses is worse than one that refuses.
    build_mid_root
    edit 's/    let charge = cfg\.charged;/    let mut charge = cfg.charged;/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a forwarded loop binding through a \`let mut\` is a scan error" 2 \
        "\`round_loop_generate\` binds \`charge\` in a shape this gate cannot"

    # The other half of the per-binding reading: a token the sites name and no binding anywhere.
    build_root
    edit 's/    let charge_phases = super::phases_charged\(\);\n//' "$ROOT/$SPEC/mtp.rs"
    gate charge "a loop naming a token it never binds is refused" 1 \
        "\`mtp_generate\` names \`charge_phases\` as its decision and never binds"

    # The looser reading's own hole, closed. A second parameter carrying a field of the same
    # name satisfies "a field of one of its own parameters" while holding a different decision,
    # so where a fn is handed the round's context the charge comes off that and nothing else.
    build_mid_root
    drafter_rollback_src "v.charged" >>"$ROOT/$SPEC/mtp.rs"
    gate charge "a drafter rollback charging off the wrong parameter is refused" 1 \
        "\`rollback\` is handed the round's context in \`ctx\` and"

    # RULE 8's read-only half is about the value and not one spelling of it: the whole
    # configuration replaced through its own reference writes no site any other reading sees.
    build_mid_root
    edit 's/    cfg: &RoundCfg,/    cfg: \&mut RoundCfg,/; s/    let charge = cfg\.charged;/    *cfg = RoundCfg::uncharged();\n    let charge = cfg.charged;/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a forwarded loop assigning through the configuration's reference is a scan error" 2 \
        "\`round_loop_generate\` takes its \`RoundCfg\` by \`&mut\` (or by"

    # The same, moved rather than assigned.
    build_mid_root
    edit 's/    cfg: &RoundCfg,/    cfg: \&mut RoundCfg,/; s/    let charge = cfg\.charged;/    let old = std::mem::replace(cfg, RoundCfg::uncharged());\n    let charge = old.charged;/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a forwarded loop swapping its configuration out is a scan error" 2 \
        "\`round_loop_generate\` takes its \`RoundCfg\` by \`&mut\` (or by"

    # Two parameters of the configuration's type. A rule that took the last one declared would
    # read the loop's token against the wrong parameter.
    build_mid_root
    edit 's/    cfg: &RoundCfg,/    cfg: \&RoundCfg,\n    fallback: \&RoundCfg,/; s/    let charge = cfg\.charged;/    let charge = fallback.charged;/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a loop taking two configurations is a scan error, not a guess" 2 \
        "\`round_loop_generate\` takes a \`RoundCfg\` this gate could not name"

    # Every needle reads code, the parameter list included. A comment naming a type in a
    # signature is prose: read as declarations, it would make a classic loop forwarded.
    build_mid_root
    edit 's/    device: Device,\n\) -> Result<Vec<ProbeStep>> \{/    device: Device, \/\/ not a RoundCfg, and not a RoundCtx either\n) -> Result<Vec<ProbeStep>> {/' "$ROOT/$SPEC/dflash.rs"
    gate charge "a comment in a parameter list is prose, not a declaration" 0 \
        "OK: 6 classic, 1 forwarded, 1 entries; census charge_phases:3 false:4 (7 sites)."

    # The property, not the spellings: a configuration handed over by value can be written the
    # same ways, and a rule that read only the lines would have to enumerate them for ever.
    build_mid_root
    edit 's/    cfg: &RoundCfg,/    cfg: RoundCfg,/; s/    let charge = cfg\.charged;/    cfg.set_charged(false);\n    let charge = cfg.charged;/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a forwarded loop taking its configuration by value is a scan error" 2 \
        "\`round_loop_generate\` takes its \`RoundCfg\` by \`&mut\` (or by"

    # The four line shapes are still read, which covers a write this gate can see on a parameter
    # it has already accepted as immutable.
    build_mid_root
    edit 's/    let charge = cfg\.charged;/    cfg.charged = false;\n    let charge = cfg.charged;/' "$ROOT/$SPEC/round_loop.rs"
    gate charge "a write to the configuration is read even where the parameter is immutable" 2 \
        "\`round_loop_generate\` rebinds or writes \`cfg\`, the configuration it"

    # A pair, because a case without its control proves nothing about which reading fired: the
    # low-level rollback after a loop's record is refused, and refused just the same when a
    # comment carrying an unbalanced brace sits in front of it. A body whose extent is read off
    # the raw line ends at that comment, and every needle after it belongs to no function.
    build_root
    edit 's/    Ok\(vec!\[\]\)\n\}/    super::rollback_round_caches(\&mut v_caches, None, \&\[\], 0, 0, false, device)?;\n    Ok(vec![])\n}/' "$ROOT/$SPEC/mtp.rs"
    gate charge "a low-level rollback after a loop record is refused" 2 \
        "\`mtp_generate\` makes 1 call(s) to the low-level"

    build_root
    edit 's/    Ok\(vec!\[\]\)\n\}/    \/\/ the round closes here }\n    super::rollback_round_caches(\&mut v_caches, None, \&\[\], 0, 0, false, device)?;\n    Ok(vec![])\n}/' "$ROOT/$SPEC/mtp.rs"
    gate charge "a comment carrying a brace does not end the body it sits in" 2 \
        "\`mtp_generate\` makes 1 call(s) to the low-level"

    # A trait declaring the drafters' interface, in the file the shared loop lives in. A scan
    # that waits for the next `{` reads the loop as `condition`'s body, so the loop is never
    # recorded and the rollback and record sites in it belong to a trait method that has neither.
    build_mid_root
    loop_file="$ROOT/$SPEC/round_loop.rs"
    {
        head -2 "$loop_file"
        drafter_trait_src
        tail -n +3 "$loop_file"
    } >"$ST_WORK/with_trait.rs"
    mv "$ST_WORK/with_trait.rs" "$loop_file"
    gate charge "a trait of bodiless declarations does not swallow the loop beneath it" 0 \
        "OK: 6 classic, 1 forwarded, 1 entries; census charge_phases:3 false:4 (7 sites)."

    # Entries with no loop to enter. The census cannot see a lost loop: its decision leaves the
    # census exactly as the entry's decision enters it, which is what a migration does on purpose.
    build_mid_root
    rm -f "$ROOT/$SPEC/round_loop.rs"
    gate charge "an entry with no forwarded loop to enter is a scan error" 2 \
        "the scan found 1 entries and no forwarded loop"

    # list-drivers reads the population the charge rules read.
    build_root
    st_case "list-drivers: every round loop is listed with its file, fn and line, sorted" 0 \
        --has "$SPEC/dflash.rs"$'\t'"dflash_generate"$'\t'1 \
        --has "$SPEC/dflash2/round.rs"$'\t'"dflash2_generate"$'\t'1 \
        --has "$SPEC/mod.rs"$'\t'"spec_generate_stochastic_cached"$'\t' \
        --lacks "round_stats_tests.rs" --lacks "mtp_tests.rs" --lacks "spec_generate_greedy"$'\t' -- \
        bash "$TOOL" list-drivers --root "$ROOT"

    build_end_root
    st_case "list-drivers: the end state lists its one loop and no entry" 0 \
        --is "$SPEC/round_loop.rs"$'\t'"round_loop_generate"$'\t'3 -- \
        bash "$TOOL" list-drivers --root "$ROOT"

    build_root
    edit 's/RoundTotals \{/RoundLedger {/g' "$ROOT/$SPEC"/*.rs "$ROOT/$SPEC/dflash2"/*.rs
    st_case "list-drivers: an empty population is a refusal, not an empty listing" 2 \
        --has "check-spec-charge: found no round loop" -- \
        bash "$TOOL" list-drivers --root "$ROOT"

    build_root
    rm -rf "${ROOT:?}/$SPEC"
    st_case "list-drivers: a missing speculative tree is a refusal" 2 \
        --has "no speculative source directory" -- \
        bash "$TOOL" list-drivers --root "$ROOT"

    # The arguments. A mode is required and named, and a root is one directory given once; a
    # typo must not run the default invocation.
    st_case "usage: no mode is refused" 2 --has "usage: bash scripts/check/spec_round_loops.sh" -- \
        bash "$TOOL"
    st_case "usage: an unknown mode is refused" 2 --has "usage: bash scripts/check/spec_round_loops.sh" -- \
        bash "$TOOL" list-driver --root "$ROOT"
    st_case "usage: an unknown option is refused" 2 --has "usage: bash scripts/check/spec_round_loops.sh" -- \
        bash "$TOOL" charge --rot "$ROOT"
    st_case "usage: --root without a directory is refused" 2 --has "usage: bash scripts/check/spec_round_loops.sh" -- \
        bash "$TOOL" charge --root
    st_case "usage: an empty --root is refused" 2 --has "usage: bash scripts/check/spec_round_loops.sh" -- \
        bash "$TOOL" charge --root ""
    st_case "usage: a repeated --root is refused" 2 --has "usage: bash scripts/check/spec_round_loops.sh" -- \
        bash "$TOOL" sampling --root "$ROOT" --root "$ROOT"
}

# ---- sampling ----------------------------------------------------------------

cases_sampling() {
    local loop_file

    build_root
    gate sampling "a clean root passes, and the private loops are in the population" 0 \
        "OK: 7 loops (0 forwarded), 0 entries, 1 guards"

    # A loop that takes no sampler at all, the state every sidecar was in.
    build_root
    edit 's/    sampler_cfg: &crate::sampler::SamplerConfig,\n//; s/super::VerifierDraw::new\(sampler_cfg\)/super::VerifierDraw::new(\&greedy())/' "$ROOT/$SPEC/mtp.rs"
    gate sampling "a loop with no sampler parameter is refused" 1 \
        "\`mtp_generate\` drives a generation but takes neither a"

    # A loop that takes the sampler and builds no draw from it. The signature satisfies a
    # caller and the loop still decodes greedily.
    build_root
    edit 's/super::VerifierDraw::new\(sampler_cfg\)/super::VerifierDraw::new(\&greedy());\n    let _ = sampler_cfg/' "$ROOT/$SPEC/dflash2/round.rs"
    gate sampling "a loop that never draws with its sampler is refused" 1 \
        "\`dflash2_generate\` is handed the request's sampler and builds"

    # One dispatch arm dropping the configuration while the others keep it.
    build_root
    edit 's/            &mut step_fn,\n            &spec_sampler_cfg,\n            dispatcher.device\(\),\n        \),\n    \};/            \&mut step_fn,\n            dispatcher.device(),\n        ),\n    };/' "$ROOT/crates/rmlx-server/src/engine/speculative.rs"
    gate sampling "one dispatch arm dropping the sampler is refused" 1 \
        "the \`Drafter::DFlash2\` arm drives a"

    # A driver added beside the seven. The path census is what sees it: a tree with eight
    # generation paths is not the tree this gate was pointed at.
    build_root
    loop_src "eagle9_generate" "false" "plain" >>"$ROOT/$SPEC/mtp.rs"
    gate sampling "an eighth generation path is a scan error" 2 \
        "the tree has 8 drafter generation paths and records"

    # The scan finding nothing must not pass. This is the failure a renamed argument would
    # produce, and it is the one a gate reports as clean.
    build_root
    edit 's/step_fn: &mut dyn FnMut\(&ProbeStep\)/emit: \&mut dyn FnMut(\&Step)/g' "$ROOT/$SPEC"/*.rs "$ROOT/$SPEC/dflash2"/*.rs
    gate sampling "a scan that matches no loop is a scan error, not a pass" 2 \
        "found no round loop"

    # A dispatch this gate cannot find is also a scan error.
    build_root
    edit 's/let result = match &drafter \{/let result = match drafter_kind {/' "$ROOT/crates/rmlx-server/src/engine/speculative.rs"
    gate sampling "a dispatch the gate cannot read is a scan error, not a pass" 2 \
        "Rule 4 scanned nothing"

    # A missing tree is a scan error.
    build_root
    rm -rf "${ROOT:?}/$SPEC"
    gate sampling "a missing speculative tree is a scan error" 2 "no speculative source directory"

    # A missing dispatch is a scan error.
    build_root
    rm -f "$ROOT/crates/rmlx-server/src/engine/speculative.rs"
    gate sampling "a missing dispatch is a scan error" 2 "no speculative dispatch"

    # A sibling test file must not be scanned: its harness takes no sampler and builds a
    # `RoundTotals`, so a scan that read it would report eight paths.
    build_root
    gate sampling "a loop-shaped fn in a sibling test file is not scanned" 0 \
        "OK: 7 loops (0 forwarded), 0 entries, 1 guards"

    # The guard's own defect class: one route that drops the configuration while the other
    # honours it. This is the fn that decides which loop a sampled request reaches.
    build_root
    edit 's/spec_generate_stochastic_cached\(verifier, step_fn, sampler_cfg, device\)/spec_generate_stochastic_cached(verifier, step_fn, device)/' "$ROOT/$SPEC/mod.rs"
    gate sampling "a guard dropping the sampler on one route is refused" 1 \
        "\`spec_generate_greedy\` runs \`spec_generate_stochastic_cached\` without passing the"

    # The shape the tree has: the path the gate used to exempt by name is an entry, and the
    # clean root that carries it passes and says so.
    build_entry_root
    gate sampling "the two-model greedy path as an entry passes, and names what it passed as" 0 \
        "OK: 6 loops (1 forwarded), 2 entries, 1 guards"

    # That path with no sampler parameter. This is exactly what the exemption waived for as long
    # as the path was a loop that took none, and nothing waives it now: the gate reads no name.
    build_entry_root
    edit 's/(fn spec_generate_greedy_cached\(.*?)    sampler_cfg: &crate::sampler::SamplerConfig,\n/$1/s; s/(RoundCfg \{.*?)            sampler_cfg,\n/$1/s' "$ROOT/$SPEC/mod.rs"
    gate sampling "the two-model greedy path buys nothing from its name" 1 \
        "\`spec_generate_greedy_cached\` starts a round loop and takes no"

    # That path taking the sampler and leaving it out of the configuration it hands over: the
    # request's temperature stops at the entry, with the signature and the guard both reading
    # correctly.
    build_entry_root
    edit 's/(fn spec_generate_greedy_cached\(.*?RoundCfg \{.*?)            sampler_cfg,\n/$1/s' "$ROOT/$SPEC/mod.rs"
    gate sampling "the two-model greedy entry that drops the sampler is refused" 1 \
        "\`spec_generate_greedy_cached\` takes \`sampler_cfg\` and leaves it out of the"

    # The tree's own census: one shared loop and seven entries, the end state the collapse
    # reaches. This is the shape every case above is evidence about.
    build_end_root
    gate sampling "the tree's own census passes, and names what it passed as" 0 \
        "OK: 1 loops (1 forwarded), 7 entries, 1 guards"

    # The mid-campaign tree: one path is an entry now, and the loop it runs is handed the
    # sampler inside the configuration the entry built.
    build_mid_root
    gate sampling "the mid-campaign tree passes, and names what it passed as" 0 \
        "OK: 7 loops (1 forwarded), 1 entries, 1 guards"

    # The shape the entries exist to refuse: a sampler an entry accepts and leaves out of the
    # configuration it hands over. Nothing else can see it: the signature is right, the name
    # appears in the body, and the loop draws from whatever the configuration carried.
    build_mid_root
    edit 's/            sampler_cfg,\n//; s/    super::round_loop_generate\(/    let _ = sampler_cfg;\n    super::round_loop_generate(/' "$ROOT/$SPEC/mtp.rs"
    gate sampling "an entry that leaves the sampler out of the configuration is refused" 1 \
        "\`mtp_generate\` takes \`sampler_cfg\` and leaves it out of the"

    # An entry that never took one. The request's temperature stops there.
    build_mid_root
    edit 's/    sampler_cfg: &crate::sampler::SamplerConfig,\n//; s/            sampler_cfg,\n//' "$ROOT/$SPEC/mtp.rs"
    gate sampling "an entry with no sampler parameter is refused" 1 \
        "\`mtp_generate\` starts a round loop and takes no"

    # The shared loop reading the sampler out of the configuration is the read RULE 1 asks for;
    # the same loop drawing from a greedy default is not.
    build_mid_root
    edit 's/super::VerifierDraw::new\(cfg\.sampler_cfg\)/super::VerifierDraw::new(\&greedy());\n    let _ = cfg.sampler_cfg/' "$ROOT/$SPEC/round_loop.rs"
    gate sampling "the shared loop that draws from no sampler is refused" 1 \
        "\`round_loop_generate\` is handed the request's sampler and builds"

    # A lost entry. The seven paths are what the campaign moves between shapes, never the count,
    # so six of them is a scan that stopped looking rather than a tree with one drafter fewer.
    build_mid_root
    printf '//! The path that used to be here.\n' >"$ROOT/$SPEC/mtp.rs"
    gate sampling "a lost entry is a scan error, not a quieter run" 2 \
        "the tree has 6 drafter generation paths and records"

    # A needle in a comment is not the call it names. The commented-out draw is what a change
    # like this leaves behind, and beside a greedy one it is a loop that ignores the request
    # while reading as one that honours it.
    build_root
    edit 's|    let mut draw = super::VerifierDraw::new\(sampler_cfg\);|    // let mut draw = super::VerifierDraw::new(sampler_cfg);\n    let mut draw = super::VerifierDraw::new(\&greedy());|' "$ROOT/$SPEC/mtp.rs"
    gate sampling "a commented-out draw is not a draw" 1 \
        "\`mtp_generate\` is handed the request's sampler and builds"

    # The same on the signature side: a commented-out parameter is not one.
    build_root
    edit 's|    sampler_cfg: &crate::sampler::SamplerConfig,|    // sampler_cfg: \&crate::sampler::SamplerConfig,|; s|super::VerifierDraw::new\(sampler_cfg\)|super::VerifierDraw::new(\&greedy())|' "$ROOT/$SPEC/gemma4_assistant.rs"
    gate sampling "a commented-out sampler parameter is not a parameter" 1 \
        "\`mtp_assistant_generate\` drives a generation but takes neither a"

    # A needle inside a string literal is text a program prints, not a construction it makes.
    build_root
    edit 's|    let mut draw = super::VerifierDraw::new\(sampler_cfg\);|    let note = "VerifierDraw::new(sampler_cfg)";\n    let mut draw = super::VerifierDraw::new(\&greedy());|' "$ROOT/$SPEC/dflash.rs"
    gate sampling "a needle inside a string literal is not a draw" 1 \
        "\`dflash_generate\` is handed the request's sampler and builds"

    # The converse, and the reason the construction is followed to its closing parenthesis: a
    # correct loop must not be refused for the width of its line.
    build_root
    edit 's|    let mut draw = super::VerifierDraw::new\(sampler_cfg\);|    let mut draw = super::VerifierDraw::new(\n        sampler_cfg,\n    );|' "$ROOT/$SPEC/eagle3.rs"
    gate sampling "a draw wrapped over two lines is still a draw" 0 \
        "OK: 7 loops (0 forwarded), 0 entries, 1 guards"

    # Every needle reads code, the parameter list included: a comment naming a type or a marker
    # in a signature is prose, and reading it would move the population.
    build_root
    edit 's|    device: Device,\n\) -> Result<Vec<ProbeStep>> \{|    device: Device, // not a RoundCfg, and no step_fn: \&mut dyn FnMut(\&ProbeStep) here\n) -> Result<Vec<ProbeStep>> {|' "$ROOT/$SPEC/dflash2/round.rs"
    gate sampling "a comment in a parameter list is prose, not a declaration" 0 \
        "OK: 7 loops (0 forwarded), 0 entries, 1 guards"

    # The dispatch scan reads code too, and it is the arm reading that is this gate's own defect
    # class: a sampler kept as a comment is passed to nothing.
    build_root
    edit 's|            &spec_sampler_cfg,\n            dispatcher.device\(\),\n        \),\n    \};|            // \&spec_sampler_cfg,\n            dispatcher.device(),\n        ),\n    };|' "$ROOT/crates/rmlx-server/src/engine/speculative.rs"
    gate sampling "a dispatch arm keeping the sampler in a comment is refused" 1 \
        "the \`Drafter::DFlash2\` arm drives a"

    # And the same through a string literal, which is text the arm prints rather than a
    # configuration it passes.
    build_root
    edit 's|            &spec_sampler_cfg,\n            dispatcher.device\(\),\n        \),\n    \};|            tracing::debug!("no spec_sampler_cfg here"),\n            dispatcher.device(),\n        ),\n    };|' "$ROOT/crates/rmlx-server/src/engine/speculative.rs"
    gate sampling "a dispatch arm naming the sampler inside a string literal is refused" 1 \
        "the \`Drafter::DFlash2\` arm drives a"

    # The same trait, and here the loop it swallows is the one that draws. The path census
    # cannot see the loss on its own: the entry that replaced the migrated drafter fills the
    # slot, one for one, so a scan that lost the shared loop still counts seven paths.
    build_mid_root
    loop_file="$ROOT/$SPEC/round_loop.rs"
    {
        head -2 "$loop_file"
        drafter_trait_src
        tail -n +3 "$loop_file"
    } >"$ST_WORK/with_trait.rs"
    mv "$ST_WORK/with_trait.rs" "$loop_file"
    gate sampling "a trait of bodiless declarations does not swallow the loop beneath it" 0 \
        "OK: 7 loops (1 forwarded), 1 entries, 1 guards"

    # And the invariant that would have caught it whatever the cause: an entry hands the sampler
    # to a loop that takes a `RoundCfg`, so entries with none to enter are a loop the scan lost.
    build_mid_root
    rm -f "$ROOT/$SPEC/round_loop.rs"
    gate sampling "an entry with no forwarded loop to enter is a scan error" 2 \
        "the scan found 1 entries and no forwarded loop"

    # The shape the second needle used to admit, and the reason it is gone: a loop that seeds its
    # own generator from the request's seed draws from a stream nothing else in the request
    # shares. It reads as reproducible on its own while being a second stream correlated with the
    # one the round's draw advances.
    build_root
    edit 's|    let mut draw = super::VerifierDraw::new\(sampler_cfg\);|    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());|' "$ROOT/$SPEC/mod.rs"
    gate sampling "a loop that seeds a second generator instead of drawing is refused" 1 \
        "\`spec_generate_greedy_cached\` is handed the request's sampler and builds"

    # A character literal holding a quote opens no string, so the needle in the literal after it
    # is still text a program prints and not a construction.
    build_root
    edit 's|    let mut draw = super::VerifierDraw::new\(sampler_cfg\);|    let note = \x27"\x27; let text = "VerifierDraw::new(sampler_cfg)";\n    let mut draw = super::VerifierDraw::new(\&greedy());|' "$ROOT/$SPEC/dflash.rs"
    gate sampling "a needle in a string literal after a quote character literal is not a draw" 1 \
        "\`dflash_generate\` is handed the request's sampler and builds"

    # The root is an argument. The variable the old gate read it from is gone, so a value left in
    # the environment points the scan nowhere.
    build_root
    st_case "sampling: the root cannot be set from the environment" 0 \
        --has "OK: 7 loops (0 forwarded), 0 entries, 1 guards" -- \
        env SPEC_SAMPLING_ROOT=/nonexistent bash "$TOOL" sampling --root "$ROOT"
}

case "$GROUP" in
charge) cases_charge ;;
sampling) cases_sampling ;;
both)
    cases_charge
    cases_sampling
    ;;
esac
st_done
