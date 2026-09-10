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
# against whatever the repository happens to contain today. It carries the same
# seven drafter paths the tree does, one of them the exempt two-model greedy
# loop and one of them the stochastic loop that draws through its own RNG rather
# than the shared draw. Two roots are built: today's, where every path is a loop
# body, and the mid-campaign one, where a drafter has been migrated and its path
# is an entry that hands the sampler to the shared loop.

set -uo pipefail

script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_spec_sampling.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

failures=0
cases=0

# One round loop: a driver's signature, the request's sampler, the draw built
# from it, and the `RoundTotals` that makes it a loop rather than a helper.
loop_src() {
  local name="$1"
  cat <<RS
pub fn $name(
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
    log_request_record(
        &RoundTotals {
            loop_kind: SpecLoop::Kind,
            rounds,
            charged: false,
        },
        &emitted,
    );
    Ok(vec![])
}
RS
}

# The two-model greedy loop: no sampler at all. It runs at temperature 0, where
# the verifier's argmax is the draw, and it is the one name the gate exempts.
greedy_cached_src() {
  cat <<'RS'
fn spec_generate_greedy_cached(
    verifier: &Architecture,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    log_request_record(
        &RoundTotals {
            loop_kind: SpecLoop::Kind,
            rounds,
            charged: false,
        },
        &emitted,
    );
    Ok(vec![])
}
RS
}

# The stochastic loop: the one acceptance rule that is not the shared one. Its
# draw is a per-request RNG stream seeded from the same configuration, which is
# a needle any loop may satisfy rather than a name the gate exempts.
stochastic_cached_src() {
  cat <<'RS'
fn spec_generate_stochastic_cached(
    verifier: &Architecture,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    log_request_record(
        &RoundTotals {
            loop_kind: SpecLoop::Kind,
            rounds,
            charged: false,
        },
        &emitted,
    );
    Ok(vec![])
}
RS
}

# The two-model entry guard: it routes a request to one of two loops by reading
# whether the sampler is active, and passes it to the one that takes it.
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
        spec_generate_greedy_cached(verifier, step_fn, device)
    }
}
RS
}

# The shared loop, handed the request's sampler inside the configuration the
# entry built for it.
forwarded_loop_src() {
  cat <<'RS'
pub(crate) fn round_loop_generate(
    verifier: &Architecture,
    cfg: &RoundCfg,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    let mut draw = super::VerifierDraw::new(cfg.sampler_cfg);
    log_request_record(
        &RoundTotals {
            loop_kind: cfg.loop_kind,
            rounds,
            charged: cfg.charged,
        },
        &emitted,
    );
    Ok(vec![])
}
RS
}

# One entry: the migrated drafter's path. It takes the request's sampler and
# hands it over inside the configuration it builds.
entry_src() {
  local name="$1"
  cat <<RS
pub fn $name(
    verifier: &Architecture,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    device: Device,
) -> Result<Vec<ProbeStep>> {
    super::round_loop_generate(
        verifier,
        &RoundCfg {
            loop_kind: SpecLoop::Kind,
            sampler_cfg,
            charged: false,
        },
        step_fn,
        device,
    )
}
RS
}

write_dispatch() {
  local root="$1"
  mkdir -p "$root/crates/rmlx-server/src/engine"
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
            n_tokens,
            &mut step_fn,
            &spec_sampler_cfg,
            dispatcher.device(),
        ),
    };
}
RS
}

# A sibling test file, which the scan must not read: the same shapes appear in
# tests deliberately and are not production paths.
write_test_sibling() {
  local root="$1"
  cat >"$root/crates/rmlx-models/src/speculative/mtp_tests.rs" <<'RS'
pub fn mtp_generate_harness(
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
) -> Result<Vec<ProbeStep>> {
    log_request_record(&RoundTotals { rounds, charged: false }, &emitted);
    Ok(vec![])
}
RS
}

# Today's tree: seven paths, every one of them a loop body.
build_root() {
  local root="$1"
  rm -rf "$root"
  mkdir -p "$root/crates/rmlx-models/src/speculative/dflash2"

  {
    printf '//! The entry guard and the two two-model loops.\n\n'
    guard_src
    printf '\n'
    greedy_cached_src
    printf '\n'
    stochastic_cached_src
    printf '\nfn summarise(rounds: usize) -> usize {\n    rounds\n}\n'
  } >"$root/crates/rmlx-models/src/speculative/mod.rs"

  loop_src "mtp_generate" >"$root/crates/rmlx-models/src/speculative/mtp.rs"
  loop_src "mtp_assistant_generate" \
    >"$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
  loop_src "dflash_generate" >"$root/crates/rmlx-models/src/speculative/dflash.rs"
  loop_src "dflash2_generate" >"$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
  loop_src "eagle3_generate" >"$root/crates/rmlx-models/src/speculative/eagle3.rs"

  write_test_sibling "$root"
  write_dispatch "$root"
}

# Mid-campaign: the MTP sidecar has been migrated. Its loop body is gone, its
# path is an entry, and the shared loop runs its rounds — seven paths still.
build_mid_root() {
  local root="$1"
  build_root "$root"
  entry_src "mtp_generate" >"$root/crates/rmlx-models/src/speculative/mtp.rs"
  {
    printf '//! The one round loop.\n\n'
    forwarded_loop_src
  } >"$root/crates/rmlx-models/src/speculative/round_loop.rs"
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

# run_script <script> <name> <expected-exit> <reason-substring> — the same, for
# a case whose edit is to the gate itself rather than to the tree.
run_script() {
  local prog="$1" name="$2" want_exit="$3" want_reason="$4"
  cases=$((cases + 1))
  local out rc
  out="$(SPEC_SAMPLING_ROOT="$work/root" bash "$prog" 2>&1)"
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
run "a clean root passes, and the private loops are in the population" 0 \
  "OK: 7 loops (0 forwarded), 0 entries, 1 guards"

# 1. A loop that takes no sampler at all — the state every sidecar was in.
build_root "$root"
perl -0pi -e 's/    sampler_cfg: &crate::sampler::SamplerConfig,\n//; s/super::VerifierDraw::new\(sampler_cfg\)/super::VerifierDraw::new(\&greedy())/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a loop with no sampler parameter is refused" 1 \
  "\`mtp_generate\` drives a generation but takes neither a"

# 2. A loop that takes the sampler and builds no draw from it. The signature
#    satisfies a caller and the loop still decodes greedily.
build_root "$root"
perl -0pi -e 's/super::VerifierDraw::new\(sampler_cfg\)/super::VerifierDraw::new(\&greedy());\n    let _ = sampler_cfg/' \
  "$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
run "a loop that never draws with its sampler is refused" 1 \
  "\`dflash2_generate\` is handed the request's sampler and builds"

# 3. One dispatch arm dropping the configuration while the others keep it.
build_root "$root"
perl -0pi -e 's/            &mut step_fn,\n            &spec_sampler_cfg,\n            dispatcher.device\(\),\n        \),\n    \};/            &mut step_fn,\n            dispatcher.device(),\n        ),\n    };/' \
  "$root/crates/rmlx-server/src/engine/speculative.rs"
run "one dispatch arm dropping the sampler is refused" 1 \
  "the \`Drafter::DFlash2\` arm drives a"

# 4. A driver added beside the seven. The path census is what sees it: a tree
#    with eight generation paths is not the tree this gate was pointed at, and
#    that is a stronger statement than the sampler rule it would also fail.
build_root "$root"
loop_src "eagle9_generate" >>"$root/crates/rmlx-models/src/speculative/mtp.rs"
run "an eighth generation path is a scan error" 2 \
  "the tree has 8 drafter generation paths and records"

# 5. The scan finding nothing must not pass. This is the failure a renamed
#    argument would produce, and it is the one a gate reports as clean.
build_root "$root"
perl -0pi -e 's/step_fn: &mut dyn FnMut\(&ProbeStep\)/emit: \&mut dyn FnMut(\&Step)/g' \
  "$root/crates/rmlx-models/src/speculative"/*.rs \
  "$root/crates/rmlx-models/src/speculative/dflash2"/*.rs
run "a scan that matches no loop is a scan error, not a pass" 2 \
  "found no round loop"

# 6. A dispatch this gate cannot find is also a scan error.
build_root "$root"
perl -0pi -e 's/let result = match &drafter \{/let result = match drafter_kind {/' \
  "$root/crates/rmlx-server/src/engine/speculative.rs"
run "a dispatch the gate cannot read is a scan error, not a pass" 2 \
  "Rule 4 scanned nothing"

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
#    builds a `RoundTotals`, so a scan that read it would report eight paths.
build_root "$root"
run "a loop-shaped fn in a sibling test file is not scanned" 0 \
  "OK: 7 loops (0 forwarded), 0 entries, 1 guards"

# 10. The guard's own defect class: one route that drops the configuration while
#     the other honours it. This is the fn that decides which loop a sampled
#     request reaches, and it was in the old gate only because it is `pub`.
build_root "$root"
perl -0pi -e 's/spec_generate_stochastic_cached\(verifier, step_fn, sampler_cfg, device\)/spec_generate_stochastic_cached(verifier, step_fn, device)/' \
  "$root/crates/rmlx-models/src/speculative/mod.rs"
run "a guard dropping the sampler on one route is refused" 1 \
  "\`spec_generate_greedy\` runs \`spec_generate_stochastic_cached\` without passing the"

# 11. The exemption is by name and by nothing else: the same body under any
#     other name is a loop that takes no sampler.
build_root "$root"
perl -0pi -e 's/spec_generate_greedy_cached/spec_generate_plain_cached/g' \
  "$root/crates/rmlx-models/src/speculative/mod.rs"
run "the exempt loop renamed is no longer exempt" 1 \
  "\`spec_generate_plain_cached\` drives a generation but takes neither a"

# 12. And the exemption is load-bearing rather than decorative: struck out of a
#     copy of the gate, the clean tree is refused, naming the one loop it covers.
build_root "$root"
sed 's/^readonly EXEMPT_LOOP=.*/readonly EXEMPT_LOOP=""/' "$script" >"$work/no_exempt.sh"
run_script "$work/no_exempt.sh" "the recorded exemption is what passes the two-model greedy loop" 1 \
  "\`spec_generate_greedy_cached\` drives a generation but takes neither a"

# 13. The mid-campaign tree: one path is an entry now, and the loop it runs is
#     handed the sampler inside the configuration the entry built.
build_mid_root "$root"
run "the mid-campaign tree passes, and names what it passed as" 0 \
  "OK: 7 loops (1 forwarded), 1 entries, 1 guards"

# 14. The shape the entries exist to refuse: a sampler an entry accepts and
#     leaves out of the configuration it hands over. Nothing else can see it —
#     the signature is right, the name appears in the body, and the loop draws
#     from whatever the configuration carried.
build_mid_root "$root"
perl -0pi -e 's/            sampler_cfg,\n//; s/\) -> Result<Vec<ProbeStep>> \{\n    super::round_loop_generate\(/) -> Result<Vec<ProbeStep>> {\n    let _ = sampler_cfg;\n    super::round_loop_generate(/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "an entry that leaves the sampler out of the configuration is refused" 1 \
  "\`mtp_generate\` takes \`sampler_cfg\` and leaves it out of the"

# 15. An entry that never took one. The request's temperature stops there.
build_mid_root "$root"
perl -0pi -e 's/    sampler_cfg: &crate::sampler::SamplerConfig,\n//; s/            sampler_cfg,\n//' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "an entry with no sampler parameter is refused" 1 \
  "\`mtp_generate\` starts a round loop and takes no"

# 16. The shared loop reading the sampler out of the configuration is the read
#     RULE 1 asks for; the same loop drawing from a greedy default is not.
build_mid_root "$root"
perl -0pi -e 's/super::VerifierDraw::new\(cfg\.sampler_cfg\)/super::VerifierDraw::new(\&greedy());\n    let _ = cfg.sampler_cfg/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "the shared loop that draws from no sampler is refused" 1 \
  "\`round_loop_generate\` is handed the request's sampler and builds"

# 17. A lost entry. The seven paths are what the campaign moves between shapes,
#     never the count, so six of them is a scan that stopped looking rather than
#     a tree with one drafter fewer.
build_mid_root "$root"
printf '//! The path that used to be here.\n' \
  >"$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a lost entry is a scan error, not a quieter run" 2 \
  "the tree has 6 drafter generation paths and records"

echo
if [ "$failures" != "0" ]; then
  echo "check-spec-sampling-fixtures: $failures of $cases cases failed"
  exit 1
fi
echo "OK: $cases cases, every rule fired for its own reason."
