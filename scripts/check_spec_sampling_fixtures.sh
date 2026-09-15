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
# seven drafter paths the tree does. Four roots are built: the all-loops shape
# the campaign started from, the mid-campaign one where a drafter has been
# migrated and its path is an entry that hands the sampler to the shared loop,
# the one where the two-model greedy path is an entry beside a loop, and the end
# state the tree now has — one shared loop and seven entries.

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
        spec_generate_greedy_cached(verifier, step_fn, sampler_cfg, device)
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

# The all-loops shape the campaign started from: seven paths, every one a body.
build_root() {
  local root="$1"
  rm -rf "$root"
  mkdir -p "$root/crates/rmlx-models/src/speculative/dflash2"

  {
    printf '//! The entry guard and the two two-model loops.\n\n'
    guard_src
    printf '\n'
    loop_src "spec_generate_greedy_cached"
    printf '\n'
    loop_src "spec_generate_stochastic_cached"
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

# A trait declaring what each drafter implements: bodiless method declarations,
# one default body among them, and a `where` clause on one of them. A `fn` item
# with no body is where a scanner that waits for the next `{` reads the
# following function as this one's body and records neither — and here the one
# it records neither of is the loop that draws.
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

# Two entries: the two-model greedy path beside a migrated sidecar, with the
# other four drafters still loop bodies. Not the tree's census — that is
# `build_tree_root` below — but the smallest root in which the greedy path wears
# the shape it wears in the tree, which is what the two readings of it need: it
# was the one name the gate exempted, and holding it to a loop's rules would be
# holding it to a shape it no longer has.
build_entry_root() {
  local root="$1"
  build_mid_root "$root"
  {
    printf '//! The entry guard, the two-model greedy entry and the stochastic loop.\n\n'
    guard_src
    printf '\n'
    entry_src "spec_generate_greedy_cached"
    printf '\n'
    loop_src "spec_generate_stochastic_cached"
    printf '\nfn summarise(rounds: usize) -> usize {\n    rounds\n}\n'
  } >"$root/crates/rmlx-models/src/speculative/mod.rs"
}

# The tree's own census, which is the end state: one shared loop and seven
# entries, the stochastic acceptance rule among them — it is a rule a drafter
# owns now, not a loop. The charge gate's fixtures carry this shape; without it
# here, every case of this suite would be taken against a tree the gate never
# actually scans.
build_tree_root() {
  local root="$1"
  build_entry_root "$root"
  {
    printf '//! The entry guard and the two two-model entries.\n\n'
    guard_src
    printf '\n'
    entry_src "spec_generate_greedy_cached"
    printf '\n'
    entry_src "spec_generate_stochastic_cached"
    printf '\nfn summarise(rounds: usize) -> usize {\n    rounds\n}\n'
  } >"$root/crates/rmlx-models/src/speculative/mod.rs"
  entry_src "mtp_assistant_generate" \
    >"$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
  entry_src "dflash_generate" >"$root/crates/rmlx-models/src/speculative/dflash.rs"
  entry_src "dflash2_generate" >"$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
  entry_src "eagle3_generate" >"$root/crates/rmlx-models/src/speculative/eagle3.rs"
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

# 11. The shape the tree has: the path the gate used to exempt by name is an
#     entry, and the clean root that carries it passes and says so.
build_entry_root "$root"
run "the two-model greedy path as an entry passes, and names what it passed as" 0 \
  "OK: 6 loops (1 forwarded), 2 entries, 1 guards"

# 12. That path with no sampler parameter. This is exactly what the exemption
#     waived for as long as the path was a loop that took none, and nothing waives
#     it now: the gate reads no name anywhere.
build_entry_root "$root"
perl -0pi -e 's/(fn spec_generate_greedy_cached\(.*?)    sampler_cfg: &crate::sampler::SamplerConfig,\n/$1/s; s/(RoundCfg \{.*?)            sampler_cfg,\n/$1/s' \
  "$root/crates/rmlx-models/src/speculative/mod.rs"
run "the two-model greedy path buys nothing from its name" 1 \
  "\`spec_generate_greedy_cached\` starts a round loop and takes no"

# 13. And that path taking the sampler and leaving it out of the configuration it
#     hands over — the request's temperature stops at the entry, with the
#     signature and the guard above it both reading correctly.
build_entry_root "$root"
perl -0pi -e 's/(fn spec_generate_greedy_cached\(.*?RoundCfg \{.*?)            sampler_cfg,\n/$1/s' \
  "$root/crates/rmlx-models/src/speculative/mod.rs"
run "the two-model greedy entry that drops the sampler is refused" 1 \
  "\`spec_generate_greedy_cached\` takes \`sampler_cfg\` and leaves it out of the"

# 14. The tree's own census: one shared loop and seven entries, which is the end
#     state the collapse reaches. This is the shape every case above is evidence
#     about, and the suite has to be able to state it.
build_tree_root "$root"
run "the tree's own census passes, and names what it passed as" 0 \
  "OK: 1 loops (1 forwarded), 7 entries, 1 guards"

# 15. The mid-campaign tree: one path is an entry now, and the loop it runs is
#     handed the sampler inside the configuration the entry built.
build_mid_root "$root"
run "the mid-campaign tree passes, and names what it passed as" 0 \
  "OK: 7 loops (1 forwarded), 1 entries, 1 guards"

# 16. The shape the entries exist to refuse: a sampler an entry accepts and
#     leaves out of the configuration it hands over. Nothing else can see it —
#     the signature is right, the name appears in the body, and the loop draws
#     from whatever the configuration carried.
build_mid_root "$root"
perl -0pi -e 's/            sampler_cfg,\n//; s/\) -> Result<Vec<ProbeStep>> \{\n    super::round_loop_generate\(/) -> Result<Vec<ProbeStep>> {\n    let _ = sampler_cfg;\n    super::round_loop_generate(/' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "an entry that leaves the sampler out of the configuration is refused" 1 \
  "\`mtp_generate\` takes \`sampler_cfg\` and leaves it out of the"

# 17. An entry that never took one. The request's temperature stops there.
build_mid_root "$root"
perl -0pi -e 's/    sampler_cfg: &crate::sampler::SamplerConfig,\n//; s/            sampler_cfg,\n//' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "an entry with no sampler parameter is refused" 1 \
  "\`mtp_generate\` starts a round loop and takes no"

# 18. The shared loop reading the sampler out of the configuration is the read
#     RULE 1 asks for; the same loop drawing from a greedy default is not.
build_mid_root "$root"
perl -0pi -e 's/super::VerifierDraw::new\(cfg\.sampler_cfg\)/super::VerifierDraw::new(\&greedy());\n    let _ = cfg.sampler_cfg/' \
  "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "the shared loop that draws from no sampler is refused" 1 \
  "\`round_loop_generate\` is handed the request's sampler and builds"

# 19. A lost entry. The seven paths are what the campaign moves between shapes,
#     never the count, so six of them is a scan that stopped looking rather than
#     a tree with one drafter fewer.
build_mid_root "$root"
printf '//! The path that used to be here.\n' \
  >"$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a lost entry is a scan error, not a quieter run" 2 \
  "the tree has 6 drafter generation paths and records"

# 20. A needle in a comment is not the call it names. The commented-out draw is
#     what a change like this leaves behind, and beside a greedy one it is a
#     loop that ignores the request while reading as one that honours it.
build_root "$root"
perl -0pi -e 's|    let mut draw = super::VerifierDraw::new\(sampler_cfg\);|    // let mut draw = super::VerifierDraw::new(sampler_cfg);\n    let mut draw = super::VerifierDraw::new(\&greedy());|' \
  "$root/crates/rmlx-models/src/speculative/mtp.rs"
run "a commented-out draw is not a draw" 1 \
  "\`mtp_generate\` is handed the request's sampler and builds"

# 21. The same on the signature side: a commented-out parameter is not one, and
#     a scan that read it would find a sampler the fn never takes.
build_root "$root"
perl -0pi -e 's|    sampler_cfg: &crate::sampler::SamplerConfig,|    // sampler_cfg: \&crate::sampler::SamplerConfig,|; s|super::VerifierDraw::new\(sampler_cfg\)|super::VerifierDraw::new(\&greedy())|' \
  "$root/crates/rmlx-models/src/speculative/gemma4_assistant.rs"
run "a commented-out sampler parameter is not a parameter" 1 \
  "\`mtp_assistant_generate\` drives a generation but takes neither a"

# 22. And a needle inside a string literal is text a program prints, not a
#     construction it makes.
build_root "$root"
perl -0pi -e 's|    let mut draw = super::VerifierDraw::new\(sampler_cfg\);|    let note = "VerifierDraw::new(sampler_cfg)";\n    let mut draw = super::VerifierDraw::new(\&greedy());|' \
  "$root/crates/rmlx-models/src/speculative/dflash.rs"
run "a needle inside a string literal is not a draw" 1 \
  "\`dflash_generate\` is handed the request's sampler and builds"

# 23. The converse, and the reason the construction is followed to its closing
#     parenthesis: a correct loop must not be refused for the width of its line.
build_root "$root"
perl -0pi -e 's|    let mut draw = super::VerifierDraw::new\(sampler_cfg\);|    let mut draw = super::VerifierDraw::new(\n        sampler_cfg,\n    );|' \
  "$root/crates/rmlx-models/src/speculative/eagle3.rs"
run "a draw wrapped over two lines is still a draw" 0 \
  "OK: 7 loops (0 forwarded), 0 entries, 1 guards"

# 24. Every needle reads code, the parameter list included: a comment naming a
#     type or a marker in a signature is prose, and reading it would move the
#     population.
build_root "$root"
perl -0pi -e 's|    device: Device,\n\) -> Result<Vec<ProbeStep>> \{|    device: Device, // not a RoundCfg, and no step_fn: \&mut dyn FnMut(\&ProbeStep) here\n) -> Result<Vec<ProbeStep>> {|' \
  "$root/crates/rmlx-models/src/speculative/dflash2/round.rs"
run "a comment in a parameter list is prose, not a declaration" 0 \
  "OK: 7 loops (0 forwarded), 0 entries, 1 guards"

# 25. The dispatch scan reads code too, and it is the arm reading that is this
#     gate's own defect class: a sampler kept as a comment is passed to nothing.
build_root "$root"
perl -0pi -e 's|            &spec_sampler_cfg,\n            dispatcher.device\(\),\n        \),\n    \};|            // \&spec_sampler_cfg,\n            dispatcher.device(),\n        ),\n    };|' \
  "$root/crates/rmlx-server/src/engine/speculative.rs"
run "a dispatch arm keeping the sampler in a comment is refused" 1 \
  "the \`Drafter::DFlash2\` arm drives a"

# 26. And the same through a string literal, which is text the arm prints rather
#     than a configuration it passes.
build_root "$root"
perl -0pi -e 's|            &spec_sampler_cfg,\n            dispatcher.device\(\),\n        \),\n    \};|            tracing::debug!("no spec_sampler_cfg here"),\n            dispatcher.device(),\n        ),\n    };|' \
  "$root/crates/rmlx-server/src/engine/speculative.rs"
run "a dispatch arm naming the sampler inside a string literal is refused" 1 \
  "the \`Drafter::DFlash2\` arm drives a"

# 27. The same trait, and here the loop it swallows is the one that draws. The
#     path census cannot see the loss on its own: the entry that replaced the
#     migrated drafter fills the slot, one for one, so a scan that lost the
#     shared loop still counts seven paths and passes.
build_mid_root "$root"
loop_file="$root/crates/rmlx-models/src/speculative/round_loop.rs"
{
  head -2 "$loop_file"
  drafter_trait_src
  tail -n +3 "$loop_file"
} >"$work/with_trait.rs"
mv "$work/with_trait.rs" "$loop_file"
run "a trait of bodiless declarations does not swallow the loop beneath it" 0 \
  "OK: 7 loops (1 forwarded), 1 entries, 1 guards"

# 28. And the invariant that would have caught it whatever the cause: an entry
#     hands the sampler to a loop that takes a `RoundCfg`, so entries with none
#     to enter are a loop the scan lost.
build_mid_root "$root"
rm -f "$root/crates/rmlx-models/src/speculative/round_loop.rs"
run "an entry with no forwarded loop to enter is a scan error" 2 \
  "the scan found 1 entries and no forwarded loop"

# 29. The shape the second needle used to admit, and the reason it is gone: a
#     loop that seeds its own generator from the request's seed draws from a
#     stream nothing else in the request shares. It reads as reproducible on its
#     own — one seed, one sequence — while being a second stream correlated with
#     the one the round's draw advances, and only one of them is the stream a
#     reproducibility pin describes.
build_root "$root"
perl -0pi -e 's|    let mut draw = super::VerifierDraw::new\(sampler_cfg\);|    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());|' \
  "$root/crates/rmlx-models/src/speculative/mod.rs"
run "a loop that seeds a second generator instead of drawing is refused" 1 \
  "\`spec_generate_greedy_cached\` is handed the request's sampler and builds"

echo
if [ "$failures" != "0" ]; then
  echo "check-spec-sampling-fixtures: $failures of $cases cases failed"
  exit 1
fi
echo "OK: $cases cases, every rule fired for its own reason."
