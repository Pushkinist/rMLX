#!/usr/bin/env bash
# scripts/check_spec_sampling.sh — CI gate: every speculative generation path is
# handed the request's sampling configuration and draws with it, and every
# drafter arm passes it.
#
# WHY
#   A round loop that takes no sampler decodes greedily. It does not fail, it
#   does not warn, and its output is fluent — so a request that asked for a
#   temperature gets an answer that looks exactly like the one it wanted and is
#   drawn from a different distribution. Every published benchmark protocol for
#   this class of model names a temperature, so a harness that sets one and
#   serves through such a loop records a greedy number under a sampled label.
#   That is what this gate exists to stop happening again: the loops were all
#   greedy at once, the drafter arms all dropped the configuration at one seam,
#   and nothing anywhere said so.
#
#   The distributional gate (`spec_sampled_distribution`) is what proves a loop
#   samples *correctly*. It needs two model snapshots and a Metal context, so it
#   runs one pair under `make gpu-test` and stands down elsewhere. This gate is
#   the part that runs everywhere and covers every path: it cannot tell a right
#   distribution from a wrong one, and it can tell that a path was never given
#   the chance to draw from either.
#
# THE POPULATION
#   Not "a `pub fn` driver". That rule found six fns while the two private
#   two-model loops sat outside the gate entirely, and it would have kept
#   finding the entries after the collapse while the one shared loop — which is
#   `pub(crate)` — stayed outside it. Widening to any visibility alone is not
#   free either: it enumerates the three emit helpers, which drive no
#   generation and take no sampler.
#
#   So the population is `check_spec_charge.sh`'s, at any visibility, plus the
#   fns that call one of them:
#
#     LOOPS   — the driver signature `step_fn: &mut dyn FnMut(&ProbeStep)` plus
#               a `RoundTotals`.
#     ENTRIES — the driver signature, no totals, no `rollback_round`, and it
#               constructs the loop's `RoundCfg`. All four conjuncts, so the
#               population is the charge gate's and not a near copy of it.
#     GUARDS  — a fn that calls a loop or an entry and is neither. There is one:
#               the two-model entry guard, which routes a request to one of two
#               loops by reading whether the sampler is active. That clause is
#               not tidiness — the guard is where a sampled request can be
#               routed to the greedy arm, which is this gate's own defect class,
#               and it was in the gate before only because it happens to be
#               `pub`.
#
#   Narrowing to the loops alone is the tempting move and it is wrong: at the
#   end of the campaign the entries are the only place a request's sampler can
#   be dropped, since each takes it and hands it to the loop's configuration,
#   and a gate that stops reading them loses its own defect class at the one
#   site that can still commit it.
#
# RULE 1 (a loop is handed the sampler and draws with it)
#   Every loop must
#     (a) declare a `sampler_cfg: &...SamplerConfig` parameter, or take the
#         `RoundCfg` that carries the request's sampler for it, and
#     (b) construct the draw it uses from that sampler — a
#         `VerifierDraw::new(...)` naming `sampler_cfg`, which is why the
#         configuration's field carrying it is named `sampler_cfg` too, so one
#         needle reads `VerifierDraw::new(sampler_cfg)` and
#         `VerifierDraw::new(cfg.sampler_cfg)` alike.
#   (b) is not redundant, and it is deliberately the construction rather than a
#   mention: a parameter added to satisfy (a) and then ignored is the same
#   defect with the signature repaired, and a sampler assigned into a context
#   field and read by nobody is the same defect with the body repaired.
#
#   One loop draws another way and is read by a second needle rather than
#   waived: `spec_generate_stochastic_cached` is the one acceptance rule that is
#   not the shared one — it seeds its whole draw stream with
#   `Pcg32::new(sampler_cfg.seed_or_default())` and scores each position's own
#   post-sampling distribution. That is still a draw constructed from the
#   request's sampler, so it is a needle any loop may satisfy and not a name
#   this gate exempts.
#
# RULE 2 (an entry hands the sampler to the loop)
#   An entry declares the parameter and the configuration it builds carries it:
#   the needle is `sampler_cfg` inside the `RoundCfg { ... }` literal, not the
#   name appearing somewhere in the body. A `sampler_cfg` an entry accepts and
#   leaves out of the configuration it hands over is exactly the shape this rule
#   exists to refuse.
#
# RULE 3 (a guard passes it on)
#   A guard declares the parameter and passes it at every call it makes to a
#   loop or an entry that takes one.
#
# RULE 4 (the dispatch)
#   In the server's speculative generator, every arm of the `match &drafter`
#   that drives a generation must pass the resolved sampler configuration. One
#   arm that does not is one drafter kind that decodes greedily, and the other
#   arms passing it is what makes that invisible in review.
#
# THE ONE EXCEPTION, RECORDED
#   `spec_generate_greedy_cached` takes no sampler at all. It is the two-model
#   greedy loop, it runs only at temperature 0 where the verifier's argmax is
#   the draw, and the guard above it routes every sampled request to the
#   stochastic loop instead. So it is exempt from RULE 1, and no caller is
#   asked to pass it a sampler it does not take. The exception is deleted when
#   that loop becomes a drafter whose `verify` draws through the round's
#   context, which is what makes it no longer true. It is the only name in this
#   file.
#
# THE CENSUS
#   The success line names the three populations, and the count of drafter
#   paths — the loops that are not the shared forwarded one, plus the entries —
#   is pinned at seven. That figure is invariant across the campaign by
#   construction: a migrated drafter's loop body becomes its entry, one for one.
#   A scan that finds six paths where the tree has seven has not passed, it has
#   stopped looking, so a count that moves is exit 2 rather than a quieter run.
#
# EXIT
#   0 clean, 1 a rule fired, 2 the gate could not scan — a missing file, no
#   loop found, a body it could not read back, or a drafter-path count that
#   moved. A scan that finds nothing must not pass.

set -uo pipefail

root="${SPEC_SAMPLING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
loops_dir="$root/crates/rmlx-models/src/speculative"
dispatch="$root/crates/rmlx-server/src/engine/speculative.rs"

# The one fn that drives a generation and takes no sampler, and the reason it
# may — see THE ONE EXCEPTION above. It is a constant and not a knob: a gate
# whose exemptions can be named from the environment is a gate that exempts
# whatever the caller wants. The recall test proves it is load-bearing by
# running a copy of this script with the name struck out.
readonly EXEMPT_LOOP="spec_generate_greedy_cached"

readonly WANT_PATHS=7

fail=0
scan_error=0

note() { printf '%s\n' "$*" >&2; }

if [ ! -d "$loops_dir" ]; then
  note "check-spec-sampling: no speculative source directory at ${loops_dir#"$root"/}"
  exit 2
fi
if [ ! -f "$dispatch" ]; then
  note "check-spec-sampling: no speculative dispatch at ${dispatch#"$root"/}"
  exit 2
fi

# The two text readers, shared with the other source-scanning gates. Every
# needle below reads a line's code with the body of its string literals blanked:
# a commented-out draw beside a greedy one, and a needle inside a literal a
# program merely prints, are both a loop that decodes greedily and a scan that
# says it does not.
# shellcheck source=scripts/lib/awk_text.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/awk_text.sh"

# ---- Pass 1: the populations, and what each member does with the sampler ----
#
# One record per fn: file, name, driver signature, `RoundTotals`, a `RoundCfg`
# parameter, a `RoundCfg` it builds, a `sampler_cfg` parameter, a draw built
# from the sampler, a configuration carrying it, an unreadable body, a
# `rollback_round` call. A body this scan cannot read back is reported rather
# than skipped. A `fn` inside a `trait` or an `impl` block is read exactly like
# a free one, and a `fn` with no body at all — what a trait declares — is closed
# at its own `;` rather than at the next function's opening brace.

records=$(
  find "$loops_dir" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
    xargs -0 awk "$AWK_TEXT_FNS"'
      function reset() {
        in_sig = 0; awaiting_body = 0; in_body = 0; fname = ""
        depth = 0; paren = 0; cfg_depth = 0; draw_depth = 0
        has_step = 0; has_stats = 0; has_cfg = 0; makes_cfg = 0
        has_param = 0; draws = 0; cfg_carries = 0; unreadable = 0; has_roll = 0
      }
      function flush() {
        if (fname != "") {
          printf "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", \
            FILENAME, fname, has_step, has_stats, has_cfg, makes_cfg, \
            has_param, draws, cfg_carries, unreadable, has_roll
        }
        reset()
      }
      FNR == 1 { flush() }

      !in_body && !in_sig && !awaiting_body &&
      /^[[:space:]]*(pub(\([a-z:]+\))?[[:space:]]+)?(async[[:space:]]+)?fn[[:space:]]+[a-z_0-9]+/ {
        flush()
        line = $0
        sub(/^.*fn[[:space:]]+/, "", line)
        sub(/[^A-Za-z0-9_].*$/, "", line)
        fname = line
        in_sig = 1
        paren = 0
      }

      in_sig {
        # A commented-out parameter is not a parameter, and a parenthesis inside
        # a comment must not end the list.
        sig = decomment($0)
        if (index(sig, "step_fn: &mut dyn FnMut(&ProbeStep)") > 0) { has_step = 1 }
        if (sig ~ /sampler_cfg:[[:space:]]*&/) { has_param = 1 }
        if (index(sig, "RoundCfg") > 0) { has_cfg = 1 }
        # A per-parameter `) -> ` is inside the list and must not end the scan,
        # which is how a `sampler_cfg` declared after `step_fn` went unseen.
        paren += gsub(/\(/, "(", sig) - gsub(/\)/, ")", sig)
        if (paren <= 0) {
          # A declaration with no body — a trait method — ends here. Waiting for
          # the next `{` would read the following function as this one`s body
          # and record neither, and a lost loop whose slot a new entry fills is
          # a census that does not move.
          if (sig_is_bodiless(sig)) { flush(); next }
          in_sig = 0
          awaiting_body = 1
          if (index(sig, "{") > 0) {
            awaiting_body = 0
            in_body = 1
            depth = gsub(/\{/, "{", sig) - gsub(/\}/, "}", sig)
            if (depth < 0) { depth = 0 }
          }
        }
        next
      }

      awaiting_body {
        head = decomment($0)
        # A `where` clause can carry the `;` that ends a bodiless declaration.
        if (sig_is_bodiless(head)) { flush(); next }
        if (index(head, "{") > 0) {
          awaiting_body = 0
          in_body = 1
          depth = gsub(/\{/, "{", head) - gsub(/\}/, "}", head)
          if (depth < 0) { depth = 0 }
        }
        next
      }

      !in_body { next }

      { code = blank_strings(decomment($0)) }

      index(code, "RoundTotals {") > 0 { has_stats = 1 }
      index(code, "rollback_round(") > 0 &&
      code !~ /fn[[:space:]]+rollback_round\(/ { has_roll = 1 }

      # The draw a loop uses, built from the request`s sampler. Two
      # constructions: the shared `VerifierDraw`, and the stochastic acceptance
      # rule`s own RNG stream seeded from the same configuration. Each is
      # followed to its closing parenthesis, so a wrapped argument list reads
      # like a single-line one — a gate that refused a correct loop for the
      # width of its line would be repaired by widening the line.
      draw_depth == 0 &&
      (index(code, "VerifierDraw::new(") > 0 || index(code, "Pcg32::new(") > 0) {
        rest = code
        if (index(code, "VerifierDraw::new(") > 0) {
          sub(/^.*VerifierDraw::new\(/, "", rest)
        } else {
          sub(/^.*Pcg32::new\(/, "", rest)
        }
        draw_depth = 1 + gsub(/\(/, "(", rest) - gsub(/\)/, ")", rest)
        if (index(rest, "sampler_cfg") > 0) { draws = 1 }
        if (draw_depth < 0) { draw_depth = 0 }
        depth += gsub(/\{/, "{", code) - gsub(/\}/, "}", code)
        if (depth <= 0) { flush() }
        next
      }
      draw_depth > 0 {
        if (index(code, "sampler_cfg") > 0) { draws = 1 }
        draw_depth += gsub(/\(/, "(", code) - gsub(/\)/, ")", code)
        if (draw_depth < 0) { draw_depth = 0 }
      }

      # The configuration an entry builds, and whether the sampler is in it. The
      # literal is followed to its closing brace, so a `sampler_cfg` further
      # down the body is not read as one the entry handed over.
      index(code, "RoundCfg {") > 0 {
        makes_cfg = 1
        rest = code
        sub(/^.*RoundCfg[[:space:]]*\{/, "", rest)
        cfg_depth = 1 + gsub(/\{/, "{", rest) - gsub(/\}/, "}", rest)
        if (index(rest, "sampler_cfg") > 0) { cfg_carries = 1 }
        if (cfg_depth <= 0) { cfg_depth = 0 }
        depth += gsub(/\{/, "{", code) - gsub(/\}/, "}", code)
        if (depth <= 0) { flush() }
        next
      }
      cfg_depth > 0 {
        if (index(code, "sampler_cfg") > 0) { cfg_carries = 1 }
        cfg_depth += gsub(/\{/, "{", code) - gsub(/\}/, "}", code)
        if (cfg_depth < 0) { cfg_depth = 0 }
      }

      {
        depth += gsub(/\{/, "{", code) - gsub(/\}/, "}", code)
        if (depth <= 0) { flush() }
      }
      END {
        if (in_sig || awaiting_body) { unreadable = 1 }
        flush()
      }
    '
)

loops=$(printf '%s\n' "$records" | awk -F'\t' '$3 == 1 && $4 == 1')
entries=$(printf '%s\n' "$records" | awk -F'\t' '$3 == 1 && $4 == 0 && $6 == 1 && $11 == 0')

if [ -z "$loops" ]; then
  note "check-spec-sampling: found no round loop under ${loops_dir#"$root"/}."
  note "  A loop is a fn taking \`step_fn: &mut dyn FnMut(&ProbeStep)\` that builds a"
  note "  \`RoundTotals\`. Either they were renamed out from under this gate, or the scan"
  note "  is broken; a gate that matched nothing must not report a pass."
  exit 2
fi

population=$(printf '%s\n%s\n' "$loops" "$entries" | awk -F'\t' '$2 != "" { print $2 }')
# BSD awk cannot take a newline inside a `-v` value, so the list crosses on
# spaces. Names are Rust identifiers; neither form can hold one.
population_flat=$(printf '%s\n' "$population" | tr '\n' ' ')

# ---- Pass 2: the guards -----------------------------------------------------
#
# A call to a member of the population, and whether the request's sampler goes
# with it. The exempt loop takes none, so a call to it is counted and not
# checked — otherwise the guard above it would be refused for honouring a
# signature this file already records.

callers=$(
  find "$loops_dir" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
    xargs -0 awk -v names="$population_flat" -v exempt="$EXEMPT_LOOP" "$AWK_TEXT_FNS"'
      BEGIN { n = split(names, list, " "); for (i = 1; i <= n; i++) { if (list[i] != "") { pop[list[i]] = 1 } } }
      function reset() {
        in_sig = 0; awaiting_body = 0; in_body = 0; fname = ""
        depth = 0; paren = 0; ncalls = 0; nmissing = 0; missing = "-"
        call_depth = 0; callee = ""; carries = 0
      }
      function close_call() {
        if (callee != exempt) {
          ncalls++
          if (!carries) { nmissing++; if (missing == "-") { missing = callee } }
        }
        call_depth = 0; callee = ""; carries = 0
      }
      function flush() {
        if (fname != "") {
          printf "%s\t%s\t%d\t%d\t%s\n", FILENAME, fname, ncalls, nmissing, missing
        }
        reset()
      }
      FNR == 1 { flush() }

      !in_body && !in_sig && !awaiting_body &&
      /^[[:space:]]*(pub(\([a-z:]+\))?[[:space:]]+)?(async[[:space:]]+)?fn[[:space:]]+[a-z_0-9]+/ {
        flush()
        line = $0
        sub(/^.*fn[[:space:]]+/, "", line)
        sub(/[^A-Za-z0-9_].*$/, "", line)
        fname = line
        in_sig = 1
        paren = 0
      }

      in_sig {
        sig = decomment($0)
        paren += gsub(/\(/, "(", sig) - gsub(/\)/, ")", sig)
        if (paren <= 0) {
          # A declaration with no body — a trait method — ends here. Waiting for
          # the next `{` would read the following function as this one`s body
          # and record neither, and a lost loop whose slot a new entry fills is
          # a census that does not move.
          if (sig_is_bodiless(sig)) { flush(); next }
          in_sig = 0
          awaiting_body = 1
          if (index(sig, "{") > 0) {
            awaiting_body = 0; in_body = 1
            depth = gsub(/\{/, "{", sig) - gsub(/\}/, "}", sig)
            if (depth < 0) { depth = 0 }
          }
        }
        next
      }
      awaiting_body {
        head = decomment($0)
        # A `where` clause can carry the `;` that ends a bodiless declaration.
        if (sig_is_bodiless(head)) { flush(); next }
        if (index(head, "{") > 0) {
          awaiting_body = 0; in_body = 1
          depth = gsub(/\{/, "{", head) - gsub(/\}/, "}", head)
          if (depth < 0) { depth = 0 }
        }
        next
      }
      !in_body { next }

      { code = blank_strings(decomment($0)) }

      call_depth > 0 {
        if (index(code, "sampler_cfg") > 0) { carries = 1 }
        call_depth += gsub(/\(/, "(", code) - gsub(/\)/, ")", code)
        if (call_depth <= 0) { close_call() }
        depth += gsub(/\{/, "{", code) - gsub(/\}/, "}", code)
        if (depth <= 0) { flush() }
        next
      }

      {
        line = code
        while (match(line, /[A-Za-z_][A-Za-z0-9_]*\(/)) {
          name = substr(line, RSTART, RLENGTH - 1)
          line = substr(line, RSTART + RLENGTH)
          if (!(name in pop) || name == fname) { continue }
          callee = name
          carries = (index(line, "sampler_cfg") > 0)
          call_depth = 1 + gsub(/\(/, "(", line) - gsub(/\)/, ")", line)
          if (call_depth <= 0) { close_call() }
          break
        }
      }

      {
        depth += gsub(/\{/, "{", code) - gsub(/\}/, "}", code)
        if (depth <= 0) { flush() }
      }
      END { flush() }
    '
)

# ---- Rule 1: the loops ------------------------------------------------------

loop_count=0
entry_count=0
path_count=0
forwarded_count=0
exempt_seen=0

while IFS=$'\t' read -r file name _sig _stats has_cfg _mk has_param draws _carries unreadable _roll; do
  [ -n "$name" ] || continue
  rel="${file#"$root"/}"
  loop_count=$((loop_count + 1))
  if [ "$has_cfg" = "1" ]; then
    forwarded_count=$((forwarded_count + 1))
  else
    path_count=$((path_count + 1))
  fi
  if [ "$unreadable" = "1" ]; then
    note "check-spec-sampling: $rel: \`$name\` opened a body this gate could not read back."
    note "  A loop whose extent cannot be determined is not scanned, and an unscanned"
    note "  loop is exactly the one that would decode greedily unnoticed."
    scan_error=1
    continue
  fi
  if [ "$name" = "$EXEMPT_LOOP" ]; then
    exempt_seen=$((exempt_seen + 1))
    continue
  fi
  if [ "$has_param" != "1" ] && [ "$has_cfg" != "1" ]; then
    note "check-spec-sampling: $rel: \`$name\` drives a generation but takes neither a"
    note "  \`sampler_cfg: &SamplerConfig\` nor the \`RoundCfg\` that carries one, so every"
    note "  request it serves decodes greedily whatever temperature the caller asked"
    note "  for, and says nothing."
    fail=1
    continue
  fi
  if [ "$draws" != "1" ]; then
    note "check-spec-sampling: $rel: \`$name\` is handed the request's sampler and builds"
    note "  no draw from it. The signature satisfies a caller and the loop still decodes"
    note "  greedily: the draw has to be constructed from the configuration, not the"
    note "  name mentioned somewhere in the body."
    fail=1
  fi
done <<<"$loops"

# ---- Rule 2: the entries ----------------------------------------------------

while IFS=$'\t' read -r file name _sig _stats _cfgp _mk has_param _draws carries unreadable _roll; do
  [ -n "$name" ] || continue
  rel="${file#"$root"/}"
  entry_count=$((entry_count + 1))
  path_count=$((path_count + 1))
  if [ "$unreadable" = "1" ]; then
    note "check-spec-sampling: $rel: \`$name\` opened a body this gate could not read back."
    scan_error=1
    continue
  fi
  if [ "$has_param" != "1" ]; then
    note "check-spec-sampling: $rel: \`$name\` starts a round loop and takes no"
    note "  \`sampler_cfg: &SamplerConfig\`, so the request's temperature stops here and"
    note "  the loop it runs draws from a distribution nobody asked for."
    fail=1
    continue
  fi
  if [ "$carries" != "1" ]; then
    note "check-spec-sampling: $rel: \`$name\` takes \`sampler_cfg\` and leaves it out of the"
    note "  configuration it hands the loop. An entry is the last place a request's"
    note "  sampler can be dropped, and a sampler that reaches no loop is the same"
    note "  greedy answer under a sampled label."
    fail=1
  fi
done <<<"$entries"

# ---- Rule 3: the guards -----------------------------------------------------

guard_count=0
while IFS=$'\t' read -r file name ncalls nmissing missing; do
  [ -n "$name" ] || continue
  [ "$ncalls" != "0" ] || continue
  printf '%s\n' "$population" | grep -qx -- "$name" && continue
  rel="${file#"$root"/}"
  guard_count=$((guard_count + 1))
  if [ "$nmissing" != "0" ]; then
    note "check-spec-sampling: $rel: \`$name\` runs \`$missing\` without passing the"
    note "  request's sampler. A guard routes a request to one generation path or"
    note "  another; one route that drops the configuration is one class of request"
    note "  answered greedily while every other route honours it."
    fail=1
  fi
done <<<"$callers"

# ---- Rule 4: the dispatch ---------------------------------------------------

arms=$(
  awk "$AWK_TEXT_FNS"'
    # This scan reads code like every other one here: an arm that keeps the
    # sampler in a comment, or names it inside a string a line merely prints,
    # passes it to nothing.
    { code = blank_strings(decomment($0)) }
    code ~ /let result = match &drafter \{/ { in_match = 1; depth = 1; next }
    in_match {
      n = gsub(/\{/, "{", code); m = gsub(/\}/, "}", code)
      if (arm != "" ) { body = body code "\n" }
      if (code ~ /^[[:space:]]*Drafter::[A-Za-z0-9_]+/) {
        if (arm != "") { printf "%s\t%d\n", arm, (index(body, "spec_sampler_cfg") > 0) }
        line = code
        sub(/^[[:space:]]*Drafter::/, "", line)
        sub(/[^A-Za-z0-9_].*$/, "", line)
        arm = line
        body = code "\n"
      }
      depth += n - m
      if (depth <= 0) {
        if (arm != "") { printf "%s\t%d\n", arm, (index(body, "spec_sampler_cfg") > 0) }
        in_match = 0; arm = ""; body = ""
      }
    }
  ' "$dispatch"
)

if [ -z "$arms" ]; then
  note "check-spec-sampling: ${dispatch#"$root"/} has no \`let result = match &drafter {\`"
  note "  dispatch this gate can read. Rule 4 scanned nothing."
  exit 2
fi

arm_count=0
while IFS=$'\t' read -r arm passes; do
  [ -n "$arm" ] || continue
  arm_count=$((arm_count + 1))
  if [ "$passes" != "1" ]; then
    note "check-spec-sampling: ${dispatch#"$root"/}: the \`Drafter::$arm\` arm drives a"
    note "  generation without passing \`spec_sampler_cfg\`, so that drafter kind alone"
    note "  decodes greedily while the others honour the request."
    fail=1
  fi
done <<<"$arms"

# An entry exists to start the shared loop, so entries without one are a loop
# the scan lost — and the path count cannot see it, because a lost loop's slot
# is filled by the entry that replaced it, one for one. That is exactly the
# shape a migration produces, and it is exactly the shape a scanner that read a
# bodiless declaration as the next function's signature produced silently.
if [ "$entry_count" != "0" ] && [ "$forwarded_count" = "0" ]; then
  note "check-spec-sampling: the scan found $entry_count entries and no forwarded loop."
  note "  An entry hands the request's sampler to the one loop that takes a \`RoundCfg\`,"
  note "  so entries with no loop to enter mean the loop was lost. The drafter-path count"
  note "  cannot see that: the lost loop's slot is the one the entry now fills."
  scan_error=1
fi

if [ "$scan_error" = "1" ]; then
  exit 2
fi
if [ "$path_count" != "$WANT_PATHS" ]; then
  note "check-spec-sampling: the tree has $path_count drafter generation paths and records"
  note "  $WANT_PATHS. A migrated drafter's loop body becomes its entry, one for one, so"
  note "  this figure does not move across the collapse. A scan that finds six paths"
  note "  where the tree has seven has not passed, it has stopped looking."
  exit 2
fi
if [ "$fail" = "1" ]; then
  exit 1
fi

read_paths=$((path_count - exempt_seen))
echo "OK: $loop_count loops ($forwarded_count forwarded), $entry_count entries, $guard_count guards — $read_paths of $path_count drafter paths take the request's sampler and draw with it (\`$EXEMPT_LOOP\` is the recorded exception); $arm_count drafter arms pass it."
