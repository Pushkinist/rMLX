#!/usr/bin/env bash
# scripts/check_spec_sampling.sh — CI gate: every speculative round loop is
# handed the request's sampling configuration, and every drafter arm passes it.
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
#   the part that runs everywhere and covers every loop: it cannot tell a right
#   distribution from a wrong one, and it can tell that a loop was never given
#   the chance to draw from either.
#
# RULE 1 (the loops)
#   In crates/rmlx-models/src/speculative/, a `pub fn` whose parameters include
#   `step_fn: &mut dyn FnMut(&ProbeStep)` is a generation driver — that argument
#   is what makes it one. Every such function must
#     (a) declare a `sampler_cfg: &...SamplerConfig` parameter, and
#     (b) mention `sampler_cfg` in its body.
#   (b) is not redundant: a parameter added to satisfy (a) and then ignored is
#   the same defect with the signature repaired.
#
# RULE 2 (the dispatch)
#   In the server's speculative generator, every arm of the `match &drafter`
#   that drives a generation must pass the resolved sampler configuration. One
#   arm that does not is one drafter kind that decodes greedily, and the other
#   arms passing it is what makes that invisible in review.
#
# EXIT
#   0 clean, 1 a rule fired, 2 the gate could not scan (missing file, no
#   drivers found — a scan that finds nothing must not pass).

set -uo pipefail

root="${SPEC_SAMPLING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
loops_dir="$root/crates/rmlx-models/src/speculative"
dispatch="$root/crates/rmlx-server/src/engine/speculative.rs"

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

# ---- Rule 1 ----------------------------------------------------------------
#
# awk walks each file once. A `pub fn` opens a candidate; its parameter list runs
# to the `) -> ` that closes it; the body runs to the line where the brace depth
# returns to zero. Output is one record per driver: file, name, has-param,
# uses-param.

drivers=$(
  find "$loops_dir" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
    xargs -0 awk '
      function flush() {
        if (name != "" && is_driver) {
          printf "%s\t%s\t%d\t%d\n", FILENAME, name, has_param, uses_param
        }
        name = ""; is_driver = 0; has_param = 0; uses_param = 0
        in_sig = 0; in_body = 0; depth = 0
      }
      FNR == 1 { flush() }
      # A new `pub fn` while one is open means the previous never had a body we
      # could read back — report it as unscannable rather than skipping it.
      /^[[:space:]]*pub fn [a-z_0-9]+\(/ {
        if (in_body) { printf "%s\t%s\t-1\t-1\n", FILENAME, name }
        flush()
        line = $0
        sub(/^[[:space:]]*pub fn /, "", line)
        sub(/\(.*$/, "", line)
        name = line
        in_sig = 1
        paren = 0
      }
      in_sig {
        if (index($0, "step_fn: &mut dyn FnMut(&ProbeStep)") > 0) { is_driver = 1 }
        if ($0 ~ /sampler_cfg: &/) { has_param = 1 }
        # The parameter list closes when its own parenthesis balances. A
        # per-parameter `) -> ` — every `FnMut(..) -> ..` argument has one — is
        # inside it and must not end the scan, which is how a `sampler_cfg`
        # declared after `step_fn` went unseen.
        paren += gsub(/\(/, "(") - gsub(/\)/, ")")
        if (paren <= 0) {
          in_sig = 0
          in_body = 1
          # The line that closes the parameter list also opens the body, so its
          # own braces start the depth count. Starting from zero instead ends
          # the body on its first `}`.
          depth = gsub(/\{/, "{") - gsub(/\}/, "}")
        }
        next
      }
      in_body {
        if (index($0, "sampler_cfg") > 0) { uses_param = 1 }
        n = gsub(/\{/, "{"); m = gsub(/\}/, "}")
        depth += n - m
        if (depth <= 0 && (n > 0 || m > 0)) { flush() }
        next
      }
      END { flush() }
    '
)

if [ -z "$drivers" ]; then
  note "check-spec-sampling: found no generation drivers under ${loops_dir#"$root"/}."
  note "  A driver is a \`pub fn\` taking \`step_fn: &mut dyn FnMut(&ProbeStep)\`. Either"
  note "  they were renamed out from under this gate, or the scan is broken; a gate that"
  note "  matched nothing must not report a pass."
  exit 2
fi

driver_count=0
while IFS=$'\t' read -r file name has uses; do
  [ -n "$name" ] || continue
  rel="${file#"$root"/}"
  driver_count=$((driver_count + 1))
  if [ "$has" = "-1" ]; then
    note "check-spec-sampling: $rel: \`$name\` opened a body this gate could not read back."
    note "  A driver whose extent cannot be determined is not scanned, and an unscanned"
    note "  driver is exactly the one that would decode greedily unnoticed."
    scan_error=1
    continue
  fi
  if [ "$has" != "1" ]; then
    note "check-spec-sampling: $rel: \`$name\` drives a generation but takes no"
    note "  \`sampler_cfg: &SamplerConfig\`, so every request it serves decodes greedily"
    note "  whatever temperature the caller asked for, and says nothing."
    fail=1
    continue
  fi
  if [ "$uses" != "1" ]; then
    note "check-spec-sampling: $rel: \`$name\` takes \`sampler_cfg\` and never reads it."
    note "  The signature satisfies a caller and the loop still decodes greedily."
    fail=1
  fi
done <<<"$drivers"

# ---- Rule 2 ----------------------------------------------------------------

arms=$(
  awk '
    /let result = match &drafter \{/ { in_match = 1; depth = 1; next }
    in_match {
      n = gsub(/\{/, "{"); m = gsub(/\}/, "}")
      if (arm != "" ) { body = body $0 "\n" }
      if ($0 ~ /^[[:space:]]*Drafter::[A-Za-z0-9_]+/) {
        if (arm != "") { printf "%s\t%d\n", arm, (index(body, "spec_sampler_cfg") > 0) }
        line = $0
        sub(/^[[:space:]]*Drafter::/, "", line)
        sub(/[^A-Za-z0-9_].*$/, "", line)
        arm = line
        body = $0 "\n"
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
  note "  dispatch this gate can read. Rule 2 scanned nothing."
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

if [ "$scan_error" = "1" ]; then
  exit 2
fi
if [ "$fail" = "1" ]; then
  exit 1
fi

echo "OK: $driver_count speculative generation drivers take and read the request's sampler; $arm_count drafter arms pass it."
