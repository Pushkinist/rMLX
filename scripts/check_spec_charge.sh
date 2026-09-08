#!/usr/bin/env bash
# scripts/check_spec_charge.sh — CI gate: each speculative round loop names one
# phase-charge decision, and it names it everywhere the decision is read.
#
# WHY
#   A round loop decides once whether its phases are charged for the work they
#   issue — `phases_charged()` in three loops, a literal `false` in four — and
#   then repeats that decision in two unrelated places: the `charge` argument of
#   every `rollback_round_caches(...)` it makes, and the `charged:` field of
#   every `RoundPhases` / `RoundStats` record it builds. Nothing holds the two
#   together. A loop whose rollback charges and whose record says it did not
#   moves the phase timings and re-attributes the work to the drafter, with
#   every token identical and every count identical.
#
#   Nothing else can see it. The equivalence pairs read the answer; the accept
#   counters read the aggregate; the per-round event stream reads `charged`, and
#   it reads `false` on both sides of any comparison by construction, because a
#   capture that enabled the switch would be measuring a different, slower run
#   than the one that ships. So a change that hard-wires one value across the
#   collapse is invisible to every observable this crate has, which is what this
#   gate is for.
#
# RULE 1 (one decision per loop)
#   Within one function, the `charge` argument of every `rollback_round_caches`
#   call and the value of every `charged:` field must be the same token.
#
# RULE 2 (the population)
#   Across the round loops, the multiset of those tokens is exactly
#   `charge_phases` three times and `false` four times. This is a census, not a
#   preference: it is here so that a loop moved from one schedule to the other,
#   or a `true` wired in, has to be written down rather than merged.
#
# EXIT
#   0 clean, 1 a rule fired, 2 the gate could not scan — a missing tree, fewer
#   round loops than the tree has, or a call whose arguments it could not read
#   back. A scan that finds nothing must not report a pass.

set -uo pipefail

root="${SPEC_CHARGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
loops_dir="$root/crates/rmlx-models/src/speculative"

# The round loops the tree ships. Fewer than this is a scan that lost one.
want_loops="${SPEC_CHARGE_WANT_LOOPS:-7}"
# The census, as `<token>:<count>` pairs, sorted.
want_census="${SPEC_CHARGE_WANT_CENSUS:-charge_phases:3 false:4}"

fail=0
scan_error=0

note() { printf '%s\n' "$*" >&2; }

if [ ! -d "$loops_dir" ]; then
  note "check-spec-charge: no speculative source directory at ${loops_dir#"$root"/}"
  exit 2
fi

# awk walks each file once and emits one record per charge site:
#
#   <file> <TAB> <function> <TAB> <kind> <TAB> <token>
#
# `kind` is `rollback` or `record`, and `token` is `?` for a call this scan
# could not read back — reported as a scan error rather than skipped, because an
# unread call is exactly the one that would carry the wrong decision.
#
# A function opens at a `fn` item and closes when its brace depth returns to
# zero. A nested `fn` inside a body is not opened: the enclosing loop is the
# unit the rule is about.
sites=$(
  find "$loops_dir" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
    xargs -0 awk '
      function emit(kind, token) {
        printf "%s\t%s\t%s\t%s\n", FILENAME, fname, kind, token
      }
      # Close the current call and report the argument in the charge position.
      function close_call(   n) {
        n = split(args, a, "\x1f")
        # caches, lin, fed, pre-round offset, target, charge, device.
        if (n != 7) { emit("rollback", "?") } else { emit("rollback", a[6]) }
        in_call = 0; args = ""
      }
      function reset() { in_body = 0; opened = 0; in_call = 0; fname = ""; depth = 0; args = "" }
      FNR == 1 { reset() }

      # -- a function item, when not already inside one --------------------
      !in_body && /^[[:space:]]*(pub(\([a-z]+\))?[[:space:]]+)?(async[[:space:]]+)?fn[[:space:]]+[a-z_0-9]+/ {
        line = $0
        sub(/^.*fn[[:space:]]+/, "", line)
        sub(/[^A-Za-z0-9_].*$/, "", line)
        fname = line
        in_body = 1
        opened = 0
        depth = 0
      }

      !in_body { next }

      # -- inside a function body ------------------------------------------
      {
        stripped = $0
        sub(/^[[:space:]]*\/\/.*$/, "", stripped)
      }

      in_call {
        if (stripped ~ /^[[:space:]]*\)/) {
          close_call()
        } else if (stripped ~ /[^[:space:]]/) {
          arg = stripped
          gsub(/^[[:space:]]+|[[:space:]]*,[[:space:]]*$/, "", arg)
          args = (args == "") ? arg : args "\x1f" arg
        }
      }

      # Only a call opens one: the definition carries no charge argument, and a
      # call written on one line is a shape this scan does not read back.
      !in_call && index(stripped, "rollback_round_caches(") > 0 &&
      stripped !~ /fn[[:space:]]+rollback_round_caches/ {
        rest = stripped
        sub(/^.*rollback_round_caches\(/, "", rest)
        if (rest ~ /[^[:space:]]/) { emit("rollback", "?") } else { in_call = 1; args = "" }
      }

      !in_call && stripped ~ /^[[:space:]]*charged:[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*,/ {
        tok = stripped
        sub(/^[[:space:]]*charged:[[:space:]]*/, "", tok)
        sub(/[^A-Za-z0-9_].*$/, "", tok)
        emit("record", tok)
      }

      {
        depth += gsub(/\{/, "{") - gsub(/\}/, "}")
        if (depth > 0) { opened = 1 }
        if (opened && depth <= 0) { reset() }
      }
    '
)

if [ -z "$sites" ]; then
  note "check-spec-charge: found no charge site under ${loops_dir#"$root"/}."
  note "  A site is a \`rollback_round_caches\` call's charge argument or a \`charged:\`"
  note "  field. Either both were renamed out from under this gate or the scan is"
  note "  broken; a gate that matched nothing must not report a pass."
  exit 2
fi

# ---- Rule 1: one token per loop -------------------------------------------

loops=$(printf '%s\n' "$sites" | cut -f1,2 | sort -u)
loop_count=0
census=""

while IFS=$'\t' read -r file fn; do
  [ -n "$fn" ] || continue
  rel="${file#"$root"/}"
  tokens=$(printf '%s\n' "$sites" | awk -F'\t' -v f="$file" -v n="$fn" \
    '$1 == f && $2 == n { print $4 }' | sort -u)
  if printf '%s\n' "$tokens" | grep -qx '?'; then
    note "check-spec-charge: $rel: \`$fn\` makes a \`rollback_round_caches\` call this"
    note "  gate could not read the arguments of. An unread call is not a checked call,"
    note "  and it is the one that would carry the other schedule."
    scan_error=1
    continue
  fi
  loop_count=$((loop_count + 1))
  if [ "$(printf '%s\n' "$tokens" | wc -l | tr -d ' ')" != "1" ]; then
    note "check-spec-charge: $rel: \`$fn\` names more than one charge decision —" \
      "$(printf '%s' "$tokens" | tr '\n' ' ')"
    note "  The rollback's \`charge\` argument and the record's \`charged:\` field are read"
    note "  by different code and must be the same token: a round that charged and"
    note "  recorded that it did not re-attributes its own work with nothing saying so."
    fail=1
    continue
  fi
  census="$census$tokens"$'\n'
done <<<"$loops"

if [ "$scan_error" = "1" ]; then
  exit 2
fi

if [ "$loop_count" -lt "$want_loops" ]; then
  note "check-spec-charge: scanned $loop_count round loops and the tree ships $want_loops."
  note "  A loop the scan lost is a loop with no gate on it, which reads as a pass."
  exit 2
fi

# ---- Rule 2: the census ----------------------------------------------------

have_census=$(printf '%s' "$census" | grep -v '^$' | sort | uniq -c |
  awk '{ printf "%s:%s ", $2, $1 }' | sed 's/ $//')

if [ "$have_census" != "$want_census" ]; then
  note "check-spec-charge: the charge census is \"$have_census\" and the tree records"
  note "  \"$want_census\"."
  note "  Three loops time their phases and charge them; four time none and charge"
  note "  none. A loop moved between the two, or a value wired in, changes what the"
  note "  phase timings mean and is written down here rather than merged."
  fail=1
fi

if [ "$fail" != "0" ]; then
  exit 1
fi

echo "OK: $loop_count speculative round loops, each naming one charge decision; census $have_census."
