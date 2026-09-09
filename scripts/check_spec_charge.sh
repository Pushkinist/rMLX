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
# WHAT A ROUND LOOP IS
#   Two conditions, both structural, neither a count:
#     (a) the signature rule `check_spec_sampling.sh` uses — a fn whose
#         parameters include `step_fn: &mut dyn FnMut(&ProbeStep)`, at any
#         visibility. That argument is what makes a fn a generation driver.
#     (b) it constructs a `RoundTotals` — the numbers a round loop closes on,
#         handed to the one function that assembles the record from them. That
#         is what separates a round loop from the fns that satisfy (a) and run
#         no round: `spec_generate_greedy`, which validates a request and
#         delegates, `emit_step`, the shared per-token emit helper, and
#         `emit_seed_token`, which emits a loop's seed and is handed the totals
#         rather than naming any. None builds a `RoundTotals`.
#   The population is derived, so a loop added or removed moves the census
#   rather than slipping past a number typed into this file.
#
# RULE 1 (one decision per loop)
#   Within one round loop, the `charge` argument of every `rollback_round_caches`
#   call and the value of every `charged:` field must be the same token. The
#   record is assembled elsewhere now, so the field the loop still writes is the
#   `charged:` of the `RoundTotals` it hands over — the decision is named at the
#   call site either way, which is the only place this gate can read it: what the
#   shared recorder then does with the value is past its reach.
#
# RULE 2 (the token means what it says)
#   A loop whose token is `charge_phases` must bind it, in the same function, to
#   exactly `phases_charged()` and nothing else — the whole right-hand side, not
#   a prefix of it. Without that the census reads a spelling: a loop that binds
#   the name to a literal, or ORs a second condition into the call, charges on
#   requests the switch did not ask for, and this gate would have counted it
#   among the three that ask.
#
# RULE 4 (nothing charges outside the population)
#   A fn that is not a round loop and calls `rollback_round_caches` is either a
#   loop the derivation lost or a rollback moved out of one. Both read as a
#   census that moved, which invites the census to be edited; both are exit 2
#   here instead.
#
# RULE 3 (the population)
#   Across the round loops, the multiset of those tokens is exactly
#   `charge_phases` three times and `false` four times. This is a census, not a
#   preference: it is here so that a loop moved from one schedule to the other,
#   or a `true` wired in, has to be written down rather than merged.
#
# EXIT
#   0 clean, 1 a rule fired, 2 the gate could not scan — a missing tree, no
#   round loop found, a driver with no charge site in it (a rollback delegated
#   to a helper is not a rollback this gate can read), a charge site outside the
#   population, or a call, a field or a binding whose shape it could not read
#   back. A scan that finds nothing must not report a pass.
#
# PORTABILITY
#   The argument list is joined on `\x1f` inside awk. That escape is what
#   BSD awk on macOS reads, which is the only awk `make ci` runs here; it is not
#   exercised against gawk or mawk, and a run under either should check that a
#   readable call is still read before trusting a clean result.

set -uo pipefail

# The only variable here: the fixtures point the scan at a synthetic root. The
# rules themselves are constants — a gate whose expectations can be relaxed from
# the environment is a gate that passes for whoever sets them.
root="${SPEC_CHARGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
loops_dir="$root/crates/rmlx-models/src/speculative"

readonly WANT_CENSUS="charge_phases:3 false:4"

fail=0
scan_error=0

note() { printf '%s\n' "$*" >&2; }

if [ ! -d "$loops_dir" ]; then
  note "check-spec-charge: no speculative source directory at ${loops_dir#"$root"/}"
  exit 2
fi

# awk walks each file once and emits one record per function:
#
#   <file> <TAB> <fn> <TAB> <driver> <TAB> <rollbacks> <TAB> <records> <TAB>
#   <binds-phases_charged> <TAB> <tokens, space-joined>
#
# `driver` is 1 for a fn meeting both conditions above. A token of `?` is a site
# whose value this scan could not read back — reported as a scan error rather
# than skipped, because an unread site is exactly the one that would carry the
# other decision.
#
# A function opens at a `fn` item and closes when its brace depth returns to
# zero. A nested `fn` inside a body is not opened: the enclosing loop is the
# unit the rules are about.
records=$(
  find "$loops_dir" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
    xargs -0 awk '
      function addtok(t) { if (!(t in toks)) { toks[t] = 1 } }
      function reset() {
        in_sig = 0; awaiting_body = 0; in_body = 0; in_call = 0; fname = ""
        depth = 0; paren = 0; args = ""; has_step = 0; has_stats = 0
        bind_ok = 0; bind_bad = 0; bind_odd = 0; nroll = 0; nrec = 0
        delete toks
      }
      function flush(   t, joined) {
        if (fname != "") {
          joined = ""
          for (t in toks) { joined = (joined == "") ? t : joined " " t }
          printf "%s\t%s\t%d\t%d\t%d\t%s\t%s\n", \
            FILENAME, fname, (has_step && has_stats), nroll, nrec, \
            (bind_odd ? "odd" : (bind_ok ? "ok" : (bind_bad ? "bad" : "none"))), joined
        }
        reset()
      }
      # The `charge` argument of a call whose arguments are one per line:
      # caches, lin, fed, pre-round offset, target, charge, device.
      function close_call(   n) {
        n = split(args, a, "\x1f")
        addtok(n == 7 ? a[6] : "?")
        nroll++
        in_call = 0; args = ""
      }
      FNR == 1 { flush() }

      # -- a function item, when not already inside one --------------------
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

      # -- the parameter list ----------------------------------------------
      in_sig {
        if (index($0, "step_fn: &mut dyn FnMut(&ProbeStep)") > 0) { has_step = 1 }
        # A per-parameter `) -> ` is inside the list and must not end the scan.
        paren += gsub(/\(/, "(") - gsub(/\)/, ")")
        if (paren <= 0) {
          in_sig = 0
          awaiting_body = 1
          # A `where` clause sits between the closing parenthesis and the body,
          # so the body opens at the next `{` and not necessarily here.
          if (index($0, "{") > 0) {
            awaiting_body = 0
            in_body = 1
            depth = gsub(/\{/, "{") - gsub(/\}/, "}")
            if (depth < 0) { depth = 0 }
          }
        }
        next
      }

      awaiting_body {
        if (index($0, "{") > 0) {
          awaiting_body = 0
          in_body = 1
          depth = gsub(/\{/, "{") - gsub(/\}/, "}")
          if (depth < 0) { depth = 0 }
        }
        next
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
        next
      }

      index(stripped, "RoundTotals {") > 0 { has_stats = 1 }

      # Three answers, not two. A plain `let charge_phases = <expr>;` is a
      # binding this scan reads: it is the call and nothing else, or it is a
      # different decision. Anything else naming the binding — a `mut`, a type
      # annotation, a right-hand side that does not end on the line — is a shape
      # it cannot read, which is a scan error and not a verdict about the loop.
      index(stripped, "charge_phases") > 0 && stripped ~ /^[[:space:]]*let[[:space:]]/ {
        if (stripped ~ /^[[:space:]]*let[[:space:]]+charge_phases[[:space:]]*=[[:space:]]*([A-Za-z_0-9]+::)*phases_charged\(\);[[:space:]]*$/) {
          bind_ok = 1
        } else if (stripped ~ /^[[:space:]]*let[[:space:]]+charge_phases[[:space:]]*=.*;[[:space:]]*$/) {
          bind_bad = 1
        } else {
          bind_odd = 1
        }
      }

      # Only a call opens one: the definition carries no charge argument, and a
      # call written on one line is a shape this scan does not read back.
      index(stripped, "rollback_round_caches(") > 0 &&
      stripped !~ /fn[[:space:]]+rollback_round_caches/ {
        rest = stripped
        sub(/^.*rollback_round_caches\(/, "", rest)
        if (rest ~ /[^[:space:]]/) { addtok("?"); nroll++ } else { in_call = 1; args = "" }
      }

      # Symmetric with the rollback side: any `charged:` is a site, and one
      # whose value is not a bare identifier is unreadable rather than absent.
      # A trailing comma is not required — a record`s last field carries none.
      index(stripped, "charged:") > 0 {
        tok = stripped
        sub(/^.*charged:[[:space:]]*/, "", tok)
        if (tok ~ /^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*(,|\}|$)/) {
          sub(/[^A-Za-z0-9_].*$/, "", tok)
          addtok(tok)
        } else {
          addtok("?")
        }
        nrec++
      }

      {
        depth += gsub(/\{/, "{") - gsub(/\}/, "}")
        if (depth <= 0) { flush() }
      }
      END { flush() }
    '
)

drivers=$(printf '%s\n' "$records" | awk -F'\t' '$3 == 1')
# RULE 4: a fn that is not a round loop and rolls a round's caches back.
orphans=$(printf '%s\n' "$records" | awk -F'\t' '$3 == 0 && $4 > 0')

if [ -z "$drivers" ]; then
  note "check-spec-charge: found no round loop under ${loops_dir#"$root"/}."
  note "  A round loop is a fn taking \`step_fn: &mut dyn FnMut(&ProbeStep)\` that builds"
  note "  a \`RoundTotals\`. Either both were renamed out from under this gate or the scan"
  note "  is broken; a gate that matched nothing must not report a pass."
  exit 2
fi

loop_count=0
census=""

while IFS=$'\t' read -r file fn _driver _nroll _nrec _bind _tokens; do
  [ -n "$fn" ] || continue
  note "check-spec-charge: ${file#"$root"/}: \`$fn\` rolls a round's caches back and is"
  note "  not one of the round loops this gate derived. Either the derivation lost a"
  note "  loop — a \`RoundTotals\` built by a constructor, a signature rustfmt wrapped, a"
  note "  \`where\` clause — or a rollback moved out of one. Both read as a census that"
  note "  moved, and editing the census would bury either."
  scan_error=1
done <<<"$orphans"

while IFS=$'\t' read -r file fn _driver nroll nrec bind tokens; do
  [ -n "$fn" ] || continue
  rel="${file#"$root"/}"
  loop_count=$((loop_count + 1))

  if printf '%s\n' "$tokens" | tr ' ' '\n' | grep -qx '?'; then
    note "check-spec-charge: $rel: \`$fn\` has a charge site this gate could not read"
    note "  the value of. An unread site is not a checked site, and it is the one that"
    note "  would carry the other decision."
    scan_error=1
    continue
  fi
  if [ "$nroll" = "0" ] || [ "$nrec" = "0" ]; then
    note "check-spec-charge: $rel: \`$fn\` has $nroll rollback and $nrec record charge sites."
    note "  A loop is checked only where both are visible: a rollback delegated to a"
    note "  helper, or a record built somewhere else, is a decision this gate cannot"
    note "  follow, and a loop it cannot follow is not a loop it checks."
    scan_error=1
    continue
  fi

  count=$(printf '%s\n' "$tokens" | tr ' ' '\n' | grep -c '[^[:space:]]')
  if [ "$count" != "1" ]; then
    note "check-spec-charge: $rel: \`$fn\` names more than one charge decision — $tokens"
    note "  The rollback's \`charge\` argument and the record's \`charged:\` field are read"
    note "  by different code and must be the same token: a round that charged and"
    note "  recorded that it did not re-attributes its own work with nothing saying so."
    fail=1
    continue
  fi
  if [ "$tokens" = "charge_phases" ] && [ "$bind" = "odd" ]; then
    note "check-spec-charge: $rel: \`$fn\` binds \`charge_phases\` in a shape this gate"
    note "  cannot read — a \`mut\`, a type annotation, or a right-hand side that does not"
    note "  end on its own line. An unread binding is not a checked one, and refusing it"
    note "  as RULE 2 would name the wrong defect."
    scan_error=1
    continue
  fi
  if [ "$tokens" = "charge_phases" ] && [ "$bind" != "ok" ]; then
    note "check-spec-charge: $rel: \`$fn\` charges on \`charge_phases\` and does not bind it"
    note "  to \`phases_charged()\` alone. The census would then be reading a spelling: a"
    note "  loop that binds the name to a literal, or ORs a second condition into the"
    note "  call, charges on requests the switch did not ask for and is counted here"
    note "  among the three that ask."
    fail=1
    continue
  fi
  census="$census$tokens"$'\n'
done <<<"$drivers"

if [ "$scan_error" = "1" ]; then
  exit 2
fi

have_census=$(printf '%s' "$census" | grep -v '^$' | LC_ALL=C sort | uniq -c |
  awk '{ printf "%s:%s ", $2, $1 }' | sed 's/ $//')

if [ "$have_census" != "$WANT_CENSUS" ]; then
  note "check-spec-charge: the charge census is \"$have_census\" and the tree records"
  note "  \"$WANT_CENSUS\"."
  note "  Three loops time their phases and charge them; four time none and charge"
  note "  none. A loop moved between the two, or a value wired in, changes what the"
  note "  phase timings mean and is written down here rather than merged."
  fail=1
fi

if [ "$fail" != "0" ]; then
  exit 1
fi

echo "OK: $loop_count speculative round loops, each naming one charge decision; census $have_census."
