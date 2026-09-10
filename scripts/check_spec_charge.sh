#!/usr/bin/env bash
# scripts/check_spec_charge.sh — CI gate: each speculative round loop names one
# phase-charge decision, and it names it everywhere the decision is read.
#
# WHY
#   A round loop decides once whether its phases are charged for the work they
#   issue — `phases_charged()` in three loops, a literal `false` in four — and
#   then repeats that decision in two unrelated places: the `charge` argument of
#   every `rollback_round(...)` it makes, and the `charged:` field of
#   every `RoundReport` it logs and of the `RoundTotals` it hands the one
#   recorder. Nothing holds the two together. A loop whose rollback charges and whose record says it did not
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
#   Within one round loop, the `charge` argument of every `rollback_round`
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
#   A fn that is not a round loop and calls `rollback_round` is either a loop
#   the derivation lost or a rollback moved out of one. Both read as a census
#   that moved, which invites the census to be edited; both are exit 2 here
#   instead.
#
#   `rollback_round` is the shared refold-or-disarm, and it is not itself a
#   site: it forwards the `charge` its caller named, down to the low-level
#   `rollback_round_caches`, whose name this needle does not match. That is the
#   same split RULE 5 makes on the record side — the decision stays at the
#   loop's own call site, where the rollback beside it can be read against it,
#   and the shared code carries it rather than naming one.
#
# RULE 6 (the low-level rollback is named in one file, and nowhere else)
#   `rollback_round` is the one rollback a round loop calls, and it is the one
#   whose `charge` argument RULE 1 reads. The low-level pair beneath it —
#   `rollback_round_caches`, which truncates and refolds, and `refold_lin_tapes`,
#   which rebuilds the recurrent state — also take a `charge`, at a call this
#   reader does not open. Anything that named one of those would make a second
#   charge decision beside the one a loop declares, and this gate would report
#   the declared one as the only one.
#
#   Two readings, because either alone has a hole. **By file**: a call from
#   anywhere but `round_common.rs` is exit 2 — a round loop, a helper the loop
#   calls, or a fn nothing calls, which is the shape a fn-shaped rule missed.
#   **By fn**: a round loop is read wherever it lives, so a loop that moves into
#   `round_common.rs` — the direction the shared skeleton is going — does not
#   inherit the file's exemption. The exemption is anchored to that one path and
#   not to a basename, so a `round_common.rs` added under any other directory is
#   read like every other file.
#
#   Both fns are private to `round_common`, so in the tree this is a second
#   reader on a property the compiler already holds — but the compiler cannot
#   see a source edit that has not been built.
#
# RULE 7 (one seam for the round event, and one file that can name its target)
#   Every round loop closes its round through `log_round`, the one emitter, and
#   inside the engine source this scan covers, the target that emitter writes on
#   is named in `round_stats.rs` alone. It is not the only copy of the string in
#   the repo — `tests/common/round_stream.rs` restates the literal, because the
#   constant is private and a capture has to name the target it declines to
#   enable at TRACE. That copy is outside this scan; it is held to the engine's
#   by `the_engine_and_its_readers_state_the_same_target_and_round_fields` in
#   `tests/spec_greedy_equivalence.rs`, which reads `round_stats.rs` and looks
#   for the declaration. What this rule enforces is the engine half. A loop that
#   keeps its charge decision honest and writes its own
#   `tracing::debug!` on that target, with the same fields in the same order,
#   produces a byte-identical line — so the pinned round-stream digests agree,
#   the equivalence pairs agree, and this gate's own census agrees. Measured:
#   that mutation passes every check in this tree. What it silently drops is the
#   charged round's carry check and the guarantee that a field added to the
#   shared record reaches that loop.
#
#   Two readings again, and both are needed. **By name**: `PHASE_TARGET` or the
#   literal `rmlx::spec::phase` outside `round_stats.rs` is exit 2 — a loop that
#   cannot name the target cannot write on it, which is the structural half. Read
#   twice for the reason RULE 6 is: by file, which is the only reading that sees
#   a mention outside any fn, and by fn, so a round loop that moved into
#   `round_stats.rs` does not inherit that file's exemption — it would otherwise
#   call the emit once, satisfy every other reading, and write a second event on
#   the target beside it. **By call**: each round loop calls `log_round(` exactly
#   once, at a statement position. Zero is a loop that left the seam; two or more
#   is a second record shape under one target, which is the shape the collapse
#   removed.
#
#   Every needle in this gate reads a line's code and not the comment beside it,
#   including this rule's by-file reading. A gate that read comments would go red
#   on the sentence explaining the seam — which the chunk introducing a skeleton
#   wrapper has to write in `round_common.rs` — and would count a `// … log_round(`
#   note as the call a loop no longer makes. The stripper is quote-aware, so a
#   `//` inside a string literal does not truncate the line.
#
#   The name is private to `round_stats.rs` in the tree, so the name reading is
#   a second reader on a property the compiler already holds — and the compiler
#   cannot see a source edit that has not been built.
#
#   **What a loop whose emit moves into the skeleton must do.** The call reading
#   is of `log_round(` by name, so a loop that closes its round through a
#   `round_common.rs` wrapper reads zero and is refused. That is deliberate while
#   the wrapper does not exist: a seam nothing names is a seam this gate cannot
#   follow. The chunk that introduces one moves this needle to it in the same
#   commit, exactly as RULE 6's needle moved when the rollback went behind
#   `rollback_round` — and, as there, the old seam has to become unreachable
#   from the loops, or the gate has lost a defect class rather than followed
#   it.
#
# RULE 5 (nothing names the decision outside the population either)
#   A fn that is not a round loop and carries a `charged:` field names a
#   decision this gate cannot check against a rollback, because there is none in
#   it. The shared recorder carries the field across from a destructured
#   binding — `charged,`, which is not a `charged:` site — so a `charged:` in
#   any fn that is not a round loop is a decision made where nothing can hold it
#   to the loop that ordered it.
#
#   The reach is fn bodies, and only those: this scan opens at a `fn` item, so a
#   `charged:` in a module-level `const` or `static` is outside it entirely. The
#   rule is about functions, and a decision parked in a constant is a shape
#   review has to catch.
#
#   A shared constructor for a loop's totals is refused by this rule, and that
#   is the intent rather than a side effect: the token has to stay at the loop's
#   own call site for the rollback beside it to be checked against, so a later
#   collapse of the per-loop `RoundTotals` literals has to keep it there.
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
#   to a helper is not a rollback this gate can read), a driver that reaches
#   past the shared rollback to the low-level pair, a driver that does not reach
#   the one round emit exactly once, the phase target named outside
#   `round_stats.rs`, a charge site outside the population, or a call, a field or
#   a binding whose shape it could not read back. A scan that finds nothing must
#   not report a pass.
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
#   <low-level rollbacks> <TAB> <round emits> <TAB> <phase-target mentions> <TAB>
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
# A line's code, with any comment removed. Quote-aware, so a `//` inside a string
# literal is not a comment and the line is not truncated at it. Shared by both of
# this gate's readings: every needle below is a substring test, and a name is
# most likely to be written without being used in the sentence explaining it.
readonly DECOMMENT='
      function decomment(s,   i, n, q, ch) {
        n = length(s); q = 0
        for (i = 1; i <= n; i++) {
          ch = substr(s, i, 1)
          if (q && ch == "\\") { i++; continue }
          if (ch == "\"") { q = !q; continue }
          if (!q && ch == "/" && substr(s, i + 1, 1) == "/") { return substr(s, 1, i - 1) }
        }
        return s
      }
'

records=$(
  find "$loops_dir" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
    xargs -0 awk "$DECOMMENT"'
      function addtok(t) { if (!(t in toks)) { toks[t] = 1 } }
      function reset() {
        in_sig = 0; awaiting_body = 0; in_body = 0; in_call = 0; fname = ""
        depth = 0; paren = 0; args = ""; has_step = 0; has_stats = 0
        bind_ok = 0; bind_bad = 0; bind_odd = 0; nroll = 0; nrec = 0; nlow = 0
        nemit = 0; ntarget = 0
        delete toks
      }
      function flush(   t, joined) {
        if (fname != "") {
          joined = ""
          for (t in toks) { joined = (joined == "") ? t : joined " " t }
          printf "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n", \
            FILENAME, fname, (has_step && has_stats), nroll, nrec, nlow, nemit, \
            ntarget, \
            (bind_odd ? "odd" : (bind_ok ? "ok" : (bind_bad ? "bad" : "none"))), joined
        }
        reset()
      }
      # The `charge` argument of a call whose arguments are one per line:
      # caches, lin, round tokens, pre-round offset, target, charge, device.
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
      { code = decomment($0) }

      in_call {
        if (code ~ /^[[:space:]]*\)/) {
          close_call()
        } else if (code ~ /[^[:space:]]/) {
          arg = code
          gsub(/^[[:space:]]+|[[:space:]]*,[[:space:]]*$/, "", arg)
          args = (args == "") ? arg : args "\x1f" arg
        }
        next
      }

      index(code, "RoundTotals {") > 0 { has_stats = 1 }

      # Three answers, not two. A plain `let charge_phases = <expr>;` is a
      # binding this scan reads: it is the call and nothing else, or it is a
      # different decision. Anything else naming the binding — a `mut`, a type
      # annotation, a right-hand side that does not end on the line — is a shape
      # it cannot read, which is a scan error and not a verdict about the loop.
      index(code, "charge_phases") > 0 && code ~ /^[[:space:]]*let[[:space:]]/ {
        if (code ~ /^[[:space:]]*let[[:space:]]+charge_phases[[:space:]]*=[[:space:]]*([A-Za-z_0-9]+::)*phases_charged\(\);[[:space:]]*$/) {
          bind_ok = 1
        } else if (code ~ /^[[:space:]]*let[[:space:]]+charge_phases[[:space:]]*=.*;[[:space:]]*$/) {
          bind_bad = 1
        } else {
          bind_odd = 1
        }
      }

      # Only a call opens one: the definition carries no charge argument, and a
      # call written on one line is a shape this scan does not read back.
      # `rollback_round_caches(` does not contain this needle, so the shared
      # helper forwarding its own `charge` is not a site.
      index(code, "rollback_round(") > 0 &&
      code !~ /fn[[:space:]]+rollback_round\(/ {
        rest = code
        sub(/^.*rollback_round\(/, "", rest)
        if (rest ~ /[^[:space:]]/) { addtok("?"); nroll++ } else { in_call = 1; args = "" }
      }

      # RULE 6: the low-level pair the shared rollback is built on. Their own
      # definitions are not calls, and neither is `rollback_round_caches` inside
      # the shared rollback — that fn is not a round loop.
      (index(code, "rollback_round_caches(") > 0 ||
       index(code, "refold_lin_tapes(") > 0) &&
      code !~ /fn[[:space:]]+(rollback_round_caches|refold_lin_tapes)[[:space:]]*\(/ {
        nlow++
      }

      # RULE 7: the one round emit. A call at a statement position, not the
      # definition — the emitter is not a round loop and is not counted against
      # itself. Read off `code`, so a trailing comment naming it is not a call;
      # and anchored, so the count is of calls a loop makes rather than of the
      # name appearing anywhere on a line.
      code ~ /^[[:space:]]*(let[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=[[:space:]]*)?([A-Za-z_][A-Za-z0-9_]*::)*log_round\(/ {
        nemit++
      }

      # RULE 7 by name, per fn. The file-level reading below cannot see which fn
      # a mention sits in, and the file that owns the target has to be allowed
      # to name it — so this is what denies a round loop that moved into that
      # file the exemption the file carries.
      (index(code, "PHASE_TARGET") > 0 || index(code, "rmlx::spec::phase") > 0) {
        ntarget++
      }

      # Symmetric with the rollback side: any `charged:` is a site, and one
      # whose value is not a bare identifier is unreadable rather than absent.
      # A trailing comma is not required — a record`s last field carries none.
      index(code, "charged:") > 0 {
        tok = code
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
# RULE 5: a fn that is not a round loop and writes a `charged:` field.
strays=$(printf '%s\n' "$records" | awk -F'\t' '$3 == 0 && $5 > 0')
# RULE 6: the low-level rollback named outside `round_common.rs`, or named
# inside it by a round loop, which the file's exemption is not for.
lowlevel=$(printf '%s\n' "$records" |
  awk -v rc="$loops_dir/round_common.rs" -F'\t' '$6 > 0 && ($1 != rc || $3 == 1)')

# RULE 7 by name, by file: the phase target belongs to `round_stats.rs`.
# Anchored to that one path, like RULE 6's exemption, so the same basename under
# another directory is read like every other file. This reading is the one that
# sees a mention outside any fn — a module-level `const` the per-fn scan below
# cannot open.
target_strays=$(
  find "$loops_dir" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
    xargs -0 awk "$DECOMMENT"'
      { c = decomment($0) }
      (index(c, "PHASE_TARGET") > 0 || index(c, "rmlx::spec::phase") > 0) &&
      !(FILENAME in seen) { seen[FILENAME] = 1; print FILENAME }
    ' |
    grep -v -x -F "$loops_dir/round_stats.rs"
)
# RULE 7 by name, by fn: a round loop that moved into `round_stats.rs` does not
# inherit that file's exemption. It would otherwise call `log_round` once, pass
# every reading here, and write a second event on the target beside it.
target_loops=$(printf '%s\n' "$records" |
  awk -v rs="$loops_dir/round_stats.rs" -F'\t' '$8 > 0 && $1 == rs && $3 == 1')

if [ -z "$drivers" ]; then
  note "check-spec-charge: found no round loop under ${loops_dir#"$root"/}."
  note "  A round loop is a fn taking \`step_fn: &mut dyn FnMut(&ProbeStep)\` that builds"
  note "  a \`RoundTotals\`. Either both were renamed out from under this gate or the scan"
  note "  is broken; a gate that matched nothing must not report a pass."
  exit 2
fi

loop_count=0
census=""

while IFS=$'\t' read -r file fn _driver _nroll _nrec _nlow _nemit _ntarget _bind _tokens; do
  [ -n "$fn" ] || continue
  note "check-spec-charge: ${file#"$root"/}: \`$fn\` rolls a round's caches back and is"
  note "  not one of the round loops this gate derived. Either the derivation lost a"
  note "  loop — a \`RoundTotals\` built by a constructor, a signature rustfmt wrapped, a"
  note "  \`where\` clause — or a rollback moved out of one. Both read as a census that"
  note "  moved, and editing the census would bury either."
  scan_error=1
done <<<"$orphans"

while IFS=$'\t' read -r file fn _driver _nroll _nrec _nlow _nemit ntarget _bind _tokens; do
  [ -n "$fn" ] || continue
  note "check-spec-charge: ${file#"$root"/}: \`$fn\` is a round loop and names the"
  note "  per-round event's target $ntarget time(s). The file that owns the target may"
  note "  name it; a loop that moved into that file does not inherit the exemption, or"
  note "  it could call the emit once and write a second event on the target beside it."
  scan_error=1
done <<<"$target_loops"

while IFS= read -r file; do
  [ -n "$file" ] || continue
  note "check-spec-charge: ${file#"$root"/} names the per-round event's target."
  note "  That target is \`round_stats.rs\`'s alone: a fn that can name it can write a"
  note "  second round event beside the one \`log_round\` writes, with the same fields"
  note "  in the same order — and a line that agrees byte for byte is invisible to the"
  note "  pinned digests, to the equivalence pairs and to this census."
  scan_error=1
done <<<"$target_strays"

while IFS=$'\t' read -r file fn _driver _nroll _nrec nlow _nemit _ntarget _bind _tokens; do
  [ -n "$fn" ] || continue
  note "check-spec-charge: ${file#"$root"/}: \`$fn\` makes $nlow call(s) to the low-level"
  note "  rollback beneath \`rollback_round\`. Those take a \`charge\` of their own at a call"
  note "  this gate does not open, so the decision it reads at a loop's own site would no"
  note "  longer be the only one made. Only \`round_common.rs\` may name them, and only"
  note "  outside a round loop."
  scan_error=1
done <<<"$lowlevel"

while IFS=$'\t' read -r file fn _driver _nroll nrec _nlow _nemit _ntarget _bind tokens; do
  [ -n "$fn" ] || continue
  note "check-spec-charge: ${file#"$root"/}: \`$fn\` writes $nrec \`charged:\` field(s) — $tokens —"
  note "  and is not a round loop, so no rollback in it says whether that is the decision"
  note "  the loop ordered. The one fn that may write the field is the shared recorder,"
  note "  which carries it across as a destructured \`charged,\` and names no value here."
  fail=1
done <<<"$strays"

while IFS=$'\t' read -r file fn _driver nroll nrec nlow nemit _ntarget bind tokens; do
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
  if [ "$nemit" != "1" ]; then
    note "check-spec-charge: $rel: \`$fn\` reaches the one round emit $nemit time(s)."
    note "  Every round loop closes its round through \`log_round\`, once. None is a loop"
    note "  that left the seam and writes its own event, which the digests cannot see"
    note "  when the line agrees; more than one is a second record shape under the one"
    note "  target, which is what the collapse removed."
    scan_error=1
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
