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
# THE THREE POPULATIONS, ALL DERIVED
#   The seven loops are collapsing onto one shared loop that is handed its
#   decision as a value. So the decision moves out of the loop and into the fn
#   that starts one, and this gate follows it rather than staying on the shape
#   it used to have. Nothing below is a name list.
#
#   (a) ROUND LOOPS. Two conditions, both structural, neither a count:
#         (a1) the signature rule `check_spec_sampling.sh` uses — a fn whose
#              parameters include `step_fn: &mut dyn FnMut(&ProbeStep)`, at any
#              visibility. That argument is what makes a fn a generation driver.
#         (a2) it constructs a `RoundTotals` — the numbers a round loop closes
#              on, handed to the one function that assembles the record from
#              them. That is what separates a round loop from the fns that
#              satisfy (a1) and run no round: `spec_generate_greedy`, which
#              validates a request and delegates, `emit_step`, the shared
#              per-token emit helper, and `emit_seed_token`, which emits a
#              loop's seed and is handed the totals rather than naming any.
#       Within (a) a loop is FORWARDED when its *parameter list* carries
#       `RoundCfg`, the configuration type that holds the charge field, and
#       CLASSIC otherwise. The parameter is resolved by that declared type; a
#       list that destructures it, or declares it twice, leaves this scan unable
#       to say which parameter the rule is about, and it refuses rather than
#       reading the last one written. Membership is read off the signature and never off
#       the binding: a loop that hard-wires its charge still takes the
#       configuration, so it is still forwarded and RULE 8 still reaches it.
#       Reading membership off the binding instead would make RULE 8
#       unfireable — a loop failing it would leave the rule's population rather
#       than fail it.
#
#   (b) ENTRIES. A fn carrying the driver signature that constructs no
#       `RoundTotals`, calls no `rollback_round`, and *constructs* `RoundCfg`.
#       One per drafter at the end of the campaign, each naming the decision
#       once, at the call that runs the loop. The last conjunct is the
#       discriminator and not a detail: without it the population admits the
#       entry guard and the emit helpers, none of which decides anything. An
#       entry that stops constructing the configuration leaves the population,
#       which is what makes a dropped decision read as a lost entry rather than
#       as a clean scan — the census reads one site fewer.
#
#   (c) DRAFTER ROLLBACKS. A fn outside (a) that calls `rollback_round` and
#       neither drives a generation nor names a `charged:` field of its own.
#       Those last two are what separate a drafter's own rollback from a round
#       loop the derivation lost: a fn that carries the driver signature, or
#       states a decision in a record, looks like a loop and is refused as one
#       (exit 2) rather than read under RULE 4's argument rule.
#
# RULE 1 (one decision per loop)
#   Within one round loop, the `charge` argument of every `rollback_round`
#   call and the value of every `charged:` field must be the same token. The
#   record is assembled elsewhere now, so the field the loop still writes is the
#   `charged:` of the `RoundTotals` it hands over — the decision is named at the
#   call site either way, which is the only place this gate can read it: what the
#   shared recorder then does with the value is past its reach. A forwarded loop
#   is read the same way; what differs is what its one token is allowed to be,
#   which is RULE 8.
#
# RULE 2 (the token means what it says)
#   A classic loop or an entry whose token is `charge_phases` must bind it, in
#   the same function, to exactly `phases_charged()` and nothing else — the
#   whole right-hand side, not a prefix of it. Without that the census reads a
#   spelling: a loop that binds the name to a literal, or ORs a second condition
#   into the call, charges on requests the switch did not ask for, and this gate
#   would have counted it among the three that ask.
#
# RULE 4 (nothing charges outside the populations)
#   A fn that is not a round loop and calls `rollback_round` is either a loop
#   the derivation lost or a drafter rolling its own state back. The first reads
#   as a census that moved, which invites the census to be edited, and is exit 2.
#   The second is population (c), and its `charge` argument carries the decision
#   handed down from the loop that ordered it. Where the fn is handed the
#   round's context — a parameter whose declared type is `RoundCtx` — the charge
#   is `<ctx>.charged` and nothing else; where it is not, the looser reading
#   applies and any field of any of its own parameters will do. The strict arm
#   closes the hole in the loose one: a second parameter carrying a field of the
#   same name satisfies "a field of one of its own parameters" while holding a
#   different decision, and nothing downstream can see which was read. A literal
#   or a `phases_charged()` is a second decision made where nothing can hold it
#   to that loop, and is exit 1; an argument this scan cannot read back is
#   exit 2.
#
#   `rollback_round` is the shared refold-or-disarm, and it is not itself a
#   site: it forwards the `charge` its caller named, down to the low-level
#   `rollback_round_caches`, whose name this needle does not match. That is the
#   same split RULE 5 makes on the record side — the decision stays at the
#   caller's own call site, where it can be read against what was handed over,
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
#   read like every other file. The shared loop lives in `round_loop.rs` beside
#   it for exactly this reason.
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
# RULE 5 (the decision is named once, and only where it can be checked)
#   A fn that is in neither population and carries a `charged:` field names a
#   decision this gate cannot check against a rollback, because there is none in
#   it. The shared recorder carries the field across from a destructured
#   binding — `charged,`, which is not a `charged:` site — so a `charged:` in
#   such a fn is a decision made where nothing can hold it to the loop that
#   ordered it, and is exit 1.
#
#   Over population (b) the same needle becomes a rule rather than a membership
#   test: an entry states exactly one `charged:`. Zero or two is exit 1 naming
#   the entry, and a field this scan cannot read back is exit 2. Folding the
#   count into membership instead would put an entry with two decisions in no
#   population at all, where no rule reaches it.
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
# RULE 3 (the census)
#   Across every charge token that is not the forwarded configuration field —
#   whether a classic loop's, an entry's, or a forwarded loop's that stopped
#   being the field it was handed — the multiset of tokens is exactly
#   `charge_phases` three times and `false` four times. This is a census, not a
#   preference: it is here so that a loop moved from one schedule to the other,
#   a `true` wired in, or an entry that quietly dropped its decision, has to be
#   written down rather than merged. A forwarded loop that names or binds the
#   field it was handed contributes nothing; the same loop hard-wiring a literal
#   contributes one and the census reads eight sites.
#
# RULE 8 (the forwarded decision is the one that was handed over)
#   A forwarded loop binds its charge token, in the same fn, to exactly the
#   configuration field it was handed — `<cfg>.charged`, the whole right-hand
#   side and not a prefix — or names that field directly at every site. Every
#   binding of that token must be that field, so a second one is exit 1 naming
#   it. Anything else is exit 1. The parameter is resolved by its declared type,
#   `RoundCfg`, and not by RULE 4's looser "a field of one of its own
#   parameters": a loop that took a second parameter with a `charged` field
#   could otherwise satisfy the rule from the wrong one.
#
#   And the configuration the loop was handed is read-only inside it. RULE 8
#   constrains the token; without this it constrains nothing, because the value
#   can be moved instead, and every way of moving it writes no `charged:` site
#   and passes every other reading. Four spellings, all exit 2: a `let` that
#   rebinds the parameter's own name — the loop building its own configuration
#   and forwarding faithfully from that — an assignment to the whole parameter
#   through its reference, an assignment to one of its fields, and a `mem::`
#   call that swaps it out. In each the decision the entry made is no longer the
#   decision the loop applies, and no reading downstream can see it.
#
#   RULE 8 and the census are complementary and neither alone covers the
#   hard-wire. A forwarded loop that writes `let charge = false;` is caught
#   twice: by RULE 8, because the binding is not the configuration field, and by
#   RULE 3, because a token bound to something other than that field is in the
#   census and the census then reads eight sites. RULE 8 can be edited out of
#   this script; the census cannot, since it is what the gate exists to state.
#   And the census alone names only a count, where RULE 8 names the line.
#
# READING A TOKEN, AND READING A BINDING
#   A token is a bare identifier, or a field access `<ident>.<ident>` closed by
#   `,`, `}` or the end of the line. Nothing else: `cfg.charged()`,
#   `cfg.charged.into()`, `cfg.charged as bool` and `!cfg.charged` are each `?`,
#   and a `?` is exit 2. The delimiter anchor is what stops the reader turning a
#   call, a cast or a negation into the field it resembles, which is precisely
#   how a decision gets inverted under a spelling the census still counts as
#   forwarded.
#
#   Bindings are recorded per binding, not per function. The only readable form
#   is a body-level `let <token> = <rhs>;` whose statement closes on its own
#   line. `let mut`, a type annotation, a right-hand side spanning more than one
#   line, a destructure, and a binding introduced by `if let`, `while let` or a
#   `match` arm are each a shape this gate cannot read: exit 2, reported rather
#   than skipped. A fn whose sites name a bare identifier it never saw bound is
#   exit 1 — the token means something the scan never read. A token bound twice
#   in one fn is exit 1 for a forwarded loop, where RULE 8 has something exact
#   to say about the second binding, and exit 2 for a classic loop and for an
#   entry, where the same-token reading has become vacuous: with a shadow, the
#   token at the `rollback_round` argument and the token in the `charged:` field
#   can be two different values under one spelling, and nothing in this scan can
#   say which binding governs which site.
#
# EXIT
#   0 clean, 1 a rule fired, 2 the gate could not scan — a missing tree, no
#   round loop found, a driver with no charge site in it (a rollback delegated
#   to a helper is not a rollback this gate can read), a driver that reaches
#   past the shared rollback to the low-level pair, a driver that does not reach
#   the one round emit exactly once, the phase target named outside
#   `round_stats.rs`, a rollback outside the populations that looks like a lost
#   loop, a configuration written to inside the loop it was handed to, or a
#   call, a field or a binding whose shape it could not read back. A scan that
#   finds nothing must not report a pass.
#
# PORTABILITY
#   The argument list is joined on `\x1f` inside awk, which BSD awk on macOS
#   reads and which is the awk `make ci` runs here. Measured under gawk 5.4.1
#   and mawk 1.3.4 as well: both give the same verdict on the tree and on every
#   case of the recall test. That is a measurement and not a guarantee — a
#   needle that reaches for an extension one of the three does not have would
#   break it silently, so a new one is worth running under all three.

set -uo pipefail

# The only variable here: the fixtures point the scan at a synthetic root. The
# rules themselves are constants — a gate whose expectations can be relaxed from
# the environment is a gate that passes for whoever sets them.
root="${SPEC_CHARGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
loops_dir="$root/crates/rmlx-models/src/speculative"

readonly WANT_CENSUS="charge_phases:3 false:4"
readonly WANT_SITES=7

fail=0
scan_error=0

note() { printf '%s\n' "$*" >&2; }

if [ ! -d "$loops_dir" ]; then
  note "check-spec-charge: no speculative source directory at ${loops_dir#"$root"/}"
  exit 2
fi

# awk walks each file once and emits one record per function, tab-separated:
#
#   1 file          6 charged: sites      11 RoundCtx parameter
#   2 fn            7 low-level rollbacks 12 constructs RoundCfg
#   3 driver sig    8 round emits         13 writes the configuration
#   4 RoundTotals   9 phase-target names  14 binding of the fn's token
#   5 rollbacks    10 RoundCfg parameter  15 parameter names
#                                         16 tokens, space-joined
#
# A token of `?` is a site whose value this scan could not read back — reported
# as a scan error rather than skipped, because an unread site is exactly the one
# that would carry the other decision.
#
# A function opens at a `fn` item and closes when its brace depth returns to
# zero. A nested `fn` inside a body is not opened: the enclosing loop is the
# unit the rules are about.
# The two text readers this gate shares with the other source-scanning gates:
# `decomment`, used everywhere below, and `blank_strings`, which this gate does
# NOT use — RULE 7 reads the per-round event's target as a literal, and blanking
# string bodies would make that rule unfireable.
# shellcheck source=lib/awk_text.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/awk_text.sh"

records=$(
  find "$loops_dir" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
    xargs -0 awk "$AWK_TEXT_FNS"'
      # A value this gate can read: a bare identifier, or one field access.
      # Anything else is `?` — a call, a cast, a negation or a chain is not the
      # field it resembles.
      function classify(v) {
        if (v ~ /^[A-Za-z_][A-Za-z0-9_]*$/) { return v }
        if (v ~ /^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$/) { return v }
        return "?"
      }
      function addtok(t) { if (!(t in toks)) { toks[t] = 1 } }
      function reset() {
        in_sig = 0; awaiting_body = 0; in_body = 0; in_call = 0; fname = ""
        depth = 0; paren = 0; args = ""; has_step = 0; has_stats = 0
        nroll = 0; nrec = 0; nlow = 0; nemit = 0; ntarget = 0
        cfg_param = ""; cfg_seen = 0; cfg_dup = 0; makes_cfg = 0; cfg_written = 0
        ctx_param = ""; ctx_seen = 0; ctx_dup = 0
        delete toks; delete bcount; delete bfirst; delete blast; delete bodd
        delete params
      }
      # Every parameter name in a signature line. A `name:` followed by a second
      # colon is a path segment and not a parameter.
      function collect_params(s,   rest, tok, ch) {
        rest = s
        while (match(rest, /[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:/)) {
          tok = substr(rest, RSTART, RLENGTH)
          ch = substr(rest, RSTART + RLENGTH, 1)
          rest = substr(rest, RSTART + RLENGTH)
          if (ch == ":") { continue }
          sub(/[[:space:]]*:$/, "", tok)
          if (tok != "") { params[tok] = 1 }
        }
      }
      # Every identifier a statement may bind, when the statement is not one
      # this gate can read. Uppercase-initial names are types and paths.
      function mark_odd(s,   lhs, rest, tok) {
        lhs = s
        sub(/=[^=].*$/, "", lhs)
        rest = lhs
        while (match(rest, /[A-Za-z_][A-Za-z0-9_]*/)) {
          tok = substr(rest, RSTART, RLENGTH)
          rest = substr(rest, RSTART + RLENGTH)
          if (tok ~ /^[A-Z]/) { continue }
          if (tok == "let" || tok == "if" || tok == "while" || tok == "mut" ||
              tok == "ref" || tok == "match") { continue }
          bodd[tok] = 1
        }
      }
      function record_binding(s,   name, rhs) {
        sub(/^[[:space:]]+/, "", s)
        if (s ~ /^let[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=[[:space:]]*[^;]+;[[:space:]]*$/) {
          name = s
          sub(/^let[[:space:]]+/, "", name)
          sub(/[[:space:]]*=.*$/, "", name)
          rhs = s
          sub(/^[^=]*=[[:space:]]*/, "", rhs)
          sub(/[[:space:]]*;[[:space:]]*$/, "", rhs)
          bcount[name]++
          if (bcount[name] == 1) { bfirst[name] = rhs }
          blast[name] = rhs
        } else {
          mark_odd(s)
        }
      }
      # What this fn`s one name-token is bound to. Literals and field accesses
      # need no binding; a fn naming more than one token fails RULE 1 first.
      function bindinfo(   t, name, cnt) {
        name = ""; cnt = 0
        for (t in toks) {
          if (t == "false" || t == "true" || t == "?") { continue }
          if (t ~ /\./) { continue }
          name = t; cnt++
        }
        if (cnt == 0) { return "-" }
        if (cnt > 1) { return "mixed" }
        if (name in bodd) { return "odd" }
        if (!(name in bcount)) { return "none" }
        if (bcount[name] > 1) { return "multi:" blast[name] }
        return "ok:" bfirst[name]
      }
      # A parameter resolved by its declared type: its name, `-` when the type
      # is not in the list, `?` when the list carries it and this scan cannot
      # say which parameter it is — unnamed, destructured, or declared twice,
      # where a rule that picked one would be reading the wrong one.
      function typed_param(seen, name, dup) {
        if (!seen) { return "-" }
        if (dup || name == "") { return "?" }
        return name
      }
      # The parameter whose declared type is `want`, read off a signature line.
      function param_of_type(line, want,   cand) {
        cand = line
        if (cand !~ ("[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:[[:space:]]*&?[[:space:]]*(mut[[:space:]]+)?" want)) { return "" }
        sub("[[:space:]]*:[[:space:]]*&?[[:space:]]*(mut[[:space:]]+)?" want ".*$", "", cand)
        sub(/^.*[^A-Za-z0-9_]/, "", cand)
        if (cand ~ /^[A-Za-z_][A-Za-z0-9_]*$/) { return cand }
        return ""
      }
      function flush(   t, joined, plist) {
        if (fname != "") {
          joined = ""
          for (t in toks) { joined = (joined == "") ? t : joined " " t }
          plist = ""
          for (t in params) { plist = (plist == "") ? t : plist " " t }
          printf "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%d\t%d\t%s\t%s\t%s\n", \
            FILENAME, fname, has_step, has_stats, nroll, nrec, nlow, nemit, \
            ntarget, typed_param(cfg_seen, cfg_param, cfg_dup), \
            typed_param(ctx_seen, ctx_param, ctx_dup), \
            makes_cfg, cfg_written, bindinfo(), (plist == "" ? "-" : plist), joined
        }
        reset()
      }
      # The `charge` argument of a call whose arguments are one per line:
      # caches, lin, round tokens, pre-round offset, target, charge, device.
      function close_call(   n) {
        n = split(args, a, "\x1f")
        addtok(n == 7 ? classify(a[6]) : "?")
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
        # Read the code and not the comment beside it, here as everywhere else:
        # a `// not a RoundCfg` in a parameter list would otherwise name a type
        # the fn does not take, and an unbalanced parenthesis inside a comment
        # would end the parameter list early.
        sig = decomment($0)
        if (index(sig, "step_fn: &mut dyn FnMut(&ProbeStep)") > 0) { has_step = 1 }
        collect_params(sig)
        # The two types resolved by declaration and not by the name of any
        # binding. A parameter list that destructures one, or declares it twice,
        # leaves this scan unable to say which parameter a rule is about — a
        # shape it refuses rather than guesses at.
        if (index(sig, "RoundCfg") > 0) {
          cfg_seen = 1
          cand = param_of_type(sig, "RoundCfg")
          if (cand != "") { if (cfg_param != "") { cfg_dup = 1 } else { cfg_param = cand } }
        }
        if (index(sig, "RoundCtx") > 0) {
          ctx_seen = 1
          cand = param_of_type(sig, "RoundCtx")
          if (cand != "") { if (ctx_param != "") { ctx_dup = 1 } else { ctx_param = cand } }
        }
        # A per-parameter `) -> ` is inside the list and must not end the scan.
        paren += gsub(/\(/, "(", sig) - gsub(/\)/, ")", sig)
        if (paren <= 0) {
          in_sig = 0
          awaiting_body = 1
          # A `where` clause sits between the closing parenthesis and the body,
          # so the body opens at the next `{` and not necessarily here.
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
        if (index(head, "{") > 0) {
          awaiting_body = 0
          in_body = 1
          depth = gsub(/\{/, "{", head) - gsub(/\}/, "}", head)
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
      (index(code, "RoundCfg {") > 0 || index(code, "RoundCfg::") > 0) { makes_cfg = 1 }

      # RULE 8, the read-only half: the configuration handed over is neither
      # rebound under its own name nor written through. Four spellings, because
      # the rule constrains the token and the value can be moved instead — a
      # rebinding, an assignment to the whole parameter through its reference, an
      # assignment to one of its fields, and a `mem::` call that swaps it out.
      cfg_param != "" &&
      (code ~ ("^[[:space:]]*let[[:space:]]+(mut[[:space:]]+)?" cfg_param "[[:space:]]*[=:]") ||
       code ~ ("^[[:space:]]*\\*?[[:space:]]*" cfg_param "[[:space:]]*=[^=]") ||
       code ~ ("^[[:space:]]*" cfg_param "\\.[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=[^=]") ||
       ((index(code, "mem::replace(") > 0 || index(code, "mem::swap(") > 0 ||
         index(code, "mem::take(") > 0) &&
        code ~ ("[^A-Za-z0-9_]" cfg_param "[^A-Za-z0-9_]"))) {
        cfg_written = 1
      }

      # Every binding, recorded per binding rather than per function: a token
      # bound twice is two values under one spelling, and a scan that kept one
      # verdict per fn would read the good one and pass the loop.
      code ~ /^[[:space:]]*(let[[:space:]]|if[[:space:]]+let[[:space:]]|while[[:space:]]+let[[:space:]])/ {
        record_binding(code)
      }
      # A match arm binds in its pattern, which is a shape this gate cannot read.
      code ~ /=>/ && code !~ /^[[:space:]]*(let[[:space:]]|if[[:space:]]+let[[:space:]])/ {
        arm = code
        sub(/=>.*$/, "", arm)
        if (arm ~ /[({]/) { mark_odd(arm) }
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
      # whose value this gate cannot read is unreadable rather than absent.
      # A trailing comma is not required — a record`s last field carries none.
      index(code, "charged:") > 0 {
        tok = code
        sub(/^.*charged:[[:space:]]*/, "", tok)
        if (tok ~ /^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?[[:space:]]*(,|\}|$)/) {
          sub(/[[:space:]]*(,|\}).*$/, "", tok)
          sub(/[[:space:]]+$/, "", tok)
          addtok(classify(tok))
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

# (a) round loops: the driver signature and a `RoundTotals`.
loops=$(printf '%s\n' "$records" | awk -F'\t' '$3 == 1 && $4 == 1')
# (b) entries: the signature, no totals, no rollback, and it builds the config.
entries=$(printf '%s\n' "$records" |
  awk -F'\t' '$3 == 1 && $4 == 0 && $5 == 0 && $12 == 1')
# RULE 4: a rollback outside (a). A fn that also drives a generation or names a
# `charged:` field is a loop the derivation lost, not a drafter's own rollback.
lost=$(printf '%s\n' "$records" |
  awk -F'\t' '!($3 == 1 && $4 == 1) && $5 > 0 && ($3 == 1 || $6 > 0)')
drafter_rollbacks=$(printf '%s\n' "$records" |
  awk -F'\t' '!($3 == 1 && $4 == 1) && $5 > 0 && $3 == 0 && $6 == 0')
# RULE 5: a `charged:` in a fn that is in neither population.
strays=$(printf '%s\n' "$records" |
  awk -F'\t' '$6 > 0 && !($3 == 1 && $4 == 1) &&
              !($3 == 1 && $4 == 0 && $5 == 0 && $12 == 1)')
# RULE 6: the low-level rollback named outside `round_common.rs`, or named
# inside it by a round loop, which the file's exemption is not for.
lowlevel=$(printf '%s\n' "$records" |
  awk -v rc="$loops_dir/round_common.rs" -F'\t' '$7 > 0 && ($1 != rc || ($3 == 1 && $4 == 1))')

# RULE 7 by name, by file: the phase target belongs to `round_stats.rs`.
# Anchored to that one path, like RULE 6's exemption, so the same basename under
# another directory is read like every other file. This reading is the one that
# sees a mention outside any fn — a module-level `const` the per-fn scan below
# cannot open.
target_strays=$(
  find "$loops_dir" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
    xargs -0 awk "$AWK_TEXT_FNS"'
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
  awk -v rs="$loops_dir/round_stats.rs" -F'\t' '$9 > 0 && $1 == rs && $3 == 1 && $4 == 1')

if [ -z "$loops" ]; then
  note "check-spec-charge: found no round loop under ${loops_dir#"$root"/}."
  note "  A round loop is a fn taking \`step_fn: &mut dyn FnMut(&ProbeStep)\` that builds"
  note "  a \`RoundTotals\`. Either both were renamed out from under this gate or the scan"
  note "  is broken; a gate that matched nothing must not report a pass."
  exit 2
fi

classic_count=0
forwarded_count=0
entry_count=0
forwarded_names=""
census=""

# `charge_token_note <rel> <fn> <bind>` — the two verdicts a bare-identifier
# token gets from its bindings, shared by classic loops and entries.
report_shadow() {
  note "check-spec-charge: $1: \`$2\` binds its charge token more than once."
  note "  With a shadow, the token at the \`rollback_round\` argument and the token in"
  note "  the \`charged:\` field can be two different values under one spelling, and"
  note "  nothing in this scan can say which binding governs which site."
}

while IFS=$'\t' read -r file fn _sig _stats _nroll _nrec _nlow _nemit _ntarget _cfg _ctx _mk _wr _bind _params _tokens; do
  [ -n "$fn" ] || continue
  note "check-spec-charge: ${file#"$root"/}: \`$fn\` rolls a round's caches back and is"
  note "  not one of the round loops this gate derived. Either the derivation lost a"
  note "  loop — a \`RoundTotals\` built by a constructor, a signature rustfmt wrapped, a"
  note "  \`where\` clause — or a rollback moved out of one. Both read as a census that"
  note "  moved, and editing the census would bury either."
  scan_error=1
done <<<"$lost"

while IFS=$'\t' read -r file fn _sig _stats _nroll _nrec _nlow _nemit ntarget _cfg _ctx _mk _wr _bind _params _tokens; do
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

while IFS=$'\t' read -r file fn _sig _stats _nroll _nrec nlow _nemit _ntarget _cfg _ctx _mk _wr _bind _params _tokens; do
  [ -n "$fn" ] || continue
  note "check-spec-charge: ${file#"$root"/}: \`$fn\` makes $nlow call(s) to the low-level"
  note "  rollback beneath \`rollback_round\`. Those take a \`charge\` of their own at a call"
  note "  this gate does not open, so the decision it reads at a loop's own site would no"
  note "  longer be the only one made. Only \`round_common.rs\` may name them, and only"
  note "  outside a round loop."
  scan_error=1
done <<<"$lowlevel"

while IFS=$'\t' read -r file fn _sig _stats _nroll nrec _nlow _nemit _ntarget _cfg _ctx _mk _wr _bind _params tokens; do
  [ -n "$fn" ] || continue
  note "check-spec-charge: ${file#"$root"/}: \`$fn\` writes $nrec \`charged:\` field(s) — $tokens —"
  note "  and is neither a round loop nor an entry that builds one's configuration, so"
  note "  nothing beside it says whether that is the decision a loop ordered. The one fn"
  note "  that may write the field is the shared recorder, which carries it across as a"
  note "  destructured \`charged,\` and names no value here."
  fail=1
done <<<"$strays"

# RULE 4's new arm: a drafter rolling its own state back carries the decision it
# was handed, as a field of one of its own parameters.
while IFS=$'\t' read -r file fn _sig _stats _nroll _nrec _nlow _nemit _ntarget _cfg ctx _mk _wr _bind params tokens; do
  [ -n "$fn" ] || continue
  rel="${file#"$root"/}"
  count=$(printf '%s\n' "$tokens" | tr ' ' '\n' | grep -c '[^[:space:]]')
  if printf '%s\n' "$tokens" | tr ' ' '\n' | grep -qx '?' || [ "$count" != "1" ]; then
    note "check-spec-charge: $rel: \`$fn\` rolls a round back on a \`charge\` this gate"
    note "  could not read back — $tokens. An unread site is not a checked site, and it"
    note "  is the one that would carry the other decision."
    scan_error=1
    continue
  fi
  if [ "$ctx" = "?" ]; then
    note "check-spec-charge: $rel: \`$fn\` takes a \`RoundCtx\` this gate could not name —"
    note "  a destructured parameter, one declared twice, or a shape it does not read. The"
    note "  rule resolves the context by its declared type, so a parameter it cannot name"
    note "  is a parameter it cannot hold this rollback's charge to."
    scan_error=1
    continue
  fi
  owner="${tokens%%.*}"
  if [ "$ctx" != "-" ]; then
    # The round's context is the one thing that carries the loop's decision
    # down, so where a fn declares one, the charge comes off it and off nothing
    # else. Any other parameter with a `charged`-shaped field would otherwise
    # satisfy the looser reading below while carrying a different decision.
    if [ "$tokens" != "$ctx.charged" ]; then
      note "check-spec-charge: $rel: \`$fn\` is handed the round's context in \`$ctx\` and"
      note "  rolls back on \`$tokens\`. A drafter's rollback charges what the loop handed"
      note "  it — \`$ctx.charged\` — and a second parameter with a field of the same name is"
      note "  a different decision under a spelling this gate would otherwise accept."
      fail=1
    fi
    continue
  fi
  if [ "$owner" = "$tokens" ] ||
    ! printf '%s\n' "$params" | tr ' ' '\n' | grep -qx -- "$owner"; then
    note "check-spec-charge: $rel: \`$fn\` rolls a round back on \`$tokens\`, which is not a"
    note "  field of one of its own parameters. A drafter's rollback carries the decision"
    note "  the loop handed it; a literal or a fresh \`phases_charged()\` here is a second"
    note "  decision made where nothing can hold it to the loop that ordered it."
    fail=1
  fi
done <<<"$drafter_rollbacks"

# Population (b): the entries.
while IFS=$'\t' read -r file fn _sig _stats _nroll nrec _nlow _nemit _ntarget _cfg _ctx _mk _wr bind _params tokens; do
  [ -n "$fn" ] || continue
  rel="${file#"$root"/}"
  entry_count=$((entry_count + 1))

  if [ "$nrec" != "1" ]; then
    note "check-spec-charge: $rel: \`$fn\` builds a round loop's configuration and states"
    note "  $nrec \`charged:\` field(s). An entry names the decision once, at the call that"
    note "  runs the loop: none is a decision dropped on the way in, and two are two"
    note "  decisions with nothing saying which one the loop applies."
    fail=1
    continue
  fi
  if printf '%s\n' "$tokens" | tr ' ' '\n' | grep -qx '?'; then
    note "check-spec-charge: $rel: \`$fn\` has a charge site this gate could not read"
    note "  the value of. An unread site is not a checked site, and it is the one that"
    note "  would carry the other decision."
    scan_error=1
    continue
  fi
  if [ "$bind" = "odd" ]; then
    note "check-spec-charge: $rel: \`$fn\` binds \`$tokens\` in a shape this gate cannot"
    note "  read — a \`mut\`, a type annotation, a destructure, or a right-hand side that"
    note "  does not end on its own line. An unread binding is not a checked one."
    scan_error=1
    continue
  fi
  if [ "${bind%%:*}" = "multi" ]; then
    report_shadow "$rel" "$fn"
    scan_error=1
    continue
  fi
  if [ "$bind" = "none" ]; then
    note "check-spec-charge: $rel: \`$fn\` names \`$tokens\` as its decision and never binds"
    note "  it. The token means something this scan never saw, so the census would be"
    note "  reading a name rather than a decision."
    fail=1
    continue
  fi
  if [ "$tokens" = "charge_phases" ] &&
    ! printf '%s' "$bind" | grep -Eq '^ok:([A-Za-z_0-9]+::)*phases_charged\(\)$'; then
    note "check-spec-charge: $rel: \`$fn\` charges on \`charge_phases\` and does not bind it"
    note "  to \`phases_charged()\` alone. The census would then be reading a spelling: an"
    note "  entry that binds the name to a literal, or ORs a second condition into the"
    note "  call, charges on requests the switch did not ask for and is counted here"
    note "  among the three that ask."
    fail=1
    continue
  fi
  census="$census$tokens"$'\n'
done <<<"$entries"

# Population (a): the round loops, classic and forwarded.
while IFS=$'\t' read -r file fn _sig _stats nroll nrec _nlow nemit _ntarget cfg _ctx _mk cfg_written bind _params tokens; do
  [ -n "$fn" ] || continue
  rel="${file#"$root"/}"
  if [ "$cfg" != "-" ]; then
    forwarded_count=$((forwarded_count + 1))
    forwarded_names="$forwarded_names \`$fn\`"
  else
    classic_count=$((classic_count + 1))
  fi

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
  if [ "$bind" = "odd" ]; then
    note "check-spec-charge: $rel: \`$fn\` binds \`$tokens\` in a shape this gate cannot"
    note "  read — a \`mut\`, a type annotation, a destructure, or a right-hand side that"
    note "  does not end on its own line. An unread binding is not a checked one, and"
    note "  refusing it as a rule about the loop would name the wrong defect."
    scan_error=1
    continue
  fi

  if [ "$cfg" != "-" ]; then
    # RULE 8: the forwarded decision is the one that was handed over.
    if [ "$cfg" = "?" ]; then
      note "check-spec-charge: $rel: \`$fn\` takes a \`RoundCfg\` this gate could not name —"
      note "  a destructured parameter, or a shape it does not read. The rule resolves the"
      note "  configuration by its declared type, so a parameter it cannot name is a"
      note "  parameter it cannot hold the loop's token to."
      scan_error=1
      continue
    fi
    if [ "$cfg_written" = "1" ]; then
      note "check-spec-charge: $rel: \`$fn\` rebinds or writes \`$cfg\`, the configuration it"
      note "  was handed. It is read-only inside the loop: a loop that builds its own and"
      note "  forwards faithfully from that writes no site any reading here can see, and"
      note "  the decision the entry made is no longer the decision the loop applies."
      scan_error=1
      continue
    fi
    if [ "${bind%%:*}" = "multi" ]; then
      note "check-spec-charge: $rel: \`$fn\` binds its charge token twice; the last binds it"
      note "  to \`${bind#multi:}\`. A forwarded loop binds its token to \`$cfg.charged\` and to"
      note "  nothing else — a second binding is a second decision under one spelling."
      fail=1
      continue
    fi
    if [ "$tokens" != "$cfg.charged" ] && [ "$bind" != "ok:$cfg.charged" ]; then
      note "check-spec-charge: $rel: \`$fn\` is handed its charge in \`$cfg\` and applies"
      note "  \`$tokens\` instead. A forwarded loop names \`$cfg.charged\` at every site, or"
      note "  binds its token to exactly that field: anything else is the loop deciding"
      note "  for itself what the entry already decided, and the census cannot see it as"
      note "  the field it was handed."
      fail=1
      # It decided for itself, so its decision is one the census reads.
      census="$census$tokens"$'\n'
      continue
    fi
    if [ "$nemit" != "1" ]; then
      note "check-spec-charge: $rel: \`$fn\` reaches the one round emit $nemit time(s)."
      note "  Every round loop closes its round through \`log_round\`, once."
      scan_error=1
      continue
    fi
    # A forwarded loop naming the field it was handed contributes no decision.
    continue
  fi

  if [ "${bind%%:*}" = "multi" ]; then
    report_shadow "$rel" "$fn"
    scan_error=1
    continue
  fi
  if [ "$bind" = "none" ]; then
    note "check-spec-charge: $rel: \`$fn\` names \`$tokens\` as its decision and never binds"
    note "  it. The token means something this scan never saw, so the census would be"
    note "  reading a name rather than a decision."
    fail=1
    continue
  fi
  if [ "$tokens" = "charge_phases" ] &&
    ! printf '%s' "$bind" | grep -Eq '^ok:([A-Za-z_0-9]+::)*phases_charged\(\)$'; then
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
done <<<"$loops"

if [ "$scan_error" = "1" ]; then
  exit 2
fi

have_census=$(printf '%s' "$census" | grep -v '^$' | LC_ALL=C sort | uniq -c |
  awk '{ printf "%s:%s ", $2, $1 }' | sed 's/ $//')
have_sites=$(printf '%s' "$census" | grep -c '[^[:space:]]')

if [ "$have_census" != "$WANT_CENSUS" ]; then
  note "check-spec-charge: the charge census is \"$have_census\" ($have_sites sites) and the"
  note "  tree records \"$WANT_CENSUS\" ($WANT_SITES sites)."
  note "  Three loops time their phases and charge them; four time none and charge"
  note "  none. A loop moved between the two, a value wired in, or an entry that"
  note "  dropped its decision, changes what the phase timings mean and is written"
  note "  down here rather than merged."
  fail=1
fi

if [ "$fail" != "0" ]; then
  exit 1
fi

echo "OK: $classic_count classic, $forwarded_count forwarded, $entry_count entries; census $have_census ($have_sites sites).${forwarded_names:+ Forwarded:$forwarded_names.}"
