#!/usr/bin/env bash
# CI gate: speculative round loops carry one phase-charge decision and the request's sampler.
#
# One population of Rust fns, read off the speculative source once, and two rule sets over it.
# Nothing is a name list; every member is derived from its signature and body.
#
#   LOOPS   — a fn whose parameters include `step_fn: &mut dyn FnMut(&ProbeStep)`, at any
#             visibility (that argument is what makes a fn a generation driver), that also
#             constructs a `RoundTotals`, the numbers a round loop closes on. The second
#             condition separates a loop from the fns that satisfy the first and run no round:
#             `spec_generate_greedy`, which validates a request and delegates, `emit_step`, the
#             shared per-token emit helper, and `emit_seed_token`, which is handed a loop's
#             totals and names none. A loop is FORWARDED when its parameter list carries
#             `RoundCfg`, the configuration type that holds the charge field and the request's
#             sampler, and CLASSIC otherwise. Membership is read off the signature and never
#             off a binding: a loop that hard-wires its charge still takes the configuration,
#             so it is still forwarded and the rules about forwarded loops still reach it.
#   ENTRIES — the driver signature, no `RoundTotals`, no `rollback_round`, and it constructs
#             a `RoundCfg`: the fn that starts the one shared loop. The last conjunct is the
#             discriminator and not a detail: without it the population admits the entry guard
#             and the emit helpers, none of which decides anything. An entry that stops
#             constructing the configuration leaves the population, which is what makes a
#             dropped decision read as a lost entry rather than as a clean scan.
#
# A `fn` inside a `trait` or an `impl` block is read exactly like a free one, and a `fn` with
# no body, which is what a trait declares, is closed at its own `;`. A fn opens at a `fn` item
# and closes when its brace depth returns to zero; a nested `fn` is not opened, because the
# enclosing loop is the unit the rules are about. Files named `*_tests.rs` and `tests.rs` are
# not read.
#
# charge: each round loop names one phase-charge decision, everywhere the decision is read.
#   Each of the seven entries decides once whether its request's phases are charged for the
#   work they issue, `phases_charged()` in three and a literal `false` in four, and hands it
#   to the one round loop, which repeats that decision in two unrelated places: the `charge`
#   argument of every `rollback_round(...)` it makes, and the `charged:` field of every
#   `RoundReport` it logs and of the `RoundTotals` it hands the one recorder. Nothing else
#   holds the two together. A loop whose rollback charges and whose record says it did not
#   moves the phase timings and re-attributes the work to the drafter, with every token and
#   every count identical, and no other observable can see it: the equivalence pairs read the
#   answer, the accept counters read the aggregate, and the per-round event stream reads
#   `charged` as `false` on both sides of any comparison by construction, because a capture
#   that enabled the switch would be measuring a different, slower run than the one that ships.
#
# sampling: every speculative generation path is handed the request's sampler and draws with it.
#   A round loop that takes no sampler decodes greedily. It does not fail, it does not warn,
#   and its output is fluent, so a request that asked for a temperature gets an answer that
#   looks exactly like the one it wanted and is drawn from a different distribution. A harness
#   that sets a temperature and serves through such a loop records a greedy number under a
#   sampled label. The distributional test (`spec_sampled_distribution`) proves a loop samples
#   correctly and needs two model snapshots and a Metal context, so it runs one pair under
#   `make gpu-test`. This mode runs everywhere and covers every path: it cannot tell a right
#   distribution from a wrong one, and it can tell that a path was never given the chance to
#   draw from either. Its population is the one above plus the GUARDS.
#
# list-drivers: prints the LOOPS, one `<file>\t<fn>\t<line>` line each, file relative to the
#   root, sorted, after the same "found no round loop" refusal the charge mode makes. It is
#   the population's one producer for `report/debt_report.py`'s driver group. The line number is
#   part of the join key a caller resolves each entry with: file and fn name alone collide when
#   two same-named fns live in one file.
#
# Each mode reads the code of a line, never the comment beside it, through the readers in
# lib/awk_text.sh. Sampling also blanks the body of string literals, so a needle inside text a
# program merely prints is no construction; charge keeps them, because it reads the per-round
# event's target as a literal. Measured under BSD awk, gawk 5.4.1 and mawk 1.3.4 with the same
# verdicts: a needle that reaches for an extension one of them lacks would break it silently,
# so run a new one under all three.
#
# LOC-exempt: one population read by two rule sets that share one scanner; the rule text is the spec
# Usage:  bash scripts/check/spec_round_loops.sh <charge|sampling|list-drivers> [--root DIR]
# Exit:   0 clean; 1 a rule fired; 2 the gate could not scan (a usage error, a missing tree or
#         dispatch, no round loop found, a body or call or binding it could not read back, a
#         drafter-path count that moved, a failed awk scan)
# Not a mode of: check_spec_metric_parity.sh — that gate compares the metric names a bench
#                script prints with the engine's list; this one reads the round loops of the
#                speculative source for two rules that are not about metric names

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/awk_text.sh
. "$REPO_ROOT/scripts/lib/awk_text.sh"

usage() {
    echo "usage: bash scripts/check/spec_round_loops.sh <charge|sampling|list-drivers> [--root DIR]" >&2
    exit 2
}

MODE="${1:-}"
case "$MODE" in
charge | list-drivers) LABEL=check-spec-charge ;;
sampling) LABEL=check-spec-sampling ;;
*) usage ;;
esac
shift

ROOT_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
    --root)
        [ $# -ge 2 ] && [ -n "$2" ] && [ -z "$ROOT_ARG" ] || usage
        ROOT_ARG="$2"
        shift 2
        ;;
    *) usage ;;
    esac
done
SCAN_ROOT="${ROOT_ARG:-$REPO_ROOT}"
LOOPS_DIR="$SCAN_ROOT/crates/rmlx-models/src/speculative"
DISPATCH="$SCAN_ROOT/crates/rmlx-server/src/engine/speculative.rs"

readonly WANT_CENSUS="charge_phases:3 false:4"
readonly WANT_SITES=7
readonly WANT_PATHS=7

fail=0
scan_error=0

note() { printf '%s\n' "$*" >&2; }

if [ ! -d "$LOOPS_DIR" ]; then
    note "$LABEL: no speculative source directory at ${LOOPS_DIR#"$SCAN_ROOT"/}"
    exit 2
fi
if [ "$MODE" = sampling ] && [ ! -f "$DISPATCH" ]; then
    note "$LABEL: no speculative dispatch at ${DISPATCH#"$SCAN_ROOT"/}"
    exit 2
fi

# ---- The scanner -------------------------------------------------------------
#
# One awk program walks each file once and writes one tab-separated record per fn. The core
# below finds the fn items, reads their signatures and extents, and records the population
# facts both modes need. A hook set supplies what differs: the view of a line, what a mode
# reads in a signature and in a body, and the rest of the record. Every record starts with
#
#   1 file     2 fn        3 driver signature   4 RoundTotals
#   5 rollback_round call  6 constructs RoundCfg  7 RoundCfg parameter  8 fn start line
#
# A hook never writes a record of its own: a body this scan cannot read back is reported
# rather than skipped. The awk text holds no apostrophe; it sits in single quotes.

SCAN_CORE='
      function reset() {
        in_sig = 0; awaiting_body = 0; in_body = 0; fname = ""; fn_line = 0
        depth = 0; paren = 0
        has_step = 0; has_stats = 0; has_roll = 0; makes_cfg = 0; cfg_seen = 0
        hook_reset()
      }
      function flush() {
        if (fname != "") { hook_emit() }
        reset()
      }
      function common_fields() {
        return sprintf("%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d", FILENAME, fname, has_step, has_stats, has_roll, makes_cfg, cfg_seen, fn_line)
      }
      function open_body(s) {
        awaiting_body = 0
        in_body = 1
        depth = gsub(/\{/, "{", s) - gsub(/\}/, "}", s)
        if (depth < 0) { depth = 0 }
      }
      FNR == 1 { flush() }

      !in_body && !in_sig && !awaiting_body &&
      /^[[:space:]]*(pub(\([a-z:]+\))?[[:space:]]+)?(async[[:space:]]+)?fn[[:space:]]+[a-z_0-9]+/ {
        flush()
        line = $0
        sub(/^.*fn[[:space:]]+/, "", line)
        sub(/[^A-Za-z0-9_].*$/, "", line)
        fname = line
        fn_line = FNR
        in_sig = 1
        paren = 0
      }

      in_sig {
        # A commented-out parameter is not a parameter, and a parenthesis inside a
        # comment must not end the list.
        sig = decomment($0)
        if (index(sig, "step_fn: &mut dyn FnMut(&ProbeStep)") > 0) { has_step = 1 }
        if (index(sig, "RoundCfg") > 0) { cfg_seen = 1 }
        hook_sig(sig)
        # A per-parameter `) -> ` is inside the list and must not end the scan.
        paren += gsub(/\(/, "(", sig) - gsub(/\)/, ")", sig)
        if (paren <= 0) {
          # A declaration with no body, a trait method, ends here. Waiting for the next
          # open brace would read the following fn as this one and record neither.
          if (sig_is_bodiless(sig)) { flush(); next }
          in_sig = 0
          awaiting_body = 1
          if (index(sig, "{") > 0) { open_body(sig) }
        }
        next
      }

      # A `where` clause sits between the closing parenthesis and the body, and can
      # carry the `;` that ends a bodiless declaration.
      awaiting_body {
        head = decomment($0)
        if (sig_is_bodiless(head)) { flush(); next }
        if (index(head, "{") > 0) { open_body(head) }
        next
      }

      !in_body { next }

      # The extent of a body is read off the code too: a brace inside a comment would
      # close the record early, and every needle after it would belong to no fn.
      {
        code = view($0)
        if (index(code, "RoundTotals {") > 0) { has_stats = 1 }
        if (index(code, "RoundCfg {") > 0) { makes_cfg = 1 }
        if (index(code, "rollback_round(") > 0 && code !~ /fn[[:space:]]+rollback_round\(/) { has_roll = 1 }
        if (!hook_line(code)) {
          depth += gsub(/\{/, "{", code) - gsub(/\}/, "}", code)
          if (depth <= 0) { flush() }
        }
      }
      END { hook_end(); flush() }
'

# Charge hooks. The extra fields, after the common eight:
#
#   9 rollback_round calls   10 charged: sites      11 low-level rollbacks   12 round emits
#  13 phase-target names     14 RoundCfg parameter  15 how it is handed over 16 RoundCtx parameter
#  17 writes the configuration  18 binding of the fn token  19 parameter names  20 tokens
#
# A token of `?` is a site whose value this scan could not read back, reported as a scan error
# rather than skipped, because an unread site is exactly the one that would carry the other
# decision.
CHARGE_HOOKS='
      function view(s) { return decomment(s) }
      # A value this gate can read: a bare identifier, or one field access. Anything
      # else is `?`: a call, a cast, a negation or a chain is not the field it resembles.
      function classify(v) {
        if (v ~ /^[A-Za-z_][A-Za-z0-9_]*$/) { return v }
        if (v ~ /^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$/) { return v }
        return "?"
      }
      function addtok(t) { if (!(t in toks)) { toks[t] = 1 } }
      function hook_reset() {
        in_call = 0; args = ""
        nroll = 0; nrec = 0; nlow = 0; nemit = 0; ntarget = 0
        cfg_param = ""; cfg_dup = 0; cfg_form = "-"; cfg_written = 0
        ctx_param = ""; ctx_seen = 0; ctx_dup = 0
        delete toks; delete bcount; delete bfirst; delete blast; delete bodd
        delete params
      }
      # Every parameter name in a signature line. A `name:` followed by a second colon
      # is a path segment and not a parameter.
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
      # Every identifier a statement may bind, when the statement is not one this gate
      # can read. Uppercase-initial names are types and paths.
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
      # What the one name-token of this fn is bound to. Literals and field accesses need
      # no binding; a fn naming more than one token fails the one-decision rule first.
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
      # A parameter resolved by its declared type: its name, `-` when the type is not in
      # the list, `?` when the list carries it and this scan cannot say which parameter
      # it is (unnamed, destructured, or declared twice), where a rule that picked one
      # would be reading the wrong one.
      function typed_param(seen, name, dup) {
        if (!seen) { return "-" }
        if (dup || name == "") { return "?" }
        return name
      }
      # The parameter whose declared type is `want`, read off a signature line, with the
      # form it is handed over in: `ref`, `mut` or `value`. A loop that cannot write the
      # configuration cannot move the value out from under the token this gate reads.
      function param_of_type(line, want,   cand, rest) {
        cand = line
        if (cand !~ ("[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:[[:space:]]*&?[[:space:]]*(mut[[:space:]]+)?" want)) { return "" }
        sub("[[:space:]]*:[[:space:]]*&?[[:space:]]*(mut[[:space:]]+)?" want ".*$", "", cand)
        sub(/^.*[^A-Za-z0-9_]/, "", cand)
        if (cand !~ /^[A-Za-z_][A-Za-z0-9_]*$/) { return "" }
        rest = line
        sub("^.*" cand "[[:space:]]*:[[:space:]]*", "", rest)
        if (rest ~ ("^&[[:space:]]*mut[[:space:]]+" want)) { return cand "\x1f" "mut" }
        if (rest ~ ("^&[[:space:]]*" want)) { return cand "\x1f" "ref" }
        return cand "\x1f" "value"
      }
      function hook_sig(sig,   cand, pf) {
        collect_params(sig)
        # The two types resolved by declaration and not by the name of any binding. A
        # parameter list that destructures one, or declares it twice, leaves this scan
        # unable to say which parameter a rule is about, and it refuses rather than guesses.
        if (index(sig, "RoundCfg") > 0) {
          cand = param_of_type(sig, "RoundCfg")
          if (cand != "") {
            split(cand, pf, "\x1f")
            if (cfg_param != "") { cfg_dup = 1 } else { cfg_param = pf[1]; cfg_form = pf[2] }
          }
        }
        if (index(sig, "RoundCtx") > 0) {
          ctx_seen = 1
          cand = param_of_type(sig, "RoundCtx")
          if (cand != "") {
            split(cand, pf, "\x1f")
            if (ctx_param != "") { ctx_dup = 1 } else { ctx_param = pf[1] }
          }
        }
      }
      # The charge argument of a call whose arguments are one per line: caches, lin,
      # round tokens, pre-round offset, target, charge, device.
      function close_call(   n, a) {
        n = split(args, a, "\x1f")
        addtok(n == 7 ? classify(a[6]) : "?")
        nroll++
        in_call = 0; args = ""
      }
      function hook_line(code,   arg, tok, rest, arm) {
        # An entry may build its configuration through a constructor as well as a literal;
        # the sampling rules read the literal alone, because they look inside it.
        if (index(code, "RoundCfg::") > 0) { makes_cfg = 1 }
        if (in_call) {
          if (code ~ /^[[:space:]]*\)/) {
            close_call()
          } else if (code ~ /[^[:space:]]/) {
            arg = code
            gsub(/^[[:space:]]+|[[:space:]]*,[[:space:]]*$/, "", arg)
            args = (args == "") ? arg : args "\x1f" arg
          }
          return 1
        }

        # The configuration handed over is neither rebound under its own name nor
        # written through. Four spellings, because the rule constrains the token and the
        # value can be moved instead: a rebinding, an assignment to the whole parameter
        # through its reference, an assignment to one of its fields, and a `mem::` call
        # that swaps it out.
        if (cfg_param != "" &&
            (code ~ ("^[[:space:]]*let[[:space:]]+(mut[[:space:]]+)?" cfg_param "[[:space:]]*[=:]") ||
             code ~ ("^[[:space:]]*\\*?[[:space:]]*" cfg_param "[[:space:]]*=[^=]") ||
             code ~ ("^[[:space:]]*" cfg_param "\\.[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=[^=]") ||
             ((index(code, "mem::replace(") > 0 || index(code, "mem::swap(") > 0 ||
               index(code, "mem::take(") > 0) &&
              code ~ ("[^A-Za-z0-9_]" cfg_param "[^A-Za-z0-9_]")))) {
          cfg_written = 1
        }

        # Every binding, recorded per binding rather than per fn: a token bound twice is
        # two values under one spelling, and a scan that kept one verdict per fn would
        # read the good one and pass the loop.
        if (code ~ /^[[:space:]]*(let[[:space:]]|if[[:space:]]+let[[:space:]]|while[[:space:]]+let[[:space:]])/) {
          record_binding(code)
        }
        # A match arm binds in its pattern, which is a shape this gate cannot read.
        if (code ~ /=>/ && code !~ /^[[:space:]]*(let[[:space:]]|if[[:space:]]+let[[:space:]])/) {
          arm = code
          sub(/=>.*$/, "", arm)
          if (arm ~ /[({]/) { mark_odd(arm) }
        }

        # Only a call opens one: the definition carries no charge argument, and a call
        # written on one line is a shape this scan does not read back.
        # `rollback_round_caches(` does not contain this needle, so the shared helper
        # forwarding its own `charge` is not a site.
        if (index(code, "rollback_round(") > 0 && code !~ /fn[[:space:]]+rollback_round\(/) {
          rest = code
          sub(/^.*rollback_round\(/, "", rest)
          if (rest ~ /[^[:space:]]/) { addtok("?"); nroll++ } else { in_call = 1; args = "" }
        }

        # The low-level pair the shared rollback is built on. Their own definitions are
        # not calls, and neither is `rollback_round_caches` inside the shared rollback.
        if ((index(code, "rollback_round_caches(") > 0 ||
             index(code, "refold_lin_tapes(") > 0) &&
            code !~ /fn[[:space:]]+(rollback_round_caches|refold_lin_tapes)[[:space:]]*\(/) {
          nlow++
        }

        # The one round emit: a call at a statement position, not the definition. Read
        # off the code, so a trailing comment naming it is not a call, and anchored, so
        # the count is of calls a loop makes rather than of the name appearing on a line.
        if (code ~ /^[[:space:]]*(let[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=[[:space:]]*)?([A-Za-z_][A-Za-z0-9_]*::)*log_round\(/) {
          nemit++
        }

        # The per-round event target, per fn. The file-level reading cannot see which fn
        # a mention sits in, and the file that owns the target has to be allowed to name
        # it, so this is what denies a round loop that moved into that file its exemption.
        if (index(code, "PHASE_TARGET") > 0 || index(code, "rmlx::spec::phase") > 0) {
          ntarget++
        }

        # Any `charged:` is a site, and one whose value this gate cannot read is
        # unreadable rather than absent. A trailing comma is not required: the last
        # field of a record carries none.
        if (index(code, "charged:") > 0) {
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
        return 0
      }
      function hook_end() { }
      function hook_emit(   t, joined, plist) {
        joined = ""
        for (t in toks) { joined = (joined == "") ? t : joined " " t }
        plist = ""
        for (t in params) { plist = (plist == "") ? t : plist " " t }
        printf "%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n", \
          common_fields(), nroll, nrec, nlow, nemit, ntarget, \
          typed_param(cfg_seen, cfg_param, cfg_dup), cfg_form, \
          typed_param(ctx_seen, ctx_param, ctx_dup), \
          cfg_written, bindinfo(), (plist == "" ? "-" : plist), joined
      }
'

# Sampling hooks, pass 1. The extra fields, after the common eight:
#
#   9 a `sampler_cfg: &...` parameter   10 a draw built from the sampler
#  11 a RoundCfg literal carrying it     12 an unreadable body
SAMPLING_HOOKS='
      function view(s) { return blank_strings(decomment(s)) }
      function hook_reset() {
        cfg_depth = 0; draw_depth = 0
        has_param = 0; draws = 0; cfg_carries = 0; unreadable = 0
      }
      function hook_sig(sig) {
        if (sig ~ /sampler_cfg:[[:space:]]*&/) { has_param = 1 }
      }
      function hook_line(code,   rest) {
        # The draw a loop uses, built from the sampler of the request: one construction,
        # the VerifierDraw of the request, which the shared acceptance walk and the stochastic
        # one both draw through. It is followed to its closing parenthesis, so a wrapped
        # argument list reads like a single-line one; a gate that refused a correct loop
        # for the width of its line would be repaired by widening the line.
        if (draw_depth == 0 && index(code, "VerifierDraw::new(") > 0) {
          rest = code
          sub(/^.*VerifierDraw::new\(/, "", rest)
          draw_depth = 1 + gsub(/\(/, "(", rest) - gsub(/\)/, ")", rest)
          if (index(rest, "sampler_cfg") > 0) { draws = 1 }
          if (draw_depth < 0) { draw_depth = 0 }
          return 0
        }
        if (draw_depth > 0) {
          if (index(code, "sampler_cfg") > 0) { draws = 1 }
          draw_depth += gsub(/\(/, "(", code) - gsub(/\)/, ")", code)
          if (draw_depth < 0) { draw_depth = 0 }
        }

        # The configuration an entry builds, and whether the sampler is in it. The
        # literal is followed to its closing brace, so a sampler_cfg further down the
        # body is not read as one the entry handed over.
        if (index(code, "RoundCfg {") > 0) {
          rest = code
          sub(/^.*RoundCfg[[:space:]]*\{/, "", rest)
          cfg_depth = 1 + gsub(/\{/, "{", rest) - gsub(/\}/, "}", rest)
          if (index(rest, "sampler_cfg") > 0) { cfg_carries = 1 }
          if (cfg_depth <= 0) { cfg_depth = 0 }
          return 0
        }
        if (cfg_depth > 0) {
          if (index(code, "sampler_cfg") > 0) { cfg_carries = 1 }
          cfg_depth += gsub(/\{/, "{", code) - gsub(/\}/, "}", code)
          if (cfg_depth < 0) { cfg_depth = 0 }
        }
        return 0
      }
      function hook_end() {
        if (in_sig || awaiting_body) { unreadable = 1 }
      }
      function hook_emit() {
        printf "%s\t%d\t%d\t%d\t%d\n", common_fields(), has_param, draws, cfg_carries, unreadable
      }
'

# Sampling hooks, pass 2: every call to a member of the population, and whether the request
# sampler goes with it. The names cross on spaces because BSD awk cannot take a newline in a
# -v value; they are Rust identifiers, which hold neither. Record: file, fn, calls made,
# calls without the sampler, the first such callee.
GUARD_HOOKS='
      BEGIN { n = split(names, list, " "); for (i = 1; i <= n; i++) { if (list[i] != "") { pop[list[i]] = 1 } } }
      function view(s) { return blank_strings(decomment(s)) }
      function hook_reset() {
        ncalls = 0; nmissing = 0; missing = "-"
        call_depth = 0; callee = ""; carries = 0
      }
      function close_call() {
        ncalls++
        if (!carries) { nmissing++; if (missing == "-") { missing = callee } }
        call_depth = 0; callee = ""; carries = 0
      }
      function hook_sig(sig) { }
      function hook_line(code,   rest, name) {
        if (call_depth > 0) {
          if (index(code, "sampler_cfg") > 0) { carries = 1 }
          call_depth += gsub(/\(/, "(", code) - gsub(/\)/, ")", code)
          if (call_depth <= 0) { close_call() }
          return 0
        }
        rest = code
        while (match(rest, /[A-Za-z_][A-Za-z0-9_]*\(/)) {
          name = substr(rest, RSTART, RLENGTH - 1)
          rest = substr(rest, RSTART + RLENGTH)
          if (!(name in pop) || name == fname) { continue }
          callee = name
          carries = (index(rest, "sampler_cfg") > 0)
          call_depth = 1 + gsub(/\(/, "(", rest) - gsub(/\)/, ")", rest)
          if (call_depth <= 0) { close_call() }
          break
        }
        return 0
      }
      function hook_end() { }
      function hook_emit() {
        printf "%s\t%s\t%d\t%d\t%s\n", FILENAME, fname, ncalls, nmissing, missing
      }
'

# scan_rs <hooks> [awk option]...: the records of every non-test source under the loops directory.
scan_rs() {
    local hooks="$1"
    shift
    find "$LOOPS_DIR" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
        xargs -0 awk "$@" "$AWK_TEXT_FNS$SCAN_CORE$hooks"
}

scan_failed() {
    note "$LABEL: the awk scan failed."
    exit 2
}

# count_tokens <space-joined tokens>: how many there are.
count_tokens() {
    printf '%s\n' "$1" | tr ' ' '\n' | grep -c '[^[:space:]]' || true
}

# has_unread_token <space-joined tokens>: true when one is the unreadable `?`.
has_unread_token() {
    printf '%s\n' "$1" | tr ' ' '\n' | grep -x '?' >/dev/null
}

# ---- charge ------------------------------------------------------------------
#
# Three populations, the first two shared:
#
#   (a) ROUND LOOPS, classic and forwarded.
#   (b) ENTRIES.
#   (c) DRAFTER ROLLBACKS. A fn outside (a) that calls `rollback_round` and neither drives a
#       generation nor names a `charged:` field of its own. Those last two are what separate
#       the rollback of a drafter from a round loop the derivation lost: a fn that carries the
#       driver signature, or states a decision in a record, looks like a loop and is refused
#       as one (exit 2) rather than read under the argument rule of RULE 4.
#
# RULE 1 (one decision per loop)
#   Within one round loop, the `charge` argument of every `rollback_round` call and the value
#   of every `charged:` field must be the same token. The record is assembled elsewhere now,
#   so the field the loop still writes is the `charged:` of the `RoundTotals` it hands over:
#   the decision is named at the call site either way, which is the only place this gate can
#   read it. What the shared recorder then does with the value is past its reach.
#
# RULE 2 (the token means what it says)
#   A classic loop or an entry whose token is `charge_phases` must bind it, in the same fn,
#   to exactly `phases_charged()` and nothing else, the whole right-hand side and not a
#   prefix of it. Without that the census reads a spelling: a loop that binds the name to a
#   literal, or ORs a second condition into the call, charges on requests the switch did not
#   ask for, and the census would count it among the three that ask.
#
# RULE 3 (the census)
#   Across every charge token that is not the forwarded configuration field, whether a
#   classic loop, an entry, or a forwarded loop that stopped being the field it was handed,
#   the multiset of tokens is exactly `charge_phases` three times and `false` four times. It
#   is here so that a loop moved from one schedule to the other, a `true` wired in, or an
#   entry that quietly dropped its decision, has to be written down rather than merged. A
#   forwarded loop that names or binds the field it was handed contributes nothing; the same
#   loop hard-wiring a literal contributes one and the census reads eight sites. The census
#   is a constant, not read from the environment: a gate whose expectation can be relaxed
#   from there passes for whoever sets it.
#
# RULE 4 (nothing charges outside the populations)
#   A fn that is not a round loop and calls `rollback_round` is either a loop the derivation
#   lost or a drafter rolling its own state back. The first reads as a census that moved,
#   which invites the census to be edited, and is exit 2. The second is population (c), and
#   its `charge` argument carries the decision handed down from the loop that ordered it.
#   Where the fn is handed the round context, a parameter whose declared type is `RoundCtx`,
#   the charge is `<ctx>.charged` and nothing else; where it is not, any field of any of its
#   own parameters will do. The strict arm closes the hole in the loose one: a second
#   parameter carrying a field of the same name satisfies "a field of one of its own
#   parameters" while holding a different decision. A literal or a `phases_charged()` is a
#   second decision made where nothing can hold it to that loop, and is exit 1; an argument
#   this scan cannot read back is exit 2.
#
#   `rollback_round` is the shared refold-or-disarm, and it is not itself a site: it forwards
#   the `charge` its caller named down to the low-level `rollback_round_caches`, whose name
#   this needle does not match. The decision stays at the call site of the caller, where it can
#   be read against what was handed over, and the shared code carries it rather than naming
#   one.
#
# RULE 5 (the decision is named once, and only where it can be checked)
#   A fn that is in neither population and carries a `charged:` field names a decision this
#   gate cannot check against a rollback, because there is none in it. The shared recorder
#   carries the field across from a destructured binding, `charged,`, which is not a
#   `charged:` site, so a `charged:` in such a fn is a decision made where nothing can hold
#   it to the loop that ordered it, and is exit 1. Over entries the same needle becomes a
#   rule rather than a membership test: an entry states exactly one `charged:`. Zero or two
#   is exit 1 naming the entry, and a field this scan cannot read back is exit 2. The reach
#   is fn bodies and only those: a `charged:` in a module-level `const` or `static` is
#   outside it, and a decision parked in a constant is a shape review has to catch. A shared
#   constructor for the totals of a loop is refused by this rule, and that is the intent: the
#   token has to stay at the call site of the loop for the rollback beside it to be checked
#   against.
#
# RULE 6 (the low-level rollback is named in one file, and nowhere else)
#   `rollback_round` is the one rollback a round loop calls, and the one whose `charge`
#   argument RULE 1 reads. The low-level pair beneath it, `rollback_round_caches` and
#   `refold_lin_tapes`, also take a `charge`, at a call this reader does not open. Anything
#   that named one of those would make a second charge decision beside the one a loop
#   declares. Two readings, because either alone has a hole. By file: a call from anywhere
#   but `round_common.rs` is exit 2, whether from a round loop, a helper the loop calls, or a
#   fn nothing calls. By fn: a round loop is read wherever it lives, so a loop that moves into
#   `round_common.rs` does not inherit the exemption of the file. The exemption is anchored to that
#   one path and not to a basename. Both fns are private to `round_common` in the tree, so
#   this is a second reader on a property the compiler already holds, which cannot see a
#   source edit that has not been built.
#
# RULE 7 (one seam for the round event, and one file that can name its target)
#   Every round loop closes its round through `log_round`, the one emitter, and inside the
#   engine source this scan covers, the target that emitter writes on is named in
#   `round_stats.rs` alone. A loop that keeps its charge decision honest and writes its own
#   `tracing::debug!` on that target, with the same fields in the same order, produces a
#   byte-identical line: the pinned round-stream digests agree, the equivalence pairs agree,
#   and this census agrees. What it silently drops is the carry check of the charged round
#   and the guarantee that a field added to the shared record reaches that loop. The test harness
#   restates the target literal outside this scan; `the_engine_and_its_readers_state_the_same_target_and_round_fields`
#   in `tests/spec_greedy_equivalence.rs` holds that copy to the engine one.
#
#   Two readings, both needed. By name: `PHASE_TARGET` or the literal `rmlx::spec::phase`
#   outside `round_stats.rs` is exit 2, read by file (the only reading that sees a mention
#   outside any fn) and by fn (so a round loop that moved into `round_stats.rs` does not
#   inherit that file exemption and write a second event beside its emit). By call: each
#   round loop calls `log_round(` exactly once, at a statement position. Zero is a loop that
#   left the seam; two or more is a second record shape under one target.
#
#   A loop whose emit moves into the skeleton moves this needle in the same commit, as RULE
#   6 needle moved when the rollback went behind `rollback_round`, and the old seam has to
#   become unreachable from the loops, or the gate has lost a defect class rather than
#   followed it.
#
# RULE 8 (the forwarded decision is the one that was handed over)
#   A forwarded loop binds its charge token, in the same fn, to exactly the configuration
#   field it was handed, `<cfg>.charged`, the whole right-hand side and not a prefix, or
#   names that field directly at every site. Every binding of that token must be that
#   field, so a second one is exit 1 naming it. The parameter is resolved by its declared
#   type, `RoundCfg`, and not by the looser reading of RULE 4: a loop that took a second parameter
#   with a `charged` field could otherwise satisfy the rule from the wrong one.
#
#   The configuration the loop was handed is read-only inside it. RULE 8 constrains the
#   token; without this it constrains nothing, because the value can be moved instead. The
#   rule is therefore on the parameter and not on the lines: the configuration is handed over
#   by `&RoundCfg`, and a loop taking it by `&mut` or by value is exit 2, since a loop that
#   cannot write it cannot move it whatever the spelling. Four line shapes are still read as
#   defence in depth: a `let` that rebinds the parameter name, an assignment to the whole
#   parameter through its reference, an assignment to one of its fields, and a `mem::` call
#   that swaps it out.
#
#   RULE 8 and the census are complementary and neither alone covers the hard-wire. A
#   forwarded loop that writes `let charge = false;` is caught twice: by RULE 8, because the
#   binding is not the configuration field, and by RULE 3, because a token bound to something
#   other than that field is in the census and the census then reads eight sites. RULE 8 can
#   be edited out of this script; the census cannot, since it is what the gate exists to
#   state. And the census alone names only a count, where RULE 8 names the line.
#
# READING A TOKEN, AND READING A BINDING
#   A token is a bare identifier, or a field access `<ident>.<ident>` closed by `,`, `}` or
#   the end of the line. Nothing else: `cfg.charged()`, `cfg.charged.into()`,
#   `cfg.charged as bool` and `!cfg.charged` are each `?`, and a `?` is exit 2. The delimiter
#   anchor stops the reader turning a call, a cast or a negation into the field it resembles,
#   which is how a decision gets inverted under a spelling the census still counts as
#   forwarded.
#
#   Bindings are recorded per binding, not per fn. The only readable form is a body-level
#   `let <token> = <rhs>;` whose statement closes on its own line. `let mut`, a type
#   annotation, a right-hand side spanning more than one line, a destructure, and a binding
#   introduced by `if let`, `while let` or a `match` arm are each a shape this gate cannot
#   read: exit 2, reported rather than skipped. A fn whose sites name a bare identifier it
#   never saw bound is exit 1: the token means something the scan never read. A token bound
#   twice in one fn is exit 1 for a forwarded loop, where RULE 8 has something exact to say
#   about the second binding, and exit 2 for a classic loop and for an entry, where the
#   same-token reading has become vacuous: with a shadow, the token at the `rollback_round`
#   argument and the token in the `charged:` field can be two different values under one
#   spelling.
#
# EXIT
#   0 clean, 1 a rule fired, 2 the gate could not scan: a missing tree, no round loop found,
#   a driver with no charge site in it (a rollback delegated to a helper is not a rollback
#   this gate can read), a driver that reaches past the shared rollback to the low-level
#   pair, a driver that does not reach the one round emit exactly once, the phase target
#   named outside `round_stats.rs`, a rollback outside the populations that looks like a
#   lost loop, a configuration written to inside the loop it was handed to, or a call, a
#   field or a binding whose shape it could not read back.

# report_shadow <rel> <fn>: the verdict a bare-identifier token gets from two bindings.
report_shadow() {
    note "$LABEL: $1: \`$2\` binds its charge token more than once."
    note "  With a shadow, the token at the \`rollback_round\` argument and the token in"
    note "  the \`charged:\` field can be two different values under one spelling, and"
    note "  nothing in this scan can say which binding governs which site."
}

run_charge() {
    local records lost target_loops target_strays lowlevel strays drafter_rollbacks
    local classic_count=0 forwarded_count=0 entry_count=0 forwarded_names="" census=""
    local file fn nroll nrec nlow nemit ntarget cfg cfg_form ctx cfg_written bind params tokens
    local rel count owner have_census have_sites

    # Columns after the common eight: 9 nroll, 10 nrec, 11 nlow, 12 nemit, 13 ntarget,
    # 14 cfg, 15 cfg_form, 16 ctx, 17 cfg_written, 18 bind, 19 params, 20 tokens.
    records=$(scan_rs "$CHARGE_HOOKS") || scan_failed
    derive_population "$records"

    # RULE 4: a rollback outside (a). A fn that also drives a generation or names a
    # `charged:` field is a loop the derivation lost, not a drafter's own rollback.
    lost=$(printf '%s\n' "$records" |
        awk -F'\t' '!($3 == 1 && $4 == 1) && $9 > 0 && ($3 == 1 || $10 > 0)')
    drafter_rollbacks=$(printf '%s\n' "$records" |
        awk -F'\t' '!($3 == 1 && $4 == 1) && $9 > 0 && $3 == 0 && $10 == 0')
    # RULE 5: a `charged:` in a fn that is in neither population.
    strays=$(printf '%s\n' "$records" |
        awk -F'\t' '$10 > 0 && !($3 == 1 && $4 == 1) &&
                    !($3 == 1 && $4 == 0 && $5 == 0 && $6 == 1)')
    # RULE 6: the low-level rollback named outside `round_common.rs`, or named inside it
    # by a round loop, which the file's exemption is not for.
    lowlevel=$(printf '%s\n' "$records" |
        awk -v rc="$LOOPS_DIR/round_common.rs" -F'\t' '$11 > 0 && ($1 != rc || ($3 == 1 && $4 == 1))')

    # RULE 7 by name, by file: the phase target belongs to `round_stats.rs`. Anchored to
    # that one path, like RULE 6's exemption, so the same basename under another directory
    # is read like every other file. This reading is the one that sees a mention outside
    # any fn, such as a module-level `const`.
    target_strays=$(
        find "$LOOPS_DIR" -name '*.rs' ! -name '*_tests.rs' ! -name 'tests.rs' -print0 |
            xargs -0 awk "$AWK_TEXT_FNS"'
              { c = decomment($0) }
              (index(c, "PHASE_TARGET") > 0 || index(c, "rmlx::spec::phase") > 0) &&
              !(FILENAME in seen) { seen[FILENAME] = 1; print FILENAME }
            ' |
            { grep -v -x -F "$LOOPS_DIR/round_stats.rs" || true; }
    )
    # RULE 7 by name, by fn: a round loop that moved into `round_stats.rs` does not
    # inherit that file's exemption.
    target_loops=$(printf '%s\n' "$records" |
        awk -v rs="$LOOPS_DIR/round_stats.rs" -F'\t' '$13 > 0 && $1 == rs && $3 == 1 && $4 == 1')

    while IFS=$'\t' read -r file fn _; do
        [ -n "$fn" ] || continue
        note "$LABEL: ${file#"$SCAN_ROOT"/}: \`$fn\` rolls a round's caches back and is"
        note "  not one of the round loops this gate derived. Either the derivation lost a"
        note "  loop — a \`RoundTotals\` built by a constructor, a signature rustfmt wrapped, a"
        note "  \`where\` clause — or a rollback moved out of one. Both read as a census that"
        note "  moved, and editing the census would bury either."
        scan_error=1
    done <<<"$lost"

    while IFS=$'\t' read -r file fn _ _ _ _ _ _ _ _ _ _ ntarget _; do
        [ -n "$fn" ] || continue
        note "$LABEL: ${file#"$SCAN_ROOT"/}: \`$fn\` is a round loop and names the"
        note "  per-round event's target $ntarget time(s). The file that owns the target may"
        note "  name it; a loop that moved into that file does not inherit the exemption, or"
        note "  it could call the emit once and write a second event on the target beside it."
        scan_error=1
    done <<<"$target_loops"

    while IFS= read -r file; do
        [ -n "$file" ] || continue
        note "$LABEL: ${file#"$SCAN_ROOT"/} names the per-round event's target."
        note "  That target is \`round_stats.rs\`'s alone: a fn that can name it can write a"
        note "  second round event beside the one \`log_round\` writes, with the same fields"
        note "  in the same order — and a line that agrees byte for byte is invisible to the"
        note "  pinned digests, to the equivalence pairs and to this census."
        scan_error=1
    done <<<"$target_strays"

    while IFS=$'\t' read -r file fn _ _ _ _ _ _ _ _ nlow _; do
        [ -n "$fn" ] || continue
        note "$LABEL: ${file#"$SCAN_ROOT"/}: \`$fn\` makes $nlow call(s) to the low-level"
        note "  rollback beneath \`rollback_round\`. Those take a \`charge\` of their own at a call"
        note "  this gate does not open, so the decision it reads at a loop's own site would no"
        note "  longer be the only one made. Only \`round_common.rs\` may name them, and only"
        note "  outside a round loop."
        scan_error=1
    done <<<"$lowlevel"

    while IFS=$'\t' read -r file fn _ _ _ _ _ _ _ nrec _ _ _ _ _ _ _ _ _ tokens; do
        [ -n "$fn" ] || continue
        note "$LABEL: ${file#"$SCAN_ROOT"/}: \`$fn\` writes $nrec \`charged:\` field(s) — $tokens —"
        note "  and is neither a round loop nor an entry that builds one's configuration, so"
        note "  nothing beside it says whether that is the decision a loop ordered. The one fn"
        note "  that may write the field is the shared recorder, which carries it across as a"
        note "  destructured \`charged,\` and names no value here."
        fail=1
    done <<<"$strays"

    # RULE 4's second arm: a drafter rolling its own state back carries the decision it
    # was handed, as a field of one of its own parameters.
    while IFS=$'\t' read -r file fn _ _ _ _ _ _ _ _ _ _ _ _ _ ctx _ _ params tokens; do
        [ -n "$fn" ] || continue
        rel="${file#"$SCAN_ROOT"/}"
        count=$(count_tokens "$tokens")
        if has_unread_token "$tokens" || [ "$count" != "1" ]; then
            note "$LABEL: $rel: \`$fn\` rolls a round back on a \`charge\` this gate"
            note "  could not read back — $tokens. An unread site is not a checked site, and it"
            note "  is the one that would carry the other decision."
            scan_error=1
            continue
        fi
        if [ "$ctx" = "?" ]; then
            note "$LABEL: $rel: \`$fn\` takes a \`RoundCtx\` this gate could not name —"
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
                note "$LABEL: $rel: \`$fn\` is handed the round's context in \`$ctx\` and"
                note "  rolls back on \`$tokens\`. A drafter's rollback charges what the loop handed"
                note "  it — \`$ctx.charged\` — and a second parameter with a field of the same name is"
                note "  a different decision under a spelling this gate would otherwise accept."
                fail=1
            fi
            continue
        fi
        if [ "$owner" = "$tokens" ] ||
            ! printf '%s\n' "$params" | tr ' ' '\n' | grep -x -- "$owner" >/dev/null; then
            note "$LABEL: $rel: \`$fn\` rolls a round back on \`$tokens\`, which is not a"
            note "  field of one of its own parameters. A drafter's rollback carries the decision"
            note "  the loop handed it; a literal or a fresh \`phases_charged()\` here is a second"
            note "  decision made where nothing can hold it to the loop that ordered it."
            fail=1
        fi
    done <<<"$drafter_rollbacks"

    # Population (b): the entries.
    while IFS=$'\t' read -r file fn _ _ _ _ _ _ _ nrec _ _ _ _ _ _ _ bind _ tokens; do
        [ -n "$fn" ] || continue
        rel="${file#"$SCAN_ROOT"/}"
        entry_count=$((entry_count + 1))

        if [ "$nrec" != "1" ]; then
            note "$LABEL: $rel: \`$fn\` builds a round loop's configuration and states"
            note "  $nrec \`charged:\` field(s). An entry names the decision once, at the call that"
            note "  runs the loop: none is a decision dropped on the way in, and two are two"
            note "  decisions with nothing saying which one the loop applies."
            fail=1
            continue
        fi
        if has_unread_token "$tokens"; then
            note "$LABEL: $rel: \`$fn\` has a charge site this gate could not read"
            note "  the value of. An unread site is not a checked site, and it is the one that"
            note "  would carry the other decision."
            scan_error=1
            continue
        fi
        if [ "$bind" = "odd" ]; then
            note "$LABEL: $rel: \`$fn\` binds \`$tokens\` in a shape this gate cannot"
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
            note "$LABEL: $rel: \`$fn\` names \`$tokens\` as its decision and never binds"
            note "  it. The token means something this scan never saw, so the census would be"
            note "  reading a name rather than a decision."
            fail=1
            continue
        fi
        if [ "$tokens" = "charge_phases" ] &&
            ! grep -Eq '^ok:([A-Za-z_0-9]+::)*phases_charged\(\)$' <<<"$bind"; then
            note "$LABEL: $rel: \`$fn\` charges on \`charge_phases\` and does not bind it"
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
    while IFS=$'\t' read -r file fn _ _ _ _ _ _ nroll nrec _ nemit _ cfg cfg_form _ cfg_written bind _ tokens; do
        [ -n "$fn" ] || continue
        rel="${file#"$SCAN_ROOT"/}"
        if [ "$cfg" != "-" ]; then
            forwarded_count=$((forwarded_count + 1))
            forwarded_names="$forwarded_names \`$fn\`"
        else
            classic_count=$((classic_count + 1))
        fi

        if has_unread_token "$tokens"; then
            note "$LABEL: $rel: \`$fn\` has a charge site this gate could not read"
            note "  the value of. An unread site is not a checked site, and it is the one that"
            note "  would carry the other decision."
            scan_error=1
            continue
        fi
        if [ "$nroll" = "0" ] || [ "$nrec" = "0" ]; then
            note "$LABEL: $rel: \`$fn\` has $nroll rollback and $nrec record charge sites."
            note "  A loop is checked only where both are visible: a rollback delegated to a"
            note "  helper, or a record built somewhere else, is a decision this gate cannot"
            note "  follow, and a loop it cannot follow is not a loop it checks."
            scan_error=1
            continue
        fi

        count=$(count_tokens "$tokens")
        if [ "$count" != "1" ]; then
            note "$LABEL: $rel: \`$fn\` names more than one charge decision — $tokens"
            note "  The rollback's \`charge\` argument and the record's \`charged:\` field are read"
            note "  by different code and must be the same token: a round that charged and"
            note "  recorded that it did not re-attributes its own work with nothing saying so."
            fail=1
            continue
        fi
        if [ "$bind" = "odd" ]; then
            note "$LABEL: $rel: \`$fn\` binds \`$tokens\` in a shape this gate cannot"
            note "  read — a \`mut\`, a type annotation, a destructure, or a right-hand side that"
            note "  does not end on its own line. An unread binding is not a checked one, and"
            note "  refusing it as a rule about the loop would name the wrong defect."
            scan_error=1
            continue
        fi

        if [ "$cfg" != "-" ]; then
            # RULE 8: the forwarded decision is the one that was handed over.
            if [ "$cfg" = "?" ]; then
                note "$LABEL: $rel: \`$fn\` takes a \`RoundCfg\` this gate could not name —"
                note "  a destructured parameter, or a shape it does not read. The rule resolves the"
                note "  configuration by its declared type, so a parameter it cannot name is a"
                note "  parameter it cannot hold the loop's token to."
                scan_error=1
                continue
            fi
            if [ "$cfg_form" != "ref" ]; then
                note "$LABEL: $rel: \`$fn\` takes its \`RoundCfg\` by \`&mut\` (or by"
                note "  value). The configuration is the entry's decision, and a loop that can write"
                note "  it can move the value out from under the token this gate reads — through the"
                note "  reference, through a \`mem::\` call, through a method — without writing a"
                note "  \`charged:\` site any reading here can see. It is handed over by \`&RoundCfg\`."
                scan_error=1
                continue
            fi
            if [ "$cfg_written" = "1" ]; then
                note "$LABEL: $rel: \`$fn\` rebinds or writes \`$cfg\`, the configuration it"
                note "  was handed. It is read-only inside the loop: a loop that builds its own and"
                note "  forwards faithfully from that writes no site any reading here can see, and"
                note "  the decision the entry made is no longer the decision the loop applies."
                scan_error=1
                continue
            fi
            if [ "${bind%%:*}" = "multi" ]; then
                note "$LABEL: $rel: \`$fn\` binds its charge token twice; the last binds it"
                note "  to \`${bind#multi:}\`. A forwarded loop binds its token to \`$cfg.charged\` and to"
                note "  nothing else — a second binding is a second decision under one spelling."
                fail=1
                continue
            fi
            if [ "$tokens" != "$cfg.charged" ] && [ "$bind" != "ok:$cfg.charged" ]; then
                note "$LABEL: $rel: \`$fn\` is handed its charge in \`$cfg\` and applies"
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
                note "$LABEL: $rel: \`$fn\` reaches the one round emit $nemit time(s)."
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
            note "$LABEL: $rel: \`$fn\` names \`$tokens\` as its decision and never binds"
            note "  it. The token means something this scan never saw, so the census would be"
            note "  reading a name rather than a decision."
            fail=1
            continue
        fi
        if [ "$tokens" = "charge_phases" ] &&
            ! grep -Eq '^ok:([A-Za-z_0-9]+::)*phases_charged\(\)$' <<<"$bind"; then
            note "$LABEL: $rel: \`$fn\` charges on \`charge_phases\` and does not bind it"
            note "  to \`phases_charged()\` alone. The census would then be reading a spelling: a"
            note "  loop that binds the name to a literal, or ORs a second condition into the"
            note "  call, charges on requests the switch did not ask for and is counted here"
            note "  among the three that ask."
            fail=1
            continue
        fi
        if [ "$nemit" != "1" ]; then
            note "$LABEL: $rel: \`$fn\` reaches the one round emit $nemit time(s)."
            note "  Every round loop closes its round through \`log_round\`, once. None is a loop"
            note "  that left the seam and writes its own event, which the digests cannot see"
            note "  when the line agrees; more than one is a second record shape under the one"
            note "  target, which is what the collapse removed."
            scan_error=1
            continue
        fi
        census="$census$tokens"$'\n'
    done <<<"$loops"

    require_forwarded_loop_for_entries "$entry_count" "$forwarded_count" \
        "  An entry builds a \`RoundCfg\` and hands it to the one loop that takes it, so" \
        "  entries with no loop to enter mean the loop was lost — a body this scan could" \
        "  not read back, or a signature it did not open. The census would not see it:" \
        "  the lost loop's decision leaves it exactly as the entry's decision enters it."

    if [ "$scan_error" = "1" ]; then
        exit 2
    fi

    have_census=$(printf '%s' "$census" | { grep -v '^$' || true; } | LC_ALL=C sort | uniq -c |
        awk '{ printf "%s:%s ", $2, $1 }' | sed 's/ $//')
    have_sites=$(printf '%s' "$census" | grep -c '[^[:space:]]' || true)

    if [ "$have_census" != "$WANT_CENSUS" ]; then
        note "$LABEL: the charge census is \"$have_census\" ($have_sites sites) and the"
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
}

# ---- sampling ----------------------------------------------------------------
#
# The population is the one above, at any visibility, plus the GUARDS: a fn that calls a loop
# or an entry and is neither. There is one, the two-model entry guard, which routes a request
# to one of two loops by reading whether the sampler is active. That clause is not tidiness:
# the guard is where a sampled request can be routed to the greedy arm, which is this mode's
# own defect class. Narrowing to the loops alone is the tempting move and it is wrong: at the
# end of the campaign the entries are the only place a request's sampler can be dropped,
# since each takes it and hands it to the loop's configuration.
#
# RULE 1 (a loop is handed the sampler and draws with it)
#   Every loop must (a) declare a `sampler_cfg: &...SamplerConfig` parameter, or take the
#   `RoundCfg` that carries the request's sampler for it, and (b) construct the draw it uses
#   from that sampler: a `VerifierDraw::new(...)` naming `sampler_cfg`, which is why the
#   configuration's field carrying it is named `sampler_cfg` too, so one needle reads
#   `VerifierDraw::new(sampler_cfg)` and `VerifierDraw::new(cfg.sampler_cfg)` alike. (b) is
#   deliberately the construction rather than a mention: a parameter added to satisfy (a) and
#   then ignored is the same defect with the signature repaired, and a sampler assigned into a
#   context field and read by nobody is the same defect with the body repaired.
#
#   One construction, not two. A bare `Pcg32::new(sampler_cfg.seed_or_default())` was read as
#   a draw too while the stochastic acceptance rule seeded its own generator; that rule now
#   draws through the round's `VerifierDraw`, so a second needle would match nothing. It is
#   gone rather than kept: a needle that matches nothing cannot fire, and what it would admit
#   is the shape the collapse removed, a second generator seeded from the request's seed,
#   which is a second stream correlated with the first while looking reproducible on its own.
#   A second generator beside a correct draw is outside this mode, since the construction it
#   requires is present; `the_requests_draw_stream_has_one_generator` in
#   `crates/rmlx-models/src/speculative/round_skeleton_tests.rs` reads it.
#
# RULE 2 (an entry hands the sampler to the loop)
#   An entry declares the parameter and the configuration it builds carries it: the needle is
#   `sampler_cfg` inside the `RoundCfg { ... }` literal, not the name appearing somewhere in
#   the body. A `sampler_cfg` an entry accepts and leaves out of the configuration it hands
#   over is exactly the shape this rule exists to refuse.
#
# RULE 3 (a guard passes it on)
#   A guard declares the parameter and passes it at every call it makes to a loop or an entry
#   that takes one.
#
# RULE 4 (the dispatch)
#   In the server's speculative generator, every arm of the `match &drafter` that drives a
#   generation must pass the resolved sampler configuration. One arm that does not is one
#   drafter kind that decodes greedily, and the other arms passing it is what makes that
#   invisible in review.
#
# NO EXCEPTION
#   This mode names no fn. It carried one while the two-model greedy loop took no sampler at
#   all and ran only at temperature 0; that loop is now an entry whose `verify` draws through
#   the round's context, so every path is read by the rules above and a name-shaped hole would
#   only be somewhere to hide.
#
# THE CENSUS
#   The success line names the three populations, and the count of drafter paths, the loops
#   that are not the shared forwarded one plus the entries, is pinned at seven. That figure is
#   invariant across the campaign by construction: a migrated drafter's loop body becomes its
#   entry, one for one. A scan that finds six paths where the tree has seven has not passed,
#   it has stopped looking, so a count that moves is exit 2 rather than a quieter run.
#
# EXIT
#   0 clean, 1 a rule fired, 2 the gate could not scan: a missing file, no loop found, a body
#   it could not read back, or a drafter-path count that moved.

run_sampling() {
    local records callers population population_flat arms
    local loop_count=0 entry_count=0 path_count=0 forwarded_count=0 guard_count=0 arm_count=0
    local file name has_cfg has_param draws carries unreadable ncalls nmissing missing arm passes rel

    # Columns after the common eight: 9 has_param, 10 draws, 11 carries, 12 unreadable.
    records=$(scan_rs "$SAMPLING_HOOKS") || scan_failed
    derive_population "$records"

    population=$(printf '%s\n%s\n' "$loops" "$entries" | awk -F'\t' '$2 != "" { print $2 }')
    population_flat=$(printf '%s\n' "$population" | tr '\n' ' ')

    callers=$(scan_rs "$GUARD_HOOKS" -v names="$population_flat") || scan_failed

    # Rule 1: the loops.
    while IFS=$'\t' read -r file name _ _ _ _ has_cfg _ has_param draws _ unreadable; do
        [ -n "$name" ] || continue
        rel="${file#"$SCAN_ROOT"/}"
        loop_count=$((loop_count + 1))
        if [ "$has_cfg" = "1" ]; then
            forwarded_count=$((forwarded_count + 1))
        else
            path_count=$((path_count + 1))
        fi
        if [ "$unreadable" = "1" ]; then
            note "$LABEL: $rel: \`$name\` opened a body this gate could not read back."
            note "  A loop whose extent cannot be determined is not scanned, and an unscanned"
            note "  loop is exactly the one that would decode greedily unnoticed."
            scan_error=1
            continue
        fi
        if [ "$has_param" != "1" ] && [ "$has_cfg" != "1" ]; then
            note "$LABEL: $rel: \`$name\` drives a generation but takes neither a"
            note "  \`sampler_cfg: &SamplerConfig\` nor the \`RoundCfg\` that carries one, so every"
            note "  request it serves decodes greedily whatever temperature the caller asked"
            note "  for, and says nothing."
            fail=1
            continue
        fi
        if [ "$draws" != "1" ]; then
            note "$LABEL: $rel: \`$name\` is handed the request's sampler and builds"
            note "  no draw from it. The signature satisfies a caller and the loop still decodes"
            note "  greedily: the draw has to be constructed from the configuration, not the"
            note "  name mentioned somewhere in the body."
            fail=1
        fi
    done <<<"$loops"

    # Rule 2: the entries.
    while IFS=$'\t' read -r file name _ _ _ _ _ _ has_param _ carries unreadable; do
        [ -n "$name" ] || continue
        rel="${file#"$SCAN_ROOT"/}"
        entry_count=$((entry_count + 1))
        path_count=$((path_count + 1))
        if [ "$unreadable" = "1" ]; then
            note "$LABEL: $rel: \`$name\` opened a body this gate could not read back."
            scan_error=1
            continue
        fi
        if [ "$has_param" != "1" ]; then
            note "$LABEL: $rel: \`$name\` starts a round loop and takes no"
            note "  \`sampler_cfg: &SamplerConfig\`, so the request's temperature stops here and"
            note "  the loop it runs draws from a distribution nobody asked for."
            fail=1
            continue
        fi
        if [ "$carries" != "1" ]; then
            note "$LABEL: $rel: \`$name\` takes \`sampler_cfg\` and leaves it out of the"
            note "  configuration it hands the loop. An entry is the last place a request's"
            note "  sampler can be dropped, and a sampler that reaches no loop is the same"
            note "  greedy answer under a sampled label."
            fail=1
        fi
    done <<<"$entries"

    # Rule 3: the guards.
    while IFS=$'\t' read -r file name ncalls nmissing missing; do
        [ -n "$name" ] || continue
        [ "$ncalls" != "0" ] || continue
        if grep -qx -- "$name" <<<"$population"; then
            continue
        fi
        rel="${file#"$SCAN_ROOT"/}"
        guard_count=$((guard_count + 1))
        if [ "$nmissing" != "0" ]; then
            note "$LABEL: $rel: \`$name\` runs \`$missing\` without passing the"
            note "  request's sampler. A guard routes a request to one generation path or"
            note "  another; one route that drops the configuration is one class of request"
            note "  answered greedily while every other route honours it."
            fail=1
        fi
    done <<<"$callers"

    # Rule 4: the dispatch. This scan reads code like every other one here: an arm that
    # keeps the sampler in a comment, or names it inside a string a line merely prints,
    # passes it to nothing.
    arms=$(
        awk "$AWK_TEXT_FNS"'
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
        ' "$DISPATCH"
    ) || scan_failed

    if [ -z "$arms" ]; then
        note "$LABEL: ${DISPATCH#"$SCAN_ROOT"/} has no \`let result = match &drafter {\`"
        note "  dispatch this gate can read. Rule 4 scanned nothing."
        exit 2
    fi

    while IFS=$'\t' read -r arm passes; do
        [ -n "$arm" ] || continue
        arm_count=$((arm_count + 1))
        if [ "$passes" != "1" ]; then
            note "$LABEL: ${DISPATCH#"$SCAN_ROOT"/}: the \`Drafter::$arm\` arm drives a"
            note "  generation without passing \`spec_sampler_cfg\`, so that drafter kind alone"
            note "  decodes greedily while the others honour the request."
            fail=1
        fi
    done <<<"$arms"

    require_forwarded_loop_for_entries "$entry_count" "$forwarded_count" \
        "  An entry hands the request's sampler to the one loop that takes a \`RoundCfg\`," \
        "  so entries with no loop to enter mean the loop was lost. The drafter-path count" \
        "  cannot see that: the lost loop's slot is the one the entry now fills."

    if [ "$scan_error" = "1" ]; then
        exit 2
    fi
    if [ "$path_count" != "$WANT_PATHS" ]; then
        note "$LABEL: the tree has $path_count drafter generation paths and records"
        note "  $WANT_PATHS. A migrated drafter's loop body becomes its entry, one for one, so"
        note "  this figure does not move across the collapse. A scan that finds six paths"
        note "  where the tree has seven has not passed, it has stopped looking."
        exit 2
    fi
    if [ "$fail" = "1" ]; then
        exit 1
    fi

    echo "OK: $loop_count loops ($forwarded_count forwarded), $entry_count entries, $guard_count guards — every one of the $path_count drafter paths carries the request's sampler to the loop that draws with it; $arm_count drafter arms pass it."
}

# ---- the shared population ---------------------------------------------------

# derive_population <records>: sets `loops` and `entries`, and refuses an empty population.
derive_population() {
    loops=$(printf '%s\n' "$1" | awk -F'\t' '$3 == 1 && $4 == 1')
    entries=$(printf '%s\n' "$1" | awk -F'\t' '$3 == 1 && $4 == 0 && $5 == 0 && $6 == 1')
    if [ -n "$loops" ]; then
        return 0
    fi
    note "$LABEL: found no round loop under ${LOOPS_DIR#"$SCAN_ROOT"/}."
    if [ "$MODE" = sampling ]; then
        note "  A loop is a fn taking \`step_fn: &mut dyn FnMut(&ProbeStep)\` that builds a"
        note "  \`RoundTotals\`. Either they were renamed out from under this gate, or the scan"
    else
        note "  A round loop is a fn taking \`step_fn: &mut dyn FnMut(&ProbeStep)\` that builds"
        note "  a \`RoundTotals\`. Either both were renamed out from under this gate or the scan"
    fi
    note "  is broken; a gate that matched nothing must not report a pass."
    exit 2
}

# require_forwarded_loop_for_entries <entries> <forwarded loops> <line>...: an entry exists to
# start the shared loop, so entries without one are a loop the scan lost, and neither mode's
# census sees it: a lost loop's decision or slot is the one the entry fills, one for one, which
# is what a migration does on purpose. The lines explain what the mode cannot see.
require_forwarded_loop_for_entries() {
    local entry_total="$1" forwarded_total="$2" line
    shift 2
    if [ "$entry_total" != "0" ] && [ "$forwarded_total" = "0" ]; then
        note "$LABEL: the scan found $entry_total entries and no forwarded loop."
        for line in "$@"; do
            note "$line"
        done
        scan_error=1
    fi
}

list_drivers() {
    local records listing="" file fn fn_line
    records=$(scan_rs "$CHARGE_HOOKS") || scan_failed
    derive_population "$records"
    while IFS=$'\t' read -r file fn fn_line; do
        [ -n "$fn" ] || continue
        listing="$listing${file#"$SCAN_ROOT"/}"$'\t'"$fn"$'\t'"$fn_line"$'\n'
    done < <(printf '%s\n' "$loops" | awk -F'\t' '{ print $1 "\t" $2 "\t" $8 }')
    printf '%s' "$listing" | grep -v '^$' | LC_ALL=C sort
}

loops=""
entries=""
case "$MODE" in
charge) run_charge ;;
sampling) run_sampling ;;
list-drivers) list_drivers ;;
esac
