#!/usr/bin/env bash
# scripts/lib/awk_text.sh — the two text readers every source-scanning gate
# needs. Sourced, not executed.
#
# WHY ONE PRODUCER
#   A gate that reads Rust source needs the same two answers everywhere: what on
#   this line is code rather than a comment, and what is code rather than the
#   contents of a string literal. A second copy of either drifts, and the way it
#   drifts is silent — a needle that matches a commented-out line reports a call
#   the code no longer makes, and one that matches inside a literal reports a
#   construction that is only being printed.
#
#   `decomment(s)`     — the line with any `//` comment removed. Quote-aware, so
#                        a `//` inside a string literal does not truncate it.
#   `blank_strings(s)` — the line with the body of every string literal replaced
#                        by spaces, the quotes kept so the shape still reads.
#
#   `sig_is_bodiless(s)` — true when a signature line ends the declaration
#                        instead of opening a body: no `{`, and the statement
#                        closed with `;`. A trait method declaration is a `fn`
#                        item with no body, and a scanner that waits for the
#                        next `{` swallows the next function whole — the one
#                        after it is never recorded at all, which is a lost
#                        population member and not a refusal. It reads the line
#                        that closes the parameter list and, for a `where`
#                        clause, every line until the body opens.
#
#                        Its boundary: it reads `decomment`'s output, so a
#                        declaration whose `;` is followed by a `/* ... */`
#                        block comment reads as a line that opens a body and
#                        swallows the next item. Neither reader knows block
#                        comments, and no `/*` occurs in the sources these
#                        gates scan. It fails closed rather than silently:
#                        the swallowed item is a lost population member, and
#                        both gates refuse entries with no forwarded loop to
#                        enter.
#
#   THE BOUNDARY. Both readers track the `"` and the backslash escape and
#   nothing else, so a character literal holding a quote (`'\"'`) and a raw
#   string (`r#"..."#`, whose inner quotes end nothing) are both mis-read — the
#   raw string fails open, leaving the rest of the line unblanked. Neither
#   occurs in the sources these gates scan today. A gate that starts scanning a
#   file with either needs a real lexer, not a wider regex here.
#
#   Use `blank_strings` where a needle must not match text a program merely
#   prints. Do NOT use it where the needle IS a literal: `check_spec_charge.sh`
#   reads the per-round event's target that way, and blanking would make that
#   rule unfireable.

# shellcheck disable=SC2034  # consumed by the gates that source this file
readonly AWK_TEXT_FNS='
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
      function sig_is_bodiless(s) {
        return (index(s, "{") == 0 && s ~ /;[[:space:]]*$/)
      }
      function blank_strings(s,   i, n, q, ch, out) {
        n = length(s); q = 0; out = ""
        for (i = 1; i <= n; i++) {
          ch = substr(s, i, 1)
          if (q) {
            if (ch == "\\") { out = out "  "; i++; continue }
            if (ch == "\"") { q = 0; out = out ch; continue }
            out = out " "
            continue
          }
          if (ch == "\"") { q = 1 }
          out = out ch
        }
        return out
      }
'
