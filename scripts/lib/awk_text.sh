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
