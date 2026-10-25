# Line readers for Rust-scanning gates: code without comments, without string bodies, bodiless fns.
#
# Sourced, not executed.
#
# WHY ONE PRODUCER
#   A gate that reads Rust source needs the same answers everywhere: what on
#   this line is code rather than a comment, what is code rather than the
#   contents of a string literal, and whether a signature line ends its
#   declaration. A second copy of any of the three drifts, and the way it
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
#   THE BOUNDARY. Both readers track the `"` and the backslash escape, and skip
#   the two character literals that hold a quote, `'"'` and `'\"'`, as fixed
#   sequences: they occur in the scanned sources and the quote in one ends
#   nothing. A raw string (`r#"..."#`) is not recognised. Its inner quote flips
#   the quote state for the rest of the line, so a `//` after it cuts the line
#   short and hides the code behind it, and a real comment after it reads as
#   code. A gate that must scan a file with one needs a real lexer, not a wider
#   regex here.
#
#   Use `blank_strings` where a needle must not match text a program merely
#   prints. Do NOT use it where the needle IS a literal: `check_spec_charge.sh`
#   reads the per-round event's target that way, and blanking would make that
#   rule unfireable.

# shellcheck disable=SC2034  # consumed by the gates that source this file
readonly AWK_TEXT_FNS='
      function char_quote_len(s, i,   t) {
        t = substr(s, i, 4)
        if (substr(t, 1, 3) == "\047\"\047") return 3
        if (t == "\047\\\"\047") return 4
        return 0
      }
      function decomment(s,   i, n, q, ch, k) {
        n = length(s); q = 0
        for (i = 1; i <= n; i++) {
          ch = substr(s, i, 1)
          if (q && ch == "\\") { i++; continue }
          if (!q && ch == "\047" && (k = char_quote_len(s, i)) > 0) { i += k - 1; continue }
          if (ch == "\"") { q = !q; continue }
          if (!q && ch == "/" && substr(s, i + 1, 1) == "/") { return substr(s, 1, i - 1) }
        }
        return s
      }
      function sig_is_bodiless(s) {
        return (index(s, "{") == 0 && s ~ /;[[:space:]]*$/)
      }
      function blank_strings(s,   i, n, q, ch, out, k) {
        n = length(s); q = 0; out = ""
        for (i = 1; i <= n; i++) {
          ch = substr(s, i, 1)
          if (q) {
            if (ch == "\\") { out = out "  "; i++; continue }
            if (ch == "\"") { q = 0; out = out ch; continue }
            out = out " "
            continue
          }
          if (ch == "\047" && (k = char_quote_len(s, i)) > 0) { out = out substr(s, i, k); i += k - 1; continue }
          if (ch == "\"") { q = 1 }
          out = out ch
        }
        return out
      }
'
