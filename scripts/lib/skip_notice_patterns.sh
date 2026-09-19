# scripts/lib/skip_notice_patterns.sh — the stand-down notice, defined once.
#
# A model-gated test that returns before asserting anything announces
# `SKIP <test>: <why>`. Two tools read that line and they must agree exactly on
# what it is:
#
#   * `scripts/run_gpu_tests.sh` reads it out of a libtest log, to list the cell
#     and to drop its count from the shader-validation census expectation.
#   * `scripts/check_named_skip_notices.sh` reads it out of the source, to fail
#     the build on a notice that names no test.
#
# A difference between the two is invisible in both directions and worse than
# either being wrong alone: a notice the source gate calls attributed and the
# runner counts as nameless passes CI and then makes every run INCOMPLETE with a
# number and no name, which is the state this pair of tools exists to end.
#
# So there is one definition. Anchoring is the reader's business — under
# `--nocapture` the notice lands after libtest's `test some::name ... ` prefix,
# so neither pattern is anchored at line start — but the SHAPE is here.

# A notice carrying a name. The single space is deliberate and is the runner's:
# `SKIP  foo:` does not match it, so the source gate must not accept that either.
NAMED_SKIP='SKIP [A-Za-z_][A-Za-z0-9_]*:'

# Any stand-down announcement, named or not. The surrounding character classes
# keep `RMLX_SKIP_GPU` and words merely containing the letters from counting.
ANY_SKIP='(^|[^A-Za-z0-9_])SKIP([^A-Za-z0-9_]|$)'

# The same notice as it is written in a Rust format string, where the test's own
# name arrives as an argument rather than as a literal. `{test}` exactly: any
# other placeholder is a name the source gate cannot check, and `{other}` reads
# identically while naming a cell that ran.
NAMED_SKIP_PLACEHOLDER='SKIP \{test\}:'
