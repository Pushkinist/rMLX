#!/usr/bin/env bash
# The exit code `rmlx baseline --record` and `rmlx eval ppl --corpus` give a
# run that measured and whose record did not land, and how a caller surfaces
# the buffer file that run left (docs/METRICS_DB.md section 13, rule 24).
#
# Usage:
#
#     source "$REPO_ROOT/scripts/lib/record_pending.sh"
#     rc=0
#     out="$("$BIN" baseline ... --record --run-class diagnostic 2>"$err")" || rc=$?
#     if [ "$rc" -eq "$RECORD_PENDING_EXIT" ]; then
#         record_pending_note "$err"   # keep the measurement, end non-zero later
#     fi
#
#     rmlx_record_file "$BIN" "$buffer_file" "$label" || status=$?   # 75 when refused

# Kept equal to `crate::exit::RECORD_PENDING` in crates/rmlx-cli/src/exit.rs.
RECORD_PENDING_EXIT=75

# Print the buffer file the run's stderr names, and the replay command, on
# stderr. <stderr-file> is what the run wrote to stderr.
record_pending_note() {
  local file
  file="$(sed -n 's/.*it is kept at \(.*\) for `rmlx metrics record --replay-pending`.*/\1/p' "$1" | tail -1)"
  [ -n "$file" ] || file="(no buffer file named on stderr; see the run log)"
  echo "record pending: $file" >&2
  echo "  measured, but the record did not land; replay it with" >&2
  echo "  \`rmlx metrics record --replay-pending\` once the cause is fixed" >&2
}

# File the record JSON <file>, already written to the pending buffer, with
# `<binary> metrics [<metrics option>...] record --file <file>`. <label> names the
# record in the error. A refusal leaves the file in the buffer, names it on stderr
# and returns RECORD_PENDING_EXIT, so the caller can keep the reading and end
# non-zero later. The recorder reads RMLX_HOME from the environment.
rmlx_record_file() {
  local binary="$1" file="$2" label="$3"
  shift 3
  if ! "$binary" metrics "$@" record --file "$file" >&2; then
    echo "  ERROR: ${label}: the record did not land; it is kept at ${file}; file it" \
         "again with \`rmlx metrics record --file\` once the cause is fixed" >&2
    return "$RECORD_PENDING_EXIT"
  fi
  echo "  [ingest] done: ${file}" >&2
}
