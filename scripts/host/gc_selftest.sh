#!/usr/bin/env bash
# Recall test for the cleanup tools: planted scratch trees, never the real target/ or traces.
#
# Recall test for: target_gc.sh --apply --all
#   traces_gc.sh --apply --all --max-count --max-total-gb --max-age-days --dir
#   target_size_report.sh
# Not a mode of: profile/gputrace_selftest.sh — that one reads .gputrace bundles; this one plants and deletes them
#
# Each tool is copied into a scratch repo under ST_WORK and run there, so its own `cd` to the repo
# root lands on the planted tree. Ages are set with `touch -t`. A stub `du` reads a planted `.kb`
# file as the size of a directory, so a boundary in gigabytes needs no gigabytes on disk. Every
# case asserts the literal exit code, what was printed and the entries left on disk. The tools are
# named in one place each: `tool_target`, `tool_traces` and `tool_report`.
#
# Usage:  bash scripts/host/gc_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a fixture could not be built

set -euo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init gc-selftest
unset MAX_AGE_DAYS

SCRATCH_REPO="$ST_WORK/repo"
GIB_KB=1048576

SCRATCH_TOOLS="$SCRATCH_REPO/scripts"

mkdir -p "$SCRATCH_TOOLS" "$ST_WORK/stubs"
cat >"$ST_WORK/stubs/du" <<'EOF2'
#!/bin/sh
for a; do last=$a; done
if [ -f "$last/.kb" ]; then
    kb=$(cat "$last/.kb")
    case "$1" in -sh) printf '%sK\t%s\n' "$kb" "$last" ;; *) printf '%s\t%s\n' "$kb" "$last" ;; esac
    exit 0
fi
exec /usr/bin/du "$@"
EOF2
chmod +x "$ST_WORK/stubs/du"
export PATH="$ST_WORK/stubs:$PATH"

# ---- the three tools, one function each ----------------------------------------------------

# tool_target [args]: prune stale target/ profiles; MAX_AGE_DAYS in the environment sets the age cut.
tool_target() {
    cp "$REPO_ROOT/scripts/target_gc.sh" "$SCRATCH_TOOLS/" || st_die "cannot copy target_gc.sh"
    st_bounded 60 bash "$SCRATCH_TOOLS/target_gc.sh" ${1+"$@"}
}

# tool_traces [args]: bound the trace bundles.
tool_traces() {
    cp "$REPO_ROOT/scripts/traces_gc.sh" "$SCRATCH_TOOLS/" || st_die "cannot copy traces_gc.sh"
    st_bounded 60 bash "$SCRATCH_TOOLS/traces_gc.sh" ${1+"$@"}
}

# tool_report [args]: print the target/ size and the hint.
tool_report() {
    cp "$REPO_ROOT/scripts/target_size_report.sh" "$SCRATCH_TOOLS/" || st_die "cannot copy target_size_report.sh"
    st_bounded 60 bash "$SCRATCH_TOOLS/target_size_report.sh" ${1+"$@"}
}

# tool_target_aged <days> <args>: tool_target with the age cut set.
tool_target_aged() {
    local days="$1"
    shift
    MAX_AGE_DAYS="$days" tool_target ${1+"$@"}
}

# ---- planting ------------------------------------------------------------------------------

fresh() {
    rm -rf "$SCRATCH_REPO/target" "$SCRATCH_REPO/.rmlx"
    mkdir -p "$SCRATCH_REPO/target" "$SCRATCH_REPO/.rmlx/traces"
}

stamp() { date -v-"$(awk -v d="$1" 'BEGIN { print int(d * 24) }')"H +%Y%m%d%H%M; }

# profile <name> <age-days>: a target/ profile whose one file is <age-days> old.
profile() {
    local dir="$SCRATCH_REPO/target/$1"
    mkdir -p "$dir"
    echo artifact >"$dir/a.o"
    touch -t "$(stamp "$2")" "$dir/a.o" "$dir"
}

# bundle <name> <age-days> [<kb>]: a trace bundle whose directory is <age-days> old, <kb> large to `du`.
bundle() {
    local dir="$SCRATCH_REPO/.rmlx/traces/$1.gputrace"
    mkdir -p "$dir"
    echo data >"$dir/capture"
    if [ -n "${3:-}" ]; then echo "$3" >"$dir/.kb"; fi
    touch -t "$(stamp "$2")" "$dir"
}

left_target() { ls "$SCRATCH_REPO/target" | tr '\n' ' ' | sed 's/ $//'; }
left_traces() { ls "$SCRATCH_REPO/.rmlx/traces" | sed 's/\.gputrace$//' | tr '\n' ' ' | sed 's/ $//'; }
left_other() { ls "$ST_WORK/other" | tr '\n' ' ' | sed 's/ $//'; }

# disk_case <name> <want-exit> <lister> <entries-left> [--has S]... [--lacks S]... -- <cmd...>:
# the exit code and output as st_case, and the entries the lister finds left on disk.
disk_case() {
    local name="$1" want="$2" lister="$3" expect="$4" checks=() why got
    shift 4
    while [ "$#" -gt 0 ] && [ "$1" != -- ]; do
        checks+=("$1" "${2-}")
        shift 2 || break
    done
    shift
    st_run "$@"
    why="$(st_check "$want" ${checks[@]+"${checks[@]}"})"
    got="$($lister)"
    if [ "$got" != "$expect" ]; then why="${why:+$why$'\n'}left on disk '$got', want '$expect'"; fi
    st_verdict "$name" "$why"
}

plant_profiles() {
    fresh
    profile dev 7.5
    profile release 1
    profile release-perf 8
    profile release-debug 6.5
    profile doc 30
    profile criterion 30
}

plant_traces() {
    fresh
    bundle t1 8
    bundle t2 7
    bundle t3 6
    bundle t4 5
    bundle t5 4
    bundle t6 3
    bundle t7 2
    bundle t8 1
}

# ---- target_gc.sh --------------------------------------------------------------------------

ALL_PROFILES="criterion dev doc release release-debug release-perf"

plant_profiles
disk_case "target report: deletes nothing, says what it would prune" 0 left_target "$ALL_PROFILES" \
    --has "would   dev" --has "keep    release-perf" --has "protected" --has "touched within 7d" \
    --has "report only" --lacks "prune   " -- tool_target

plant_profiles
disk_case "target --apply: prunes the stale profile and nothing else" 0 left_target "criterion doc release release-debug release-perf" \
    --has "prune   dev" --lacks "prune   release" --lacks "prune   doc" --has "reclaimed" -- tool_target --apply

plant_profiles
disk_case "target --apply --all: the protected profile goes too" 0 left_target "criterion doc release release-debug" \
    --has "prune   dev" --has "prune   release-perf" --lacks "protected" -- tool_target --apply --all

plant_profiles
disk_case "target report --all: still deletes nothing" 0 left_target "$ALL_PROFILES" \
    --has "would   release-perf" --lacks "prune   " -- tool_target --all

plant_profiles
disk_case "target: MAX_AGE_DAYS=3 moves the age cut down" 0 left_target "criterion doc release release-perf" \
    --has "prune   dev" --has "prune   release-debug" --has "touched within 3d" -- tool_target_aged 3 --apply

plant_profiles
disk_case "target: MAX_AGE_DAYS=10 moves the age cut up" 0 left_target "$ALL_PROFILES" \
    --lacks "prune   " --has "touched within 10d" -- tool_target_aged 10 --apply

plant_profiles
disk_case "target: a profile 6.5 days old stays under the 7 day cut" 0 left_target "criterion doc release release-debug release-perf" \
    --has "keep    release-debug" -- tool_target --apply

fresh
profile mixed 20
echo fresh >"$SCRATCH_REPO/target/mixed/b.o"
touch -t "$(stamp 1)" "$SCRATCH_REPO/target/mixed/b.o"
disk_case "target: one recent file keeps an otherwise stale profile" 0 left_target "mixed" \
    --has "keep    mixed" -- tool_target --apply

rm -rf "$SCRATCH_REPO/target"
st_case "target: no target/ is a no-op" 0 --has "no target/ directory" -- tool_target --apply

plant_profiles
disk_case "target: an unknown argument is a usage error, even after --apply, and deletes nothing" 2 left_target "$ALL_PROFILES" \
    --has "unknown argument: --bogus" -- tool_target --apply --bogus

# ---- traces_gc.sh --------------------------------------------------------------------------

ALL_BUNDLES="t1 t2 t3 t4 t5 t6 t7 t8"

plant_traces
disk_case "traces report: deletes nothing, says what it would prune" 0 left_traces "$ALL_BUNDLES" \
    --has "would   t1.gputrace" --has "beyond the newest 6" --has "would   t2.gputrace" --has "keep    t3.gputrace" \
    --has "report only" --has "prune 2 bundle(s)" --lacks "prune   " -- tool_traces

plant_traces
disk_case "traces --apply: prunes exactly the two beyond the newest six" 0 left_traces "t3 t4 t5 t6 t7 t8" \
    --has "prune   t1.gputrace" --has "prune   t2.gputrace" --lacks "prune   t3" --has "2 pruned" -- tool_traces --apply

plant_traces
disk_case "traces --max-count 3: keeps exactly the newest three" 0 left_traces "t6 t7 t8" \
    --has "beyond the newest 3" --has "5 pruned" -- tool_traces --apply --max-count 3

plant_traces
disk_case "traces --max-count 8 with eight bundles: nothing to prune" 0 left_traces "$ALL_BUNDLES" \
    --has "nothing to prune" --lacks "would   " -- tool_traces --max-count 8

plant_traces
disk_case "traces --max-count 7 with eight bundles: prunes exactly one" 0 left_traces "t2 t3 t4 t5 t6 t7 t8" \
    --has "prune   t1.gputrace" --has "1 pruned" -- tool_traces --apply --max-count 7

plant_traces
disk_case "traces --max-count 0: the newest bundle is still kept" 0 left_traces "t8" \
    --has "keep    t8.gputrace" --has "newest bundle (use --all to override)" --has "7 pruned" -- tool_traces --apply --max-count 0

plant_traces
disk_case "traces --max-count 0 --all: the newest goes too" 0 left_traces "" \
    --has "prune   t8.gputrace" --has "8 pruned" -- tool_traces --apply --max-count 0 --all

plant_traces
disk_case "traces report --max-count 0 --all: deletes nothing" 0 left_traces "$ALL_BUNDLES" \
    --has "would   t8.gputrace" --lacks "prune   " -- tool_traces --max-count 0 --all

fresh
bundle a 7.5
bundle b 6.5
bundle c 3
bundle d 1
disk_case "traces --max-age-days 7: prunes the bundle 7.5 days old, keeps the one 6.5 days old" 0 left_traces "b c d" \
    --has "prune   a.gputrace" --has "stale >7d" --lacks "prune   b" -- tool_traces --apply --max-age-days 7
fresh
bundle a 7.5
bundle b 6.5
bundle c 3
bundle d 1
disk_case "traces --max-age-days 2: prunes every bundle older than the cut" 0 left_traces "d" \
    --has "prune   a.gputrace" --has "prune   b.gputrace" --has "prune   c.gputrace" -- tool_traces --apply --max-age-days 2
fresh
bundle a 7.5
bundle b 6.5
disk_case "traces: no age rule unless the flag is given" 0 left_traces "a b" --has "0 pruned" -- tool_traces --apply

fresh
bundle only 30
disk_case "traces: a lone stale bundle is the newest and is kept" 0 left_traces "only" \
    --has "newest bundle" --lacks "prune   " -- tool_traces --apply --max-age-days 1
disk_case "traces: a lone stale bundle goes with --all" 0 left_traces "" \
    --has "prune   only.gputrace" -- tool_traces --apply --max-age-days 1 --all

fresh
bundle s1 8 "$GIB_KB"
bundle s2 6 "$GIB_KB"
disk_case "traces size: total equal to the cap prunes nothing" 0 left_traces "s1 s2" \
    --has "0 pruned" -- tool_traces --apply --max-total-gb 2
fresh
bundle s1 8 "$GIB_KB"
bundle s2 6 "$GIB_KB"
bundle s3 4 "$GIB_KB"
bundle s4 2 "$GIB_KB"
disk_case "traces size: evicts oldest first and stops once the rest fits" 0 left_traces "s3 s4" \
    --has "prune   s1.gputrace" --has "over the 2G cap" --has "prune   s2.gputrace" \
    --has "4G -> 2G, 2 pruned (reclaimed 2G)" -- tool_traces --apply --max-total-gb 2
fresh
bundle s1 8 "$GIB_KB"
bundle s2 6 "$GIB_KB"
bundle s3 4 "$GIB_KB"
disk_case "traces size: three bundles of 1G under a 2G cap lose exactly the oldest" 0 left_traces "s2 s3" \
    --has "prune   s1.gputrace" --has "1 pruned" -- tool_traces --apply --max-total-gb 2
fresh
bundle s1 8 "$GIB_KB"
bundle s2 6 "$GIB_KB"
bundle s3 4 "$GIB_KB"
disk_case "traces size report: deletes nothing" 0 left_traces "s1 s2 s3" \
    --has "would   s1.gputrace" --lacks "prune   " -- tool_traces --max-total-gb 2
fresh
bundle s1 8 "$GIB_KB"
bundle s2 6 "$GIB_KB"
disk_case "traces size: a cap of 0 still keeps the newest" 0 left_traces "s2" \
    --has "newest bundle" -- tool_traces --apply --max-total-gb 0

fresh
bundle t1 3
bundle t2 2
mkdir -p "$SCRATCH_REPO/.rmlx/traces/mst/keep-me"
echo note >"$SCRATCH_REPO/.rmlx/traces/notes.txt"
disk_case "traces: only .gputrace bundles are candidates" 0 left_traces "mst notes.txt" \
    --has "prune   t1.gputrace" --has "prune   t2.gputrace" -- tool_traces --apply --max-count 0 --all

fresh
st_case "traces: an empty directory is a no-op" 0 --has "holds no .gputrace bundles" -- tool_traces --apply
rm -rf "$SCRATCH_REPO/.rmlx"
st_case "traces: no directory is a no-op" 0 --has "no .rmlx/traces — nothing to do" -- tool_traces --apply

fresh
mkdir -p "$ST_WORK/other/b1.gputrace" "$ST_WORK/other/b2.gputrace"
touch -t "$(stamp 3)" "$ST_WORK/other/b1.gputrace"
bundle bystander 5
disk_case "traces --dir: the directory named is the one bounded" 0 left_other "" \
    --has "prune   b1.gputrace" --has "prune   b2.gputrace" -- tool_traces --apply --max-count 0 --all --dir "$ST_WORK/other"
disk_case "traces --dir: the default directory is left alone" 0 left_traces "bystander" \
    --has "nothing to do" -- tool_traces --apply --dir "$ST_WORK/none"

plant_traces
disk_case "traces: an unknown argument is a usage error and deletes nothing" 2 left_traces "$ALL_BUNDLES" \
    --has "unknown argument: --bogus" --has "usage:" -- tool_traces --apply --bogus
disk_case "traces: a flag with no value is a usage error" 2 left_traces "$ALL_BUNDLES" \
    --has "--max-count needs a value" -- tool_traces --apply --max-count
disk_case "traces: a value that is not a whole number is a usage error" 2 left_traces "$ALL_BUNDLES" \
    --has "--max-total-gb must be a whole number (got 'x')" -- tool_traces --apply --max-total-gb x
disk_case "traces: --help prints the usage and deletes nothing" 0 left_traces "$ALL_BUNDLES" \
    --has "usage:" -- tool_traces --apply --help

# ---- target_size_report.sh -----------------------------------------------------------------

# sized_target <kb>: a target/ that `du` reports at <kb>.
sized_target() {
    fresh
    echo "$1" >"$SCRATCH_REPO/target/.kb"
}

sized_target $((60 * GIB_KB))
st_case "size report: past the threshold prints the hint" 0 --has "target/ is 60G (>= 50G advisory threshold)" --has "make target-gc" -- tool_report
sized_target $((50 * GIB_KB))
st_case "size report: at the threshold prints the hint" 0 --has "target/ is 50G (>= 50G advisory threshold)" -- tool_report
sized_target $((50 * GIB_KB - 1))
st_case "size report: just under the threshold prints no hint" 0 --has "target/: " --lacks "advisory threshold" -- tool_report
sized_target $((10 * GIB_KB))
st_case "size report: a custom threshold is read from the first argument" 0 --has "target/ is 10G (>= 10G advisory threshold)" -- tool_report 10
sized_target $((10 * GIB_KB))
st_case "size report: a custom threshold above the size prints no hint" 0 --lacks "advisory threshold" -- tool_report 11
rm -rf "$SCRATCH_REPO/target"
st_case "size report: no target/ is a no-op" 0 --has "target/ absent" -- tool_report

st_done
