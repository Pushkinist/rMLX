#!/usr/bin/env bash
# scripts/kv_update_census_selftest.sh — recall test for
# `kv_update_census.py`. Every case is one edit to an otherwise valid synthetic
# tree, and asserts both the exit code and the figure or reason the producer
# printed.
#
# WHY BOTH
#   The producer reports a measured figure (exit 0) and an inability to measure
#   (exit 2) on different exits on purpose. A structural metric that silently
#   printed `0` for a tree it could not read would report the restructure as
#   finished on the day the scan broke. A case that asserted only "non-zero"
#   would pass against a producer that had stopped finding any site at all.
#
#   Nothing here hard-codes a figure the producer derives from the real tree:
#   the fixtures are their own trees, and each case states the number its own
#   planted source implies.
#
# EXIT CODES
#   0  every case produced the expected exit code and output
#   1  a case did not
#   2  the fixtures themselves could not be built

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CENSUS="${REPO_ROOT}/scripts/kv_update_census.py"

[ -f "${CENSUS}" ] || { echo "ERROR: missing ${CENSUS}" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || {
    echo "ERROR: python3 not on PATH" >&2
    exit 2
}

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

STORAGE_REL="crates/rmlx-kv-quant/src/storage/kv_storage.rs"
QUANT_REL="crates/rmlx-kv-quant/src/quant.rs"
UPDATE_REL="crates/rmlx-kv-quant/src/kvcache/update.rs"

# Build a fresh tree under $1. Four variants, four quant spellings, so a
# threshold of 2 is "half the enum" and a site naming two variants is at the
# bar. Every case edits a copy of this.
build_tree() { # build_tree ROOT
    local root="$1"
    rm -rf "${root}"
    mkdir -p "${root}/$(dirname "${STORAGE_REL}")" "${root}/$(dirname "${UPDATE_REL}")"
    cat >"${root}/${STORAGE_REL}" <<'EOF'
pub enum KvStorage {
    Alpha { k: Option<QuantK>, v: Option<QuantV>, max_seq: i32 },
    Beta { k: Option<QuantK>, v: Option<QuantV>, max_seq: i32, bits: u8 },
    Gamma { k: Option<QuantK>, max_seq: i32 },
    Delta { state: MixedKvState, max_seq: i32 },
}

pub fn resident(s: &KvStorage) -> usize {
    match s {
        KvStorage::Alpha { .. } => 1,
        KvStorage::Beta { .. } => 2,
        KvStorage::Gamma { .. } => 3,
        KvStorage::Delta { .. } => 4,
    }
}
EOF
    cat >"${root}/${QUANT_REL}" <<'EOF'
pub enum KvQuant {
    One,
    Two,
    Three,
    Four,
}

pub fn label(q: KvQuant) -> &'static str {
    match q {
        KvQuant::One => "one",
        KvQuant::Two => "two",
        KvQuant::Three => "three",
        KvQuant::Four => "four",
    }
}
EOF
    cat >"${root}/${UPDATE_REL}" <<'EOF'
pub fn dispatch(s: &KvStorage) -> usize {
    match s {
        KvStorage::Alpha { .. } | KvStorage::Beta { .. } => update_alpha(),
        KvStorage::Gamma { .. } => update_gamma(),
        KvStorage::Delta { .. } => update_delta(),
    }
}

fn update_alpha() -> usize {
    let a = 1;
    a
}

fn update_gamma() -> usize {
    2
}

fn update_delta() -> usize {
    3
}

fn not_an_update_body() -> usize {
    4
}
EOF
}

run() { # run ROOT MODE [EXTRA...]
    local root="$1"; shift
    python3 "${CENSUS}" --root "${root}" "$@" 2>&1
}

failures=0
check() { # check LABEL ROOT WANT_EXIT WANT_PATTERN MODE [EXTRA...]
    local label="$1" root="$2" want_exit="$3" want_pat="$4"; shift 4
    local out rc
    out="$(run "${root}" "$@")"
    rc=$?
    if [ "${rc}" -ne "${want_exit}" ]; then
        echo "FAIL  ${label}: exit ${rc}, expected ${want_exit}" >&2
        echo "${out}" | head -5 >&2
        failures=$((failures + 1))
        return
    fi
    if ! grep -qE -- "${want_pat}" <<<"${out}"; then
        echo "FAIL  ${label}: exit ${rc} as expected but no line matching /${want_pat}/" >&2
        echo "${out}" | head -8 >&2
        failures=$((failures + 1))
        return
    fi
    echo "ok    ${label}  (exit ${rc}, output matched)"
}

T="${WORK}/clean"
build_tree "${T}"

# 0 — the unedited tree. Three sites name two or more of the four variants:
# the two enum files' own dispatch and the update file's. Without this the
# whole file could be passing because the producer refuses every tree.
check "clean tree counts its sites" "${T}" 0 "^match-sites 3$" match-sites --threshold 2
check "clean tree: all three force a touch" "${T}" 0 "^forcing-sites 3$" match-sites --threshold 2
check "clean tree: per-file count" "${T}" 0 "^file ${UPDATE_REL} 1$" match-sites --threshold 2

# 1 — a fourth site planted in a file the producer was not told about. A
# hand-written file list would miss it.
T="${WORK}/hidden"; build_tree "${T}"
mkdir -p "${T}/crates/rmlx-kv-ssd/src"
cat >"${T}/crates/rmlx-kv-ssd/src/block_io.rs" <<'EOF'
pub fn spill(s: &KvStorage) -> usize {
    match s {
        KvStorage::Alpha { .. } => 1,
        KvStorage::Beta { .. } => 2,
        KvStorage::Gamma { .. } => 3,
        KvStorage::Delta { .. } => 4,
    }
}
EOF
check "a site in an unnamed file is found" "${T}" 0 "^match-sites 4$" match-sites --threshold 2

# 2 — a site collapsed onto one arm behind a helper. The count must drop, which
# is the whole claim the restructure makes.
T="${WORK}/collapsed"; build_tree "${T}"
cat >"${T}/${UPDATE_REL}" <<'EOF'
pub fn dispatch(s: &KvStorage) -> usize {
    update_shared(s)
}

fn update_shared(_s: &KvStorage) -> usize {
    1
}
EOF
check "a collapsed site drops the count" "${T}" 0 "^match-sites 2$" match-sites --threshold 2

# 3 — the same three sites, but the update file's has a catch-all arm. It is
# still a site; it no longer forces a touch, and the two figures say so
# separately.
T="${WORK}/catchall"; build_tree "${T}"
cat >"${T}/${UPDATE_REL}" <<'EOF'
pub fn dispatch(s: &KvStorage) -> usize {
    match s {
        KvStorage::Alpha { .. } => 1,
        KvStorage::Beta { .. } => 2,
        KvStorage::Gamma { .. } => 3,
        other => fallback(other),
    }
}

fn update_alpha() -> usize {
    1
}
EOF
check "a catch-all arm is still a site" "${T}" 0 "^match-sites 3$" match-sites --threshold 2
check "a catch-all arm forces no touch" "${T}" 0 "^forcing-sites 2$" match-sites --threshold 2

# 4 — a match naming one variant of four is a case analysis, not an
# enumeration, and stays under the bar.
T="${WORK}/small"; build_tree "${T}"
cat >"${T}/${UPDATE_REL}" <<'EOF'
pub fn dispatch(s: &KvStorage) -> usize {
    match s {
        KvStorage::Alpha { .. } => 1,
        _ => 0,
    }
}
EOF
check "a one-variant match is under the bar" "${T}" 0 "^match-sites 2$" match-sites --threshold 2

# 5 — the bar is derived from the enum, not fixed: at a bar of 3 the update
# file's own two-variant arm pattern still names three variants across the
# match and stays in, while the clean tree measured at 5 finds nothing.
T="${WORK}/bar"; build_tree "${T}"
check "a bar above the enum finds nothing" "${T}" 0 "^match-sites 0$" match-sites --threshold 5

# 6 — `match` inside a comment and inside a string literal is text, not a site.
T="${WORK}/text"; build_tree "${T}"
cat >>"${T}/${UPDATE_REL}" <<'EOF'

// match s { KvStorage::Alpha { .. } => 1, KvStorage::Beta { .. } => 2,
// KvStorage::Gamma { .. } => 3, KvStorage::Delta { .. } => 4 }
fn reason() -> &'static str {
    "match s { KvStorage::Alpha => 1, KvStorage::Beta => 2, KvStorage::Gamma => 3 }"
}
EOF
check "commented and quoted matches are not sites" "${T}" 0 "^match-sites 3$" match-sites --threshold 2

# 7 — no crates directory at all. `unavailable`, never `0`.
T="${WORK}/bare"; rm -rf "${T}"; mkdir -p "${T}"
check "a tree with no crates directory" "${T}" 2 "unavailable: crates" match-sites --threshold 2

# 8 — a crates directory holding no variant reference.
T="${WORK}/nomention"; build_tree "${T}"
rm -rf "${T}/crates/rmlx-kv-quant/src/kvcache"
printf 'pub enum KvStorage { Alpha { k: Option<QuantK>, v: Option<QuantV>, max_seq: i32 } }\n' \
    >"${T}/${STORAGE_REL}"
printf 'pub enum KvQuant { One }\n' >"${T}/${QUANT_REL}"
check "no file names a variant" "${T}" 2 "unavailable: no source file" match-sites --threshold 2

# 9 — the enum is gone. The producer cannot derive a bar and says so.
T="${WORK}/noenum"; build_tree "${T}"
printf 'pub struct NotAnEnum;\n' >"${T}/${STORAGE_REL}"
check "the storage enum is missing" "${T}" 2 "unavailable: .*KvStorage not found" match-sites --threshold 2

# 10 — a brace that never closes. A scan that cannot read a file back reports
# that, rather than the sites it managed to find before the file.
T="${WORK}/unbalanced"; build_tree "${T}"
printf 'pub fn broken(s: &KvStorage) -> usize {\n    match s {\n        KvStorage::Alpha { .. } => 1,\n        KvStorage::Beta { .. } => 2,\n' \
    >>"${T}/${UPDATE_REL}"
check "an unterminated block is a refusal" "${T}" 2 "unavailable: .*update.rs" match-sites --threshold 2

# 11 — the shape census. Four planted variants, one per shape class.
T="${WORK}/shapes"; build_tree "${T}"
check "shape census: two slots and max_seq" "${T}" 0 "^shape kv_slots 1$" variants
check "shape census: a scalar knob beside the slots" "${T}" 0 "^shape kv_slots_plus 1$" variants
check "shape census: K only" "${T}" 0 "^shape k_only 1$" variants
check "shape census: state the shape cannot reach" "${T}" 0 "^shape other 1$" variants
check "shape census: the shared-shape total" "${T}" 0 "^shape store_slots_total 3$" variants

# 12 — a variant that grows a field outside the scalar-knob set leaves the
# shared shape. It must not be folded in quietly.
T="${WORK}/newfield"; build_tree "${T}"
sed -i.bak 's/    Beta { k: Option<QuantK>, v: Option<QuantV>, max_seq: i32, bits: u8 },/    Beta { k: Option<QuantK>, v: Option<QuantV>, max_seq: i32, table: Table },/' \
    "${T}/${STORAGE_REL}"
check "a state field leaves the shared shape" "${T}" 0 "^shape store_slots_total 2$" variants

# 13 — the update-body population: bodies only, and a fn that is not one of
# them is not counted.
T="${WORK}/bodies"; build_tree "${T}"
check "update bodies are counted" "${T}" 0 "^update-bodies 3$" update-bodies
check "a body's lines are its braces, not its signature" "${T}" 0 "^body update_gamma line=[0-9]+ lines=3$" update-bodies

# 14 — a declaration with no body is not a body.
T="${WORK}/decl"; build_tree "${T}"
cat >>"${T}/${UPDATE_REL}" <<'EOF'

trait Update {
    fn update_declared(&self) -> usize;
}
EOF
check "a bodiless declaration is not a body" "${T}" 0 "^update-bodies 3$" update-bodies

# 15 — no update body at all. `unavailable`, never `0`.
T="${WORK}/nobodies"; build_tree "${T}"
printf 'pub fn dispatch() -> usize { 1 }\n' >"${T}/${UPDATE_REL}"
check "no update body is a refusal" "${T}" 2 "unavailable: .*update_\\* fn" update-bodies

# 16 — the update file is gone.
T="${WORK}/nofile"; build_tree "${T}"
rm -f "${T}/${UPDATE_REL}"
check "a missing update file is a refusal" "${T}" 2 "unavailable: .*is not a file" update-bodies

# 17 — the reference count, both ways.
T="${WORK}/refs"; build_tree "${T}"
check "variant references are counted" "${T}" 0 "KvStorage=4 distinct=4" refs --file "${UPDATE_REL}"
T="${WORK}/norefs"; build_tree "${T}"
printf 'pub fn nothing() -> usize { 1 }\n' >"${T}/${UPDATE_REL}"
check "a file naming no variant is a refusal" "${T}" 2 "unavailable: .*names no" refs --file "${UPDATE_REL}"

if [ "${failures}" -gt 0 ]; then
    echo >&2
    echo "ERROR: ${failures} case(s) did not reproduce the expected behaviour." >&2
    exit 1
fi
echo "OK: the KV update census reports every planted figure, and refuses every tree it cannot measure."
