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
#   The fixtures are their own trees, and each case states the number its own
#   planted source implies. The last cases pin the real tree's site figures,
#   so a change that moves one re-pins it in the same change.
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
check "clean tree: no subset site" "${T}" 0 "^subset-sites 0$" match-sites --threshold 2
check "clean tree: no table site" "${T}" 0 "^table-sites 0$" match-sites --threshold 2

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
check "a catch-all match under the bar is a subset site" "${T}" 0 \
    "^subset ${UPDATE_REL}:2 kind=wildcard enum=KvStorage variants=1$" match-sites --threshold 2

# 5 — the bar is derived from the enum, not fixed. Each of the three planted
# sites names all four variants, so it is in at a bar of 3 and out at a bar of
# 5.
T="${WORK}/bar"; build_tree "${T}"
check "a bar at three keeps the four-variant sites" "${T}" 0 "^match-sites 3$" match-sites --threshold 3
check "a bar above the enum finds nothing" "${T}" 0 "^match-sites 0$" match-sites --threshold 5

# 5a — the derived bar itself, with no --threshold to supply one. Four variants
# per enum, so the bar is 2, and a site naming exactly two is at it. Without
# this case the derivation could be replaced by any constant at or below 2 and
# every other case here would stay green.
T="${WORK}/derived"; build_tree "${T}"
cat >>"${T}/${UPDATE_REL}" <<'EOF'

pub fn pair(s: &KvStorage) -> usize {
    match s {
        KvStorage::Alpha { .. } => 1,
        KvStorage::Beta { .. } => 2,
        _ => 0,
    }
}
EOF
check "the derived bar is half the storage enum" "${T}" 0 "^enum KvStorage variants=4 threshold=2$" match-sites
check "the derived bar is half the quant enum" "${T}" 0 "^enum KvQuant variants=4 threshold=2$" match-sites
check "a two-variant site sits at the derived bar" "${T}" 0 "^match-sites 4$" match-sites

# 5b — a fifth variant moves the bar to 3, and the two-variant site drops out
# with it. The bar and the count move together, which a fixed bar cannot do.
sed -i.bak 's/    Delta { state: MixedKvState, max_seq: i32 },/    Delta { state: MixedKvState, max_seq: i32 },\n    Epsilon { k: Option<QuantK>, max_seq: i32 },/' \
    "${T}/${STORAGE_REL}"
check "a fifth variant moves the derived bar" "${T}" 0 "^enum KvStorage variants=5 threshold=3$" match-sites
check "the two-variant site drops below the moved bar" "${T}" 0 "^match-sites 3$" match-sites

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
check "a body's lines are its braces, not its signature" "${T}" 0 \
    "^body update_gamma file=${UPDATE_REL} line=[0-9]+ lines=3$" update-bodies

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

# 16 — a codec family's own update file joins the population. The bodies are
# the same bodies wherever they sit, so splitting one out must not move the
# count or the line total.
T="${WORK}/family"; build_tree "${T}"
python3 - "${T}/${UPDATE_REL}" "${T}/$(dirname "${UPDATE_REL}")/update_gamma.rs" <<'PYEOF'
import sys

src, dst = sys.argv[1], sys.argv[2]
body = "fn update_gamma() -> usize {\n    2\n}\n"
text = open(src).read()
assert body in text
open(src, "w").write(text.replace(body + "\n", ""))
open(dst, "w").write(body)
PYEOF
check "a family file's bodies join the population" "${T}" 0 "^update-bodies 3$" update-bodies
check "a moved body keeps the line total" "${T}" 0 "^update-body-lines 10$" update-bodies
check "a moved body names the file it sits in" "${T}" 0 \
    "^body update_gamma file=.*update_gamma\\.rs line=[0-9]+ lines=3$" update-bodies

# 17 — every update file is gone. The directory is there, so the refusal names
# the glob and not the directory.
T="${WORK}/nofile"; build_tree "${T}"
rm -f "${T}/${UPDATE_REL}"
check "a glob that matches no file is a refusal" "${T}" 2 "unavailable: .*matches no file" update-bodies

# 18 — the whole update directory is gone.
T="${WORK}/nodir"; build_tree "${T}"
rm -rf "${T}/$(dirname "${UPDATE_REL}")"
check "a missing update directory is a refusal" "${T}" 2 "unavailable: .*is not a directory" update-bodies

# 19 — the reference count, both ways.
T="${WORK}/refs"; build_tree "${T}"
check "variant references are counted" "${T}" 0 "KvStorage=4 distinct=4" refs --file "${UPDATE_REL}"
T="${WORK}/norefs"; build_tree "${T}"
printf 'pub fn nothing() -> usize { 1 }\n' >"${T}/${UPDATE_REL}"
check "a file naming no variant is a refusal" "${T}" 2 "unavailable: .*names no" refs --file "${UPDATE_REL}"

# 18 — a wide match planted in a test file. The producer measures production
# source, so it is not counted; `--include-tests` is the one way to see it.
# Without this case the test-file exclusion could be disabled and every other
# case here would stay green.
T="${WORK}/testfile"; build_tree "${T}"
cat >"${T}/crates/rmlx-kv-quant/src/kvcache/update_tests.rs" <<'EOF'
fn probe(s: &KvStorage) -> usize {
    match s {
        KvStorage::Alpha { .. } => 1,
        KvStorage::Beta { .. } => 2,
        KvStorage::Gamma { .. } => 3,
        KvStorage::Delta { .. } => 4,
    }
}
EOF
check "a site in a test file is not counted" "${T}" 0 "^match-sites 3$" match-sites --threshold 2
check "--include-tests counts the test file's site" "${T}" 0 "^match-sites 4$" match-sites --threshold 2 --include-tests

# 20 — a match over `Option<KvQuant>` names the variants inside `Some(..)`, and
# its `None` arm is a variant of `Option`, not a catch-all. It forces a touch.
T="${WORK}/option"; build_tree "${T}"
cat >>"${T}/${UPDATE_REL}" <<'EOF'

pub fn label_of(q: Option<KvQuant>) -> usize {
    match q {
        Some(KvQuant::One) => 1,
        Some(KvQuant::Two) => 2,
        Some(KvQuant::Three) => 3,
        Some(KvQuant::Four) => 4,
        None => 0,
    }
}
EOF
check "a match over Option<KvQuant> is a site" "${T}" 0 "^match-sites 4$" match-sites --threshold 2
check "its None arm is not a catch-all" "${T}" 0 "^forcing-sites 4$" match-sites --threshold 2

# 21 — the `Self::` spelling inside `impl KvStorage` and `impl KvQuant`. The
# compiler forces a touch on each of these exactly as on a `KvStorage::` arm.
# Two enums, so a producer that resolves `Self` to one fixed enum fails one of
# the two.
T="${WORK}/selfpath"; build_tree "${T}"
cat >>"${T}/${STORAGE_REL}" <<'EOF'

impl KvStorage {
    pub fn reset(&mut self) -> usize {
        match self {
            Self::Alpha { .. } => 1,
            Self::Beta { .. } => 2,
            Self::Gamma { .. } => 3,
            Self::Delta { .. } => 4,
        }
    }
}
EOF
cat >>"${T}/${QUANT_REL}" <<'EOF'

impl KvQuant {
    pub fn index(&self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
        }
    }
}
EOF
check "Self:: arms in both impls are two more sites" "${T}" 0 "^match-sites 5$" match-sites --threshold 2
check "the storage impl's Self:: site is its file's second" "${T}" 0 "^file ${STORAGE_REL} 2$" match-sites --threshold 2
check "the quant impl's Self:: site is its file's second" "${T}" 0 "^file ${QUANT_REL} 2$" match-sites --threshold 2

# 22 — an alias import. `S::Alpha` is `KvStorage::Alpha` to the compiler.
T="${WORK}/alias"; build_tree "${T}"
cat >>"${T}/${UPDATE_REL}" <<'EOF'

use KvStorage as S;

pub fn aliased(s: &S) -> usize {
    match s {
        S::Alpha { .. } => 1,
        S::Beta { .. } => 2,
        S::Gamma { .. } => 3,
        S::Delta { .. } => 4,
    }
}
EOF
check "an alias-path match is a site" "${T}" 0 "^match-sites 4$" match-sites --threshold 2

# 23 — a glob import. The arms name bare variants.
T="${WORK}/glob"; build_tree "${T}"
cat >>"${T}/${UPDATE_REL}" <<'EOF'

use KvQuant::*;

pub fn bare(q: KvQuant) -> usize {
    match q {
        One => 1,
        Two => 2,
        Three => 3,
        Four => 4,
    }
}
EOF
check "a glob-import match is a site" "${T}" 0 "^match-sites 4$" match-sites --threshold 2

# 24 — the per-codec dispatch moved one level down, into an enum a KvStorage
# variant holds. A new codec with a new store still forces a touch there, so a
# restructure that only moves the sites must not read as a reduction.
T="${WORK}/slotenum"; build_tree "${T}"
sed -i.bak 's/    Alpha { k: Option<QuantK>, v: Option<QuantV>, max_seq: i32 },/    Alpha { k: KSlot, v: Option<QuantV>, max_seq: i32 },/' \
    "${T}/${STORAGE_REL}"
cat >>"${T}/${STORAGE_REL}" <<'EOF'

pub enum KSlot {
    Q8(QuantK),
    Turbo(QuantKTurbo),
    Iso(QuantIso),
    Rotor(QuantRotor),
}

pub fn slot_bytes(k: &KSlot) -> usize {
    match k {
        KSlot::Q8(_) => 1,
        KSlot::Turbo(_) => 2,
        KSlot::Iso(_) => 3,
        KSlot::Rotor(_) => 4,
    }
}
EOF
check "an enum a KvStorage field holds joins the codec enums" "${T}" 0 "^enum KSlot variants=4 threshold=2$" match-sites --threshold 2
check "a match over an enum a KvStorage field holds is a site" "${T}" 0 "^match-sites 4$" match-sites --threshold 2

# 25 — a forcing site rewritten as a `matches!` subset. The count drops and a
# new codec now defaults to `false` there with no compile error. The producer
# must report the subset, so the drop cannot read as a reduction.
T="${WORK}/matchesmacro"; build_tree "${T}"
cat >"${T}/${QUANT_REL}" <<'EOF'
pub enum KvQuant {
    One,
    Two,
    Three,
    Four,
}

pub fn low(q: KvQuant) -> bool {
    matches!(q, KvQuant::One | KvQuant::Two)
}
EOF
check "a matches! subset is not a match site" "${T}" 0 "^match-sites 2$" match-sites --threshold 2
check "a matches! subset is reported as a subset site" "${T}" 0 "^subset-sites 1$" match-sites --threshold 2

# 26 — a string spelling table. Its patterns are string literals and its arm
# bodies name every variant. A new codec that is missing here compiles and
# cannot be parsed.
T="${WORK}/spelling"; build_tree "${T}"
cat >>"${T}/${QUANT_REL}" <<'EOF'

pub fn parse(s: &str) -> Option<KvQuant> {
    match s {
        "one" => Some(KvQuant::One),
        "two" => Some(KvQuant::Two),
        "three" => Some(KvQuant::Three),
        "four" => Some(KvQuant::Four),
        _ => None,
    }
}
EOF
check "a spelling table is not a match site" "${T}" 0 "^match-sites 3$" match-sites --threshold 2
check "a spelling table is reported as a table site" "${T}" 0 "^table-sites 1$" match-sites --threshold 2

# 27 — negative control for case 21: `Self::` arms inside `impl OtherEnum`,
# whose variant names are the same as the storage enum's. `Self` is
# `OtherEnum` there, so the match is not a site.
T="${WORK}/otherimpl"; build_tree "${T}"
cat >>"${T}/${UPDATE_REL}" <<'EOF'

pub enum OtherEnum {
    Alpha,
    Beta,
    Gamma,
    Delta,
}

impl OtherEnum {
    pub fn index(&self) -> usize {
        match self {
            Self::Alpha => 1,
            Self::Beta => 2,
            Self::Gamma => 3,
            Self::Delta => 4,
        }
    }
}
EOF
check "Self:: arms in an impl of another enum are not a site" "${T}" 0 "^match-sites 3$" match-sites --threshold 2

# 28 — negative control for case 22: an alias of an unrelated enum, spelling
# the storage enum's variant names.
T="${WORK}/otheralias"; build_tree "${T}"
cat >>"${T}/${UPDATE_REL}" <<'EOF'

use OtherEnum as S;

pub fn aliased(s: &S) -> usize {
    match s {
        S::Alpha => 1,
        S::Beta => 2,
        S::Gamma => 3,
        S::Delta => 4,
    }
}
EOF
check "an alias of an unrelated enum is not a site" "${T}" 0 "^match-sites 3$" match-sites --threshold 2

# 29 — negative control for case 24: the same slot enum and the same match,
# but no KvStorage field holds the enum. Only a held enum joins the codec
# enums, so a producer that counted every enum in the tree fails here.
T="${WORK}/freeenum"; build_tree "${T}"
cat >>"${T}/${STORAGE_REL}" <<'EOF'

pub enum KSlot {
    Q8(QuantK),
    Turbo(QuantKTurbo),
    Iso(QuantIso),
    Rotor(QuantRotor),
}

pub fn slot_bytes(k: &KSlot) -> usize {
    match k {
        KSlot::Q8(_) => 1,
        KSlot::Turbo(_) => 2,
        KSlot::Iso(_) => 3,
        KSlot::Rotor(_) => 4,
    }
}
EOF
check "an enum no KvStorage field holds is not a site" "${T}" 0 "^match-sites 3$" match-sites --threshold 2

# 30 — a held enum that two files define. The producer cannot tell which one
# the field holds, and says so rather than pick one.
T="${WORK}/twoslots"; build_tree "${T}"
sed -i.bak 's/    Alpha { k: Option<QuantK>, v: Option<QuantV>, max_seq: i32 },/    Alpha { k: KSlot, v: Option<QuantV>, max_seq: i32 },/' \
    "${T}/${STORAGE_REL}"
printf 'pub enum KSlot { Q8(QuantK), Turbo(QuantKTurbo) }\n' >>"${T}/${STORAGE_REL}"
printf 'pub enum KSlot { Iso(QuantIso) }\n' >>"${T}/${UPDATE_REL}"
check "a held enum defined twice is a refusal" "${T}" 2 "unavailable: enum KSlot, held by a KvStorage field, is defined in 2 files" \
    match-sites --threshold 2

# 31 — negative control for case 25: a `matches!` over an enum that is not a
# codec enum is not a subset site.
T="${WORK}/othermatches"; build_tree "${T}"
cat >>"${T}/${UPDATE_REL}" <<'EOF'

pub fn low(o: Other) -> bool {
    matches!(o, Other::Alpha | Other::Beta)
}
EOF
check "a matches! over a non-codec enum is not a subset site" "${T}" 0 "^subset-sites 0$" match-sites --threshold 2

# 32 — negative control for case 26: a string table whose bodies name one of
# the four variants, under the bar. It is a lookup, not a spelling table.
T="${WORK}/shorttable"; build_tree "${T}"
cat >>"${T}/${QUANT_REL}" <<'EOF'

pub fn parse_one(s: &str) -> Option<KvQuant> {
    match s {
        "one" => Some(KvQuant::One),
        "uno" => Some(KvQuant::One),
        _ => None,
    }
}
EOF
check "a string table under the bar is not a table site" "${T}" 0 "^table-sites 0$" match-sites --threshold 2

# 33 — a storage enum with a `None` variant, a glob import of it, and a match
# over `Option<..>` with a `None =>` arm, in one file. The bare-variant match
# reaches the bar of 3 only through its `None` arm, so the glob must resolve
# that `None` to the storage enum. The `Option<KvQuant>` match is one site, not
# one per enum. The `Option<KvStorage>` match names two variants inside
# `Some(..)`; its `None` is `Option`'s, and reads as a third variant only in a
# producer that ignores the `Some(..)` beside it.
T="${WORK}/globnone"; build_tree "${T}"
sed -i.bak 's/    Delta { state: MixedKvState, max_seq: i32 },/    Delta { state: MixedKvState, max_seq: i32 },\n    None { max_seq: i32 },/' \
    "${T}/${STORAGE_REL}"
cat >>"${T}/${UPDATE_REL}" <<'EOF'

use KvStorage::*;

pub fn bare(s: &KvStorage) -> usize {
    match s {
        Alpha { .. } => 1,
        Beta { .. } => 2,
        None { .. } => 0,
        _ => 9,
    }
}

pub fn label_of(q: Option<KvQuant>) -> usize {
    match q {
        Some(KvQuant::One) => 1,
        Some(KvQuant::Two) => 2,
        Some(KvQuant::Three) => 3,
        Some(KvQuant::Four) => 4,
        None => 0,
    }
}

pub fn maybe(s: Option<&KvStorage>) -> usize {
    match s {
        Some(Alpha { .. }) => 1,
        Some(Beta { .. }) => 2,
        Some(_) => 3,
        None => 0,
    }
}
EOF
check "a glob-imported None variant counts toward the bar" "${T}" 0 \
    "^site ${UPDATE_REL}:[0-9]+ enum=KvStorage variants=3 arms=4 catch_all=yes$" match-sites
check "Option's None beside Some(..) is not a storage variant" "${T}" 0 "^match-sites 5$" match-sites
check "the Option<KvQuant> match counts once and forces a touch" "${T}" 0 "^forcing-sites 4$" match-sites

# 34 — the real tree, pinned. A change that moves a figure re-pins it in the
# same change, so a restructure states its reduction here, and a new subset
# or table site cannot land unnoticed.
check "real tree: match sites" "${REPO_ROOT}" 0 "^match-sites 29$" match-sites
check "real tree: forcing sites" "${REPO_ROOT}" 0 "^forcing-sites 29$" match-sites
check "real tree: subset sites" "${REPO_ROOT}" 0 "^subset-sites 64$" match-sites
check "real tree: table sites" "${REPO_ROOT}" 0 "^table-sites 2$" match-sites

if [ "${failures}" -gt 0 ]; then
    echo >&2
    echo "ERROR: ${failures} case(s) did not reproduce the expected behaviour." >&2
    exit 1
fi
echo "OK: the KV update census reports every planted figure, and refuses every tree it cannot measure."
