//! Unit tests for the golden-token snapshot probes, the resolution choice, and
//! the decision-to-return mapping.
//!
//! These are the only part of the golden harness that runs without a model, and
//! they exist because the harness's failure mode is silence: a golden whose
//! snapshot does not resolve returns before asserting anything and libtest
//! reports `ok`. The cases below fix, in code, which configurations must arm the
//! gate, which may skip it, and which must break it.
//!
//! **Oracle independence, stated precisely.** The verdict table — which
//! configuration runs, skips or fails — shares nothing with the code under test:
//! each case names its expected outcome as a literal. The *path composition*
//! does overlap: `make_snapshot` writes the same `config.json` /
//! `tokenizer.json` names [`super::SNAPSHOT_FILES`] lists, and joins the slug to
//! the root the way [`super::slug_snapshot`] does. A change to either of those
//! two conventions would move test and subject together, so they are pinned by
//! `snapshot_files_are_the_ones_the_harness_opens` reading the constant against
//! the literals the harness passes to `Tokenizer::from_file` / `load_model`.
//!
//! **Duplication.** `tests/common/mod.rs` is compiled into seven test binaries,
//! so these cases run seven times under `make test`. That is inherent to the
//! `tests/common` module pattern — the alternative is a helper crate for a
//! hundred lines of resolution logic. They are pure, sub-millisecond and use
//! per-case tempdirs, so the cost is accepted rather than worked around.

use std::path::{Path, PathBuf};

use super::{apply, choose, override_snapshot, slug_snapshot, Gate, GoldenModel, Snapshot};

const SLUG: &str = "vendor__model-8b-2bit";
const ARCH: &str = "ExampleForCausalLM";
const OTHER_ARCH: &str = "OtherForCausalLM";

const MODEL: GoldenModel = GoldenModel {
    slug: SLUG,
    archs: &[ARCH],
};

/// Create `<parent>/<name>/` holding every file a runnable snapshot needs.
fn make_snapshot(parent: &Path, name: &str) -> PathBuf {
    let dir = parent.join(name);
    std::fs::create_dir_all(&dir).expect("create snapshot dir");
    for f in super::SNAPSHOT_FILES {
        std::fs::write(dir.join(f), b"{}").expect("write snapshot file");
    }
    dir
}

fn as_str(p: &Path) -> &str {
    p.to_str().expect("temp paths are utf-8")
}

/// Pin the required-file list against what the harness actually opens by name,
/// so "runnable" cannot drift away from "loads without panicking".
#[test]
fn snapshot_files_are_the_ones_the_harness_opens() {
    assert!(
        super::SNAPSHOT_FILES.contains(&"config.json"),
        "arch::load_model and model_arch both read config.json"
    );
    assert!(
        super::SNAPSHOT_FILES.contains(&"tokenizer.json"),
        "run_golden_test does Tokenizer::from_file(model_path.join(\"tokenizer.json\"))"
    );
}

// ── slug probe ───────────────────────────────────────────────────────────

/// A models root alone must arm the golden. Every `make` target exports
/// `RMLX_O_MODELS_ROOT`, so this is the configuration in which the gate
/// actually runs; without it a golden needed a per-run variable nobody sets and
/// reported success by returning.
#[test]
fn models_root_alone_resolves_the_snapshot() {
    let root = tempfile::tempdir().expect("tempdir");
    let dir = make_snapshot(root.path(), SLUG);

    assert_eq!(
        slug_snapshot(Some(as_str(root.path())), SLUG),
        Snapshot::Found(dir),
        "a models root holding the slug must resolve it"
    );
}

/// An existing root that does not hold this slug is an absence, not an error.
/// Nobody holds every snapshot, and failing here would make the gate
/// permanently red on any partial mirror.
#[test]
fn a_root_without_the_slug_is_absent() {
    let root = tempfile::tempdir().expect("tempdir");
    make_snapshot(root.path(), "some-other-model");

    let got = slug_snapshot(Some(as_str(root.path())), SLUG);

    assert!(
        matches!(got, Snapshot::Absent(_)),
        "got {got:?} for a root holding a different model"
    );
}

/// A root that is set but is not an existing directory is one keystroke
/// disarming all five gates at once — the widest blast radius in this harness.
/// It must break the run, not skip it.
#[test]
fn a_root_that_does_not_exist_is_misconfigured() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("o-modles-typo");

    let got = slug_snapshot(Some(as_str(&root)), SLUG);

    assert!(
        matches!(got, Snapshot::Misconfigured(_)),
        "got {got:?} for a root that is not a directory"
    );
}

/// A root pointing at a plain file is the same class as one pointing nowhere.
#[test]
fn a_root_that_is_a_file_is_misconfigured() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("not-a-dir");
    std::fs::write(&root, b"x").expect("write file");

    let got = slug_snapshot(Some(as_str(&root)), SLUG);

    assert!(matches!(got, Snapshot::Misconfigured(_)), "got {got:?}");
}

/// An interrupted download leaves `config.json` and nothing else — that is the
/// first file written. It must read as absent, exactly like a directory that is
/// not there at all, rather than resolving and panicking later at the
/// tokenizer's `expect`.
#[test]
fn a_config_only_directory_under_the_root_is_absent() {
    let root = tempfile::tempdir().expect("tempdir");
    let partial = root.path().join(SLUG);
    std::fs::create_dir_all(&partial).expect("create dir");
    std::fs::write(partial.join("config.json"), b"{}").expect("write config.json");

    let got = slug_snapshot(Some(as_str(root.path())), SLUG);

    assert!(
        matches!(got, Snapshot::Absent(_)),
        "a half-written snapshot must not resolve as runnable; got {got:?}"
    );
}

/// No configuration at all — a developer without weights, and the hosted CI.
#[test]
fn nothing_configured_is_absent() {
    let got = slug_snapshot(None, SLUG);
    assert!(matches!(got, Snapshot::Absent(_)), "got {got:?}");
    assert!(
        matches!(slug_snapshot(Some(""), SLUG), Snapshot::Absent(_)),
        "an empty root must not resolve to the filesystem root"
    );
}

// ── override probe ───────────────────────────────────────────────────────

/// Unset, or the empty string a shell uses to mean unset.
#[test]
fn an_unset_or_empty_override_is_none() {
    assert!(override_snapshot(None).is_none());
    assert!(override_snapshot(Some("")).is_none());
}

#[test]
fn an_override_naming_a_runnable_snapshot_is_found() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = make_snapshot(tmp.path(), "single");

    assert_eq!(
        override_snapshot(Some(as_str(&dir))),
        Some(Snapshot::Found(dir))
    );
}

/// A named path that is not a runnable snapshot is configuration present and
/// wrong — a typo, a moved snapshot, a half-finished download. Reporting
/// absence there is how it turns into a green run that asserted nothing. All
/// three shapes are distinct on disk and a laxer check would pass some of them.
#[test]
fn an_override_naming_anything_unrunnable_is_misconfigured() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let absent = tmp.path().join("was-moved-away");

    let empty = tmp.path().join("no-files");
    std::fs::create_dir_all(&empty).expect("create dir");

    let partial = tmp.path().join("config-only");
    std::fs::create_dir_all(&partial).expect("create dir");
    std::fs::write(partial.join("config.json"), b"{}").expect("write config.json");

    for dir in [&absent, &empty, &partial] {
        let got = override_snapshot(Some(as_str(dir)));
        assert!(
            matches!(got, Some(Snapshot::Misconfigured(_))),
            "{} must be a hard failure, got {got:?}",
            dir.display()
        );
    }
}

// ── choose: the resolution decision ──────────────────────────────────────

fn found(p: &str) -> Snapshot {
    Snapshot::Found(PathBuf::from(p))
}

/// With no override, the slug decides.
#[test]
fn the_slug_runs_when_it_matches_and_no_override_is_set() {
    assert_eq!(
        choose(None, "", found("/snap/slug"), ARCH, &MODEL),
        Gate::Run(PathBuf::from("/snap/slug"))
    );
}

/// An override for THIS architecture wins over the slug: that is what makes
/// `RMLX_REGEN_GOLDENS=1 RMLX_KV_TEST_MODEL=<path>` record from the named
/// snapshot, and what lets one golden be compared against a non-slug checkpoint
/// deliberately. Ranking the slug first would silently ignore the named path.
#[test]
fn a_matching_override_outranks_the_slug() {
    assert_eq!(
        choose(
            Some(found("/named")),
            ARCH,
            found("/snap/slug"),
            ARCH,
            &MODEL
        ),
        Gate::Run(PathBuf::from("/named")),
        "the named path must win, or regen records the wrong checkpoint"
    );
}

/// The override names another architecture — a KV developer's export. It says
/// nothing about this golden, so the slug must still arm it. Standing down here
/// is the original silent-skip defect.
#[test]
fn an_override_for_another_arch_falls_through_to_the_slug() {
    assert_eq!(
        choose(
            Some(found("/named-gemma4")),
            OTHER_ARCH,
            found("/snap/slug"),
            ARCH,
            &MODEL
        ),
        Gate::Run(PathBuf::from("/snap/slug")),
        "an override for another arch must not disarm this golden"
    );
}

/// Fall-through with nothing to fall through to still skips, and the reason
/// names both halves so the operator can see the override stood down.
#[test]
fn an_override_for_another_arch_with_no_slug_skips_naming_both() {
    let got = choose(
        Some(found("/named-gemma4")),
        OTHER_ARCH,
        Snapshot::Absent("root holds no slug".to_owned()),
        "",
        &MODEL,
    );
    match got {
        Gate::Skip(why) => {
            assert!(why.contains("root holds no slug"), "{why}");
            assert!(why.contains(super::SINGLE_MODEL_VAR), "{why}");
        }
        other => panic!("expected Skip naming both halves, got {other:?}"),
    }
}

/// A misconfigured override fails even when the slug would have resolved — the
/// operator named a path and it is wrong, and a fall-through would bury that.
#[test]
fn a_misconfigured_override_fails_even_when_the_slug_resolves() {
    assert_eq!(
        choose(
            Some(Snapshot::Misconfigured("bad override".to_owned())),
            "",
            found("/snap/slug"),
            ARCH,
            &MODEL
        ),
        Gate::Fail("bad override".to_owned()),
        "a wrong pointer must not be masked by a working slug"
    );
}

/// The slug names THIS golden's snapshot, so a wrong arch there is a wrong
/// pointer, not a different model's business.
#[test]
fn arch_mismatch_on_the_slug_snapshot_fails() {
    let got = choose(None, "", found("/snap/slug"), OTHER_ARCH, &MODEL);
    assert!(
        matches!(got, Gate::Fail(_)),
        "the slug named the snapshot, so a wrong arch must break the gate; got {got:?}"
    );
}

/// An unreadable or absent `config.json` yields an empty arch string. That must
/// not accidentally match a golden's expected-arch list.
#[test]
fn an_unreadable_config_does_not_match_any_arch() {
    let got = choose(None, "", found("/snap/slug"), "", &MODEL);
    assert!(matches!(got, Gate::Fail(_)), "got {got:?}");
}

/// Absence skips, misconfiguration fails — the split the whole design rests on,
/// pinned at the decision level where no `Found` path is involved.
#[test]
fn slug_absence_skips_and_slug_misconfiguration_fails() {
    assert!(matches!(
        choose(
            None,
            "",
            Snapshot::Absent("nothing set".to_owned()),
            "",
            &MODEL
        ),
        Gate::Skip(_)
    ));
    assert!(matches!(
        choose(
            None,
            "",
            Snapshot::Misconfigured("bad root".to_owned()),
            "",
            &MODEL
        ),
        Gate::Fail(_)
    ));
}

// ── apply: decision to return value ──────────────────────────────────────

#[test]
fn apply_returns_the_path_on_run() {
    assert_eq!(
        apply(Gate::Run(PathBuf::from("/snap")), "t"),
        Some(PathBuf::from("/snap"))
    );
}

#[test]
fn apply_returns_none_on_skip() {
    assert_eq!(apply(Gate::Skip("no model".to_owned()), "t"), None);
}

/// The step that makes a wrong pointer visible. Without the panic a `Fail`
/// would collapse back into a silent skip, which is the defect this harness
/// exists to close.
#[test]
#[should_panic(expected = "t: bad pointer")]
fn apply_panics_on_fail() {
    apply(Gate::Fail("bad pointer".to_owned()), "t");
}
