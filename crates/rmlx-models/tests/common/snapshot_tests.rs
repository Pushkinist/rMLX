//! Unit tests for the golden-token snapshot resolver and its run/skip/fail
//! decision.
//!
//! These are the only part of the golden harness that runs without a model, and
//! they exist because the harness's failure mode is silence: a golden whose
//! snapshot does not resolve returns before asserting anything and libtest
//! reports `ok`. The tests below fix, in code, which configurations must arm the
//! gate, which may skip it, and which must break it.
//!
//! The oracle is the directory tree each test builds by hand — no shared
//! arithmetic with the resolver, which only ever asks whether a `config.json`
//! is present at a path it composed.

use std::path::{Path, PathBuf};

use super::{gate, pick_snapshot, Gate, GoldenModel, Snapshot, Source};

const SLUG: &str = "vendor__model-8b-2bit";
const ARCH: &str = "ExampleForCausalLM";

const MODEL: GoldenModel = GoldenModel {
    slug: SLUG,
    archs: &[ARCH],
};

/// Create `<parent>/<name>/config.json` and return the snapshot directory.
fn make_snapshot(parent: &Path, name: &str) -> PathBuf {
    let dir = parent.join(name);
    std::fs::create_dir_all(&dir).expect("create snapshot dir");
    std::fs::write(dir.join("config.json"), b"{}").expect("write config.json");
    dir
}

fn as_str(p: &Path) -> &str {
    p.to_str().expect("temp paths are utf-8")
}

/// A models root alone must arm the golden. Every `make` target exports
/// `RMLX_O_MODELS_ROOT`, so this is the configuration in which the gate
/// actually runs; without it a golden needed a per-run variable nobody sets and
/// reported success by returning.
#[test]
fn models_root_alone_resolves_the_snapshot() {
    let root = tempfile::tempdir().expect("tempdir");
    let dir = make_snapshot(root.path(), SLUG);

    let got = pick_snapshot(None, Some(as_str(root.path())), SLUG);

    assert_eq!(
        got,
        Snapshot::Found {
            path: dir,
            from: Source::ModelsRoot
        },
        "a models root holding the slug must resolve it"
    );
}

/// The single-model override outranks the root: it is what
/// `make model-check-full MODEL=…` sets, and that target means one model.
#[test]
fn single_model_override_outranks_the_root() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("root");
    std::fs::create_dir_all(&root).expect("create root");
    make_snapshot(&root, SLUG);
    let single = make_snapshot(tmp.path(), "single");

    let got = pick_snapshot(Some(as_str(&single)), Some(as_str(&root)), SLUG);

    assert_eq!(
        got,
        Snapshot::Found {
            path: single,
            from: Source::SingleModelOverride
        }
    );
}

/// The resolver reads exactly two variables. A per-architecture
/// `RMLX_TEST_MODEL_*` export — which the documented workspace-test workflow
/// sets persistently in a dev shell — must not reach a byte-exact fixture: a
/// same-family substitute (a QAT rebuild, a re-quantized sibling) passes the
/// architecture check and would land as a token mismatch indistinguishable from
/// a decode regression.
///
/// The signature is the enforcement — `pick_snapshot` has nowhere to put a
/// third source — so this test pins the arity and the resolved outcome: with a
/// root configured, the answer is the slug path no matter what else is exported.
#[test]
fn only_the_root_and_the_single_model_override_are_consulted() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("root");
    std::fs::create_dir_all(&root).expect("create root");
    let canonical = make_snapshot(&root, SLUG);
    // A same-arch snapshot elsewhere on disk, of the kind a per-arch variable
    // would point at. It must not win.
    make_snapshot(tmp.path(), "vendor__model-8b-qat-4bit");

    let got = pick_snapshot(None, Some(as_str(&root)), SLUG);

    assert_eq!(
        got,
        Snapshot::Found {
            path: canonical,
            from: Source::ModelsRoot
        },
        "resolution must come from the slug under the root, not from any other \
         snapshot present on the machine"
    );
}

/// A variable naming a directory that is not a snapshot is configuration
/// present and wrong. Reporting absence there is how a typo'd or stale path
/// turns into a green run that asserted nothing.
#[test]
fn a_named_path_that_is_not_a_snapshot_is_misconfigured_not_absent() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let empty = tmp.path().join("no-config-json");
    std::fs::create_dir_all(&empty).expect("create dir");

    let got = pick_snapshot(Some(as_str(&empty)), None, SLUG);

    assert!(
        matches!(got, Snapshot::Misconfigured(_)),
        "a variable naming {} must be a hard failure, got {got:?}",
        empty.display()
    );
}

/// A named path that does not exist at all is the same class as one that exists
/// without a config: the variable is set and wrong.
#[test]
fn a_named_path_that_does_not_exist_is_misconfigured() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let missing = tmp.path().join("was-moved-away");

    let got = pick_snapshot(Some(as_str(&missing)), None, SLUG);

    assert!(
        matches!(got, Snapshot::Misconfigured(_)),
        "got {got:?} for a path that does not exist"
    );
}

/// A models root that does not hold this slug is an absence, not an error.
/// Nobody holds every snapshot, and a hard failure here would make the gate
/// permanently red on any partial mirror.
#[test]
fn a_root_without_the_slug_is_absent() {
    let root = tempfile::tempdir().expect("tempdir");
    make_snapshot(root.path(), "some-other-model");

    let got = pick_snapshot(None, Some(as_str(root.path())), SLUG);

    assert!(
        matches!(got, Snapshot::Absent(_)),
        "got {got:?} for a root holding a different model"
    );
}

/// The Makefile falls back to a repo-local `models/` dir that need not exist, so
/// a root pointing nowhere must skip rather than break every golden for every
/// developer who has not set one.
#[test]
fn a_root_that_does_not_exist_is_absent() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("models");

    let got = pick_snapshot(None, Some(as_str(&root)), SLUG);

    assert!(matches!(got, Snapshot::Absent(_)), "got {got:?}");
}

/// No configuration at all — a developer without weights, and the hosted CI.
#[test]
fn nothing_configured_is_absent() {
    let got = pick_snapshot(None, None, SLUG);
    assert!(matches!(got, Snapshot::Absent(_)), "got {got:?}");
}

/// An exported-but-empty variable is how a shell spells "unset". Treating it as
/// a path makes every lookup a misconfiguration and takes the whole suite down.
#[test]
fn empty_variables_are_treated_as_unset() {
    let root = tempfile::tempdir().expect("tempdir");
    let dir = make_snapshot(root.path(), SLUG);

    let got = pick_snapshot(Some(""), Some(as_str(root.path())), SLUG);

    assert_eq!(
        got,
        Snapshot::Found {
            path: dir,
            from: Source::ModelsRoot
        },
        "an empty override must fall through to the root"
    );

    assert!(
        matches!(pick_snapshot(None, Some(""), SLUG), Snapshot::Absent(_)),
        "an empty root must not resolve to the filesystem root"
    );
}

/// A matching arch runs, whichever source produced the path.
#[test]
fn a_matching_arch_runs_from_every_source() {
    for from in [Source::SingleModelOverride, Source::ModelsRoot] {
        let path = PathBuf::from("/snapshots").join(SLUG);
        let snapshot = Snapshot::Found {
            path: path.clone(),
            from,
        };
        assert_eq!(gate(snapshot, ARCH, &MODEL), Gate::Run(path), "{from:?}");
    }
}

/// The single-model override names ONE model for a run of per-arch goldens, so
/// the goldens for the other architectures are meant to stand down.
#[test]
fn arch_mismatch_under_the_single_model_override_skips() {
    let snapshot = Snapshot::Found {
        path: PathBuf::from("/snapshots/some-other-arch"),
        from: Source::SingleModelOverride,
    };
    assert!(
        matches!(gate(snapshot, "OtherForCausalLM", &MODEL), Gate::Skip(_)),
        "one model for a whole run must not fail the goldens it does not cover"
    );
}

/// The slug names THIS golden's snapshot. A mismatch there is a wrong pointer,
/// and skipping on it hides the wrong pointer behind a green run.
#[test]
fn arch_mismatch_on_the_slug_snapshot_fails() {
    let snapshot = Snapshot::Found {
        path: PathBuf::from("/snapshots").join(SLUG),
        from: Source::ModelsRoot,
    };
    assert!(
        matches!(gate(snapshot, "OtherForCausalLM", &MODEL), Gate::Fail(_)),
        "the slug named the snapshot, so a wrong arch must break the gate"
    );
}

/// An unreadable or absent `config.json` yields an empty arch string. That must
/// not accidentally match a golden's expected-arch list.
#[test]
fn an_unreadable_config_does_not_match_any_arch() {
    let snapshot = Snapshot::Found {
        path: PathBuf::from("/snapshots").join(SLUG),
        from: Source::ModelsRoot,
    };
    assert!(matches!(gate(snapshot, "", &MODEL), Gate::Fail(_)));
}

/// Absence skips, misconfiguration fails — the whole point of keeping the two
/// apart.
#[test]
fn absence_skips_and_misconfiguration_fails() {
    assert!(matches!(
        gate(Snapshot::Absent("no root".to_owned()), "", &MODEL),
        Gate::Skip(_)
    ));
    assert!(matches!(
        gate(Snapshot::Misconfigured("bad path".to_owned()), "", &MODEL),
        Gate::Fail(_)
    ));
}
