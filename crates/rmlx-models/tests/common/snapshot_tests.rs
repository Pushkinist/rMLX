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
//! does overlap: `make_snapshot` writes files under a directory it joins to a
//! slug the way [`super::slug_snapshot`] does. That overlap is itself pinned by
//! `a_dir_built_from_the_constants_satisfies_every_harness_call_site`, whose
//! oracle is a list transcribed from the call sites in `rmlx-loader` and
//! `run_golden_test` rather than from the constants under test — so an
//! under-specified constant fails there instead of being ratified.
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

/// Not `regen`: the ordinary read path, where a stood-down override is a note.
const CHECKING: bool = false;
/// `RMLX_REGEN_GOLDENS` set: the harness is about to WRITE a committed fixture.
const RECORDING: bool = true;

/// Every path the golden harness opens BY NAME, transcribed from the call sites
/// rather than from the constants under test:
///
/// * `run_golden_test` → `Tokenizer::from_file(model_path.join("tokenizer.json"))`
/// * `model_arch` / `arch::load_model` → `config.json`
/// * `rmlx_loader::load_shard_index` (`rmlx-loader/src/shards.rs`) → tries
///   `model.safetensors.index.json`, else `model.safetensors`, else errors.
const HARNESS_OPENS_ALL: [&str; 2] = ["config.json", "tokenizer.json"];
const HARNESS_OPENS_ANY: [&str; 2] = ["model.safetensors.index.json", "model.safetensors"];

/// Create `<parent>/<name>/` holding every file a runnable snapshot needs,
/// with `architectures[0]` set to `arch`.
fn make_snapshot(parent: &Path, name: &str, arch: &str) -> PathBuf {
    let dir = parent.join(name);
    std::fs::create_dir_all(&dir).expect("create snapshot dir");
    std::fs::write(
        dir.join("config.json"),
        format!(r#"{{"architectures":["{arch}"]}}"#),
    )
    .expect("write config.json");
    std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write tokenizer.json");
    std::fs::write(dir.join("model.safetensors"), b"").expect("write weights");
    dir
}

fn as_str(p: &Path) -> &str {
    p.to_str().expect("temp paths are utf-8")
}

/// Build a directory from the harness's own required-file constants and assert
/// every name the harness opens is present in it.
///
/// This is the falsifiable direction the previous membership assertion could
/// not reach: it fails when the constants UNDER-specify a snapshot, which is
/// how the list was actually wrong — it omitted the weight entrypoints, so a
/// download interrupted before its multi-GB shards arrived resolved as runnable
/// and panicked inside `load_shard_index`.
#[test]
fn a_dir_built_from_the_constants_satisfies_every_harness_call_site() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("exactly-the-constants");
    std::fs::create_dir_all(&dir).expect("create dir");
    for f in super::REQUIRED_FILES {
        std::fs::write(dir.join(f), b"{}").expect("write required file");
    }
    // The weight side is a disjunction in the loader, so one entrypoint is what
    // a real snapshot carries; take the first the constants name.
    let first_weight = super::WEIGHT_ENTRYPOINTS
        .first()
        .expect("at least one weight entrypoint");
    std::fs::write(dir.join(first_weight), b"").expect("write weight entrypoint");

    for opened in HARNESS_OPENS_ALL {
        assert!(
            dir.join(opened).exists(),
            "the harness opens {opened} by name, but a directory built from \
             REQUIRED_FILES/WEIGHT_ENTRYPOINTS does not contain it"
        );
    }
    assert!(
        HARNESS_OPENS_ANY.iter().any(|f| dir.join(f).exists()),
        "load_shard_index needs one of {HARNESS_OPENS_ANY:?}, but a directory \
         built from the constants contains neither"
    );
    assert!(
        super::is_snapshot_dir(&dir),
        "a directory built from the constants must satisfy the check that reads them"
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
    let dir = make_snapshot(root.path(), SLUG, ARCH);

    assert_eq!(
        slug_snapshot(Some(as_str(root.path())), SLUG),
        Snapshot::Found {
            path: dir,
            arch: ARCH.to_owned()
        },
        "a models root holding the slug must resolve it, carrying its arch"
    );
}

/// An existing root that does not hold this slug is an absence, not an error.
/// Nobody holds every snapshot, and failing here would make the gate
/// permanently red on any partial mirror.
#[test]
fn a_root_without_the_slug_is_absent() {
    let root = tempfile::tempdir().expect("tempdir");
    make_snapshot(root.path(), "some-other-model", ARCH);

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

/// The two half-written shapes a download actually leaves. Both must read as
/// absent — exactly like a directory that is not there at all — rather than
/// resolving and panicking later at a tokenizer or shard-index `expect`.
///
/// The no-shards case is the modal one: small JSON files land first and the
/// multi-GB shards last.
#[test]
fn a_half_written_snapshot_under_the_root_is_absent() {
    for present in [
        vec!["config.json"],
        vec!["config.json", "tokenizer.json"], // JSONs done, shards still coming
    ] {
        let root = tempfile::tempdir().expect("tempdir");
        let partial = root.path().join(SLUG);
        std::fs::create_dir_all(&partial).expect("create dir");
        for f in &present {
            std::fs::write(partial.join(f), b"{}").expect("write partial file");
        }

        let got = slug_snapshot(Some(as_str(root.path())), SLUG);

        assert!(
            matches!(got, Snapshot::Absent(_)),
            "a snapshot holding only {present:?} must not resolve as runnable; got {got:?}"
        );
    }
}

/// Either weight entrypoint completes a snapshot — sharded checkpoints ship the
/// index, single-file ones ship the bare `model.safetensors`. Requiring a
/// specific one would reject half the real snapshots.
#[test]
fn either_weight_entrypoint_completes_a_snapshot() {
    for weight in super::WEIGHT_ENTRYPOINTS {
        let root = tempfile::tempdir().expect("tempdir");
        let dir = root.path().join(SLUG);
        std::fs::create_dir_all(&dir).expect("create dir");
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"architectures":["{ARCH}"]}}"#),
        )
        .expect("write config.json");
        std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write tokenizer.json");
        std::fs::write(dir.join(weight), b"").expect("write weight entrypoint");

        assert!(
            matches!(
                slug_snapshot(Some(as_str(root.path())), SLUG),
                Snapshot::Found { .. }
            ),
            "{weight} alone must complete a snapshot"
        );
    }
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
    let dir = make_snapshot(tmp.path(), "single", OTHER_ARCH);

    assert_eq!(
        override_snapshot(Some(as_str(&dir))),
        Some(Snapshot::Found {
            path: dir,
            arch: OTHER_ARCH.to_owned()
        })
    );
}

/// A named path that is not a runnable snapshot is configuration present and
/// wrong — a typo, a moved snapshot, a half-finished download. Reporting
/// absence there is how it turns into a green run that asserted nothing. Each
/// shape is distinct on disk and a laxer check would pass some of them.
#[test]
fn an_override_naming_anything_unrunnable_is_misconfigured() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let absent = tmp.path().join("was-moved-away");

    let empty = tmp.path().join("no-files");
    std::fs::create_dir_all(&empty).expect("create dir");

    let config_only = tmp.path().join("config-only");
    std::fs::create_dir_all(&config_only).expect("create dir");
    std::fs::write(config_only.join("config.json"), b"{}").expect("write config.json");

    let no_shards = tmp.path().join("jsons-but-no-shards");
    std::fs::create_dir_all(&no_shards).expect("create dir");
    std::fs::write(no_shards.join("config.json"), b"{}").expect("write config.json");
    std::fs::write(no_shards.join("tokenizer.json"), b"{}").expect("write tokenizer.json");

    for dir in [&absent, &empty, &config_only, &no_shards] {
        let got = override_snapshot(Some(as_str(dir)));
        assert!(
            matches!(got, Some(Snapshot::Misconfigured(_))),
            "{} must be a hard failure, got {got:?}",
            dir.display()
        );
    }
}

// ── choose: the resolution decision ──────────────────────────────────────

fn found(p: &str, arch: &str) -> Snapshot {
    Snapshot::Found {
        path: PathBuf::from(p),
        arch: arch.to_owned(),
    }
}

fn run_path(gate: &Gate) -> &Path {
    match gate {
        Gate::Run { path, .. } => path,
        other => panic!("expected Run, got {other:?}"),
    }
}

/// With no override, the slug decides.
#[test]
fn the_slug_runs_when_it_matches_and_no_override_is_set() {
    assert_eq!(
        choose(None, found("/snap/slug", ARCH), &MODEL, CHECKING),
        Gate::Run {
            path: PathBuf::from("/snap/slug"),
            note: None
        }
    );
}

/// An override for THIS architecture wins over the slug: that is what makes
/// `RMLX_REGEN_GOLDENS=1 RMLX_KV_TEST_MODEL=<path>` record from the named
/// snapshot, and what lets one golden be compared against a non-slug checkpoint
/// deliberately. Ranking the slug first would silently ignore the named path.
#[test]
fn a_matching_override_outranks_the_slug() {
    let got = choose(
        Some(found("/named", ARCH)),
        found("/snap/slug", ARCH),
        &MODEL,
        CHECKING,
    );
    assert_eq!(
        run_path(&got),
        Path::new("/named"),
        "the named path must win, or regen records the wrong checkpoint"
    );
}

/// The override names another architecture — a KV developer's export. It says
/// nothing about this golden, so the slug must still arm it. Standing down here
/// is the original silent-skip defect.
#[test]
fn an_override_for_another_arch_falls_through_to_the_slug() {
    let got = choose(
        Some(found("/named-gemma4", OTHER_ARCH)),
        found("/snap/slug", ARCH),
        &MODEL,
        CHECKING,
    );
    assert_eq!(
        run_path(&got),
        Path::new("/snap/slug"),
        "an override for another arch must not disarm this golden"
    );
}

/// ...and the operator is told, because a resolution that discarded something
/// they named must not pass in silence.
#[test]
fn a_stood_down_override_is_reported_on_the_run_it_produced() {
    let got = choose(
        Some(found("/named-gemma4", OTHER_ARCH)),
        found("/snap/slug", ARCH),
        &MODEL,
        CHECKING,
    );
    match got {
        Gate::Run { note: Some(n), .. } => {
            assert!(n.contains(OTHER_ARCH), "{n}");
            assert!(n.contains(super::SINGLE_MODEL_VAR), "{n}");
        }
        other => panic!("expected Run carrying a note, got {other:?}"),
    }
}

/// Under regen the same configuration must FAIL. Writing a committed fixture
/// from a snapshot the operator did not name, while discarding the one they
/// did, gives the golden untraceable provenance — and regenerating the whole
/// set with one override would give each fixture a different origin silently.
#[test]
fn a_stood_down_override_fails_when_recording_a_fixture() {
    let got = choose(
        Some(found("/named-gemma4", OTHER_ARCH)),
        found("/snap/slug", ARCH),
        &MODEL,
        RECORDING,
    );
    match got {
        Gate::Fail(why) => {
            assert!(why.contains(super::REGEN_VAR), "{why}");
            assert!(why.contains(SLUG), "{why}");
        }
        other => panic!("expected Fail while recording, got {other:?}"),
    }
}

/// Recording against the golden the override DOES serve stays allowed — that is
/// the whole point of the variable.
#[test]
fn a_matching_override_still_records() {
    let got = choose(
        Some(found("/named", ARCH)),
        found("/snap/slug", ARCH),
        &MODEL,
        RECORDING,
    );
    assert_eq!(run_path(&got), Path::new("/named"));
}

/// Fall-through with nothing to fall through to still skips, and the reason
/// names both halves so the operator can see the override stood down.
#[test]
fn an_override_for_another_arch_with_no_slug_skips_naming_both() {
    let got = choose(
        Some(found("/named-gemma4", OTHER_ARCH)),
        Snapshot::Absent("root holds no slug".to_owned()),
        &MODEL,
        CHECKING,
    );
    match got {
        Gate::Skip(why) => {
            assert!(why.contains("root holds no slug"), "{why}");
            assert!(why.contains(super::SINGLE_MODEL_VAR), "{why}");
        }
        other => panic!("expected Skip naming both halves, got {other:?}"),
    }
}

/// An override whose config is unreadable is a broken pointer, not a statement
/// about another architecture — only a legible, different arch may fall
/// through. The slug branch treats the same empty string as fatal, and the two
/// halves must not disagree on identical input.
#[test]
fn an_override_with_an_unreadable_config_fails_rather_than_falling_through() {
    let got = choose(
        Some(found("/named-corrupt", "")),
        found("/snap/slug", ARCH),
        &MODEL,
        CHECKING,
    );
    assert!(
        matches!(got, Gate::Fail(_)),
        "an unreadable config in a directory the operator NAMED must break the \
         run, not quietly resolve a different snapshot; got {got:?}"
    );
}

/// A misconfigured override fails even when the slug would have resolved — the
/// operator named a path and it is wrong, and a fall-through would bury that.
#[test]
fn a_misconfigured_override_fails_even_when_the_slug_resolves() {
    assert_eq!(
        choose(
            Some(Snapshot::Misconfigured("bad override".to_owned())),
            found("/snap/slug", ARCH),
            &MODEL,
            CHECKING
        ),
        Gate::Fail("bad override".to_owned()),
        "a wrong pointer must not be masked by a working slug"
    );
}

/// The slug names THIS golden's snapshot, so a wrong arch there is a wrong
/// pointer, not a different model's business.
#[test]
fn arch_mismatch_on_the_slug_snapshot_fails() {
    let got = choose(None, found("/snap/slug", OTHER_ARCH), &MODEL, CHECKING);
    assert!(
        matches!(got, Gate::Fail(_)),
        "the slug named the snapshot, so a wrong arch must break the gate; got {got:?}"
    );
}

/// An unreadable or absent `config.json` yields an empty arch string. That must
/// not accidentally match a golden's expected-arch list.
#[test]
fn an_unreadable_slug_config_does_not_match_any_arch() {
    let got = choose(None, found("/snap/slug", ""), &MODEL, CHECKING);
    assert!(matches!(got, Gate::Fail(_)), "got {got:?}");
}

/// Absence skips, misconfiguration fails — the split the whole design rests on,
/// pinned at the decision level where no `Found` path is involved.
#[test]
fn slug_absence_skips_and_slug_misconfiguration_fails() {
    assert!(matches!(
        choose(
            None,
            Snapshot::Absent("nothing set".to_owned()),
            &MODEL,
            CHECKING
        ),
        Gate::Skip(_)
    ));
    assert!(matches!(
        choose(
            None,
            Snapshot::Misconfigured("bad root".to_owned()),
            &MODEL,
            CHECKING
        ),
        Gate::Fail(_)
    ));
}

// ── apply: decision to return value ──────────────────────────────────────

#[test]
fn apply_returns_the_path_on_run() {
    assert_eq!(
        apply(
            Gate::Run {
                path: PathBuf::from("/snap"),
                note: None
            },
            "t"
        ),
        Some(PathBuf::from("/snap"))
    );
    assert_eq!(
        apply(
            Gate::Run {
                path: PathBuf::from("/snap"),
                note: Some("stood down".to_owned())
            },
            "t"
        ),
        Some(PathBuf::from("/snap")),
        "a note must not change what resolves"
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
