use super::*;

#[test]
fn arch_unsupported_display() {
    let e = Error::ArchUnsupported {
        arch: "FakeArch".to_owned(),
    };
    assert_eq!(e.to_string(), "arch unsupported: FakeArch");
    assert!(matches!(e, Error::ArchUnsupported { .. }));
}

#[test]
fn kv_storage_mismatch_display() {
    let e = Error::KvStorageMismatch {
        expected: "K8V4",
        got: "None",
    };
    assert_eq!(
        e.to_string(),
        "kv storage mismatch: expected K8V4, got None"
    );
    assert!(matches!(e, Error::KvStorageMismatch { .. }));
}

#[test]
fn ssd_tier_already_installed_display() {
    let e = Error::SsdTierAlreadyInstalled;
    assert_eq!(
        e.to_string(),
        "ssd tier config already installed \u{2014} refusing to re-install"
    );
    assert!(matches!(e, Error::SsdTierAlreadyInstalled));
}

#[test]
fn unimplemented_display() {
    let e = Error::Unimplemented("truncate_to: GDN cannot snapshot mid-sequence");
    assert_eq!(
        e.to_string(),
        "unimplemented: truncate_to: GDN cannot snapshot mid-sequence"
    );
    assert!(matches!(e, Error::Unimplemented(_)));
}

/// The ceiling error fires on both prefill and decode, so its message must not
/// bake in a phase name. A decode-path violation reported as "prefill" is the
/// user-visible mislabel this test guards against.
#[test]
fn kv_ceiling_exceeded_display_is_phase_neutral() {
    let e = Error::KvCeilingExceeded {
        requested: 641,
        ceiling: 640,
    };
    assert_eq!(
        e.to_string(),
        "kv request exceeds max-ctx ceiling: requested=641, ceiling=640"
    );
    assert!(
        !e.to_string().contains("prefill"),
        "ceiling message must not name a phase — it fires on decode too"
    );
}

/// The hard-cap error likewise fires on both phases; its message stays
/// phase-neutral.
#[test]
fn kv_hard_cap_exceeded_display_is_phase_neutral() {
    let e = Error::KvHardCapExceeded {
        requested: 4097,
        cap: 4096,
    };
    assert_eq!(
        e.to_string(),
        "kv request exceeds hard cap: requested=4097, cap=4096"
    );
    assert!(
        !e.to_string().contains("prefill"),
        "hard-cap message must not name a phase — it fires on decode too"
    );
}

/// Only genuinely transient failures are migratable; a ceiling/hard-cap
/// rejection (same bound on every attempt) and other structural errors are not.
#[test]
fn is_migratable_classifies_transient_and_permanent() {
    assert!(Error::Mlx("watchdog".to_owned()).is_migratable());
    assert!(Error::Other("recovered panic".to_owned()).is_migratable());

    assert!(!Error::KvCeilingExceeded {
        requested: 641,
        ceiling: 640,
    }
    .is_migratable());
    assert!(!Error::KvHardCapExceeded {
        requested: 4097,
        cap: 4096,
    }
    .is_migratable());
    assert!(!Error::SmokeProbe("NaN".to_owned()).is_migratable());
    assert!(!Error::Config("bad".to_owned()).is_migratable());
}
