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
