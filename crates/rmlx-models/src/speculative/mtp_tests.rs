use super::*;

/// Compile-check: the public MTP surface exists with the expected sigs.
#[test]
fn mtp_module_compiles() {
    // Reference the items so the symbols are checked at compile time
    // without spelling out their (clippy-flagged complex) fn types.
    let _load = MtpDrafter::load;
    let _ = _load;
}
