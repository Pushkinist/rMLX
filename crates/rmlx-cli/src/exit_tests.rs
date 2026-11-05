use super::*;

/// The scripts that tell "measured, record pending" from a crash read the code
/// from their own copy of it; each copy must be this one.
#[test]
fn every_script_copy_of_the_record_pending_code_is_this_one() {
    let copies = [
        (
            "scripts/lib/record_pending.sh",
            include_str!("../../../scripts/lib/record_pending.sh"),
            "RECORD_PENDING_EXIT=",
        ),
        (
            "scripts/eval/wikitext2_ppl.py",
            include_str!("../../../scripts/eval/wikitext2_ppl.py"),
            "RECORD_PENDING_EXIT = ",
        ),
    ];
    for (path, text, key) in copies {
        let value = text
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .unwrap_or_else(|| panic!("{path} has no `{key}` line"));
        assert_eq!(value.trim(), RECORD_PENDING.to_string(), "{path}");
    }
}
