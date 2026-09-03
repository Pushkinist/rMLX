use super::{
    decode_config, decode_config_from_notes, decode_config_is_well_formed, predicate, NotesVerdict,
    CELL_COLUMNS,
};

#[test]
fn the_predicate_binds_every_cell_column_once() {
    let (sql, next) = predicate(1);
    assert_eq!(next, 1 + CELL_COLUMNS.len(), "next free parameter index");
    for (offset, col) in CELL_COLUMNS.iter().enumerate() {
        let op = if col.nullable { "IS" } else { "=" };
        let want = format!("{} {op} ?{}", col.name, offset + 1);
        assert!(sql.contains(&want), "missing `{want}` in:\n{sql}");
    }
}

/// A nullable column compared with `=` never matches, so an ordinary-decode
/// cell would read as having no champion at all.
#[test]
fn a_nullable_column_is_compared_null_safely() {
    let (sql, _) = predicate(1);
    for col in CELL_COLUMNS.iter().filter(|c| c.nullable) {
        assert!(
            !sql.contains(&format!("{} = ", col.name)),
            "{} is nullable and must be compared with IS",
            col.name
        );
    }
}

#[test]
fn the_partition_list_is_the_cell_key_plus_metric() {
    let list = super::partition_columns();
    for col in CELL_COLUMNS {
        assert!(list.contains(col.name), "{} missing from {list}", col.name);
    }
    assert!(list.ends_with(", metric"), "{list}");
}

// ── Deriving `decode_config` from notes ───────────────────────────────────────

/// The one place the format is written, pinned against the string
/// `scripts/spec_bench.sh` records for the same run.
#[test]
fn decode_config_format_is_stable() {
    assert_eq!(decode_config("mtp", 5), "mtp/block=5");
    assert_eq!(decode_config("dflash", 16), "dflash/block=16");
}

#[test]
fn notes_naming_a_drafter_classify_as_that_arm() {
    for (notes, want) in [
        (
            "config=mtp6b draft_kind=mtp block_size=6 decode_tps=client-side",
            "mtp/block=6",
        ),
        (
            "config=dflash16 draft_kind=dflash block_size=16",
            "dflash/block=16",
        ),
        (
            "config=eagle5 draft_kind=eagle block_size=5 tag=x",
            "eagle/block=5",
        ),
        (
            "config=mtp draft_kind=mtp block_size=5 tag=canary-code",
            "mtp/block=5",
        ),
    ] {
        assert_eq!(
            decode_config_from_notes(notes),
            NotesVerdict::Speculative(want.to_owned()),
            "{notes}"
        );
    }
}

/// An early `spec_bench.sh` put the drafter flags on both arms. A row that says
/// it is the no-drafter arm is one, whatever else the line carries.
#[test]
fn notes_saying_there_was_no_drafter_win_over_ones_naming_it() {
    for notes in [
        "config=normal draft_kind=mtp block_size=5",
        "config=normal draft_kind=none tag=canary-code",
        "config=base draft_kind=none decode_tps=client-side",
    ] {
        assert_eq!(
            decode_config_from_notes(notes),
            NotesVerdict::Plain,
            "{notes}"
        );
    }
}

#[test]
fn notes_that_say_nothing_are_not_guessed_at() {
    for notes in [
        "",
        "label=kvbytes_e2b",
        "perf_ab.sh ABBA n=6/arm",
        "config=mtp draft_kind=mtp block_size=many",
    ] {
        assert_eq!(
            decode_config_from_notes(notes),
            NotesVerdict::Silent,
            "{notes}"
        );
    }
}

// ── The `decode_config` grammar ───────────────────────────────────────────────

/// Every spelling an emitter in this tree produces is a term of the grammar,
/// including the speculative arm's, which predates it.
#[test]
fn the_shipped_decode_configurations_are_well_formed() {
    for config in [
        "mtp/block=5",
        "dflash/block=16",
        "eagle/block=5",
        "prefill_chunk=1024",
        "prefill_chunk=1024,spec/block=5",
    ] {
        assert!(decode_config_is_well_formed(config), "{config}");
    }
    assert!(decode_config_is_well_formed(&decode_config("mtp", 5)));
}

/// The rejections that protect cell identity: a term order that would split
/// one configuration across two cells, a repeated key that says two things at
/// once, and the shapes that are not terms at all.
#[test]
fn a_malformed_decode_configuration_is_refused() {
    for config in [
        "",                                      // empty is not "no configuration"; NULL is
        "plain",                                 // no `=`
        "prefill_chunk=",                        // no value
        "=1024",                                 // no key
        "prefill chunk=1024",                    // whitespace in a key
        "prefill_chunk = 1024",                  // whitespace around `=`
        "PrefillChunk=1024",                     // upper case in a key
        "spec/block=5,prefill_chunk=1024",       // terms out of key order
        "prefill_chunk=1024,prefill_chunk=2048", // one key, two values
        "prefill_chunk=1024,",                   // empty trailing term
    ] {
        assert!(!decode_config_is_well_formed(config), "{config}");
    }
}

/// The backfill is the grammar's second writer, and it composes its value out
/// of free-form notes. A drafter name the notes spell in any other way must
/// leave the row unclassified rather than store a string the ingest path would
/// have refused — `observations` is append-only, so a wrong value there stays.
#[test]
fn notes_that_compose_an_illegal_configuration_are_not_classified() {
    for notes in [
        "config=mtp draft_kind=Eagle3 block_size=5",
        "config=mtp draft_kind=mtp-v2! block_size=5",
        "config=mtp draft_kind=/block block_size=5",
    ] {
        assert_eq!(
            decode_config_from_notes(notes),
            NotesVerdict::Silent,
            "{notes}"
        );
    }
    // The spelling the bench scripts actually write still classifies.
    assert_eq!(
        decode_config_from_notes("config=mtp draft_kind=mtp block_size=5"),
        NotesVerdict::Speculative("mtp/block=5".to_string())
    );
}
