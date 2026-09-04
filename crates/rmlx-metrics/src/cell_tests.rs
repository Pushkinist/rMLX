use super::{
    decode_config, decode_config_from_notes, decode_config_is_all_defaults,
    decode_config_is_well_formed, decode_config_names_a_drafter, predicate, NotesVerdict,
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

/// The one place the format is written, pinned against the string the engine
/// logs and `scripts/spec_bench.sh` records for the same run.
#[test]
fn decode_config_format_is_stable() {
    assert_eq!(decode_config("mtp", 5, None), "mtp/block=5");
    assert_eq!(decode_config("dflash", 16, None), "dflash/block=16");
}

/// An adaptive arm is a different cell from the fixed arm at the same ceiling,
/// and both are legal `decode_config` values in term order.
#[test]
fn adaptive_depth_is_a_separate_well_formed_cell() {
    let adaptive = decode_config("dflash", 16, Some("accept_rate"));
    assert_eq!(adaptive, "dflash/block=16,dflash/depth=accept_rate");
    assert!(decode_config_is_well_formed(&adaptive), "{adaptive}");
    assert_ne!(adaptive, decode_config("dflash", 16, None));
    assert!(!decode_config_is_all_defaults(&adaptive), "{adaptive}");

    let confidence = decode_config("mtp", 8, Some("confidence"));
    assert_eq!(confidence, "mtp/block=8,mtp/depth=confidence");
    assert!(decode_config_is_well_formed(&confidence), "{confidence}");
}

/// The block reaches the term as the engine holds it. A narrower parameter
/// would truncate or saturate here, and the row would name a block nothing ran
/// in a table that cannot take it back out.
#[test]
fn a_block_beyond_a_narrower_type_is_not_truncated() {
    let huge = u32::MAX as usize + 2;
    assert_eq!(
        decode_config("mtp", huge, None),
        format!("mtp/block={huge}")
    );
    assert!(decode_config_is_well_formed(&decode_config(
        "mtp", huge, None
    )));

    // The same value read back out of a row's notes, for the same reason.
    let notes = format!("config=mtp draft_kind=mtp block_size={huge}");
    assert_eq!(
        decode_config_from_notes(&notes),
        NotesVerdict::Speculative(format!("mtp/block={huge}")),
    );
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
        "dflash/block=16,dflash/depth=accept_rate",
        "two_model/block=5",
    ] {
        assert!(decode_config_is_well_formed(config), "{config}");
    }
    assert!(decode_config_is_well_formed(&decode_config("mtp", 5, None)));
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

// ── Spellings of the engine's own defaults ───────────────────────────────────

/// `NULL` is the engine at its defaults, so a string saying the same thing is a
/// second spelling of one configuration — and two spellings are two cells.
#[test]
fn a_configuration_spelling_only_defaults_is_recognised() {
    let head = rmlx_core::kv_boundary::DEFAULT_BOUNDARY_HEAD_N;
    let tail = rmlx_core::kv_boundary::DEFAULT_BOUNDARY_TAIL_N;
    assert!(decode_config_is_all_defaults(&format!(
        "kv_boundary/head={head},kv_boundary/tail={tail}"
    )));
}

/// A term off its default, a key with no single default, and a malformed
/// string are all not "the engine as shipped" — the first two say something the
/// column exists to record, and the third is the grammar check's business.
#[test]
fn a_configuration_that_says_something_is_not_all_defaults() {
    let head = rmlx_core::kv_boundary::DEFAULT_BOUNDARY_HEAD_N;
    let tail = rmlx_core::kv_boundary::DEFAULT_BOUNDARY_TAIL_N;
    for config in [
        format!("kv_boundary/head={head},kv_boundary/tail={}", tail + 1),
        format!("kv_boundary/head={},kv_boundary/tail={tail}", head + 1),
        "kv_boundary/head=0,kv_boundary/tail=0".to_string(),
        // No default in the table: absence means no drafter at all, so no
        // spelling of `mtp/block` is redundant.
        "mtp/block=5".to_string(),
        // A per-architecture default cannot be recognised from the term.
        "prefill_chunk=1024".to_string(),
        // Mixed: one term at its default, one not.
        format!("kv_boundary/head={head},kv_boundary/tail=4"),
        // Malformed — a different question, answered by the grammar check.
        "kv boundary head=2".to_string(),
        String::new(),
    ] {
        assert!(
            !decode_config_is_all_defaults(&config),
            "{config:?} must not be read as the engine at its defaults"
        );
    }
}

/// Not every `decode_config` is a drafter. A reader that keyed on "the column
/// is not NULL" would pull a prefill-chunk sweep into the speculative table and
/// report a blank round loop for it.
#[test]
fn only_a_block_term_names_a_drafter() {
    for config in [
        "mtp/block=5",
        "dflash/block=16,dflash/depth=accept_rate",
        "two_model/block=5",
        "prefill_chunk=1024,spec/block=5",
    ] {
        assert!(decode_config_names_a_drafter(config), "{config}");
    }
    for config in [
        "prefill_chunk=1024",
        "kv_boundary/head=3,kv_boundary/tail=9",
        "mtp/blocking=5",
        "block=5",
        "",
        "mtp/block=",
    ] {
        assert!(!decode_config_names_a_drafter(config), "{config}");
    }
}
