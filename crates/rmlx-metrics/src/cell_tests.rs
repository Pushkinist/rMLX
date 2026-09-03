use super::{predicate, CELL_COLUMNS};

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
