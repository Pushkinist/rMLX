//! Tests for the `rmlx info --list-cache-types` listings.
//!
//! The residency table is the one surface in the tree that publishes a
//! resident-KV ratio to an operator — `check_kv_codec_disposition.sh` RULE 7
//! rejects one written anywhere else, and the `--kv-quant` help points here.
//! A surface with that job and no test is a surface that can quietly stop
//! covering a codec, or start printing a column of `1.000x`, with every gate
//! green. `preset_table_tests.rs::no_preset_is_a_memory_lever` is the shape
//! being followed: sweep the type, assert the property.

use super::{kv_quant_residency_table, RESIDENCY_TABLE_HEADING};
use rmlx_kv_quant::{KvQuant, ALL_KV_QUANTS};

/// The listing's data lines, in order: everything after the two header lines
/// and before the trailing prose.
fn codec_rows(rendered: &str) -> Vec<&str> {
    rendered
        .lines()
        .skip_while(|l| !l.starts_with("-----"))
        .skip(1)
        .take_while(|l| !l.trim().is_empty())
        .collect()
}

/// Coverage comes from the type. A hand-written expected list is the defect
/// this whole branch exists to remove.
#[test]
fn every_kv_quant_has_a_row() {
    let rendered = kv_quant_residency_table();
    let rows = codec_rows(&rendered);
    assert_eq!(
        rows.len(),
        ALL_KV_QUANTS.len(),
        "the listing printed {} rows for {} codecs — a codec an operator can \
         pass is a codec the listing has to price:\n{rendered}",
        rows.len(),
        ALL_KV_QUANTS.len()
    );
    for (row, q) in rows.iter().zip(ALL_KV_QUANTS) {
        assert!(
            row.starts_with(&q.to_string()),
            "expected a row for '{q}', got '{row}'"
        );
    }
}

/// The unquantised reference is 1.000x of itself on both topologies. Anchors
/// the ratio column: a table rendering bytes against the wrong denominator
/// fails here first.
#[test]
fn the_bf16_reference_reads_unity_on_both_topologies() {
    let rendered = kv_quant_residency_table();
    let row = codec_rows(&rendered)
        .into_iter()
        .find(|r| r.starts_with(&KvQuant::None.to_string()))
        .unwrap_or_else(|| panic!("no row for the bf16 reference:\n{rendered}"));
    assert_eq!(
        row.matches("1.000x").count(),
        2,
        "'{row}' must read 1.000x in both columns"
    );
    assert!(
        row.contains("16.000"),
        "'{row}' must read 16.000 bits per value — two bf16 buffers and nothing else"
    );
}

/// The two columns differ for exactly the family whose mirror follows the
/// stack's topology, and agree for every other codec.
///
/// This is the property a single-column listing could not express, and the one
/// that made the old help wrong in opposite directions on two architectures.
/// Which codecs those are is read from the predicates, never listed here.
#[test]
fn the_columns_differ_exactly_where_the_mirror_follows_shares_kv() {
    let rendered = kv_quant_residency_table();
    for (row, &q) in codec_rows(&rendered).iter().zip(ALL_KV_QUANTS) {
        let topology_sensitive = q.feeds_bf16_k_at_decode(false) != q.feeds_bf16_k_at_decode(true)
            || q.feeds_bf16_v_at_decode(false) != q.feeds_bf16_v_at_decode(true);
        // codec, then four numeric fields (bits, ratio) x (dense, shared),
        // then the disposition prose.
        let f: Vec<&str> = row.split_whitespace().collect();
        assert!(f.len() >= 5, "row too short: '{row}'");
        let dense = (f[1], f[2]);
        let shared = (f[3], f[4]);
        assert_eq!(
            dense != shared,
            topology_sensitive,
            "'{q}' is {}topology-sensitive but its two columns {} — row '{row}'",
            if topology_sensitive { "" } else { "not " },
            if dense == shared { "agree" } else { "differ" }
        );
    }
}

/// At least one live codec is strictly under the bf16 reference on a dense
/// stack, and at least one is strictly over it.
///
/// Without this the table could render every ratio as `1.000x` — the exact
/// shape of a byte model that stopped reading the codec — and the two tests
/// above would still pass.
#[test]
fn the_listing_spans_both_sides_of_the_reference() {
    let rendered = kv_quant_residency_table();
    let rows = codec_rows(&rendered);
    let ratios: Vec<f64> = rows
        .iter()
        .zip(ALL_KV_QUANTS)
        .map(|(_, &q)| {
            let bf16 = KvQuant::None.estimated_resident_bytes_per_layer(4096, 128, 8, false) as f64;
            q.estimated_resident_bytes_per_layer(4096, 128, 8, false) as f64 / bf16
        })
        .collect();
    assert!(
        ratios.iter().any(|r| *r < 0.999),
        "no codec renders below the bf16 reference — a listing of nothing but \
         1.000x is what a byte model that stopped reading the codec looks like:\n{rendered}"
    );
    assert!(
        ratios.iter().any(|r| *r > 1.001),
        "no codec renders above the bf16 reference:\n{rendered}"
    );
    // And the rendered text agrees with the arithmetic, not just the arithmetic
    // with itself.
    for (row, ratio) in rows.iter().zip(&ratios) {
        let want = format!("{ratio:5.3}x");
        assert!(
            row.contains(want.trim()),
            "row '{row}' does not carry its own dense ratio {}",
            want.trim()
        );
    }
}

/// The heading the disposition gate binds the help's pointer to is the heading
/// the listing actually renders.
#[test]
fn the_listing_opens_with_its_heading() {
    assert!(
        kv_quant_residency_table().contains(RESIDENCY_TABLE_HEADING),
        "the rendered listing does not carry RESIDENCY_TABLE_HEADING"
    );
}
