//! The per-codec facts of today's code, held against a literal table.
//!
//! The table (`codec_facts_table.rs`) has one row per `Display` spelling, and
//! each row is written out by hand. The tests here make two claims:
//!
//! * The table and [`ALL_KV_QUANTS`] are a bijection: each codec has exactly
//!   one row and each row names exactly one codec.
//! * For each codec, each predicate gives the value its row states.
//!
//! A per-codec descriptor can give one row to both a predicate and the code
//! that the predicate describes. Then a wrong row agrees with itself and passes
//! every structural check. This table is written apart from that row, so a
//! wrong row turns a cell red here.
//!
//! What the table cannot see: a new codec. Its author writes its row too, so a
//! wrong fact in a new row is for review to find.

use super::{KvQuant, ALL_KV_QUANTS};

include!("codec_facts_table.rs");

/// The literal row for `spelling`, if the table has one.
pub(crate) fn facts_for(spelling: &str) -> Option<&'static CodecFacts> {
    CODEC_FACTS.iter().find(|row| row.spelling == spelling)
}

/// The row for `quant`. A codec with no row is a failure of the bijection
/// test, so the other tests stop here with a message that names it.
fn row(quant: KvQuant) -> &'static CodecFacts {
    let spelling = quant.to_string();
    facts_for(&spelling).unwrap_or_else(|| {
        panic!("{spelling}: no row in codec_facts_table.rs — add the codec's facts as literals")
    })
}

#[test]
fn the_table_and_all_kv_quants_are_a_bijection() {
    let spellings: Vec<String> = ALL_KV_QUANTS.iter().map(ToString::to_string).collect();

    for spelling in &spellings {
        let rows = CODEC_FACTS
            .iter()
            .filter(|row| row.spelling == spelling.as_str())
            .count();
        assert_eq!(rows, 1, "{spelling}: the table has {rows} rows, not 1");
    }
    for row in CODEC_FACTS {
        let codecs = spellings
            .iter()
            .filter(|s| s.as_str() == row.spelling)
            .count();
        assert_eq!(
            codecs, 1,
            "table row {}: {codecs} codecs in ALL_KV_QUANTS have this spelling, not 1",
            row.spelling
        );
    }
    assert_eq!(
        CODEC_FACTS.len(),
        ALL_KV_QUANTS.len(),
        "the table and ALL_KV_QUANTS have different lengths"
    );
}

#[test]
fn every_codec_states_the_facts_its_row_holds() {
    for &quant in ALL_KV_QUANTS {
        let want = row(quant);
        let name = want.spelling;
        assert_eq!(
            quant.decode_reads_packed_store(),
            want.decode_reads_packed_store,
            "{name}: decode_reads_packed_store"
        );
        assert_eq!(
            quant.materialises_packed_store(),
            want.materialises_packed_store,
            "{name}: materialises_packed_store"
        );
        for shares_kv in [false, true] {
            let i = usize::from(shares_kv);
            assert_eq!(
                quant.feeds_bf16_k_at_decode(shares_kv),
                want.feeds_bf16_k[i],
                "{name}: feeds_bf16_k_at_decode({shares_kv})"
            );
            assert_eq!(
                quant.feeds_bf16_v_at_decode(shares_kv),
                want.feeds_bf16_v[i],
                "{name}: feeds_bf16_v_at_decode({shares_kv})"
            );
        }
        assert_eq!(quant.carries_msl(), want.carries_msl, "{name}: carries_msl");
        assert_eq!(
            quant.approx_code_bits(),
            want.approx_code_bits,
            "{name}: approx_code_bits"
        );
        let (k_store, v_store) = quant.side_stores();
        let got_stores = (
            k_store.map(|s| format!("{s:?}")),
            v_store.map(|s| format!("{s:?}")),
        );
        let want_stores = (
            want.side_stores.0.map(str::to_string),
            want.side_stores.1.map(str::to_string),
        );
        assert_eq!(got_stores, want_stores, "{name}: side_stores (k, v)");
        assert_eq!(
            quant.k_below_8bit(),
            want.k_below_8bit,
            "{name}: k_below_8bit"
        );
        assert_eq!(
            quant.uses_mixed_path(),
            want.uses_mixed_path,
            "{name}: uses_mixed_path"
        );
        assert_eq!(
            quant.is_k_only_iso_rotor(),
            want.is_k_only_iso_rotor,
            "{name}: is_k_only_iso_rotor"
        );
        assert_eq!(
            quant.mixed_params(),
            want.mixed_params,
            "{name}: mixed_params"
        );
    }
}

/// `cpu_hot_path_reason` reads the rotor QJL switch at call time, so the class
/// is read with the switch off and then on. The switch is the `RMLX_ROTOR_QJL`
/// environment variable while no CLI value is latched, which is the state of
/// this test binary.
#[test]
#[allow(unsafe_code, reason = "env write under env_lock")]
fn every_codec_states_the_cpu_hot_path_class_its_row_holds() {
    let _guard = crate::test_utils::env_lock();
    assert!(
        !crate::rotor_qjl::rotor_qjl_cli_is_set(),
        "a test latched the CLI rotor QJL value, so the environment variable no longer moves \
         the switch and the QJL-gated class cannot be read"
    );
    for &quant in ALL_KV_QUANTS {
        let want = row(quant);
        // SAFETY: env lock held — no concurrent env reader/writer.
        unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };
        let off = quant.cpu_hot_path_reason().is_some();
        // SAFETY: env lock held — no concurrent env reader/writer.
        unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };
        let on = quant.cpu_hot_path_reason().is_some();
        let got = match (off, on) {
            (false, false) => CpuHotPath::Never,
            (true, true) => CpuHotPath::Always,
            (false, true) => CpuHotPath::WhenQjl,
            (true, false) => panic!(
                "{}: a CPU hot path with QJL off and none with QJL on fits no class",
                want.spelling
            ),
        };
        assert_eq!(
            got, want.cpu_hot_path,
            "{}: cpu_hot_path_reason class",
            want.spelling
        );
    }
}
