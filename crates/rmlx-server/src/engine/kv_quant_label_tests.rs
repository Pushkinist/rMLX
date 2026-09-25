//! The metrics label of every codec, held against the literal codec table of
//! `rmlx-kv-quant`. The label bytes are what the metrics DB groups rows by, so
//! they must not move when the label table moves into the codec crate.

use super::kv_quant_label;
use rmlx_kv_quant::ALL_KV_QUANTS;

#[allow(
    dead_code,
    reason = "the table states every codec fact; this crate reads only the spelling and the metrics label"
)]
mod codec_facts {
    include!("../../../rmlx-kv-quant/src/codec_facts_table.rs");
}

#[test]
fn every_codec_writes_the_metrics_label_its_row_holds() {
    for &quant in ALL_KV_QUANTS {
        let spelling = quant.to_string();
        let Some(row) = codec_facts::CODEC_FACTS
            .iter()
            .find(|row| row.spelling == spelling)
        else {
            panic!("{spelling}: no row in rmlx-kv-quant's codec_facts_table.rs");
        };
        assert_eq!(
            kv_quant_label(Some(quant)),
            row.metrics_label,
            "{spelling}: metrics label"
        );
    }
    assert_eq!(
        kv_quant_label(None),
        "auto",
        "no codec override: metrics label"
    );
}
