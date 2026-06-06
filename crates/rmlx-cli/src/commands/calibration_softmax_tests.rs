//! Unit tests for `SoftmaxMassSink::budget_for_distribution`.

use super::SoftmaxMassSink;

/// Scores chosen so the softmax distribution is exactly [0.5, 0.3, 0.1, 0.05, 0.05].
#[test]
fn softmax_mass_known_distribution() {
    let target = [0.5_f32, 0.3, 0.1, 0.05, 0.05];
    // s_i = ln(p_i) -> softmax(s) = p (any constant offset is normalised out).
    let scores: Vec<f32> = target.iter().map(|p| p.ln()).collect();

    // target_mass slightly below the exact cumulative to absorb f32 rounding
    // when reconstructing the softmax distribution from log-probabilities.
    // After sorting: [0.5, 0.3, 0.1, 0.05, 0.05].
    // cumulative through k=2 is 0.8; pick 0.79 so f32 jitter still hits.
    let b = SoftmaxMassSink::budget_for_distribution(&scores, 0.79, 1);
    assert_eq!(b, 2, "expected budget=2 for cumulative-mass ~0.8, got {b}");

    // through k=3 is 0.9; pick 0.89.
    let b = SoftmaxMassSink::budget_for_distribution(&scores, 0.89, 1);
    assert_eq!(b, 3, "expected budget=3 for cumulative-mass ~0.9, got {b}");

    // through k=4 is 0.95; pick 0.94.
    let b = SoftmaxMassSink::budget_for_distribution(&scores, 0.94, 1);
    assert_eq!(b, 4, "expected budget=4 for cumulative-mass ~0.95, got {b}");
}

#[test]
fn softmax_mass_uniform_distribution() {
    // Equal scores -> uniform softmax over 10 -> p_i = 0.1.
    let scores = [0.0_f32; 10];
    // target_mass=0.5 -> cumulative 0.1*5 = 0.5 -> budget = 5.
    let b = SoftmaxMassSink::budget_for_distribution(&scores, 0.5, 1);
    assert_eq!(
        b, 5,
        "expected budget=5 for uniform target_mass=0.5, got {b}"
    );

    // target_mass=0.95 -> need 0.1*10 = 1.0 -> budget = 10 (must cover).
    let b = SoftmaxMassSink::budget_for_distribution(&scores, 0.95, 1);
    assert_eq!(b, 10);
}

#[test]
fn softmax_mass_pathological_floor_overrides() {
    // Distribution: one key at probability ~0.99, others negligible.
    // log-prob scale: pick scores so softmax yields [0.99, 0.001, 0.001, ...].
    let mut probs = vec![0.001_f32; 11];
    probs[0] = 0.99;
    // Normalise (already nearly normalised; force exact).
    let s: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= s;
    }
    let scores: Vec<f32> = probs.iter().map(|p| p.ln()).collect();

    // No floor: target_mass=0.9 satisfied by the single peak -> budget=1.
    let b = SoftmaxMassSink::budget_for_distribution(&scores, 0.9, 1);
    assert_eq!(b, 1, "expected raw budget=1 on single-peak, got {b}");

    // With floor=16 the budget is overridden to 16 (saturated at len if smaller).
    let b = SoftmaxMassSink::budget_for_distribution(&scores, 0.9, 16);
    assert_eq!(b, 16, "expected floor=16 override, got {b}");
}

#[test]
fn softmax_mass_zero_floor_clamps_to_one() {
    let scores = [0.0_f32; 4];
    // floor=0 normalised to 1.
    let b = SoftmaxMassSink::budget_for_distribution(&scores, 0.25, 0);
    assert_eq!(
        b, 1,
        "expected budget=1 at target 0.25 over uniform-4 with floor=0"
    );
}

#[test]
fn softmax_mass_empty_scores_yields_floor() {
    let scores: Vec<f32> = Vec::new();
    let b = SoftmaxMassSink::budget_for_distribution(&scores, 0.9, 8);
    assert_eq!(b, 8);
}

#[test]
fn softmax_mass_all_negative_inf_yields_floor() {
    let scores = vec![f32::NEG_INFINITY; 5];
    let b = SoftmaxMassSink::budget_for_distribution(&scores, 0.9, 4);
    assert_eq!(b, 4);
}

#[test]
fn expand_to_q_heads_passthrough() {
    // SoftmaxMassSink stores budgets per Q-head directly (the within-GQA-group
    // `max` is applied at `record` time). `expand_to_q_heads` becomes a clone
    // of the per-Q-head table.
    let mut sink = SoftmaxMassSink::new(2, 16, 4, 64, 0.95, 16);
    let row0: Vec<u32> = (0..16).map(|i| 100 + i as u32).collect();
    let row1: Vec<u32> = (0..16).map(|i| 200 + i as u32).collect();
    sink.budgets[0] = row0.clone();
    sink.budgets[1] = row1.clone();
    let expanded = sink.expand_to_q_heads();
    assert_eq!(expanded.len(), 2);
    assert_eq!(expanded[0].len(), 16);
    assert_eq!(expanded[0], row0);
    assert_eq!(expanded[1], row1);
}

#[test]
fn sink_table_shape_per_q_head() {
    // budgets table is [num_layers][n_q_heads], not [num_layers][n_kv_heads].
    // Confirms the per-Q-head storage shape.
    let num_layers = 3_usize;
    let n_q_heads = 12_usize;
    let n_kv_heads = 4_usize;
    let sink = SoftmaxMassSink::new(num_layers, n_q_heads, n_kv_heads, 64, 0.95, 16);
    assert_eq!(sink.budgets.len(), num_layers);
    for row in &sink.budgets {
        assert_eq!(row.len(), n_q_heads, "row must be n_q_heads-wide");
        for &b in row {
            assert_eq!(b, 16, "init value must equal the floor");
        }
    }
}
