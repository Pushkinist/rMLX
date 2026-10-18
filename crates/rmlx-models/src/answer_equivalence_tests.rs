use super::*;

const N_TOKENS: usize = 256;

/// `stream` with the element at `at` replaced.
fn with_at<T: Clone>(stream: &[T], at: usize, value: T) -> Vec<T> {
    let mut out = stream.to_vec();
    if let Some(slot) = out.get_mut(at) {
        *slot = value;
    }
    out
}

/// What the divergence oracle reads, and when it stands down.
#[test]
fn the_divergence_oracle_reads_the_first_differing_position_or_nothing() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let flipped = with_at(&plain, 66, 9999);
    let tied = with_at(&[5.0f32; N_TOKENS], 66, 0.01);
    assert_eq!(
        divergence_confidence(&flipped, &plain, &tied),
        Some((66, 0.0)),
        "the position read is the first the arms differ at"
    );

    // Two identical arms have no divergence to judge, and an empty margin list
    // stands the oracle down rather than reading position 0 of nothing.
    assert_eq!(divergence_confidence(&plain, &plain, &tied), None);
    assert_eq!(divergence_confidence(&flipped, &plain, &[]), None);

    // A divergence past the end of the reference arm has no margin behind it;
    // the length guard owns that pair.
    let extended: Vec<u32> = plain.iter().copied().chain([1_u32]).collect();
    assert_eq!(divergence_confidence(&extended, &plain, &tied), None);
}

/// The oracle's own denominator: the confidence is a rank over the reference
/// arm's margins, so it must not move when every margin is scaled.
///
/// A threshold in nats would move. This is the property that lets one ceiling
/// cover two models whose logits are not on the same scale.
#[test]
fn the_confidence_is_a_rank_and_not_a_number_of_nats() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let flipped = with_at(&plain, 40, 7777);

    let base: Vec<f32> = (0..N_TOKENS).map(|i| 0.1 + (i % 17) as f32).collect();
    let scaled: Vec<f32> = base.iter().map(|m| m * 250.0).collect();
    assert_eq!(
        divergence_confidence(&flipped, &plain, &base),
        divergence_confidence(&flipped, &plain, &scaled),
        "scaling every margin must not move the rank"
    );
}

#[test]
fn the_common_prefix_stops_at_the_first_difference_or_the_shorter_stream() {
    assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 4]), 2);
    assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]), 3);
    assert_eq!(common_prefix_len(&[], &[1]), 0);
}

/// The gap is best minus second best whatever order the candidates arrive in,
/// and a position that reported fewer than two has no gap to be mistaken for a
/// tie.
#[test]
fn the_margin_is_the_gap_between_the_two_best_candidates() {
    assert_eq!(top_two_margin([-0.5, -1.0]), 0.5);
    assert_eq!(top_two_margin([-1.0, -0.5]), 0.5);
    assert_eq!(top_two_margin([-0.5, -2.0, -0.75]), 0.25);
    assert_eq!(top_two_margin([-2.0, -0.75, -0.5]), 0.25);
    assert_eq!(top_two_margin([-1.0, -1.0]), 0.0);
    assert_eq!(top_two_margin([-0.5, -0.5, -3.0]), 0.0);
    assert_eq!(top_two_margin([-0.5]), f32::INFINITY);
    assert_eq!(top_two_margin([]), f32::INFINITY);
}

fn run(len: usize) -> Vec<u32> {
    (0..len as u32).collect()
}

/// The length ratio is read before the rank, so a short run is refused even
/// where its divergence sits at a tie.
#[test]
fn a_short_run_is_read_before_its_rank() {
    let plain = run(100);
    let spec: Vec<u32> = plain.iter().copied().take(50).collect();
    let margins = with_at(&[1.0f32; 100], 50, 0.0);
    assert_eq!(
        read_divergence(&spec, &plain, &margins),
        Divergence::ShortRun {
            ratio: 0.5,
            at: Some((50, 0.0))
        }
    );
}

/// `below` decisions of the reference run are less sure than the one at 10.
fn margins_with_rank(below: usize) -> Vec<f32> {
    let mut margins = vec![1.0f32; 100];
    for slot in margins.iter_mut().rev().take(below) {
        *slot = 0.5;
    }
    margins
}

#[test]
fn a_flip_is_a_near_tie_up_to_the_ceiling_and_confident_above_it() {
    let plain = run(100);
    let flipped = with_at(&plain, 10, 9999);
    // 12 of 100 decisions less sure than the flipped one is the ceiling itself.
    assert_eq!(
        read_divergence(&flipped, &plain, &margins_with_rank(12)),
        Divergence::NearTie(10, 0.12)
    );
    assert!(matches!(
        read_divergence(&flipped, &plain, &margins_with_rank(13)),
        Divergence::Confident(10, _)
    ));
}

#[test]
fn identical_runs_and_a_run_that_outlasts_the_reference_are_unread() {
    let plain = run(100);
    let margins = vec![1.0f32; 100];
    assert_eq!(
        read_divergence(&plain, &plain, &margins),
        Divergence::Unread
    );
    assert_eq!(
        read_divergence(&run(110), &plain, &margins),
        Divergence::Unread
    );
}
