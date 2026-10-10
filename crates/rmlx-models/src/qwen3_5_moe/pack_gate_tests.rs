//! The gate and the fixture reader, on synthetic logits.

use super::*;

const VOCAB: usize = 64;
const TOP: usize = 8;

/// `n` values in [-1, 1) from a fixed recurrence.
fn noise(seed: u64, n: usize) -> Vec<f32> {
    (0..n as u64)
        .map(|i| {
            let u = ((seed + i) * 1_103_515_245 + 12_345) % 2_147_483_648;
            (u as f64 / 1_073_741_824.0 - 1.0) as f32
        })
        .collect()
}

/// Full synthetic logit rows: a spread of values with a lead of 1.0 for the
/// first id, so every position is decided at a small floor.
fn synthetic_rows(positions: usize) -> Vec<Vec<f32>> {
    (0..positions)
        .map(|p| {
            let mut row: Vec<f32> = noise(p as u64 * 7919, VOCAB)
                .iter()
                .map(|v| 3.0 * v)
                .collect();
            let lead = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) + 1.0;
            if let Some(first) = row.get_mut(p % VOCAB) {
                *first = lead;
            }
            row
        })
        .collect()
}

/// The reference positions of `rows`.
fn positions_of(rows: &[Vec<f32>]) -> Vec<Position> {
    rows.iter()
        .map(|row| {
            let mut ranked: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            ranked.truncate(TOP);
            Position {
                top_ids: ranked.iter().map(|r| r.0).collect(),
                top: ranked.iter().map(|r| r.1).collect(),
                lse: logsumexp(row),
            }
        })
        .collect()
}

/// `rows` plus `amplitude` times noise of `seed`.
fn perturbed(rows: &[Vec<f32>], amplitude: f32, seed: u64) -> Vec<Vec<f32>> {
    rows.iter()
        .enumerate()
        .map(|(p, row)| {
            row.iter()
                .zip(noise(seed + p as u64 * 104_729, row.len()))
                .map(|(v, n)| v + amplitude * n)
                .collect()
        })
        .collect()
}

fn seen_of(reference: &[Position], rows: &[Vec<f32>]) -> Result<Vec<Seen>> {
    reference
        .iter()
        .zip(rows)
        .map(|(at, row)| Seen::of(at, row))
        .collect()
}

/// The reference, a floor at amplitude 0.01, and the statistics of each
/// candidate in `candidates` (rows) against it.
fn judged(candidates: &[Vec<Vec<f32>>]) -> Result<(Vec<Position>, Stats, Vec<Stats>)> {
    let base = synthetic_rows(32);
    let reference = positions_of(&base);
    let floor = stats(
        &reference,
        &seen_of(&reference, &perturbed(&base, 0.01, 1))?,
    )?;
    let judged = candidates
        .iter()
        .map(|c| stats(&reference, &seen_of(&reference, c)?))
        .collect::<Result<_>>()?;
    Ok((reference, floor, judged))
}

#[test]
fn the_gate_passes_the_reference_itself_and_a_candidate_at_the_floor() -> Result<()> {
    let base = synthetic_rows(32);
    let (reference, floor, judged) = judged(&[base.clone(), perturbed(&base, 0.01, 2)])?;
    assert!(
        floor.top_max > 0.0 && floor.kl > 0.0,
        "the floor is not zero: {floor:?}"
    );
    for s in &judged {
        let broken = gate(&reference, &floor, s);
        assert!(broken.is_empty(), "{s:?} breaks {broken:?}");
    }
    Ok(())
}

#[test]
fn the_gate_fails_a_planted_shift_and_names_the_rule() -> Result<()> {
    let base = synthetic_rows(32);
    // Position 5: the second id rises 0.5 above the first, a lead of 1.0
    // that the floor cannot move.
    let mut swapped = base.clone();
    let at = positions_of(&base);
    let (first, second) = match at.get(5).map(|p| p.top_ids.as_slice()) {
        Some([first, second, ..]) => (*first, *second),
        _ => return Err(Error::Model("synthetic position 5".to_owned())),
    };
    if let Some(row) = swapped.get_mut(5) {
        let lead = row.get(first).copied().unwrap_or_default();
        if let Some(v) = row.get_mut(second) {
            *v = lead + 0.5;
        }
    }
    let shifted: Vec<Vec<f32>> = base
        .iter()
        .map(|row| row.iter().map(|v| v + 0.05).collect())
        .collect();
    let (reference, floor, judged) = judged(&[swapped, shifted, perturbed(&base, 0.03, 3)])?;
    let names: Vec<Vec<String>> = judged.iter().map(|s| gate(&reference, &floor, s)).collect();
    let has = |i: usize, needle: &str| {
        names
            .get(i)
            .is_some_and(|b| b.iter().any(|m| m.contains(needle)))
    };
    assert!(has(0, "decided position 5"), "swap: {names:?}");
    assert!(has(1, "reach_max"), "uniform shift of 0.05: {names:?}");
    assert!(
        has(2, "top_rms") && has(2, "kl"),
        "noise at 3 x the floor: {names:?}"
    );
    Ok(())
}

#[test]
fn the_control_check_needs_every_statistic_past_ten_times_the_floor() -> Result<()> {
    let base = synthetic_rows(32);
    let (_, floor, judged) = judged(&[perturbed(&base, 0.5, 4), perturbed(&base, 0.05, 5)])?;
    let far = judged.first().map(|s| control_short(&floor, s));
    let near = judged.get(1).map(|s| control_short(&floor, s));
    assert_eq!(far, Some(Vec::new()), "noise at 50 x the floor");
    // The divergence grows with the square of the noise, so at 5 x the floor
    // only the two linear statistics fall short.
    assert!(
        near.as_ref().is_some_and(|short| short.len() == 2
            && short.iter().any(|m| m.contains("reach_max"))
            && short.iter().any(|m| m.contains("top_rms"))),
        "noise at 5 x the floor: {near:?}"
    );
    Ok(())
}

/// Two positions, three top logits each; the candidate moves the first
/// position by 0.3 and 0.4 and swaps the argmax of the second.
#[test]
fn stats_reads_the_largest_and_the_rms_difference_and_the_argmax() -> Result<()> {
    let reference = vec![
        Position {
            top_ids: vec![7, 3, 5],
            top: vec![4.0, 3.0, 2.0],
            lse: 4.5,
        },
        Position {
            top_ids: vec![1, 2, 0],
            top: vec![4.0, 3.0, 2.0],
            lse: 4.5,
        },
    ];
    let seen = vec![
        Seen {
            top: vec![4.3, 3.4, 2.0],
            lse: 4.5,
            argmax: 7,
        },
        Seen {
            top: vec![4.0, 3.0, 2.0],
            lse: 4.5,
            argmax: 2,
        },
    ];
    let s = stats(&reference, &seen)?;
    let rms = ((0.3f64 * 0.3 + 0.4 * 0.4) / 6.0).sqrt();
    assert!((s.top_max - 0.4).abs() < 1e-6, "{s:?}");
    assert!((s.reach_max - 0.4).abs() < 1e-6, "{s:?}");
    assert!((s.top_rms - rms).abs() < 1e-6, "{s:?} against {rms}");
    assert_eq!(s.disagree, vec![1]);
    Ok(())
}

/// At positions whose margin the floor can close, an argmax change is
/// allowed up to `GATE x max(floor disagreements, 1)` positions.
#[test]
fn the_gate_counts_argmax_changes_at_undecided_positions_against_a_budget() {
    let near_tie = Position {
        top_ids: vec![0, 1],
        top: vec![1.0, 0.999],
        lse: 2.0,
    };
    let reference = vec![near_tie; 4];
    let floor = Stats {
        reach_max: 0.01,
        top_max: 0.01,
        top_rms: 0.01,
        kl: 1e-4,
        disagree: Vec::new(),
    };
    let with = |disagree: Vec<usize>| Stats {
        reach_max: 0.0,
        top_max: 0.0,
        top_rms: 0.0,
        kl: 0.0,
        disagree,
    };
    let two = gate(&reference, &floor, &with(vec![0, 1]));
    let three = gate(&reference, &floor, &with(vec![0, 1, 2]));
    assert!(two.is_empty(), "{two:?}");
    assert!(
        three.len() == 1 && three.iter().all(|m| m.contains("budget 2")),
        "{three:?}"
    );
}

/// The floor of the two gap cases below: 0.1 on each logit statistic.
fn tenth_floor() -> Stats {
    Stats {
        reach_max: 0.1,
        top_max: 0.1,
        top_rms: 0.1,
        kl: 1e-3,
        disagree: Vec::new(),
    }
}

/// A difference at a rank no sampler draws (reference probability below
/// 1e-4) does not decide the gate; the largest difference over all 20 ranks
/// is still reported.
#[test]
fn a_difference_at_a_rank_no_sampler_reaches_does_not_decide_the_gate() -> Result<()> {
    let row: Vec<f32> = [12.0, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 0.0]
        .into_iter()
        .chain(std::iter::repeat_n(-20.0, VOCAB - 8))
        .collect();
    let reference = positions_of(std::slice::from_ref(&row));
    let rare = reference.first().and_then(|at| {
        at.top_ids
            .last()
            .map(|&id| (id, at.top.last().copied(), at.lse))
    });
    let Some((rare_id, Some(rare_logit), lse)) = rare else {
        return Err(Error::Model("synthetic position".to_owned()));
    };
    assert!(f64::from(rare_logit - lse).exp() < 1e-4, "rank 7 is rare");
    let mut moved = row;
    if let Some(v) = moved.get_mut(rare_id) {
        *v += 0.5;
    }
    let candidate = stats(&reference, &seen_of(&reference, &[moved])?)?;
    assert!((candidate.top_max - 0.5).abs() < 1e-6, "{candidate:?}");
    assert!(candidate.reach_max < 1e-6, "{candidate:?}");
    let broken = gate(&reference, &tenth_floor(), &candidate);
    assert!(broken.is_empty(), "{broken:?}");
    Ok(())
}

/// Rank 0 is read even where a flat distribution puts it below the
/// reachable probability.
#[test]
fn the_largest_difference_always_reads_rank_0() -> Result<()> {
    let mut row = vec![0.0f32; 20_000];
    if let Some(first) = row.first_mut() {
        *first = 0.1;
    }
    let reference = positions_of(std::slice::from_ref(&row));
    assert!(
        reference.first().is_some_and(|at| f64::from(
            at.top.first().copied().unwrap_or(0.0) - at.lse
        )
        .exp()
            < 1e-4),
        "rank 0 is below the reachable probability"
    );
    if let Some(first) = row.first_mut() {
        *first += 0.5;
    }
    let candidate = stats(&reference, &seen_of(&reference, &[row])?)?;
    assert!((candidate.reach_max - 0.5).abs() < 1e-6, "{candidate:?}");
    Ok(())
}

/// An argmax change is refused at a position whose reference top-2 margin
/// is above the floor's largest difference, even below twice it.
#[test]
fn an_argmax_change_above_the_floor_margin_breaks_the_gate() {
    let reference = vec![Position {
        top_ids: vec![0, 1],
        top: vec![1.0, 0.85],
        lse: 2.0,
    }];
    let candidate = Stats {
        reach_max: 0.0,
        top_max: 0.0,
        top_rms: 0.0,
        kl: 0.0,
        disagree: vec![0],
    };
    let broken = gate(&reference, &tenth_floor(), &candidate);
    assert!(
        broken.iter().any(|m| m.contains("decided position 0")),
        "{broken:?}"
    );
}

/// Write a synthetic fixture with `tensors` (name, dtype, shape, bytes).
fn write_fixture(
    path: &Path,
    tensors: &[(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)],
) -> Result<()> {
    let views = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes)
                .map(|v| ((*name).to_owned(), v))
                .map_err(|e| Error::Model(format!("{name}: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let bytes = safetensors::serialize(views, None).map_err(|e| Error::Model(e.to_string()))?;
    std::fs::write(path, bytes).map_err(|e| Error::Model(e.to_string()))
}

fn le_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn the_reader_returns_what_the_fixture_holds_and_refuses_a_misfit() -> Result<()> {
    use safetensors::Dtype::{F32, I32};
    let (seqs, len, k) = (2usize, 3usize, 4usize);
    let tops = vec![seqs, len, k];
    let grid = vec![seqs, len];
    let n = seqs * len;
    let ids: Vec<i32> = (0..n as i32).map(|i| 100 + i).collect();
    let top_ids: Vec<i32> = (0..(n * k) as i32).collect();
    let top: Vec<f32> = (0..n * k).map(|i| 10.0 - i as f32 * 0.25).collect();
    let lse: Vec<f32> = (0..n).map(|i| 11.0 + i as f32).collect();
    let floor_lse: Vec<f32> = lse.iter().map(|v| v + 0.5).collect();
    let floor_argmax: Vec<i32> = (0..n as i32).map(|i| i * 4 + 1).collect();
    let full = |floor_top: (Vec<usize>, &[f32])| {
        vec![
            ("ids", I32, grid.clone(), le_i32(&ids)),
            ("reference_top_ids", I32, tops.clone(), le_i32(&top_ids)),
            ("reference_top_logits", F32, tops.clone(), le_f32(&top)),
            ("reference_logsumexp", F32, grid.clone(), le_f32(&lse)),
            ("floor_top_logits", F32, floor_top.0, le_f32(floor_top.1)),
            ("floor_logsumexp", F32, grid.clone(), le_f32(&floor_lse)),
            ("floor_argmax", I32, grid.clone(), le_i32(&floor_argmax)),
        ]
    };
    let dir = tempfile::tempdir().map_err(|e| Error::Model(e.to_string()))?;
    let path = dir.path().join("logits.safetensors");

    write_fixture(&path, &full((tops.clone(), &top)))?;
    let r = VendorReference::read(&path)?;
    assert_eq!(r.ids, vec![vec![100, 101, 102], vec![103, 104, 105]]);
    assert_eq!(r.positions.len(), n);
    let last = r
        .positions
        .last()
        .map(|p| (p.top_ids.clone(), p.top.clone(), p.lse));
    assert_eq!(
        last,
        Some((vec![20, 21, 22, 23], vec![5.0, 4.75, 4.5, 4.25], 16.0))
    );
    let floor_last = r.floor.last().map(|s| (s.top.clone(), s.lse, s.argmax));
    assert_eq!(floor_last, Some((vec![5.0, 4.75, 4.5, 4.25], 16.5, 21)));

    let wider: Vec<f32> = (0..n * (k + 1)).map(|i| i as f32).collect();
    write_fixture(&path, &full((vec![seqs, len, k + 1], &wider)))?;
    let misfit = VendorReference::read(&path)
        .map(|_| ())
        .map_err(|e| e.to_string());
    assert!(
        misfit
            .as_ref()
            .is_err_and(|e| e.contains("floor_top_logits")),
        "a floor of another width: {misfit:?}"
    );

    let mut missing = full((tops.clone(), &top));
    missing.retain(|t| t.0 != "floor_argmax");
    write_fixture(&path, &missing)?;
    let misfit = VendorReference::read(&path)
        .map(|_| ())
        .map_err(|e| e.to_string());
    assert!(
        misfit.as_ref().is_err_and(|e| e.contains("floor_argmax")),
        "a missing tensor: {misfit:?}"
    );
    Ok(())
}
