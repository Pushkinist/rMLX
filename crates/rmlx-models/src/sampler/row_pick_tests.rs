//! Unit tests for the token pick from a host logit row.

use super::{pick_at, RowPicker};
use crate::sampler::{Pcg32, DEFAULT_SEED};
use rmlx_core::error::Result;

const MASKED: f32 = f32::NEG_INFINITY;
const SEED: u64 = 7;

/// Weights 1, 2, 0 and 5 at temperature 1; 1, 4, 0 and 25 at temperature 0.5;
/// 1, 1.414, 0 and 2.236 at temperature 2.
fn row() -> Vec<f32> {
    vec![0.0, 2.0_f32.ln(), MASKED, 5.0_f32.ln()]
}

/// Sixteen ids with the same logit.
fn flat_row() -> Vec<f32> {
    vec![0.5; 16]
}

fn stream(seed: u64, temperature: f32, picks: usize) -> Result<Vec<u32>> {
    let mut picker = RowPicker::new(seed);
    let row = flat_row();
    (0..picks).map(|_| picker.pick(&row, temperature)).collect()
}

/// A temperature of 0 gives the largest logit and leaves the generator where
/// it was.
#[test]
fn temperature_zero_gives_the_largest_logit_and_draws_nothing() -> Result<()> {
    let mut picker = RowPicker::new(SEED);
    for _ in 0..5 {
        assert_eq!(picker.pick(&row(), 0.0)?, 3);
        assert_eq!(picker.pick(&flat_row(), 0.0)?, 0);
    }
    assert_eq!(picker.rng.next_u32(), Pcg32::new(SEED).next_u32());
    Ok(())
}

/// A temperature above 0 gives the token at the point `uniform` of the
/// cumulative distribution of `softmax(row / temperature)`. The same point
/// gives another token at another temperature.
#[test]
fn a_temperature_above_zero_gives_the_token_at_that_point_of_the_cdf() -> Result<()> {
    let cases: [(f32, &[(f32, usize)]); 3] = [
        (
            1.0,
            &[
                (0.0, 0),
                (0.12, 0),
                (0.13, 1),
                (0.37, 1),
                (0.38, 3),
                (0.999, 3),
            ],
        ),
        (0.5, &[(0.03, 0), (0.04, 1), (0.16, 1), (0.17, 3), (0.2, 3)]),
        (2.0, &[(0.2, 0), (0.23, 1), (0.5, 1), (0.53, 3)]),
    ];
    for (temperature, points) in cases {
        for &(uniform, token) in points {
            assert_eq!(
                pick_at(&row(), temperature, uniform)?,
                token,
                "temperature {temperature}, point {uniform}"
            );
        }
    }
    Ok(())
}

/// An id whose logit is `-inf` is never picked, at no point of the
/// distribution and at no temperature.
#[test]
fn a_masked_logit_is_never_picked() -> Result<()> {
    let rows: [&[f32]; 4] = [
        &[MASKED, 1.0, 0.5, 0.2],
        &[1.0, MASKED, MASKED, 0.2],
        &[0.3, 1.0, 0.5, MASKED],
        &[MASKED, MASKED, 0.1, MASKED],
    ];
    for row in rows {
        for temperature in [0.2, 1.0, 2.0] {
            for step in 0..=1000 {
                let uniform = (step as f32 / 1000.0).min(0.999_999);
                let token = pick_at(row, temperature, uniform)?;
                assert!(
                    row.get(token).is_some_and(|logit| logit.is_finite()),
                    "token {token} at temperature {temperature}, point {uniform}"
                );
            }
        }
        let mut picker = RowPicker::new(SEED);
        for temperature in [0.0, 0.7] {
            for _ in 0..200 {
                let token = picker.pick(row, temperature)? as usize;
                assert!(row.get(token).is_some_and(|logit| logit.is_finite()));
            }
        }
    }
    Ok(())
}

/// The same seed gives the same stream, and it is the stream of the shared
/// generator: one draw for each pick, in order. Another seed gives another
/// stream.
#[test]
fn the_same_seed_gives_the_same_stream() -> Result<()> {
    let first = stream(SEED, 1.0, 64)?;
    assert_eq!(first, stream(SEED, 1.0, 64)?);

    let mut rng = Pcg32::new(SEED);
    let row = flat_row();
    let mut from_generator = Vec::new();
    for _ in 0..64 {
        from_generator.push(pick_at(&row, 1.0, rng.next_f32())? as u32);
    }
    assert_eq!(first, from_generator);

    assert_ne!(first, stream(SEED + 1, 1.0, 64)?);
    Ok(())
}

/// A pick at temperature 0 between two sampled picks does not move the stream.
#[test]
fn a_greedy_pick_does_not_move_the_stream() -> Result<()> {
    let row = flat_row();
    let mut mixed = RowPicker::new(SEED);
    let a = mixed.pick(&row, 1.0)?;
    mixed.pick(&row, 0.0)?;
    let b = mixed.pick(&row, 1.0)?;
    let mut rng = Pcg32::new(SEED);
    let want = [
        pick_at(&row, 1.0, rng.next_f32())? as u32,
        pick_at(&row, 1.0, rng.next_f32())? as u32,
    ];
    assert_eq!([a, b], want);
    Ok(())
}

/// A row with a `NaN` logit cannot be sampled: it is an error, not a token.
#[test]
fn a_row_that_is_not_a_number_is_an_error() {
    assert!(pick_at(&[0.0, f32::NAN, 1.0], 1.0, 0.5).is_err());
}

/// A row with a `NaN` or `+inf` logit is an error at every temperature, also
/// when another logit is finite: the greedy pick would take that logit, and the
/// average log-probability of the decode, which a reply states, would not be a
/// number.
#[test]
fn a_row_with_a_logit_that_is_not_a_number_is_an_error_at_every_temperature() {
    let mut picker = RowPicker::new(SEED);
    for bad in [f32::NAN, f32::INFINITY] {
        for row in [vec![0.0, bad, 1.0], vec![bad, 2.0], vec![3.0, MASKED, bad]] {
            for temperature in [0.0, 0.7] {
                let picked = picker.pick(&row, temperature);
                assert!(
                    matches!(&picked, Err(error) if error.to_string().contains("not a number")),
                    "{row:?} at temperature {temperature}: {picked:?}"
                );
            }
        }
    }
}

/// A row in which every id is masked has no token to pick: it is an error at
/// each temperature, not a masked id. A row with no finite logit and a logit
/// that is not a number is an error that names that cause, not a mask.
#[test]
fn a_row_with_no_logit_to_pick_is_an_error_that_names_the_cause() {
    let mut picker = RowPicker::new(SEED);
    let rows = [
        (vec![MASKED; 4], "each logit is masked"),
        (Vec::new(), "each logit is masked"),
        (vec![f32::NAN; 4], "not a number"),
        (vec![MASKED, f32::NAN, MASKED], "not a number"),
    ];
    for (row, cause) in rows {
        for temperature in [0.0, 0.7] {
            let picked = picker.pick(&row, temperature);
            assert!(
                matches!(&picked, Err(error) if error.to_string().contains(cause)),
                "{row:?} at temperature {temperature}: {picked:?}"
            );
        }
    }
}

/// The temperature of a pick reaches its draw: for the same generator values,
/// the stream at one temperature is the stream of that temperature's
/// distribution, and it is not the stream at another temperature.
#[test]
fn the_same_draws_give_another_stream_at_another_temperature() -> Result<()> {
    let row = row();
    let stream_at = |temperature: f32| -> Result<Vec<u32>> {
        let mut picker = RowPicker::new(SEED);
        (0..64).map(|_| picker.pick(&row, temperature)).collect()
    };
    let want_at = |temperature: f32| -> Result<Vec<u32>> {
        let mut rng = Pcg32::new(SEED);
        (0..64)
            .map(|_| pick_at(&row, temperature, rng.next_f32()).map(|id| id as u32))
            .collect()
    };
    let (cold, hot) = (stream_at(0.5)?, stream_at(2.0)?);
    assert_eq!(cold, want_at(0.5)?);
    assert_eq!(hot, want_at(2.0)?);
    assert_ne!(cold, hot);
    Ok(())
}

/// The picker of a request with no seed draws the stream of the sampler's
/// default seed, on every construction.
#[test]
fn the_default_picker_draws_the_stream_of_the_default_seed() -> Result<()> {
    let row = flat_row();
    let mut seeded = RowPicker::new(DEFAULT_SEED);
    let want: Vec<u32> = (0..64)
        .map(|_| seeded.pick(&row, 1.0))
        .collect::<Result<_>>()?;
    for _ in 0..2 {
        let mut picker = RowPicker::default();
        let got: Vec<u32> = (0..64)
            .map(|_| picker.pick(&row, 1.0))
            .collect::<Result<_>>()?;
        assert_eq!(got, want);
    }
    Ok(())
}
